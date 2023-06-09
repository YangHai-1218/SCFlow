from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
from collections import OrderedDict

from .builder import REFINERS
from .base_flow_refiner import BaseFlowRefiner
from models.encoder import build_encoder
from models.loss import build_loss
from models.utils import (
    get_flow_from_delta_pose_and_depth, 
    filter_flow_by_mask, cal_epe,
    filter_flow_by_depth,
    get_2d_3d_corr_by_fw_flow,
    solve_pose_by_pnp, 
    remap_pose_to_origin_resoluaion)





@REFINERS.register_module()
class RAFTRefinerFlowMask(BaseFlowRefiner):
    """RAFT model. Supervised version. Predict flow.

    Args:
        num_levels (int): Number of levels in .
        radius (int): Number of radius in  .
        cxt_channels (int): Number of channels of context feature.
        h_channels (int): Number of channels of hidden feature in .
        cxt_encoder (dict): Config dict for building context encoder.
        freeze_bn (bool, optional): Whether to freeze batchnorm layer or not.
            Default: False.
    """

    def __init__(self,
                 seperate_encoder: bool,
                 cxt_channels: int,
                 h_channels: int,
                 cxt_encoder: dict,
                 encoder: dict,
                 decoder: dict,
                 renderer: Optional[dict]=None,
                 flow_loss_cfg: Optional[dict]=None,
                 occlusion_loss_cfg: Optional[dict]=None,
                 max_flow: float=400.,
                 render_augmentations: Optional[list]=None,
                 filter_invalid_flow_by_mask: bool=True,
                 filter_invalid_flow_by_depth: bool=False,
                 freeze_bn: bool = False,
                 train_cfg: Optional[dict] = dict(),
                 test_cfg: Optional[dict] = dict(),
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            encoder=encoder, 
            decoder=decoder, 
            seperate_encoder=seperate_encoder,
            filter_invalid_flow_by_mask=filter_invalid_flow_by_mask,
            filter_invalid_flow_by_depth=filter_invalid_flow_by_depth,
            renderer=renderer, 
            render_augmentations=render_augmentations,
            max_flow=max_flow,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        self.context = build_encoder(cxt_encoder)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels
        self.vis_tensorboard = False
        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels

        if flow_loss_cfg is not None:
            self.flow_loss_func = build_loss(flow_loss_cfg)
        if occlusion_loss_cfg is not None:
            self.occlusion_loss_func = build_loss(occlusion_loss_cfg)
        if freeze_bn:
            self.freeze_bn()
        self.test_iter_num = self.test_cfg.get('iters') if 'iters' in self.test_cfg else self.decoder.iters

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    
    def extract_feat(
        self, images_0, images_1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        assert not(len(images_0.shape) == 3 and len(images_1.shape) == 3)

        # for multiview forward flow
        if len(images_1.shape) == 3:
            # (C, H, W)
            view_num = len(images_0)
            image1_feat = self.real_encoder(images_1[None])
            image1_feat = image1_feat.expand(view_num, -1, -1, -1)
        else:
            image1_feat = self.real_encoder(images_1)

        # for multiview backward flow
        if len(images_0.shape) == 3:
            view_num = len(images_1)
            image0_feat = self.render_encoder(images_0[None])
            image0_feat = image0_feat.expand(view_num, -1, -1, -1)
            cxt_feat = self.context(images_0[None])
            cxt_feat = cxt_feat.expand(view_num, -1, -1, -1)
        else:
            image0_feat = self.render_encoder(images_0)
            cxt_feat = self.context(images_0)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return image0_feat, image1_feat, h_feat, cxt_feat

    def get_flow(
            self,
            render_images,
            real_images,
            init_flow=None,
    ) -> Dict[str, torch.Tensor]:
        feat_render, feat_real, h_feat, cxt_feat = self.extract_feat(render_images, real_images)
        if init_flow is None:
            B, _, H, W = feat_real.shape
            init_flow = torch.zeros((B, 2, H, W), device=feat_real.device)
        output = self.decoder(
            feat_render, feat_real, init_flow, h_feat, cxt_feat)
        return output
    
    
    def forward_single_view(self, data, data_batch, return_pose=True):
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        rendered_images, real_images = data['rendered_images'], data['real_images']
        rendered_depths, internel_k = data['rendered_depths'], data['internel_k']
        per_img_patch_num, labels = data['per_img_patch_num'], data['labels']

        iters = self.decoder.iters
        self.decoder.iters = self.test_iter_num
        sequence_flow, sequence_occlusion = self.get_flow(rendered_images, real_images)
        self.decoder.iters = iters
        batch_flow = sequence_flow[-1]
        batch_occlusion = sequence_occlusion[-1].squeeze(dim=1)

        
        if return_pose:
            results = self.solve_pose(
                batch_flow, rendered_depths, ref_rotations, ref_translations, internel_k, labels, per_img_patch_num, batch_occlusion)
            batch_internel_k = torch.split(internel_k, per_img_patch_num)
            batch_rotations, batch_translations = results['rotations'], results['translations']
            image_metas = data_batch['img_metas']
            batch_rotations, batch_translations = remap_pose_to_origin_resoluaion(batch_rotations, batch_translations, batch_internel_k, image_metas)
            results['rotations'] = batch_rotations
            results['translations'] = batch_translations
            return results
        else:
            # for validation
            return batch_flow, batch_occlusion, data


        



    
    def loss(self, data_batch):
        data = self.format_data_train_sup(data_batch)
        log_vars = OrderedDict()
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        internel_k, rendered_depths, rendered_masks = data['internel_k'], data['rendered_depths'], data['rendered_masks']
        gt_masks = data['gt_masks']

        # get flow
        sequence_flow, sequence_occlusion = self.get_flow(rendered_images, real_images)
        gt_flow = get_flow_from_delta_pose_and_depth(
            ref_rotations, ref_translations,
            gt_rotations, gt_translations,
            rendered_depths, internel_k, invalid_num=self.max_flow
        )

        # filter invalid flow
        if self.filter_invalid_flow_by_mask:
            gt_flow = filter_flow_by_mask(gt_flow, gt_masks, invalid_num=self.max_flow)
        if self.filter_invalid_flow_by_depth:
            gt_rendered_depths = data['gt_rendered_depths']
            gt_flow = filter_flow_by_depth(gt_flow, gt_rendered_depths, rendered_depths, invalid_num=self.max_flow)

        gt_occlusion_mask = (torch.sum(gt_flow, dim=1, keepdim=False) < self.max_flow).to(torch.float32)

        loss_flow, seq_loss_flow_list = self.flow_loss_func(
            sequence_flow, gt_flow=gt_flow, valid=rendered_masks
        )
        sequence_occlusion = [occlusion.squeeze(dim=1) for occlusion in sequence_occlusion]
        loss_occlusion, seq_loss_occlusion_list = self.occlusion_loss_func(
            sequence_occlusion, gt_mask=gt_occlusion_mask, valid=rendered_masks,
        )
        
        for i in range(len(seq_loss_flow_list)):
            log_vars.update({f'seq_{i}_flow_loss':seq_loss_flow_list[i].item()})
            log_vars.update({f'seq_{i}_occ_loss':seq_loss_occlusion_list[i].item()})
        
        loss = loss_flow + loss_occlusion
        
        pred_flow = sequence_flow[-1]
        pred_flow = pred_flow*rendered_masks[:, None]
        log_vars.update({'loss_occ':loss_occlusion.item(), 'loss_flow':loss_flow.item(), 'loss':loss.item()})
        if self.vis_tensorboard:
            log_imgs = self.add_vis_images(
                **dict(
                    real_images=real_images, render_images=rendered_images,
                    gt_flow=gt_flow, pred_flow=pred_flow,
                    gt_mask=gt_masks.to(torch.float32), pred_mask=sequence_occlusion,
                )
            )
            return loss, log_imgs, log_vars
        else:
            return loss, None, log_vars
    
    def train_step(self, data_batch, optimizer, **kwargs):
        loss, log_imgs, log_vars = self.loss(data_batch)
        if log_imgs is not None:
            outputs = dict(
                loss = loss,
                log_vars = log_vars,
                log_imgs = log_imgs,
                num_samples = len(data_batch['img_metas']),
            )
        else:
            outputs = dict(
                loss = loss,
                log_vars = log_vars,
                num_samples = len(data_batch['img_metas']),
            )
        return outputs
    
    def eval_epe_and_occ(self, batch_flow, batch_occlusion, rendered_depths, gt_rotations, gt_translations, internel_k, ref_rotations, ref_translations, gt_masks, reduction='mean'):
        metric_dict = OrderedDict()
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        epe = cal_epe(gt_flow, batch_flow, None, reduction=reduction, max_flow=self.max_flow)
        for k in epe:
            metric_dict[f'epe_{k}'] = epe[k]
    

        if gt_masks is not None:
            # calculate the flow of pixels with visible correspondences in the real image
            noc_gt_flow = filter_flow_by_mask(gt_flow, gt_masks, self.max_flow)
            epe_noc = cal_epe(noc_gt_flow, batch_flow, None, reduction=reduction, max_flow=self.max_flow)
            for k in epe_noc:
                metric_dict[f'epe_noc_{k}'] = epe_noc[k]
        else:
            noc_gt_flow = gt_flow

        occlusion_gt = torch.sum(noc_gt_flow**2, dim=1).sqrt() < self.max_flow
        occ = torch.abs(occlusion_gt.to(batch_occlusion.dtype) - batch_occlusion).mean()
        metric_dict['occ'] = occ
        return metric_dict
    
    def val_step(self, data_batch):
        data = self.format_data_test(data_batch)
        pred_flow, pred_occlusion, _ = self.forward_single_view(data, data_batch, return_pose=False)
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        internel_k = data['internel_k']
        rendered_depths = data['rendered_depths']
        gt_masks = data.get('gt_masks', None)

        metric = self.eval_epe_and_occ(
            pred_flow, pred_occlusion, 
            rendered_depths, gt_rotations, gt_translations, internel_k, 
            ref_rotations, ref_translations, gt_masks, reduction='total_mean')
       
        log_vars = OrderedDict()
        for k in metric:
            log_vars[k] = metric[k].item()
        return dict(
            log_vars = log_vars,
            num_samples = len(pred_flow),
        ) 


    
    def forward(self, data_batch, return_loss=False):
        data = self.format_data_test(data_batch)
        return self.forward_single_view(data, data_batch)