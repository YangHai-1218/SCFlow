from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from .builder import REFINERS
from .base_refiner import BaseRefiner
from ..encoder import build_encoder
from ..loss import build_loss, RAFTLoss
from models.utils import (
    get_flow_from_delta_pose_and_depth,
    remap_pose_to_origin_resoluaion, 
    filter_flow_by_mask, cal_3d_2d_corr)




@REFINERS.register_module()
class SCFlowRefiner(BaseRefiner):
    def __init__(self,
                 seperate_encoder: bool,
                 cxt_channels: int,
                 h_channels: int,
                 cxt_encoder: dict,
                 encoder: dict,
                 decoder: dict,
                 renderer: dict,
                 pose_loss_cfg: dict,
                 flow_loss_cfg: dict,
                 mask_loss_cfg: dict,
                 max_flow: float = 400,
                 render_augmentations:list = None,
                 filter_invalid_flow: bool = True,
                 freeze_encoder: bool = False,
                 freeze_bn: bool = False,
                 train_cfg: Optional[dict] = None,
                 test_cfg: Optional[dict] = None,
                 init_cfg: Optional[Union[list, dict]] = None) -> None:
        super().__init__(
            encoder=encoder, 
            decoder=decoder, 
            seperate_encoder=seperate_encoder, 
            renderer=renderer, 
            render_augmentations=render_augmentations,
            max_flow=max_flow,
            train_cfg=train_cfg, 
            test_cfg=test_cfg,
            init_cfg=init_cfg)
        
        self.context = build_encoder(cxt_encoder)
        self.h_channels = h_channels
        self.cxt_channels = cxt_channels


        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels

        self.pose_loss_func = build_loss(pose_loss_cfg)
        self.flow_loss_func = build_loss(flow_loss_cfg)
        self.mask_loss_func = build_loss(mask_loss_cfg)
        if freeze_bn:
            self.freeze_bn()
        if freeze_encoder:
            self.freeze_encoder()
        self.filter_invalid_flow = filter_invalid_flow
        self.test_by_flow = self.test_cfg.get('by_flow', False)
        self.test_iter_num = self.test_cfg.get('iters') if 'iters' in self.test_cfg else self.decoder.iters
    
    def freeze_encoder(self):
        for m in self.real_encoder.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        for m in self.render_encoder.modules():
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def to(self, device):
        super().to(device)
        self.pose_loss_func.to(device)

    def extract_feat(
        self, render_images, real_images,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from images.

        Args:
            imgs (Tensor): The concatenated input images.

        Returns:
            Tuple[Tensor, Tensor, Tensor, Tensor]: The feature from the first
                image, the feature from the second image, the hidden state
                feature for GRU cell and the contextual feature.
        """
        real_feat = self.real_encoder(real_images)
        render_feat = self.render_encoder(render_images)
        cxt_feat = self.context(render_images)

        h_feat, cxt_feat = torch.split(
            cxt_feat, [self.h_channels, self.cxt_channels], dim=1)
        h_feat = torch.tanh(h_feat)
        cxt_feat = torch.relu(cxt_feat)

        return render_feat, real_feat, h_feat, cxt_feat

    def get_pose(
            self,
            render_images,
            real_images,
            ref_rotation,
            ref_translation,
            depth,
            internel_k,
            label,
            init_flow=None,
    ) -> Dict[str, torch.Tensor]:
        """Forward function for RAFT when model training.

        Args:
            imgs (Tensor): The concatenated input images.
            flow_init (Tensor, optional): The initialized flow when warm start.
                Default to None.

        Returns:
            Dict[str, Tensor]: The losses of output.
        """

        feat_render, feat_real, h_feat, cxt_feat = self.extract_feat(render_images, real_images)
        if init_flow is None:
            N, _, H, W = real_images.shape
            init_flow = feat_render.new_zeros((N, 2, H, W), dtype=torch.float32, device=feat_render.device)
        return self.decoder(
            feat_render, feat_real, h_feat, cxt_feat,
            ref_rotation, ref_translation,
            depth, internel_k, init_flow=init_flow, 
            label=label, invalid_flow_num=0.)


    
    def forward_single_pass(self, data, data_batch, return_loss=False):
        labels = data['labels']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        internel_k, rendered_depths, rendered_masks = data['internel_k'], data['rendered_depths'], data['rendered_masks']
        per_img_patch_num = data['per_img_patch_num']


        iters = self.decoder.iters
        self.decoder.iters = self.test_iter_num
        sequence_flow_from_pose, sequence_flow_from_pred, seq_rotations, seq_translations, sequence_masks, seq_delta_rotations, seq_delta_translations = \
            self.get_pose(
                rendered_images, real_images,
                ref_rotations, ref_translations, 
                rendered_depths, internel_k, labels
            )
        self.decoder.iters = iters

        batch_rotations = seq_rotations[-1]
        batch_translations = seq_translations[-1]
        batch_rotations = torch.split(batch_rotations, per_img_patch_num)
        batch_translations = torch.split(batch_translations, per_img_patch_num)
        batch_labels = torch.split(labels, per_img_patch_num)
        batch_scores = torch.split(torch.ones_like(labels, dtype=torch.float32), per_img_patch_num)
        image_metas = data_batch['img_metas']
        batch_internel_k = torch.split(internel_k, per_img_patch_num)
        
        batch_rotations, batch_translations = remap_pose_to_origin_resoluaion(batch_rotations, batch_translations, batch_internel_k, image_metas)
        return dict(
            rotations = batch_rotations,
            translations = batch_translations,
            labels = batch_labels,
            scores = batch_scores,
        )

    


    def loss(self, data_batch):
        log_vars = OrderedDict()
        data = self.format_data_train_sup(data_batch)
        init_add_error_mean, init_add_error_std = data['init_add_error_mean'], data['init_add_error_std']
        init_log_info = dict(init_add_mean=init_add_error_mean.item(), init_add_std=init_add_error_std.item())
        log_vars.update(init_log_info)

        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        rendered_depths, rendered_masks = data['rendered_depths'], data['rendered_masks']
        internel_k, labels = data['internel_k'], data['labels']
        gt_masks = data['gt_masks']
        # get flow
        sequence_flow_from_pose, sequence_flow_from_pred, seq_rotations, seq_translations, sequence_masks, seq_delta_rotations, seq_delta_translations = \
            self.get_pose(
                rendered_images, real_images,
                ref_rotations, ref_translations, 
                rendered_depths, internel_k, labels
            )
        gt_flow = get_flow_from_delta_pose_and_depth(
            ref_rotations, ref_translations,
            gt_rotations, gt_translations,
            rendered_depths, internel_k, invalid_num=self.max_flow
        )
        if self.filter_invalid_flow:
            gt_flow = filter_flow_by_mask(gt_flow, gt_masks, invalid_num=self.max_flow)

        if not isinstance(self.pose_loss_func.loss_func, RAFTLoss):
            img_metas = data_batch['img_metas']
            # fixed scale for each image
            scale_factors = torch.from_numpy(
                np.concatenate([img_meta['scale_factor'][..., 0] for img_meta in img_metas])).to(gt_flow.device)
            loss_pose, seq_pose_loss_list = self.pose_loss_func(
                seq_rotations, seq_translations, 
                gt_r=gt_rotations, gt_t=gt_translations, 
                labels=labels, scale_factors=scale_factors
            )
        else:
            loss_pose, seq_pose_loss_list = self.pose_loss_func(
                sequence_flow_from_pose, gt_flow=gt_flow, valid=rendered_masks 
            )
        
        loss_flow, seq_flow_loss_list = self.flow_loss_func(
            sequence_flow_from_pred, gt_flow=gt_flow, valid=rendered_masks
        )
        gt_occlusion_mask = (torch.sum(gt_flow, dim=1, keepdim=False) < self.max_flow).to(torch.float32)
        sequence_masks = [mask.squeeze(dim=1) for mask in sequence_masks]
        loss_mask, seq_mask_loss_list = self.mask_loss_func(
            sequence_masks, gt_mask=gt_occlusion_mask, valid=rendered_masks
        )

        
        for seq_i in range(len(seq_flow_loss_list)):
            i_flow_loss = seq_flow_loss_list[seq_i].item()
            i_pose_loss = seq_pose_loss_list[seq_i].item()
            i_mask_loss = seq_mask_loss_list[seq_i].item()
            log_vars.update({f'seq_{seq_i}_pose_loss': i_pose_loss})
            log_vars.update({f'seq_{seq_i}_flow_loss': i_flow_loss})
            log_vars.update({f'seq_{seq_i}_mask_loss': i_mask_loss})

        
    
        loss = loss_pose + loss_flow + loss_mask
        pred_pose_flow = sequence_flow_from_pose[-1]
        pred_flow = sequence_flow_from_pred[-1]
        pred_flow = pred_flow * rendered_masks[:, None]
        log_imgs = self.add_vis_images(
            **dict(
                real_images=real_images, render_images=rendered_images,
                gt_flow=gt_flow, pose_flow=pred_pose_flow, pred_flow=pred_flow,
            )
        )
        log_vars.update({'loss_mask':loss_mask.item(), 'loss_flow':loss_flow.item(), 'loss_pose':loss_pose.item(), 'loss':loss.item()})
        return loss, log_imgs, log_vars, seq_rotations, seq_translations

