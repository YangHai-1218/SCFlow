from typing import Dict, Optional, Tuple, Union, Sequence
import torch
import torch.nn as nn
from collections import OrderedDict

from .builder import REFINERS
from .base_flow_refiner import BaseFlowRefiner
from ..encoder import build_encoder
from ..loss import build_loss
from ..utils import (
    get_flow_from_delta_pose_and_depth, filter_flow_by_mask, 
    remap_pose_to_origin_resoluaion)
import numpy as np




@REFINERS.register_module()
class RAFTRefinerFlow(BaseFlowRefiner):
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
                 loss_cfg: Optional[dict]=None,
                 max_flow: float=400.,
                 render_augmentations: Optional[Sequence]=None,
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

        assert self.h_channels == self.decoder.h_channels
        assert self.cxt_channels == self.decoder.cxt_channels
        assert self.h_channels + self.cxt_channels == self.context.out_channels

        if loss_cfg is not None:
            self.loss_func = build_loss(loss_cfg)
        if freeze_bn:
            self.freeze_bn()
        self.test_iter_num = self.test_cfg.get('iters') if 'iters' in self.test_cfg else self.decoder.iters
        

    def freeze_bn(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def measure_runtime(self):
        super().measure_runtime()
        self.max_runtime, self.min_runtime = 0, 1000000
        self.total_runtime = 0
        self.feat_time = 0
        self.feat_cxt_time = 0
        self.iter_time = 0
        self.solve_pose_time = 0
        self.runtime_record = []
    
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

    def get_flow(
            self,
            render_images,
            real_images,
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
        rendered_masks = data['rendered_masks']

        iters = self.decoder.iters
        self.decoder.iters = self.test_iter_num
        sequence_flow = self.get_flow(rendered_images, real_images)
        self.decoder.iters = iters
       
        batch_flow = sequence_flow[-1]
        if return_pose:
            results = self.solve_pose(
                batch_flow, rendered_depths, ref_rotations, ref_translations, internel_k, labels, per_img_patch_num)
            if self.test_cfg.get('vis_result', False):
                self.visualize_and_save(data, sequence_flow, torch.cat(results['rotations']), torch.cat(results['translations']))
            if self.test_cfg.get('vis_seq_flow', False):
                self.visualize_sequence_flow_and_fw(data, sequence_flow)
            batch_internel_k = torch.split(internel_k, per_img_patch_num)
            batch_rotations, batch_translations = results['rotations'], results['translations']
            image_metas = data_batch['img_metas']
            batch_rotations, batch_translations = remap_pose_to_origin_resoluaion(batch_rotations, batch_translations, batch_internel_k, image_metas)
            results['rotations'] = batch_rotations
            results['translations'] = batch_translations
            return results
        else:
            return batch_flow, data


            
            


    
    def loss(self, data_batch):
        data = self.format_data_train_sup(data_batch)
        log_vars = OrderedDict()
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        real_images, rendered_images = data['real_images'], data['rendered_images']
        internel_k, rendered_depths, rendered_masks = data['internel_k'], data['rendered_depths'], data['rendered_masks']

        # get flow
        sequence_flow, feat_render, feat_real= self.get_flow(rendered_images, real_images)
        gt_flow = get_flow_from_delta_pose_and_depth(
            ref_rotations, ref_translations,
            gt_rotations, gt_translations,
            rendered_depths, internel_k, invalid_num=self.max_flow
        )
        if self.filter_invalid_flow:
            gt_masks = data['gt_masks']
            gt_flow = filter_flow_by_mask(gt_flow, gt_masks, invalid_num=self.max_flow)

        loss, seq_loss_list = self.loss_func(
            sequence_flow, gt_flow=gt_flow, valid=rendered_masks
        )
        
        for i in range(len(seq_loss_list)):
            log_vars.update({f'seq_{i}_loss':seq_loss_list[i].item()})
        
        pred_flow = sequence_flow[-1]
        pred_flow = pred_flow*rendered_masks[:, None]
        log_imgs = self.add_vis_images(
            **dict(
                real_images=real_images, render_images=rendered_images,
                gt_flow=gt_flow, pred_flow=pred_flow
            )
        )
        log_vars.update(loss=loss.item())
        return loss, log_imgs, log_vars
    
    def train_step(self, data_batch, optimizer, **kwargs):
        loss, log_imgs, log_vars = self.loss(data_batch)
        outputs = dict(
            loss = loss,
            log_vars = log_vars,
            log_imgs = log_imgs,
            num_samples = len(data_batch['img_metas']),
        )
        return outputs

    
    def forward(self, data_batch, return_loss=False):
        data = self.format_data_test(data_batch)
        return self.forward_single_view(data, data_batch)
