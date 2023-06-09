from typing import Dict, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from .builder import DECODERS
from ..utils import get_flow_from_delta_pose_and_points, get_pose_from_delta_pose, CorrLookup, cal_3d_2d_corr
from .raft_decoder import MotionEncoder, XHead, ConvGRU, CorrelationPyramid
from ..head import build_head





@DECODERS.register_module()
class SCFlowDecoder(BaseModule):
    """The decoder of RAFT Net.

    The decoder of RAFT Net, which outputs list of upsampled flow estimation.

    Args:
        net_type (str): Type of the net. Choices: ['Basic', 'Small'].
        num_levels (int): Number of levels used when calculating
            correlation tensor.
        radius (int): Radius used when calculating correlation tensor.
        iters (int): Total iteration number of iterative update of RAFTDecoder.
        corr_op_cfg (dict): Config dict of correlation operator.
            Default: dict(type='CorrLookup').
        gru_type (str): Type of the GRU module. Choices: ['Conv', 'SeqConv'].
            Default: 'SeqConv'.
        feat_channels (Sequence(int)): features channels of prediction module.
        mask_channels (int): Output channels of mask prediction layer.
            Default: 64.
        conv_cfg (dict, optional): Config dict of convolution layers in motion
            encoder. Default: None.
        norm_cfg (dict, optional): Config dict of norm layer in motion encoder.
            Default: None.
        act_cfg (dict, optional): Config dict of activation layer in motion
            encoder. Default: None.
    """
    _h_channels = {'Basic': 128, 'Small': 96}
    _cxt_channels = {'Basic': 128, 'Small': 64}

    def __init__(
        self,
        net_type: str,
        num_levels: int,
        radius: int,
        iters: int,
        detach_flow: bool,
        detach_mask: bool,
        detach_pose: bool,
        mask_flow: bool,
        mask_corr: bool,
        pose_head_cfg: dict(),
        depth_transform: str='exp',
        detach_depth_for_xy: bool=False,
        corr_lookup_cfg: dict = dict(align_corners=True),
        gru_type: str = 'SeqConv',
        feat_channels: Union[int, Sequence[int]] = 256,
        conv_cfg: Optional[dict] = None,
        norm_cfg: Optional[dict] = None,
        act_cfg: Optional[dict] = None,
    ) -> None:
        super().__init__()
        assert net_type in ['Basic', 'Small']
        assert type(feat_channels) in (int, tuple, list)
        self.corr_block = CorrelationPyramid(num_levels=num_levels)

        feat_channels = feat_channels if isinstance(tuple,
                                                    list) else [feat_channels]
        self.net_type = net_type
        self.num_levels = num_levels
        self.radius = radius
        self.detach_flow = detach_flow
        self.detach_mask = detach_mask
        self.detach_pose = detach_pose
        self.detach_depth_for_xy = detach_depth_for_xy
        self.mask_flow = mask_flow
        self.mask_corr = mask_corr
        self.depth_transform = depth_transform
        self.h_channels = self._h_channels.get(net_type)
        self.cxt_channels = self._cxt_channels.get(net_type)
        self.iters = iters
        corr_lookup_cfg['radius'] = radius
        self.corr_lookup = CorrLookup(**corr_lookup_cfg)
        self.encoder = MotionEncoder(
            num_levels=num_levels,
            radius=radius,
            net_type=net_type,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        self.gru_type = gru_type
        self.gru = self.make_gru_block()
        self.pose_pred = build_head(pose_head_cfg)
        self.flow_pred = XHead(self.h_channels, feat_channels, 2, x='flow')
        self.mask_pred = XHead(self.h_channels, feat_channels, 1, x='mask')
        self.delta_flow_encoder = nn.Sequential(*self.make_delta_flow_encoder(
            2, channels=[128, 64], kernels=[7, 3], paddings=[3, 1], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
        self.mask_encoder = nn.Sequential(*self.make_delta_flow_encoder(
            1, channels=[64, 32], kernels=[3, 3], paddings=[1, 1], conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg))
    

    def make_delta_flow_encoder(self, in_channel, channels, kernels, paddings, conv_cfg, norm_cfg, act_cfg):
        encoder = []

        for ch, k, p in zip(channels, kernels, paddings):

            encoder.append(
                ConvModule(
                    in_channels=in_channel,
                    out_channels=ch,
                    kernel_size=k,
                    padding=p,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
            in_channel = ch
        return encoder



    def make_gru_block(self):
        return ConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 2 + self.cxt_channels,
            net_type=self.gru_type)
        
    def _downsample(self, flow:torch.Tensor, mask:torch.Tensor):
        scale = 2**(self.num_levels - 1)
        N, _, H, W = flow.shape

        mask = mask.view(N, 1, scale*scale, H/scale, W/scale)
        mask = torch.softmax(mask, dim=2)
        
        downflow = F.unfold(flow/scale, [scale, scale], padding=1, stride=scale)
        downflow = downflow.view(N, 2, scale*scale, H/scale, W/scale)
        # shape (N, 2, H/scale, W/scale)
        downflow = torch.sum(mask * downflow, dim=2)
        return downflow





    def forward(self, feat_render: torch.Tensor, feat_real: torch.Tensor,
                h_feat: torch.Tensor, cxt_feat: torch.Tensor,
                ref_rotation: torch.Tensor, ref_translation: torch.Tensor,
                depth: torch.Tensor, internel_k: torch.Tensor, 
                label:torch.Tensor, init_flow:torch.Tensor,
                invalid_flow_num: float) -> Sequence[torch.Tensor]:
        """Forward function for RAFTDecoder.

        Args:
            feat1 (Tensor): The feature from the first input image, shape (N, C, H, W)
            feat2 (Tensor): The feature from the second input image, shape (N, C, H, W).
            h_feat (Tensor): The hidden state for GRU cell, shape (N, C, H, W).
            cxt_feat (Tensor): The contextual feature from the first image, shape (N, C, H, W).
            ref_rotation (Tensor): The rotation which is used to render the renderering image.
            ref_translation (Tensor): The translation which is used to render the rendering image.
            depth (Tensor): The depth for rendering images.
            internel_k (Tensor): The camera parameters.
            label (Tensor): The label for training.

        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """
        corr_pyramid = self.corr_block(feat_render, feat_real)
        update_rotation = ref_rotation
        update_translation = ref_translation
        rotation_preds, translation_preds = [], []
        delta_rotation_preds, delta_translation_preds = [], []
        flow_from_pose, flow_from_pred = [], []
        mask_preds = []
        scale = 2**(self.num_levels - 1)
        N, H, W = depth.size()
        flow = init_flow
        
        points_2d_list, points_3d_list = [], []
        for i in range(N):
            points_2d, points_3d = cal_3d_2d_corr(depth[i], internel_k[i], ref_rotation[i], ref_translation[i])
            points_2d_list.append(points_2d)
            points_3d_list.append(points_3d)
        init_mask = torch.ones((N, 1, H, W), dtype=init_flow.dtype, device=init_flow.device)
        init_mask = F.interpolate(init_mask, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)
        mask = init_mask
        for i in range(self.iters):
            if self.detach_flow:
                flow = flow.detach()
            if self.detach_mask:
                mask = mask.detach()
            flow = 1/scale * F.interpolate(
                flow, scale_factor=(1/scale, 1/scale), mode='bilinear', align_corners=True)
            corr = self.corr_lookup(corr_pyramid, flow)
            # mask occluded pixels for correlation
            if self.mask_corr:
                corr = corr * mask
            # mask occluded pixels for currently estimated flow
            if self.mask_flow:
                motion_feat = self.encoder(corr, flow * mask)
            else:
                motion_feat = self.encoder(corr, flow)
            x = torch.cat([cxt_feat, motion_feat], dim=1)
            h_feat = self.gru(h_feat, x)
            # predict delta flow
            delta_flow_pred = self.flow_pred(h_feat)
            # predict mask
            mask = self.mask_pred(h_feat)
            mask = torch.sigmoid(mask)

            # predict delta pose from the concatentation of delta flow feat, hidden feat and mask feat
            delta_flow_feat = self.delta_flow_encoder(delta_flow_pred)
            mask_feat = self.mask_encoder(mask)
            delta_rotation, delta_translation = self.pose_pred(
                torch.cat([h_feat, delta_flow_feat, mask_feat], axis=1), label)
            
            # construct predict flow
            flow_pred = flow + delta_flow_pred
            flow_pred = scale * F.interpolate(
                flow_pred, scale_factor=(scale, scale), mode='bilinear', align_corners=True)

            upsample_mask_pred = F.interpolate(
                mask, scale_factor=(scale, scale), mode='bilinear', align_corners=True)
            
            # compute updated pose
            update_rotation, update_translation = get_pose_from_delta_pose(
                delta_rotation, delta_translation, 
                update_rotation.detach() if self.detach_pose else update_rotation,
                update_translation.detach() if self.detach_pose else update_translation,
                depth_transform=self.depth_transform, 
                detach_depth_for_xy=self.detach_depth_for_xy
            )

            # render flow
            flow = get_flow_from_delta_pose_and_points(
                update_rotation, update_translation, internel_k, 
                points_2d_list, points_3d_list, H, W, 
                invalid_num=invalid_flow_num
            )
            rotation_preds.append(update_rotation)
            translation_preds.append(update_translation)
            delta_rotation_preds.append(delta_rotation)
            delta_translation_preds.append(delta_translation)
            flow_from_pose.append(flow)
            flow_from_pred.append(flow_pred)
            mask_preds.append(upsample_mask_pred)
        return flow_from_pose, flow_from_pred, rotation_preds, translation_preds, mask_preds, delta_rotation_preds, delta_translation_preds