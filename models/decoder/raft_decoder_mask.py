# Copyright (c) OpenMMLab. All rights reserved.
import math
from turtle import forward
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule

from models.utils.corr_lookup import CorrLookup

from .builder import DECODERS
from .raft_decoder import CorrelationPyramid, MotionEncoder, XHead, ConvGRU




@DECODERS.register_module()
class RAFTDecoderMask(BaseModule):
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
        corr_lookup_cfg: dict = dict(type='CorrLookup', align_corners=True),
        gru_type: str = 'SeqConv',
        feat_channels: Union[int, Sequence[int]] = 256,
        mask_channels: int = 64,
        convex_unsample_flow: bool=True,
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
        self.h_channels = self._h_channels.get(net_type)
        self.cxt_channels = self._cxt_channels.get(net_type)
        self.iters = iters
        self.mask_channels = mask_channels * (2 * radius + 1)
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
        self.flow_pred = XHead(self.h_channels, feat_channels, 2, x='flow')
        self.occlusion_pred = XHead(self.h_channels, feat_channels, 1, x='mask')

        if net_type == 'Basic':
            self.mask_pred = XHead(
                self.h_channels, feat_channels, self.mask_channels, x='mask')
        self.convex_upsample_flow = convex_unsample_flow

    def make_gru_block(self):
        return ConvGRU(
            self.h_channels,
            self.encoder.out_channels[0] + 2 + self.cxt_channels,
            net_type=self.gru_type)

    def upsample_flow(self,
                  flow: torch.Tensor,
                  mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex
        combination.

        Args:
            flow (Tensor): The optical flow with the shape [N, 2, H/8, W/8].
            mask (Tensor, optional): The leanable mask with shape
                [N, grid_size x scale x scale, H/8, H/8].

        Returns:
            Tensor: The output optical flow with the shape [N, 2, H, W].
        """
        scale = 2**(self.num_levels - 1)
        grid_size = self.radius * 2 + 1
        grid_side = int(math.sqrt(grid_size))
        N, _, H, W = flow.shape
        if mask is None:
            new_size = (scale * H, scale * W)
            return scale * F.interpolate(
                flow, size=new_size, mode='bilinear', align_corners=True)
        # predict a (Nx8×8×9xHxW) mask
        mask = mask.view(N, 1, grid_size, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        # extract local grid with 3x3 side  padding = grid_side//2
        upflow = F.unfold(scale * flow, [grid_side, grid_side], padding=1)
        # upflow with shape N, 2, 9, 1, 1, H, W
        upflow = upflow.view(N, 2, grid_size, 1, 1, H, W)

        # take a weighted combination over the neighborhood grid 3x3
        # upflow with shape N, 2, 8, 8, H, W
        upflow = torch.sum(mask * upflow, dim=2)
        upflow = upflow.permute(0, 1, 4, 2, 5, 3)
        return upflow.reshape(N, 2, scale * H, scale * W)
    
    def upsample_mask(self, 
                    occlusion: torch.Tensor,
                    mask:Optional[torch.Tensor] = None)->torch.Tensor:
        scale = 2**(self.num_levels - 1)
        grid_size = self.radius * 2 + 1
        grid_side = int(math.sqrt(grid_size))
        N, _, H, W = occlusion.shape
        if mask is None:
            new_size = (scale * H, scale * W)
            return F.interpolate(
                occlusion, size=new_size, mode='bilinear', align_corners=True)
        
        mask = mask.view(N, 1, grid_size, scale, scale, H, W)
        mask = torch.softmax(mask, dim=2)

        upocclusion = F.unfold(occlusion, [grid_side, grid_side], padding=1)
        upocclusion = upocclusion.view(N, 1, grid_size, 1, 1, H, W)
        upocclusion = torch.sum(upocclusion * mask, dim=2)
        upocclusion = upocclusion.permute(0, 1, 4, 2, 5, 3)
        return upocclusion.reshape(N, 1, scale*H, scale*W)


    def forward(self, feat1: torch.Tensor, feat2: torch.Tensor,
                flow: torch.Tensor, h_feat: torch.Tensor,
                cxt_feat: torch.Tensor) -> Sequence[torch.Tensor]:
        """Forward function for RAFTDecoder.

        Args:
            feat1 (Tensor): The feature from the first input image.
            feat2 (Tensor): The feature from the second input image.
            flow (Tensor): The initialized flow when warm start.
            h (Tensor): The hidden state for GRU cell.
            cxt_feat (Tensor): The contextual feature from the first image.

        Returns:
            Sequence[Tensor]: The list of predicted optical flow.
        """

        corr_pyramid = self.corr_block(feat1, feat2)
        upflow_preds, upocclusion_preds = [], []
        delta_flow = torch.zeros_like(flow)
        for _ in range(self.iters):
            flow = flow.detach()
            corr = self.corr_lookup(corr_pyramid, flow)
            motion_feat = self.encoder(corr, flow)
            x = torch.cat([cxt_feat, motion_feat], dim=1)
            h_feat = self.gru(h_feat, x)

            delta_flow = self.flow_pred(h_feat)
            flow = flow + delta_flow

            occlusion = self.occlusion_pred(h_feat)
            occlusion = torch.sigmoid(occlusion)
            
            if hasattr(self, 'mask_pred'):
                if self.convex_upsample_flow:
                    mask = .25 * self.mask_pred(h_feat)
                else:
                    mask = None
            else:
                mask = None
            upflow = self.upsample_flow(flow, mask)
            upflow_preds.append(upflow)

            upocclusion = self.upsample_mask(occlusion, mask)
            upocclusion_preds.append(upocclusion)

        return upflow_preds, upocclusion_preds