from math import sqrt

import torch
from mmcv.cnn import build_activation_layer
from mmcv.runner import BaseModule
from mmcv.ops import Correlation


class CorrBlock(BaseModule):
    """Basic Correlation Block.

    A block used to calculate correlation.

    Args:
        corr (dict): Config dict for build correlation operator.
        act_cfg (dict): Config dict for activation layer.
        normalize (bool): Whether to normalize features.
        scaled (bool): Whether to use scaled correlation by the number of
            elements involved to calculate correlation or not.
            Default: False.
        scale_mode (str): How to scale correlation. The value includes
        `'dimension'` and `'sqrt dimension'`, but it doesn't work when
        scaled = True. Default to `'dimension'`.
    """

    def __init__(self,
                 corr_cfg: dict,
                 act_cfg: dict = dict(type='LeakyReLU', negative_slope=0.1),
                 normalize_cfg = dict(normalize=False, center=False, across_channels=True, across_images=True),
                 scaled: bool = False,
                 scale_mode: str = 'dimension') -> None:
        super().__init__()

        assert scale_mode in ('dimension', 'sqrt dimension'), (
            'scale_mode must be \'dimension\' or \'sqrt dimension\' '
            f'but got {scale_mode}')

        corr = Correlation(**corr_cfg)
        act = build_activation_layer(act_cfg)
        self.scaled = scaled
        self.scale_mode = scale_mode
        self.normalize_cfg  = normalize_cfg
        self.kernel_size = corr.kernel_size
        self.corr_block = [corr, act]
        self.stride = corr_cfg.get('stride', 1)

    def normalize_feature(self, feature):
        ''' Northrmalize feature by its mean and std
        
        Args:
            feature (torch.Tensor): shape (B, C, H, W)
        '''
        if self.normalize_cfg.get('normalize', False) == False and self.normalize_cfg.get('center', False) == False:
            return feature
        
        axes = [-1, -2]
        if self.normalize_cfg.get('across_channels'):
            axes.append(-3)
        if self.normalize_cfg.get('across_images'):
            axes.append(-4)
        axes = tuple(axes)
        
        mean = torch.mean(feature, dim=axes, keepdim=True)
        std = torch.std(feature, dim=axes, keepdim=True)
        if self.normalize_cfg.get('center'):
            feature = feature - mean
        
        if self.normalize_cfg.get('normalize'):
            feature = feature / (std + 1e-8)
        return feature


    def forward(self, feat1: torch.Tensor,
                feat2: torch.Tensor) -> torch.Tensor:
        """Forward function for CorrBlock.

        Args:
            feat1 (Tensor): The feature from the first image.
            feat2 (Tensor): The feature from the second image.

        Returns:
            Tensor: The correlation between feature1 and feature2.
        """
        N, C, H, W = feat1.shape
        scale_factor = 1.

        feat1 = self.normalize_feature(feat1)
        feat2 = self.normalize_feature(feat2)

        if self.scaled:

            if 'sqrt' in self.scale_mode:
                scale_factor = sqrt(float(C * self.kernel_size**2))
            else:
                scale_factor = float(C * self.kernel_size**2)

        corr = self.corr_block[0](feat1, feat2) / scale_factor

        corr = corr.view(N, -1, H // self.stride, W // self.stride)

        out = self.corr_block[1](corr)

        return out

    def __repr__(self):
        s = super().__repr__()
        s += f'\nscaled={self.scaled}'
        s += f'\nscale_mode={self.scale_mode}'
        return s
