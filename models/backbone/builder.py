from mmcv.utils import build_from_cfg, Registry

BACKBONES = Registry('backbone')

def build_backbone(cfg):
    return build_from_cfg(cfg, BACKBONES)