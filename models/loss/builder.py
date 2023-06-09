from mmcv.utils import Registry, build_from_cfg

LOSSES = Registry('loss')

def build_loss(cfg):
    return build_from_cfg(cfg, LOSSES)