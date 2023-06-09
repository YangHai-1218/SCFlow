from mmcv.utils import Registry, build_from_cfg

HEAD = Registry('head')

def build_head(cfg):
    return build_from_cfg(cfg, HEAD)