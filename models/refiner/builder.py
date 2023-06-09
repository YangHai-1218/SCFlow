from mmcv.utils import Registry, build_from_cfg

REFINERS = Registry('refiner')

def build_refiner(cfg):
    return build_from_cfg(cfg, REFINERS)