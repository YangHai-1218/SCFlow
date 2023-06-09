from mmcv.utils import Registry, build_from_cfg

ENCODERS = Registry('encoder')

def build_encoder(cfg):
    return build_from_cfg(cfg, ENCODERS)