from mmcv.utils import Registry, build_from_cfg

DECODERS = Registry('decoder')

def build_decoder(cfg):
    return build_from_cfg(cfg, DECODERS)