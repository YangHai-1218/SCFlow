from mmcv.utils import Registry, build_from_cfg


PIPELINES = Registry('pipeline')

def build_pipeline(cfg):
    return build_from_cfg(cfg, PIPELINES)