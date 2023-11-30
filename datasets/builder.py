from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import ConcatDataset

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

def build_dataset(cfg):
    if cfg.type == 'ConcatDataset':
        cfg.datasets = [build_from_cfg(d, DATASETS) for d in cfg.datasets]
        return build_from_cfg(cfg, DATASETS)
    else:
        return build_from_cfg(cfg, DATASETS)
