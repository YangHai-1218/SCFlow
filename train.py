import os, time
import argparse
from functools import partial
from os import path as osp
import warnings
import torch
import mmcv
from mmcv import Config
from mmcv.utils import get_logger
from mmcv.parallel import collate
from mmcv.parallel.data_parallel import MMDataParallel
from mmcv.parallel.distributed import MMDistributedDataParallel
from mmcv.runner import (
    build_runner, get_dist_info, build_optimizer, init_dist, 
    EvalHook, DistEvalHook, Fp16OptimizerHook)
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler

from models import build_refiner
from datasets import build_dataset, MultiSourceSampler
from tools.eval import single_gpu_test, multi_gpu_test





def build_eval_hook(cfg, distributed, dataloader):
    eval_cfg = cfg.get('evaluation', {})
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_hook = DistEvalHook if distributed else EvalHook
    test_fn = multi_gpu_test if distributed else single_gpu_test
    eval_hook = eval_hook(dataloader, test_fn=test_fn, **eval_cfg)
    return eval_hook   


def parse_args():
    parser = argparse.ArgumentParser(description='Train a pose refiner')
    parser.add_argument('--config', default='configs/refine_models/scflow.py', help='train config file path')
    parser.add_argument('--work-dir', type=str, help='working dir')
    parser.add_argument('--resume-from', type=str)
    parser.add_argument('--launcher', default='none', choices=['none', 'slurm', 'mpi', 'pytorch'], help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def build_dataloader(cfg, dataset, dataset_cfg, distributed, shuffle):
    if dataset_cfg.get('multisourcesample', None) is not None:
        sampler = MultiSourceSampler(
            dataset, cfg.data.samples_per_gpu, dataset_cfg.multisourcesample.source_ratio, shuffle, seed=1218)
        dataloader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=cfg.data.samples_per_gpu * cfg.num_gpus,
            num_workers=cfg.data.workers_per_gpu * cfg.num_gpus,
            collate_fn=partial(collate, samples_per_gpu=cfg.data.samples_per_gpu),
            shuffle=False,
            persistent_workers=True
        )
    else:
        if distributed:
            rank, world_size = get_dist_info()
            sampler =  DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
            batch_size = cfg.data.samples_per_gpu
            num_workers = cfg.data.workers_per_gpu
        else:
            if shuffle:
                sampler = RandomSampler(dataset)
            else:
                sampler = SequentialSampler(dataset)
            batch_size = cfg.data.samples_per_gpu * cfg.num_gpus
            num_workers = cfg.data.workers_per_gpu * cfg.num_gpus
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=partial(collate, samples_per_gpu=cfg.data.samples_per_gpu),
            shuffle=False,
            persistent_workers=False
        )
    return dataloader



if __name__ == '__main__':
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    args = parse_args()

    cfg_path = args.config
    launcher = args.launcher
    
    cfg = Config.fromfile(cfg_path)
    if launcher != 'none':
        distributed = True
        init_dist(launcher, **cfg.get('dist_param', {}))
        _, world_size = get_dist_info()
    else:
        distributed = False


    # create work dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(cfg_path)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_logger('Flow-6D', log_file=log_file)

    # log some basic info
    if distributed:
        logger.info(f"Distributed training: {distributed}, {world_size} GPUS using")
    else:
        logger.info(f"Distributed training: {distributed}")

    # build model
    model = build_refiner(cfg.model)
    # init weights
    model.init_weights()

    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model.to(torch.device('cuda'))
        model = MMDistributedDataParallel(
            model,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters
        )
    else:
        gpu_ids = list(range(cfg.num_gpus))
        model = MMDataParallel(model.cuda(gpu_ids[0]), device_ids=gpu_ids)
    
    # build optimizer 
    optimizer = build_optimizer(model, cfg.optimizer)

    # fp16 setting
    fp16_config = cfg.get('fp16', None)
    if fp16_config is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_config, distributed=distributed
        )
    else:
        optimizer_config = cfg.optimizer_config

    # build Runner
    runner = build_runner(cfg.runner, 
                        default_args=dict(
                            model=model,
                            optimizer=optimizer,
                            work_dir=cfg.work_dir,
                            logger=logger,
                            meta=None,
                        ))
    
    # register hooks
    runner.register_training_hooks(cfg.lr_config, 
                                    optimizer_config, 
                                    cfg.checkpoint_config,
                                    cfg.log_config,
                                    cfg.get('momentum_config', None),
                                    custom_hooks_config=cfg.get('custom_hooks', None))
    
    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif args.resume_from is not None:
        runner.resume(args.resume_from)
    elif cfg.get('load_from', None):
        runner.load_checkpoint(cfg.load_from)
    
    # build dataset
    train_dataset = build_dataset(cfg.data.train)
    # logger.info(f'Dataset Info:{repr(train_dataset)}')
    datasets = [train_dataset]
    # build dataloader
    dataloaders = [build_dataloader(cfg, ds, cfg.data.train ,distributed, shuffle=True) for ds in datasets]

    # register validation hook
    if cfg.get('evaluation', False):
        if cfg.data.get('test_samples_per_gpu', None) is not None:
            logger.info(f"Number of samples-per-gpu for validation is set to {cfg.data.test_samples_per_gpu}")
            samples_per_gpu = cfg.data.samples_per_gpu
            cfg.data.samples_per_gpu = cfg.data.test_samples_per_gpu
        val_dataset = build_dataset(cfg.data.val)
        val_dataloader = build_dataloader(
            cfg,
            val_dataset,
            cfg.data.val,
            distributed,
            shuffle=False
        )
        eval_hook = build_eval_hook(cfg, distributed, val_dataloader)
        runner.register_hook(eval_hook, priority='LOW')
    

    work_flow = cfg.get('work_flow', [('train', 1)])
    if len(work_flow) == 2:
        # do validation using val_step, we need to build validation dataloader
        val_dataset = build_dataset(cfg.data.val)
        val_dataloader = build_dataloader(
            cfg,
            val_dataset,
            distributed,
            shuffle=True
        )
        dataloaders.append(val_dataloader)
    runner.run(dataloaders, work_flow)
    