import argparse
import time
from os import path as osp
from functools import partial

import mmcv
import torch
import random
import numpy as np
from mmcv import Config, DictAction
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from mmcv.runner import init_dist, get_dist_info, load_checkpoint
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel, collate
from datasets import build_dataset
from models import build_refiner
from tools.eval import single_gpu_test, multi_gpu_test

# profile

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test a pose estimator'
    )
    parser.add_argument(
        '--config', help='test config file path', default='configs/flow_refine/mvc_raft_flow.py')
    parser.add_argument(
        '--checkpoint', type=str, help='checkpoint file', default='',)
    parser.add_argument(
        '--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results'
    )
    parser.add_argument(
        '--eval', action='store_true', help='whether to evaluate the results')
    parser.add_argument(
        '--format-only', action='store_true', help='whether to save the results in BOP format')
    parser.add_argument(
        '--save-dir', type=str, default='debug/results', help='directory for saving the formatted results')
    parser.add_argument(
        '--eval-options',
        action=DictAction,
        nargs='+',
        help='custom options for formating results, the key-value pair in xxx=yyy'
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    return args


def build_dataloader(cfg, dataset, distributed, shuffle):
    if cfg.data.get('test_samples_per_gpu', None) is not None:
        samples_per_gpu = cfg.data.test_samples_per_gpu
    else:
        samples_per_gpu = cfg.data.samples_per_gpu
    if distributed:
        rank, world_size = get_dist_info()
        sampler =  DistributedSampler(dataset, world_size, rank, shuffle=shuffle)
        batch_size = samples_per_gpu
        num_workers = cfg.data.workers_per_gpu
    else:
        if shuffle:
            sampler = RandomSampler(dataset)
        else:
            sampler = SequentialSampler(dataset)
        batch_size = samples_per_gpu * cfg.num_gpus
        num_workers = cfg.data.workers_per_gpu * cfg.num_gpus
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        shuffle=False,
    )
    return dataloader

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args  = parse_args()
    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pickle file')
    cfg = Config.fromfile(args.config)
    
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_param', {}))

    rank, _ = get_dist_info()
    if cfg.get('work_dir', None) is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        json_file = osp.join(cfg.work_dir, f'eval_{timestamp}.json') 
    # build dataset 
    dataset = build_dataset(cfg.data.test)
    dataloader = build_dataloader(cfg, dataset, distributed, shuffle=False)
    model = build_refiner(cfg.model)
    checkpoint = load_checkpoint(model, args.checkpoint)
        
    
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        start = time.time()
        outputs = single_gpu_test(model, dataloader, validate=args.eval)
        end = time.time()
        per_image_consume_time = (end - start)/len(dataset)
        print(f"per image consume time:{per_image_consume_time}")
    else:
        model.to(torch.device('cuda'))
        model = MMDistributedDataParallel(
            model, 
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, dataloader, gpu_collect=args.gpu_collect)
    
    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, args.save_dir, time=per_image_consume_time, **kwargs)
        if args.eval:
            eval_kwargs = cfg.evaluation
            eval_kwargs.pop('interval')
            if 'save_best' in eval_kwargs:
                eval_kwargs.pop('save_best')
            if 'rule' in eval_kwargs:
                eval_kwargs.pop('rule')
            metric = dataset.evaluate(outputs, **eval_kwargs)
            if cfg.work_dir is not None and rank == 0:
                mmcv.dump(dict(config=args.config, checkpoint=args.checkpoint, metric=metric), json_file)
                cfg.dump(json_file.replace('json', 'py'))