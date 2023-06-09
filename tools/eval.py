import torch
import mmcv
import numpy as np
import shutil
import pickle
import tempfile
import random
from pathlib import Path
from os import path as osp
from torch import distributed as dist
from mmcv.runner import get_dist_info
from mmcv.image import tensor2imgs
import pycocotools.mask as mask_util







def format_result(batch_preds):

    results_batch = []
    # choose a key
    random_key = random.choice(list(batch_preds.keys()))

    batch_size = len(batch_preds[random_key])
    for i in range(batch_size):
        result_dict = {}
        for key in batch_preds.keys():
            result_dict[key] = batch_preds[key][i].cpu().numpy()
        results_batch.append(result_dict)
    return results_batch




def single_gpu_test(model,
                    data_loader,
                    k=[1066.778, 0.0, 312.9869, 0.0, 1067.487, 241.3109, 0.0, 0.0, 1.0],
                    # k=[572.4114, 0.0, 325.2611, 0.0, 573.57043, 242.04899, 0.0, 0.0, 1.0],
                    validate=True):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        if not getattr(model.module, 'requires_grad_when_eval', False):
            with torch.no_grad():
                # call forward function
                batch_preds = model(data, return_loss=False)
        else:
            batch_preds = model(data, return_loss=False)
        
        result = format_result(batch_preds)
        batch_size = len(result)        
        if validate:
            result_new_list = []
            img_metas = data['img_metas'].data[0]
            annots = data['annots']
            for i in range(batch_size):
                result_new = dict(pred=result[i], gt={})
                result_new['img_metas'] = img_metas[i]
                result_new_list.append(result_new)
            result = result_new_list
        else:
            result_new_list = []
            img_metas = data['img_metas'].data[0]
            for i in range(batch_size):
                result_new = dict(
                    pred=result[i], 
                    img_metas=img_metas[i],
                )
                result_new_list.append(result_new)
            result = result_new_list

        results.extend(result)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model,
                  data_loader,
                  validate=True,
                  tmpdir=None,
                  gpu_collect=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            batch_preds = model(data, return_loss=False)
        result = format_result(batch_preds)
        batch_size = len(result)

        if validate:
            result_new_list = []
            img_metas = data['img_metas'].data[0]
            for i in range(batch_size):
                result_new = dict(pred=result[i], gt={})
                keys = ['labels', 'rotations', 'translations']
                if 'ori_gt_labels' in img_metas[i]:
                    gt_obj_num = len(img_metas[i]['ori_gt_labels'])
                    for key in keys:
                        result_new['gt'][key] = img_metas[i]['ori_gt_' + key]
                    gt_k = np.repeat(img_metas[i]['ori_k'][None], repeats=gt_obj_num, axis=0)
                    result_new['gt']['k'] = gt_k
                result_new['img_metas'] = img_metas[i]
                result_new_list.append(result_new)
            result = result_new_list
        else:
            result_new_list = []
            img_metas = data['img_metas'].data[0]
            for i in range(batch_size):
                result_new = dict(
                    pred=result[i], 
                    img_metas=img_metas[i],
                )
                result_new_list.append(result_new)
            result = result_new_list
        results.extend(result)

        if rank == 0:
            for _ in range(batch_size * world_size):
                prog_bar.update()
    
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results






def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
        
    
def intersect_and_union(pred_mask,
                        gt_mask):
    """Calculate intersection and Union.
    Args:
        pred_label (ndarray | str): Prediction segmentation map
            or predict result filename.
        label (ndarray | str): Ground truth segmentation map
            or label filename.
        num_classes (int): Number of categories.
        ignore_index (int): Index that will be ignored in evaluation.
        label_map (dict): Mapping old labels to new labels. The parameter will
            work only when label is str. Default: dict().
        reduce_zero_label (bool): Whether ignore zero label. The parameter will
            work only when label is str. Default: False.
     Returns:
         torch.Tensor: The intersection of prediction and ground truth
            histogram on all classes.
         torch.Tensor: The union of prediction and ground truth histogram on
            all classes.
         torch.Tensor: The prediction histogram on all classes.
         torch.Tensor: The ground truth histogram on all classes.
    """
    if isinstance(gt_mask, BitmapMasks):
        gt_mask = gt_mask.masks
    elif isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.numpy()
    
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.numpy()
    
    if gt_mask.dtype != np.bool_:
        gt_mask = gt_mask.astype(np.bool_)
    
    expanded_gt_mask = np.expand_dims(gt_mask, axis=1)
    expanded_pred_mask = np.expand_dims(pred_mask, axis=0)

    intersect = expanded_pred_mask & expanded_gt_mask
    area_intersect = intersect.sum(axis=(-1, -2))
    
    area_pred_mask = pred_mask.sum(axis=(-1, -2))
    area_gt_mask = gt_mask.sum(axis=(-1, -2))
    area_union = area_gt_mask[..., None] + area_pred_mask[None] - area_intersect

    return area_intersect, area_union, area_pred_mask, area_gt_mask