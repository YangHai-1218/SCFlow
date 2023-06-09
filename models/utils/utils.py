from functools import partial
import torch
from torch import distributed as dist, imag
import numpy as np
from mmcv.ops import Correlation

from models.utils.warp import Warp

def multi_apply(func, *args, **kwargs):
    """Apply function to a list of arguments.

    Note:
        This function applies the ``func`` to multiple inputs and
        map the multiple outputs of the ``func`` into different
        list. Each list contains the same type of outputs corresponding
        to different inputs.

    Args:
        func (Function): A function that will be applied to a list of
            arguments

    Returns:
        tuple(list): A tuple containing multiple list, each list contains \
            a kind of returned results by the function
    """
    pfunc = partial(func, **kwargs) if kwargs else func
    map_results = map(pfunc, *args)
    return tuple(map(list, zip(*map_results)))

def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    target = torch.stack(target, 0)
    level_targets = []
    start = 0
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets

def random_sample(seq, sample_num):
    '''
    Randomly sample 'sample_num' elements from seq, based on torch.randperm
    Note: Use >1.9 version pytorch, https://github.com/pytorch/pytorch/issues/63726  
    '''
    total_num = seq.size(0)
    if sample_num < 0:
        # if sample num < 0, return an empty tensor
        return seq.new_zeros((0, ))
    if total_num > sample_num:
        random_inds = torch.randperm(total_num, device=seq.device)
        smapled_inds = random_inds[:sample_num]
        return seq[smapled_inds]
    else:
        return seq
    

def reduce_mean(tensor):
    if not (dist.is_initialized() and dist.is_available()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor


def tensor_image_to_cv2(images:torch.Tensor):
    if images.ndim == 4:
        images = images.permute(0, 2, 3, 1).cpu().data.numpy()
        images = np.ascontiguousarray(images[..., ::-1] * 255).astype(np.uint8)
        return images
    elif images.ndim == 3:
        images = images.cpu().data.numpy()
        images = np.ascontiguousarray(images * 255).astype(np.uint8)
        return images


def simple_forward_warp(images, flow, mask, background_color=(0.5, 0.5, 0.5)):
    warped_images = torch.zeros_like(images)
    warped_images[:, 0] = background_color[0]
    warped_images[:, 1] = background_color[1]
    warped_images[:, 2] = background_color[2]
    height, width = images.size(2), images.size(3)
    num_images = len(images)
    for i in range(num_images): 
        mask_i, flow_i, image_i = mask[i], flow[i], images[i]
        points_y, points_x = torch.nonzero(mask_i, as_tuple=True)
        points_flow = flow_i[:, mask_i.to(torch.bool)]
        warped_x, warped_y = points_x+points_flow[0, :], points_y+points_flow[1, :]
        points_color = image_i[:, mask_i.to(torch.bool)]
        warped_y = torch.clamp(warped_y, min=0, max=height-1)
        warped_x = torch.clamp(warped_x, min=0, max=width-1)
        warped_images[i, :, warped_y.to(torch.int64), warped_x.to(torch.int64)] = points_color
    return warped_images