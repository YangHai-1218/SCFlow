import torch
from torch.nn import functional as F
from .warp import coords_grid, Warp


def filter_flow_by_mask(flow, gt_mask, invalid_num=400, mode='bilinear', align_corners=False):
    '''Check if flow is valid. 
    When the flow pointed point not in the target image mask or falls out of the target image, the flow is invalid.
    Args:
        flow (tensor): flow from source image to target image, shape (N, 2, H, W)
        mask (tensor): mask of the target image, shape (N, H, W)
    '''
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    mask = gt_mask[:, None].to(flow.dtype)
    grid = coords_grid(flow)
    mask = F.grid_sample(
        mask,
        grid,
        mode=mode,
        padding_mode='zeros',
        align_corners=align_corners
    )
    not_valid_mask = (mask < 0.9) | not_valid_mask[:, None]
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow

def filter_flow_by_depth(
    flow:torch.Tensor, depth1:torch.Tensor, depth0:torch.Tensor, invalid_num=400, thr=0.2):
    # flow is from image 0 to image 1
    # https://github.com/zju3dv/LoFTR/blob/master/src/loftr/utils/geometry.py
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    mask0, mask1 = depth0 > 0, depth1 > 0
    grid = coords_grid(flow)
    depth1_masked, depth0_masked = depth1.clone(), depth0.clone()
    depth1_masked[~mask1.bool()] = 0.
    depth0_masked[~mask0.bool()] = 0.
    depth1_expanded, depth0_expanded = depth1_masked[:, None], depth0_masked[:, None]
    warped_depth = F.grid_sample(depth1_expanded, grid, padding_mode='zeros', mode='bilinear', align_corners=True)
    consistent_mask = ((depth0_expanded - warped_depth).abs() / (depth0_expanded + 0.1)) < thr
    
    not_valid_mask = not_valid_mask[:, None] & (~ consistent_mask)
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow

def filter_flow_by_face_index(
    flow:torch.Tensor, face_index1:torch.Tensor, face_index2:torch.Tensor, invalid_num=400):
    not_valid_mask = (flow[:, 0] >= invalid_num) & (flow[:, 1] >= invalid_num)
    face_index1, face_index2 = face_index1.to(torch.float32), face_index2.to(torch.float32)
    grid = coords_grid(flow)
    face_index1_expanded, face_index2_expanded = face_index1[:, None], face_index2[:, None]
    warped_face_index2 = F.grid_sample(face_index2_expanded, grid, padding_mode='zeros', mode='nearest', align_corners=True)
    consisent_mask = warped_face_index2 == face_index1_expanded
    
    not_valid_mask = not_valid_mask[:, None] | (~ consisent_mask)
    not_valid_mask = not_valid_mask.expand_as(flow)
    flow[not_valid_mask] = invalid_num
    return flow




def cal_epe(flow_tgt, flow_pred, mask, max_flow=400, reduction='mean', threshs=[1, 3, 5]):
    mag = torch.sum(flow_tgt**2, dim=1).sqrt()
    if mask is not None:
        # filter the noisy sample with too large flow and without explicit correspondence
        valid_mask = ((mag < max_flow) & (mask >=0.5))
    else:
        valid_mask = (mag< max_flow)
    flow_error = torch.sum((flow_tgt - flow_pred)**2, dim=1).sqrt()
    if reduction == 'none':
        flow_error = flow_error * valid_mask.to(flow_error)
        return flow_error
    elif reduction == 'mean':
        flow_acc = dict()
        total_valid_pixel_num = valid_mask.sum(dim=(-1, -2)) + 1e-10
        flow_acc['mean'] = (flow_error * valid_mask.to(flow_error)).sum(dim=(-1, -2)) / total_valid_pixel_num
        flow_error[valid_mask] = 1e+8
        for thresh in threshs:
            flow_acc[f'{thresh}px'] = (flow_error < thresh).sum(dim=(-1, -2))/ total_valid_pixel_num 
    elif reduction  == 'total_mean':
        flow_acc = dict()
        total_valid_pixel_num = valid_mask.sum(dim=(-1, -2, -3)) + 1e-10
        flow_acc['mean'] = (flow_error * valid_mask.to(flow_error.dtype)).sum(dim=(-1,-2,-3)) / total_valid_pixel_num
        for thresh in threshs:
            flow_acc[f'{thresh}px'] = (flow_error[valid_mask] < thresh).sum() / total_valid_pixel_num
    return flow_acc

def flow_to_coords(flow: torch.Tensor):
    """Generate shifted coordinate grid based based input flow.
    Args:
        flow (Tensor): Estimated optical flow.
    Returns:
        Tensor: Coordinate that shifted by input flow with shape (B, 2, H, W).
    """
    B, _, H, W = flow.shape
    xx = torch.arange(0, W, device=flow.device, requires_grad=False)
    yy = torch.arange(0, H, device=flow.device, requires_grad=False)
    coords = torch.meshgrid(yy, xx)
    coords = torch.stack(coords[::-1], dim=0).float()
    coords = coords[None].repeat(B, 1, 1, 1) + flow
    return coords