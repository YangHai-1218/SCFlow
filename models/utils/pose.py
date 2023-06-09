from typing import Sequence, Optional
import torch
from torch.nn import functional as F
import cv2
import numpy as np
import kornia
from datasets.pose import remap_pose


def interpolate_depth(points_x, points_y, depth):
    h, w = depth.shape
    normalized_points_x = points_x * 2 / w - 1.
    normalized_points_y = points_y * 2 / h - 1.
    grid = torch.stack([normalized_points_x, normalized_points_y], dim=-1)
    depth_interploated = F.grid_sample(
        depth[None, None],
        grid[None, None],
        mode='bilinear',
        padding_mode='zeros'
    )
    depth_interploated = depth_interploated[0, 0, 0]
    return depth_interploated
    


def lift_2d_to_3d(
    points_x:torch.Tensor, points_y:torch.Tensor, points_depth:torch.Tensor, 
    internel_k:torch.Tensor, rotation:Optional[torch.Tensor]=None, translation:Optional[torch.Tensor]=None):
    '''Unproject 2d points to 3d
    With only internel_k, return the 3d points defined in the camera frame
    If rotation and translation are given, the 3d points defined in the object frame will also be returned.
    '''
    assert len(points_x) == len(points_y) == len(points_depth)
    homo_points = torch.stack([points_x, points_y, torch.ones_like(points_x)], dim=-1).float()
    points_camera_frame = homo_points * points_depth[..., None]
    points_camera_frame = torch.mm(torch.inverse(internel_k), points_camera_frame.t()).t()
    if rotation is not None and translation is not None:
        points_object_frame = torch.mm(torch.inverse(rotation), (points_camera_frame - translation[None]).t()).t()
        return points_camera_frame, points_object_frame
    else:
        return points_camera_frame


def cal_3d_2d_corr(depth, internel_k, rotation, translation, occlusion=None):
    '''Calculate 2D-3D correspondance
    Args:
        depth (Tensor): shape (H, W)
        internel_k (Tensor): shape (3, 3)
        rotations (Tensor): shape (3, 3)
        translations (Tensor): shape (3)
        occlusioh (Tensor): shape (H, W)
    return:
        points_2d (Tensor): shape (N, 2), foreground 2d points, xy format
        points_3d (Tensor): shape (N, 3), corresponding 3d location, xyz format     
        
    '''
    mask = depth > 0
    if occlusion is not None:
        mask = mask & occlusion
    points2d_y, points2d_x = torch.nonzero(mask, as_tuple=True)
    points_depth = depth[mask]
    _, points3d_object_frame = lift_2d_to_3d(
        points2d_x.float(), points2d_y.float(), points_depth, internel_k, rotation, translation)
    return torch.stack([points2d_x, points2d_y], dim=-1).float(), points3d_object_frame

def get_flow_from_delta_pose_and_points(rotation_dst, translation_dst, k, points_2d_list, points_3d_list, height, width, invalid_num=400.):
    '''Calculate flow from source image to target image
    Args:
        rotation_dst (Tensor): rotation matrix for target image, shape (n, 3, 3)
        translation_dst (Tensor): translation vector for target image, shape (n, 3)
        k (Tensor): camera intrinsic for source image and taregt image, shape (n, 3, 3)
        points_2d_list (Tensor): source image 2d points, (x,y), each element has shape (N, 2), where N is the number of points
        points_3d_list (Tensor): source image 3d points, (x,y,z), each element has shape (N, 3), where N is the number of points
        height, width (int): patch image resolution 
        invalid_num (float): set invalid flow to this number
    '''
    num_images = len(rotation_dst)
    flow = rotation_dst.new_ones((num_images, 2, height, width)) * invalid_num
    for i in range(num_images):
        points_2d, points_3d = points_2d_list[i], points_3d_list[i]
        points_3d_transpose = points_3d.t()
        points_2d_dst = torch.mm(k[i], torch.mm(rotation_dst[i], points_3d_transpose)+translation_dst[i][:, None]).t()
        points_2d_dst_x, points_2d_dst_y = points_2d_dst[:, 0]/points_2d_dst[:, 2], points_2d_dst[:, 1]/points_2d_dst[:, 2]
        flow_x, flow_y = points_2d_dst_x - points_2d[:, 0], points_2d_dst_y - points_2d[:, 1]
        flow = flow.to(flow_x.dtype)
        flow[i, 0, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)] = flow_x
        flow[i, 1, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)] = flow_y
    return flow



def get_flow_from_delta_pose_and_depth(
    rotation_src, translation_src, rotation_dst, translation_dst, depth_src, k, invalid_num=400):
    '''Calculate flow from source image to target image
    Args:
        rotation_src (Tensor): rotation matrix for source image, shape (n, 3, 3)
        translation_src (Tensor): translatio vector for source image, shape (n, 3)
        rotation_dst (Tenosr): rotation matrix for target image, shape (n, 3, 3)
        translation_dst (Tensor): translation vector for target image, shape (n, 3)
        depth_src (Tensor): depth for source image, shape (n, H, W)
        k (Tensor): camera intrinsic for source image and taregt image. 
    return:
        flow (Tensor): flow from the source image to the target image, shape (n, 2, H, W)
    
    '''
    num_images = rotation_src.shape[0]
    height, width = depth_src.shape[-2:]
    flow = depth_src.new_ones((num_images, 2, height, width), ) * invalid_num
    for i in range(num_images):
        points_2d_src, points_3d_src = cal_3d_2d_corr(depth_src[i], k[i], rotation_src[i], translation_src[i])
        points_3d_src_transpose = points_3d_src.t()
        points_2d_dst = torch.mm(k[i], torch.mm(rotation_dst[i], points_3d_src_transpose)+translation_dst[i][:, None]).t()
        points_2d_dst_x, points_2d_dst_y = points_2d_dst[:, 0], points_2d_dst[:, 1]
        points_2d_dst_x = points_2d_dst_x / points_2d_dst[:, 2]
        points_2d_dst_y = points_2d_dst_y / points_2d_dst[:, 2]
        flow_x = points_2d_dst_x - points_2d_src[:, 0]
        flow_y = points_2d_dst_y - points_2d_src[:, 1]
        flow = flow.to(flow_x.dtype)
        flow[i, 0, points_2d_src[:, 1].to(torch.int64), points_2d_src[:, 0].to(torch.int64)] = flow_x
        flow[i, 1, points_2d_src[:, 1].to(torch.int64), points_2d_src[:, 0].to(torch.int64)] = flow_y
    return flow


def get_pose_from_delta_pose(rotation_delta, translation_delta, rotation_src, translation_src, weight=10., depth_transform='exp', detach_depth_for_xy=False):
    '''Get transformed pose
    Args:
        rotation_delta (Tensor): quaternion to represent delta rotation shape (n, 4)(Quaternions) or (n, 6)(orth 6D )
        translation_delta (Tensor): translation to represent delta translation shape (n, 3)
        rotation_src (Tensor): rotation matrix to represent source rotation shape (n, 3, 3)
        translation_src (Tensor): translation vector to represent source translation shape (n, 3)
    '''
    if rotation_delta.size(1) == 4:
        rotation_delta = kornia.geometry.conversions.quaternion_to_rotation_matrix(rotation_delta)
    else:
        rotation_delta = get_rotation_matrix_from_ortho6d(rotation_delta)
    rotation_dst = torch.bmm(rotation_delta, rotation_src)
    if depth_transform == 'exp':
        vz = torch.div(translation_src[:, 2], torch.exp(translation_delta[:, 2]))
    else:
        # vz = torch.div(translation_src[:, 2], translation_delta[:, 2] + 1)
        vz = translation_src[:, 2] * (translation_delta[:, 2] + 1)
    if detach_depth_for_xy:
        vx = torch.mul(vz.detach(), torch.addcdiv(translation_delta[:, 0] / weight, translation_src[:, 0], translation_src[:, 2]))
        vy = torch.mul(vz.detach(), torch.addcdiv(translation_delta[:, 1] / weight, translation_src[:, 1], translation_src[:, 2]))
    else:
        vx = torch.mul(vz, torch.addcdiv(translation_delta[:, 0] / weight, translation_src[:, 0], translation_src[:, 2]))
        vy = torch.mul(vz, torch.addcdiv(translation_delta[:, 1] / weight, translation_src[:, 1], translation_src[:, 2]))
    translation_dst = torch.stack([vx, vy, vz], dim=-1)
    return rotation_dst, translation_dst



def get_rotation_matrix_from_ortho6d(ortho6d):
    '''
    https://github.com/papagina/RotationContinuity/blob/sanity_test/code/tools.py L47
    '''
    x_raw = ortho6d[:,0:3]#batch*3
    y_raw = ortho6d[:,3:6]#batch*3
        
    x = F.normalize(x_raw, p=2, dim=1) #batch*3
    z = torch.cross(x, y_raw, dim=1)
    z = F.normalize(z, p=2, dim=1)
    y = torch.cross(z, x, dim=1)#batch*3
        
    x = x.view(-1,3,1)
    y = y.view(-1,3,1)
    z = z.view(-1,3,1)
    matrix = torch.cat((x,y,z), 2) #batch*3*3
    return matrix

def save_xyzrgb(points_3d, save_path:str, points_2d=None, image=None):
    if save_path.endswith('xyzrgb'):
        assert image is not None
        assert points_2d is not None
        color = image[:, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)]
        save_content = torch.cat([points_3d, color.transpose(0, 1)], dim=1)
    else:
        save_content = points_3d
    np.savetxt(save_path, save_content.cpu().data.numpy())


def get_2d_3d_corr_by_fw_flow(fw_flow, rendered_depths, ref_rotations, ref_translations, internel_k, valid_mask=None):
    '''
    Return a list of tuple, each element has three components, 
        2d points in the ref image, 2d points in the tgt image, and corresponding 3d points 
    '''
    num_images = len(fw_flow)
    points_corr = []
    for i in range(num_images):
        if valid_mask is not None:
            points_2d, points_3d = cal_3d_2d_corr(
                rendered_depths[i], internel_k[i], ref_rotations[i], ref_translations[i], valid_mask[i])
        else:
            points_2d, points_3d = cal_3d_2d_corr(
                    rendered_depths[i], internel_k[i], ref_rotations[i], ref_translations[i])
        pred_flow = fw_flow[i]
        points_flow = pred_flow[:, points_2d[:, 1].to(torch.int64), points_2d[:, 0].to(torch.int64)].t()
        transformed_2d_points = points_2d + points_flow
        points_corr.append((points_2d, transformed_2d_points, points_3d))
    return points_corr


def solve_pose_by_pnp(points_2d:torch.Tensor, points_3d:torch.Tensor, internel_k:torch.Tensor, **kwargs):
    '''
    Args:
        points_2d (Tensor): xy coordinates of 2d points, shape (N, 2)
        points_3d (Tenosr): xyz coordinates of 3d points, shape (N, 3)
        internel_k (Tensor): camera intrinsic, shape (3, 3)
        kwargs (dict):
    '''
    if points_2d.size(0) < 4:
        return None, None, False
    if kwargs.get('solve_pose_mode', 'ransacpnp') == 'ransacpnp':
        ransacpnp_parameter = kwargs.get('solve_pose_param', {})
        reprojectionError = ransacpnp_parameter.get('reprojectionerror', 3.0)
        iterationscount = ransacpnp_parameter.get('iterationscount', 100)
        retval, rotation_pred, translation_pred, inliers = cv2.solvePnPRansac(
            points_3d.cpu().numpy(), 
            points_2d.cpu().numpy(),
            internel_k.cpu().numpy(),
            None, flags=cv2.SOLVEPNP_EPNP, reprojectionError=reprojectionError, iterationsCount=iterationscount
        )
        rotation_pred = cv2.Rodrigues(rotation_pred)[0].reshape(3, 3)
    elif kwargs.get('solve_pose_mode', 'ransacpnp') == 'progressive-x': 
        import pyprogressivex
        pose_ests, _ = pyprogressivex.find6DPoses(
            x1y1 = points_2d.cpu().data.numpy().astype(np.float64),
            x2y2z2 = points_3d.cpu().data.numpy().astype(np.float64),
            K = internel_k.cpu().numpy().astype(np.float64),
            threshold = 2,  
            neighborhood_ball_radius=20,
            spatial_coherence_weight=0.1,
            maximum_tanimoto_similarity=0.9,
            max_iters=400,
            minimum_point_number=6,
            maximum_model_number=1)
        if pose_ests.shape[0] == 0:
            retval = False
        else:
            retval = True
            rotation_pred = pose_ests[0:3, :3]
            translation_pred = pose_ests[0:3, 3]
    else:
        raise RuntimeError(f"Not supported pnp solver :{kwargs.get('solve_pose_mode')}")
    if retval:
        translation_pred = translation_pred.reshape(-1)
        if np.isnan(rotation_pred.sum()) or np.isnan(translation_pred.sum()):
            retval = False
    return rotation_pred, translation_pred, retval

def remap_points_to_origin_resolution(points_2d:torch.Tensor, transform_matrix:torch.Tensor):
    '''
    Remap 2d points on crop and resized patch to original image.
    '''
    num_points = len(points_2d)
    homo_points_2d = torch.cat([points_2d, points_2d.new_ones(size=(num_points, 1))], dim=-1)
    inverse_transform_matrix = torch.linalg.inv(transform_matrix)
    remapped_points_2d = torch.matmul(inverse_transform_matrix[:2, :], homo_points_2d.transpose(0, 1)).transpose(0, 1)
    return remapped_points_2d




def remap_pose_to_origin_resoluaion(pred_rotations_list, pred_translations_list, internel_k_list, meta_info_list):
    '''
    Remap pose predictions to original image resolution for all the objects in an image.
    As we perform different kinds of camera calibration, the remapped pose should follow the same way.
    '''
    num_images = len(pred_rotations_list)
    assert len(pred_translations_list) == len(internel_k_list) == len(meta_info_list) == num_images
    remapped_pred_rotations_list, remapped_pred_translations_list = [], []
    for j in range(num_images):
        pred_rotations, pred_translations = pred_rotations_list[j], pred_translations_list[j]
        internel_k, meta_info = internel_k_list[j], meta_info_list[j]
        if meta_info['geometry_transform_mode'] == 'adapt_intrinsic':
            remapped_pred_rotations_list.append(pred_rotations)
            remapped_pred_translations_list.append(pred_translations)
        else:
            transform_matrixs = meta_info['transform_matrix']
            inverse_transform_matrixs = np.linalg.inv(transform_matrixs)
            keypoints_3d = meta_info['keypoints_3d']
            pre_obj_num = len(pred_rotations)
            pred_rotations_np = pred_rotations.cpu().data.numpy()
            pred_translations_np = pred_translations.cpu().data.numpy()
            internel_k_np = internel_k.cpu().data.numpy()
            remapped_rotations, remapped_translations = [], []
            if meta_info['geometry_transform_mode'] == 'target_intrinsic':
                # 3*3 but not N*3*3, because the ori_k is for the whole image
                ori_k = meta_info['ori_k'] 
                for i in range(pre_obj_num):
                    remapped_rotation, remapped_translation, diff_pixel = remap_pose(
                        internel_k_np[i], pred_rotations_np[i], pred_translations_np[i], keypoints_3d[i], ori_k, inverse_transform_matrixs[i]
                    )
                    remapped_rotations.append(remapped_rotation)
                    remapped_translations.append(remapped_translation)
            elif meta_info['geometry_transform_mode'] == 'keep_intrinsic':
                for i in range(pre_obj_num):
                    remapped_rotation, remapped_translation, diff_pixel = remap_pose(
                        internel_k_np[i], pred_rotations_np[i], pred_translations_np[i], keypoints_3d[i], internel_k_np[i], inverse_transform_matrixs[i]
                    )
                    remapped_rotations.append(remapped_rotation)
                    remapped_translations.append(remapped_translation)
            else:
                raise RuntimeError
            remapped_rotations = torch.from_numpy(np.stack(remapped_rotations, axis=0)).to(torch.float32).to(pred_translations.device)
            remapped_translations = torch.from_numpy(np.stack(remapped_translations, axis=0)).to(torch.float32).to(pred_rotations.device)
            remapped_pred_translations_list.append(remapped_translations)
            remapped_pred_rotations_list.append(remapped_rotations)
    return remapped_pred_rotations_list, remapped_pred_translations_list

