import warnings
import cv2
import numpy as np
import torch
from os import path as osp
from glob import glob
import trimesh

def load_mesh(mesh_path, ext='.ply'):
    if osp.isdir(mesh_path):
        mesh_paths = glob(osp.join(mesh_path, '*'+ext))
        mesh_paths = sorted(mesh_paths)
    else:
        mesh_paths = [mesh_path]
    meshs = [trimesh.load(p) for p in mesh_paths]
    return meshs

def project_3d_point(pt3d, K, rotation, translation, transform_matrix=None, return_3d=False):
    '''
    Project 3D points in world coordinate to 2D image coordinate
    Args:
        pt3d (ndarray | torch.tensor): shape (n, 3), 3D points in world coordinate
            n is the number of vertices
        K (ndarray | torch.tensor): shape (N, 3, 3) or (3, 3), intrinsic matrix
            N is the number of images
        rotation (ndarray | torch.tensor): shape (N, 3, 3) or (3, 3), rotation matrix
        translation (ndarray | torch.tensor): shape (N, 3, 1) or (3, 1), translation vector
        transform_matrix (ndarray | torch.tensor): shape (3, 3), if not None, transform the projected points
    return:
        projected_points (ndarray): shape (n, 2)
    '''

    assert pt3d.ndim == 2, "Only support single object projection"
    if rotation.ndim - translation.ndim ==  1:
        translation = translation[..., None]
    else:
        assert rotation.ndim == translation.ndim
    multi_image = rotation.ndim >= 3
    if transform_matrix is not None:
        assert transform_matrix.ndim == rotation.ndim

    
    # shape (N, 3, n) or (3, n)
    pts_3d_camera = np.matmul(rotation, pt3d.transpose()) + translation

    if multi_image:
        # shape (N, 3, n)
        # TODO change to numpy
        K_ = torch.from_numpy(K)
        pts_3d_camera_ = torch.from_numpy(pts_3d_camera)
        pts_2d = torch.bmm(K_, pts_3d_camera_)
        if transform_matrix is not None:
            transform_matrix = torch.from_numpy(transform_matrix)
            pts_2d = torch.bmm(transform_matrix, pts_2d)
        pts_2d = pts_2d.numpy()
        pts_2d = pts_2d.transpose((0, 2, 1))
    else:
        # shape (3, n)
        pts_2d = np.matmul(K, pts_3d_camera)
        if transform_matrix is not None:
            pts_2d = np.matmul(transform_matrix, pts_2d)
        pts_2d = pts_2d.transpose()

    pts_2d[..., 0] = pts_2d[..., 0]/ (pts_2d[..., -1] + 1e-8)
    pts_2d[..., 1] = pts_2d[..., 1]/ (pts_2d[..., -1] + 1e-8)
    pts_2d = pts_2d[..., :-1]

    if return_3d:
        if multi_image:
            return pts_2d, pts_3d_camera.transpose((0, 2, 1))
        else:
            return pts_2d, pts_3d_camera.transpose()
    else:
        return pts_2d





def remap_pose(srcK, srcR, srcT, pt3d, dstK, transform_M):
    '''
    Compute rotations and translation under the new intrinsics parameter 
        and the corresponding transform martix
    It is solving the problem "dstK*(R_new*p+T_new) = trans_M*srcK*(srcR*p+srcT)"
    '''
    dst_2d_pts = project_3d_point(pt3d, srcK, srcR, srcT, transform_matrix=transform_M)

    retval, rot, trans = cv2.solvePnP(
        pt3d.reshape(-1, 1, 3),
        dst_2d_pts.reshape(-1, 1, 2),
        dstK,
        None,
        flags=cv2.SOLVEPNP_EPNP)
    
    if retval:
        newR = cv2.Rodrigues(rot)[0]
        newT = trans.reshape(-1)

        new_projected_2d = project_3d_point(pt3d, dstK, newR, newT)
        diff_in_pix = np.linalg.norm(new_projected_2d - dst_2d_pts, axis=1).mean()
        return newR, newT, diff_in_pix
    else:
        warnings.warn('Error in pose mapping')
        return srcR, srcT, -1

def eval_rot_error(gt_r:np.ndarray, pred_r:np.ndarray):
    error_cos = np.trace(np.matmul(pred_r, np.linalg.inv(gt_r)), axis1=1, axis2=2)
    error_cos = 0.5 * (error_cos - 1.0)
    error_cos = np.clip(error_cos, a_min=-1.0, a_max=1.0)
    error =  np.arccos(error_cos)
    error = 180.0 * error / np.pi
    return error

def eval_tran_error(gt_t:np.ndarray, pred_t:np.ndarray):
    error = np.linalg.norm(gt_t - pred_t, axis=-1)
    gt_depth, pred_depth = gt_t[:, -1], pred_t[:, -1]
    error_depth = np.linalg.norm(gt_depth[:, None] - pred_depth[:, None], axis=-1)
    error_xy = np.linalg.norm(gt_t[:, :2] - pred_t[:, :2], axis=-1)
    return error, error_depth, error_xy
