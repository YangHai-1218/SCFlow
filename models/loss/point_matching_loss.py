from os import path as osp
import glob
import trimesh
import torch
from torch import nn
from torch.nn import functional as F
from pytorch3d.ops import knn_points
from .builder import LOSSES
import numpy as np


EPS = 1e-8

@LOSSES.register_module()
class PointMatchingLoss(nn.Module):
    def __init__(self, 
                symmetry_types,
                mesh_diameter,
                scale_xy=False,
                scale_depth=False,
                scale_depth_factor=1.,
                use_perspective_shape=False,
                mesh_path=None,
                loss_weight=1.0,
                reduction='mean',
                loss_type='l2',
                ):
        super().__init__()
        self.symmetry_types = symmetry_types
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.mesh_diameter = mesh_diameter
        self.use_perspective_shape = use_perspective_shape
        self.scale_depth = scale_depth
        self.scale_xy = scale_xy
        self.scale_depth_factor = scale_depth_factor
        assert loss_type in ['l1', 'l2']
        self.loss_type = int(loss_type[-1])
        if not self.use_perspective_shape:
            self.meshes = self._load_mesh(mesh_path)
        else:
            self.meshes = None
    
    def to(self, device):
        if self.meshes is not None:
            for i, mesh in enumerate(self.meshes):
                self.meshes[i] = mesh.to(device)

        
    
    def _load_mesh(self, mesh_path, ext='.ply'):
        if osp.isdir(mesh_path):
            mesh_paths = glob.glob(osp.join(mesh_path, '*'+ext))
            mesh_paths = sorted(mesh_paths)
        else:
            mesh_paths = [mesh_path]
        meshes = [trimesh.load(p) for p in mesh_paths]
        meshes = [torch.from_numpy(mesh.vertices.view(np.ndarray).astype(np.float32)) for mesh in meshes]
        return meshes

    
    def forward(self, 
                pred_r:torch.Tensor, pred_t:torch.Tensor,
                gt_r:torch.Tensor,gt_t:torch.Tensor,
                labels:torch.Tensor, points_list=None, scale_factors=None):
        if self.use_perspective_shape:
            assert points_list is not None
        if self.scale_xy or self.scale_depth:
            assert scale_factors is not None
        loss = 0.
        batch_size = len(pred_r)
        scaled_pred_t, scaled_gt_t = torch.clone(pred_t), torch.clone(gt_t)
        
        if self.scale_xy:
            scaled_pred_t[..., :2] = pred_t[..., :2] * scale_factors[:, None]
            scaled_gt_t[..., :2] = gt_t[..., :2] * scale_factors[:, None]
        if self.scale_depth:
            scaled_pred_t[..., -1] = pred_t[..., -1] * scale_factors * self.scale_depth_factor
            scaled_gt_t[..., -1] = gt_t[..., -1] * scale_factors * self.scale_depth_factor
        else:
            scaled_pred_t[..., -1] = pred_t[..., -1] * self.scale_depth_factor
            scaled_gt_t[..., -1] = gt_t[..., -1] * self.scale_depth_factor
        for i in range(batch_size):
            if self.use_perspective_shape:
                points = points_list[i]
            else:
                points = self.meshes[labels[i]]
            pred = torch.matmul(pred_r[i], points.transpose(0, 1)) + scaled_pred_t[i][:, None]
            target = torch.matmul(gt_r[i], points.transpose(0, 1)) + scaled_gt_t[i][:, None]
            pred = pred.transpose(0, 1)
            target = target.transpose(0, 1)
            
            if 'cls_'+str(labels[i].item()+1) in self.symmetry_types:
                knn = knn_points(target[None], pred[None], K=1)
                idx = knn.idx[0, :, 0]
                loss_i = torch.mean(torch.linalg.norm(pred[idx] - target, dim=-1, ord=self.loss_type))
            else:
                loss_i = torch.mean(torch.linalg.norm(pred - target, dim=-1, ord=self.loss_type))
            loss_i = loss_i / self.mesh_diameter[labels[i].item()]
            loss = loss + loss_i 
        if self.reduction == 'mean':
            loss = loss / batch_size
        return self.loss_weight * loss


@LOSSES.register_module()
class DisentanglePointMatchingLoss(nn.Module):
    '''
    Disentangled pointmatching loss: https://arxiv.org/abs/1905.12365
    code:https://github.com/THU-DA-6D-Pose-Group/GDR-Net/blob/core/gdrn_modeling/losses/pm_loss.py
    '''
    def __init__(self, 
                symmetry_types,
                mesh_diameter,
                scale_xy=False,
                scale_depth=False,
                scale_depth_factor=1.,
                use_perspective_shape=False,
                disentangle_z=False,
                mesh_path=None,
                loss_weight=1.0,
                reduction='mean',
                loss_type='l2',
                ):
        super().__init__()
        self.symmetry_types = symmetry_types
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.mesh_diameter = mesh_diameter
        self.use_perspective_shape = use_perspective_shape
        self.scale_depth = scale_depth
        self.scale_xy = scale_xy
        self.scale_depth_factor = scale_depth_factor
        self.disentagle_z = disentangle_z
        assert loss_type in ['l1', 'l2']
        self.loss_type = int(loss_type[-1])
        if not self.use_perspective_shape:
            self.meshes = self._load_mesh(mesh_path)
        else:
            self.meshes = None
    
    def to(self, device):
        if self.meshes is not None:
            for i, mesh in enumerate(self.meshes):
                self.meshes[i] = mesh.to(device)

        
    
    def _load_mesh(self, mesh_path, ext='.ply'):
        if osp.isdir(mesh_path):
            mesh_paths = glob.glob(osp.join(mesh_path, '*'+ext))
            mesh_paths = sorted(mesh_paths)
        else:
            mesh_paths = [mesh_path]
        meshes = [trimesh.load(p) for p in mesh_paths]
        meshes = [torch.from_numpy(mesh.vertices.view(np.ndarray).astype(np.float32)) for mesh in meshes]
        return meshes

    
    def forward(self, 
                pred_r:torch.Tensor, pred_t:torch.Tensor,
                gt_r:torch.Tensor,gt_t:torch.Tensor,
                labels:torch.Tensor, points_list=None, scale_factors=None):
        if self.use_perspective_shape:
            assert points_list is not None
        if self.scale_xy or self.scale_depth:
            assert scale_factors is not None
        loss = 0.
        batch_size = len(pred_r)
        scaled_pred_t, scaled_gt_t = torch.clone(pred_t), torch.clone(gt_t)
        
        if self.scale_xy:
            scaled_pred_t[..., :2] = pred_t[..., :2] * scale_factors[:, None]
            scaled_gt_t[..., :2] = gt_t[..., :2] * scale_factors[:, None]
        if self.scale_depth:
            scaled_pred_t[..., -1] = pred_t[..., -1] * scale_factors * self.scale_depth_factor
            scaled_gt_t[..., -1] = gt_t[..., -1] * scale_factors * self.scale_depth_factor
        else:
            scaled_pred_t[..., -1] = pred_t[..., -1] * self.scale_depth_factor
            scaled_gt_t[..., -1] = gt_t[..., -1] * self.scale_depth_factor
        for i in range(batch_size):
            if self.use_perspective_shape:
                points = points_list[i]
            else:
                points = self.meshes[labels[i]]

            points_gt_rot = torch.matmul(gt_r[i], points.transpose(0, 1)).transpose(0, 1)
            points_gt_rt = points_gt_rot + scaled_gt_t[i][None]
            # rotation part, pred rotation, ground truth translation
            points_pred_rot = torch.matmul(pred_r[i], points.transpose(0, 1)).transpose(0, 1) + scaled_gt_t[i][None]
            if 'cls_'+str(labels[i].item()+1) in self.symmetry_types:
                knn = knn_points(points_gt_rt[None], points_pred_rot[None], K=1)
                idx = knn.idx[0, :, 0]
                points_pred_rot = points_pred_rot[idx]
            loss_rotation_i = torch.mean(torch.linalg.norm(points_pred_rot - points_gt_rt, dim=-1, ord=self.loss_type))
            # translation part
            if self.disentagle_z:
                # depth part, ground rotation, ground truth xy, pred depth
                scaled_pred_depth_i_clone = scaled_gt_t[i].clone()
                scaled_pred_depth_i_clone[-1] = scaled_pred_t[i, -1]
                points_pred_z = points_gt_rot + scaled_pred_depth_i_clone[None]
                loss_depth_i = torch.mean(torch.linalg.norm(points_pred_z - points_gt_rt, dim=-1, ord=self.loss_type))
                # xy part
                scaled_pred_xy_i_clone = scaled_pred_t[i].clone()
                scaled_pred_xy_i_clone[-1] = scaled_gt_t[i, -1]
                points_pred_xy = points_gt_rot + scaled_pred_xy_i_clone[None]
                loss_xy_i = torch.mean(torch.linalg.norm(points_pred_xy - points_gt_rt, dim=-1, ord=self.loss_type))
                loss_trans_i = loss_depth_i + loss_xy_i
            else:
                points_pred_trans = points_gt_rot + scaled_pred_t[i][None]
                loss_trans_i = torch.mean(torch.linalg.norm(points_pred_trans - points_gt_rt, dim=-1, ord=self.loss_type))
        
            loss_i = loss_trans_i + loss_rotation_i
            loss_i = loss_i / self.mesh_diameter[labels[i].item()]
            loss = loss + loss_i 
        if self.reduction == 'mean':
            loss = loss / batch_size
        return self.loss_weight * loss
        
        
@LOSSES.register_module()
class RotPointMatchingLoss(nn.Module):
    def __init__(self, 
                symmetry_types,
                mesh_diameter,
                use_perspective_shape=False,
                mesh_path=None,
                loss_weight=1.0,
                loss_type='l2',
                reduction='mean',
                ):
        super().__init__()
        self.symmetry_types = symmetry_types
        self.loss_weight = loss_weight
        self.reduction = reduction
        self.mesh_diameter = mesh_diameter
        self.use_perspective_shape = use_perspective_shape
        assert loss_type in ['l1', 'l2']
        self.loss_type = int(loss_type[-1])
        if not self.use_perspective_shape:
            self.meshes = self._load_mesh(mesh_path)
        else:
            self.meshes = None
    
    def to(self, device):
        if self.meshes is not None:
            for i, mesh in enumerate(self.meshes):
                self.meshes[i] = mesh.to(device)

        
    
    def _load_mesh(self, mesh_path, ext='.ply'):
        if osp.isdir(mesh_path):
            mesh_paths = glob.glob(osp.join(mesh_path, '*'+ext))
            mesh_paths = sorted(mesh_paths)
        else:
            mesh_paths = [mesh_path]
        meshes = [trimesh.load(p) for p in mesh_paths]
        meshes = [torch.from_numpy(mesh.vertices.view(np.ndarray).astype(np.float32)) for mesh in meshes]
        return meshes

    
    def forward(self, 
                pred_r:torch.Tensor, gt_r:torch.Tensor,
                labels:torch.Tensor, points_list=None):
        if self.use_perspective_shape:
            assert points_list is not None
        loss = 0.
        batch_size = len(pred_r)
        for i in range(batch_size):
            if self.use_perspective_shape:
                points = points_list[i]
            else:
                points = self.meshes[labels[i]]
            
            pred = torch.matmul(pred_r[i], points.transpose(0, 1))
            target = torch.matmul(gt_r[i], points.transpose(0, 1))
            pred = pred.transpose(0, 1)
            target = target.transpose(0, 1)
            
            if 'cls_'+str(labels[i].item()+1) in self.symmetry_types:
                knn = knn_points(target[None], pred[None], K=1)
                idx = knn.idx[0, :, 0]
                loss_i = torch.mean(torch.linalg.vector_norm(pred[idx] - target, dim=-1, ord=self.loss_type))
            else:
                loss_i = torch.mean(torch.linalg.vector_norm(pred - target, dim=-1, ord=self.loss_type))
            loss_i = loss_i / self.mesh_diameter[labels[i].item()]
            loss = loss + loss_i 
        if self.reduction == 'mean':
            loss = loss / batch_size
        return self.loss_weight * loss
