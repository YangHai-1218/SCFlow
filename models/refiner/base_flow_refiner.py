
from typing import Optional, Dict, Sequence
import torch
import numpy as np
from .base_refiner import BaseRefiner
from ..utils import (
    solve_pose_by_pnp, get_flow_from_delta_pose_and_depth, 
    cal_epe, filter_flow_by_mask, get_2d_3d_corr_by_fw_flow)




class BaseFlowRefiner(BaseRefiner):
    def __init__(self, 
                encoder: Optional[Dict]=None, 
                decoder: Optional[Dict]=None, 
                seperate_encoder: bool=False, 
                filter_invalid_flow_by_mask: bool=False,
                filter_invalid_flow_by_depth: bool=False,
                renderer: Optional[Dict]=None, 
                render_augmentations: Optional[Sequence[Dict]]=None, 
                train_cfg: dict={}, 
                test_cfg: dict={}, 
                init_cfg: dict={}, 
                max_flow: int=400):
        super().__init__(encoder, decoder, seperate_encoder, renderer, render_augmentations, train_cfg, test_cfg, init_cfg, max_flow)
        self.filter_invalid_flow_by_mask = filter_invalid_flow_by_mask
        self.filter_invalid_flow_by_depth = filter_invalid_flow_by_depth
        self.solve_pose_space = test_cfg.get('solve_pose_space', 'transformed')
        assert self.solve_pose_space in ['origin', 'transformed'] 
    
    def get_flow(self):
        raise NotImplementedError

        
    def format_data_train_sup(self, data_batch):
        data = super().format_data_train_sup(data_batch)
        if self.filter_invalid_flow_by_depth:
            gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
            internel_k, labels = data['internel_k'], data['labels']
            render_outputs = self.renderer(gt_rotations, gt_translations, internel_k, labels)
            gt_rendered_depths = render_outputs['fragments'].zbuf
            gt_rendered_depths = gt_rendered_depths[..., 0]
            data.update(gt_rendered_depths = gt_rendered_depths)
            return data
        else:
            return data
        
    def random_sample_points(self, points_2d, points_3d, sample_points_num):
        assert len(points_2d) == len(points_3d)
        num_points = len(points_2d)
        if sample_points_num > num_points:
            return points_2d, points_3d
        rand_index = torch.randperm(num_points-1, device=points_2d.device)[:sample_points_num]
        return points_2d[rand_index], points_3d[rand_index]

    def topk_sample_points(self, points_2d, points_3d, confidence, sample_points_num):
        assert len(points_2d) == len(points_3d)
        num_points = len(points_2d)
        if sample_points_num > num_points:
            return points_2d, points_3d
        _, index = torch.topk(confidence, k=sample_points_num)
        return points_2d[index], points_3d[index]
    
    def sample_points(self, points_2d, points_3d, sample_cfg, points_confidence=None):
        sample_points_num = sample_cfg.get('num', 1000)
        sample_points_mode = sample_cfg.get('mode', 'random')
        if sample_points_mode == 'random':
            return self.random_sample_points(points_2d, points_3d, sample_points_num)
        else:
            return self.topk_sample_points(points_2d, points_3d, points_confidence, sample_points_num)
        

    def val_step(self, data_batch):
        pred_flow, data = self.forward(data_batch)
        gt_rotations, gt_translations = data['gt_rotations'], data['gt_translations']
        ref_rotations, ref_translations = data['ref_rotations'], data['ref_translations']
        internel_k, rendered_depths, rendered_masks = data['internel_k'], data['rendered_depths'], data['rendered_masks']
        gt_masks = data['gt_masks']
        epe, epe_noc = self.eval_epe(
            pred_flow, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations, rendered_masks, gt_masks, reduction='total_mean')
        log_vars = {'epe':epe.item(), 'epe_noc':epe_noc.item()}
        return dict(
            log_vars=log_vars
        )
    
    def eval_epe(self, batch_flow, rendered_depths, ref_rotations, ref_translations, internel_k, gt_rotations, gt_translations, gt_masks=None, reduction='mean'):
        gt_flow = get_flow_from_delta_pose_and_depth(ref_rotations, ref_translations, gt_rotations, gt_translations, rendered_depths, internel_k, invalid_num=self.max_flow)
        epe = cal_epe(gt_flow, batch_flow, reducion=reduction, max_flow=self.max_flow)
        if gt_masks is not None:
            noc_gt_flow = filter_flow_by_mask(gt_flow, gt_masks, self.max_flow)
            valid_mask = torch.sum(noc_gt_flow**2, dim=1).sqrt() < self.max_flow
            epe_noc = cal_epe(noc_gt_flow, batch_flow, valid_mask, reduction)
        else:
            epe_noc = epe
        return epe, epe_noc
    

    def solve_pose(self, 
                   batch_flow : torch.Tensor, 
                   rendered_depths : torch.Tensor, 
                   ref_rotations : torch.Tensor, 
                   ref_translations : torch.Tensor, 
                   internel_k : torch.Tensor, 
                   labels : torch.Tensor, 
                   per_img_patch_num : torch.Tensor, 
                   occlusion: Optional[torch.Tensor]=None):
        batch_rotations, batch_translations = [], []
        num_images = len(rendered_depths)
        if occlusion is not None:
            occlusion_thresh = self.test_cfg.get('occ_thresh', 0.5)
            valid_mask = occlusion > occlusion_thresh
        else:
            valid_mask = None 

        points_corr = get_2d_3d_corr_by_fw_flow(batch_flow, rendered_depths, ref_rotations, ref_translations, internel_k, valid_mask)
        sample_points_cfg = self.test_cfg.get('sample_points', None) 
        retval_flag = []
        for i in range(num_images):
            ref_points_2d, tgt_points_2d, points_3d = points_corr[i]
            if sample_points_cfg is not None:
                if occlusion is not None:
                    points_confidence = occlusion[i, ref_points_2d[:, 1].to(torch.int64), ref_points_2d[:, 0].to(torch.int64)]
                    tgt_points_2d, points_3d = self.sample_points(tgt_points_2d, points_3d, sample_points_cfg, points_confidence)
                else:
                    tgt_points_2d, points_3d = self.sample_points(tgt_points_2d, points_3d, sample_points_cfg)

            rotation_pred, translation_pred, retval = solve_pose_by_pnp(tgt_points_2d, points_3d, internel_k[i], **self.test_cfg)
            if retval:
                rotation_pred = torch.from_numpy(rotation_pred)[None].to(torch.float32).to(ref_rotations.device)
                translation_pred = torch.from_numpy(translation_pred)[None].to(torch.float32).to(ref_rotations.device)
                retval_flag.append(True)
            else:
                rotation_pred = ref_rotations[i][None]
                translation_pred = ref_translations[i][None]
                retval_flag.append(False)
            batch_rotations.append(rotation_pred)
            batch_translations.append(translation_pred)

        batch_rotations = torch.split(torch.cat(batch_rotations), per_img_patch_num)
        batch_translations = torch.split(torch.cat(batch_translations), per_img_patch_num)
        batch_labels = torch.split(labels, per_img_patch_num)
        batch_scores = torch.split(torch.ones_like(labels, dtype=torch.float32), per_img_patch_num)
        batch_retval_flag = torch.split(torch.tensor(retval_flag, device=labels.device, dtype=torch.bool), per_img_patch_num)
        
        batch_rotations = [p[r] for p,r in zip(batch_rotations, batch_retval_flag)]
        batch_translations = [p[r] for p,r in zip(batch_translations, batch_retval_flag)]
        batch_labels = [l[r] for l,r in zip(batch_labels, batch_retval_flag)]
        batch_scores = [s[r] for s,r in zip(batch_scores, batch_retval_flag)]
        return dict(
            rotations=batch_rotations,
            translations=batch_translations,
            scores=batch_scores,
            labels=batch_labels,
        ) 