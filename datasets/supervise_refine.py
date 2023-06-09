import itertools
import mmcv
import numpy as np
from os import path as osp
from typing import Sequence, Optional
from .builder import DATASETS
from .base_dataset import BaseDataset
from terminaltables import AsciiTable






@DATASETS.register_module()
class SuperviseTrainDataset(BaseDataset):
    def __init__(self,
                data_root,
                image_list,
                pipeline,
                gt_annots_root: str,
                keypoints_json: str,
                keypoints_num: int,
                class_names: tuple,
                min_visib_fract=0.,
                min_visib_px_num=0,
                sample_num=1,
                label_mapping: dict = None,
                target_label: list = None,
                meshes_eval: str = None,
                mesh_symmetry: dict = {},
                mesh_diameter: list = []):
        super().__init__(
            data_root=data_root,
            image_list=image_list,
            keypoints_json=keypoints_json,
            pipeline=pipeline,
            class_names=class_names,
            label_mapping=label_mapping,
            keypoints_num=keypoints_num,
            meshes_eval=meshes_eval,
            mesh_symmetry=mesh_symmetry,
            mesh_diameter=mesh_diameter,
            target_label=target_label
        )
        self.min_visib_fract = min_visib_fract
        self.sample_num = sample_num
        self.gt_annots_root = gt_annots_root
        self.pose_json_tmpl = "{:06d}/scene_gt.json"
        self.info_json_tmpl = "{:06d}/scene_gt_info.json"
        self.camera_json_tmpl = osp.join(self.gt_annots_root, "{:06}/scene_camera.json")
        self.mask_path_tmpl = "{:06d}/mask_visib/{:06}_{:06}.png"
        self.gt_seq_pose_annots = self._load_pose_annots()
        self.min_visib_px_num = min_visib_px_num
        self.cal_total_sample_num()
    
    def cal_total_sample_num(self):
        table_data = [['class'] + list(self.class_names) + ['total']]
        sample_num_per_obj = {name:{'total_sample_num':0, 'valid_sample_num':0} for name in self.class_names}
        for sequence in self.gt_seq_pose_annots:
            gt_seq_infos, gt_seq_pose_annots = self.gt_seq_pose_annots[sequence]['gt_info'], self.gt_seq_pose_annots[sequence]['pose']
            for k in gt_seq_infos:
                gt_img_infos = gt_seq_infos[k]
                gt_img_pose_annots = gt_seq_pose_annots[k]
                for i in range(len(gt_img_infos)):
                    obj_info, obj_annot = gt_img_infos[i], gt_img_pose_annots[i]
                    ori_label = obj_annot['obj_id']
                    sample_num_per_obj[self.class_names[ori_label-1]]['total_sample_num'] += 1
                    if self.label_mapping is not None:
                        if ori_label not in self.label_mapping:
                            continue
                        label = self.label_mapping[ori_label]
                    else:
                        label = ori_label
                    if self.target_label is not None:
                        if label not in self.target_label:
                            continue
                    if obj_info['visib_fract'] < self.min_visib_fract:
                        continue
                    sample_num_per_obj[self.class_names[ori_label-1]]['valid_sample_num'] += 1
    
        for k in ['total_sample_num', 'valid_sample_num']:
            table_data.append(
                [k] + [sample_num_per_obj[name][k] for name in sample_num_per_obj] + [sum([sample_num_per_obj[name][k] for name in sample_num_per_obj])]
            )
        self.total_sample_num = AsciiTable(table_data).table


    def _load_pose_annots(self):
        sequences = set([p.split(self.data_root)[1].split('/')[1] for p in self.img_files])
        sequences = sorted(list(sequences))
        gt_seq_pose_annots = dict()
        for sequence in sequences:
            gt_pose_json_path = osp.join(self.gt_annots_root, self.pose_json_tmpl.format(int(sequence)))
            gt_info_json_path = osp.join(self.gt_annots_root, self.info_json_tmpl.format(int(sequence)))
            camera_json_path = self.camera_json_tmpl.format(int(sequence))
            gt_pose_annots = mmcv.load(gt_pose_json_path)
            gt_infos = mmcv.load(gt_info_json_path)
            camera_annots = mmcv.load(camera_json_path)
            gt_seq_pose_annots[sequence] = dict(pose=gt_pose_annots, camera=camera_annots, gt_info=gt_infos)
        return gt_seq_pose_annots
    
    def anneal_visib_thr(self):
        if self.anneal_visib_fract_rate is None:
            self.min_visib_fract = self.min_visib_fract_start
        else:
            self.min_visib_fract = self.min_visib_fract_start - self.anneal_visib_fract_rate * self.step.value


    
    
    def getitem(self, index):
        img_path = self.img_files[index]
        _,  seq_name, _, img_name = img_path.rsplit('/', 3)
        img_id = int(osp.splitext(img_name)[0])
        
        gt_seq_annots = self.gt_seq_pose_annots[seq_name]
        # load ground truth pose annots
        if str(img_id) in gt_seq_annots['pose']:
            gt_pose_annots = gt_seq_annots['pose'][str(img_id)]
        else:
            gt_pose_annots = gt_seq_annots['pose']["{:06}".format(img_id)]
        # load camera intrisic
        if str(img_id) in gt_seq_annots['camera']:
            camera_annots = gt_seq_annots['camera'][str(img_id)]
        else:
            camera_annots = gt_seq_annots['camera']["{:06}".format(img_id)]
        # load ground truth annotation related info, e.g., bbox, bbox_visib
        if str(img_id) in gt_seq_annots['gt_info']:
            gt_infos = gt_seq_annots['gt_info'][str(img_id)]
        else:
            gt_infos = gt_seq_annots['gt_info']["{:06}".format(img_id)]
        
        
        # we assume one obejct appear only once.
        gt_obj_num = len(gt_pose_annots)
        gt_rotations, gt_translations, gt_labels, gt_bboxes = [], [], [], []
        gt_mask_paths = []
        for i in range(gt_obj_num):
            obj_id = gt_pose_annots[i]['obj_id']
            if self.target_label is not None:
                if obj_id not in self.target_label:
                    continue
            if self.label_mapping is not None:
                if obj_id not in self.label_mapping:
                    continue
                obj_id = self.label_mapping[obj_id]
            visib_fract = gt_infos[i]['visib_fract']
            if visib_fract < self.min_visib_fract:
                continue
            visib_px_count = gt_infos[i]['px_count_visib']
            if visib_px_count < self.min_visib_px_num:
                continue
            gt_rotations.append(np.array(gt_pose_annots[i]['cam_R_m2c'], dtype=np.float32).reshape(3, 3))
            gt_translations.append(np.array(gt_pose_annots[i]['cam_t_m2c'], dtype=np.float32).reshape(-1))
            gt_labels.append(obj_id)
            gt_bboxes.append(np.array(gt_infos[i]['bbox_obj'], dtype=np.float32).reshape(-1))
            if 'mask_id' in gt_infos[i]:
                mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(int(seq_name), img_id, gt_infos[i]['mask_id']))
            else:
                mask_path = osp.join(self.data_root, self.mask_path_tmpl.format(int(seq_name), img_id, i))
            gt_mask_paths.append(mask_path)

        if len(gt_labels) == 0:
            return None
        gt_rotations = np.stack(gt_rotations, axis=0)
        gt_translations = np.stack(gt_translations, axis=0)
        gt_labels = np.array(gt_labels, dtype=np.int64) - 1
        gt_keypoints_3d = self.keypoints_3d[gt_labels]
        gt_bboxes = np.stack(gt_bboxes, axis=0)
        # ground truth bboxes are xywh format
        gt_bboxes[..., 2:] = gt_bboxes[..., :2] + gt_bboxes[..., 2:]
        obj_num = len(gt_rotations)

        if self.sample_num == -1:
            sample_num = obj_num
        else:
            sample_num = self.sample_num
        choosen_obj_index = np.random.choice(list(range(obj_num)), sample_num)
        gt_rotations = gt_rotations[choosen_obj_index]
        gt_translations = gt_translations[choosen_obj_index]
        gt_labels = gt_labels[choosen_obj_index]
        gt_bboxes = gt_bboxes[choosen_obj_index]
        gt_keypoints_3d = gt_keypoints_3d[choosen_obj_index]
        gt_mask_paths = np.array(gt_mask_paths)[choosen_obj_index].tolist()
        
        k = np.array(camera_annots['cam_K'], dtype=np.float32).reshape(3,3)
        k = np.repeat(k[None], repeats=sample_num, axis=0)
        results_dict = dict()

        results_dict['pose_fields'] = [('gt_rotations', 'gt_translations', 'gt_keypoints_3d'), ('ref_rotations', 'ref_translations', 'ref_keypoints_3d')]
        results_dict['bbox_fields'] = ['gt_bboxes', 'ref_bboxes']
        results_dict['mask_fields'] = ['gt_masks']
        results_dict['label_fields'] = ['labels']
        
        results_dict['annot_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields'] \
                                        + list(itertools.chain(*results_dict['pose_fields'])) + ['k']
        results_dict['aux_fields'] = results_dict['bbox_fields'] + results_dict['mask_fields'] + results_dict['label_fields']
        results_dict['gt_rotations'] = gt_rotations
        results_dict['gt_translations'] = gt_translations
        results_dict['gt_keypoints_3d'] = gt_keypoints_3d
        results_dict['ref_keypoints_3d'] = gt_keypoints_3d
        results_dict['ori_gt_rotations'] = gt_rotations.copy()
        results_dict['ori_gt_translations'] = gt_translations.copy()
        results_dict['labels'] = gt_labels
        results_dict['gt_bboxes'] = gt_bboxes
        results_dict['gt_mask_path'] = gt_mask_paths
        results_dict['k'] = results_dict['ori_k'] = k
        results_dict['img_path'] = img_path
        results_dict = self.transformer(results_dict)

        return results_dict