from mmcv.parallel import DataContainer
from mmcv.utils import build_from_cfg
import numpy as np
from .builder import PIPELINES, build_pipeline
from ..utils import to_tensor
from ..pose import project_3d_point, load_mesh
from ..mask import BitmapMasks



@PIPELINES.register_module()
class ProjectKeypoints:
    '''
    Project 3D points to 2D image plane, add 'keypoints_2d' key and 'keypoints_3d_camera_frame' key
    '''
    def __init__(self, 
                clip_border=False):
        self.clip_border = clip_border

    def __call__(self, results):
        # this is 3d keypoints defined on object space
        keypoints_3d = results['gt_keypoints_3d']
        translations, rotations = results['gt_translations'], results['gt_rotations']
        num_obj = len(translations)
        k = results['k']
        keypoints_2d, keypoints_3d_camera_frame = [], []
        for i in range(num_obj):
            keypoint_3d = keypoints_3d[i]
            rotation, translation = rotations[i], translations[i]
            k_i = k[i]
            keypoint_2d, keypoint_3d_camera_frame = project_3d_point(keypoint_3d, k_i, rotation, translation, return_3d=True)
            keypoints_2d.append(keypoint_2d)
            keypoints_3d_camera_frame.append(keypoint_3d_camera_frame)
        keypoints_2d = np.stack(keypoints_2d, axis=0).astype(np.float32)
        keypoints_3d_camera_frame = np.stack(keypoints_3d_camera_frame, axis=0).astype(np.float32)
        results['gt_keypoints_3d_camera'] = keypoints_3d_camera_frame
        results['gt_keypoints_2d'] = keypoints_2d
        return results

@PIPELINES.register_module()
class ComputeBbox:
    '''
    Compute the bbox for the jittered pose, aka reference pose, add 'ref_bboxes' key
    '''
    def __init__(self, 
                mesh_dir, 
                clip_border=True, 
                filter_invalid=True,
                pose_field=['ref_rotations', 'ref_translations'], 
                bbox_field='ref_bboxes'):
        self.mesh_dir = mesh_dir
        self.meshes = load_mesh(mesh_dir)
        mesh_vertices = [mesh.vertices.view(np.ndarray).astype(np.float32) for mesh in self.meshes]
        self.mesh_vertices = [vertices[np.random.choice(vertices.shape[0], 1000)] for vertices in mesh_vertices]
        self.clip_border = clip_border
        self.filter_invalid = filter_invalid
        self.pose_field = pose_field
        self.bbox_field = bbox_field


    def __call__(self, results):
        labels = results['labels']
        ref_rotations, ref_translations = results[self.pose_field[0]], results[self.pose_field[1]]
        ks = results['k']
        obj_num = len(labels)
        bboxes = []
        for i in range(obj_num):
            ref_rotation, ref_translation = ref_rotations[i], ref_translations[i]
            label, k = labels[i], ks[i]
            points_2d = project_3d_point(self.mesh_vertices[label], k, ref_rotation, ref_translation)
            points_x, points_y = points_2d[:, 0], points_2d[:, 1]
            left, right = points_x.min(), points_x.max()
            top, bottom = points_y.min(), points_y.max()
            bbox = np.array([left, top, right, bottom], dtype=np.float32)
            bboxes.append(bbox)
        if obj_num > 0:
            bboxes = np.stack(bboxes, axis=0)
        else:
            bboxes = np.zeros((0, 4), dtype=np.float32)
        if self.clip_border:
            height, width, _ = results['img'].shape
            left, right = np.clip(bboxes[:, 0], a_min=0, a_max=width), np.clip(bboxes[:, 2], a_min=0, a_max=width)
            top, bottom = np.clip(bbox[:, 1], a_min=0, a_max=height), np.clip(bbox[:, 3], a_min=0, a_max=height)
            bbox = np.stack([left, top, right, bottom], axis=1)
        if self.filter_invalid:
            ori_shape = results['ori_shape']
            if np.sum((bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1]) > ori_shape[0] * ori_shape[1]):
                return None
        results[self.bbox_field] = bboxes
        return results



@PIPELINES.register_module()
class ToTensor:
    def __init__(self, stack_keys=['img']):
        self.stack_keys = stack_keys

    def __call__(self, results):
        # format image
        image_keys = results.get('image_filelds', ['img'])
        for key in image_keys:
            img = results[key]
            if isinstance(img, (list, tuple)):
                # multiple patches
                ndim = img[0].ndim
                assert all(i.ndim == ndim for i in img)
                img = [np.expand_dims(img, -1) if ndim < 3 else i for i in img]
                img = [np.ascontiguousarray(i.transpose(2, 0, 1)) for i in img]
                stacked_image = np.stack(img, axis=0)
                img = stacked_image
            else:
                assert isinstance(img, np.ndarray), f"Expect img to be 'np.ndarray', but got {type(img)}"
                if img.ndim == 4:
                    img = np.ascontiguousarray(img.transpose(0, 3, 1, 2)) # (N, H, W, C) -> (N, C, H, W)
                elif img.ndim == 3:
                    img = np.ascontiguousarray(img.transpose(2, 0, 1)) # (H, W, C) -> (C, H, W)
                else:
                    raise RuntimeError
                
            if key in self.stack_keys:
                results[key] = DataContainer(to_tensor(img), stack=True)
            else:
                results[key] = DataContainer(to_tensor(img), stack=False)
        
        if 'depths' in results:
            depths = results['depths']
            depths = np.stack(depths, axis=0)
            results['depths'] = depths
        
        for key in results.get('annot_fields'):
            if key not in results or 'masks' in key:
                continue
            if key in self.stack_keys:
                results[key] = DataContainer(to_tensor(results[key]), stack=True)
            else:
                results[key] = DataContainer(to_tensor(results[key]), stack=False)
        
        if results.get('mask_fields', False):
            for field in results.get('mask_fields', ['masks']):
                if isinstance(results[field], (list, tuple)):
                    masks = results[field]
                    height, width = results[field][0].height, results[field][0].width
                    masks = BitmapMasks(masks, height, width)
                    results[field] = DataContainer(masks, cpu_only=True)
                else:
                    results[field] = DataContainer(results[field], cpu_only=True)
        return results

@PIPELINES.register_module()
class Collect:
    def __init__(self, 
                keys=('img',), 
                annot_keys=None,
                meta_keys=('img_path', 'ori_shape', 'ori_k', 'k',
                            'img_shape', 'img_norm_cfg')):
        self.keys = keys
        self.meta_keys = meta_keys
        self.annot_keys = annot_keys
        
    
    def __call__(self, results):
        data, annot, img_meta = {}, {}, {}
        if self.annot_keys is None:
            annot_keys = results.get('annot_fields')
        else:
            annot_keys = self.annot_keys
        for key in self.meta_keys:
            img_meta[key] = results[key]
        for key in annot_keys:
            if key not in annot_keys:
                continue
            annot[key] = results[key]
        for key in self.keys:
            data[key] = results[key]
        data['img_metas'] = DataContainer(img_meta, cpu_only=True)
        if len(annot) > 0:
            # when testing without annotation, don't register 'annots' key
            data['annots'] = annot
        return data


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_pipeline(transform)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """

        for t in self.transforms:
            data = t(data)
            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += f'    {t}'
        format_string += '\n)'
        return format_string
