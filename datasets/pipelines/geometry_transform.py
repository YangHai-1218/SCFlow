from typing import Optional
import mmcv
import cv2
import random
import numpy as np

from scipy.spatial.transform import Rotation

from .builder import PIPELINES
from ..pose import eval_rot_error, remap_pose, eval_tran_error, load_mesh
from collections.abc import Sequence



def update_transform_matrix(transform_matrix, results):
    if results.get('transform_matrix', None) is None:
        results['transform_matrix'] = transform_matrix
    else:
        results['transform_matrix'] = np.matmul(transform_matrix, results['transform_matrix'])
    return results

@PIPELINES.register_module()
class RemapPose:
    '''
    This class represents the general Geomtery Augmentation.
    In general, A geometry augmentation will affect the pose or camera intrinsic.
    We abstract it into the following three modes:
        1). Keep the pose, while changing the camera intrinsic. 
                Multiply the camera intrisic with the transform matrix.
        2). Keep the camera intrisic, while changing the pose.
                Use PnP to solve the transformed Pose.
        3). Change the intrinsic to the target intrisic, 
                and solve the pose in this intrinsic and transform.
    '''
    def __init__(self, keep_intrinsic=True, dst_k=None):
        if keep_intrinsic:
            assert dst_k is None, "while keep intrinsic, can not specific the taget k"
            self.mode = 'keep_intrinsic'
        else:
            if dst_k is not None:
                assert isinstance(dst_k, Sequence)
                self.dst_k = np.array(dst_k, dtype=np.float32).reshape(3, 3)
                self.mode = 'target_intrinsic'
            else:
                self.mode = 'adapt_intrinsic'
    
    def adapt_intrisic(self, transform_matrix, intrinsic):
        '''
        Args:
            transform_matrix (ndarray): 2d transform matrix, shape (3, 3) or shape (n, 3, 3)
            intrinsic (ndarray): camera intrinsic parameter, shape (3, 3) or shape (n, 3, 3)
        ''' 
        assert transform_matrix.ndim == intrinsic.ndim 
        new_intrisic = np.matmul(transform_matrix, intrinsic)
        return new_intrisic
    
    def transform_target_intrinsic(self, transform_matrix, intrinsic, rotations, translations, keypoints_3d):
        '''
        Args:
            transform_matrix (ndarray): transform matrix, shape  (n, 3, 3)
            intrinsic (ndarray): camera intrinsic parameters, shape (n, 3, 3)
            rotation (ndarray): rotation, shape (n, 3, 3)
            translation (ndarray): translation, shape (n, 3)
            keypoints_3d (ndarray): preset 3D keypoints, shape (n, keypoints_num, 3) 
        NOTE: when transform matrix is (n, 3, 3), this 'n' may be less than the 'n' for translation/rotation shape (n, 3)/(n, 3, 3).

        '''
        
        assert transform_matrix.ndim == 3 and rotations.ndim == 3 and \
                translations.ndim ==2 and keypoints_3d.ndim == 3 and intrinsic.ndim == 3
        num_obj = rotations.shape[0]    
        assert transform_matrix.shape[0] == num_obj and translations.shape[0] == num_obj and  \
            keypoints_3d.shape[0] == num_obj and  intrinsic.shape[0] == num_obj
        new_intrinsic = np.repeat(self.dst_k[None], num_obj, axis=0)

        new_rotation_list, new_translation_list = [], []
        for i in range(num_obj):
            new_rotation, new_translation, diff_in_pixel = remap_pose(
                srcK=intrinsic[i], 
                srcR=rotations[i], 
                srcT=translations[i], 
                pt3d=keypoints_3d[i], 
                dstK=self.dst_k,
                transform_M=transform_matrix[i])
            new_rotation_list.append(new_rotation)
            new_translation_list.append(new_translation)
        new_rotations = np.stack(new_rotation_list, axis=0)
        new_translations = np.stack(new_translation_list, axis=0)
        return new_rotations, new_translations, new_intrinsic

    def transform_pose(self, transform_matrix, intrinsic, rotations, translations, keypoints_3d):
        '''
        Transform the object pose according the 2D transform matrix, while keeping the camera intrinsic.
        '''
        assert transform_matrix.ndim == 3 and rotations.ndim == 3 and \
                translations.ndim == 2 and keypoints_3d.ndim == 3 and intrinsic.ndim == 3
        num_obj = rotations.shape[0]    
        assert transform_matrix.shape[0] == num_obj and translations.shape[0] == num_obj and  \
            keypoints_3d.shape[0] == num_obj and  intrinsic.shape[0] == num_obj
        
        new_rotation_list, new_translation_list = [], []
        for i in range(num_obj):
            new_rotation, new_translation, _ = remap_pose(
                srcK=intrinsic[i], 
                srcR=rotations[i], 
                srcT=translations[i], 
                pt3d=keypoints_3d[i], 
                dstK=intrinsic[i],
                transform_M=transform_matrix[i])
            new_rotation_list.append(new_rotation)
            new_translation_list.append(new_translation)
        new_rotations = np.stack(new_rotation_list, axis=0)
        new_translations = np.stack(new_translation_list, axis=0)
        return new_rotations, new_translations, intrinsic




    def __call__(self, results):
        transform_matrix = results.get('transform_matrix')
        intrinsic = results.get('k')
        if self.mode == 'adapt_intrinsic':
            new_intrinsic = self.adapt_intrisic(transform_matrix, intrinsic)
            results['k'] = new_intrinsic
        else:
            pose_fields = results.get('pose_fields', [('rotations', 'translations', 'keypoints_3d')])
            assert mmcv.is_list_of(pose_fields, tuple)
            if len(pose_fields) == 2:
                obj_num = min([len(results.get(pose_fields[0][0])), len(results.get(pose_fields[1][0]))])
            else:
                obj_num = len(results.get(pose_fields[0][0]))
        
            for pose_field in pose_fields:
                rotations, translations, keypoints_3d = results.get(pose_field[0]), results.get(pose_field[1]), results.get(pose_field[2])
                if self.mode == 'keep_intrinsic':
                    new_rotations, new_translations, new_intrinsic = self.transform_pose(
                        transform_matrix, intrinsic, rotations[:obj_num], translations[:obj_num], keypoints_3d[:obj_num]
                    )
                else:   
                    new_rotations, new_translations, new_intrinsic = self.transform_target_intrinsic(
                        transform_matrix, intrinsic, rotations[:obj_num], translations[:obj_num], keypoints_3d[:obj_num])
                    
                rotations[:obj_num] = new_rotations
                translations[:obj_num] = new_translations
                results[pose_field[0]] = rotations
                results[pose_field[1]] = translations
            
            results['k'] = new_intrinsic
        results['geometry_transform_mode'] = self.mode
        return results



@PIPELINES.register_module()
class Crop:
    '''
    Randomly select one object, and crop its corresponding patch.
    Args:
        select_num (str): select one object or all objects in an image to be cropped.
            'all' mode is usually used in testing case, which will cause the number of cropped patches for each image uncertrain.
            In this case, we should not stack the image patches directly.
    return:
        results (dict): 
            'img': (H, W, 3) ndarray --> list[ndarray], cropped images.
            'k': (3, 3) ndarray --> (n, 3, 3) ndarray, camera intrinsic for each cropped image.

    '''
    def __init__(self, 
                size_range=(0.8, 1.2), 
                crop_bbox_field='ref_bboxes',
                keep_ratio=False,
                aspect_ratio=1., 
                pad_val=128,
                min_expand=0,
                clip_border=True):
        self.size_range = size_range
        self.aspect_ratio = aspect_ratio
        self.crop_bbox_field = crop_bbox_field
        self.keep_ratio = keep_ratio
        self.pad_val = pad_val
        self.clip_border = clip_border
        self.min_expand = min_expand

    def _crop_bbox(self, crop_bboxes, results):
        x1, y1 = crop_bboxes[:, 0], crop_bboxes[:, 1]
        bbox_offset = np.stack([x1, y1, x1, y1], axis=1)
        obj_num = crop_bboxes.shape[0]
        for field in results.get('bbox_fields', ['bboxes']):
            bboxes = results[field]
            cropped_bboxes = bboxes[:obj_num] - bbox_offset
            bboxes[:obj_num] = cropped_bboxes
            results[field] = bboxes
        return results

    
    def _crop_mask(self, crop_bboxes, results):
        if not results.get('mask_fields', False):
            return results
        obj_num = crop_bboxes.shape[0]
        for field in results.get('mask_fields', ['masks']):
            masks = results[field]
            cropped_masks = []
            for i in range(len(masks)):
                mask = masks[i]
                if i >= obj_num:
                    cropped_masks.append(mask)
                else:
                    cropped_masks.append(mask.crop(crop_bboxes[i]))
            results[field] = cropped_masks
        return results
    
    def _crop_depth(self, crop_bboxes, results):
        if not results.get('depth_fields', False):
            return results
        for field in results.get('depth_fields', ['depths']):
            depths = results[field]
            cropped_depths = mmcv.imcrop(depths, crop_bboxes, pad_fill=0)
            results[field] = cropped_depths
        return results          

    def __call__(self, results):
        if len(results['labels']) == 0:
            results['k'] = np.zeros((0, 3, 3), dtype=np.float32)
            return results
        assert self.crop_bbox_field in results
        bboxes = results.get(self.crop_bbox_field)
        h, w, _ = results['img_shape']
        
        obj_num = len(bboxes)
        orig_image = results['img']
        transform_matrixs, cropped_images, cropped_bboxes = [], [], []
        for idx in range(obj_num):
            size_ratio = random.uniform(*self.size_range)
            bbox = bboxes[idx]
            x1, y1, x2, y2 = bbox
            bbox_w, bbox_h = x2 - x1, y2 - y1
            xc, yc = (x1 + x2)/2, (y1 + y2)/2
            if not self.keep_ratio:
                bbox_w = max(bbox_w, bbox_h * self.aspect_ratio)
                bbox_h = max(bbox_w/self.aspect_ratio, bbox_h)
            old_bbox_w, old_bbox_h = bbox_w, bbox_h
            new_bbox_w, new_bbox_h = bbox_w*size_ratio, bbox_h*size_ratio
            if self.min_expand > 0:
                bbox_w, bbox_h = max(old_bbox_w + 2*self.min_expand, new_bbox_w), max(old_bbox_h + 2*self.min_expand, new_bbox_h)
            else:
                bbox_w, bbox_h = new_bbox_w, new_bbox_h

            
            if self.clip_border:
                crop_x1, crop_x2 = int(np.clip(xc - bbox_w/2, 0, w)), int(np.clip(xc + bbox_w/2, 0, w))
                crop_y1, crop_y2 = int(np.clip(yc - bbox_h/2, 0, h)), int(np.clip(y2 + bbox_h/2, 0, h))
            else:
                crop_x1, crop_x2 = int(xc - bbox_w/2), int(xc + bbox_w/2)
                crop_y1, crop_y2 = int(yc - bbox_h/2), int(yc + bbox_h/2)
            
            transform_matrixs.append(
                np.array(
                    [[1., 0, -crop_x1],
                     [0, 1., -crop_y1],
                     [0, 0, 1.]], dtype=np.float32)
            )
            cropped_bboxes.append(np.array(
                [crop_x1, crop_y1, crop_x2, crop_y2]))
        cropped_bboxes = np.stack(cropped_bboxes, axis=0)
        transform_matrix = np.stack(transform_matrixs, axis=0)
        cropped_images = mmcv.imcrop(orig_image, cropped_bboxes, scale=1, pad_fill=self.pad_val)
        image_shapes = [image.shape for image in cropped_images]

        results['img'] = cropped_images
        results['img_shape'] = image_shapes

        results = update_transform_matrix(transform_matrix, results)
        results = self._crop_bbox(cropped_bboxes, results)
        results = self._crop_mask(cropped_bboxes, results)
        results = self._crop_depth(cropped_bboxes, results)
        return results



@PIPELINES.register_module()
class Resize:
    '''Resize the image & mask.

    Args:
        img_scale (tuple): (h, w) or (s)
        keep_ratio (bool): whether keeping ratio. 
            If keep ratio and the image scale is (w, h), choose max(h, w) as the target scale
    '''
    def __init__(self, img_scale, keep_ratio=True):
        
        if isinstance(img_scale, (list, tuple)):
            if len(img_scale) == 1:
                self.img_scale = (img_scale[0], img_scale[1])
            else:
                self.img_scale = tuple(img_scale)
        else:
            assert isinstance(img_scale, int)
            self.img_scale = (img_scale, img_scale)
        self.keep_ratio = keep_ratio
    
    def _resize_img(self, results):
        images = results['img']
        num_patches = len(images)
        resized_patches, image_shapes = [], []
        transform_matrixs, scale_factors = [], []

        for i in range(num_patches):
            patch = images[i]
            if self.keep_ratio:
                resized_patch, scale_factor = mmcv.imrescale(patch, self.img_scale, return_scale=True)
                transform_matrixs.append(
                    np.array(
                        [[scale_factor, 0, 0],
                        [0, scale_factor, 0],
                        [0, 0, 1.]], dtype=np.float32
                    )
                )
                w_scale, h_scale = scale_factor, scale_factor
            else:
                resized_patch, w_scale, h_scale = mmcv.imresize(patch, self.img_scale, return_scale=True)
                transform_matrixs.append(
                    np.array(
                        [[w_scale, 0, 0],
                        [0, h_scale, 0],
                        [0, 0, 1]], dtype=np.float32
                    )
                )
            image_shapes.append(resized_patch.shape)
            resized_patches.append(resized_patch)
            scale_factors.append(np.array([w_scale, h_scale, w_scale, h_scale], dtype=np.float32))
        scale_factors = np.stack(scale_factors, axis=0)
        transform_matrix = np.stack(transform_matrixs, axis=0)
        results['img'] = resized_patches
        results['scale_factor'] = scale_factors
        results['img_shape'] = image_shapes
        
        return results, transform_matrix


    def _resize_bboxes(self, results):
        scale_factor = results['scale_factor']
        obj_num = len(scale_factor)
        for field in results.get('bbox_fields', ['bboxes']):
            bboxes = results[field]
            new_bboxes = bboxes[:obj_num] * scale_factor
            bboxes[:obj_num] = new_bboxes
            results[field] = bboxes
        return results

    def _resize_masks(self, results):
        if not results.get('mask_fields', False):
            return results
        for field in results.get('mask_fields', ['masks']):
            masks = results[field]
            resized_masks = []
            for mask in masks:
                if self.keep_ratio:
                    rescaled_mask = mask.rescale(self.img_scale)
                    resized_masks.append(rescaled_mask)
                else:
                    resized_mask = mask.resize(self.img_scale)
                    resized_masks.append(resized_mask)
            results[field] = resized_masks
        return results
    
    def _resize_depths(self, results):
        if not results.get('depth_fields', False):
            return results
        for field in results.get('depth_fields', ['depths']):
            depths = results[field]
            num_patches = len(depths)
            resized_depths = []
            for i in range(num_patches):
                depth = depths[i]
                if self.keep_ratio:
                    resized_depth = mmcv.imrescale(depth, self.img_scale, return_scale=False)
                else:
                    resized_depth = mmcv.imresize(depth, self.img_scale, return_scale=False)
                resized_depths.append(resized_depth)
            results[field] = resized_depths
        return results

                
    def __call__(self, results):
        if len(results['labels']) == 0:
            return results
        results, transform_matrix = self._resize_img(results)
        results = update_transform_matrix(transform_matrix, results)
        results = self._resize_bboxes(results)
        results = self._resize_masks(results)
        results = self._resize_depths(results)
        return results


@PIPELINES.register_module()
class Pad:
    '''
    Pad the image & masks
    There are two padding modes: 
        (1) pad to a fixed size(size)
        (2) pad to the minimum size that is divisible by some number(size_divisor)
    
    Args:
        size (tupe, optional): Fixxed padding shape, (H, W)
        size_divisor (int, optional): The divisor of padded size
        pad_val(dict, optional): A dict for padding value.
    '''
    def __init__(self,
                size=None, 
                center=False,
                pad_val=dict(img=0, mask=0)):
        assert isinstance(size, (list, tuple))
        assert len(size) == 2

        self.size = tuple(size)
        self.pad_val = pad_val
        self.center = center
        
    
    def _pad_img(self, results):
        pad_val = self.pad_val.get('img', 0)
        images = results['img']
        padded_images, padding_list = [], []
        transform_matrixs = []
        for img in images:
            if self.center:
                image_h, image_w, _ = img.shape
                padding_top = int(self.size[0]/2 - image_h/2)
                padding_left = int(self.size[1]/2 - image_w/2)
                padding_bottom = self.size[0] - image_h - padding_top
                padding_right = self.size[1] - image_w - padding_left
                padding = (padding_left, padding_top, padding_right, padding_bottom)
                padded_img = mmcv.impad(
                    img, padding=padding, pad_val=pad_val
                )
                transform_matrix = np.array(
                    [[1., 0., padding_left],
                    [0., 1., padding_top],
                    [0., 0., 1.]], dtype=np.float32
                )
            else:
                padding = (0, 0, self.size[1] - img.shape[1], self.size[0] - img.shape[0])
                padded_img = mmcv.impad(
                    img, padding=padding, pad_val=pad_val)
                transform_matrix = np.eye(3, dtype=np.float32)
            transform_matrixs.append(transform_matrix)
            padded_images.append(padded_img)
            padding_list.append(padding)
        results['img'] = padded_images
        results['padding'] = padding_list
        results['img_shape'] = [i.shape for i in padded_images]
        transform_matrixs = np.stack(transform_matrixs, axis=0)
        return results, transform_matrixs
    
    def _pad_mask(self, results):
        if not results.get('mask_fields', False):
            return results
        pad_val = self.pad_val.get('mask', 0)
        padding_list = results['padding']
        for field in results.get('mask_fields', ['masks']):
            masks = results[field]
            obj_num = len(padding_list)
            padded_masks = []
            for i, mask in enumerate(masks):
                if i >= obj_num:
                    padded_masks.append(mask)
                else:
                    padding = list(padding_list[i])
                    mask_height, mask_width = mask.height, mask.width
                    padding[2] = self.size[1] - mask_width - padding[0]
                    padding[3] = self.size[0] - mask_height - padding[1]
                    padding = tuple(padding)
                    padded_masks.append(mask.pad(padding=padding, pad_val=pad_val)) 
            results[field] = padded_masks
        return results
    
    def _pad_depth(self, results):
        if not results.get('depth_fields', False):
            return results 
        pad_val = self.pad_val.get('depth', 0)
        padding_list = results['padding']
        for field in results.get('depth_fields', ['depths']):
            depths = results[field]
            obj_num = len(padding_list)
            padded_depths = []
            for i in range(obj_num):
                padded_depth = mmcv.impad(depths[i], padding=padding_list[i], pad_val=pad_val)
                padded_depths.append(padded_depth)
            results[field] = padded_depths 
        return results

    def __call__(self, results):
        if len(results['labels']) == 0:
            results['img'] = np.zeros((0, self.size[0], self.size[1], 3), dtype=np.uint8)
            return results
        results, transform_matrixs = self._pad_img(results)
        results = self._pad_mask(results)
        results = self._pad_depth(results)
        results = update_transform_matrix(transform_matrixs, results)
        return results

