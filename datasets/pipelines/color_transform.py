import random
import mmcv
import cv2
import glob
import warnings
from os import path as osp
import numpy as np
from typing import Sequence
from .builder import PIPELINES
from ..mask import BitmapMasks


class ColorTransform:
    def __init__(self, patch_level=True, image_keys=['img']):
        self.augment_with_mask = False
        self.image_keys = image_keys
        self.patch_level = patch_level

    def augment(self):
        raise NotImplementedError
    
    def forward_wo_mask_patch(self, results):
        image_fields = self.image_keys
        for key in image_fields:
            patches = results[key]
            augmented_patches = []
            for patch in patches:
                augmented_patch = self.augment(patch)
                augmented_patches.append(augmented_patch)
            results[key] = augmented_patches
        return results
    
    def forward_w_mask_patch(self, results):
        image_fields = self.image_keys
        for key in image_fields:
            patches = results[key]
            masks = results[results.get('augment_masks', 'gt_masks')]
            augmented_patches = []
            for patch, mask in zip(patches, masks):
                augmented_patch = self.augment(patch, mask)
                augmented_patches.append(augmented_patch)
            results[key] = augmented_patches
        return results
    
    def forward_wo_mask(self, results):
        image_fields = self.image_keys
        for key in image_fields:
            img = results[key]
            results[key] = self.augment(img)
        return results 
    
    def forward_w_mask(self, results):
        image_fields = self.image_keys
        for key in image_fields:
            img = results[key]
            mask = results.get('augment_masks', 'gt_masks')
            results[key] = self.augment(img, mask)
        return results


    def __call__(self, results):
        if self.patch_level:
            if self.augment_with_mask:
                return self.forward_w_mask_patch(results)
            else:
                return self.forward_wo_mask_patch(results)
        else:
            if self.augment_with_mask:
                return self.forward_w_mask(results)
            else:
                return self.forward_wo_mask(results)
    
        


@PIPELINES.register_module()
class RandomHSV(ColorTransform):
    def __init__(self, h_ratio, s_ratio, v_ratio, p=1.0, patch_level=True, image_keys=['img']):
        super(RandomHSV, self).__init__(patch_level, image_keys=image_keys)
        self.h_ratio = h_ratio
        self.s_ratio = s_ratio
        self.v_ratio = v_ratio
        self.p = p
    
    def augment(self, img):
        if random.random() > self.p:
            return img
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # hue, sat, val
        h = img_hsv[:, :, 0].astype(np.float32)  # hue
        s = img_hsv[:, :, 1].astype(np.float32)  # saturation
        v = img_hsv[:, :, 2].astype(np.float32)  # value
        a = random.uniform(-1, 1) * self.h_ratio + 1
        b = random.uniform(-1, 1) * self.s_ratio + 1
        c = random.uniform(-1, 1) * self.v_ratio + 1
        h *= a
        s *= b
        v *= c
        img_hsv[:, :, 0] = h if a < 1 else h.clip(None, 179)
        img_hsv[:, :, 1] = s if b < 1 else s.clip(None, 255)
        img_hsv[:, :, 2] = v if c < 1 else v.clip(None, 255)
        return cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

@PIPELINES.register_module()
class RandomNoise(ColorTransform):
    def __init__(self, noise_ratio, p=1.0, patch_level=True, image_keys=['img']):
        super(RandomNoise, self).__init__(patch_level, image_keys=image_keys)
        self.noise_ratio = noise_ratio
        self.p = p

    def augment(self, img):
        if random.random() > self.p:
            return img
        noise_sigma = random.uniform(0, self.noise_ratio)
        gauss = np.random.normal(0, noise_sigma, img.shape) * 255
        img = img + gauss
        img[img > 255] = 255
        img[img < 0] = 0
        return np.uint8(img)

@PIPELINES.register_module()
class RandomSmooth(ColorTransform):
    def __init__(self, max_kernel_size=7, p=1.0, patch_level=True, image_keys=['img']):
        super(RandomSmooth, self).__init__(patch_level, image_keys=image_keys)
        self.max_kernel_size = int(max_kernel_size)
        self.kernel_sizes = [i*2+1 for i in range(self.max_kernel_size//2+1)]
        self.p = p

    def augment(self, img):
        if random.random() > self.p:
            return img
        kernel_size = random.choice(self.kernel_sizes)
        img = cv2.blur(img, (kernel_size, kernel_size))
        return img

@PIPELINES.register_module()
class RandomSharpness(ColorTransform):
    def __init__(self, 
                kernel_sizes=[5, 7, 9, 11], 
                p=1., 
                patch_level=True, 
                image_keys=['img']):
        super().__init__(patch_level, image_keys=image_keys)
        self.kernel_sizes = kernel_sizes
        self.p = p
    
    def augment(self, img):
        if random.random() > self.p:
            return img
        ks = random.choice(self.kernel_sizes)
        img_s = cv2.blur(img, (ks, ks))
        if random.random() < 0.5:
            edge = img / (img_s.astype(np.float32) + 0.01)
        else:
            edge = img - img_s.astype(np.float32)
        edge = cv2.normalize(edge, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        alpha = random.uniform(0.5, 0.95)
        img = img * (1-alpha) + edge * alpha
        img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        return img

@PIPELINES.register_module()
class RandomGray(ColorTransform):
    def __init__(self, p=1., patch_level=True, image_keys=['img']):
        super().__init__(patch_level, image_keys=image_keys)
        self.p = p
    
    def augment(self, img):
        if random.random() > self.p:
            return img 
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img



@PIPELINES.register_module()
class RandomBackground(ColorTransform):
    def __init__(self, 
                background_dir:str, 
                p:float=0.8, 
                file_client_args:dict=dict(backend='disk'), 
                flag:str='color', 
                patch_level:bool=True):
        super().__init__(patch_level)
        self.background_dir = background_dir
        self.background_images = glob.glob(osp.join(background_dir, '*.jpg')) + \
                                glob.glob(osp.join(background_dir, '*.png'))
        if len(self.background_images) == 0:
            raise RuntimeError(f'No background images found in {background_dir}')
        self.p = p
        self.augment_with_mask = True
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.flag = flag

    def augment(self, image, mask):
        if random.random() > self.p:
            return image
        random_bg = self.sample_random_background()
        if random_bg is None:
            return image
        if image.shape[2] == 4:
            image = self.merge_background_by_alpha(image, random_bg)
        else:
            image = self.merge_background_by_mask(image, random_bg, mask)
        return image
    
    def sample_random_background(self):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        bg_img_path = random.choice(self.background_images)
        try:
            img_bytes = self.file_client.get(filepath=bg_img_path)
            bg_img = mmcv.imfrombytes(img_bytes, flag=self.flag, channel_order='bgr')
            if bg_img is None:
                return None
            else:
                return bg_img
        except:
            warnings.warn(f"Error in loading background image:{bg_img_path}")
            return None

    def merge_background_by_alpha(self, foreImg, backImg):
        assert(foreImg.shape[2] == 4)
        forergb = foreImg[:, :, :3]
        alpha = foreImg[:, :, 3] / 255.0
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.repeat(alpha, 3).reshape(foreImg.shape[0], foreImg.shape[1], 3)
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg
    
    def merge_background_by_mask(self, foreImg, backImg, maskImg):
        forergb = foreImg[:, :, :3]
        if forergb.shape != backImg.shape:
            backImg = cv2.resize(backImg, (foreImg.shape[1], foreImg.shape[0]))
        alpha = np.ones((foreImg.shape[0], foreImg.shape[1], 3), np.float32)
        background_mask = maskImg.get_background_mask()
        alpha[background_mask] = 0
        mergedImg = np.uint8(backImg * (1 - alpha) + forergb * alpha)
        # backImg[alpha > 128] = forergb[alpha > 128]
        return mergedImg
    


@PIPELINES.register_module()
class Normalize(ColorTransform):
    def __init__(self, mean, std, to_rgb=True, patch_level=True, image_keys=['img']):
        super(Normalize, self).__init__(patch_level, image_keys)
        if isinstance(mean, Sequence):
            mean = np.array(mean, dtype=np.float32)
        if isinstance(std, Sequence):
            std = np.array(std, dtype=np.float32)
        self.mean = mean
        self.std = std
        self.to_rgb = to_rgb
    
    def augment(self, img):
        img = mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
        return img
    
    def __call__(self, results):
        if len(results['labels']) > 0:
            results = super(Normalize, self).__call__(results)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb
        )
        return results


@PIPELINES.register_module()
class RandomOcclusion:
    def __init__(self, 
                p=0, 
                bbox_field='gt_bboxes',
                mask_field='gt_masks',
                size_range=(0.02, 0.7), 
                ratio_range=(0.5, 2.0),
                min_bbox_size=20):
        self.p = p
        self.size_range = size_range
        self.ratio_range = ratio_range
        self.bbox_field = bbox_field
        self.mask_field = mask_field
        self.min_bbox_size = min_bbox_size
    
    def __call__(self, results):
        images = results['img']
        bboxes = results[self.bbox_field]
        masks = results[self.mask_field]
        x1, y1, x2, y2 = bboxes[..., 0], bboxes[..., 1], bboxes[..., 2], bboxes[..., 3]
        bbox_size = (x2 - x1) * (y2 - y1)
        num_patches = len(bboxes)
        assert num_patches == len(bboxes)
        new_images, new_masks = [], []
        for i in range(len(bboxes)):
            img = images[i]
            mask = masks[i]
            if random.random() > self.p or bbox_size[i] < self.min_bbox_size:
                new_images.append(img)
                new_masks.append(mask)
                continue
            img = images[i]
            mask = masks[i]
            height, width, _ = img.shape
            
            size = random.uniform(*self.size_range) * bbox_size[i]
            ratio = random.uniform(*self.ratio_range)
            ew = int(np.sqrt(size * ratio))
            eh = int(np.sqrt(size / ratio))
            ecx = random.uniform(x1[i], x2[i])
            ecy = random.uniform(y1[i], y2[i])
            esx = int(np.clip((ecx - ew/2 + 0.5), 0, width-1))
            esy = int(np.clip((ecy - eh/2 + 0.5), 0, height-1))
            eex = int(np.clip((ecx + ew/2 + 0.5), 0, width-1))
            eey = int(np.clip((ecy + eh/2 + 0.5), 0, height-1))
            img[esy:eey, esx:eex, :] = np.random.randint(256, size=(eey-esy, eex-esx, 3))
            occlusion_mask = np.zeros((height, width), dtype=np.uint8)
            occlusion_mask[esy:eey, esx:eex] = 1
            new_mask = mask.merge_background_mask(occlusion_mask)
            new_masks.append(BitmapMasks([new_mask], height, width))
            new_images.append(img)
        results['img'] = new_images
        results[self.mask_field] = new_masks
        return results

@PIPELINES.register_module()
class RandomOcclusionV2:
    def __init__(self, 
                augment_mask_field, 
                data_root, 
                image_list, 
                file_client_args, 
                p=1.0, 
                scale_range=(0.5, 1.0), 
                rotate_range=(-45, 45)):
        self.data_root = data_root
        with open(image_list, 'r') as f:
            image_list = f.readlines()
            image_list = [osp.join(self.data_root, image_path.strip()) for image_path in image_list]
        self.image_list = image_list
        self.file_client = mmcv.FileClient(**file_client_args)
        self.augmet_mask_field = augment_mask_field
        self.p = p
        self.scale_range = scale_range
        self.rotate_range = rotate_range
    
    def __call__(self, results):
        if random.random() > self.p:
            return results
        img = results['img']
        mask = results[self.augmet_mask_field]
        h, w,  _ = img.shape
        random_occlusion_path = random.choice(self.image_list)
        img_bytes = self.file_client.get(random_occlusion_path)
        occlusion_bgr = mmcv.imfrombytes(img_bytes, flag='color', channel_order='bgr')

        occlusion_background = (occlusion_bgr[..., 0] == 0) & (occlusion_bgr[..., 1] == 0) & (occlusion_bgr[..., 2]==0)
        occlusion_foreground = (1 - occlusion_background).astype(np.uint8)
        occlusion_foreground = BitmapMasks([occlusion_foreground], height=h, width=w)
        occlusion_bbox = occlusion_foreground.get_bboxes()[0]
        occlusion_left, occlusion_right = occlusion_bbox[0], occlusion_bbox[2]
        occlusion_top, occlusion_bottom = occlusion_bbox[1], occlusion_bbox[3]

        origin_bbox = mask.get_bboxes()[0]
        origin_left, origin_right = origin_bbox[0], origin_bbox[2]
        origin_top, origin_bottom = origin_bbox[1], origin_bbox[3]
        # generate random translation matrix
        occlusion_cx, occlusion_cy = (occlusion_left + occlusion_right)/2, (occlusion_top + occlusion_bottom)/2
        pleft = random.randint(int(occlusion_cx - origin_right), int(occlusion_cx - origin_left))
        ptop = random.randint(int(occlusion_cy - origin_bottom), int(occlusion_cy - origin_top))
        shiftM = np.array(
            [[1., 0., -pleft],
            [0., 1., -ptop],
            [0., 0., 1.]],
            dtype=np.float32)
        # generate random rotate and scale matrix
        origin_area = mask.areas[0]
        occlusion_area = occlusion_foreground.areas[0]
        scale = np.sqrt(origin_area / occlusion_area)
        scale_factor = random.uniform(scale*self.scale_range[0], scale*self.scale_range[1])
        angle = random.uniform(*self.rotate_range)
        rsM = cv2.getRotationMatrix2D((occlusion_cx, occlusion_cy), angle=angle, scale=scale_factor)
        rsM = np.concatenate((rsM, [[0, 0, 1]]), axis=0)
        transform_matrix = np.matmul(shiftM, rsM)
        occlusion_bgr = cv2.warpAffine(
            occlusion_bgr, transform_matrix[:2, :], (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0)
        )
        occlusion_foreground = occlusion_foreground.warpaffine(transform_matrix[:2, :], w, h)
        # merge the origin image and occlusion image by mask
        occlusion_foreground_float = occlusion_foreground.masks[0].astype(np.float32)[..., None]
        augmented_image = (1 - occlusion_foreground_float) * img + occlusion_foreground_float * occlusion_bgr
        augmented_image = augmented_image.astype(np.uint8)
        results['img'] = augmented_image

        mask_fields  = results.get('mask_field', ['gt_masks'])
        for mask_field in mask_fields:
            mask = results.get(mask_field)
            mask = mask.merge_background_mask(occlusion_foreground.masks[0])
            results[mask_field] = mask
        return results

