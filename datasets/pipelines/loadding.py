import mmcv
import numpy as np
from .builder import PIPELINES
from ..mask import BitmapMasks



@PIPELINES.register_module()
class LoadImages:
    def __init__(self, 
                color_type='color',
                to_float32=False,
                file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
    
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        filepath = results['img_path']
        img_bytes = self.file_client.get(filepath=filepath)
        img = mmcv.imfrombytes(img_bytes, flag=self.color_type)
        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results



        


@PIPELINES.register_module()
class LoadMasks:
    def __init__(self,
                binarize=True,
                merge=False,
                file_client_args=dict(backend='disk'),
                eps=1e-5):
        self.binarize = binarize
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.eps = eps
        self.merge = merge
    
    def __call__(self, results):
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        
        mask_paths = results['gt_mask_path']
        height, width, _ = results['img_shape']
        
        masks = []
        for path in mask_paths:
            img_bytes = self.file_client.get(filepath=path)
            mask = mmcv.imfrombytes(img_bytes, flag='unchanged')
            if self.binarize:
                dtype = mask.dtype
                if mask.max() < self.eps:
                    mask[...] = 0
                else:
                    mask = (mask / mask.max()).astype(dtype)
            masks.append(mask)
        
        mask = BitmapMasks(masks, height, width)
        results['gt_masks'] = mask
        return results