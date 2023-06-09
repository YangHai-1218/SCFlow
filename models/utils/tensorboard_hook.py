import torch
import numpy as np
from mmcv.utils import build_from_cfg

from mmcv.runner.hooks.logger import TensorboardLoggerHook, TextLoggerHook
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.dist_utils import master_only


@HOOKS.register_module()
class TensorboardImgLoggerHook(TensorboardLoggerHook):
    def __init__(self,
                log_dir=None,
                interval=10,
                ignore_last=True,
                reset_flag=False,
                by_epoch=True,
                image_format='CHW'):
        super().__init__(
            log_dir=log_dir, 
            interval=interval, 
            ignore_last=ignore_last, 
            reset_flag=reset_flag, 
            by_epoch=by_epoch)
        self.image_format = image_format
    
    @master_only
    def log(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return
        tags = self.get_loggable_tags(runner, allow_text=True)
        for tag, val in tags.items():
            if isinstance(val, str):
                self.writer.add_text(tag, val, self.get_iter(runner))
            else:
                self.writer.add_scalar(tag, val, self.get_iter(runner))
        
        if 'log_imgs' in runner.outputs:
            log_imgs = runner.outputs['log_imgs']
            for tag, val in log_imgs.items():
                if isinstance(val, torch.Tensor):
                    self.writer.add_image(tag, val, self.get_iter(runner), dataformats=self.image_format)
                elif isinstance(val, np.ndarray):
                    if val.dtype == np.uint8:
                        val = val.astype(np.float32) / 255
                    elif val.dtype == np.float32:
                        pass 
                    else:
                        raise RuntimeError(f"Expect np.ndarray to be in np.flot32 or np.uint8 dtype, but got {val.dtype}")
                    self.writer.add_image(tag, val, self.get_iter(runner), dataformats=self.image_format)
                elif isinstance(val, (list, tuple)):
                    if isinstance(val[0], torch.Tensor):
                        val = torch.stack(val, dim=0)
                    elif isinstance(val[0], np.ndarray):
                        val = np.stack(val, axis=0)
                    else:
                        raise RuntimeError(f'Unexpected data type:{type(val[0])}')
                    self.writer.add_images(tag, val, self.get_iter(runner), dataformats='N'+self.image_format)
                else:
                    raise RuntimeError(f'Unexpected data type:{type(val)}')
