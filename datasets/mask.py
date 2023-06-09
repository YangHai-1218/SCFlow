import cv2
import numpy as np
import torch
import mmcv
import itertools
from mmcv.ops.roi_align import roi_align





class BitmapMasks:
    """This class represents masks in the form of bitmaps.

    Args:
        masks (ndarray): ndarray of masks in shape (N, H, W), where N is
            the number of objects.
        height (int): height of masks
        width (int): width of masks

    Example:
        >>> from mmdet.core.mask.structures import *  # NOQA
        >>> num_masks, H, W = 3, 32, 32
        >>> rng = np.random.RandomState(0)
        >>> masks = (rng.rand(num_masks, H, W) > 0.1).astype(np.int)
        >>> self = BitmapMasks(masks, height=H, width=W)

        >>> # demo crop_and_resize
        >>> num_boxes = 5
        >>> bboxes = np.array([[0, 0, 30, 10.0]] * num_boxes)
        >>> out_shape = (14, 14)
        >>> inds = torch.randint(0, len(self), size=(num_boxes,))
        >>> device = 'cpu'
        >>> interpolation = 'bilinear'
        >>> new = self.crop_and_resize(
        ...     bboxes, out_shape, inds, device, interpolation)
        >>> assert len(new) == num_boxes
        >>> assert new.height, new.width == out_shape
    """

    def __init__(self, masks, height, width):
        self.height = height
        self.width = width
        if len(masks) == 0:
            self.masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            assert isinstance(masks, (list, np.ndarray))
            if isinstance(masks, list):
                if isinstance(masks[0], np.ndarray):
                    assert masks[0].ndim == 2  # (H, W)
                elif isinstance(masks[0], BitmapMasks):
                    assert all([height == mask.height for mask in masks])
                    assert all([width == mask.width for mask in masks])
                    masks = list(itertools.chain(*[np.split(mask.masks, len(mask), axis=0) for mask in masks]))
                else:
                    raise RuntimeError(f"Not supported data type, expect 'ndarray' or 'BitMapMasks', but got {type(masks[0])}")
            else:
                assert masks.ndim == 3  # (N, H, W)

            self.masks = np.stack(masks).reshape(-1, height, width)
            assert self.masks.shape[1] == self.height
            assert self.masks.shape[2] == self.width

    def __getitem__(self, index):
        """Index the BitmapMask.

        Args:
            index (int | ndarray): Indices in the format of integer or ndarray.

        Returns:
            :obj:`BitmapMasks`: Indexed bitmap masks.
        """
        masks = self.masks[index].reshape(-1, self.height, self.width)
        return BitmapMasks(masks, self.height, self.width)

    def __iter__(self):
        return iter(self.masks)

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += f'num_masks={len(self.masks)}, '
        s += f'height={self.height}, '
        s += f'width={self.width})'
        return s

    def __len__(self):
        """Number of masks."""
        return len(self.masks)
    
    def warpaffine(self,  transform_matrix, target_width=None, target_height=None, flags=cv2.INTER_NEAREST, pad_val=0):
        if target_height is None:
            target_height = self.height
        if target_width is None:
            target_width = self.width
        if len(self.masks) == 0:
            transformed_mask = np.empty((0, target_height, target_width), dtype=np.uint8)
        else:
            transformed_mask = np.stack([
                cv2.warpAffine(mask, transform_matrix, (target_width, target_height), flags=flags, borderValue=pad_val)
                for mask in self.masks
            ])
        return BitmapMasks(transformed_mask, target_height, target_width)

    def rescale(self, scale, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.rescale`."""
        if len(self.masks) == 0:
            new_w, new_h = mmcv.rescale_size((self.width, self.height), scale)
            rescaled_masks = np.empty((0, new_h, new_w), dtype=np.uint8)
        else:
            rescaled_masks = np.stack([
                mmcv.imrescale(mask, scale, interpolation=interpolation)
                for mask in self.masks
            ])
        height, width = rescaled_masks.shape[1:]
        return BitmapMasks(rescaled_masks, height, width)

    def resize(self, out_shape, interpolation='nearest'):
        """See :func:`BaseInstanceMasks.resize`."""
        if len(self.masks) == 0:
            resized_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            resized_masks = np.stack([
                mmcv.imresize(
                    mask, out_shape[::-1], interpolation=interpolation)
                for mask in self.masks
            ])
        return BitmapMasks(resized_masks, *out_shape)

    def flip(self, flip_direction='horizontal'):
        """See :func:`BaseInstanceMasks.flip`."""
        assert flip_direction in ('horizontal', 'vertical', 'diagonal')

        if len(self.masks) == 0:
            flipped_masks = self.masks
        else:
            flipped_masks = np.stack([
                mmcv.imflip(mask, direction=flip_direction)
                for mask in self.masks
            ])
        return BitmapMasks(flipped_masks, self.height, self.width)

    def pad(self, out_shape=None, padding=None, pad_val=0):
        """See :func:`BaseInstanceMasks.pad`."""
        assert (out_shape is not None) ^ (padding is not None)
        if len(self.masks) == 0:
            if out_shape is None:
                out_height = int(self.height + padding[1] + padding[-1])
                out_width = int(self.width + padding[0] + padding[2])
                out_shape = (out_height, out_width)
            padded_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            if out_shape is not None:
                padded_masks = np.stack([
                    mmcv.impad(mask, shape=out_shape, pad_val=pad_val)
                    for mask in self.masks
                ])
            else:
                padded_masks = np.stack([
                    mmcv.impad(mask, padding=padding, pad_val=pad_val)
                    for mask in self.masks
                ])
                out_height = int(self.height + padding[1] + padding[3])
                out_width = int(self.width + padding[0] + padding[2])
                out_shape = (out_height, out_width)
        return BitmapMasks(padded_masks, *out_shape)
            

    def crop(self, bbox):
        """See :func:`BaseInstanceMasks.crop`."""
        assert isinstance(bbox, np.ndarray)
        assert bbox.ndim == 1
        h, w = bbox[3] - bbox[1] + 1, bbox[2] - bbox[0] + 1
        
        if len(self.masks) == 0:
            cropped_masks = np.empty((0, h, w), dtype=np.uint8)
        else:
            cropped_masks = []
            for mask in self.masks:
                cropped_masks.append(
                    mmcv.imcrop(mask, bbox, pad_fill=0)
                )
        return BitmapMasks(cropped_masks, h, w)

    def crop_and_resize(self,
                        bboxes,
                        out_shape,
                        inds,
                        device='cpu',
                        interpolation='bilinear',
                        binarize=True):
        """See :func:`BaseInstanceMasks.crop_and_resize`."""
        if len(self.masks) == 0:
            empty_masks = np.empty((0, *out_shape), dtype=np.uint8)
            return BitmapMasks(empty_masks, *out_shape)

        # convert bboxes to tensor
        if isinstance(bboxes, np.ndarray):
            bboxes = torch.from_numpy(bboxes).to(device=device)
        if isinstance(inds, np.ndarray):
            inds = torch.from_numpy(inds).to(device=device)

        num_bbox = bboxes.shape[0]
        fake_inds = torch.arange(
            num_bbox, device=device).to(dtype=bboxes.dtype)[:, None]
        rois = torch.cat([fake_inds, bboxes], dim=1)  # Nx5
        rois = rois.to(device=device)
        if num_bbox > 0:
            gt_masks_th = torch.from_numpy(self.masks).to(device).index_select(
                0, inds).to(dtype=rois.dtype)
            targets = roi_align(gt_masks_th[:, None, :, :], rois, out_shape,
                                1.0, 0, 'avg', True).squeeze(1)
            if binarize:
                resized_masks = (targets >= 0.5).cpu().numpy()
            else:
                resized_masks = targets.cpu().numpy()
        else:
            resized_masks = []
        return BitmapMasks(resized_masks, *out_shape)

    def expand(self, expanded_h, expanded_w, top, left):
        """See :func:`BaseInstanceMasks.expand`."""
        if len(self.masks) == 0:
            expanded_mask = np.empty((0, expanded_h, expanded_w),
                                     dtype=np.uint8)
        else:
            expanded_mask = np.zeros((len(self), expanded_h, expanded_w),
                                     dtype=np.uint8)
            expanded_mask[:, top:top + self.height,
                          left:left + self.width] = self.masks
        return BitmapMasks(expanded_mask, expanded_h, expanded_w)

    def translate(self,
                  out_shape,
                  offset,
                  direction='horizontal',
                  fill_val=0,
                  interpolation='bilinear'):
        """Translate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            offset (int | float): The offset for translate.
            direction (str): The translate direction, either "horizontal"
                or "vertical".
            fill_val (int | float): Border value. Default 0 for masks.
            interpolation (str): Same as :func:`mmcv.imtranslate`.

        Returns:
            BitmapMasks: Translated BitmapMasks.

        Example:
            >>> from mmdet.core.mask.structures import BitmapMasks
            >>> self = BitmapMasks.random(dtype=np.uint8)
            >>> out_shape = (32, 32)
            >>> offset = 4
            >>> direction = 'horizontal'
            >>> fill_val = 0
            >>> interpolation = 'bilinear'
            >>> # Note, There seem to be issues when:
            >>> # * out_shape is different than self's shape
            >>> # * the mask dtype is not supported by cv2.AffineWarp
            >>> new = self.translate(out_shape, offset, direction, fill_val,
            >>>                      interpolation)
            >>> assert len(new) == len(self)
            >>> assert new.height, new.width == out_shape
        """
        if len(self.masks) == 0:
            translated_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            translated_masks = mmcv.imtranslate(
                self.masks.transpose((1, 2, 0)),
                offset,
                direction,
                border_value=fill_val,
                interpolation=interpolation)
            if translated_masks.ndim == 2:
                translated_masks = translated_masks[:, :, None]
            translated_masks = translated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(translated_masks, *out_shape)

    def shear(self,
              out_shape,
              magnitude,
              direction='horizontal',
              border_value=0,
              interpolation='bilinear'):
        """Shear the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            magnitude (int | float): The magnitude used for shear.
            direction (str): The shear direction, either "horizontal"
                or "vertical".
            border_value (int | tuple[int]): Value used in case of a
                constant border.
            interpolation (str): Same as in :func:`mmcv.imshear`.

        Returns:
            BitmapMasks: The sheared masks.
        """
        if len(self.masks) == 0:
            sheared_masks = np.empty((0, *out_shape), dtype=np.uint8)
        else:
            sheared_masks = mmcv.imshear(
                self.masks.transpose((1, 2, 0)),
                magnitude,
                direction,
                border_value=border_value,
                interpolation=interpolation)
            if sheared_masks.ndim == 2:
                sheared_masks = sheared_masks[:, :, None]
            sheared_masks = sheared_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(sheared_masks, *out_shape)

    def rotate(self, out_shape, angle, center=None, scale=1.0, fill_val=0):
        """Rotate the BitmapMasks.

        Args:
            out_shape (tuple[int]): Shape for output mask, format (h, w).
            angle (int | float): Rotation angle in degrees. Positive values
                mean counter-clockwise rotation.
            center (tuple[float], optional): Center point (w, h) of the
                rotation in source image. If not specified, the center of
                the image will be used.
            scale (int | float): Isotropic scale factor.
            fill_val (int | float): Border value. Default 0 for masks.

        Returns:
            BitmapMasks: Rotated BitmapMasks.
        """
        if len(self.masks) == 0:
            rotated_masks = np.empty((0, *out_shape), dtype=self.masks.dtype)
        else:
            rotated_masks = mmcv.imrotate(
                self.masks.transpose((1, 2, 0)),
                angle,
                center=center,
                scale=scale,
                border_value=fill_val)
            if rotated_masks.ndim == 2:
                # case when only one mask, (h, w)
                rotated_masks = rotated_masks[:, :, None]  # (h, w, 1)
            rotated_masks = rotated_masks.transpose(
                (2, 0, 1)).astype(self.masks.dtype)
        return BitmapMasks(rotated_masks, *out_shape)

    @property
    def areas(self):
        """Compute the area with the mask"""
        return self.masks.sum((1, 2))
    
    def copy(self):
        '''Similar to np.ndarray.copy()
        '''
        return BitmapMasks(self.masks.copy(), self.height, self.width)

    def to_ndarray(self):
        """See :func:`BaseInstanceMasks.to_ndarray`."""
        return self.masks

    def to_tensor(self, dtype, device):
        """See :func:`BaseInstanceMasks.to_tensor`."""
        return torch.tensor(self.masks, dtype=dtype, device=device)

    def get_bboxes(self):
        num_masks = len(self)
        boxes = np.zeros((num_masks, 4), dtype=np.float32)
        x_any = self.masks.any(axis=1)
        y_any = self.masks.any(axis=2)
        for idx in range(num_masks):
            x = np.where(x_any[idx, :])[0]
            y = np.where(y_any[idx, :])[0]
            if len(x) > 0 and len(y) > 0:
                # use +1 for x_max and y_max so that the right and bottom
                # boundary of instance masks are fully included by the box
                boxes[idx, :] = np.array([x[0], y[0], x[-1] + 1, y[-1] + 1],
                                         dtype=np.float32)
        return boxes
    
    def get_background_mask(self):
        background_mask = np.ones((self.height, self.width), dtype=np.bool_)
        for mask in self.masks:
            background_mask = (mask != 1) & background_mask
        return background_mask    

    def merge_background_mask(self, background_mask:np.ndarray):
        '''Given a new background mask, merge the background mask into each mask
        Args:
            background_mask (np.ndarray): shape (H, W), 0 or 1, 1 is background while 0 is foreground.
        '''
        assert background_mask.shape[0] == self.height and background_mask.shape[1] == self.width
        if len(self.masks) == 0:
            new_masks = np.empty((0, self.height, self.width), dtype=np.uint8)
        else:
            if not isinstance(background_mask, np.bool_):
                background_mask = background_mask.astype(np.bool_)
            new_masks = []
            for mask in self.masks:
                mask = mask.astype(np.bool_)
                new_mask = np.logical_and(~background_mask, mask).astype(np.uint8)
                new_masks.append(new_mask)
        return BitmapMasks(new_masks, self.height, self.width)    
    
    def cal_iof(self, new_mask):
        '''Calculate the iof of current mask and new mask, where the new mask is the foreground
        '''
        iof_list = []
        for mask in self.masks:
            union  = mask.astype(np.bool_) & new_mask.astype(np.bool_)
            area = new_mask.sum()
            union = union.sum()
            if area == 0:
                iof = 1.0
            else:
                iof = union / area
            iof_list.append(iof)
        return np.array(iof_list) 
