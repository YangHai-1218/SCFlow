import matplotlib.pyplot as plt

import numpy as np
import cv2
from PIL import Image
import torch
import mmcv
import json
from collections.abc import Sequence


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')




def dumps_json(data, indent=2, depth=2):
    assert depth > 0
    space = ' '*indent
    s = json.dumps(data, indent=indent)
    lines = s.splitlines()
    N = len(lines)
    # determine which lines to be shortened
    is_over_depth_line = lambda i: i in range(N) and lines[i].startswith(space*(depth+1))
    is_open_bracket_line = lambda i: not is_over_depth_line(i) and is_over_depth_line(i+1)
    is_close_bracket_line = lambda i: not is_over_depth_line(i) and is_over_depth_line(i-1)
    # 
    def shorten_line(line_index):
        if not is_open_bracket_line(line_index):
            return lines[line_index]
        # shorten over-depth lines
        start = line_index
        end = start
        while not is_close_bracket_line(end):
            end += 1
        has_trailing_comma = lines[end][-1] == ','
        _lines = [lines[start][-1], *lines[start+1:end], lines[end].replace(',','')]
        d = json.dumps(json.loads(' '.join(_lines)))
        return lines[line_index][:-1] + d + (',' if has_trailing_comma else '')
    # 
    s = '\n'.join([
        shorten_line(i)
        for i in range(N) if not is_over_depth_line(i) and not is_close_bracket_line(i)
    ])
    #
    return s


def pil_to_cv2(image):
    cv2_image = np.array(image)
    # RGB to BGR
    cv2_image = cv2_image[:, :, ::-1].copy()
    return cv2_image

def cv2_to_pil(image):
    # BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return pil_image
