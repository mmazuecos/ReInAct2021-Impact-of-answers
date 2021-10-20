import numpy as np

import torch

import gzip

import json

from PIL import Image, ImageDraw, ImageFont

from tqdm import tqdm

import time

from datetime import datetime

from colorama import Fore, Style, init
init()

def xyxy2xywh(boxes, inplace=False, as_int=False):
    """Convert boxes format: (x1, y1, x2, y2) -> (x, y, w, h)

    Args:
      boxes: input boxes in (x1, y1, x2, y2) format
      inplace: if True, replace input boxes with their converted versions
      as_int: if True, interpret the input as integer coordinates (takes into
              account the +1 offset when computing the box width and height)
    Returns:
      boxes in (x, y, w, h) format
    """
    assert (
        (isinstance(boxes, np.ndarray) or torch.is_tensor(boxes))
        and boxes.ndim == 2
        and boxes.shape[1] == 4
    )
    if not inplace:
        boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[:, 2] = boxes[:, 2] - boxes[:, 0] + int(as_int)
    boxes[:, 3] = boxes[:, 3] - boxes[:, 1] + int(as_int)
    return boxes


def xywh2xyxy(boxes, inplace=False, as_int=False):
    """convert boxes format: (x, y, w, h) -> (x1, y1, x2, y2)

    Args:
      boxes: input boxes in (x, y, w, h) format
      inplace: if True, replace input boxes with their converted versions
      as_int: if True, interpret the input as integer coordinates (takes into
              account the -1 offset when computing the box x2 and y2 coords)
    Returns:
      boxes in (x1, y1, x2, y2) format
    """
    assert (
        (isinstance(boxes, np.ndarray) or torch.is_tensor(boxes))
        and boxes.ndim == 2
        and boxes.shape[1] == 4
    )
    if not inplace:
        boxes = boxes.clone() if torch.is_tensor(boxes) else boxes.copy()
    boxes[:, 2] = boxes[:, 0] + boxes[:, 2] - int(as_int)
    boxes[:, 3] = boxes[:, 1] + boxes[:, 3] - int(as_int)
    return boxes
