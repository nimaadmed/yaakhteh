from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import numbers
import types
import time
import collections

class Pad_resize_conditional(object):
    """Pad the given PIL.Image on all sides with the given "pad" value.

    Args:
        padding (int or sequence): Padding on each border. If a sequence of
            length 4, it is used to pad left, top, right and bottom borders respectively.
        fill: Pixel fill value. Default is 0.
    """

    def __init__(self, size, ratio=1.4, fill=(255,255,255), interpolation=Image.BILINEAR):
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size
        self.ratio = ratio
        self.fill = fill
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be padded.

        Returns:
            PIL.Image: Padded image.
        """
        border = 0.1
        b, a = img.size
        # print(a/b)
        if a/b < (self.ratio-border):
            # print('yes')
            x = int(b*self.ratio - a)
            # print(x)
            padding = (0, 0, 0, x)
            img = ImageOps.expand(img, border=padding, fill=self.fill)

        ow = self.size
        oh = int(ow/self.ratio)
        # print(oh,ow)
        # time.sleep(3)
        return img.resize((oh, ow), self.interpolation)
