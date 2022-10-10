
from symbol import pass_stmt
from turtle import forward
import torch
import cv2
import mmcv
import numpy as np
import random
from einops import rearrange

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class transform:
    def __init__(self):
        pass
    
    def __call__(self,image):
        '''
        image: T x H x W x C
        '''
        nums = image.shape[0]
        img_res = None
        for i in range(nums):
            img = image[i]
            img = resize(img, (-1,256))
            img = randomResizedCrop(img)
            img = resize(img, (224,224), keep_ratio=False)
            img = flip(img)
            img = normalize(img, **img_norm_cfg)
            img = torch.tensor(img).unsqueeze(0)
            if img_res == None:
                img_res = img
            else:
                img_res = torch.cat((img_res, img), dim=0)
        
        img_res = rearrange(img_res, 't h w c -> c t h w')
        return img_res


def resize(img, scale, keep_ratio = True, interpolation='bilinear'):
    max_long_edge = max(scale)
    max_short_edge = min(scale)
    if max_short_edge == -1:
        # assign np.inf to long edge for rescaling short edge later.
        scale = (np.inf, max_long_edge)
    img_h, img_w = img.shape[0], img.shape[1]
    if keep_ratio:
        new_w, new_h = mmcv.rescale_size((img_w, img_h), scale)
    else:
        new_w, new_h = scale
    
    return mmcv.imresize(img, (new_w, new_h), interpolation=interpolation)


def get_crop_bbox(img_shape,
                      area_range,
                      aspect_ratio_range,
                      max_attempts=10):
        """Get a crop bbox given the area range and aspect ratio range.

        Args:
            img_shape (Tuple[int]): Image shape
            area_range (Tuple[float]): The candidate area scales range of
                output cropped images. Default: (0.08, 1.0).
            aspect_ratio_range (Tuple[float]): The candidate aspect
                ratio range of output cropped images. Default: (3 / 4, 4 / 3).
                max_attempts (int): The maximum of attempts. Default: 10.
            max_attempts (int): Max attempts times to generate random candidate
                bounding box. If it doesn't qualified one, the center bounding
                box will be used.
        Returns:
            (list[int]) A random crop bbox within the area range and aspect
            ratio range.
        """
        assert 0 < area_range[0] <= area_range[1] <= 1
        assert 0 < aspect_ratio_range[0] <= aspect_ratio_range[1]

        img_h, img_w = img_shape
        area = img_h * img_w

        min_ar, max_ar = aspect_ratio_range
        aspect_ratios = np.exp(
            np.random.uniform(
                np.log(min_ar), np.log(max_ar), size=max_attempts))
        target_areas = np.random.uniform(*area_range, size=max_attempts) * area
        candidate_crop_w = np.round(np.sqrt(target_areas *
                                            aspect_ratios)).astype(np.int32)
        candidate_crop_h = np.round(np.sqrt(target_areas /
                                            aspect_ratios)).astype(np.int32)

        for i in range(max_attempts):
            crop_w = candidate_crop_w[i]
            crop_h = candidate_crop_h[i]
            if crop_h <= img_h and crop_w <= img_w:
                x_offset = random.randint(0, img_w - crop_w)
                y_offset = random.randint(0, img_h - crop_h)
                return x_offset, y_offset, x_offset + crop_w, y_offset + crop_h

        # Fallback
        crop_size = min(img_h, img_w)
        x_offset = (img_w - crop_size) // 2
        y_offset = (img_h - crop_size) // 2
        return x_offset, y_offset, x_offset + crop_size, y_offset + crop_size



def randomResizedCrop(img, area_range=(0.08, 1.0),
                 aspect_ratio_range=(3 / 4, 4 / 3)):
    img_h, img_w = img.shape[0], img.shape[1]

    left, top, right, bottom = get_crop_bbox(
        (img_h, img_w), area_range, aspect_ratio_range)

    crop_bbox = np.array([left, top, right, bottom])
    x1, y1, x2, y2 = crop_bbox
    return img[y1:y2, x1:x2]

def flip(img, flip_ratio = 0.5, direction = "horizontal"):
    if np.random.rand() < flip_ratio:
        img = mmcv.imflip_(img, direction)
    
    return img

def normalize(img, mean, std, to_bgr=False):
    img = np.float32(img)
    mean = np.array(mean)
    std = np.array(std)
    img = mmcv.imnormalize_(img, mean, std, to_bgr)
    return img
