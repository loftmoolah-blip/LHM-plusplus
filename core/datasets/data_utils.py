# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 16:33:26
# @Function      : Data Stream Processing Functions.
import bisect
from copy import deepcopy

import cv2
import numpy as np
import torch

from core.structures.bbox import Bbox


def src_center_crop_according_to_mask(
    img, mask, aspect_standard, enlarge_ratio, head_bbox=None
):
    """This function crops the input image and mask around the nonzero region of the mask.
    The cropping region's aspect ratio is standardized to match `aspect_standard` and can be enlarged randomly
    between `enlarge_ratio[0]` and `enlarge_ratio[1]`.
    Optionally, if a `head_bbox` (bounding box of the head) is provided, its coordinates will be offset
    to match the new cropped region.
    Returns the cropped image, cropped mask, the left and top offset (relative to original image),
    and the (possibly offset) head_bbox.
    """

    ys, xs = np.where(mask > 0)

    if len(xs) == 0 or len(ys) == 0:
        raise Exception("empty mask")

    x_min = np.min(xs)
    x_max = np.max(xs)
    y_min = np.min(ys)
    y_max = np.max(ys)

    height, width, c = img.shape

    fore_bbox = Bbox([x_min, y_min, x_max, y_max], mode="whwh")

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_cur = (
            np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
            + enlarge_ratio_min
        )

        fore_bbox = fore_bbox.scale(enlarge_ratio_cur, width, height)

    _l, _t, _r, _b = fore_bbox.get_box()
    crop_img = img[_t:_b, _l:_r]
    crop_mask = mask[_t:_b, _l:_r]

    if head_bbox is not None:
        head_bbox.offset(-_l, -_t)

    return crop_img, crop_mask, _l, _t, head_bbox


def collected_imgs_with_target_ratio(img_list, target_ratio, tgt_max_size=1024):
    """
    Resize and pad each image in the list to a target ratio and size for batching.

    Args:
        img_list (List[np.ndarray]): List of input images in HWC format.
        target_ratio (float): The desired aspect ratio (width/height) of the output images.
        tgt_max_size (int, optional): The maximum height of the resized/padded images. Defaults to 1024.

    Returns:
        torch.Tensor: Batched tensor of shape (N, 3, H, W), where images are resized and padded
            to share the same size and aspect ratio, with values in range [0, 255].

    Notes:
        The function finds the maximum height and width among images,
        selects the closest allowed target size, rescales each image down while maintaining aspect ratio,
        then center-pads each image to (height, width) determined by target ratio and max size.
        Images are converted from numpy arrays to torch tensors with CHW format.
    """

    tgt_max_size_list = [840]

    max_width = 0
    max_height = 0
    for img in img_list:
        height, width = img.shape[:2]
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    assert max_height > 0 and max_width > 0, print("do not find images")

    idx = bisect.bisect_left(tgt_max_size_list, max_height)
    idx = max(idx - 1, 0)
    tgt_max_size = tgt_max_size_list[idx]
    tgt_max_width_size = (int(tgt_max_size / target_ratio) // 14) * 14

    height, width = tgt_max_size, tgt_max_width_size

    collect_imgs = []

    for img in img_list:
        draw = np.ones((tgt_max_size, tgt_max_width_size, 3), dtype=img.dtype)

        input_h, input_w, _ = img.shape

        scale = min(height / input_h, width / input_w)

        if scale < 1.0:
            new_size = (int(input_w * scale), int(input_h * scale))
            resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
        else:
            resized_img = img

        y_offset = (height - resized_img.shape[0]) // 2
        x_offset = (width - resized_img.shape[1]) // 2

        if resized_img.shape[2] == 3 and draw.shape[2] == 3:
            draw[
                y_offset : y_offset + resized_img.shape[0],
                x_offset : x_offset + resized_img.shape[1],
            ] = resized_img

        else:
            raise NotImplementedError("color channel not match")

        collect_imgs.append(
            torch.from_numpy(draw).float().permute(2, 0, 1).unsqueeze(0)
        )

    return torch.cat(collect_imgs, dim=0)


class PadRatio:
    """Pad the input images to a target ratio and size for batching."""

    def __init__(self, target_ratio, tgt_max_size_list=[840], val=False, **kwargs):
        """
        PadRatio pads and resizes each image in the input list to fit within a target aspect ratio and maximum size(s).
        The function computes the largest height and width among the input images, determines an appropriate
        target output shape according to the given ratio and a sorted list of allowed maximum sizes, and
        resizes each image (scaling down if necessary, centering with white padding) to a common batch size
        suitable for batching in deep learning pipelines.

        Args:
            target_ratio (float or str): The target aspect ratio (height/width) to pad images to.
                If provided as a string, it will be evaluated to a float.
            tgt_max_size_list (list of int, optional): List of allowed maximum sizes for output images.
                Used to select the appropriate size for the current batch. Default: [840].
            val (bool, optional): If True, disables random scaling (used for validation). Default: False.
            **kwargs: Extra arguments (ignored).

        Usage:
            pad = PadRatio(target_ratio=1.2, tgt_max_size_list=[672, 840, 1008])
            imgs = pad([img1, img2, img3])
        """

        self.target_ratio = (
            eval(target_ratio) if isinstance(target_ratio, str) else target_ratio
        )
        self.tgt_max_size_list = tgt_max_size_list
        self.val = val

    def __call__(self, img_list):
        """
        Pads and resizes a list of images to a common batch size with a target aspect ratio.

        Each image in `img_list` is padded to match the aspect ratio specified by `target_ratio`,
        and resized (if necessary) so that the largest side does not exceed `tgt_max_size_list`.
        The output images are centered with white padding to ensure all are identical in shape,
        which is useful for batching in deep learning. This function is compatible with both training and validation
        workflows, with randomized scaling disabled when `val` is True.

        Args:
            img_list (list of numpy.ndarray): List of input images of shape (H, W, 3), dtype uint8 or float32.

        Returns:
            torch.Tensor: Batched tensor of shape (N, 3, tgt_max_size, tgt_max_width_size), where N is the number of images.
        """

        target_ratio = self.target_ratio
        tgt_max_size_list = self.tgt_max_size_list

        max_width = 0
        max_height = 0
        for img in img_list:
            height, width = img.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        assert max_height > 0 and max_width > 0, print("do not find images")

        idx = bisect.bisect_left(tgt_max_size_list, max_height)
        idx = max(idx - 1, 0)
        tgt_max_size = tgt_max_size_list[idx]
        tgt_max_width_size = (int(tgt_max_size / target_ratio) // 14) * 14

        height, width = tgt_max_size, tgt_max_width_size
        collect_imgs = []

        max_width = 0
        max_height = 0
        for img in img_list:
            height, width = img.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        assert max_height > 0 and max_width > 0, print("do not find images")

        idx = bisect.bisect_left(tgt_max_size_list, max_height)
        idx = max(idx - 1, 0)
        tgt_max_size = tgt_max_size_list[idx]
        tgt_max_width_size = (int(tgt_max_size / target_ratio) // 14) * 14

        height, width = tgt_max_size, tgt_max_width_size

        collect_imgs = []

        for img in img_list:
            draw = np.ones((tgt_max_size, tgt_max_width_size, 3), dtype=img.dtype)

            input_h, input_w, _ = img.shape

            scale = min(height / input_h, width / input_w)

            if scale < 1.0:
                new_size = (int(input_w * scale), int(input_h * scale))
                resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_img = img

            y_offset = (height - resized_img.shape[0]) // 2
            x_offset = (width - resized_img.shape[1]) // 2

            if resized_img.shape[2] == 3 and draw.shape[2] == 3:
                draw[
                    y_offset : y_offset + resized_img.shape[0],
                    x_offset : x_offset + resized_img.shape[1],
                ] = resized_img

            else:
                raise NotImplementedError("color channel not match")

            collect_imgs.append(draw)
        return collect_imgs

    def __repr__(self):
        return (
            f" {self.__class__.__name__}({self.target_ratio}, {self.tgt_max_size_list})"
        )


class PadRatioWithScale(PadRatio):
    """pad ratio follow by a scale image,
    if the image is smaller than the target size,
    it will be scaled to the target size
    """

    def scale_img_keep_aspect(self, img, t_h, t_w):
        """
        Scales an input image while keeping its aspect ratio, so that the result matches the target height (t_h)
        and width (t_w). The scaling factor is chosen such that the scaled image fits within (t_h, t_w) without
        exceeding it, and the aspect ratio is preserved. If not in validation mode, a small random variation is
        applied to the scale.

        Args:
            img (np.ndarray): Input image in HWC format.
            t_h (int): Target height.
            t_w (int): Target width.

        Returns:
            np.ndarray: Scaled image, still preserving its aspect ratio, and potentially with a random scale
            variation if not in validation mode.
        """

        h, w = img.shape[:2]
        scale_h, scale_w = t_h / h, t_w / w
        scale = min(scale_h, scale_w)
        assert scale > 1, f"{scale} must be larger than 1"

        if not self.val:
            scale_min = scale * 0.95
            random_scale = scale_min + (scale - scale_min) * np.random.rand()
        else:
            random_scale = scale
        new_w, new_h = (int(w * random_scale), int(h * random_scale))
        resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return resized_img

    def __call__(self, img_list):
        """
        Pad each image in the input list according to a scaled ratio determined by the target size.
        If the image is smaller than the target, it will be upscaled with preserved aspect ratio.
        Padding is added so that the final image matches the computed target size (multiples of 14).

        Args:
            img_list (List[np.ndarray]): List of images (H, W, 3) to be padded and/or scaled.

        Returns:
            collect_imgs (List[np.ndarray]): List of images (H', W', 3) where each image is scaled (if needed) and
                padded to meet target size constraints, with shapes that are multiples of 14.
        """

        target_ratio = self.target_ratio
        tgt_max_size_list = self.tgt_max_size_list

        max_width = 0
        max_height = 0
        for img in img_list:
            height, width = img.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        assert max_height > 0 and max_width > 0, print("do not find images")

        idx = bisect.bisect_left(tgt_max_size_list, max_height)
        idx = max(idx - 1, 0)
        tgt_max_size = tgt_max_size_list[idx]
        tgt_max_width_size = (int(tgt_max_size / target_ratio) // 14) * 14

        height, width = tgt_max_size, tgt_max_width_size
        collect_imgs = []

        max_width = 0
        max_height = 0
        for img in img_list:
            height, width = img.shape[:2]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        assert max_height > 0 and max_width > 0, print("do not find images")

        idx = bisect.bisect_left(tgt_max_size_list, max_height)
        idx = max(idx - 1, 0)
        tgt_max_size = tgt_max_size_list[idx]
        tgt_max_width_size = (int(tgt_max_size / target_ratio) // 14) * 14

        height, width = tgt_max_size, tgt_max_width_size

        collect_imgs = []

        for img in img_list:
            draw = np.ones((height, width, 3), dtype=img.dtype)

            input_h, input_w, _ = img.shape

            scale = min(height / input_h, width / input_w)
            if scale <= 1.0:
                if not self.val:
                    scale_min = 0.95 * scale
                    random_scale = scale_min + (scale - scale_min) * np.random.rand()
                else:
                    random_scale = scale
                new_size = (int(input_w * random_scale), int(input_h * random_scale))
                resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            else:
                resized_img = self.scale_img_keep_aspect(img, height, width)

            y_offset = (height - resized_img.shape[0]) // 2
            x_offset = (width - resized_img.shape[1]) // 2

            if resized_img.shape[2] == 3 and draw.shape[2] == 3:
                draw[
                    y_offset : y_offset + resized_img.shape[0],
                    x_offset : x_offset + resized_img.shape[1],
                ] = resized_img
            else:
                raise NotImplementedError("color channel not match")

            collect_imgs.append(draw)
            # torch.from_numpy(draw).float().permute(2, 0, 1).unsqueeze(0)
        return collect_imgs


class UpRandomCrop:
    """Randomly crop an image to a specified aspect ratio range."""

    def __init__(self, low, high, val=False, **kwargs):
        self.low = low
        self.high = high
        self.val = val

    def random_body_crop(self, img):
        """
        Randomly crops the top portion of the input image along the vertical axis according to a random ratio
        between `self.low` and `self.high` (both between 0 and 1). The cropped image is then padded at the top
        with the background value (constant 1) to retain the original height.
        If the generated ratio is greater than or equal to 1, the original image is returned unchanged.

        Args:
            img (np.ndarray): The input image as a numpy array of shape (H, W, C).

        Returns:
            np.ndarray: The randomly cropped and padded image with the same shape as the input.
        """

        h, w, _ = img.shape

        # [low~high]
        ratio = np.random.uniform(self.low, self.high)

        if ratio < 1.0:
            ratio_h = int(h * ratio)
            crop_img = img[:ratio_h, :, :]

            pad_h = h - ratio_h

            padded_image = np.pad(
                crop_img,
                ((pad_h, 0), (0, 0), (0, 0)),
                mode="constant",
                constant_values=1,
            )

            return padded_image
        else:
            return img

    def __call__(self, img_list):
        if self.val:
            return img_list
        return [self.random_body_crop(img) for img in img_list]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.low}, {self.high})"


class ToTensor:
    def __init__(self, **kwargs):
        pass

    def __call__(self, img_list):

        img_list = [
            torch.from_numpy(draw).float().permute(2, 0, 1).unsqueeze(0)
            for draw in img_list
        ]
        return torch.cat(img_list, dim=0)


class Flip:
    """
    Flip
    ----
    Data augmentation transformation class for flipping images in a list.

    Args:
        horizontal (bool): If True, flip images horizontally (left-right). If False, flip vertically (up-down).
        val (bool): If True, disables random flipping and simply returns the original image list (used for validation).
        **kwargs: Additional keyword arguments (ignored).

    Methods:
        random_flip(img): Randomly flips the image based on the horizontal flag with a probability of 0.5.
        __call__(img_list): Applies the flip transformation to each image in the list if not in validation mode.
        __repr__(): Returns a string representation of the transformation.
    """

    def __init__(self, horizontal=True, val=False, **kwargs):
        self.horizontal = horizontal
        self.val = val

    def random_flip(self, img):
        if self.horizontal:
            if np.random.rand() > 0.5:
                return np.flip(img, axis=1)
        else:
            if np.random.rand() > 0.5:
                return np.flip(img, axis=0)
        return img

    def __call__(self, img_list):

        if self.val:
            return img_list

        return [self.random_flip(img) for img in img_list]

    def __repr__(self):
        return f"{self.__class__.__name__}(horizontal={self.horizontal})"


class SrcImagePipeline:
    """
    SrcImagePipeline
    ----------------
    A configurable pipeline for processing a list of images through multiple, composable transformation steps.

    This class allows chaining multiple augmentation and preprocessing steps (such as cropping, padding, flipping, and tensor conversion)
    by specifying each step as a dictionary with a 'name' key and its specific arguments.

    Args:
        *args: Each argument is a dictionary specifying a pipeline step:
            - 'name' (str): The name of the transformation ('UpRandomCrop', 'PadRatio', 'PadRatioWithScale', 'ToTensor', or 'Flip')
            - Other keys correspond to arguments for the transformation's constructor.

    Example:
        pipeline = SrcImagePipeline(
            {'name': 'UpRandomCrop', 'crop_size': 256, 'some_arg': val},
            {'name': 'Flip', 'horizontal': True},
            {'name': 'ToTensor'}
        )
        processed_imgs = pipeline(img_list)

    """

    def __init__(self, *args):
        self.valid_pipeline = {
            "UpRandomCrop": UpRandomCrop,
            "PadRatio": PadRatio,
            "PadRatioWithScale": PadRatioWithScale,
            "ToTensor": ToTensor,
            "Flip": Flip,
        }

        self.pipeline = []

        for kwargs in args:
            kwargs = deepcopy(kwargs)
            pipeline_name = kwargs.pop("name")
            if pipeline_name in self.valid_pipeline.keys():
                self.pipeline.append(self.valid_pipeline[pipeline_name](**kwargs))
            else:
                raise ValueError(f"{pipeline_name} is not a valid pipeline")

    def __call__(self, img_list):
        for pipeline in self.pipeline:
            img_list = pipeline(img_list)

        return img_list

    def __repr__(self):
        return "Pipeline: " + " -> ".join([p.__repr__() for p in self.pipeline])
