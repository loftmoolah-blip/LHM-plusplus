# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu & Peihao Li
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-07 23:56:19
# @Function      : video_human_dataset, used tar to I/O data.
# @Describ:  Due to the fragmented nature of the input video files, we developed this function to read the files using tar.

import json
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

sys.path.append("./")

from core.datasets.video_base_human_dataset import (
    VideoBaseHumanDataset,
    smplx_params_to_tensor,
)
from core.datasets.video_tar import TarFrameDataset
from core.utils.bbox import Bbox
from core.utils.proxy import no_proxy

__all__ = ["VideoHumanDataset"]


class VideoHumanDataset(VideoBaseHumanDataset):
    """Video Human Dataset for loading human video data from tar archives.

    This dataset handles video data with SMPLX parameters, supporting multi-view
    rendering and source image processing. It loads data from tar archives for
    efficient I/O with fragmented video files.

    Attributes:
        sample_side_views (int): Number of side views to sample per item.
        render_image_res (int): Resolution for rendered images.
        render_region_size (Tuple[int, int]): Size of the rendering region [H, W].
        source_image_res (int): Resolution for source images.
        aspect_standard (float): Standard aspect ratio (height/width).
        enlarge_ratio (List[float]): Min and max ratios for random enlargement.
        is_val (bool): Whether this is a validation dataset.
        use_flame (bool): Whether to use FLAME parameters for face modeling.
        src_head_size (int): Size of source head images.
    """

    def __init__(
        self,
        root_dirs: str,
        meta_path: str,
        sample_side_views: int,
        render_image_res_low: int,
        render_image_res_high: int,
        render_region_size: Union[int, List[int], Tuple[int, int]],
        source_image_res: int,
        repeat_num: int = 1,
        crop_range_ratio_hw: Optional[List[float]] = None,
        aspect_standard: float = 5.0 / 3,
        enlarge_ratio: Optional[List[float]] = None,
        debug: bool = False,
        is_val: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the Video Human Dataset.

        Args:
            root_dirs (str): Root directory path containing tar archive files.
            meta_path (str): Path to the metadata JSON file with frame information.
            sample_side_views (int): Number of side views to sample per data item.
            render_image_res_low (int): Low resolution for rendered images (unused).
            render_image_res_high (int): High resolution for rendered images.
            render_region_size (Union[int, List[int], Tuple[int, int]]): Size of the
                rendering region. Can be a single int or [H, W] tuple.
            source_image_res (int): Resolution for source reference images.
            repeat_num (int, optional): Number of times to repeat the dataset for
                epoch extension. Defaults to 1.
            crop_range_ratio_hw (Optional[List[float]], optional): Height and width
                crop ratios for data augmentation. Defaults to [1.0, 1.0].
            aspect_standard (float, optional): Standard aspect ratio (height/width).
                Defaults to 5.0/3.
            enlarge_ratio (Optional[List[float]], optional): [min, max] range for
                random enlargement augmentation. Defaults to [0.9, 1.1].
            debug (bool, optional): Enable debug mode for verbose output. Defaults to False.
            is_val (bool, optional): Whether this is a validation dataset (deterministic
                sampling). Defaults to False.
            **kwargs: Additional keyword arguments:
                - multiply (int): Multiplier for size calculations. Default: 14.
                - use_flame (bool): Use FLAME parameters for face. Default: False.
                - src_head_size (int): Source head image size. Default: 448.
                - ref_img_size (int): Reference image size. Default: 8.
                - womask (bool): Reference image without mask. Default: False.

        Note:
            Currently, render_image_res_low and render_image_res_high must be equal.
        """
        if crop_range_ratio_hw is None:
            crop_range_ratio_hw = [1.0, 1.0]
        if enlarge_ratio is None:
            enlarge_ratio = [0.9, 1.1]

        super().__init__(root_dirs, meta_path)

        # View and resolution settings
        self.sample_side_views = sample_side_views
        self.render_image_res_low = render_image_res_low
        self.render_image_res_high = render_image_res_high

        # Convert render_region_size to tuple if needed
        if not isinstance(render_region_size, (list, tuple)):
            render_region_size = (render_region_size, render_region_size)  # [H, W]
        self.render_region_size = render_region_size
        self.source_image_res = source_image_res

        # Data augmentation settings
        self.uids = self.uids * repeat_num
        self.crop_range_ratio_hw = crop_range_ratio_hw
        self.aspect_standard = aspect_standard
        self.enlarge_ratio = enlarge_ratio

        # Mode settings
        self.debug = debug
        self.is_val = is_val

        # Validate and set render resolution
        assert self.render_image_res_low == self.render_image_res_high, (
            f"Low and high resolutions must match: "
            f"{self.render_image_res_low} != {self.render_image_res_high}"
        )
        self.render_image_res = self.render_image_res_low

        # Additional configuration from kwargs
        self.multiply = kwargs.get("multiply", 14)
        self.use_flame = kwargs.get("use_flame", False)
        self.src_head_size = kwargs.get("src_head_size", 448)
        self.ref_img_size = kwargs.get("ref_img_size", 8)
        self.womask = kwargs.get("womask", False)

        print(
            f"VideoHumanDataset initialized: "
            f"data_len={len(self.uids)}, repeat_num={repeat_num}, "
            f"debug={debug}, is_val={is_val}"
        )

    def load_rgb_image_with_aug_bg(
        self,
        rgb,
        mask,
        bg_color,
        pad_ratio,
        max_tgt_size,
        aspect_standard,
        enlarge_ratio,
        render_tgt_size,
        multiply,
        intr,
        head_bbox=None,
    ):
        """
        Load an RGB image and its mask, apply data augmentations including background replacement,
        padding, resizing, and cropping according to mask and head bounding box. Returns tensors
        suitable for model input and optionally provides head-specific crops and masks.

        Args:
            rgb (np.ndarray): Input RGB image of shape (H, W, 3) or (H, W, 4) if alpha present.
            mask (np.ndarray or None): Input mask image of shape (H, W), or None if alpha is used.
            bg_color (float): Background color value (in range [0, 1]) used to replace background pixels.
            pad_ratio (float): Ratio to pad the image (relative to original size).
            max_tgt_size (int): Maximum size (pixels) to which shortest image edge is resized before further steps.
            aspect_standard (float): The aspect ratio (H/W) to enforce during center cropping.
            enlarge_ratio (list of float or tuple): [min, max] ratios to randomly enlarge the cropping region.
            render_tgt_size (int): Target size (scalar, e.g. 512) for the SMPL-X estimator's image preprocessor.
            multiply (int): Multiplication factor for aligning to specific model/grid requirements.
            intr (np.ndarray or torch.Tensor): Camera intrinsics matrix (to be updated accordingly).
            head_bbox (Bbox or None, optional): Bounding box around the head region in 'whwh' mode; will be updated by padding/cropping.

        Returns:
            tuple: (
                rgb (torch.Tensor): [1, 3, H, W] float32, background filled,
                mask (torch.Tensor): [1, 1, H, W] float32, binary mask after all augmentations,
                padding_mask (torch.Tensor): [1, 1, H, W] float32, 1 where not padded, 0 for the padded regions,
                intr (torch.Tensor): Camera intrinsic matrix (4x4),
                head_rgb (torch.Tensor): [1, 3, h, w] float32, cropped head region if available else zeros,
                head_only_mask (torch.Tensor): [1, 1, H, W] float32, mask with only head region,
                use_head (bool): Whether head crop is valid/available.
            )

        Note:
            - Will center-pad and later resize and optionally crop the image before conversion to tensor.
            - The output 'use_head' is True if a valid bounding box was provided, False otherwise.
            - Assumes input images are numpy arrays, returns torch tensors.
        """

        def mask_padding(img_np):
            ori_h, ori_w = img_np.shape[:2]

            h = round((1 + pad_ratio) * ori_h)
            w = round((1 + pad_ratio) * ori_w)

            mask_pad_np = np.ones((h, w), dtype=np.uint8)

            offset_h, offset_w = (h - img_np.shape[0]) // 2, (w - img_np.shape[1]) // 2
            mask_pad_np[
                offset_h : offset_h + img_np.shape[0] :,
                offset_w : offset_w + img_np.shape[1],
            ] = 0.0

            return mask_pad_np / 1.0, offset_h, offset_w

        if pad_ratio > 0:
            rgb, head_bbox = self.img_center_padding(rgb, pad_ratio, head_bbox)

        rgb = rgb / 255.0
        if mask is not None:
            if pad_ratio > 0:
                mask, _ = self.img_center_padding(mask, pad_ratio, None)
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]  # [H, W]

        padding_mask, _, _ = mask_padding(mask)

        mask = (mask > 0.5).astype(np.float32)
        padding_mask = (padding_mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # resize to specific size require by preprocessor of smplx-estimator.

        rgb, head_bbox = self.resize_image_keepaspect_np(rgb, max_tgt_size, head_bbox)
        mask, _ = self.resize_image_keepaspect_np(mask, max_tgt_size, None)
        padding_mask, _ = self.resize_image_keepaspect_np(
            padding_mask, max_tgt_size, None
        )

        padding_mask_h, padding_mask_w = np.where(padding_mask == 0)
        padding_edge = [
            padding_mask_h.min(),
            padding_mask_h.max(),
            padding_mask_w.min(),
            padding_mask_w.max(),
        ]

        # crop image to enlarge human area.
        rgb, mask, padding_mask, offset_x, offset_y, head_bbox = (
            self.center_crop_according_to_mask(
                rgb,
                mask,
                padding_mask,
                aspect_standard,
                enlarge_ratio,
                padding_edge=padding_edge,
                head_bbox=head_bbox,
            )
        )

        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

        # resize to render tgt_size for training for face
        if head_bbox is not None:
            src_head_rgb = self.crop_head(rgb, head_bbox)
            src_head_rgb = self.img_padding_square(src_head_rgb, bg_color)

            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.src_head_size, self.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
            use_head = True
        else:
            src_head_rgb = np.zeros(
                (self.src_head_size, self.src_head_size, 3)
            )  # null image
            use_head = False

        # resize to render_tgt_size for training for human body
        tgt_hw_size, ratio_y, ratio_x, head_bbox = self.calc_new_tgt_size_by_aspect(
            cur_hw=rgb.shape[:2],
            aspect_standard=aspect_standard,
            tgt_size=render_tgt_size,
            multiply=multiply,
            head_bbox=head_bbox,
        )

        rgb = cv2.resize(
            rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )
        mask = cv2.resize(
            mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )
        padding_mask = cv2.resize(
            padding_mask,
            dsize=(tgt_hw_size[1], tgt_hw_size[0]),
            interpolation=cv2.INTER_AREA,
        )

        # head_only_mask
        if head_bbox is not None:
            head_only_mask = self.head_only_mask(mask, head_bbox)
        else:
            head_only_mask = np.zeros_like(mask)

        intr = self.scale_intrs(intr, ratio_x=ratio_x, ratio_y=ratio_y)

        assert (
            abs(intr[0, 2] * 2 - rgb.shape[1]) < 7.0
        ), f"{intr[0, 2] * 2}, {rgb.shape[1]}"
        assert (
            abs(intr[1, 2] * 2 - rgb.shape[0]) < 7.0
        ), f"{intr[1, 2] * 2}, {rgb.shape[0]}"
        intr[0, 2] = rgb.shape[1] // 2
        intr[1, 2] = rgb.shape[0] // 2

        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)
        mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)
        padding_mask = (
            torch.from_numpy(padding_mask[:, :, None])
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )
        head_only_mask = (
            torch.from_numpy(head_only_mask[:, :, None])
            .float()
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        src_head_rgb = (
            torch.from_numpy(src_head_rgb).float().permute(2, 0, 1).unsqueeze(0)
        )

        return rgb, mask, padding_mask, intr, src_head_rgb, head_only_mask, use_head

    def load_src_rgb_image_with_aug_bg(
        self,
        rgb,
        mask,
        bg_color,
        pad_ratio,
        max_tgt_size,
        aspect_standard,
        enlarge_ratio,
        render_tgt_size,
        multiply,
        intr,
        head_bbox=None,
    ):
        """
        Load a source RGB image and its mask, apply data augmentations including background replacement,
        padding, resizing, and optional random cropping of the body for source image processing in SMPL-X pipeline.

        Args:
            rgb (np.ndarray): Input RGB(A) image of shape (H, W, 3) or (H, W, 4) if alpha present.
            mask (np.ndarray or None): Input mask image of shape (H, W), or None if alpha is used.
            bg_color (float): Background color value (in range [0, 1]) to fill the background for masked-out regions.
            pad_ratio (float): Padding ratio to apply before resizing (relative to original size).
            max_tgt_size (int): Maximum target size (pixels) for the shortest dimension during resizing.
            aspect_standard (float): Target aspect ratio (H/W) to apply during cropping.
            enlarge_ratio (list or tuple): [min, max] range to randomly enlarge the cropping region.
            render_tgt_size (int): Target size for the final rendered image (typically scalar, e.g. 512).
            multiply (int): Multiplication factor for output size alignment.
            intr (np.ndarray or torch.Tensor): Camera intrinsic matrix (4x4) to update when scaling or cropping.
            head_bbox (Bbox, optional): Bounding box for head region to update as image is transformed.

        Returns:
            tuple:
                - rgb (torch.Tensor): [1, 3, H, W] float32, image with background filled.
                - mask (torch.Tensor): [1, 1, H, W] float32, binary mask after augmentations.
                - padding_mask (torch.Tensor): [1, 1, H, W] float32, 1 for valid pixels, 0 for padding.
                - intr (torch.Tensor): Camera intrinsic matrix after augmentations (4x4).
                - src_head_rgb (torch.Tensor): [1, 3, h, w] float32, cropped head image if available, else zeros.
                - head_only_mask (torch.Tensor): [1, 1, H, W] float32, mask containing only the head region.
                - use_head (bool): Whether a valid head crop/region was found and used.

        Note:
            - This method is similar to load_rgb_image_with_aug_bg, intended specifically for source image pipelines.
            - Image and mask are converted to torch tensors. Head crops and masks are produced if a corresponding bounding box is provided.
            - The intrinsic matrix is scaled and its principal point updated as needed after transformations.
        """

        def random_body_crop(img):
            h, w, _ = img.shape

            # [0.5~1.05]
            # 50% random crop, 50% no crop
            ratio = np.random.uniform(0.5, 1.05)
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

        if pad_ratio > 0:
            rgb, head_bbox = self.img_center_padding(rgb, pad_ratio, head_bbox)

        rgb = rgb / 255.0
        if mask is not None:
            if pad_ratio > 0:
                mask, _ = self.img_center_padding(mask, pad_ratio, None)
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]  # [H, W]

        mask = (mask > 0.5).astype(np.float32)
        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # resize to specific size require by preprocessor of smplx-estimator.

        rgb, head_bbox = self.resize_image_keepaspect_np(rgb, max_tgt_size, head_bbox)
        mask, _ = self.resize_image_keepaspect_np(mask, max_tgt_size, None)

        # crop image to enlarge human area.
        rgb, mask, _, offset_x, offset_y, head_bbox = (
            self.center_crop_according_to_mask(
                rgb,
                mask,
                None,
                aspect_standard,
                enlarge_ratio,
                padding_edge=None,
                head_bbox=head_bbox,
            )
        )

        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

        # resize to render tgt_size for training for face
        if head_bbox is not None:
            src_head_rgb = self.crop_head(rgb, head_bbox)
            src_head_rgb = self.img_padding_square(src_head_rgb, bg_color)

            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.src_head_size, self.src_head_size),
                interpolation=cv2.INTER_AREA,
            )  # resize to dino size
            use_head = True
        else:
            src_head_rgb = np.zeros(
                (self.src_head_size, self.src_head_size, 3)
            )  # null image
            use_head = False

        # resize to render_tgt_size for training for human body
        tgt_hw_size, ratio_y, ratio_x, head_bbox = self.calc_new_tgt_size_by_aspect(
            cur_hw=rgb.shape[:2],
            aspect_standard=aspect_standard,
            tgt_size=render_tgt_size,
            multiply=multiply,
            head_bbox=head_bbox,
        )

        rgb = random_body_crop(rgb)

        rgb = cv2.resize(
            rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA
        )

        rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)

        src_head_rgb = (
            torch.from_numpy(src_head_rgb).float().permute(2, 0, 1).unsqueeze(0)
        )

        return rgb, src_head_rgb, use_head

    def _process_frame_data(self, tar_item, frame_id, shape_param=None):
        """
        Process a single frame's data and extract relevant information for subsequent processing.

        Args:
            tar_item (TarFrameDataset): The dataset object containing frame information.
            frame_id (str or int): The identifier for the frame to process.

        Returns:
            tuple:
                - img (np.ndarray): The RGB image of the frame.
                - mask (np.ndarray): The segmentation mask corresponding to the image.
                - smplx_params (dict): SMPL-X parameters (in tensor format).
                - flame_params (dict or None): FLAME parameters, if available.
                - bg_color (float): Background color selected for augmentation.
                - max_tgt_size (int): The maximal target size for resizing.
                - c2w (torch.Tensor): Camera-to-world transformation matrix.
                - intrinsic (torch.Tensor): Camera intrinsic matrix.
                - head_bbox (Bbox or None): Head bounding box if FLAME params exist, else None.
        """

        query_values = tar_item[frame_id]
        img = query_values["imgs_png"]
        mask = query_values["samurai_seg"]
        smplx_params_raw = query_values["smplx_params"]
        flame_params = query_values["flame_params"]

        smplx_params = smplx_params_to_tensor(smplx_params_raw)

        c2w, intrinsic = self._load_pose(smplx_params)

        if flame_params is not None:
            head_bbox = flame_params["frame_bbox"]
            head_bbox = list(map(int, head_bbox))
            head_bbox = Bbox(head_bbox, mode="whwh")
        else:
            head_bbox = None

        if self.use_flame and flame_params is not None:
            smplx_params["expr"] = torch.FloatTensor(flame_params["expcode"])

            # replace with flame's jaw_pose
            smplx_params["jaw_pose"] = torch.FloatTensor(flame_params["posecode"][3:])
            smplx_params["leye_pose"] = torch.FloatTensor(flame_params["eyecode"][:3])
            smplx_params["reye_pose"] = torch.FloatTensor(flame_params["eyecode"][3:])
        else:
            smplx_params["expr"] = torch.FloatTensor([0.0] * 100)

        bg_color = random.choice([0.0, 0.5, 1.0])
        if self.is_val:
            bg_color = 1.0

        pad_ratio = float(smplx_params_raw["pad_ratio"])
        max_tgt_size = int(max(smplx_params["img_size_wh"]))

        return (
            img,
            mask,
            smplx_params,
            bg_color,
            max_tgt_size,
            c2w,
            intrinsic,
            head_bbox,
            pad_ratio,
        )

    def get_tgt_view_img(self, tar_item, uid, all_frames, frame_id_list):
        """
        Get target view images and related information for a set of frames.

        Args:
            tar_item (dict): Dictionary containing data for each frame.
            uid: Unique identifier for the sample.
            all_frames (list): List of all frames available.
            frame_id_list (list): List of frame IDs to process as target views.

        Returns:
            tuple: A tuple containing processed data for each frame in frame_id_list.
                - c2ws: List of camera-to-world transformation matrices (torch.Tensor).
                - intrs: List of camera intrinsic matrices (torch.Tensor).
                - rgbs: List of processed RGB images (as tensors).
                - bg_colors: List of background color values used.
                - masks: List of processed binary masks.
                - padding_masks: List of padding masks.
                - head_rgbs: List of cropped head images (if any).
                - head_only_masks: List of head-only region masks.
                - smplx_params_list: List of SMPL-X parameter dictionaries (as tensors).
                - use_heads: List indicating whether a valid head crop/region was found and used (bool).
                - shape_param: The shared shape parameter (betas) across all frames.
        """

        # source images
        (
            c2ws,
            intrs,
            rgbs,
            bg_colors,
            masks,
            padding_masks,
            head_rgbs,
            head_only_masks,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        smplx_params_list = []
        use_heads = []

        shape_param = None

        for frame_id in frame_id_list:
            (
                img,
                mask,
                smplx_params,
                bg_color,
                max_tgt_size,
                c2w,
                intrinsic,
                head_bbox,
                pad_ratio,
            ) = self._process_frame_data(tar_item, frame_id)

            if shape_param is None:
                shape_param = smplx_params["betas"]

            rgb, mask, padding_mask, intrinsic, head_rgb, head_only_mask, use_head = (
                self.load_rgb_image_with_aug_bg(
                    img,
                    mask=mask,
                    bg_color=bg_color,
                    pad_ratio=pad_ratio,
                    max_tgt_size=max_tgt_size,
                    aspect_standard=self.aspect_standard,
                    enlarge_ratio=self.enlarge_ratio if not self.is_val else [1.0, 1.0],
                    render_tgt_size=self.render_image_res,
                    multiply=16,
                    intr=intrinsic,
                    head_bbox=head_bbox,
                )
            )
            c2ws.append(c2w)
            rgbs.append(rgb)
            padding_masks.append(padding_mask)
            bg_colors.append(bg_color)
            intrs.append(intrinsic)
            smplx_params_list.append(smplx_params)
            masks.append(mask)

            # head accumulation.
            head_only_masks.append(head_only_mask)
            head_rgbs.append(head_rgb)
            use_heads.append(use_head)

        c2ws = torch.stack(c2ws, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs, dim=0)  # [N, 4, 4]
        rgbs = torch.cat(rgbs, dim=0)  # [N, 3, H, W]
        bg_colors = (
            torch.tensor(bg_colors, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        )  # [N, 3]
        masks = torch.cat(masks, dim=0)  # [N, 1, H, W]
        padding_masks = torch.cat(padding_masks, dim=0)  # [N, 1, H, W]

        head_only_masks = torch.cat(head_only_masks, dim=0)  # [N, 1, H, W]
        head_rgbs = torch.cat(head_rgbs, dim=0)  # [N, 3, H, W]
        use_heads = torch.tensor(use_heads).unsqueeze(-1)

        smplx_params_tmp = defaultdict(list)
        for smplx in smplx_params_list:
            for k, v in smplx.items():
                smplx_params_tmp[k].append(v)
        for k, v in smplx_params_tmp.items():
            smplx_params_tmp[k] = torch.stack(v)
        smplx_params = smplx_params_tmp
        # TODO check different betas for same person
        smplx_params["betas"] = shape_param

        return (
            c2ws,
            intrs,
            rgbs,
            bg_colors,
            masks,
            padding_masks,
            head_rgbs,
            head_only_masks,
            use_heads,
            smplx_params,
        )

    def get_src_view_img(self, tar_item, uid, all_frames, frame_id_source_list):
        """
        Get source view images and related information for a set of source frames.

        Args:
            tar_item (dict): Dictionary containing data for each frame.
            uid: Unique identifier for the sample.
            all_frames (list): List of all frames available.
            frame_id_source_list (list): List of frame IDs to process as source views.

        Returns:
            tuple:
                - source_rgbs (list): List of source RGB images after augmentation (as tensors).
                - source_head_rgbs (list): List of cropped head region images from source (as tensors).
                - src_use_heads (list): List of booleans indicating valid head regions.
        """

        source_rgbs, source_head_rgbs = [], []
        src_use_heads = []
        # load src img
        for frame_id in frame_id_source_list:
            img, mask, _, _, max_tgt_size, _, intrinsic, head_bbox, _ = (
                self._process_frame_data(tar_item, frame_id)
            )

            # white bg
            bg_color = 1.0

            rgb, src_head_rgb, src_use_head = self.load_src_rgb_image_with_aug_bg(
                img,
                mask=mask,
                bg_color=bg_color,
                pad_ratio=np.random.choice([0.0, 0.1, 0.2], size=1)[0],
                max_tgt_size=max_tgt_size,
                aspect_standard=self.aspect_standard,
                enlarge_ratio=self.enlarge_ratio if not self.is_val else [1.0, 1.0],
                render_tgt_size=self.source_image_res,
                multiply=self.multiply,
                intr=intrinsic,
                head_bbox=head_bbox,
            )

            source_rgbs.append(rgb)
            source_head_rgbs.append(src_head_rgb)
            src_use_heads.append(src_use_head)

        source_rgbs = torch.cat(source_rgbs, dim=0)
        source_head_rgbs = torch.cat(source_head_rgbs, dim=0)
        src_use_heads = torch.tensor(src_use_heads).unsqueeze(-1)

        return source_rgbs, source_head_rgbs, src_use_heads

    @no_proxy
    def inner_get_item(self, idx: int) -> Dict[str, Any]:
        """Get a single data item by index.

        Args:
            idx: Index of the item to retrieve.

        Returns:
            Dictionary containing:
                - uid: Unique identifier string
                - source_rgbs: Source RGB images [N1, 3, H, W]
                - render_image: Rendered RGB images [N, 3, H, W]
                - render_mask: Segmentation masks [N, 1, H, W]
                - render_padding_mask: Padding masks [N, 1, H, W]
                - source_head_rgbs: Source head region images [N1, 3, H, W]
                - render_head_mask: Head region masks [N, 1, H, W]
                - use_heads: Head usage flags [N, 1]
                - c2ws: Camera-to-world matrices [N, 4, 4]
                - intrs: Camera intrinsics [N, 4, 4]
                - render_full_resolutions: Image resolutions [N, 2]
                - render_bg_colors: Background colors [N, 3]
                - Additional SMPLX parameters
        """

        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        all_frames = uid_info["img_path"]

        valid_frame_items = tar_item.valid_frame_items
        assert len(valid_frame_items) > 5

        if not self.is_val:
            frame_id_list = self.train_sample_views(valid_frame_items)
        else:
            frame_id_list = self.val_sample_views(valid_frame_items)

        assert self.sample_side_views + 1 == len(frame_id_list)

        np.random.shuffle(frame_id_list)

        (
            c2ws,
            intrs,
            rgbs,
            bg_colors,
            masks,
            padding_masks,
            head_rgbs,
            head_only_masks,
            use_heads,
            smplx_params,
        ) = self.get_tgt_view_img(tar_item, uid, all_frames, frame_id_list)

        def isnan_smplx_params(data):
            for key in data.keys():
                if torch.isnan(data[key].sum()).any():
                    return True
            return False

        assert (
            not torch.isnan(rgbs.sum()).any()
            and not torch.isnan(masks.sum())
            and not isnan_smplx_params(smplx_params)
        )

        # reference images
        frame_id_source_list = frame_id_list[0:1]
        source_rgbs, source_head_rgbs, src_use_heads = self.get_src_view_img(
            tar_item, uid, all_frames, frame_id_source_list
        )

        render_image = rgbs
        render_mask = masks
        render_head_mask = head_only_masks
        render_padding_mask = padding_masks
        tgt_size = render_image.shape[2:4]  # [H, W]
        assert (
            abs(intrs[0, 0, 2] * 2 - render_image.shape[3]) <= 1.1
        ), f"{intrs[0, 0, 2] * 2}, {render_image.shape}"
        assert (
            abs(intrs[0, 1, 2] * 2 - render_image.shape[2]) <= 1.1
        ), f"{intrs[0, 1, 2] * 2}, {render_image.shape}"

        # assert src_use_heads.sum() == 1  # only train when head exists

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs.clamp(0, 1),  # [N1, 3, H, W]
            "render_image": render_image.clamp(0, 1),  # [N, 3, H, W]
            "render_mask": render_mask.clamp(0, 1),  # [ N, 1, H, W]
            "render_padding_mask": render_padding_mask.clamp(0, 1),  # [N, 3, H, W]
            "source_head_rgbs": source_head_rgbs.clamp(0, 1),  # [N1, 3, H, W]
            "render_head_mask": render_head_mask.clamp(0, 1),  # [ N, 1, H, W]
            "use_heads": use_heads.float().clamp(0, 1),  # [N, 1]
            "c2ws": c2ws,  # [N, 4, 4]
            "intrs": intrs,  # [N, 4, 4]
            "render_full_resolutions": torch.tensor(
                [tgt_size], dtype=torch.float32
            ).repeat(
                self.sample_side_views + 1, 1
            ),  # [N, 2]
            "render_bg_colors": bg_colors,  # [N, 3]
        }

        ret.update(smplx_params)

        return ret


if __name__ == "__main__":
    import cv2

    # root_dir = "./example_data"
    root_dir = "./train_data/ClothVideo/selected_dataset_v5_tar"
    meta_path = (
        "./train_data/ClothVideo/label/valid_selected_datasetv5_val_filter40K.json"
    )

    root_dir = "/mnt/workspaces/dataset/video_human_datasets/synthetic_data_tar"
    # meta_path = "./train_data/ClothVideo/label/valid_LHM_synthetic_dataset_val_100.json"
    meta_path = (
        "./train_data/ClothVideo/label/valid_LHM_synthetic_dataset_train_5K.json"
    )

    root_dir = "/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v5_tar/"
    meta_path = "/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_val_filter460-self-rotated-69.json"
    dataset = VideoHumanDataset(
        root_dirs=root_dir,
        meta_path=meta_path,
        sample_side_views=10,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        source_image_res=384,
        debug=False,
        use_flame=True,
        random_sample_ratio=1.0,
        processing_pipeline=[
            dict(name="Flip"),
            dict(name="PadRatio", target_ratio=5 / 3, tgt_max_size_list=[840]),
            dict(name="UpRandomCrop", low=0.5, high=1.1),
            dict(name="ToTensor"),
        ],
    )

    save_path = "./debug_vis_synthetic"

    for idx in range(0, 20):
        data = dataset[idx]

        os.makedirs(f"{save_path}/dataloader", exist_ok=True)
        for i in range(data["source_rgbs"].shape[0]):
            cv2.imwrite(
                f"{save_path}/dataloader/source_rgbs_{i}_b{idx}.jpg",
                (
                    (
                        data["source_rgbs"][i].permute(1, 2, 0).numpy()[:, :, (2, 1, 0)]
                        * 255
                    ).astype(np.uint8)
                ),
            )

        for i in range(data["source_head_rgbs"].shape[0]):
            cv2.imwrite(
                f"{save_path}/dataloader/source_head_rgbs_{i}_b{idx}.jpg",
                (
                    (
                        data["source_head_rgbs"][i]
                        .permute(1, 2, 0)
                        .numpy()[:, :, (2, 1, 0)]
                        * 255
                    ).astype(np.uint8)
                ),
            )
        for i in range(data["render_padding_mask"].shape[0]):
            cv2.imwrite(
                f"{save_path}/dataloader/render_padding_mask_{i}_b{idx}.jpg",
                ((data["render_padding_mask"][i][0].numpy() * 255).astype(np.uint8)),
            )

        for i in range(data["render_image"].shape[0]):
            cv2.imwrite(
                f"{save_path}/dataloader/rgbs{i}_b{idx}.jpg",
                (
                    (
                        data["render_image"][i]
                        .permute(1, 2, 0)
                        .numpy()[:, :, (2, 1, 0)]
                        * 255
                    ).astype(np.uint8)
                ),
            )

            render_image = data["render_image"][i]
            render_head_mask = data["render_head_mask"][i]
            render_head_image = render_image * render_head_mask

            cv2.imwrite(
                f"{save_path}/dataloader/head_rgbs{i}_b{idx}.jpg",
                (
                    (
                        render_head_image.permute(1, 2, 0).numpy()[:, :, (2, 1, 0)]
                        * 255
                    ).astype(np.uint8)
                ),
            )
