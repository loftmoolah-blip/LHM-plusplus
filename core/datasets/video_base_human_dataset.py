# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu & Peihao Li
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-07 23:56:19
# @Function      : video_base_human_dataset, used tar to I/O data.
# @Describ:  Base class for video human dataset.

import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

sys.path.append("./")

from core.datasets.base import BaseDataset
from core.utils.bbox import Bbox

__all__ = ["VideoBaseHumanDataset"]


def smplx_params_to_tensor(smplx_raw_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Convert SMPLX parameters from dict to tensors.

    Args:
        smplx_raw_data (Dict[str, Any]): Dictionary containing raw SMPLX parameters.
            Expected to contain keys like 'betas', 'body_pose', 'root_pose', etc.

    Returns:
        Dict[str, torch.Tensor]: Dictionary with tensor-converted SMPLX parameters.
            Excludes any keys containing 'pad_ratio'.

    Example:
        >>> raw_data = {'betas': [0.1, 0.2], 'pad_ratio': 0.5}
        >>> tensor_data = smplx_params_to_tensor(raw_data)
        >>> 'betas' in tensor_data  # True
        >>> 'pad_ratio' in tensor_data  # False
    """
    return {
        k: torch.FloatTensor(v)
        for k, v in smplx_raw_data.items()
        if "pad_ratio" not in k
    }


class VideoBaseHumanDataset(BaseDataset):
    """Video Base Human Dataset for loading human video data from tar archives.

    This dataset is the base class for all video human datasets.
    It provides the base functionality for all video human datasets.
    """

    @staticmethod
    def _load_pose(pose: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and construct camera pose and intrinsic parameter matrices.

        Args:
            pose (Dict[str, Any]): Dictionary containing camera parameters:
                - focal (List[float]): Focal lengths [fx, fy] in pixels.
                - princpt (List[float]): Principal point [cx, cy] in pixels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - c2w (torch.Tensor): Camera-to-world transformation matrix of
                    shape (4, 4). Currently returns identity matrix.
                - intrinsic (torch.Tensor): Camera intrinsic matrix of shape (4, 4)
                    containing focal lengths and principal point.

        Note:
            The c2w matrix is currently set to identity, meaning the camera is
            at the origin with no rotation.
        """
        # Construct intrinsic matrix
        intrinsic = torch.eye(4, dtype=torch.float32)
        intrinsic[0, 0] = pose["focal"][0]  # fx
        intrinsic[1, 1] = pose["focal"][1]  # fy
        intrinsic[0, 2] = pose["princpt"][0]  # cx
        intrinsic[1, 2] = pose["princpt"][1]  # cy

        # Construct camera-to-world matrix (identity for now)
        c2w = torch.eye(4, dtype=torch.float32)

        return c2w, intrinsic

    def img_center_padding(
        self,
        img_np: np.ndarray,
        pad_ratio: float,
        head_bbox: Optional[Bbox] = None,
    ) -> Tuple[np.ndarray, Optional[Bbox]]:
        """Apply center padding to an image with zero values.

        This function pads the image by a given ratio while keeping the original
        image centered. Useful for data augmentation and ensuring consistent
        aspect ratios.

        Args:
            img_np (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            pad_ratio (float): Ratio of padding relative to original size.
                For example, 0.1 adds 10% padding on each side.
            head_bbox (Optional[Bbox], optional): Bounding box for the head region.
                Will be adjusted to account for padding offset. Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[Bbox]]: A tuple containing:
                - img_pad_np (np.ndarray): Padded image with zeros.
                - head_bbox (Optional[Bbox]): Updated bounding box with adjusted
                    coordinates, or None if input was None.

        Example:
            >>> img = np.ones((100, 100, 3), dtype=np.uint8)
            >>> padded_img, _ = dataset.img_center_padding(img, pad_ratio=0.2)
            >>> padded_img.shape  # (120, 120, 3)
        """
        ori_h, ori_w = img_np.shape[:2]

        # Calculate new dimensions with padding
        h = round((1 + pad_ratio) * ori_h)
        w = round((1 + pad_ratio) * ori_w)

        # Create zero-padded image
        if len(img_np.shape) > 2:
            img_pad_np = np.zeros((h, w, img_np.shape[2]), dtype=np.uint8)
        else:
            img_pad_np = np.zeros((h, w), dtype=np.uint8)

        # Calculate offsets for centering
        offset_h = (h - img_np.shape[0]) // 2
        offset_w = (w - img_np.shape[1]) // 2

        # Place original image in center
        img_pad_np[
            offset_h : offset_h + img_np.shape[0],
            offset_w : offset_w + img_np.shape[1],
        ] = img_np

        # Update bounding box coordinates
        if head_bbox is not None:
            head_bbox.offset(offset_w, offset_h)

        return img_pad_np, head_bbox

    def img_padding_square(
        self,
        img_np: np.ndarray,
        bg_color: Union[float, Tuple[float, ...]],
    ) -> np.ndarray:
        """Pad image to square shape with specified background color.

        Centers the input image in a square canvas, padding with the given
        background color. Useful for preparing images for networks that expect
        square inputs.

        Args:
            img_np (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            bg_color (Union[float, Tuple[float, ...]]): Background color value(s)
                for padding. Single value for grayscale, tuple for RGB.

        Returns:
            np.ndarray: Square padded image of shape (max_dim, max_dim, C) or
                (max_dim, max_dim) where max_dim = max(H, W).

        Example:
            >>> img = np.ones((100, 150, 3), dtype=np.uint8)
            >>> square_img = dataset.img_padding_square(img, bg_color=0.5)
            >>> square_img.shape  # (150, 150, 3)
        """
        ori_h, ori_w = img_np.shape[:2]
        max_scale = max(ori_h, ori_w)

        # Create square canvas with background color
        if len(img_np.shape) > 2:
            img_pad_np = np.full(
                (max_scale, max_scale, img_np.shape[2]), bg_color, dtype=img_np.dtype
            )
        else:
            img_pad_np = np.full((max_scale, max_scale), bg_color, dtype=img_np.dtype)

        # Calculate offsets for centering
        offset_h = (max_scale - ori_h) // 2
        offset_w = (max_scale - ori_w) // 2

        # Place original image in center
        img_pad_np[
            offset_h : offset_h + ori_h,
            offset_w : offset_w + ori_w,
        ] = img_np

        return img_pad_np

    def resize_image_keepaspect_np(
        self,
        img: np.ndarray,
        max_tgt_size: int,
        head_bbox: Optional[Bbox] = None,
    ) -> Tuple[np.ndarray, Optional[Bbox]]:
        """Resize image while preserving aspect ratio.

        Similar to PIL.ImageOps.contain(), this function resizes the image so that
        its longest side equals max_tgt_size while maintaining the original aspect
        ratio.

        Args:
            img (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            max_tgt_size (int): Maximum target size for the longest dimension.
            head_bbox (Optional[Bbox], optional): Bounding box for the head region.
                Will be scaled proportionally. Defaults to None.

        Returns:
            Tuple[np.ndarray, Optional[Bbox]]: A tuple containing:
                - resized_img (np.ndarray): Resized image maintaining aspect ratio.
                - head_bbox (Optional[Bbox]): Scaled bounding box, or None if
                    input was None.

        Example:
            >>> img = np.ones((200, 100, 3), dtype=np.uint8)
            >>> resized, _ = dataset.resize_image_keepaspect_np(img, max_tgt_size=100)
            >>> resized.shape  # (100, 50, 3)
        """
        h, w = img.shape[:2]
        ratio = max_tgt_size / max(h, w)
        new_h, new_w = round(h * ratio), round(w * ratio)

        # Scale bounding box if provided
        if head_bbox is not None:
            head_bbox = head_bbox.scale_bbox(h, w, new_h, new_w)

        # Resize image with area interpolation for best quality when downsampling
        resized_img = cv2.resize(
            img, dsize=(new_w, new_h), interpolation=cv2.INTER_AREA
        )

        return resized_img, head_bbox

    def center_crop_according_to_mask(
        self,
        img: np.ndarray,
        mask: np.ndarray,
        padding_mask: Optional[np.ndarray],
        aspect_standard: float,
        enlarge_ratio: List[float],
        padding_edge: Optional[List[int]] = None,
        head_bbox: Optional[Bbox] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], int, int, Optional[Bbox]]:
        """Perform center crop on image according to mask with specified aspect ratio.

        This function crops the image to focus on the masked region while maintaining
        a standard aspect ratio and applying random enlargement for data augmentation.

        Args:
            img (np.ndarray): Input image array of shape (H, W, 3).
            mask (np.ndarray): Binary mask array of shape (H, W) where >0 indicates
                the region of interest.
            padding_mask (Optional[np.ndarray]): Optional padding mask of shape (H, W).
            aspect_standard (float): Target aspect ratio (height/width) to maintain.
            enlarge_ratio (List[float]): [min, max] ratios for random enlargement.
            padding_edge (Optional[List[int]], optional): Edge coordinates of padding
                as [h_min, h_max, w_min, w_max]. Used to detect full body visibility.
                Defaults to None.
            head_bbox (Optional[Bbox], optional): Bounding box for head region.
                Will be adjusted for crop offset. Defaults to None.

        Returns:
            Tuple containing:
                - new_img (np.ndarray): Cropped image.
                - new_mask (np.ndarray): Cropped mask.
                - padding_mask (Optional[np.ndarray]): Cropped padding mask or None.
                - offset_x (int): Horizontal offset of the crop.
                - offset_y (int): Vertical offset of the crop.
                - head_bbox (Optional[Bbox]): Adjusted head bounding box or None.

        Raises:
            Exception: If the mask is empty (no pixels > 0).

        Note:
            The function ensures the crop is centered and maintains the specified
            aspect ratio within a tolerance of 3%.
        """
        ys, xs = np.where(mask > 0)

        if len(xs) == 0 or len(ys) == 0:
            raise Exception("Empty mask: no foreground pixels found")

        # Get bounding box of masked region
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)

        # Check if body is truncated by padding edges (unused variable, kept for compatibility)
        if padding_edge is not None:
            ph_min, ph_max, pw_min, pw_max = padding_edge
            # Check if mask touches padding edges (within 3 pixels)
            is_not_full_body = (
                abs(y_max - ph_max) < 3
                or abs(y_min - ph_min) < 3
                or abs(x_max - pw_max) < 3
                or abs(x_min - pw_min) < 3
            )

        # Calculate image center
        center_x, center_y = img.shape[1] // 2, img.shape[0] // 2

        # Calculate half-widths to cover masked region from center
        half_w = max(abs(center_x - x_min), abs(center_x - x_max))
        half_h = max(abs(center_y - y_min), abs(center_y - y_max))
        aspect = half_h / half_w

        # Adjust dimensions to match target aspect ratio
        if aspect >= aspect_standard:
            half_w = round(half_h / aspect_standard)
        else:
            half_h = round(half_w * aspect_standard)

        # Apply random enlargement within feasible bounds
        enlarge_ratio_min, enlarge_ratio_max = enlarge_ratio
        enlarge_ratio_max_real = min(center_y / half_h, center_x / half_w)
        enlarge_ratio_max = min(enlarge_ratio_max_real, enlarge_ratio_max)
        enlarge_ratio_min = min(enlarge_ratio_max_real, enlarge_ratio_min)
        enlarge_ratio_cur = (
            np.random.rand() * (enlarge_ratio_max - enlarge_ratio_min)
            + enlarge_ratio_min
        )
        half_h = round(enlarge_ratio_cur * half_h)
        half_w = round(enlarge_ratio_cur * half_w)

        # Validate crop dimensions
        assert half_h <= center_y, f"Crop height {half_h} exceeds center {center_y}"
        assert half_w <= center_x, f"Crop width {half_w} exceeds center {center_x}"
        assert (
            abs(half_h / half_w - aspect_standard) < 0.03
        ), f"Aspect ratio mismatch: {half_h / half_w} vs {aspect_standard}"

        # Calculate crop offsets
        offset_x = center_x - half_w
        offset_y = center_y - half_h

        # Perform cropping
        new_img = img[
            offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w
        ]
        new_mask = mask[
            offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w
        ]

        if padding_mask is not None:
            padding_mask = padding_mask[
                offset_y : offset_y + 2 * half_h, offset_x : offset_x + 2 * half_w
            ]

        # Adjust head bounding box for crop offset
        if head_bbox is not None:
            head_bbox.offset(-offset_x, -offset_y)

        return new_img, new_mask, padding_mask, offset_x, offset_y, head_bbox

    def crop_head_debug(
        self,
        img: np.ndarray,
        head_bbox: Bbox,
        scale: float = 1.0,
        save_path: str = "head_img.png",
    ) -> np.ndarray:
        """Crop and save head region from image for debugging purposes.

        Args:
            img (np.ndarray): Input image array of shape (H, W) or (H, W, C).
            head_bbox (Bbox): Bounding box defining the head region.
            scale (float, optional): Scale factor to apply to pixel values.
                Defaults to 1.0.
            save_path (str, optional): Path to save the cropped head image.
                Defaults to "head_img.png".

        Returns:
            np.ndarray: Cropped head image as BGR uint8 array.

        Note:
            This function converts RGB to BGR for OpenCV compatibility and
            saves the result to disk.
        """
        l, t, r, b = head_bbox.get_box()
        l, t = max(l, 0), max(t, 0)

        # Convert grayscale to RGB if needed
        if len(img.shape) == 2:
            img = np.concatenate([img[:, :, None]] * 3, axis=2)

        # Scale and convert to uint8
        img = (img * scale).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Crop head region
        head_img = img[t:b, l:r, :]
        cv2.imwrite(save_path, head_img)

        return head_img

    def crop_head(self, img: np.ndarray, head_bbox: Bbox) -> np.ndarray:
        """Crop head region from image.

        Args:
            img (np.ndarray): Input image array of shape (H, W, C).
            head_bbox (Bbox): Bounding box defining the head region.

        Returns:
            np.ndarray: Cropped head image of shape (h, w, C) where h and w
                are determined by the bounding box dimensions.

        Note:
            Negative coordinates are clamped to zero to avoid index errors.
        """
        l, t, r, b = head_bbox.get_box()
        l, t = max(l, 0), max(t, 0)
        head_img = img[t:b, l:r, :]
        return head_img

    def head_only_mask(self, mask: np.ndarray, head_bbox: Bbox) -> np.ndarray:
        """Create a mask containing only the head area.

        Args:
            mask (np.ndarray): Input binary mask of shape (H, W).
            head_bbox (Bbox): Bounding box defining the head region.

        Returns:
            np.ndarray: New mask of shape (H, W) with only head region preserved,
                all other areas set to zero.

        Example:
            >>> mask = np.ones((100, 100))
            >>> bbox = Bbox([20, 20, 40, 40], mode='whwh')
            >>> head_mask = dataset.head_only_mask(mask, bbox)
            >>> # Only region [20:40, 20:40] is non-zero
        """
        l, t, r, b = head_bbox.get_box()
        l, t = max(l, 0), max(t, 0)

        new_mask = np.zeros_like(mask)
        new_mask[t:b, l:r] = mask[t:b, l:r]

        return new_mask

    def scale_intrs(
        self,
        intrs: Union[torch.Tensor, np.ndarray],
        ratio_x: float,
        ratio_y: float,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Scale camera intrinsic parameters by given ratios.

        Args:
            intrs (Union[torch.Tensor, np.ndarray]): Intrinsic matrix or matrices.
                Can be shape (4, 4) or (N, 4, 4).
            ratio_x (float): Horizontal scaling ratio.
            ratio_y (float): Vertical scaling ratio.

        Returns:
            Union[torch.Tensor, np.ndarray]: Scaled intrinsic parameters with
                focal lengths and principal points adjusted.
        """
        if len(intrs.shape) >= 3:
            intrs[:, 0] = intrs[:, 0] * ratio_x
            intrs[:, 1] = intrs[:, 1] * ratio_y
        else:
            intrs[0] = intrs[0] * ratio_x
            intrs[1] = intrs[1] * ratio_y
        return intrs

    def uniform_sample_in_chunk(
        self,
        sample_num: int,
        sample_data: np.ndarray,
    ) -> List[Any]:
        """Uniformly sample one element from each chunk (random selection).

        Divides the data into `sample_num` chunks and randomly selects one
        element from each chunk. Useful for diverse temporal sampling.

        Args:
            sample_num (int): Number of chunks and samples to generate.
            sample_data (np.ndarray): Array of data to sample from.

        Returns:
            List[Any]: List of sampled elements, one from each chunk.
        """
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(np.random.choice(chunk))
        return select_list

    def uniform_sample_in_chunk_det(
        self,
        sample_num: int,
        sample_data: np.ndarray,
    ) -> List[Any]:
        """Uniformly sample one element from each chunk (deterministic selection).

        Divides the data into `sample_num` chunks and selects the middle element
        from each chunk. Used for validation to ensure reproducibility.

        Args:
            sample_num (int): Number of chunks and samples to generate.
            sample_data (np.ndarray): Array of data to sample from.

        Returns:
            List[Any]: List of sampled elements (middle of each chunk).
        """
        chunks = np.array_split(sample_data, sample_num)
        select_list = []
        for chunk in chunks:
            select_list.append(chunk[len(chunk) // 2])
        return select_list

    def calc_new_tgt_size(
        self,
        cur_hw: Tuple[int, int],
        tgt_size: int,
        multiply: int,
    ) -> Tuple[Tuple[int, int], float, float]:
        """Calculate new target size maintaining aspect ratio and multiplier.

        Args:
            cur_hw (Tuple[int, int]): Current (height, width).
            tgt_size (int): Target size for the minimum dimension.
            multiply (int): Multiplier for quantizing dimensions (e.g., 8, 16).

        Returns:
            Tuple containing:
                - tgt_size (Tuple[int, int]): New (height, width) quantized to multiply.
                - ratio_y (float): Height scaling ratio.
                - ratio_x (float): Width scaling ratio.
        """
        ratio = tgt_size / min(cur_hw)
        tgt_size = (int(ratio * cur_hw[0]), int(ratio * cur_hw[1]))
        tgt_size = (
            int(tgt_size[0] / multiply) * multiply,
            int(tgt_size[1] / multiply) * multiply,
        )
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]
        return tgt_size, ratio_y, ratio_x

    def calc_new_tgt_size_by_aspect(
        self,
        cur_hw: Tuple[int, int],
        aspect_standard: float,
        tgt_size: int,
        multiply: int,
        head_bbox: Optional[Bbox] = None,
    ) -> Tuple[Tuple[int, int], float, float, Optional[Bbox]]:
        """Calculate new target size based on standard aspect ratio.

        Args:
            cur_hw (Tuple[int, int]): Current (height, width).
            aspect_standard (float): Standard aspect ratio (height/width).
            tgt_size (int): Target size for width dimension.
            multiply (int): Multiplier for quantizing dimensions.
            head_bbox (Optional[Bbox], optional): Head bounding box to scale.
                Defaults to None.

        Returns:
            Tuple containing:
                - tgt_size (Tuple[int, int]): New (height, width) quantized to multiply.
                - ratio_y (float): Height scaling ratio.
                - ratio_x (float): Width scaling ratio.
                - head_bbox (Optional[Bbox]): Scaled head bounding box or None.

        Raises:
            AssertionError: If current aspect ratio differs from standard by >3%.
        """
        assert abs(cur_hw[0] / cur_hw[1] - aspect_standard) < 0.03, (
            f"Current aspect {cur_hw[0] / cur_hw[1]} differs from "
            f"standard {aspect_standard} by >3%"
        )

        tgt_size = (tgt_size * aspect_standard, tgt_size)
        tgt_size = (
            int(tgt_size[0] / multiply) * multiply,
            int(tgt_size[1] / multiply) * multiply,
        )
        ratio_y, ratio_x = tgt_size[0] / cur_hw[0], tgt_size[1] / cur_hw[1]

        if head_bbox is not None:
            head_bbox = head_bbox.scale_bbox(
                cur_hw[1], cur_hw[0], tgt_size[1], tgt_size[0]
            )
        return tgt_size, ratio_y, ratio_x, head_bbox

    def _load_hand_bbox(
        self,
        keypoint_path: str,
    ) -> Optional[Tuple[Bbox, Bbox]]:
        """Load left and right hand bounding boxes from keypoint file.

        Args:
            keypoint_path (str): Path to JSON file containing keypoint information.

        Returns:
            Optional[Tuple[Bbox, Bbox]]: Tuple of (left_hand_bbox, right_hand_bbox)
                or None if file doesn't exist or doesn't contain exactly one instance.

        Note:
            Assumes keypoints 91:112 correspond to left hand and 112: to right hand.
        """
        if not os.path.exists(keypoint_path):
            return None

        with open(keypoint_path, "r") as f:
            instance = json.load(f)["instance_info"]

        if len(instance) != 1:
            return None

        keypoints = instance[0]["keypoints"]
        lhand_keypoints = np.array(keypoints[91:112])
        rhand_keypoints = np.array(keypoints[112:])

        # Compute left hand bounding box
        lhand_bbox = [
            lhand_keypoints[:, 0].min(),
            lhand_keypoints[:, 1].min(),
            lhand_keypoints[:, 0].max(),
            lhand_keypoints[:, 1].max(),
        ]
        lhand_bbox = Bbox(list(map(int, lhand_bbox)), mode="whwh")

        # Compute right hand bounding box
        rhand_bbox = [
            rhand_keypoints[:, 0].min(),
            rhand_keypoints[:, 1].min(),
            rhand_keypoints[:, 0].max(),
            rhand_keypoints[:, 1].max(),
        ]
        rhand_bbox = Bbox(list(map(int, rhand_bbox)), mode="whwh")

        return (lhand_bbox, rhand_bbox)

    def train_sample_views(
        self,
        all_frames_res: List[Any],
    ) -> List[Any]:
        """Sample views for training with random strategies.

        Implements two sampling strategies:
        - 70% chance: Uniform sampling from chunks (diverse temporal coverage)
        - 30% chance: Completely random sampling

        Args:
            all_frames_res (List[Any]): List of available frame identifiers.

        Returns:
            List[Any]: List of sampled frame identifiers of length
                (sample_side_views + 1).

        Note:
            If fewer frames than required, sampling is done with replacement.
        """
        all_frames_res = list(all_frames_res)
        all_frame_numpy = np.asarray(all_frames_res)

        if len(all_frames_res) >= self.sample_side_views:
            if np.random.rand() < 0.7:
                # Uniform chunk sampling for temporal diversity
                frame_id_list = self.uniform_sample_in_chunk(
                    self.sample_side_views + 1, np.arange(len(all_frames_res))
                )
            else:
                # Completely random sampling
                frame_id_list = np.random.choice(
                    len(all_frames_res),
                    size=self.sample_side_views + 1,
                    replace=False,
                )
        else:
            # Not enough frames, sample with replacement
            frame_id_list = np.random.choice(
                len(all_frames_res), size=self.sample_side_views + 1, replace=True
            )

        sample_flame_id = all_frame_numpy[frame_id_list].tolist()
        return sample_flame_id

    def val_sample_views(
        self,
        all_frames_res: List[Any],
    ) -> List[Any]:
        """Sample views for validation with deterministic strategy.

        Uses deterministic sampling for reproducibility during validation:
        - Selects middle element from each uniform chunk
        - Uses linear interpolation if fewer frames than required

        Args:
            all_frames_res (List[Any]): List of available frame identifiers.

        Returns:
            List[Any]: List of deterministically sampled frame identifiers
                of length (sample_side_views + 1).
        """
        all_frames_res = list(all_frames_res)
        all_frame_numpy = np.asarray(all_frames_res)

        if len(all_frames_res) >= self.sample_side_views:
            # Deterministic chunk sampling (middle of each chunk)
            frame_id_list = self.uniform_sample_in_chunk_det(
                self.sample_side_views + 1, np.arange(len(all_frames_res))
            )
        else:
            # Linear interpolation for insufficient frames
            frame_id_list = np.linspace(
                0,
                len(all_frames_res) - 1,
                num=self.sample_side_views + 1,
                endpoint=True,
            )
            frame_id_list = [round(e) for e in frame_id_list]

        sample_flame_id = all_frame_numpy[frame_id_list].tolist()
        return sample_flame_id
