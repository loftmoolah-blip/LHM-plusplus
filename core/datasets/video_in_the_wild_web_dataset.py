# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu & Peihao Li
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-07 23:56:19
# @Function      : video_human_dataset, used tar to I/O data.
# @Describ:  in the wild video


# from megfile import smart_path_join, smart_open
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from rembg import remove

sys.path.append("./")

from engine.pose_estimation.pose_estimator import PoseEstimator
from core.datasets.data_utils import src_center_crop_according_to_mask
from core.datasets.video_human_dataset_a4o import VideoHumanA4ODataset
from core.utils.proxy import no_proxy

__all__ = ["WebInTheWildEval", "WebInTheWildHeurEval"]


class WebInTheWildEval(VideoHumanA4ODataset):
    """Evaluation dataset for web-based in-the-wild images.

    Processes images from web sources with automatic pose estimation and
    background removal using rembg. Designed for evaluation of single-image
    or few-image human reconstruction.

    Attributes:
        heuristic_sampling (bool): Whether to use heuristic sampling (unused).
        pose_estimator (PoseEstimator): SMPLX pose estimator for extracting
            body shape parameters.
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
        random_sample_ratio: float = 0.9,
        heuristic_sampling: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize web-based in-the-wild evaluation dataset.

        Args:
            root_dirs (str): Root directory containing image folders.
            meta_path (str): Path to metadata file (currently unused).
            sample_side_views (int): Number of side views to sample.
            render_image_res_low (int): Low resolution for rendering.
            render_image_res_high (int): High resolution for rendering.
            render_region_size (Union[int, List[int], Tuple[int, int]]): Rendering
                region size.
            source_image_res (int): Resolution for source images.
            repeat_num (int, optional): Dataset repetition factor. Defaults to 1.
            crop_range_ratio_hw (Optional[List[float]], optional): Crop ratios.
                Defaults to [1.0, 1.0].
            aspect_standard (float, optional): Standard aspect ratio (h/w).
                Defaults to 5.0/3.
            enlarge_ratio (Optional[List[float]], optional): Enlargement range.
                Defaults to [0.9, 1.1].
            debug (bool, optional): Enable debug mode. Defaults to False.
            is_val (bool, optional): Whether this is validation set. Defaults to False.
            random_sample_ratio (float, optional): Random sampling probability.
                Defaults to 0.9.
            heuristic_sampling (bool, optional): Use heuristic sampling (unused).
                Defaults to False. **kwargs: Additional arguments passed to parent class.

        Note:
            - Automatically initializes PoseEstimator on CUDA device.
            - Requires pretrained_models/human_model_files/ for pose estimation.
        """
        if crop_range_ratio_hw is None:
            crop_range_ratio_hw = [1.0, 1.0]
        if enlarge_ratio is None:
            enlarge_ratio = [0.9, 1.1]

        super(WebInTheWildEval, self).__init__(
            root_dirs,
            meta_path,
            sample_side_views,
            render_image_res_low,
            render_image_res_high,
            render_region_size,
            source_image_res,
            repeat_num,
            crop_range_ratio_hw,
            aspect_standard,
            enlarge_ratio,
            debug,
            is_val,
            random_sample_ratio,
            **kwargs,
        )

        # Sampling strategy (currently unused)
        self.heuristic_sampling = heuristic_sampling

        # Initialize pose estimator
        self.pose_estimator = PoseEstimator(
            "./pretrained_models/human_model_files/", device="cuda"
        )

    def get_src_view_img(
        self,
        img_root: str,
        sample_views: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load and process source view images from file paths.

        Reads images from disk, estimates pose from the first image, processes
        all images with background removal and augmentation, then pads to fixed size.

        Args:
            img_root (str): Root directory containing the images (unused).
            sample_views (List[str]): List of image file paths to load.

        Returns:
            Tuple containing:
                - source_rgbs (torch.Tensor): Source RGB images [N, 3, H, W],
                    zero-padded to [ref_img_size, 3, H, W].
                - source_head_rgbs (torch.Tensor): Source head images [N, 3, h, w],
                    zero-padded to [ref_img_size, 3, h, w].
                - src_use_heads (torch.Tensor): Head validity flags [N, 1],
                    zero-padded to [ref_img_size, 1].
                - betas (torch.Tensor): SMPLX shape parameters from pose estimation.
                - tmp_view_size (torch.Tensor): Boolean mask indicating valid
                    reference images [ref_img_size].

        Note:
            - Pose estimation is performed only on the first image.
            - All images are processed through self.pipeline for augmentation.
            - Fixed background color of 1.0 (white) is used.
        """
        source_rgbs, source_head_rgbs = [], []
        src_use_heads = []
        betas = None

        # Load and process source images
        for img_path in sample_views:
            img = cv2.imread(img_path)
            img = img[..., ::-1]  # BGR2RGB
            bg_color = 1.0

            # Estimate pose from first image only
            if betas is None:
                shape_pose = self.pose_estimator(img_path)
                betas = torch.from_numpy(shape_pose.beta)

            rgb, src_head_rgb, src_use_head = self.load_src_rgb_image_with_aug_bg(
                img,
                bg_color=bg_color,
                pad_ratio=0.0,
                aspect_standard=5.0 / 3.0,
                enlarge_ratio=[1.0, 1.0],
                render_tgt_size=self.source_image_res,
                multiply=self.multiply,
            )

            source_rgbs.append(rgb)
            source_head_rgbs.append(src_head_rgb)
            src_use_heads.append(src_use_head)

        # Apply augmentation pipeline
        source_rgbs = self.pipeline(source_rgbs)

        source_head_rgbs = torch.cat(source_head_rgbs, dim=0)
        src_use_heads = torch.tensor(src_use_heads).unsqueeze(-1)

        # Create validity mask
        tmp_view_size = torch.zeros(self.ref_img_size).to(src_use_heads)
        tmp_view_size[: source_rgbs.shape[0]] = 1

        # Pad to fixed size if needed
        if self.ref_img_size > 1:
            tmp_source_rgbs = torch.zeros(self.ref_img_size, *source_rgbs.shape[1:]).to(
                source_rgbs
            )
            tmp_source_head_rgbs = torch.zeros(
                self.ref_img_size, *source_head_rgbs.shape[1:]
            ).to(source_head_rgbs)
            tmp_src_use_heads = torch.zeros(
                self.ref_img_size, *src_use_heads.shape[1:]
            ).to(src_use_heads)

            tmp_source_rgbs[: source_rgbs.shape[0]] = source_rgbs
            tmp_source_head_rgbs[: source_head_rgbs.shape[0]] = source_head_rgbs
            tmp_src_use_heads[: src_use_heads.shape[0]] = src_use_heads

            return (
                tmp_source_rgbs,
                tmp_source_head_rgbs,
                tmp_src_use_heads,
                betas,
                tmp_view_size,
            )
        else:
            return source_rgbs, source_head_rgbs, src_use_heads, betas, tmp_view_size

    def _load_uids(self, meta_path: Optional[str]) -> List[str]:
        """Load unique identifiers (UIDs) from directory structure.

        Args:
            meta_path (Optional[str]): Path to metadata file. Currently unused,
                UIDs are loaded from directory structure.

        Returns:
            List[str]: List of folder paths, each containing images for one subject.

        Raises:
            NotImplementedError: If meta_path is provided (not yet supported).

        Note:
            Expects directory structure: root_dirs/folder/folder.png
        """
        if meta_path is None:
            folders = os.listdir(self.root_dirs)
            uids = [
                os.path.join(self.root_dirs, folder, os.path.basename(folder) + ".png")
                for folder in folders
            ]
        else:
            raise NotImplementedError("Loading from meta_path not yet supported")

        return uids

    @no_proxy
    def inner_get_item(self, idx: int, view: int = 4) -> Dict[str, Any]:
        """Get a single evaluation data item from web images.

        Loads images from a folder, sorts them by seed number, and processes
        them for evaluation. No rendering targets are provided.

        Args:
            idx (int): Index of the item to retrieve.
            view (int, optional): Number of views to sample (unused). Defaults to 4.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - uid (str): Unique identifier (parent folder name).
                - source_rgbs (torch.Tensor): Source images [N, 3, H, W].
                - render_image (None): Placeholder for compatibility.
                - render_mask (None): Placeholder for compatibility.
                - render_padding_mask (None): Placeholder for compatibility.
                - source_head_rgbs (torch.Tensor): Source head images [N, 3, h, w].
                - render_head_mask (None): Placeholder for compatibility.
                - use_heads (torch.Tensor): Head validity flags [N, 1].
                - ref_imgs_bool (torch.Tensor): Valid reference image mask [N].
                - c2ws (None): Placeholder for compatibility.
                - intrs (None): Placeholder for compatibility.
                - render_full_resolutions (None): Placeholder for compatibility.
                - render_bg_colors (torch.Tensor): Background color [1, 3].
                - betas (torch.Tensor): SMPLX shape parameters.

        Raises:
            NotImplementedError: If heuristic_sampling is True.

        Note:
            Images are sorted by seed number extracted from filename format:
            *-seedXXX.png
        """

        img_root = self.uids[idx]
        img_paths = os.listdir(img_root)
        img_paths = sorted(
            img_paths,
            key=lambda x: int(x.replace(".png", "").split("-")[-1].replace("seed", "")),
        )
        all_frames = [os.path.join(img_root, img_path) for img_path in img_paths]

        # Sample views
        if not self.heuristic_sampling:
            sample_views = all_frames[: self.ref_img_size]
        else:
            raise NotImplementedError("Heuristic sampling not supported")

        uid = img_root.split("/")[-2]

        # Load source view images
        source_rgbs, source_head_rgbs, src_use_heads, betas, ref_img_tensor = (
            self.get_src_view_img(img_root, sample_views)
        )

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs.clamp(0, 1),
            "render_image": None,
            "render_mask": None,
            "render_padding_mask": None,
            "source_head_rgbs": source_head_rgbs.clamp(0, 1),
            "render_head_mask": None,
            "use_heads": src_use_heads.float().clamp(0, 1),
            "ref_imgs_bool": ref_img_tensor,
            "c2ws": None,
            "intrs": None,
            "render_full_resolutions": None,
            "render_bg_colors": torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).float(),
        }

        smplx_params = dict(
            betas=betas.float(),
        )

        ret.update(smplx_params)
        return ret

    def __getitem__(self, idx: int, view: int = 4) -> Dict[str, Any]:
        """Get item by index.

        Args:
            idx (int): Dataset index.
            view (int, optional): Number of views (unused). Defaults to 4.

        Returns:
            Dict[str, Any]: Data dictionary from inner_get_item.
        """
        return self.inner_get_item(idx, view)

    def load_src_rgb_image_with_aug_bg(
        self,
        rgb: np.ndarray,
        bg_color: float,
        pad_ratio: float,
        aspect_standard: float,
        enlarge_ratio: List[float],
        render_tgt_size: int,
        multiply: int,
    ) -> Tuple[np.ndarray, torch.Tensor, bool]:
        """Load and process source RGB image with automatic background removal.

        Uses rembg for automatic background removal, then applies optional
        super-resolution and center cropping.

        Args:
            rgb (np.ndarray): Input RGB image of shape (H, W, 3), uint8.
            bg_color (float): Background fill color (in range [0, 1]).
            pad_ratio (float): Padding ratio (unused).
            aspect_standard (float): Target aspect ratio (height/width).
            enlarge_ratio (List[float]): [min, max] range for random enlargement.
            render_tgt_size (int): Target rendering size (unused).
            multiply (int): Size multiplier (unused).

        Returns:
            Tuple[np.ndarray, torch.Tensor, bool]: A tuple containing:
                - rgb (np.ndarray): Processed RGB image as float32 array.
                - src_head_rgb (torch.Tensor): Zero head image [1, 3, h, w].
                - use_head (bool): Always False (no head processing).

        Note:
            - Uses rembg library for automatic background segmentation.
            - No head detection/cropping is performed.
        """

        # Automatic background removal using rembg
        mask = remove(rgb)[..., 3]

        # Normalize to [0, 1]
        rgb = rgb / 255.0

        # Process mask
        if mask is not None:
            mask = mask / 255.0
        else:
            # Use alpha channel as mask
            assert rgb.shape[2] == 4, "RGB must have 4 channels if mask is None"
            mask = rgb[:, :, 3]

        mask = (mask > 0.5).astype(np.float32)

        # Center crop to enlarge human area
        rgb, mask, offset_x, offset_y, head_bbox = src_center_crop_according_to_mask(
            rgb,
            mask,
            aspect_standard,
            enlarge_ratio,
            head_bbox=None,
        )

        # No head processing for web images
        src_head_rgb = np.zeros((self.src_head_size, self.src_head_size, 3))
        use_head = False

        src_head_rgb = (
            torch.from_numpy(src_head_rgb).float().permute(2, 0, 1).unsqueeze(0)
        )

        return rgb, src_head_rgb, use_head


class WebInTheWildHeurEval(WebInTheWildEval):
    """Web in-the-wild evaluation dataset with heuristic view selection.

    Extends WebInTheWildEval to support heuristic view sampling based on
    viewpoint labels (front, back, left, right, etc.) embedded in filenames.

    Note:
        Expects filenames with viewpoint labels: *-{viewpoint}-*.png
        where viewpoint is one of: front, back, left, right, frontleft,
        backleft, frontright, backright.
    """

    def _load_uids(self, meta_path: Optional[str]) -> List[str]:
        """Load UIDs and filter out empty folders.

        Args:
            meta_path (Optional[str]): Path to metadata file (currently unused).

        Returns:
            List[str]: List of valid folder paths containing at least one image.

        Raises:
            NotImplementedError: If meta_path is provided (not yet supported).
        """
        if meta_path is None:
            folders = os.listdir(self.root_dirs)
            uids = [
                os.path.join(self.root_dirs, os.path.basename(folder))
                for folder in folders
            ]
        else:
            raise NotImplementedError("Loading from meta_path not yet supported")

        # Filter out empty folders
        valid_uid = []
        for uid in uids:
            imgs = os.listdir(uid)
            if len(imgs) > 0:
                valid_uid.append(uid)

        return valid_uid

    @no_proxy
    def inner_get_item(self, idx: int, view: int = 4) -> Dict[str, Any]:
        """Get evaluation data item with heuristic viewpoint-based sampling.

        Groups images by viewpoint labels extracted from filenames, then
        samples images in a round-robin fashion across viewpoints for
        balanced coverage.

        Args:
            idx (int): Index of the item to retrieve.
            view (int, optional): Number of views to sample (unused). Defaults to 4.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - uid (str): Unique identifier (folder name).
                - source_rgbs (torch.Tensor): Source images [N, 3, H, W].
                - render_image (None): Placeholder for compatibility.
                - render_mask (None): Placeholder for compatibility.
                - render_padding_mask (None): Placeholder for compatibility.
                - source_head_rgbs (torch.Tensor): Source head images [N, 3, h, w].
                - render_head_mask (None): Placeholder for compatibility.
                - use_heads (torch.Tensor): Head validity flags [N, 1].
                - ref_imgs_bool (torch.Tensor): Valid reference image mask [N].
                - c2ws (None): Placeholder for compatibility.
                - intrs (None): Placeholder for compatibility.
                - render_full_resolutions (None): Placeholder for compatibility.
                - render_bg_colors (torch.Tensor): Background color [1, 3].
                - betas (torch.Tensor): SMPLX shape parameters.

        Raises:
            NotImplementedError: If heuristic_sampling is True.

        Note:
            Viewpoint label is extracted from the second-to-last component of
            filename when split by '-': *-{viewpoint}-*.png
        """
        crop_ratio_h, crop_ratio_w = self.crop_range_ratio_hw

        img_root = self.uids[idx]
        img_paths = os.listdir(img_root)
        img_paths = sorted(img_paths)
        all_frames = [os.path.join(img_root, img_path) for img_path in img_paths]

        # Viewpoint keys for heuristic sampling
        heur_keys = [
            "front",
            "back",
            "left",
            "right",
            "frontleft",
            "backleft",
            "frontright",
            "backright",
        ]
        heur_frames = defaultdict(list)

        # Group frames by viewpoint label
        for frame in all_frames:
            heur_key = frame.split("/")[-1].split("-")[-2]
            heur_frames[heur_key].append(frame)

        # Round-robin sampling across viewpoints
        if not self.heuristic_sampling:
            sample_views = []
            idx = 0
            while len(sample_views) < self.ref_img_size and len(sample_views) < len(
                all_frames
            ):
                heur_key = heur_keys[idx]
                if len(heur_frames[heur_key]) > 0:
                    sample_views.append(heur_frames[heur_key].pop(0))
                idx += 1
                idx = idx % len(heur_keys)
        else:
            raise NotImplementedError("Heuristic sampling mode not supported")

        uid = img_root.split("/")[-1].replace(".png", "")

        # Load source view images
        source_rgbs, source_head_rgbs, src_use_heads, betas, ref_img_tensor = (
            self.get_src_view_img(img_root, sample_views)
        )

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs.clamp(0, 1),
            "render_image": None,
            "render_mask": None,
            "render_padding_mask": None,
            "source_head_rgbs": source_head_rgbs.clamp(0, 1),
            "render_head_mask": None,
            "use_heads": src_use_heads.float().clamp(0, 1),
            "ref_imgs_bool": ref_img_tensor,
            "c2ws": None,
            "intrs": None,
            "render_full_resolutions": None,
            "render_bg_colors": torch.Tensor([1.0, 1.0, 1.0]).unsqueeze(0).float(),
        }

        smplx_params = dict(
            betas=betas.float(),
        )

        ret.update(smplx_params)
        return ret
