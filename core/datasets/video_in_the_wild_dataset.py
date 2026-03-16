# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu & Peihao Li
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-07 23:56:19
# @Function      : In-the-wild video dataset for evaluation and benchmarking

import os
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

sys.path.append("./")

from core.datasets.video_human_dataset_a4o import VideoHumanA4ODataset
from core.datasets.video_tar import TarFrameDataset
from core.utils.bbox import Bbox
from core.utils.proxy import no_proxy

__all__ = ["VideoInTheWildEval", "VideoInTheWildBenchmark"]


class VideoInTheWildEval(VideoHumanA4ODataset):
    """Evaluation dataset for in-the-wild videos.

    This dataset processes unconstrained real-world videos for evaluation,
    supporting uniform or heuristic view sampling.

    Attributes:
        heuristic_sampling (bool): Whether to use heuristic sampling based on
            precomputed view selections.
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
        """Initialize in-the-wild evaluation dataset.

        Args:
            root_dirs (str): Root directory containing tar archive files.
            meta_path (str): Path to metadata JSON file.
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
            heuristic_sampling (bool, optional): Use precomputed heuristic view
                selection instead of uniform sampling. Defaults to False.
            **kwargs: Additional arguments passed to parent class.
        """
        if crop_range_ratio_hw is None:
            crop_range_ratio_hw = [1.0, 1.0]
        if enlarge_ratio is None:
            enlarge_ratio = [0.9, 1.1]

        super(VideoInTheWildEval, self).__init__(
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

        # View sampling strategy
        self.heuristic_sampling = heuristic_sampling

    @no_proxy
    def inner_get_item(self, idx: int, view: int = 4) -> Dict[str, Any]:
        """Get a single evaluation data item with specified number of views.

        Loads source images from in-the-wild videos for evaluation. No rendering
        targets are provided (all set to None).

        Args:
            idx (int): Index of the item to retrieve.
            view (int, optional): Number of views to sample. Defaults to 4.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - uid (str): Unique identifier.
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

        Note:
            - Uniform sampling: linearly samples `view` frames across the sequence.
            - Heuristic sampling: uses precomputed view selections from metadata.
            - Special handling for 'xiaolin' sequences with different sampling.
        """

        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        all_frames = uid_info["img_path"]
        valid_frame_items = list(tar_item.valid_frame_items)

        assert (
            len(valid_frame_items) >= 5
        ), f"Need at least 5 frames, got {len(valid_frame_items)}"

        # Select views based on sampling strategy
        if not self.heuristic_sampling:
            # Uniform sampling across the sequence
            sample_views = list(
                np.linspace(0, len(valid_frame_items) - 1, view, dtype=int)
            )
            sample_views = sorted(sample_views)
            sample_views = [valid_frame_items[i] for i in sample_views]
        else:
            # Heuristic sampling from precomputed selections
            if "xiaolin" in uid:
                # Special handling for xiaolin sequences
                sample_frame_idx = uid_info[f"16_view"][:8][:: 8 // view]
            else:
                sample_frame_idx = uid_info[f"16_view"][:: 16 // view]
            sample_views = [
                int(sample_frame.replace(".png", "").replace(".jpg", ""))
                for sample_frame in sample_frame_idx
            ]

        # Load source view images
        source_rgbs, source_head_rgbs, src_use_heads, betas, ref_img_tensor = (
            self.get_src_view_img(tar_item, uid, all_frames, sample_views)
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
        """Get item by index with specified view count.

        Args:
            idx (int): Dataset index.
            view (int, optional): Number of views. Defaults to 4.

        Returns:
            Dict[str, Any]: Data dictionary from inner_get_item.
        """
        return self.inner_get_item(idx, view)

    def load_src_rgb_image_with_aug_bg(
        self,
        rgb: np.ndarray,
        mask: Optional[np.ndarray],
        bg_color: Union[float, Tuple[float, ...]],
        pad_ratio: float,
        max_tgt_size: int,
        aspect_standard: float,
        enlarge_ratio: List[float],
        render_tgt_size: int,
        multiply: int,
        intr: Union[torch.Tensor, np.ndarray],
        head_bbox: Optional[Bbox] = None,
    ) -> Tuple[np.ndarray, torch.Tensor, bool]:
        """Load and process source RGB image.

        Delegates to parent class method for processing.

        Args:
            rgb (np.ndarray): Input RGB image of shape (H, W, 3) or (H, W, 4).
            mask (Optional[np.ndarray]): Input mask of shape (H, W), or None to
                use alpha channel.
            bg_color (Union[float, Tuple[float, ...]]): Background fill color.
            pad_ratio (float): Padding ratio (unused).
            max_tgt_size (int): Maximum target size (unused).
            aspect_standard (float): Target aspect ratio (height/width).
            enlarge_ratio (List[float]): [min, max] range for random enlargement.
            render_tgt_size (int): Target rendering size (unused).
            multiply (int): Size multiplier (unused).
            intr (Union[torch.Tensor, np.ndarray]): Camera intrinsics (unused).
            head_bbox (Optional[Bbox], optional): Head region bounding box.
                Defaults to None.

        Returns:
            Tuple[np.ndarray, torch.Tensor, bool]: A tuple containing:
                - rgb (np.ndarray): Processed RGB image as float32 array.
                - src_head_rgb (torch.Tensor): Cropped head image [1, 3, H, W].
                - use_head (bool): Whether head crop is valid.

        """
        return super().load_src_rgb_image_with_aug_bg(
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
            head_bbox,
        )


class VideoInTheWildBenchmark(VideoInTheWildEval):
    """Benchmark dataset for comprehensive evaluation of in-the-wild videos.

    Extends VideoInTheWildEval to include test view IDs for thorough evaluation
    across all frames in the sequence.

    Note:
        This class is specifically designed for benchmark testing where we need
        to track which views are used for testing.
    """

    @no_proxy
    def inner_get_item(self, idx: int, view: int = 4) -> Dict[str, Any]:
        """Get benchmark data item with test view tracking.

        Similar to parent class but adds test_views_id to the output for tracking
        which frames should be evaluated.

        Args:
            idx (int): Index of the item to retrieve.
            view (int, optional): Number of views to sample. Defaults to 4.

        Returns:
            Dict[str, Any]: Dictionary containing all parent class outputs plus:
                - test_views_id (List[int]): Frame IDs designated for testing.

        Note:
            - For view <= 16: Uses first `view` frames from precomputed 16_view.
            - For view > 16: Uses precomputed {view}_view metadata.
            - Prints sample_frame_idx for debugging when view <= 16.
        """

        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        all_frames = uid_info["img_path"]
        valid_frame_items = list(tar_item.valid_frame_items)

        assert (
            len(valid_frame_items) >= 5
        ), f"Need at least 5 frames, got {len(valid_frame_items)}"

        # Select views based on sampling strategy
        if not self.heuristic_sampling:
            # Uniform sampling across the sequence
            sample_views = list(
                np.linspace(0, len(valid_frame_items) - 1, view, dtype=int)
            )
            sample_views = sorted(sample_views)
            sample_views = [valid_frame_items[i] for i in sample_views]
        else:
            # Heuristic sampling from precomputed selections
            if view <= 16:
                sample_frame_idx = uid_info[f"16_view"][:view]
                print(sample_frame_idx)  # Debug output
            else:
                sample_frame_idx = uid_info[f"{view}_view"]

            sample_views = [
                int(sample_frame.replace(".png", "").replace(".jpg", ""))
                for sample_frame in sample_frame_idx
            ]

        # Load source view images
        source_rgbs, source_head_rgbs, src_use_heads, betas, ref_img_tensor = (
            self.get_src_view_img(tar_item, uid, all_frames, sample_views)
        )

        # Get test view IDs for evaluation tracking
        test_views = uid_info["img_path"]
        test_views_id = [int(view.replace(".png", "")) for view in test_views]

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
            "test_views_id": test_views_id,
        }

        smplx_params = dict(
            betas=betas.float(),
        )

        ret.update(smplx_params)
        return ret
