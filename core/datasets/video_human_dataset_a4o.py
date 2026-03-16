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
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

sys.path.append("./")

from core.datasets.data_utils import SrcImagePipeline, src_center_crop_according_to_mask
from core.datasets.video_human_dataset import VideoHumanDataset
from core.datasets.video_tar import TarFrameDataset
from core.utils.bbox import Bbox
from core.utils.proxy import no_proxy

__all__ = ["VideoHumanA4ODataset", "VideoHumanA4ODatasetEval", "VideoHumanA4OBenchmark"]


class VideoHumanA4ODataset(VideoHumanDataset):
    """Video Human A4O (Any-in-One) Dataset with flexible multi-view support.

    This dataset extends VideoHumanDataset with:
    - Variable number of reference images (1 to ref_img_size)
    - Source image processing pipeline with data augmentation
    - Optional canonical pose constraints for unseen regions
    - Random multi-view sampling strategies

    Attributes:
        random_sample_ratio (float): Probability of sampling random views vs input views.
        constrain_cano (bool): Whether to use canonical pose constraints.
        pipeline (SrcImagePipeline): Image processing pipeline for source images.
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
        constrain_cano: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize Video Human A4O Dataset.

        Args:
            root_dirs (str): Root directory containing tar archive files.
            meta_path (str): Path to metadata JSON file.
            sample_side_views (int): Number of side views to sample for rendering.
            render_image_res_low (int): Low resolution for rendered images.
            render_image_res_high (int): High resolution for rendered images.
            render_region_size (Union[int, List[int], Tuple[int, int]]): Rendering
                region size.
            source_image_res (int): Resolution for source reference images.
            repeat_num (int, optional): Dataset repetition factor. Defaults to 1.
            crop_range_ratio_hw (Optional[List[float]], optional): Crop ratios [h, w].
                Defaults to [1.0, 1.0].
            aspect_standard (float, optional): Standard aspect ratio (h/w).
                Defaults to 5.0/3.
            enlarge_ratio (Optional[List[float]], optional): Enlargement range [min, max].
                Defaults to [0.9, 1.1].
            debug (bool, optional): Enable debug mode. Defaults to False.
            is_val (bool, optional): Whether this is validation set. Defaults to False.
            random_sample_ratio (float, optional): Probability of random view sampling.
                Defaults to 0.9.
            constrain_cano (bool, optional): Use canonical pose constraints.
                Defaults to False.
            **kwargs: Additional arguments including:
                - processing_pipeline (List[Dict]): Image processing pipeline config.
                - multiply (int): Size multiplier.
                - use_flame (bool): Use FLAME face model.
                - src_head_size (int): Source head image size.
                - ref_img_size (int): Number of reference images.

        Raises:
            KeyError: If 'processing_pipeline' not in kwargs.
        """
        if crop_range_ratio_hw is None:
            crop_range_ratio_hw = [1.0, 1.0]
        if enlarge_ratio is None:
            enlarge_ratio = [0.9, 1.1]

        super(VideoHumanA4ODataset, self).__init__(
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
            **kwargs,
        )

        # Multi-view sampling configuration
        self.random_sample_ratio = random_sample_ratio

        # Canonical pose constraints
        self.constrain_cano = constrain_cano
        if self.constrain_cano:
            self._load_constrain_cano()

        # Source image processing pipeline
        processing_list = kwargs["processing_pipeline"]
        self.pipeline = SrcImagePipeline(*processing_list)

    def _load_constrain_cano(self) -> None:
        """Load canonical pose constraints from predefined JSON file.

        Loads canonical (T-pose or A-pose) constraints including camera parameters
        and SMPLX parameters for regularizing unseen body parts during training.

        Note:
            TODO: Remove hardcoded path and make it configurable.

        Side Effects:
            Sets self.constrain_params with canonical constraints including:
                - c2ws (torch.Tensor): Canonical camera-to-world matrices.
                - intrs (torch.Tensor): Canonical camera intrinsics.
                - smplx_params (Dict[str, torch.Tensor]): Canonical SMPLX params.
        """
        # TODO: Make this path configurable via constructor or config file
        cano_path = (
            "/mnt/workspaces/rmbg/papers/heyuan/stablenorml/"
            "d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/"
            "clean_labels/constrain_cano.json"
        )

        with open(cano_path, "r") as reader:
            cano_constrain = json.load(reader)
            smplx_params = cano_constrain.pop("smplx_params")

        # Convert to tensors
        smplx_tmp = {}
        for key, v in smplx_params.items():
            smplx_tmp[key] = torch.asarray(v)

        cano_constrain["c2ws"] = torch.asarray(cano_constrain["c2ws"])
        cano_constrain["intrs"] = torch.asarray(cano_constrain["intrs"])
        cano_constrain["smplx_params"] = smplx_tmp

        self.constrain_params = cano_constrain

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
        """Load and process source RGB image with background augmentation (A4O version).

        This is a simplified version of the parent class method optimized for
        source image processing without padding masks.

        Args:
            rgb (np.ndarray): Input RGB image of shape (H, W, 3) or (H, W, 4).
            mask (Optional[np.ndarray]): Input mask of shape (H, W), or None to
                use alpha channel.
            bg_color (Union[float, Tuple[float, ...]]): Background fill color.
            pad_ratio (float): Padding ratio (unused in this version).
            max_tgt_size (int): Maximum target size for resizing.
            aspect_standard (float): Target aspect ratio (height/width).
            enlarge_ratio (List[float]): [min, max] range for random enlargement.
            render_tgt_size (int): Target rendering size (unused in this version).
            multiply (int): Size multiplier (unused in this version).
            intr (Union[torch.Tensor, np.ndarray]): Camera intrinsics (unused).
            head_bbox (Optional[Bbox], optional): Head region bounding box.
                Defaults to None.

        Returns:
            Tuple[np.ndarray, torch.Tensor, bool]: A tuple containing:
                - rgb (np.ndarray): Processed RGB image as float32 array.
                - src_head_rgb (torch.Tensor): Cropped head image [1, 3, H, W].
                - use_head (bool): Whether head crop is valid.

        Note:
            If self.womask is True, keeps original RGB without background filling.
        """
        # Normalize to [0, 1]
        rgb = rgb / 255.0
        if mask is not None:
            mask = mask / 255.0
        else:
            # Use alpha channel as mask
            assert rgb.shape[2] == 4, "RGB must have 4 channels if mask is None"
            mask = rgb[:, :, 3]

        mask = (mask > 0.5).astype(np.float32)

        # Apply or skip background filling
        if self.womask:
            rgb = rgb[:, :, :3]
        else:
            rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # Center crop according to mask
        rgb, mask, offset_x, offset_y, head_bbox = src_center_crop_according_to_mask(
            rgb, mask, aspect_standard, enlarge_ratio, head_bbox=head_bbox
        )

        # Resize while keeping aspect ratio
        rgb, head_bbox = self.resize_image_keepaspect_np(rgb, max_tgt_size, head_bbox)
        mask, _ = self.resize_image_keepaspect_np(mask, max_tgt_size, None)

        # Process head region if available
        if head_bbox is not None:
            src_head_rgb = self.crop_head(rgb, head_bbox)
            src_head_rgb = self.img_padding_square(src_head_rgb, bg_color)
            src_head_rgb = cv2.resize(
                src_head_rgb,
                dsize=(self.src_head_size, self.src_head_size),
                interpolation=cv2.INTER_AREA,
            )
            use_head = True
        else:
            src_head_rgb = np.zeros((self.src_head_size, self.src_head_size, 3))
            use_head = False

        src_head_rgb = (
            torch.from_numpy(src_head_rgb).float().permute(2, 0, 1).unsqueeze(0)
        )

        return rgb, src_head_rgb, use_head

    def get_src_view_img(
        self,
        tar_item: TarFrameDataset,
        uid: str,
        all_frames: List[str],
        frame_id_source_list: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get source view images with A4O variable-view padding.

        Extends parent class method to support variable numbers of reference images
        with zero-padding up to self.ref_img_size.

        Args:
            tar_item (TarFrameDataset): Tar dataset containing frame data.
            uid (str): Unique identifier for the sample.
            all_frames (List[str]): List of all available frame paths.
            frame_id_source_list (List[str]): List of frame IDs to use as sources.

        Returns:
            Tuple containing:
                - source_rgbs (torch.Tensor): Source RGB images [N, 3, H, W],
                    zero-padded to [ref_img_size, 3, H, W].
                - source_head_rgbs (torch.Tensor): Source head images [N, 3, h, w],
                    zero-padded to [ref_img_size, 3, h, w].
                - src_use_heads (torch.Tensor): Head validity flags [N, 1],
                    zero-padded to [ref_img_size, 1].
                - betas (torch.Tensor): SMPLX shape parameters.
                - ref_img_tensor (torch.Tensor): Boolean mask indicating valid
                    reference images [ref_img_size].

        Note:
            Uses self.pipeline to apply data augmentation transforms to source images.
        """
        source_rgbs, source_head_rgbs = [], []
        src_use_heads = []
        betas = None

        # Load source images
        for frame_id in frame_id_source_list:

            img, mask, smplx_params, _, max_tgt_size, _, intrinsic, head_bbox, _ = (
                self._process_frame_data(tar_item, frame_id)
            )
            if betas is None:
                betas = smplx_params["betas"]
            # only white bg
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

        source_rgbs = self.pipeline(source_rgbs)
        source_head_rgbs = torch.cat(source_head_rgbs, dim=0)
        src_use_heads = torch.tensor(src_use_heads).unsqueeze(-1)

        tmp_view_size = torch.zeros(self.ref_img_size).to(src_use_heads)
        tmp_view_size[: source_rgbs.shape[0]] = 1

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

    def load_cano_image_with_aug_bg(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
        bg_color: Union[float, Tuple[float, ...]],
        aspect_standard: float,
        enlarge_ratio: List[float],
        render_tgt_size: int,
        multiply: int,
        intr: Union[torch.Tensor, np.ndarray],
        head_bbox: Optional[Bbox] = None,
    ):
        """Load canonical pose image with background augmentation.

        Processes images from canonical (T-pose or A-pose) renders for use as
        constraints during training.

        Args:
            rgb (np.ndarray): Input RGB image of shape (H, W, 3).
            mask (np.ndarray): Binary mask of shape (H, W) or (H, W, 3).
            bg_color (Union[float, Tuple[float, ...]]): Background fill color.
            aspect_standard (float): Target aspect ratio (height/width).
            enlarge_ratio (List[float]): [min, max] range for random enlargement.
            render_tgt_size (int): Target rendering resolution.
            multiply (int): Size multiplier for alignment.
            intr (Union[torch.Tensor, np.ndarray]): Camera intrinsics to update.
            head_bbox (Optional[Bbox], optional): Head bounding box. Defaults to None.

        Returns:
            Tuple containing:
                - rgb (torch.Tensor): Processed RGB [1, 3, H, W].
                - mask (torch.Tensor): Binary mask [1, 1, H, W].
                - padding_mask (torch.Tensor): Padding mask [1, 1, H, W].
                - intr (Union[torch.Tensor, np.ndarray]): Updated intrinsics.
                - src_head_rgb (torch.Tensor): Head image [1, 3, h, w] (zeros).
                - head_only_mask (torch.Tensor): Head mask [1, 1, H, W] (zeros).
                - use_head (bool): Always False for canonical poses.

        Note:
            Currently returns zero head images as canonical poses typically don't
            have detailed face information.
        """

        rgb = rgb / 255.0
        mask = mask / 255.0
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])
        padding_mask = np.zeros_like(mask)
        # crop image to enlarge human area.
        rgb, mask, padding_mask, offset_x, offset_y, head_bbox = (
            self.center_crop_according_to_mask(
                rgb,
                mask,
                padding_mask,
                aspect_standard,
                enlarge_ratio,
                padding_edge=None,
                head_bbox=head_bbox,
            )
        )
        cv2.imwrite("fuck.png", (rgb * 255).astype(np.uint8))

        intr[0, 2] -= offset_x
        intr[1, 2] -= offset_y

        # head.
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
        # head_only_rgb = head_only_mask[..., None] * rgb
        # cv2.imwrite("head_only_rgb.png", (head_only_rgb * 255).astype(np.uint8))

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

    def get_cano_constrain(
        self,
        uid: str,
        sample_num: int = 2,
    ):
        """Get canonical pose constraints for a specific UID.

        Loads pre-rendered canonical pose images and their corresponding SMPLX
        parameters to regularize unseen body parts during training.

        Args:
            uid (str): Unique identifier for finding canonical pose renders.
            sample_num (int, optional): Number of views to sample. Defaults to 2.

        Returns:
            Tuple containing:
                - c2ws (torch.Tensor): Camera-to-world matrices [N, 4, 4].
                - intrs (torch.Tensor): Camera intrinsics [N, 4, 4].
                - rgbs (torch.Tensor): RGB images [N, 3, H, W].
                - bg_colors (torch.Tensor): Background colors [N, 3].
                - masks (torch.Tensor): Binary masks [N, 1, H, W].
                - padding_masks (torch.Tensor): Padding masks [N, 1, H, W].
                - head_rgbs (torch.Tensor): Head images [N, 3, h, w].
                - head_only_masks (torch.Tensor): Head masks [N, 1, H, W].
                - use_heads (torch.Tensor): Head validity flags [N, 1].
                - smplx_params (Dict[str, torch.Tensor]): SMPLX parameters.

        Raises:
            NotImplementedError: If canonical pose folder doesn't exist.

        Note:
            Expects canonical renders in debug/cano_output/{uid}/
        """
        cano_folder = os.path.join("debug/cano_output", uid)
        motion_id = self.constrain_params["motion_id"][1:]

        constrain_params = deepcopy(self.constrain_params)

        # render params
        c2ws = constrain_params["c2ws"]
        intrs = constrain_params["intrs"]
        smplx_params = constrain_params["smplx_params"]
        c2ws = torch.from_numpy(np.asarray(c2ws)).float()
        intrs = torch.from_numpy(np.asarray(intrs)).float()
        sample_view = np.random.choice([0, 1, 2, 3], sample_num, replace=False)

        c2ws_list = []
        intrs_list = []
        rgbs = []
        padding_masks = []
        bg_colors = []
        masks = []
        head_only_masks = []
        head_rgbs = []
        use_heads = []
        smplx_params_list = []

        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "trans",
            "focal",
            "princpt",
            "img_size_wh",
            "expr",
        ]
        smplx_params_tensor = dict()
        for key in smplx_keys:
            smplx_params_tensor[key] = torch.from_numpy(
                np.asarray(smplx_params[key])
            ).float()[0]

        if os.path.exists(cano_folder):
            # obtain rgb imgs
            # obtain mask imgs
            for view in sample_view:
                view_id = motion_id[view]
                rgb_img_path = os.path.join(cano_folder, f"rgb_{view_id:05d}.png")
                mask_path = os.path.join(cano_folder, f"mask_{view_id:05d}.png")
                bgr_img = cv2.imread(rgb_img_path)
                rgb_img = bgr_img[:, :, ::-1].copy()
                mask = cv2.imread(mask_path)

                # sample bg color
                bg_color = random.choice([0.0, 0.5, 1.0])
                # bg_color = 1.0
                if self.is_val:
                    bg_color = 1.0

                # index start fromr 1
                view_c2ws = c2ws[0, view + 1]
                view_intrs = intrs[0, view + 1]
                (
                    rgb,
                    mask,
                    padding_mask,
                    view_intrs,
                    head_rgb,
                    head_only_mask,
                    use_head,
                ) = self.load_cano_image_with_aug_bg(
                    rgb_img,
                    mask,
                    bg_color,
                    aspect_standard=self.aspect_standard,
                    enlarge_ratio=self.enlarge_ratio if not self.is_val else [1.0, 1.0],
                    render_tgt_size=self.render_image_res,
                    multiply=self.multiply,
                    intr=view_intrs,
                    head_bbox=None,
                )

                c2ws_list.append(view_c2ws)
                intrs_list.append(view_intrs)
                rgbs.append(rgb)
                padding_masks.append(padding_mask)
                bg_colors.append(bg_color)
                masks.append(mask)

                # head accumulation.
                head_only_masks.append(head_only_mask)
                head_rgbs.append(head_rgb)
                use_heads.append(use_head)

                smplx_param_view = dict()
                for key in smplx_keys:
                    smplx_param_view[key] = smplx_params_tensor[key][view + 1]
                # smplx_tensor
                smplx_params_list.append(smplx_param_view)
        else:
            raise NotImplementedError

        c2ws = torch.stack(c2ws_list, dim=0)  # [N, 4, 4]
        intrs = torch.stack(intrs_list, dim=0)  # [N, 4, 4]
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

    @no_proxy
    def inner_get_item(self, idx: int) -> Dict[str, Any]:
        """Get a single training data item with A4O multi-view support.

        Loads both rendering target views and variable number of source reference
        images, optionally adding canonical pose constraints.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: Dictionary containing:
                - uid (str): Unique identifier.
                - source_rgbs (torch.Tensor): Source images [N1, 3, H, W].
                - render_image (torch.Tensor): Rendering targets [N, 3, H, W].
                - render_mask (torch.Tensor): Rendering masks [N, 1, H, W].
                - render_padding_mask (torch.Tensor): Padding masks [N, 1, H, W].
                - source_head_rgbs (torch.Tensor): Source head images [N1, 3, h, w].
                - render_head_mask (torch.Tensor): Head region masks [N, 1, H, W].
                - use_heads (torch.Tensor): Head validity flags [N, 1].
                - ref_imgs_bool (torch.Tensor): Valid reference image mask [N1].
                - c2ws (torch.Tensor): Camera matrices [N, 4, 4].
                - intrs (torch.Tensor): Intrinsics [N, 4, 4].
                - render_full_resolutions (torch.Tensor): Resolutions [N, 2].
                - render_bg_colors (torch.Tensor): Background colors [N, 3].
                - SMPLX parameters (multiple keys).

        Note:
            - With probability random_sample_ratio, samples random views as sources.
            - If constrain_cano is True, adds canonical pose views to targets.
            - Source images are padded to ref_img_size with zeros.
        """

        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        all_frames = uid_info["img_path"]

        valid_frame_items = tar_item.valid_frame_items
        assert len(valid_frame_items) > 5

        ref_img_size = self.ref_img_size
        max_img_size = min(ref_img_size, len(valid_frame_items))

        valid_img_size = [1] + [
            valid_size for valid_size in range(2, max_img_size + 1, 2)
        ]

        # build src image
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

        if self.constrain_cano:
            (
                cano_c2ws,
                cano_intrs,
                cano_rgbs,
                cano_bg_colors,
                cano_masks,
                cano_padding_masks,
                cano_head_rgbs,
                cano_head_only_masks,
                cano_use_heads,
                cano_smplx_params,
            ) = self.get_cano_constrain(uid)

            # concat cano and tgt
            c2ws = torch.cat([cano_c2ws, c2ws], dim=0)
            intrs = torch.cat([cano_intrs, intrs], dim=0)
            rgbs = torch.cat([cano_rgbs, rgbs], dim=0)
            bg_colors = torch.cat([cano_bg_colors, bg_colors], dim=0)
            masks = torch.cat([cano_masks, masks], dim=0)
            padding_masks = torch.cat([cano_padding_masks, padding_masks], dim=0)
            head_rgbs = torch.cat([cano_head_rgbs, head_rgbs], dim=0)
            use_heads = torch.cat([cano_use_heads, use_heads], dim=0)

            betas = smplx_params.pop("betas")
            for key in smplx_params.keys():
                smplx_params[key] = torch.cat(
                    [cano_smplx_params[key], smplx_params[key]], dim=0
                )
            smplx_params["betas"] = betas

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

        # reference images && 10 % choose input view images
        if not self.is_val and np.random.rand() < self.random_sample_ratio:
            src_view_imgs_size = np.random.choice(valid_img_size)
            frame_id_source_list = np.random.choice(
                list(valid_frame_items), size=src_view_imgs_size, replace=False
            )
        else:
            frame_id_source_list = frame_id_list[:max_img_size]

        source_rgbs, source_head_rgbs, src_use_heads, _, ref_img_tensor = (
            self.get_src_view_img(tar_item, uid, all_frames, frame_id_source_list)
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
            "ref_imgs_bool": ref_img_tensor,  # [N]
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


class VideoHumanA4ODatasetEval(VideoHumanA4ODataset):
    """Video Human A4O Evaluation Dataset.

    Evaluation-only dataset that loads source images without rendering targets.
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
        constrain_cano: bool = False,
        **kwargs: Any,
    ) -> None:
        super(VideoHumanA4ODatasetEval, self).__init__(
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
            constrain_cano,
            **kwargs,
        )

    @no_proxy
    def inner_get_item(self, idx, view=16):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """

        def uniform_sample(lst, n_samples):
            if not lst or n_samples <= 0:
                return []
            if n_samples <= 1:
                return [lst[0]]
            if n_samples >= len(lst):
                return lst[:]

            indices = [
                round(i * (len(lst) - 1) / (n_samples - 1)) for i in range(n_samples)
            ]
            return [lst[idx] for idx in indices]

        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        all_frames = uid_info["img_path"]

        valid_frame_items = list(tar_item.valid_frame_items)

        assert len(valid_frame_items) > 5

        ref_img_size = self.ref_img_size
        max_img_size = min(ref_img_size, len(valid_frame_items))
        frame_id_source_list = uniform_sample(valid_frame_items, max_img_size)

        source_rgbs, source_head_rgbs, src_use_heads, betas, ref_img_tensor = (
            self.get_src_view_img(tar_item, uid, all_frames, frame_id_source_list)
        )

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs.clamp(0, 1),  # [N1, 3, H, W]
            "render_image": None,  # [N, 3, H, W]
            "render_mask": None,  # [ N, 1, H, W]
            "render_padding_mask": None,  # [N, 3, H, W]
            "source_head_rgbs": source_head_rgbs.clamp(0, 1),  # [N1, 3, H, W]
            "render_head_mask": None,  # [ N, 1, H, W]
            "use_heads": src_use_heads.float().clamp(0, 1),  # [N, 1]
            "ref_imgs_bool": ref_img_tensor,  # [N]
            "c2ws": None,  # [N, 4, 4]
            "intrs": None,  # [N, 4, 4]
            "render_full_resolutions": None,
            "render_bg_colors": torch.Tensor([1.0, 1.0, 1.0])
            .unsqueeze(0)
            .float(),  # [N, 3]
        }

        smplx_params = dict(
            betas=betas.float(),
        )

        ret.update(smplx_params)
        return ret

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
        def to_image(image_numpy):
            image_numpy = (image_numpy * 255).astype(np.uint8)
            image_numpy = image_numpy[..., ::-1]  # RGB2BGR

            return image_numpy

        rgb = rgb / 255.0
        if mask is not None:
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]  # [H, W]

        mask = (mask > 0.5).astype(np.float32)

        if self.womask:
            rgb = rgb[:, :, :3]
        else:
            rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

        # crop image to enlarge human area.
        rgb, mask, offset_x, offset_y, head_bbox = src_center_crop_according_to_mask(
            rgb, mask, aspect_standard, enlarge_ratio, head_bbox=head_bbox
        )

        # resize to render tgt_size for training for face
        src_head_rgb = np.zeros(
            (self.src_head_size, self.src_head_size, 3)
        )  # null image
        use_head = False

        src_head_rgb = (
            torch.from_numpy(src_head_rgb).float().permute(2, 0, 1).unsqueeze(0)
        )

        return rgb, src_head_rgb, use_head

    def __getitem__(self, idx, view):
        return self.inner_get_item(idx, view)

    def getimgs(self, idx):
        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)
        all_frames = uid_info["img_path"]
        valid_frame_items = list(tar_item.valid_frame_items)

        rgb_imgs_numpy = [(tar_item[i]["imgs_png"], i) for i in valid_frame_items]
        return rgb_imgs_numpy, uid


class VideoHumanA4OBenchmark(VideoHumanA4ODataset):
    """Video Human A4O Benchmark Dataset.

    Benchmark dataset that loads ALL available frames as rendering targets
    (instead of sampling a subset), using uniform sampling for source views.
    Used for comprehensive evaluation on complete sequences.

    Note:
        Unlike training/validation, this uses all valid frames for rendering
        to enable thorough quality assessment.
    """

    @no_proxy
    def inner_get_item(self, idx: int) -> Dict[str, Any]:
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """

        def isnan_smplx_params(data):
            for key in data.keys():
                if torch.isnan(data[key].sum()).any():
                    return True
            return False

        def uniform_sample(lst, n_samples):
            if not lst or n_samples <= 0:
                return []
            if n_samples >= len(lst):
                return lst[:]

            indices = [
                round(i * (len(lst) - 1) / (n_samples - 1)) for i in range(n_samples)
            ]
            return [lst[idx] for idx in indices]

        uid_info = self.uids[idx]
        uid = uid_info["id"]
        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        all_frames = uid_info["img_path"]
        valid_frame_items = tar_item.valid_frame_items

        assert len(valid_frame_items) > 5

        ref_img_size = self.ref_img_size
        max_img_size = min(ref_img_size, len(valid_frame_items))

        valid_img_size = [1] + [
            valid_size for valid_size in range(2, max_img_size + 1, 2)
        ]

        frame_id_list = list(valid_frame_items)
        frame_id_list = sorted(frame_id_list)

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

        # add cano constrain params
        if self.constrain_cano:
            (
                cano_c2ws,
                cano_intrs,
                cano_rgbs,
                cano_bg_colors,
                cano_masks,
                cano_padding_masks,
                cano_head_rgbs,
                cano_head_only_masks,
                cano_use_heads,
                cano_smplx_params,
            ) = self.get_cano_constrain(uid)

            # concat cano and tgt
            c2ws = torch.cat([cano_c2ws, c2ws], dim=0)
            intrs = torch.cat([cano_intrs, intrs], dim=0)
            rgbs = torch.cat([cano_rgbs, rgbs], dim=0)
            bg_colors = torch.cat([cano_bg_colors, bg_colors], dim=0)
            masks = torch.cat([cano_masks, masks], dim=0)
            padding_masks = torch.cat([cano_padding_masks, padding_masks], dim=0)
            head_rgbs = torch.cat([cano_head_rgbs, head_rgbs], dim=0)
            use_heads = torch.cat([cano_use_heads, use_heads], dim=0)

            betas = smplx_params.pop("betas")
            for key in smplx_params.keys():
                smplx_params[key] = torch.cat(
                    [cano_smplx_params[key], smplx_params[key]], dim=0
                )
            smplx_params["betas"] = betas

        assert (
            not torch.isnan(rgbs.sum()).any()
            and not torch.isnan(masks.sum())
            and not isnan_smplx_params(smplx_params)
        )

        if max_img_size < 16:
            frame_id_source_list = uniform_sample(frame_id_list, 16)[:max_img_size]
        else:
            frame_id_source_list = uniform_sample(frame_id_list, max_img_size)

        source_rgbs, source_head_rgbs, src_use_heads, _, ref_img_tensor = (
            self.get_src_view_img(tar_item, uid, all_frames, frame_id_source_list)
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
            "ref_imgs_bool": ref_img_tensor,  # [N]
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

    def __getitem__(self, idx):
        return self.inner_get_item(idx)


if __name__ == "__main__":
    import cv2

    # root_dir = "./example_data"
    root_dir = "./train_data/ClothVideo/selected_dataset_v5_tar"
    meta_path = "/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv5_train_filter40K-self-rotated-7147.json"

    root_dir = "/mnt/workspaces/dataset/video_human_datasets/selected_dataset_v6_tar/"
    meta_path = "/mnt/workspaces/dataset/video_human_datasets/clean_labels/valid_selected_datasetv6_Mesh_filter_5W_2.json"

    root_dir = "/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/selected_dataset_v5_tar/"
    meta_path = "/mnt/workspaces/rmbg/papers/heyuan/stablenorml/d6f8c7e1a2b3c4d5e6f7a8b9c0d1e2f3/tmp/video_human_datasets/clean_labels/valid_selected_datasetv5_val_filter460-self-rotated-69.json"

    dataset = VideoHumanA4ODataset(
        root_dirs=root_dir,
        meta_path=meta_path,
        sample_side_views=10,
        render_image_res_low=384,
        render_image_res_high=384,
        render_region_size=(682, 384),
        enlarge_ratio=(1.0, 1.1),
        source_image_res=512,
        debug=False,
        use_flame=True,
        ref_img_size=8,
        is_val=False,
        womask=True,
        random_sample_ratio=0.95,
        constrain_cano=True,
        processing_pipeline=[
            dict(name="Flip"),
            dict(name="PadRatio", target_ratio=5 / 3, tgt_max_size_list=[840]),
            dict(name="ToTensor"),
        ],
    )

    # for idx  in range(0, len(dataset)):
    #     data = dataset[idx]

    for idx in range(20):

        print(f"processing {idx}")

        item = dataset[idx]

        print(item["source_rgbs"].shape)
        os.makedirs("debug_vis_lhm/dataloader", exist_ok=True)
        for i in range(item["source_rgbs"].shape[0]):
            cv2.imwrite(
                f"debug_vis_lhm/dataloader/source_rgbs_b{idx:03d}_{i:03d}.jpg",
                (
                    (
                        item["source_rgbs"][i].permute(1, 2, 0).numpy()[:, :, (2, 1, 0)]
                        * 255
                    ).astype(np.uint8)
                ),
            )

        print(item["render_image"].shape)
        for i in range(item["render_image"].shape[0]):
            cv2.imwrite(
                f"debug_vis_lhm/dataloader/render_rgbs_b{idx:03d}_{i:03d}.jpg",
                (
                    (
                        item["render_image"][i]
                        .permute(1, 2, 0)
                        .numpy()[:, :, (2, 1, 0)]
                        * 255
                    ).astype(np.uint8)
                ),
            )
