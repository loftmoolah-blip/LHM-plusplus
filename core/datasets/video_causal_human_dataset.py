# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu & Peihao Li
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-07 23:56:19
# @Function      : video_human_dataset, used tar to I/O data.
# @Describ:  Due to the fragmented nature of the input video files, we developed this function to read the files using tar.

import glob

# from megfile import smart_path_join, smart_open
import json
import os
import random
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append("./")

from core.datasets.video_human_dataset_a4o import VideoHumanA4ODatasetEval
from core.datasets.video_tar import TarFrameDataset
from core.utils.bbox import Bbox
from core.utils.proxy import no_proxy

__all__ = ["VideoHumanCausalDataset"]


class VideoHumanCausalDataset(VideoHumanA4ODatasetEval):

    @no_proxy
    def inner_get_item(self, idx, view_id=16):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """

        uid_info = self.uids[idx]
        motion_path = uid_info["img_path"]

        uid = uid_info["id"]

        sample_frame_idx = uid_info[f"{view_id}_view"]
        all_frames = uid_info["img_path"]
        sample_frame_idx = [
            int(sample_frame.replace(".png", "").replace(".jpg", ""))
            for sample_frame in sample_frame_idx
        ]

        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)
        frame_id_source_list = sample_frame_idx

        source_rgbs = self.get_src_view_img(
            tar_item, uid, all_frames, frame_id_source_list
        )

        bg_colors = (
            torch.tensor(1.0, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        )  # [N, 3]

        # assert src_use_heads.sum() == 1  # only train when head exists

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs,  # [N1, 3, H, W]
            "render_bg_colors": bg_colors,  # [N, 3]
            "sample_frame_idx": sample_frame_idx,
        }
        return ret

    def get_motion_sequence(self, idx):

        uid_info = self.uids[idx]
        uid = uid_info["id"]

        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        valid_frame_items = list(tar_item.valid_frame_items)
        valid_frame_items = sorted(valid_frame_items)

        smplx_params_list = []

        for valid_frame in valid_frame_items:

            outs = tar_item.getattr(valid_frame, "smplx_params")

            smplx_params_list.append(outs["smplx_params"])

        return smplx_params_list, valid_frame_items

    def __getitem__(self, idx, view_id=16):
        return self.inner_get_item(idx, view_id)

    def get_src_view_img(self, tar_item, uid, all_frames, frame_id_source_list):

        source_rgbs = []

        # load src img
        for frame_id in frame_id_source_list:

            query_values = tar_item[frame_id]
            img = query_values["imgs_png"]
            mask = query_values["samurai_seg"]

            rgb = self.load_src_rgb_image_with_aug_bg(
                img,
                mask=mask,
                bg_color=1.0,
                pad_ratio=0.1,
            )

            source_rgbs.append(rgb)

        return source_rgbs

    def load_src_rgb_image_with_aug_bg(
        self,
        rgb,
        mask,
        bg_color,
        pad_ratio,
        head_bbox=None,
    ):

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

        rgb = self.center_crop_according_to_mask(rgb, mask)

        return torch.from_numpy(rgb).float().permute(2, 0, 1).clamp(0, 1)

    def center_crop_according_to_mask(self, img, mask):
        """
        img: [H, W, 3]
        mask: [H, W]
        """
        # Validate dimensions
        if img.shape[:2] != mask.shape:
            raise ValueError("The dimensions of img and mask must match.")

        # Find non-zero mask bounding box
        y_indices, x_indices = np.where(mask > 0)
        if y_indices.size == 0 or x_indices.size == 0:
            raise ValueError("The mask has no positive areas to crop.")

        min_y, max_y = np.min(y_indices), np.max(y_indices)
        min_x, max_x = np.min(x_indices), np.max(x_indices)

        # Calculate padding
        height, width = img.shape[:2]
        pad_y = int(0.05 * (max_y - min_y))
        pad_x = int(0.05 * (max_x - min_x))

        # Apply padding while checking bounds
        min_y = max(min_y - pad_y, 0)
        max_y = min(max_y + pad_y, height)
        min_x = max(min_x - pad_x, 0)
        max_x = min(max_x + pad_x, width)

        # Crop the image
        cropped_img = img[min_y:max_y, min_x:max_x]

        return cropped_img


class InTheWildDataset(VideoHumanA4ODatasetEval):
    """Src Image Inputs"""

    @no_proxy
    def inner_get_item(self, idx, view_id=16):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """

        uid_info = self.uids[idx]

        uid = uid_info["id"]

        sample_frame_idx = uid_info[f"{view_id}_view"]
        sample_frame_idx = [
            int(sample_frame.replace(".png", "").replace(".jpg", ""))
            for sample_frame in sample_frame_idx
        ]
        frame_id_source_list = sample_frame_idx

        root_dirs = os.path.join(self.root_dirs, uid)
        img_dirs = os.path.join(root_dirs, "crop_sr_src_imgs")

        source_rgb_list = []

        for source_id in frame_id_source_list:
            source_img_path = os.path.join(img_dirs, f"{source_id:05d}.png")
            rgb = self.load_src_rgb_image(source_img_path)

            source_rgb_list.append(rgb)

        source_rgbs = torch.cat(source_rgb_list, dim=0)

        bg_colors = (
            torch.tensor(1.0, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        )  # [N, 3]

        # assert src_use_heads.sum() == 1  # only train when head exists

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs,  # [N1, 3, H, W]
            "render_bg_colors": bg_colors,  # [N, 3]
            "sample_frame_idx": sample_frame_idx,
        }
        return ret

    def load_src_rgb_image(self, rgb, source_image_res=384, crop_img_ratio=1.0):
        target_ratio = self.aspect_standard
        target_h = source_image_res * target_ratio
        target_w = source_image_res

        img_np = np.asarray(Image.open(rgb).convert("RGB"))

        if crop_img_ratio < 1.0:
            height, width = img_np.shape[:2]
            max_edge = max(width, height)
            if max_edge > 2000:
                scale_ratio = 2 * target_h / max_edge

                new_width = int(width * scale_ratio)
                new_height = int(height * scale_ratio)

                img_np = cv2.resize(
                    img_np, (new_width, new_height), interpolation=cv2.INTER_AREA
                )

            height, width = img_np.shape[:2]
            crop_height = int(height * crop_img_ratio)
            crop_width = int(crop_height / target_ratio)

            crop_left = 0
            if crop_width < width:
                crop_left = (width - crop_width) // 2

            img_np = img_np[:crop_height, crop_left : crop_left + crop_width]

        original_height, original_width = img_np.shape[:2]
        original_ratio = original_width / original_height

        target_ratio = 1.0 / target_ratio

        if original_ratio > target_ratio:
            # Current width is too large, pad height
            new_width = original_width
            new_height = int(original_width / target_ratio)
            pad_height = new_height - original_height
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_width = (0, 0)
            pad_height = (pad_top, pad_bottom)
        else:
            # Current height is too large, pad width
            new_height = original_height
            new_width = int(original_height * target_ratio)
            pad_width = new_width - original_width
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            pad_height = (0, 0)
            pad_width = (pad_left, pad_right)

        # Pad image to reach the new dimensions
        padded_img_np = np.pad(
            img_np,
            (pad_height, pad_width, (0, 0)),
            mode="constant",
            constant_values=255,
        )
        resize_img = cv2.resize(
            padded_img_np, (int(target_w), int(target_h)), interpolation=cv2.INTER_AREA
        )

        rgb = (
            torch.from_numpy(resize_img / 255.0)
            .float()
            .permute(2, 0, 1)
            .clamp(0, 1)
            .unsqueeze(0)
        )

        return rgb

    def get_motion_sequence(self, idx):

        uid_info = self.uids[idx]
        uid = uid_info["id"]

        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        valid_frame_items = list(tar_item.valid_frame_items)
        valid_frame_items = sorted(valid_frame_items)

        smplx_params_list = []

        for valid_frame in valid_frame_items:

            smplx_params = tar_item.getattr(valid_frame, "smplx_params")["smplx_params"]
            try:
                flame_params = tar_item.getattr(valid_frame, "flame_params")[
                    "flame_params"
                ]

                smplx_params["expr"] = torch.FloatTensor(flame_params["expcode"])

                # replace with flame's jaw_pose
                smplx_params["jaw_pose"] = torch.FloatTensor(
                    flame_params["posecode"][3:]
                )
                smplx_params["leye_pose"] = torch.FloatTensor(
                    flame_params["eyecode"][:3]
                )
                smplx_params["reye_pose"] = torch.FloatTensor(
                    flame_params["eyecode"][3:]
                )
            except:
                smplx_params["expr"] = torch.FloatTensor([0.0] * 100)

            smplx_params_list.append(smplx_params)

        return smplx_params_list, valid_frame_items


class HeursiticInTheWildDataset(InTheWildDataset):

    @no_proxy
    def inner_get_item(self, idx, view_id=16):
        """
        Loaded contents:
            rgbs: [M, 3, H, W]
            poses: [M, 3, 4], [R|t]
            intrinsics: [3, 2], [[fx, fy], [cx, cy], [weight, height]]
        """

        uid_info = self.uids[idx]

        uid = uid_info["id"]

        # sample_frame_idx = uid_info[f"{view_id}_view"]
        sample_frame_idx = uid_info[f"16_view"][:: 16 // view_id]

        half_sample_frame_inputs = uid_info[f"4_view"]
        sample_frame_idx = [
            int(sample_frame.replace(".png", "").replace(".jpg", ""))
            for sample_frame in sample_frame_idx
        ]
        half_sample_frame_inputs = [
            int(sample_frame.replace(".png", "").replace(".jpg", ""))
            for sample_frame in half_sample_frame_inputs
        ]

        frame_id_source_list = sample_frame_idx

        root_dirs = os.path.join(self.root_dirs, uid)
        img_dirs = os.path.join(root_dirs, "crop_sr_src_imgs")

        source_rgb_list = []

        for source_id in frame_id_source_list:

            source_img_path = os.path.join(img_dirs, f"{source_id:05d}.png")
            # if view_id > 4 and source_id in half_sample_frame_inputs:
            #     # crop 60% input img
            #     rgb = self.load_src_rgb_image(source_img_path, crop_img_ratio = 0.6)
            # else:
            #     rgb = self.load_src_rgb_image(source_img_path)
            rgb = self.load_src_rgb_image(source_img_path)
            source_rgb_list.append(rgb)

        source_rgbs = torch.cat(source_rgb_list, dim=0)

        bg_colors = (
            torch.tensor(1.0, dtype=torch.float32).unsqueeze(-1).repeat(1, 3)
        )  # [N, 3]

        # assert src_use_heads.sum() == 1  # only train when head exists

        ret = {
            "uid": uid,
            "source_rgbs": source_rgbs,  # [N1, 3, H, W]
            "render_bg_colors": bg_colors,  # [N, 3]
            "sample_frame_idx": sample_frame_idx,
        }
        return ret

    def get_betas(self, idx):
        uid_info = self.uids[idx]
        uid = uid_info["id"]

        tar_uid = os.path.join(self.root_dirs, uid) + ".tar"
        tar_item = TarFrameDataset(tar_uid)

        valid_frame_items = list(tar_item.valid_frame_items)
        valid_frame = sorted(valid_frame_items)[0]

        smplx_params = tar_item.getattr(valid_frame, "smplx_params")["smplx_params"]

        return torch.FloatTensor(smplx_params["betas"]).unsqueeze(0).float()
