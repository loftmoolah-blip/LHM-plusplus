# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Zhe Li
# @Function      : Copy from Animatable Gaussian

import os
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CanoBlendWeightVolume(nn.Module):
    def __init__(self, data_path):
        super().__init__()

        data = np.load(data_path)

        diff_weight_volume = data["diff_weight_volume"]
        diff_weight_volume = diff_weight_volume.transpose((3, 0, 1, 2))[None]
        # base_weight_volume = base_weight_volume.transpose((3, 2, 1, 0))[None]

        self.register_buffer(
            "diff_weight_volume", torch.from_numpy(diff_weight_volume).to(torch.float32)
        )
        self.res_x, self.res_y, self.res_z = self.diff_weight_volume.shape[2:]
        self.joint_num = self.diff_weight_volume.shape[1]

        ori_weight_volume = torch.from_numpy(
            data["ori_weight_volume"].transpose((3, 0, 1, 2))[None]
        ).to(torch.float32)
        self.register_buffer("ori_weight_volume", ori_weight_volume)

        if "sdf_volume" in data:
            smpl_sdf_volume = data["sdf_volume"]
            if len(smpl_sdf_volume.shape) == 3:
                smpl_sdf_volume = smpl_sdf_volume[..., None]
            smpl_sdf_volume = smpl_sdf_volume.transpose((3, 0, 1, 2))[None]
            self.register_buffer(
                "smpl_sdf_volume", torch.from_numpy(smpl_sdf_volume).to(torch.float32)
            )

        volume_bounds = torch.from_numpy(data["volume_bounds"]).to(torch.float32)
        center = torch.from_numpy(data["center"]).to(torch.float32)
        smpl_bounds = torch.from_numpy(data["smpl_bounds"]).to(torch.float32)

        self.register_buffer("volume_bounds", volume_bounds)
        self.register_buffer("center", center)
        self.register_buffer("smpl_bounds", smpl_bounds)

        volume_len = self.volume_bounds[1] - self.volume_bounds[0]
        self.voxel_size = volume_len / torch.tensor(
            [self.res_x - 1, self.res_y - 1, self.res_z - 1]
        ).to(volume_len)

    def forward(self, pts, requires_scale=True, volume_type="diff"):
        """
        :param pts: (B, N, 3), x y z
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 55)
        """
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (
                self.volume_bounds[1] - self.volume_bounds[0]
            )[None, None]

        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid[..., [2, 1, 0]]
        grid = grid[:, :, None, None]

        weight_volume = (
            self.diff_weight_volume if volume_type == "diff" else self.ori_weight_volume
        )

        base_w = F.grid_sample(
            weight_volume.expand(B, -1, -1, -1, -1),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        base_w = base_w[:, :, :, 0, 0].permute(0, 2, 1)
        return base_w

    def forward_weight_grad(self, pts, requires_scale=True):
        """
        :param pts: (B, N, 3)
        :param requires_scale: bool, scale pts to [0, 1]
        :return: (B, N, 24)
        """
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (
                self.volume_bounds[1] - self.volume_bounds[0]
            )[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        base_g = F.grid_sample(
            self.base_gradient_volume.view(
                self.joint_num * 3, self.res_x, self.res_y, self.res_z
            )[None].expand(B, -1, -1, -1, -1),
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        )
        base_g = base_g[:, :, :, 0, 0].permute(0, 2, 1).reshape(B, N, -1, 3)
        return base_g

    def forward_sdf(self, pts, requires_scale=True):
        if requires_scale:
            pts = (pts - self.volume_bounds[None, None, 0]) / (
                self.volume_bounds[1] - self.volume_bounds[0]
            )[None, None]
        B, N, _ = pts.shape
        grid = 2 * pts - 1
        grid = grid.reshape(-1, 3)[:, [2, 1, 0]]
        grid = grid[None, :, None, None]

        sdf = F.grid_sample(
            self.smpl_sdf_volume.expand(B, -1, -1, -1, -1),
            grid,
            padding_mode="border",
            align_corners=True,
        )
        sdf = sdf[:, :, :, 0, 0].permute(0, 2, 1)

        return sdf
