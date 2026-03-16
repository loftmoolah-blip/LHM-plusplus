# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-05-09 19:23:25
# @Function      : deform head

import math
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.models.rendering.utils.utils import trunc_exp
from core.outputs.output import GaussianAppOutput

from .layers import Mlp
from .layers.block import Block


def inverse_sigmoid(x):

    if isinstance(x, float):
        x = torch.tensor(x).float()

    return torch.log(x / (1 - x))


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Modulate the input tensor using scaling and shifting parameters."""
    return x * (1 + scale) + shift


class BaseDeformHead(nn.Module):
    """Base class for deformation heads with shared functionality."""

    def setup_functions(self):
        self.scaling_activation = trunc_exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = F.normalize

    def _init_common(
        self,
        dim_in: int,
        img_dim_in: int,
        trunk_depth: int,
        attr_dict: dict,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        use_pose_embed: bool = False,
        **kwargs
    ):
        self.setup_functions()
        self.trunk_depth = trunk_depth
        self.attr_dict = attr_dict
        self.order = list(attr_dict.keys())
        self.target_dim = sum(v if v is not None else 0 for v in attr_dict.values())

        # Normalizations
        self.img_norm = nn.LayerNorm(img_dim_in)
        self.point_norm = nn.LayerNorm(dim_in)
        self.use_pose_embed = use_pose_embed

        # Modulation components
        if use_pose_embed:
            self.embed_pose = nn.Linear(img_dim_in, dim_in)
            self.poseLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(dim_in, 3 * dim_in, bias=True)
            )
        else:
            self.poseLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(img_dim_in, 3 * dim_in, bias=True)
            )

        self.adaln_norm = nn.LayerNorm(dim_in, elementwise_affine=False, eps=1e-6)

        # Transformer blocks if depth > 0
        if trunk_depth > 0:
            self.point_trunks = nn.Sequential(
                *[
                    Block(
                        dim=dim_in,
                        num_heads=num_heads,
                        mlp_ratio=mlp_ratio,
                        init_values=init_values,
                    )
                    for _ in range(trunk_depth)
                ]
            )

        # Regression head
        self.regress_head = Mlp(
            dim_in, dim_in // 2, out_features=self.target_dim, drop=0.0
        )
        nn.init.constant_(self.regress_head.fc2.weight, 0)
        nn.init.constant_(self.regress_head.fc2.bias, 0)

    def _process_pose_tokens(self, img_tokens, use_mean=True):
        """Extract and normalize pose tokens."""
        B, S, P, C = img_tokens.shape
        assert B == 1

        if use_mean:
            pose_tokens = img_tokens[:, :, 0].mean(dim=1, keepdim=True).view(1, C)
        else:
            pose_tokens = img_tokens[:, 0, 0].view(1, C)

        return self.img_norm(pose_tokens)

    def _trunk_forward(self, query_points_tokens, pose_tokens):
        """Shared forward pass through the modulation trunk."""
        pose_tokens = pose_tokens.unsqueeze(1)
        B, S, C = pose_tokens.shape

        # Generate modulation parameters

        if self.use_pose_embed:
            pose_tokens = self.embed_pose(pose_tokens)
        shift_msa, scale_msa, gate_msa = self.poseLN_modulation(pose_tokens).chunk(
            3, dim=-1
        )

        # Apply modulation
        modulated_tokens = gate_msa * modulate(
            self.adaln_norm(query_points_tokens), shift_msa, scale_msa
        )
        modulated_tokens = modulated_tokens + query_points_tokens

        # Process through transformer blocks if they exist, Not Only MLP regress
        if hasattr(self, "point_trunks"):
            modulated_tokens = self.point_trunks(modulated_tokens)

        return self.regress_head(self.point_norm(modulated_tokens))


class DeformHead(BaseDeformHead):
    """Full deformation head predicting all attributes."""

    def __init__(
        self,
        dim_in: int = 896,
        img_dim_in: int = 1024,
        trunk_depth: int = 2,
        sh_degree=None,
        fix_opacity=False,
        fix_rotation=False,
        use_rgb=True,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        **kwargs
    ):
        super().__init__()

        attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3 if not use_rgb else 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": 1 if not fix_opacity else None,
            "rotation": 4 if not fix_rotation else None,
        }

        self._init_common(
            dim_in,
            img_dim_in,
            trunk_depth,
            attr_dict,
            num_heads,
            mlp_ratio,
            init_values,
            use_pose_embed=False,
            **kwargs
        )

    def forward(self, query_points_tokens, img_tokens, **kwargs):
        pose_tokens = self._process_pose_tokens(img_tokens, use_mean=True)
        offset = self._trunk_forward(query_points_tokens, pose_tokens)

        return {
            "shs": offset[..., :3],
            "scaling": offset[..., 3:6],
            "xyz": offset[..., 6:9],
            "opacity": offset[..., 9:10],
            "rotation": offset[..., 10:],
        }


class DeformHeadMLP(BaseDeformHead):
    """Lightweight MLP-only deformation head."""

    def __init__(
        self,
        dim_in: int = 896,
        img_dim_in: int = 1024,
        trunk_depth: int = 0,  # MLP-only version
        sh_degree=None,
        fix_opacity=False,
        fix_rotation=False,
        use_rgb=True,
        **kwargs
    ):
        super().__init__()

        attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3 if not use_rgb else 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": 1 if not fix_opacity else None,
            "rotation": 4 if not fix_rotation else None,
        }

        self._init_common(
            dim_in, img_dim_in, trunk_depth, attr_dict, use_pose_embed=False, **kwargs
        )

    def forward(self, query_points_tokens, img_tokens, **kwargs):
        pose_tokens = self._process_pose_tokens(img_tokens, use_mean=True)
        offset = self._trunk_forward(query_points_tokens, pose_tokens)

        return {
            "shs": offset[..., :3],
            "scaling": offset[..., 3:6],
            "xyz": offset[..., 6:9],
            "opacity": offset[..., 9:10],
            "rotation": offset[..., 10:],
        }


class DeformXYZHead(BaseDeformHead):
    """XYZ-only deformation head with tanh scaling."""

    def __init__(
        self,
        dim_in: int = 896,
        img_dim_in: int = 1024,
        trunk_depth: int = 2,
        num_heads: int = 16,
        mlp_ratio: int = 4,
        init_values: float = 0.01,
        deform_scale_size: float = 0.01,
        **kwargs
    ):
        super().__init__()

        self._init_common(
            dim_in,
            img_dim_in,
            trunk_depth,
            {"xyz": 3},
            num_heads,
            mlp_ratio,
            init_values,
            use_pose_embed=False,
            **kwargs
        )
        self.deform_scale_size = deform_scale_size

    def forward(self, query_points_tokens, img_tokens, **kwargs):
        pose_tokens = self._process_pose_tokens(img_tokens, use_mean=True)
        offset = self._trunk_forward(query_points_tokens, pose_tokens)
        xyz = self.deform_scale_size * torch.tanh(offset)

        return {"xyz": xyz.squeeze(0)}


class DeformXYZMLP(BaseDeformHead):
    """XYZ-only MLP deformation head with tanh scaling."""

    def __init__(
        self,
        dim_in: int = 896,
        img_dim_in: int = 1024,
        trunk_depth: int = 0,  # MLP-only version
        deform_scale_size: float = 0.01,
        **kwargs
    ):
        super().__init__()

        self._init_common(
            dim_in, img_dim_in, trunk_depth, {"xyz": 3}, use_pose_embed=False, **kwargs
        )
        self.deform_scale_size = deform_scale_size

    def forward(self, query_points_tokens, img_tokens, **kwargs):
        pose_tokens = self._process_pose_tokens(img_tokens, use_mean=True)
        offset = self._trunk_forward(query_points_tokens, pose_tokens)
        xyz = self.deform_scale_size * torch.tanh(offset)

        return {"xyz": xyz.squeeze(0)}
