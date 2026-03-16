# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Conv decoder head for dense prediction

"""
Convolutional head module for dense prediction tasks.

This module provides a decoder head that upsamples low-resolution feature maps
to high-resolution outputs (e.g., depth, normal maps). It originates from MoGe
(https://github.com/microsoft/moge) and supports UV coordinate conditioning
for aspect-ratio awareness.

Classes:
    ResidualConvBlock: A residual block with group normalization and convolution.
    ConvHead: Main decoder head with progressive upsampling and multi-output support.
"""

from typing import List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_view_plane_uv(
    width: int,
    height: int,
    aspect_ratio: Optional[float] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Generates normalized UV coordinates for a view plane grid.

    Creates a 2D grid of UV coordinates normalized such that the left-top corner
    is (-width/diagonal, -height/diagonal) and the right-bottom corner is
    (width/diagonal, height/diagonal). This normalization provides consistent
    coordinate ranges across different aspect ratios.

    Args:
        width (int): Grid width (number of columns).
        height (int): Grid height (number of rows).
        aspect_ratio (float, optional): Width/height ratio. Defaults to width/height
            if not specified.
        dtype (torch.dtype, optional): Output tensor dtype.
        device (torch.device, optional): Output tensor device.

    Returns:
        torch.Tensor: UV coordinates of shape (height, width, 2), where the last
            dimension contains (u, v) coordinates.
    """
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
    span_y = 1 / (1 + aspect_ratio**2) ** 0.5

    u = torch.linspace(
        -span_x * (width - 1) / width,
        span_x * (width - 1) / width,
        width,
        dtype=dtype,
        device=device,
    )
    v = torch.linspace(
        -span_y * (height - 1) / height,
        span_y * (height - 1) / height,
        height,
        dtype=dtype,
        device=device,
    )
    u, v = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([u, v], dim=-1)
    return uv


class ResidualConvBlock(nn.Module):
    """A residual convolution block with GroupNorm and configurable activation.

    Applies two convolutions with normalization and activation, adding a skip
    connection. When input and output channel dimensions differ, a 1x1 conv
    projects the shortcut to match dimensions.

    Attributes:
        layers (nn.Sequential): Main convolution path (Norm -> Act -> Conv -> Norm -> Act -> Conv).
        skip_connection (nn.Module): Projection for residual (1x1 conv or Identity).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        hidden_channels: Optional[int] = None,
        padding_mode: str = "replicate",
        activation: Literal["relu", "leaky_relu", "silu", "elu"] = "relu",
        norm: Literal["group_norm", "layer_norm"] = "group_norm",
    ):
        """Initializes the ResidualConvBlock.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int, optional): Number of output channels. Defaults to
                in_channels.
            hidden_channels (int, optional): Hidden layer channels. Defaults to
                in_channels.
            padding_mode (str): Padding mode for convolutions. Defaults to "replicate".
            activation (str): Activation function. One of "relu", "leaky_relu",
                "silu", "elu". Defaults to "relu".
            norm (str): Normalization type. One of "group_norm", "layer_norm".
                Defaults to "group_norm".
        """
        super(ResidualConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation == "relu":
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == "leaky_relu":
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation == "silu":
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == "elu":
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        num_groups = hidden_channels // 32 if norm == "group_norm" else 1
        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(
                in_channels,
                hidden_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
            nn.GroupNorm(num_groups, hidden_channels),
            activation_cls(),
            nn.Conv2d(
                hidden_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                padding_mode=padding_mode,
            ),
        )

        self.skip_connection = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the residual block to the input.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (N, C_out, H, W).
        """
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class ConvHead(nn.Module):
    """Convolutional decoder head for dense prediction with progressive upsampling.

    Decodes patch-level features into high-resolution outputs (e.g., depth, normal).
    Uses transposed convolutions for 2x upsampling stages, optionally concatenates
    normalized UV coordinates for aspect-ratio awareness, and supports multiple
    output heads.

    Shape:
        - Input hidden_states: (B, N_patch, D) or (B, D, H_patch, W_patch) after project.
        - Output: List of tensors, each (B, C_out_i, H_img, W_img).

    Attributes:
        upsample_blocks (nn.ModuleList): Blocks for (H_patch, W_patch) -> (H_patch*8, W_patch*8).
        output_block (nn.ModuleList): Per-output-head decoder blocks.
        using_uv (bool): Whether to concatenate UV coordinates.
    """

    def __init__(
        self,
        num_features: int,
        dim_in: int,
        dim_out: List[int],
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal["group_norm", "layer_norm"] = "group_norm",
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1,
        projects: Optional[nn.Module] = None,
        using_uv: bool = True,
    ):
        """Initializes the ConvHead.

        Args:
            num_features (int): Unused; kept for API compatibility.
            dim_in (int): Unused; kept for API compatibility.
            dim_out (List[int]): Output channel dimension for each head.
            dim_proj (int): Projected feature dimension (first upsample block input).
                Defaults to 512.
            dim_upsample (List[int]): Channel dimensions for each upsample stage.
                Defaults to [256, 128, 128].
            dim_times_res_block_hidden (int): Hidden channels = out_ch * this factor.
                Defaults to 1.
            num_res_blocks (int): Number of residual blocks per upsample stage.
                Defaults to 1.
            res_block_norm (str): Normalization in residual blocks. Defaults to
                "group_norm".
            last_res_blocks (int): Residual blocks in output head. Defaults to 0.
            last_conv_channels (int): Channels before final 1x1/3x3 output conv.
                Defaults to 32.
            last_conv_size (int): Kernel size of final output convolution.
                Defaults to 1.
            projects (nn.Module, optional): Projection module from hidden_states to
                spatial feature maps. If None, hidden_states are assumed to already
                be spatial.
            using_uv (bool): Whether to concatenate UV coordinates. Defaults to True.
        """
        super().__init__()

        self.using_uv = using_uv
        self.projects = projects

        self.upsample_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    self._make_upsampler(in_ch + 2 if using_uv else in_ch, out_ch),
                    *(
                        ResidualConvBlock(
                            out_ch,
                            out_ch,
                            dim_times_res_block_hidden * out_ch,
                            activation="relu",
                            norm=res_block_norm,
                        )
                        for _ in range(num_res_blocks)
                    ),
                )
                for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
            ]
        )

        self.output_block = nn.ModuleList(
            [
                self._make_output_block(
                    dim_upsample[-1] + 2 if using_uv else dim_upsample[-1],
                    dim_out_,
                    dim_times_res_block_hidden,
                    last_res_blocks,
                    last_conv_channels,
                    last_conv_size,
                    res_block_norm,
                )
                for dim_out_ in dim_out
            ]
        )

    def _make_upsampler(self, in_channels: int, out_channels: int) -> nn.Sequential:
        """Builds a 2x upsampler: ConvTranspose2d(2x2) + Conv2d(3x3).

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Sequential: Upsampler module.
        """
        upsampler = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
        )
        # Initialize transposed conv to nearest-neighbor-like upsampling
        upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
        return upsampler

    def _make_output_block(
        self,
        dim_in: int,
        dim_out: int,
        dim_times_res_block_hidden: int,
        last_res_blocks: int,
        last_conv_channels: int,
        last_conv_size: int,
        res_block_norm: Literal["group_norm", "layer_norm"],
    ) -> nn.Sequential:
        """Builds an output decoder block (conv -> res blocks -> final conv).

        Args:
            dim_in (int): Input channels.
            dim_out (int): Output channels.
            dim_times_res_block_hidden (int): Hidden channel factor for res blocks.
            last_res_blocks (int): Number of residual blocks.
            last_conv_channels (int): Intermediate channels.
            last_conv_size (int): Final conv kernel size.
            res_block_norm (str): Normalization type.

        Returns:
            nn.Sequential: Output block module.
        """
        return nn.Sequential(
            nn.Conv2d(
                dim_in,
                last_conv_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                padding_mode="replicate",
            ),
            *(
                ResidualConvBlock(
                    last_conv_channels,
                    last_conv_channels,
                    dim_times_res_block_hidden * last_conv_channels,
                    activation="relu",
                    norm=res_block_norm,
                )
                for _ in range(last_res_blocks)
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                last_conv_channels,
                dim_out,
                kernel_size=last_conv_size,
                stride=1,
                padding=last_conv_size // 2,
                padding_mode="replicate",
            ),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        patch_h: Optional[int] = None,
        patch_w: Optional[int] = None,
    ):
        """Decodes hidden states to high-resolution dense outputs.

        Upsamples from patch resolution (H_img/7, W_img/7) to image resolution
        (H_img, W_img), optionally using UV coordinates for aspect-ratio awareness.

        Args:
            hidden_states (torch.Tensor): Patch-level features (B, N, D) or similar.
            image (torch.Tensor, optional): Reference image (B, C, H, W) to infer
                H, W. If None, patch_h and patch_w must be provided.
            patch_h (int, optional): Patch grid height (img_h // 7).
            patch_w (int, optional): Patch grid width (img_w // 7).

        Returns:
            List[torch.Tensor] or torch.Tensor: Per-head outputs, each of shape
                (B, C_out_i, H_img, W_img).
        """
        if image is not None:
            img_h, img_w = image.shape[-2:]
            patch_h, patch_w = img_h // 7, img_w // 7
        else:
            assert patch_h is not None and patch_w is not None
            img_h = patch_h * 7
            img_w = patch_w * 7

        # Project patch tokens to spatial feature map if needed
        if self.projects is not None:
            x = (
                self.projects(hidden_states)
                .permute(0, 2, 1)
                .unflatten(2, (patch_h, patch_w))
                .contiguous()
            )
        else:
            x = hidden_states

        # Upsample: (patch_h, patch_w) -> *2 -> *4 -> *8
        for block in self.upsample_blocks:
            if self.using_uv:
                uv = normalized_view_plane_uv(
                    width=x.shape[-1],
                    height=x.shape[-2],
                    aspect_ratio=img_w / img_h,
                    dtype=x.dtype,
                    device=x.device,
                )
                uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
                x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        # Final bilinear interpolation to image resolution
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)

        if self.using_uv:
            uv = normalized_view_plane_uv(
                width=x.shape[-1],
                height=x.shape[-2],
                aspect_ratio=img_w / img_h,
                dtype=x.dtype,
                device=x.device,
            )
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)

        output = [
            torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            for block in self.output_block
        ]

        return output
