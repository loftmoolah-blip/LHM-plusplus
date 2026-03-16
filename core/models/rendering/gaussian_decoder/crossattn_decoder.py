# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-14 21:50:20
# @Function      : CrossAttn Decoder.

"""
Cross-Attention Decoder for Gaussian Splatting-based Neural Rendering.

This module implements a cross-attention decoder that leverages multi-modal
features for enhanced Gaussian splatting reconstruction. The decoder supports
various feature fusion strategies including DINOv2 and ResNet18 features
for improved visual quality and detail preservation in 3D reconstruction tasks.
"""

import math

import torch.nn as nn
import torch.nn.functional as F

from core.modules.cross_attn import CrossAttnBlock


class DecoderCrossAttn(nn.Module):
    """
    Cross-Attention Decoder for multi-modal Gaussian splatting reconstruction.

    This decoder leverages cross-attention mechanisms to fuse point cloud queries
    with contextual features from various modalities. It supports hierarchical
    decoding with coarse and fine levels, where fine-level decoding incorporates
    additional visual features from pre-trained encoders such as DINOv2 or ResNet18.

    The decoder architecture enables enhanced detail preservation and visual quality
    by allowing the model to attend to relevant visual features during the
    Gaussian parameter prediction process.

    Attributes:
        query_dim (int): Dimension of the query features (point cloud features).
        context_dim (int): Dimension of the context features (latent features).
        cross_attn (CrossAttnBlock): Primary cross-attention block for coarse decoding.
        cross_attn_color (CrossAttnBlock, optional): Secondary cross-attention block for fine decoding.
        encoder (nn.Module, optional): Visual encoder for extracting image features.
    """

    def __init__(
        self, query_dim, context_dim, num_heads, mlp=False, decode_with_extra_info=None
    ):
        """
        Initialize the Cross-Attention Decoder.

        Args:
            query_dim (int): Dimension of query features from point cloud.
            context_dim (int): Dimension of context features from latent representation.
            num_heads (int): Number of attention heads for multi-head attention.
            mlp (bool, optional): Whether to use MLP feedforward in cross-attention.
                Defaults to False.
            decode_with_extra_info (dict, optional): Configuration for fine-level decoding
                with additional visual features. Supported types:
                - "dinov2p14_feat": Use pre-computed DINOv2 features
                - "decoder_dinov2p14_feat": Use DINOv2 encoder for real-time feature extraction
                - "decoder_resnet18_feat": Use ResNet18 encoder for feature extraction

        Note:
            The decoder supports hierarchical decoding where coarse features are first
            processed through the primary cross-attention, and optionally refined using
            visual features through a secondary cross-attention mechanism.
        """
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.cross_attn = CrossAttnBlock(
            inner_dim=query_dim,
            cond_dim=context_dim,
            num_heads=num_heads,
            feedforward=mlp,
            eps=1e-5,
        )
        self.decode_with_extra_info = decode_with_extra_info
        if decode_with_extra_info is not None:
            if decode_with_extra_info["type"] == "dinov2p14_feat":
                context_dim = decode_with_extra_info["cond_dim"]
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=context_dim,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                from core.models.encoders.dinov2_wrapper import Dinov2Wrapper

                self.encoder = Dinov2Wrapper(
                    model_name="dinov2_vits14_reg", freeze=False, encoder_feat_dim=384
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=384,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )
            elif decode_with_extra_info["type"] == "decoder_resnet18_feat":
                from core.models.encoders.xunet_wrapper import XnetWrapper

                self.encoder = XnetWrapper(
                    model_name="resnet18", freeze=False, encoder_feat_dim=64
                )
                self.cross_attn_color = CrossAttnBlock(
                    inner_dim=query_dim,
                    cond_dim=64,
                    num_heads=num_heads,
                    feedforward=False,
                    eps=1e-5,
                )

    def resize_image(self, image, multiply):
        """
        Resize input image to ensure dimensions are divisible by the specified multiplier.

        This function is commonly used to ensure compatibility with encoder architectures
        that require specific input dimensions (e.g., ResNet18 requires input dimensions
        to be divisible by 32 for optimal feature extraction).

        Args:
            image (torch.Tensor): Input image tensor with shape (B, C, H, W).
            multiply (int): Multiplier to ensure dimensions are divisible by this value.

        Returns:
            torch.Tensor: Resized image tensor with shape (B, C, new_h, new_w),
                where new_h and new_w are the smallest dimensions >= H and W
                that are divisible by the multiplier.
        """
        B, _, H, W = image.shape
        new_h, new_w = (
            math.ceil(H / multiply) * multiply,
            math.ceil(W / multiply) * multiply,
        )
        image = F.interpolate(
            image, (new_h, new_w), align_corners=True, mode="bilinear"
        )
        return image

    def forward(self, pcl_query, pcl_latent, extra_info=None):
        """
        Forward pass through the cross-attention decoder.

        Performs hierarchical decoding with optional fine-level refinement using
        additional visual features. The process involves:
        1. Coarse decoding: Cross-attention between point cloud queries and latent features
        2. Fine decoding (optional): Additional cross-attention with visual features

        Args:
            pcl_query (torch.Tensor): Query features from point cloud with shape (N, query_dim).
            pcl_latent (torch.Tensor): Context features from latent representation with shape (M, context_dim).
            extra_info (dict, optional): Additional information for fine-level decoding.
                Contains:
                - "image": Input image tensor for feature extraction (when using encoder)
                - "image_feats": Pre-computed image features (when using pre-computed features)

        Returns:
            torch.Tensor or dict:
                - If no extra info: Returns coarse features tensor
                - If extra info provided: Returns dictionary with:
                    - "coarse": Coarse-level features
                    - "fine": Fine-level features enhanced with visual information

        Note:
            The fine-level decoding enhances the coarse features by incorporating
            visual information through cross-attention, enabling better detail
            preservation and visual quality in the final reconstruction.
        """
        out = self.cross_attn(pcl_query, pcl_latent)
        if self.decode_with_extra_info is not None:
            out_dict = {}
            out_dict["coarse"] = out
            if self.decode_with_extra_info["type"] == "dinov2p14_feat":
                out = self.cross_attn_color(out, extra_info["image_feats"])
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                img_feat = self.encoder(extra_info["image"])
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_resnet18_feat":
                image = extra_info["image"]
                image = self.resize_image(image, multiply=32)
                img_feat = self.encoder(image)
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
        return out
