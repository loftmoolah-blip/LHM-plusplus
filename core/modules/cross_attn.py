# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Cross-attention transformer block

import torch
import torch.nn as nn


class CrossAttnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """

    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(
        self,
        inner_dim: int,
        cond_dim: int,
        num_heads: int,
        eps: float = None,
        attn_drop: float = 0.0,
        attn_bias: bool = False,
        mlp_ratio: float = 4.0,
        mlp_drop: float = 0.0,
        feedforward=False,
    ):
        super().__init__()
        # TODO check already apply normalization
        # self.norm_q = nn.LayerNorm(inner_dim, eps=eps)
        # self.norm_k = nn.LayerNorm(cond_dim, eps=eps)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim,
            num_heads=num_heads,
            kdim=cond_dim,
            vdim=cond_dim,
            dropout=attn_drop,
            bias=attn_bias,
            batch_first=True,
        )

        self.mlp = None
        if feedforward:
            self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=inner_dim,
                num_heads=num_heads,
                dropout=attn_drop,
                bias=attn_bias,
                batch_first=True,
            )
            self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
            self.mlp = nn.Sequential(
                nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(mlp_drop),
                nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
                nn.Dropout(mlp_drop),
            )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = self.cross_attn(
            self.norm_q(x), self.norm_k(cond), cond, need_weights=False
        )[0]
        if self.mlp is not None:
            before_sa = self.norm2(x)
            x = (
                x
                + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
            )
            x = x + self.mlp(self.norm3(x))
        return x
