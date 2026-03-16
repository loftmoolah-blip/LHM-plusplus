# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-17 19:07:39
# @Function      : CrossAttn's Transformer block


from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.utils import is_torch_version

from core.modules.attention_modules import CrossAttnBlock


class CrossAttnDecoderBlock(nn.Module):
    r"""
    Given Query Features, we use Crossattention to query the Context Features.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        eps: float = None,
        trunk_depth: int = 1,
        gradient_checkpointing: bool = False,
    ):
        super(CrossAttnDecoderBlock, self).__init__()

        self.intermediate_layer_idx_info = {
            "debug": [0],  # debug-query version
        }

        self.attnblocks = nn.ModuleList(
            [
                CrossAttnBlock(
                    hidden_size=query_dim,
                    context_dim=context_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    eps=eps,
                )
                for _ in range(trunk_depth)
            ]
        )

        self.gradient_checkpointing = gradient_checkpointing  # use checkpoint tricks

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states_list,
        temb: torch.FloatTensor = None,
        patch_start_idx: int = 5,
    ):
        """
        Forward pass of the transformer_dit model.
        Args:
            hidden_states (torch.FloatTensor): Input hidden states. Query Points features
            encoder_hidden_states (torch.FloatTensor): Encoder hidden states. Context features
            temb:(torch.FloatTensor, optional): Optional tensor for embedding. Defaults to None.
        Returns:
            List[torch.FloatTensor, torch.FloatTensor]: List containing the updated hidden states and encoder hidden states.
        """

        output_hidden_states_list = []

        for layer_idx, intermediate_idx in enumerate(
            self.intermediate_layer_idx_info["debug"]
        ):
            layer = self.attnblocks[layer_idx]
            tokens = encoder_hidden_states_list[intermediate_idx][
                :, :, patch_start_idx:
            ].contiguous()
            B, S, P, C = tokens.shape

            mvtokens = tokens.view(B, S * P, C)

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = (
                    {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                )

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    mvtokens,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    mvtokens,
                )

            hidden_states = layer(hidden_states, mvtokens)

            output_hidden_states_list.append(hidden_states)

        return output_hidden_states_list[-1]
