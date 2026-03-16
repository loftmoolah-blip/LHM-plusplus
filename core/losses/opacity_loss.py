# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 16:30:44
# @Function      : Opacity Loss

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["Opacity_Loss", "Heuristic_Opacity_Loss"]


def get_body_infos(mesh_infos):
    """Get indices for different body parts."""
    return {
        "head": torch.where(mesh_infos["is_face"])[0],
        "body": torch.where(~mesh_infos["is_face"])[0],
        "lower_body": torch.where(mesh_infos["is_lower_body"])[0],
        "upper_body": torch.where(mesh_infos["is_upper_body"])[0],
        "hands": torch.cat(
            [
                torch.where(mesh_infos["is_rhand"])[0],
                torch.where(mesh_infos["is_lhand"])[0],
            ]
        ),
        "constrain_body": torch.where(mesh_infos["is_constrain_body"])[0],
    }


class Opacity_Loss(nn.Module):
    """Opacity regularization loss."""

    def forward(self, opacity, epsilon=0.05, **params):
        """Empirically, the minimum opacity is 0.05"""

        champed_opacity = torch.clamp(opacity, min=epsilon)
        return -torch.log(champed_opacity).mean()


class Heuristic_Opacity_Loss(nn.Module):
    """Heuristic Opacity Loss with per-body-part weighting."""

    def __init__(self, group_dict, group_body_mapping):
        super().__init__()

        self.group_dict = group_dict  # register weights for different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _opacity_loss, mesh_meta=None):
        _loss = 0.0

        group_body_mapping = get_body_infos(mesh_meta)

        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = group_body_mapping[key]
            _loss += key_weights * _opacity_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, opacity, epsilon=0.05, **params):
        """
        Compute heuristic opacity loss with per-body-part weighting.

        Args:
            opacity: Gaussian opacity parameters.
            epsilon: Minimum opacity threshold (empirically 0.05).
            **params: Additional parameters including 'mesh_meta'.

        Returns:
            Weighted opacity loss.
        """
        champed_opacity = torch.clamp(opacity, min=epsilon)
        _opacity_loss = -torch.log(champed_opacity)

        return self._heurisitic_loss(_opacity_loss, params["mesh_meta"])
