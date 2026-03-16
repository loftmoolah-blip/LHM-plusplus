# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 16:30:44
# @Function      : ASAP Loss

import torch
import torch.nn as nn

__all__ = ["ASAP_Loss", "Heuristic_ASAP_Loss"]


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


class ASAP_Loss(nn.Module):
    """As Spherical As Possible (ASAP) Loss for Gaussian scaling regularization."""

    def forward(self, scaling, r=5, **params):
        """where r is the radius of the ball between max-axis and min-axis."""

        _scale = scaling

        _scale_min = torch.min(_scale, dim=-1)[0]
        _scale_max = torch.max(_scale, dim=-1)[0]

        scale_ratio = _scale_max / (_scale_min + 1e-6)

        _ball_loss = (torch.clamp(scale_ratio, min=r) - r).mean()

        return _ball_loss


class Heuristic_ASAP_Loss(nn.Module):
    """Heuristic ASAP Loss with per-body-part weighting."""

    def __init__(self, group_dict, group_body_mapping):
        super().__init__()

        self.group_dict = group_dict  # register weights for different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _ball_loss, mesh_meta=None):
        _loss = 0.0

        group_body_mapping = get_body_infos(mesh_meta)

        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = group_body_mapping[key]
            _loss += key_weights * _ball_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, scaling, r=5, **params):
        """
        Compute heuristic ASAP loss with per-body-part weighting.

        Args:
            scaling: Gaussian scaling parameters.
            r: Radius threshold for the ball constraint.
            **params: Additional parameters including 'mesh_meta'.

        Returns:
            Weighted ASAP loss.
        """
        _scale = scaling

        _scale_min = torch.min(_scale, dim=-1)[0]
        _scale_max = torch.max(_scale, dim=-1)[0]

        scale_ratio = _scale_max / (_scale_min + 1e-6)

        _ball_loss = torch.clamp(scale_ratio, min=r) - r

        return self._heurisitic_loss(_ball_loss, params["mesh_meta"])
