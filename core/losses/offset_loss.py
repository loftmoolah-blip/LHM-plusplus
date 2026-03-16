# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 16:30:44
# @Function      : ACAP Loss
import torch
import torch.nn as nn

__all__ = ["ACAP_Loss", "Heuristic_ACAP_Loss"]


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


class ACAP_Loss(nn.Module):
    """As Close As Possible (ACAP) Loss for offset regularization."""

    def forward(self, offset, d=0.05625, **params):
        """Empirically, where d is the thresold of distance points leave from 1.8/32 = 0.0562."""

        offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return offset_loss.mean()


class Heuristic_ACAP_Loss(nn.Module):
    """Heuristic ACAP Loss with per-body-part weighting."""

    def __init__(self, group_dict, group_body_mapping):
        super().__init__()

        self.group_dict = group_dict  # register weights for different body parts
        self.group_body_mapping = group_body_mapping  # mapping of body parts to group

    def _heurisitic_loss(self, _offset_loss, mesh_meta=None):
        _loss = 0.0

        group_body_mapping = get_body_infos(mesh_meta)

        for key in self.group_dict.keys():
            key_weights = self.group_dict[key]
            group_mapping_idx = group_body_mapping[key]
            _loss += key_weights * _offset_loss[:, group_mapping_idx].mean()

        return _loss

    def forward(self, offset, d=0.05625, **params):
        """
        Compute heuristic ACAP loss with per-body-part weighting.

        Args:
            offset: Gaussian offset parameters.
            d: Distance threshold (empirically 1.8/32 = 0.0562).
            **params: Additional parameters including 'mesh_meta'.

        Returns:
            Weighted ACAP loss.
        """
        _offset_loss = torch.clamp(offset.norm(p=2, dim=-1), min=d) - d

        return self._heurisitic_loss(_offset_loss, params["mesh_meta"])
