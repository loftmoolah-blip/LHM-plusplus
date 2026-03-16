"""A class to define GSNet Output
"""

# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : GaussianAppOutput dataclass

from dataclasses import dataclass

import numpy as np
from torch import Tensor

from .base import BaseOutput


@dataclass
class GaussianAppOutput(BaseOutput):
    """
    Output of the Gaussian Appearance output.

    Attributes:

    """

    offset_xyz: Tensor
    opacity: Tensor
    rotation: Tensor
    scaling: Tensor
    shs: Tensor
    use_rgb: bool
