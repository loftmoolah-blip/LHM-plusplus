# -*- coding: utf-8 -*-
# Copyright (c) 2023-2024, Jiaxiang Tang
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Torch/numpy math operations (dot, cross, etc.)

from typing import Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


# torch / numpy math utils
def dot(x: Union[Tensor, ndarray], y: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
    """dot product (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        y (Union[Tensor, ndarray]): y, [..., C]

    Returns:
        Union[Tensor, ndarray]: x dot y, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sum(x * y, -1, keepdims=True)
    else:
        return torch.sum(x * y, -1, keepdim=True)


def length(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """length of an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: length, [..., 1]
    """
    if isinstance(x, np.ndarray):
        return np.sqrt(np.maximum(np.sum(x * x, axis=-1, keepdims=True), eps))
    else:
        return torch.sqrt(torch.clamp(dot(x, x), min=eps))


def safe_normalize(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """normalize an array (along the last dim).

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)


def normalize(x: Union[Tensor, ndarray], eps=1e-20) -> Union[Tensor, ndarray]:
    """normalize an array (along the last dim). alias of safe_normalize.

    Args:
        x (Union[Tensor, ndarray]): x, [..., C]
        eps (float, optional): eps. Defaults to 1e-20.

    Returns:
        Union[Tensor, ndarray]: normalized x, [..., C]
    """

    return x / length(x, eps)
