# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : CLI for SMPL-X pose estimation from single image

import argparse
import os
import pdb
import sys

import torch

sys.path.append("./")

from engine.pose_estimation.pose_estimator import PoseEstimator


def get_parse() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed arguments object.
    """
    parser = argparse.ArgumentParser(description="infer smplx pose demo")
    parser.add_argument(
        "-i",
        "--input",
        default="assets/pose_img.jpg",
        type=str,
        help="input image path",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parse()

    pose_estimator = PoseEstimator(
        "./pretrained_models/human_model_files/", device="cpu"
    )
    pose_estimator.to("cuda")
    pose_estimator.device = "cuda"
    with torch.no_grad():
        smplx_output = pose_estimator(args.input, is_shape=False)

    pdb.set_trace()
