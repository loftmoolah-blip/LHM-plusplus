# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Head rotation and front-face check from SMPL-X pose

from rot6d import *


def compute_head_global_rotation(smplx_pose):
    HEAD_PARENT_CHAIN = [
        0,
        3,
        6,
        9,
        12,
        15,
    ]  # pelvis, spine1, spine2, spine3, neck, head
    global_head_rotmat = torch.eye((3)).to(smplx_pose.device)
    for idx in HEAD_PARENT_CHAIN:
        rotmat = axis_angle_to_matrix(smplx_pose[idx])
        global_head_rotmat = torch.matmul(global_head_rotmat, rotmat)
    return global_head_rotmat


def is_front_face(smplx_pose, threshold=15):
    global_head_rotmat = compute_head_global_rotation(smplx_pose)
    head_euler = matrix_to_euler_angles(global_head_rotmat, "XYZ")
    PI = 3.1415926
    head_euler[0] = (head_euler[0] + 2 * PI) % (2 * PI) - PI

    threshold = PI * threshold / 180
    if (
        abs(head_euler[0]) <= threshold
        and abs(head_euler[1]) < threshold
        and abs(head_euler[2]) < threshold
    ):
        return True, head_euler

    return False, head_euler
