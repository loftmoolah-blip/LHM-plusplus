# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-15 16:04:19
# @Function      : Camera and Projection Matrix

import math

import numpy as np
import torch


def generate_rotation_matrix_y(degrees):
    """
    Generates a rotation matrix for rotation around the Y-axis by a given angle in degrees.

    Args:
        degrees (float): The rotation angle in degrees.

    Returns:
        numpy.ndarray: A 3x3 rotation matrix representing rotation about the Y-axis.
    """

    theta = math.radians(degrees)
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    R = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]

    return np.asarray(R, dtype=np.float32)


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """
    Computes the world-to-view (camera) transformation matrix.

    Args:
        R (numpy.ndarray): A 3x3 rotation matrix.
        t (numpy.ndarray): A 3-element translation vector.
        translate (numpy.ndarray, optional): Additional translation to apply to the camera center after transformation. Defaults to np.array([0.0, 0.0, 0.0]).
        scale (float, optional): Scaling factor for the camera center position. Defaults to 1.0.

    Returns:
        numpy.ndarray: A 4x4 world-to-view transformation matrix suitable for use in computer graphics pipelines.
    """

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """
    Constructs a perspective projection matrix.

    Args:
        znear (float): The near clipping plane distance.
        zfar (float): The far clipping plane distance.
        fovX (float): The horizontal field of view in radians.
        fovY (float): The vertical field of view in radians.

    Returns:
        torch.Tensor: A 4x4 projection matrix suitable for 3D rendering.
    """

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y


class Camera:
    """
    Camera class for representing a pinhole or perspective camera model.

    Attributes:
        FoVx (float): Horizontal field of view in radians.
        FoVy (float): Vertical field of view in radians.
        height (int): Image height in pixels.
        width (int): Image width in pixels.
        world_view_transform (torch.Tensor): 4x4 matrix transforming world coordinates to camera (view) coordinates.
        zfar (float): Far clipping plane distance.
        znear (float): Near clipping plane distance.
        trans (np.ndarray): Camera translation vector applied after transformation.
        scale (float): Scale factor applied to the camera center.
        projection_matrix (torch.Tensor): 4x4 projection matrix for camera intrinsics.
        full_proj_transform (torch.Tensor): Combined view and projection transform matrix.
        camera_center (torch.Tensor): 3D location of the camera center in world coordinates.
        intrinsic (torch.Tensor): Camera intrinsic matrix.

    Methods:
        from_c2w(c2w, intrinsic, height, width):
            Instantiates a Camera object from a camera-to-world matrix and intrinsics.
    """

    def __init__(
        self,
        w2c,
        intrinsic,
        FoVx,
        FoVy,
        height,
        width,
        trans=np.array([0.0, 0.0, 0.0]),
        scale=1.0,
    ) -> None:
        """
        Initializes the Camera object with extrinsics, intrinsics, field of view and additional parameters.

        Args:
            w2c (torch.Tensor): 4x4 world-to-camera extrinsic transformation matrix (transposed and used as view transform).
            intrinsic (torch.Tensor): 3x3 camera intrinsic matrix.
            FoVx (float): Horizontal field of view in radians.
            FoVy (float): Vertical field of view in radians.
            height (int): Image height in pixels.
            width (int): Image width in pixels.
            trans (np.ndarray, optional): Camera translation vector to apply after other transforms (default: [0.0, 0.0, 0.0]).
            scale (float, optional): Scale factor applied to the camera center (default: 1.0).
        """

        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = height
        self.width = width
        self.world_view_transform = w2c.transpose(0, 1)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(w2c.device)
        )
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.intrinsic = intrinsic

    @staticmethod
    def from_c2w(c2w, intrinsic, height, width):
        """
        Creates a Camera object from a camera-to-world (c2w) matrix and intrinsic parameters.

        Args:
            c2w (torch.Tensor): 4x4 camera-to-world extrinsic matrix.
            intrinsic (torch.Tensor): 3x3 camera intrinsic matrix.
            height (int): Image height in pixels.
            width (int): Image width in pixels.

        Returns:
            Camera: An instance of the Camera class constructed from the provided parameters.
        """

        w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )

        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,
        )

    @staticmethod
    def from_c2w_center_modfied(c2w, intrinsic, height, width):
        """
        Creates a Camera object from a camera-to-world (c2w) matrix and intrinsic parameters,
        but modifies the intrinsic matrix so that the principal point is set to the image center.

        Args:
            c2w (torch.Tensor): 4x4 camera-to-world extrinsic matrix.
            intrinsic (torch.Tensor): 3x3 camera intrinsic matrix.
            height (int): Image height in pixels.
            width (int): Image width in pixels.

        Returns:
            Camera: An instance of the Camera class constructed from the provided parameters, with adjusted intrinsic center.
        """

        w2c = torch.inverse(c2w)
        intrinsic = intrinsic.clone()

        intrinsic[0, 2] = width / 2.0
        intrinsic[1, 2] = height / 2.0

        FoVx, FoVy = intrinsic_to_fov(
            intrinsic,
            w=torch.tensor(width, device=w2c.device),
            h=torch.tensor(height, device=w2c.device),
        )

        return Camera(
            w2c=w2c,
            intrinsic=intrinsic,
            FoVx=FoVx,
            FoVy=FoVy,
            height=height,
            width=width,
        )
