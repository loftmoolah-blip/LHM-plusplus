## -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-05-05 13:33:18
# @Function      : LHMPP inference codes.

"""
LHM++ (Efficient Large Human Reconstruction Model) All-for-One inference runner.

This module provides an inference runner for the LHMPP model, which is
designed for comprehensive human avatar reconstruction and rendering tasks.
The model integrates multiple components including pose estimation, mesh
reconstruction, and neural rendering capabilities.
"""

from accelerate.logging import get_logger

from core.runners import REGISTRY_RUNNERS

logger = get_logger(__name__)
from .base_inferrer import Inferrer


@REGISTRY_RUNNERS.register("infer.human_lrm_a4o")
class HumanLRMA4OInferer(Inferrer):
    """
    LHM++ All-for-One inference for comprehensive human avatar reconstruction.

    This class extends the base Inferrer to provide specialized inference capabilities
    for the LHM++ model. It handles the initialization of model components,
    loss functions, and optimization strategies required for human avatar reconstruction
    and rendering tasks.

    The trainer integrates multiple loss functions including pixel-wise, perceptual,
    total variation, ball, and offset losses to ensure high-quality reconstruction
    of human avatars from input images or video sequences.
    """

    def __init__(self):
        """
        Initialize the LHM++ inference.

        Sets up the model architecture, optimization components, and loss functions
        required for human avatar reconstruction. The initialization process includes:
        - Model instantiation with configuration parameters
        - Optimizer and scheduler setup for training/inference
        - Loss function initialization for comprehensive reconstruction quality
        """
        super().__init__()

        self.model = self._build_model(self.cfg)
        self.optimizer = self._build_optimizer(self.model, self.cfg)
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)

        (
            self.pixel_loss_fn,
            self.perceptual_loss_fn,
            self.tv_loss_fn,
            self.ball_loss,
            self.offset_loss,
        ) = self._build_loss_fn(self.cfg)

    def _build_model(self, cfg):
        """
        Build and initialize the LHM++ model architecture.

        Args:
            cfg: Configuration object containing model parameters and experiment settings.

        Returns:
            torch.nn.Module: Initialized LHM++ model instance.

        Raises:
            AssertionError: If the experiment type does not match 'lrm'.
            NotImplementedError: If the specified model name is not supported.
        """
        assert (
            cfg.experiment.type == "lrm"
        ), f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"

        model_name = cfg.model.get("model_name", "LHMA4O")

        if model_name == "LHMA4O":
            from core.models import ModelHumanA4OLRM as ModelHumanLRM
        else:
            raise NotImplementedError

        model = ModelHumanLRM(**cfg.model)
        return model

    def register_hooks(self):
        pass

    def get_smplx_params(self, data):
        """
        Extract SMPL-X model parameters from input data dictionary.

        SMPL-X (SMPL eXpressive) is a statistical 3D human body model that extends
        SMPL with additional parameters for facial expressions and hand poses.
        This method filters and extracts the relevant SMPL-X parameters from
        the input data dictionary.

        Args:
            data (dict): Input data dictionary containing various model parameters.

        Returns:
            dict: Dictionary containing only the SMPL-X specific parameters including:
                - root_pose: Global root orientation
                - body_pose: Joint rotations for body joints
                - jaw_pose: Jaw rotation for facial animation
                - leye_pose, reye_pose: Left and right eye rotations
                - lhand_pose, rhand_pose: Left and right hand joint rotations
                - expr: Facial expression parameters
                - trans: Global translation
                - betas: Shape parameters (PCA coefficients)
        """
        smplx_params = {}
        smplx_keys = [
            "root_pose",
            "body_pose",
            "jaw_pose",
            "leye_pose",
            "reye_pose",
            "lhand_pose",
            "rhand_pose",
            "expr",
            "trans",
            "betas",
        ]
        for k, v in data.items():
            if k in smplx_keys:
                # print(k, v.shape)
                smplx_params[k] = data[k]
        return smplx_params
