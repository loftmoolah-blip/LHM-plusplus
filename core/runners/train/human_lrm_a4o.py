# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-17 10:24:29
# @Function      : LHMPP training files.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.logging import get_logger
from einops import rearrange
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from core.runners import REGISTRY_RUNNERS
from core.utils.profiler import DummyProfiler

from .base_trainer import Trainer

logger = get_logger(__name__)
# torch.autograd.set_detect_anomaly(True)


@REGISTRY_RUNNERS.register("train.human_lrm_a4o")
class HumanLRMA4OTrainer(Trainer):
    """
    Trainer for Efficient Large Human Model(LHMPP) architecture.

    This class implements the training pipeline for the LHMPP model, which is designed
    for human avatar generation and reconstruction tasks. The trainer handles the complete
    training workflow including data loading, model optimization, loss computation, and
    evaluation procedures.

    The trainer extends the base Trainer class and provides specialized implementations
    for human avatar generation tasks, incorporating domain-specific loss functions
    and evaluation metrics.

    Args:
        Inherits from base Trainer class configuration.

    Attributes:
        model: The LHMPP model instance for human avatar generation
        optimizer: AdamW optimizer with configurable learning rate scheduling
        scheduler: Cosine annealing scheduler with warmup
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        pixel_loss_fn: Pixel-wise reconstruction loss function
        perceptual_loss_fn: LPIPS-based perceptual loss function
        tv_loss_fn: Total variation regularization loss
        ball_loss: ASAP (As Spherical As Possible) loss for Gaussian regularization
        offset_loss: ACAP (As Compact As Possible) loss for Gaussian positioning
        opacity_loss: Opacity regularization loss for transparency control
    """

    def __init__(self):
        """
        Initialize the HumanLRMA4OTrainer with all necessary components.

        This constructor sets up the complete training pipeline by initializing:
        1. Data loaders for training and validation datasets
        2. The LHM-A4O model architecture
        3. Optimizer with configurable parameters
        4. Learning rate scheduler with cosine annealing and warmup
        5. Loss functions for various aspects of human avatar reconstruction

        The initialization follows a modular approach where each component is
        built independently based on configuration parameters, enabling flexible
        experimentation with different architectures and training strategies.
        """
        super().__init__()

        # Initialize data loaders for training and validation
        self.train_loader, self.val_loader = self._build_dataloader(self.cfg)

        # Initialize the LHM-A4O model architecture
        self.model = self._build_model(self.cfg)

        # Initialize optimizer with configurable learning rate and momentum parameters
        self.optimizer = self._build_optimizer(self.model, self.cfg)

        # Initialize learning rate scheduler with cosine annealing and warmup
        self.scheduler = self._build_scheduler(self.optimizer, self.cfg)

        # Initialize comprehensive loss function suite
        (
            self.pixel_loss_fn,  # Pixel-wise reconstruction loss (L1/L2)
            self.perceptual_loss_fn,  # LPIPS-based perceptual loss
            self.tv_loss_fn,  # Total variation regularization
            self.ball_loss,  # ASAP (As Spherical As Possible) loss
            self.offset_loss,  # ACAP (As Compact As Possible) loss
            self.opacity_loss,  # Opacity regularization loss
        ) = self._build_loss_fn(self.cfg)

    def _build_model(self, cfg):
        """
        Build the LHM-A4O model architecture based on configuration parameters.
        Args:
            cfg: Configuration object containing model parameters and architecture settings

        Returns:
            ModelHumanLRM: Initialized LHM-A4O model instance

        Raises:
            AssertionError: If the experiment type does not match "lrm"
            NotImplementedError: If the specified model name is not supported

        Note:
            Currently supports only "LHMA4O" model variant. The model integrates
            transformer-based architecture with 3D Gaussian splatting for efficient
            human avatar rendering and reconstruction.
        """
        # Validate experiment configuration matches expected type
        assert (
            cfg.experiment.type == "lrm"
        ), f"Config type {cfg.experiment.type} does not match with runner {self.__class__.__name__}"

        # Extract model name from configuration, defaulting to LHMA4O
        model_name = cfg.model.get("model_name", "LHMA4O")

        # Initialize model based on specified architecture
        if model_name == "LHMA4O":
            from core.models import ModelHumanA4OLRM as ModelHumanLRM
        else:
            raise NotImplementedError(
                f"Model architecture '{model_name}' is not implemented"
            )

        # Instantiate model with configuration parameters
        model = ModelHumanLRM(**cfg.model)

        return model

    def _build_optimizer(self, model: nn.Module, cfg):
        """
        Build the AdamW optimizer with configurable parameter groups and hyperparameters.

        This method creates an AdamW optimizer with different learning rates for different
        parameter groups within the model. The learning rate is initially set to 0.0 and
        will be updated by the learning rate scheduler during training. This approach
        allows for fine-grained control over the optimization process for different
        components of the LHM-A4O model.

        Args:
            model (nn.Module): The LHM-A4O model instance to optimize
            cfg: Configuration object containing optimizer hyperparameters

        Returns:
            torch.optim.AdamW: Configured AdamW optimizer instance

        Note:
            The optimizer uses AdamW variant which incorporates weight decay regularization
            and is particularly effective for transformer-based architectures. The beta
            parameters control the exponential moving averages of gradients and squared
            gradients, with default values typically set to (0.9, 0.999) for stable
            optimization of large-scale models.
        """
        # Obtain parameter groups with potentially different learning rates
        opt_groups = model.obtain_params(cfg)

        # Initialize AdamW optimizer with configurable hyperparameters
        optimizer = torch.optim.AdamW(
            opt_groups,
            lr=0.0,  # Initial learning rate (will be set by scheduler)
            betas=(cfg.train.optim.beta1, cfg.train.optim.beta2),  # Momentum parameters
        )

        return optimizer

    def _build_scheduler(self, optimizer, cfg):
        """
        Build the learning rate scheduler with cosine annealing and warmup.

        This method creates a cosine annealing scheduler with linear warmup, which is
        particularly effective for training large transformer-based models. The scheduler
        gradually increases the learning rate during the warmup phase and then applies
        cosine annealing for the remaining training steps, helping to achieve better
        convergence and final performance.

        Args:
            optimizer: The optimizer to schedule learning rates for
            cfg: Configuration object containing scheduler parameters

        Returns:
            CosineWarmupScheduler: Configured learning rate scheduler instance

        Raises:
            NotImplementedError: If the specified scheduler type is not supported

        Note:
            The scheduler calculates the total number of global batches considering
            distributed training, gradient accumulation, and multiple epochs. This
            ensures accurate learning rate scheduling across the entire training process.
            The warmup phase helps stabilize training in the early stages, while
            cosine annealing provides smooth decay toward the end of training.
        """
        # Calculate local batches per epoch considering distributed training
        local_batches_per_epoch = math.floor(
            len(self.train_loader) / self.accelerator.num_processes
        )

        # Calculate total global batches across all epochs with gradient accumulation
        total_global_batches = cfg.train.epochs * math.ceil(
            local_batches_per_epoch / self.cfg.train.accum_steps
        )

        # Get effective warmup iterations from configuration
        effective_warmup_iters = cfg.train.scheduler.warmup_real_iters

        # Log scheduler configuration for debugging
        logger.debug(
            f"======== Scheduler effective max iters: {total_global_batches} ========"
        )
        logger.debug(
            f"======== Scheduler effective warmup iters: {effective_warmup_iters} ========"
        )

        # Initialize scheduler based on specified type
        if cfg.train.scheduler.type == "cosine":
            from core.utils.scheduler import CosineWarmupScheduler

            scheduler = CosineWarmupScheduler(
                optimizer=optimizer,
                warmup_iters=effective_warmup_iters,
                max_iters=total_global_batches,
            )
        else:
            raise NotImplementedError(
                f"Scheduler type {cfg.train.scheduler.type} not implemented"
            )
        return scheduler

    def _build_dataloader(self, cfg):
        """
        Build data loaders for training and validation datasets.

        This method creates MixerDataset instances for both training and validation,
        which combines multiple human avatar datasets with different resolutions and
        rendering configurations. The dataset supports multi-view human reconstruction
        with SMPL-X parameter integration and provides both source images and rendered
        views for training the LHM-A4O model.

        Args:
            cfg: Configuration object containing dataset and data loader parameters

        Returns:
            tuple: (train_loader, val_loader) - PyTorch DataLoader instances for training and validation

        Note:
            The MixerDataset handles multiple dataset subsets, supports different image
            resolutions for source and rendered images, and provides comprehensive
            data augmentation and preprocessing for human avatar training. The data
            loaders are configured with appropriate batch sizes, worker processes,
            and memory pinning for optimal training performance.
        """
        from core.datasets import MixerDataset

        # Build training dataset with comprehensive configuration
        train_dataset = MixerDataset(
            split="train",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            repeat_num=(
                cfg.dataset.repeat_num if hasattr(cfg.dataset, "repeat_num") else 1
            ),
            multiply=cfg.dataset.multiply if hasattr(cfg.dataset, "multiply") else 14,
            debug=cfg.dataset.debug if hasattr(cfg.dataset, "debug") else False,
            is_val=False,
        )

        # Build validation dataset with identical configuration
        val_dataset = MixerDataset(
            split="val",
            subsets=cfg.dataset.subsets,
            sample_side_views=cfg.dataset.sample_side_views,
            render_image_res_low=cfg.dataset.render_image.low,
            render_image_res_high=cfg.dataset.render_image.high,
            render_region_size=cfg.dataset.render_image.region,
            source_image_res=cfg.dataset.source_image_res,
            repeat_num=(
                cfg.dataset.repeat_num if hasattr(cfg.dataset, "repeat_num") else 1
            ),
            multiply=cfg.dataset.multiply if hasattr(cfg.dataset, "multiply") else 14,
            debug=cfg.dataset.debug if hasattr(cfg.dataset, "debug") else False,
            is_val=True,  # Validation mode
        )

        # Build training data loader with optimized settings
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=True,  # Shuffle training data
            drop_last=True,  # Drop incomplete batches
            num_workers=cfg.dataset.num_train_workers,  # Number of worker processes
            pin_memory=cfg.dataset.pin_mem,  # Pin memory for GPU transfer
            persistent_workers=True,  # Keep workers alive between epochs
        )

        # Build validation data loader with appropriate settings
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=cfg.val.batch_size,
            shuffle=False,  # No shuffling for validation
            drop_last=False,  # Keep all validation samples
            num_workers=cfg.dataset.num_val_workers,  # Number of worker processes
            pin_memory=cfg.dataset.pin_mem,  # Pin memory for GPU transfer
            persistent_workers=False,  # No persistent workers for validation
        )

        return train_loader, val_loader

    def _build_loss_fn(self, cfg):
        """
        Build comprehensive loss functions for LHM-A4O training.

        This method constructs a suite of loss functions specifically designed for
        human avatar generation and reconstruction. The loss functions include:

        1. Pixel-wise reconstruction losses (L1/L2)
        2. Perceptual losses (LPIPS) for high-quality rendering
        3. Total variation regularization for smoothness
        4. ASAP (As Spherical As Possible) loss for Gaussian regularization
        5. ACAP (As Compact As Possible) loss for Gaussian positioning
        6. Opacity regularization for transparency control

        The method supports both classical and heuristic variants of the geometric
        regularization losses, allowing for flexible training strategies based on
        the specific requirements of human avatar generation.

        Args:
            cfg: Configuration object containing loss function parameters

        Returns:
            tuple: (pixel_loss_fn, perceptual_loss_fn, tv_loss_fn, ball_loss, offset_loss, opacity_loss)
                - pixel_loss_fn: Pixel-wise reconstruction loss (L1/L2)
                - perceptual_loss_fn: LPIPS-based perceptual loss
                - tv_loss_fn: Total variation regularization loss
                - ball_loss: ASAP loss for Gaussian shape regularization
                - offset_loss: ACAP loss for Gaussian positioning
                - opacity_loss: Opacity regularization loss
        """
        from core.losses import (
            ACAP_Loss,
            ASAP_Loss,
            Heuristic_ACAP_Loss,
            Heuristic_ASAP_Loss,
            Heuristic_Opacity_Loss,
            LPIPSLoss,
            Opacity_Loss,
            PixelLoss,
            TVLoss,
        )

        # Build pixel-wise reconstruction loss function
        pixel_loss_fn = PixelLoss(cfg.train.loss_func.get("pixel_loss", "mse"))

        # Extract loss configuration handles for geometric regularization
        ball_loss_handle = cfg.train.loss_func.get("ball_loss", None)
        offset_loss_handle = cfg.train.loss_func.get("offset_loss", None)
        opacity_loss_handle = cfg.train.loss_func.get("opacity_loss", None)
        # Build ASAP (As Spherical As Possible) loss for Gaussian shape regularization
        if ball_loss_handle is not None:
            if ball_loss_handle["type"] == "classical":
                ball_loss_func = ASAP_Loss()
            elif ball_loss_handle["type"] == "heuristic":
                # Heuristic variant with body part grouping for better regularization
                ball_loss_func = Heuristic_ASAP_Loss(
                    group_dict=ball_loss_handle["group"],
                    group_body_mapping=self.model.renderer.smplx_model.get_body_infos(),
                )
            else:
                raise NotImplementedError(
                    f"ASAP loss type '{ball_loss_handle['type']}' not implemented"
                )
        else:
            ball_loss_func = None

        # Build ACAP (As Compact As Possible) loss for Gaussian positioning
        if offset_loss_handle is not None:
            if offset_loss_handle["type"] == "classical":
                offset_loss_func = ACAP_Loss()
            elif offset_loss_handle["type"] == "heuristic":
                # Heuristic variant with body part grouping for anatomical constraints
                offset_loss_func = Heuristic_ACAP_Loss(
                    group_dict=offset_loss_handle["group"],
                    group_body_mapping=self.model.renderer.smplx_model.get_body_infos(),
                )
            else:
                raise NotImplementedError(
                    f"ACAP loss type '{offset_loss_handle['type']}' not implemented"
                )
        else:
            offset_loss_func = None

        # Build opacity regularization loss for transparency control
        if opacity_loss_handle is not None:
            if opacity_loss_handle["type"] == "classical":
                opacity_loss_func = Opacity_Loss()
            elif opacity_loss_handle["type"] == "heuristic":
                # Heuristic variant with body part grouping for anatomical constraints
                opacity_loss_func = Heuristic_Opacity_Loss(
                    group_dict=opacity_loss_handle["group"],
                    group_body_mapping=self.model.renderer.smplx_model.get_body_infos(),
                )
            else:
                raise NotImplementedError(
                    f"Opacity loss type '{opacity_loss_handle['type']}' not implemented"
                )
        else:
            opacity_loss_func = None

        # Initialize perceptual loss with main process synchronization
        with self.accelerator.main_process_first():
            perceptual_loss_fn = LPIPSLoss(device=self.device, prefech=True)

        # Initialize total variation regularization loss
        tv_loss_fn = TVLoss()

        return (
            pixel_loss_fn,
            perceptual_loss_fn,
            tv_loss_fn,
            ball_loss_func,
            offset_loss_func,
            opacity_loss_func,
        )

    def register_hooks(self):
        pass

    def get_smplx_params(self, data):
        """
        Extract SMPL-X parameters from input data dictionary.

        This method filters and extracts SMPL-X (SMPL eXtended) parameters from the
        input data dictionary. SMPL-X is a parametric human body model that extends
        the original SMPL model with additional parameters for facial expressions,
        hand poses, and jaw articulation, enabling comprehensive human avatar modeling.

        Args:
            data (dict): Input data dictionary containing various parameters including SMPL-X parameters

        Returns:
            dict: Dictionary containing only the SMPL-X parameters extracted from input data

        Note:
            SMPL-X parameters include:
            - root_pose: Global root joint rotation (3 parameters)
            - body_pose: Body joint rotations (63 parameters for 21 joints)
            - jaw_pose: Jaw articulation parameters (3 parameters)
            - leye_pose, reye_pose: Left and right eye pose parameters (3 parameters each)
            - lhand_pose, rhand_pose: Left and right hand pose parameters (45 parameters each)
            - expr: Facial expression parameters (10 parameters)
            - trans: Global translation (3 parameters)
            - betas: Shape parameters controlling body proportions (10 parameters)
        """
        smplx_params = {}
        smplx_keys = [
            "root_pose",  # Global root joint rotation
            "body_pose",  # Body joint rotations
            "jaw_pose",  # Jaw articulation
            "leye_pose",  # Left eye pose
            "reye_pose",  # Right eye pose
            "lhand_pose",  # Left hand pose
            "rhand_pose",  # Right hand pose
            "expr",  # Facial expressions
            "trans",  # Global translation
            "betas",  # Shape parameters
        ]

        # Extract SMPL-X parameters from input data
        for k, v in data.items():
            if k in smplx_keys:
                smplx_params[k] = data[k]
        return smplx_params

    def cross_copy(self, data):
        """
        Perform cross-copying operation for cross-rendering data augmentation.

        This method creates cross-rendering data by copying each sample from the batch
        and combining it with all other samples in the batch. This technique is used
        for cross-rendering scenarios where different human identities are rendered
        with swapped poses or appearances, enhancing model generalization and robustness.

        The operation transforms input data from shape [B, 1, ...] to [B, B, ...] where
        each element [i, j, ...] contains the i-th sample with the j-th sample's data
        (except when i=j, where it contains the original sample).

        Args:
            data (torch.Tensor): Input tensor with shape [B, 1, ...] where B is batch size

        Returns:
            torch.Tensor: Cross-copied tensor with shape [B, B, ...]

        Raises:
            AssertionError: If input data does not have exactly 1 element in dimension 1

        Note:
            This method is particularly useful for cross-rendering scenarios in human
            avatar generation, where different identities are rendered with swapped
            poses or appearances to improve model generalization and reduce overfitting
            to specific identity-pose combinations.
        """
        B = data.shape[0]
        assert data.shape[1] == 1, f"Expected data shape [B, 1, ...], got {data.shape}"

        new_data = []
        for i in range(B):
            # Start with the original sample
            B_i = [data[i]]
            # Add all other samples in the batch
            for j in range(B):
                if j != i:
                    B_i.append(data[j])
            # Concatenate along the view dimension
            new_data.append(torch.concat(B_i, dim=0))

        # Stack all cross-copied samples
        new_data = torch.stack(new_data, dim=0)

        return new_data

    def prepare_cross_render_data(self, data):
        """
        Prepare data for cross-rendering by applying cross-copying to relevant components.

        This method prepares input data for cross-rendering scenarios by applying the
        cross-copying operation to camera parameters, rendering data, and SMPL-X pose
        parameters (excluding shape parameters). Cross-rendering enables training the
        model to handle identity-pose combinations that were not seen during training,
        improving generalization capabilities.

        The method modifies the input data dictionary by cross-copying:
        - Camera parameters (c2ws, intrs)
        - Rendering data (render_image, render_mask, render_bg_colors)
        - SMPL-X pose parameters (excluding betas shape parameters)

        Args:
            data (dict): Input data dictionary containing rendering and SMPL-X parameters

        Returns:
            dict: Modified data dictionary with cross-copied components

        Raises:
            AssertionError: If render_image does not have exactly 1 view per sample

        Note:
            Shape parameters (betas) are excluded from cross-copying as they represent
            intrinsic body proportions that should remain consistent with the identity.
            Only pose-related parameters are cross-copied to enable pose transfer
            between different identities while maintaining identity-specific characteristics.
        """
        B, N_v, C, H, W = data["render_image"].shape
        assert N_v == 1, f"Expected single view per sample, got {N_v} views"

        # Apply cross-copying to camera parameters
        data["c2ws"] = self.cross_copy(data["c2ws"])
        data["intrs"] = self.cross_copy(data["intrs"])

        # Apply cross-copying to rendering data
        data["render_full_resolutions"] = self.cross_copy(
            data["render_full_resolutions"]
        )
        data["render_image"] = self.cross_copy(data["render_image"])
        data["render_mask"] = self.cross_copy(data["render_mask"])
        data["render_bg_colors"] = self.cross_copy(data["render_bg_colors"])

        # Apply cross-copying to SMPL-X pose parameters (excluding shape parameters)
        smplx_params = self.get_smplx_params(data)
        for key in smplx_params.keys():
            if "betas" not in key:  # Preserve identity-specific shape parameters
                data[key] = self.cross_copy(data[key])

        return data

    def get_loss_weight(self, loss_weight):
        """
        Compute dynamic loss weight based on training step and configuration.

        This method supports both static and dynamic loss weight scheduling. For dynamic
        scheduling, it implements linear interpolation between start and end values
        over a specified range of training steps. This allows for curriculum learning
        strategies where different loss components are emphasized at different stages
        of training.

        Args:
            loss_weight: Loss weight configuration, either:
                - float/int: Static weight value
                - str: Dynamic weight in format "start_step:start_value:end_value:end_step"

        Returns:
            float: Computed loss weight for current training step

        Raises:
            NotImplementedError: If loss_weight format is not supported

        Note:
            Dynamic weight scheduling is particularly useful for human avatar training
            where different loss components (e.g., pixel loss, perceptual loss, geometric
            regularization) may need different emphasis at different training stages.
            The linear interpolation ensures smooth transitions between weight values.
        """
        if isinstance(loss_weight, str) and ":" in loss_weight:
            # Parse dynamic weight configuration
            start_step, start_value, end_value, end_step = map(
                float, loss_weight.split(":")
            )
            current_step = self.global_step

            # Compute linear interpolation between start and end values
            value = start_value + (end_value - start_value) * max(
                min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0
            )
            return value
        elif isinstance(loss_weight, (float, int)):
            # Static weight value
            return loss_weight
        else:
            raise NotImplementedError(
                f"Unsupported loss_weight format: {type(loss_weight)}"
            )

    def gray_resize_for_identity(self, out, size=128):
        """
        Convert RGB image to grayscale and resize for face identity processing.

        This method converts RGB images to grayscale using standard luminance weights
        and resizes them to a fixed size for face identity network processing. The
        conversion uses the standard RGB-to-grayscale weights (0.2989, 0.5870, 0.1140)
        which correspond to the human eye's sensitivity to different color channels.

        Args:
            out (torch.Tensor): Input RGB tensor with shape [B, 3, H, W]
            size (int, optional): Target size for resizing. Defaults to 128.

        Returns:
            torch.Tensor: Grayscale tensor with shape [B, 1, size, size]

        Note:
            This method is used for preprocessing face regions before feeding them
            to the face identity network. The grayscale conversion removes color
            information that might interfere with identity recognition, while the
            fixed size ensures consistent input dimensions for the identity network.
        """
        # Convert RGB to grayscale using standard luminance weights
        out_gray = (
            0.2989 * out[:, 0, :, :]  # Red channel weight
            + 0.5870 * out[:, 1, :, :]  # Green channel weight
            + 0.1140 * out[:, 2, :, :]  # Blue channel weight
        )
        out_gray = out_gray.unsqueeze(1)  # Add channel dimension

        # Resize to target size using bilinear interpolation
        out_gray = F.interpolate(
            out_gray, (size, size), mode="bilinear", align_corners=False
        )
        return out_gray

    def get_identity_loss(self, preds, gts, render_masks, use_heads):
        """
        Compute face identity loss for face region consistency.

        This method computes identity loss by comparing face regions between predicted
        and ground truth images using a pre-trained face identity network. The loss
        is designed to maintain facial identity consistency across different poses
        and expressions. However, this loss was found to be unstable during training
        and is not used in the final version of the model.

        Args:
            preds (torch.Tensor): Predicted rendered images [B, N, C, H, W]
            gts (torch.Tensor): Ground truth images [B, N, C, H, W]
            render_masks (torch.Tensor): Rendering masks for face detection [B, N, C, H, W]
            use_heads (torch.Tensor): Boolean tensor indicating which samples contain heads [B, N, 1]

        Returns:
            torch.Tensor: Face identity loss value, or 0.0 if no valid face regions found

        Note:
            This loss function is currently deprecated due to training instability.
            The method extracts face regions using bounding box detection, converts
            them to grayscale, and compares identity features extracted by a pre-trained
            face recognition network. While conceptually useful for identity preservation,
            the implementation was found to be unstable and removed from the final model.
        """

        def min_bbox(mask):
            mask = (mask > 0.5).float()
            y_indices, x_indices = torch.where(mask > 0)
            x_min = x_indices.min()
            x_max = x_indices.max()
            y_min = y_indices.min()
            y_max = y_indices.max()

            return [y_min, y_max, x_min, x_max]

        if self.model.module.use_face_id:

            B, N, C, H, W = preds.shape

            preds = rearrange(preds, "b n c h w -> (b n) c h w")
            gts = rearrange(gts, "b n c h w -> (b n) c h w")
            render_masks = rearrange(render_masks, "b n c h w -> (b n) c h w")
            render_masks = render_masks.squeeze(1)
            use_heads = rearrange(use_heads, "b n 1-> (b n) 1")

            face_id_loss = 0.0
            identity_gt_list = []
            identity_out_list = []
            for pred, gt, render_mask, use_head in zip(
                preds, gts, render_masks, use_heads
            ):
                if use_head.sum() == 1:

                    try:
                        y_min, y_max, x_min, x_max = min_bbox(render_mask)
                        head_pred = pred[:, y_min:y_max, x_min:x_max]
                        head_gt = gt[:, y_min:y_max, x_min:x_max]

                        out_gray = self.gray_resize_for_identity(head_pred.unsqueeze(0))
                        gt_gray = self.gray_resize_for_identity(head_gt.unsqueeze(0))

                        identity_gt = self.model.module.id_face_net(gt_gray).detach()
                        identity_out = self.model.module.id_face_net(out_gray)

                        identity_gt_list.append(identity_gt)
                        identity_out_list.append(identity_out)
                    except:
                        continue

            if use_heads.sum() > 0 and len(identity_gt_list) > 0:
                identity_gt = torch.cat(identity_gt_list, dim=0)
                identity_out = torch.cat(identity_out_list, dim=0)
                face_id_loss = F.l1_loss(identity_out, identity_gt)

                return face_id_loss
            else:
                return torch.tensor(0.0).to(preds)

        else:
            return None

    def forward_loss_local_step(self, data):
        """
        Perform forward pass and compute comprehensive loss for a single training step.

        This method executes the forward pass through the LHM-A4O model and computes
        a comprehensive set of losses including pixel-wise reconstruction, perceptual,
        geometric regularization, and identity preservation losses. The method handles
        both standard and cross-rendering scenarios, providing detailed loss breakdowns
        for monitoring training progress.

        Args:
            data (dict): Input data dictionary containing:
                - render_image: Ground truth rendered images [B, N_v, C, H, W]
                - render_mask: Rendering masks [B, N_v, C, H, W]
                - render_padding_mask: SMPL-X estimator padding masks [B, N_v, C, H, W]
                - use_heads: Boolean tensor for head presence [B, N_v, 1]
                - render_head_mask: Head region masks [B, N_v, C, H, W]
                - source_rgbs: Source RGB images for conditioning
                - c2ws: Camera-to-world transformation matrices
                - intrs: Camera intrinsic parameters
                - render_bg_colors: Background colors for rendering
                - SMPL-X parameters: Pose and shape parameters

        Returns:
            dict: Comprehensive loss dictionary containing:
                - outputs: Rendered results (comp_rgb, comp_mask)
                - loss: Total weighted loss
                - loss_pixel: Pixel-wise reconstruction loss
                - loss_masked_pixel: Masked pixel loss
                - loss_perceptual: LPIPS perceptual loss
                - loss_masked_perceptual: Masked perceptual loss
                - loss_face_id: Face identity loss (if enabled)
                - loss_mask: Mask prediction loss
                - loss_offset: ACAP loss for Gaussian positioning
                - ball_loss: ASAP loss for Gaussian regularization
                - loss_opacity: Opacity regularization loss
                - PSNR_GS: Peak Signal-to-Noise Ratio
                - Masked_PSNR_GS: Masked PSNR
        """
        # Extract ground truth data
        render_image = data["render_image"]
        render_mask = data["render_mask"]
        render_padding_mask = data["render_padding_mask"]  # SMPL-X estimator padding

        # Extract head-related data
        use_heads = data["use_heads"]
        render_head_mask = data["render_head_mask"]

        B, N_v, C, H, W = render_image.shape
        smplx_params = self.get_smplx_params(data)

        # Define PSNR computation function
        psnr_fn = lambda x, y: -10.0 * torch.log10(
            nn.functional.mse_loss(x, y).detach()
        )

        # Forward pass through the LHM-A4O model
        # The model outputs:
        # - latent_points: Image feature tokens
        # - comp_rgb: Rendered RGB images
        # - comp_rgb_bg: Background rendering
        # - comp_mask: Gaussian alpha values
        # - comp_depth: Rendered depth maps
        # - 3dgs: 3D Gaussian splatting model

        # Execute forward pass through the LHM-A4O model
        outputs = self.model(
            image=data["source_rgbs"],  # Source RGB images for conditioning
            source_c2ws=None,  # Source camera poses (not used)
            source_intrs=None,  # Source camera intrinsics (not used)
            render_c2ws=data["c2ws"],  # Target camera poses for rendering
            render_intrs=data["intrs"],  # Target camera intrinsics
            render_bg_colors=data[
                "render_bg_colors"
            ],  # Background colors for rendering
            smplx_params=smplx_params,  # SMPL-X pose and shape parameters
            source_head_rgbs=data["source_head_rgbs"],  # Source head region images
            df_data=None,  # Discriminator features (for GAN loss if enabled)
            query_pts_path=data.get("query_pts_path", None),  # Mesh-based query points
            ref_imgs_bool=data.get(
                "ref_imgs_bool", None
            ),  # Reference image data for multi-view inputs
        )

        # Initialize loss computation variables
        loss = 0.0
        loss_head_pixel = None
        loss_pixel = None
        loss_masked_pixel = None
        loss_D = None  # StyleGAN2-human discriminator loss
        loss_offset = None  # ACAP (As Compact As Possible) loss
        loss_opacity = None  # Opacity regularization loss
        loss_perceptual = None  # LPIPS perceptual loss
        loss_masked_perceptual = None  # Masked perceptual loss
        loss_face_id = None  # Face identity loss
        loss_mask = None  # Mask prediction loss
        ball_loss = None  # ASAP (As Spherical As Possible) loss
        masked_psnr_gs, psnr_gs = None, None  # PSNR metrics

        real_num_view = data["real_num_view"]

        # Compute masked pixel loss for human region reconstruction
        if (
            self.get_loss_weight(self.cfg.train.loss.get("masked_pixel_weight", 0))
            > 0.0
        ):
            # Apply rendering mask to focus on human regions
            pred_rgb = outputs["comp_rgb"][:, :real_num_view] * render_mask[
                :, :real_num_view
            ] + 1.0 * (1 - render_mask[:, :real_num_view])
            gt_rgb = render_image[:, :real_num_view] * render_mask[
                :, :real_num_view
            ] + 1.0 * (1 - render_mask[:, :real_num_view])

            # Compute pixel-wise reconstruction loss (L1/L2)
            loss_masked_pixel = self.pixel_loss_fn(pred_rgb, gt_rgb)
            masked_psnr_gs = psnr_fn(pred_rgb, gt_rgb)

            # Compute perceptual loss for high-quality rendering
            loss_masked_perceptual = self.perceptual_loss_fn(pred_rgb, gt_rgb)

            # Add weighted losses to total loss
            loss += loss_masked_pixel * self.get_loss_weight(
                self.cfg.train.loss.masked_pixel_weight
            )
            loss += loss_masked_perceptual * self.get_loss_weight(
                self.cfg.train.loss.masked_pixel_weight
            )
        # Compute standard pixel loss with padding mask handling
        if self.get_loss_weight(self.cfg.train.loss.pixel_weight) > 0.0:
            render_padding_mask = render_padding_mask[:, :real_num_view]

            # Apply padding mask to exclude SMPL-X estimator padding regions
            valid_comp_rgb = outputs["comp_rgb"][:, :real_num_view] * (
                1 - render_padding_mask
            )
            valid_render_rgb = render_image[:, :real_num_view] * (
                1 - render_padding_mask
            )

            # Compute pixel-wise reconstruction loss on valid regions
            loss_pixel = self.pixel_loss_fn(valid_comp_rgb, valid_render_rgb)

            # Add pixel loss to total loss
            loss += loss_pixel * self.get_loss_weight(self.cfg.train.loss.pixel_weight)

            # Compute perceptual loss for high-quality rendering
            if self.get_loss_weight(self.cfg.train.loss.perceptual_weight) > 0.0:
                loss_perceptual = self.perceptual_loss_fn(
                    valid_comp_rgb, valid_render_rgb
                )
                loss += loss_perceptual * self.get_loss_weight(
                    self.cfg.train.loss.perceptual_weight
                )

            # Compute PSNR for monitoring
            psnr_gs = psnr_fn(valid_comp_rgb, valid_render_rgb)

        # Compute mask prediction loss for alpha channel accuracy
        if self.get_loss_weight(self.cfg.train.loss.mask_weight) > 0.0:
            comp_mask = outputs["comp_mask"][:, :real_num_view]
            render_mask = render_mask[:, :real_num_view]
            render_padding_mask = render_padding_mask[:, :real_num_view]

            # Apply padding mask to exclude SMPL-X estimator padding regions
            valid_render_mask = render_mask * (1 - render_padding_mask)
            valid_comp_mask = comp_mask * (1 - render_padding_mask)

            # Compute mask prediction loss
            loss_mask = self.pixel_loss_fn(valid_comp_mask, valid_render_mask)
            loss += loss_mask * self.get_loss_weight(self.cfg.train.loss.mask_weight)

        # Extract mesh metadata for geometric regularization losses
        mesh_meta = outputs["mesh_meta"]

        # Compute ASAP (As Spherical As Possible) loss for Gaussian shape regularization
        if (
            self.ball_loss is not None
            and self.get_loss_weight(self.cfg.train.loss.asap_weight) > 0.0
        ):
            ball_loss = self.ball_loss(outputs["scaling_output"], mesh_meta=mesh_meta)
            loss += ball_loss * self.get_loss_weight(self.cfg.train.loss.asap_weight)

        # Compute ACAP (As Compact As Possible) loss for Gaussian positioning
        if (
            self.offset_loss is not None
            and self.get_loss_weight(self.cfg.train.loss.acap_weight) > 0.0
        ):
            loss_offset = self.offset_loss(
                outputs["offset_output"], mesh_meta=mesh_meta
            )
            loss += loss_offset * self.get_loss_weight(self.cfg.train.loss.acap_weight)

        # Compute opacity regularization loss for transparency control
        if (
            self.opacity_loss is not None
            and self.get_loss_weight(self.cfg.train.loss.opacity_weight) > 0.0
        ):
            loss_opacity = self.opacity_loss(
                outputs["opacity_output"], mesh_meta=mesh_meta
            )
            loss += loss_opacity * self.get_loss_weight(
                self.cfg.train.loss.opacity_weight
            )

        # Compute face identity loss for identity preservation (if enabled)
        if self.get_loss_weight(self.cfg.train.loss.face_id_weight) > 0.0:
            loss_face_id = self.get_identity_loss(
                outputs["comp_rgb"], render_image, render_head_mask, use_heads=use_heads
            )
            loss += loss_face_id * self.get_loss_weight(
                self.cfg.train.loss.face_id_weight
            )

        # Clean up input data to free memory
        del data

        # Prepare rendered results for visualization and monitoring
        render_results = dict(
            comp_rgb=outputs["comp_rgb"].detach().cpu(),
            comp_mask=outputs["comp_mask"].detach().cpu(),
        )

        # Return comprehensive loss dictionary for training monitoring
        return {
            "outputs": render_results,
            "loss": loss,
            "loss_pixel": loss_pixel,
            "loss_masked_pixel": loss_masked_pixel,
            "loss_opacity": loss_opacity,
            "loss_perceptual": loss_perceptual,
            "loss_masked_perceptual": loss_masked_perceptual,
            "loss_face_id": loss_face_id,
            "loss_D": loss_D,
            "loss_mask": loss_mask,
            "loss_offset": loss_offset,
            "ball_loss": ball_loss,
            "PSNR_GS": psnr_gs,
            "Masked_PSNR_GS": masked_psnr_gs,
        }

    def adopt_weight(self, weight, global_step, threshold=0, value=0.0):
        """
        Apply threshold-based weight scheduling.

        This method implements a simple threshold-based weight scheduling where
        the weight is set to a specified value before a threshold step and
        remains at the original weight after the threshold.

        Args:
            weight: Original weight value
            global_step: Current training step
            threshold: Step threshold for weight change
            value: Weight value to use before threshold

        Returns:
            float: Adjusted weight value
        """
        if global_step < threshold:
            weight = value
        return weight

    def calculate_adaptive_weight(
        self, nll_loss, g_loss, last_layer, discriminator_weight=1
    ):
        """
        Calculate adaptive discriminator weight based on gradient norms.

        This method computes an adaptive weight for discriminator loss based on
        the ratio of negative log-likelihood gradients to generator gradients.
        This helps balance the discriminator and generator losses during training.

        Args:
            nll_loss: Negative log-likelihood loss
            g_loss: Generator loss
            last_layer: Last layer of the model for gradient computation
            discriminator_weight: Base discriminator weight multiplier

        Returns:
            torch.Tensor: Adaptive discriminator weight
        """
        # Compute gradients for both losses
        nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        # Calculate adaptive weight based on gradient norm ratio
        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * discriminator_weight
        return d_weight

    def disc_preprocess(self, img):
        """
        Preprocess images for discriminator input.

        This method prepares images for discriminator processing by reshaping,
        normalizing, and resizing them according to discriminator requirements.
        The preprocessing includes flattening batch and view dimensions, converting
        from [0,1] to [-1,1] range, and resizing to the discriminator's expected
        input size.

        Args:
            img (torch.Tensor): Input images with shape [B, N_v, C, H, W]

        Returns:
            torch.Tensor: Preprocessed images with shape [B*N_v, C, H, W]

        Note:
            This method is used for discriminator training when adversarial
            losses are enabled. The preprocessing ensures that images are in
            the correct format and size for the discriminator network.
        """
        # Reshape from [B, N_v, C, H, W] to [B*N_v, C, H, W]
        img = torch.flatten(img, 0, 1)

        # Convert from [0,1] to [-1,1] range for discriminator input
        img = 2 * img - 1

        # Resize to discriminator's expected input size if specified
        if hasattr(self.accelerator.unwrap_model(self.model_disc), "input_size"):
            tgt_size = self.accelerator.unwrap_model(self.model_disc).input_size
            img = nn.functional.interpolate(img, (tgt_size, tgt_size))

        # Ensure float type
        img = img.float()

        return img

    def forward_to_get_loss(self, data):
        """
        Perform forward pass, compute loss, and execute backward propagation.

        This method orchestrates the complete training step including forward pass,
        loss computation, backward propagation, gradient clipping, and optimizer
        updates. It handles distributed training synchronization and gradient
        accumulation automatically through the accelerator framework.

        Args:
            data (dict): Input data dictionary for the training step

        Returns:
            dict: Loss dictionary containing all computed losses and metrics

        Note:
            This method integrates with the Hugging Face Accelerate framework
            for distributed training, automatic mixed precision, and gradient
            accumulation. The gradient clipping helps stabilize training for
            large transformer-based models.
        """
        # Perform forward pass and compute comprehensive losses
        loss_dict = self.forward_loss_local_step(data)

        # Execute backward propagation through the accelerator
        self.accelerator.backward(loss_dict["loss"])

        # Apply gradient clipping if enabled and gradients are synchronized
        if (
            self.accelerator.sync_gradients
            and self.cfg.train.optim.clip_grad_norm > 0.0
        ):
            self.accelerator.clip_grad_norm_(
                self.model.parameters(), self.cfg.train.optim.clip_grad_norm
            )

        # Update model parameters and reset gradients
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss_dict

    def model_step(self, step):
        """
        Update model hyperparameters at each training step.

        This method calls the model's hyper_step function to update any dynamic
        hyperparameters or internal states that need to be adjusted during training.
        This is particularly useful for models with learnable parameters that
        change over time, such as adaptive attention mechanisms or dynamic
        architectural components.

        Args:
            step (int): Current training step number

        Note:
            The hyper_step method is implemented by the model itself and may
            include operations like updating attention weights, adjusting
            architectural parameters, or modifying internal model states
            based on the current training progress.
        """
        self.model.module.hyper_step(step)

    def train_epoch(
        self,
        pbar: tqdm,
        loader: torch.utils.data.DataLoader,
        profiler: torch.profiler.profile,
    ):
        """
        Execute a complete training epoch with comprehensive monitoring.

        This method runs a full training epoch, handling data loading, forward/backward
        passes, loss computation, optimizer updates, and comprehensive logging. It
        supports distributed training, gradient accumulation, and real-time monitoring
        of training progress through loss tracking and image visualization.

        Args:
            pbar (tqdm): Progress bar for training monitoring
            loader (torch.utils.data.DataLoader): Data loader for training samples
            profiler (torch.profiler.profile): PyTorch profiler for performance monitoring

        Note:
            The method handles cross-rendering scenarios, periodic evaluation,
            checkpoint saving, and image monitoring. It integrates with the
            Hugging Face Accelerate framework for distributed training and
            automatic mixed precision.
        """
        # Set model to training mode
        self.model.train()

        local_step_losses = []
        global_step_losses = []

        logger.debug(f"======== Starting epoch {self.current_epoch} ========")
        loss_disc = None

        for idx, data in enumerate(loader):
            data["source_rgbs"] = data["source_rgbs"].to(self.weight_dtype)
            if (
                self.has_disc
                and hasattr(self.cfg.model.disc, "cross_render")
                and self.cfg.model.disc.cross_render
            ):
                data = self.prepare_cross_render_data(data)
                data["real_num_view"] = 1
            else:
                data["real_num_view"] = data["render_image"].shape[1]

            logger.debug(f"======== Starting global step {self.global_step} ========")

            if not self.has_disc:
                generator_step = True
                with self.accelerator.accumulate(self.model):

                    loss_dict = self.forward_to_get_loss(data)
                    outs = loss_dict.pop("outputs")

                    # track local losses
                    loss_dict["loss_disc"] = None
                    loss_dict["loss_gen"] = None
                    _loss_keys = sorted(list(loss_dict.keys()))
                    local_step_losses.append(
                        torch.stack(
                            [
                                (
                                    loss_dict[_loss_key].detach()
                                    if loss_dict[_loss_key] is not None
                                    else torch.tensor(float("nan"), device=self.device)
                                )
                                for _loss_key in _loss_keys
                            ]
                        )
                    )
            else:
                raise NotImplementedError("Discriminator is not implemented yet.")

            # track global step
            if self.accelerator.sync_gradients:
                profiler.step()
                if generator_step:  # only use when open discrimnator
                    self.scheduler.step()
                    self.model_step(self.global_step)  # update model
                if self.has_disc and (not generator_step):
                    self.scheduler_disc.step()

                logger.debug(f"======== Scheduler step ========")
                self.global_step += 1  # default global step starts from 0

                # accumulate local step losses  using accelerator
                global_step_loss = (
                    self.accelerator.gather(torch.stack(local_step_losses))
                    .mean(dim=0)
                    .cpu()
                )

                loss_list = global_step_loss.unbind()
                loss_keys = sorted(list(loss_dict.keys()))
                loss_kwargs = dict()
                for loss, _loss_key in zip(loss_list, loss_keys):
                    loss_kwargs[_loss_key] = loss.item()

                self.log_scalar_kwargs(
                    step=self.global_step, split="train", **loss_kwargs
                )
                self.log_optimizer(
                    step=self.global_step, attrs=["lr"], group_ids=[0, 1]
                )
                local_step_losses = []
                global_step_losses.append(global_step_loss)

                # manage display
                pbar.update(1)
                description = {
                    **loss_kwargs,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
                description = "[TRAIN STEP]" + ", ".join(
                    f"{k}={tqdm.format_num(v)}"
                    for k, v in description.items()
                    if not math.isnan(v)
                )
                pbar.set_description(description)

                # periodic actions
                if self.global_step % self.cfg.saver.checkpoint_global_steps == 0:
                    self.save_checkpoint()
                if self.global_step % self.cfg.val.global_step_period == 0:
                    self.evaluate()
                    self.model.train()
                    if self.has_disc:
                        self.model_disc.train()

                if (
                    self.global_step % self.cfg.logger.image_monitor.train_global_steps
                    == 0
                ):
                    # RGB
                    src_rgbs = data["source_rgbs"][:, 0:1]
                    render_results = outs["comp_rgb"]

                    _, _, _, r_h, r_w = render_results.shape
                    src_rgbs = F.interpolate(
                        rearrange(src_rgbs, "b 1 c h w -> (b) c h w"),
                        size=(r_h, r_w),
                        mode="bilinear",
                        align_corners=False,
                    )
                    src_rgbs = rearrange(src_rgbs, "b c h w -> b 1 c h w")
                    vis_render_imgs = torch.cat(
                        [src_rgbs.detach().cpu(), render_results], dim=1
                    )
                    vis_gt_imgs = torch.cat([src_rgbs, data["render_image"]], dim=1)

                    # accumulate_list
                    show_accumulate_imgs = dict(
                        rgb=dict(
                            inputs=vis_render_imgs, targets=vis_gt_imgs.detach().cpu()
                        ),  # rgb
                        mask=dict(
                            inputs=outs["comp_mask"].detach().cpu(),
                            targets=data["render_mask"].detach().cpu(),
                        ),  # mask
                    )

                    self.log_images_monitor(
                        step=self.global_step,
                        split="train",
                        results=show_accumulate_imgs,
                    )

                # progress control
                if self.global_step >= self.N_max_global_steps:
                    self.accelerator.set_trigger()
                    break

        # track epoch
        self.current_epoch += 1
        epoch_losses = torch.stack(global_step_losses).mean(dim=0)
        epoch_losses_list = epoch_losses.unbind()
        epoch_loss_dict = {}

        loss_keys = sorted(list(loss_dict.keys()))

        for loss, _loss_key in zip(epoch_losses_list, loss_keys):
            epoch_loss_dict[_loss_key] = loss.item()

        self.log_scalar_kwargs(
            epoch=self.current_epoch,
            split="train",
            **epoch_loss_dict,
        )
        logger.info(
            f"[TRAIN EPOCH] {self.current_epoch}/{self.cfg.train.epochs}: "
            + ", ".join(
                f"{k}={tqdm.format_num(v)}"
                for k, v in epoch_loss_dict.items()
                if not math.isnan(v)
            )
        )

    def train(self):

        starting_local_step_in_epoch = (
            self.global_step_in_epoch * self.cfg.train.accum_steps
        )
        skipped_loader = self.accelerator.skip_first_batches(
            self.train_loader, starting_local_step_in_epoch
        )
        logger.info(
            f"======== Skipped {starting_local_step_in_epoch} local batches ========"
        )
        # after resume model, initialize current hyper values
        self.model_step(self.global_step)

        with tqdm(
            range(0, self.N_max_global_steps),
            initial=self.global_step,
            disable=(not self.accelerator.is_main_process),
        ) as pbar:

            profiler = (
                torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    schedule=torch.profiler.schedule(
                        wait=10,
                        warmup=10,
                        active=100,
                    ),
                    on_trace_ready=torch.profiler.tensorboard_trace_handler(
                        os.path.join(
                            self.cfg.logger.tracker_root,
                            self.cfg.experiment.parent,
                            self.cfg.experiment.child,
                        )
                    ),
                    record_shapes=True,
                    profile_memory=True,
                    with_stack=True,
                )
                if self.cfg.logger.enable_profiler
                else DummyProfiler()
            )

            with profiler:
                self.optimizer.zero_grad()
                if self.has_disc:
                    self.optimizer_disc.zero_grad()
                for _ in range(self.current_epoch, self.cfg.train.epochs):

                    loader = skipped_loader or self.train_loader
                    skipped_loader = None
                    self.train_epoch(pbar=pbar, loader=loader, profiler=profiler)
                    if self.accelerator.check_trigger():
                        break

            logger.info(
                f"======== Training finished at global step {self.global_step} ========"
            )

            # final checkpoint and evaluation
            self.save_checkpoint()
            self.evaluate()

    @torch.no_grad()
    @torch.compiler.disable
    def evaluate(self, epoch: int = None):
        """
        Perform comprehensive evaluation on validation dataset.

        This method evaluates the model's performance on the validation dataset
        without gradient computation. It computes all training losses for monitoring
        and logs comprehensive metrics including PSNR, perceptual losses, and
        geometric regularization losses. The method also generates visualization
        images for qualitative assessment.

        Args:
            epoch (int, optional): Current epoch number for logging purposes

        Note:
            The method uses torch.no_grad() to disable gradient computation for
            memory efficiency and torch.compiler.disable to avoid compilation
            issues during evaluation. It handles distributed evaluation and
            comprehensive logging of validation metrics.
        """
        # Set model to evaluation mode
        self.model.eval()

        max_val_batches = self.cfg.val.debug_batches or len(self.val_loader)
        running_losses = []
        sample_data, sample_outs = None, None

        for data in tqdm(
            self.val_loader,
            disable=(not self.accelerator.is_main_process),
            total=max_val_batches,
        ):
            data["source_rgbs"] = data["source_rgbs"].to(self.weight_dtype)
            if (
                self.has_disc
                and hasattr(self.cfg.model.disc, "cross_render")
                and self.cfg.model.disc.cross_render
            ):
                data = self.prepare_cross_render_data(data)
                data["real_num_view"] = 1
            else:
                data["real_num_view"] = data["render_image"].shape[1]

            if len(running_losses) >= max_val_batches:
                logger.info(
                    f"======== Early stop validation at {len(running_losses)} batches ========"
                )
                break

            loss_dict = self.forward_loss_local_step(data)

            outs = loss_dict.pop("outputs")
            sample_data, sample_outs = data, outs

            _loss_keys = sorted(list(loss_dict.keys()))

            # loss accumulate
            running_losses.append(
                torch.stack(
                    [
                        (
                            loss_dict[_loss_key].detach()
                            if loss_dict[_loss_key] is not None
                            else torch.tensor(float("nan"), device=self.device)
                        )
                        for _loss_key in _loss_keys
                    ]
                )
            )

            # log each step
            src_rgbs = sample_data["source_rgbs"][:, :1]
            render_results = sample_outs["comp_rgb"]
            _, _, _, r_h, r_w = render_results.shape
            src_rgbs = F.interpolate(
                rearrange(src_rgbs, "b 1 c h w -> (b) c h w"),
                size=(r_h, r_w),
                mode="bilinear",
                align_corners=False,
            )
            src_rgbs = rearrange(src_rgbs, "b c h w -> b 1 c h w")

            sample_vis_render_imgs = torch.cat(
                [src_rgbs.detach().cpu(), render_results], dim=1
            )
            sample_vis_gt_imgs = torch.cat(
                [src_rgbs.detach().cpu(), sample_data["render_image"].detach().cpu()],
                dim=1,
            )

            # accumulate_list
            show_accumulate_imgs = dict(
                rgb=dict(
                    inputs=sample_vis_render_imgs, targets=sample_vis_gt_imgs
                ),  # rgb
            )

            self.log_eval_images_monitor(
                step=self.global_step,
                split="val",
                results=show_accumulate_imgs,
                rank_id=self.accelerator.process_index,
                eval_id=len(running_losses),
            )

        total_losses = (
            self.accelerator.gather(torch.stack(running_losses)).mean(dim=0).cpu()
        )

        total_loss_list = total_losses.unbind()
        loss_keys = sorted(list(loss_dict.keys()))
        total_loss_dict = dict()

        for loss, _loss_key in zip(total_loss_list, loss_keys):
            total_loss_dict[_loss_key] = loss.item()

        if epoch is not None:
            self.log_scalar_kwargs(
                epoch=epoch,
                split="val",
                **total_loss_dict,
            )
            logger.info(
                f"[VAL EPOCH] {epoch}/{self.cfg.train.epochs}: "
                + ", ".join(
                    f"{k}={tqdm.format_num(v)}"
                    for k, v in total_loss_dict.items()
                    if not math.isnan(v)
                )
            )
        else:
            self.log_scalar_kwargs(
                step=self.global_step,
                split="val",
                **total_loss_dict,
            )
            logger.info(
                f"[VAL STEP] {self.global_step}/{self.N_max_global_steps}: "
                + ", ".join(
                    f"{k}={tqdm.format_num(v)}"
                    for k, v in total_loss_dict.items()
                    if not math.isnan(v)
                )
            )

    def log_pair_image_monitor_each_process(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        renders: torch.Tensor = None,
        gts: torch.Tensor = None,
        prefix=None,
    ):
        M = renders.shape[1]

        merged = torch.stack([renders, gts], dim=1).view(-1, *renders.shape[2:])
        renders, gts = renders.view(-1, *renders.shape[2:]), gts.view(
            -1, *gts.shape[2:]
        )
        renders, gts, merged = (
            make_grid(renders, nrow=M),
            make_grid(gts, nrow=M),
            make_grid(merged, nrow=M),
        )
        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f"{split}" if split else "ImageSplit"
        split = split + prefix if prefix is not None else split

        self.log_images_each_process(
            {
                f"render/{split}": renders.unsqueeze(0),
                f"gt/{split}": gts.unsqueeze(0),
                f"merge/{split}": merged.unsqueeze(0),
            },
            log_progress,
        )

    def log_pair_image_monitor(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        renders: torch.Tensor = None,
        gts: torch.Tensor = None,
        prefix=None,
    ):
        self.log_pair_image_monitor_each_process(
            epoch, step, split, renders, gts, prefix
        )

    def log_image_monitor(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        renders: torch.Tensor = None,
        prefix=None,
    ):
        self.log_image_monitor_each_process(epoch, step, split, renders, prefix)

    def log_image_monitor_each_process(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        renders: torch.Tensor = None,
        prefix=None,
    ):

        M = renders.shape[1]
        renders = renders.view(-1, *renders.shape[2:])
        renders = make_grid(renders, nrow=M)

        log_type, log_progress = self._get_str_progress(epoch, step)
        split = f"{split}" if split else "ImageSplit"
        split = split + prefix if prefix is not None else split
        self.log_images_each_process(
            {
                f"rendered/{split}": renders.unsqueeze(0),
            },
            log_progress,
        )

    @Trainer.control("on_main_process")
    def log_images_monitor(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        results=None,
    ):
        """
        Log training/validation images for monitoring and visualization.

        This method logs images during training and validation for visual monitoring
        of model performance. It supports both single image logging and paired
        image logging (predictions vs ground truth) with configurable sample limits
        and automatic prefixing for different image types.

        Args:
            epoch (int, optional): Current epoch number for logging
            step (int, optional): Current training step for logging
            split (str, optional): Dataset split (train/val) for logging
            results (dict): Dictionary containing image results with structure:
                - key: Image type identifier (e.g., 'rgb', 'mask')
                - value: Dict with 'inputs' and optionally 'targets' tensors

        Note:
            This method is decorated with @Trainer.control("on_main_process") to
            ensure logging only occurs on the main process in distributed training.
            It automatically handles sample limiting and prefixes for different
            image types in the monitoring system.
        """
        for key in results.keys():
            result = results[key]
            inputs = result["inputs"][: self.cfg.logger.image_monitor.samples_per_log]
            targets = result["targets"]

            if targets is None:
                # Log single images (predictions only)
                self.log_image_monitor(
                    step=step, split=split, renders=inputs, prefix=f"_{key}"
                )
            else:
                # Log paired images (predictions vs ground truth)
                targets = targets[: self.cfg.logger.image_monitor.samples_per_log]
                self.log_pair_image_monitor(
                    step=step,
                    split=split,
                    renders=inputs,
                    gts=targets,
                    prefix=f"_{key}",
                )

    def log_eval_images_monitor(
        self,
        epoch: int = None,
        step: int = None,
        split: str = None,
        results=None,
        rank_id=None,
        eval_id=None,
    ):
        """
        Log evaluation images with distributed training support.

        This method logs images during evaluation with support for distributed
        training by including rank and evaluation identifiers in the image names.
        This allows tracking of evaluation results across different processes
        and evaluation steps in distributed training scenarios.

        Args:
            epoch (int, optional): Current epoch number for logging
            step (int, optional): Current training step for logging
            split (str, optional): Dataset split (train/val) for logging
            results (dict): Dictionary containing image results
            rank_id (int, optional): Process rank ID for distributed training
            eval_id (int, optional): Evaluation step identifier

        Note:
            The method adds rank and evaluation identifiers to image names
            to distinguish between different processes and evaluation steps
            in distributed training scenarios. This ensures proper tracking
            and organization of evaluation results.
        """
        for key in results.keys():
            result = results[key]
            inputs = result["inputs"][: self.cfg.logger.image_monitor.samples_per_log]
            targets = result["targets"]

            if targets is None:
                # Log single images with distributed training identifiers
                self.log_image_monitor(
                    step=step,
                    split=split,
                    renders=inputs,
                    prefix=f"_{key}_eval_{eval_id:02d}_rank_{rank_id:03d}",
                )
            else:
                # Log paired images with distributed training identifiers
                targets = targets[: self.cfg.logger.image_monitor.samples_per_log]
                self.log_pair_image_monitor(
                    step=step,
                    split=split,
                    renders=inputs,
                    gts=targets,
                    prefix=f"_{key}_eval_{eval_id:02d}_rank_{rank_id:03d}",
                )
