# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-10-20 10:00:00
# @Function      : LHM++ Inference Utils Tool

import base64
import glob
import json
import os
import tempfile
from typing import Dict, List, Optional

import gradio as gr
import numpy as np
import torch
from PIL import Image

SMPLX_EXPRESSION_DIM = 100
from contextlib import contextmanager


def get_smplx_params(
    data: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Extract and prepare SMPL-X parameters for model input.

    Filters relevant SMPL-X parameter keys from the input data dictionary
    and moves them to the specified device with an added batch dimension.

    Args:
        data: Dictionary containing SMPL-X parameters and other data.
        device: Target device (e.g., 'cuda' or 'cpu') for the parameters.

    Returns:
        Dictionary containing only SMPL-X parameters with shape (1, ...)
        on the specified device.
    """
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

    smplx_params = {k: data[k].unsqueeze(0).to(device) for k in smplx_keys if k in data}

    return smplx_params


def obtain_motion_sequence(motion_dir: str) -> List[Dict[str, torch.Tensor]]:
    """Load motion sequence data from SMPL-X and FLAME parameter files.

    Reads SMPL-X parameter JSON files from the specified directory and optionally
    merges them with corresponding FLAME parameters for facial expressions and poses.

    Args:
        motion_dir: Path to directory containing SMPL-X parameter JSON files.

    Returns:
        List of dictionaries containing SMPL-X parameters as torch tensors.
        Each dictionary includes body pose, facial expressions, and eye poses.
    """
    motion_files = sorted(glob.glob(os.path.join(motion_dir, "*.json")))
    smplx_list = []

    for motion_file in motion_files:
        # Load SMPL-X parameters
        with open(motion_file, "r") as f:
            smplx_params = json.load(f)

        # Try to load corresponding FLAME parameters
        flame_path = motion_file.replace("smplx_params", "flame_params")
        if os.path.exists(flame_path):
            with open(flame_path, "r") as f:
                flame_params = json.load(f)

            # Override SMPL-X parameters with FLAME data
            smplx_params["expr"] = torch.FloatTensor(flame_params["expcode"])
            smplx_params["jaw_pose"] = torch.FloatTensor(flame_params["posecode"][3:])
            smplx_params["leye_pose"] = torch.FloatTensor(flame_params["eyecode"][:3])
            smplx_params["reye_pose"] = torch.FloatTensor(flame_params["eyecode"][3:])
        else:
            # Use zero expressions if FLAME params not available
            # smplx_params["expr"] = torch.zeros(SMPLX_EXPRESSION_DIM)
            # Using SMPLX expression
            pass

        smplx_list.append(smplx_params)

    return smplx_list


def assert_input_image(input_image: Optional[np.ndarray]) -> None:
    """Validate input image is not None.

    Args:
        input_image: Input image array to validate.

    Raises:
        gr.Error: If input image is None.
    """
    if input_image is None:
        raise gr.Error("No image selected or uploaded!")


def prepare_working_dir() -> tempfile.TemporaryDirectory:
    """Create a temporary working directory.

    Returns:
        Temporary directory object.
    """
    return tempfile.TemporaryDirectory()


def init_preprocessor() -> None:
    """Initialize the global preprocessor for image preprocessing."""
    from core.utils.preprocess import Preprocessor

    global preprocessor
    preprocessor = Preprocessor()


def preprocess_fn(
    image_in: np.ndarray,
    remove_bg: bool,
    recenter: bool,
    working_dir: tempfile.TemporaryDirectory,
) -> str:
    """Preprocess input image with optional background removal and recentering.

    Args:
        image_in: Input image as numpy array.
        remove_bg: Whether to remove background.
        recenter: Whether to recenter the subject.
        working_dir: Temporary directory for storing intermediate files.

    Returns:
        Path to the preprocessed image.

    Raises:
        AssertionError: If preprocessing fails.
    """
    image_raw = os.path.join(working_dir.name, "raw.png")
    with Image.fromarray(image_in) as img:
        img.save(image_raw)

    image_out = os.path.join(working_dir.name, "rembg.png")
    success = preprocessor.preprocess(
        image_path=image_raw, save_path=image_out, rmbg=remove_bg, recenter=recenter
    )

    if not success:
        raise RuntimeError("Preprocessing failed")

    return image_out


def get_image_base64(path: str) -> str:
    """Convert image file to base64 encoded string.

    Args:
        path: Path to image file.

    Returns:
        Base64 encoded image string with data URI prefix.
    """
    with open(path, "rb") as f:
        encoded_string = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{encoded_string}"


def get_available_device() -> str:
    """Returns the current CUDA device or ``"cpu"`` if CUDA is unavailable.

    When CUDA is available, uses the device ID from ``torch.cuda.current_device()``
    so the caller runs on the same device as the active CUDA context.

    Returns:
        str: Device string (e.g., ``"cuda:0"`` or ``"cpu"``).
    """
    if torch.cuda.is_available():
        current_device_id = torch.cuda.current_device()
        device = f"cuda:{current_device_id}"
    else:
        device = "cpu"
    return device


@contextmanager
def easy_memory_manager(model: torch.nn.Module, device: str = "cuda"):
    """Context manager that moves a model to GPU during use and back to CPU afterward.

    Reduces GPU memory footprint by transferring the model off CUDA and clearing
    the cache when the block exits. Use when running inference in memory-constrained
    environments.

    Args:
        model (torch.nn.Module): The model to manage. Will be moved to the current
            CUDA device (or CPU if unavailable) for the duration of the context.
        device (str, optional): Unused; kept for API compatibility. Default: ``"cuda"``.

    Yields:
        torch.nn.Module: The model, ready for use on the target device.

    Example:
        >>> with easy_memory_manager(model):
        ...     output = model(input_tensor)
    """
    device = get_available_device()
    model.to(device)
    try:
        yield model
    finally:
        model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
