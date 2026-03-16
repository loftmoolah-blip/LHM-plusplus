# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-03 10:28:35
# @Function      : Easy to use PSNR metric
import os
import sys

sys.path.append("./")
import math
import pdb

import cv2
import numpy as np
import skimage
import torch
from PIL import Image
from tqdm import tqdm


def write_json(path, x):
    import json

    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def img_center_padding(img_np, pad_ratio=0.2, background=1):
    """Pad an image onto a larger centered canvas with a uniform background.

    The canvas is enlarged by a multiplicative factor of ``1 + pad_ratio``
    along both spatial dimensions, and the original image is placed at the
    center of the new canvas. The background is filled with ones (white) when
    ``background == 1`` and zeros (black) when ``background == 0``.

    Parameters
    ----------
    img_np : numpy.ndarray
        Input image array of shape ``(H, W, C)``.
    pad_ratio : float, optional
        Relative padding ratio applied to both height and width. For example,
        ``pad_ratio = 0.2`` yields a 20% enlargement.
    background : int, optional
        Background fill flag: ``1`` uses ones (white), ``0`` uses zeros (black).

    Returns
    -------
    tuple
        A 3-tuple ``(img_pad_np, offset_w, offset_h)`` where ``img_pad_np`` is
        the padded image (same dtype as input), and ``offset_w``/``offset_h``
        are the column/row offsets where the original image begins on the
        padded canvas.
    """

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    if background == 1:
        img_pad_np = np.ones((w, h, 3), dtype=img_np.dtype)
    else:
        img_pad_np = np.zeros((w, h, 3), dtype=img_np.dtype)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np, offset_w, offset_h


def compute_psnr(src, tar):
    """Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.

    This routine delegates to ``skimage.metrics.peak_signal_noise_ratio`` with
    ``data_range = 1`` and therefore assumes that both inputs have been
    normalized to the range ``[0, 1]``. Higher PSNR indicates closer similarity.

    Parameters
    ----------
    src : numpy.ndarray
        Source image (e.g., reconstruction or prediction), normalized to ``[0, 1]``.
    tar : numpy.ndarray
        Target image (e.g., ground truth), normalized to ``[0, 1]``.

    Returns
    -------
    float
        PSNR value in decibels (dB).
    """
    psnr = skimage.metrics.peak_signal_noise_ratio(tar, src, data_range=1)
    return psnr


def get_parse():
    """Construct and parse command-line arguments for PSNR benchmarking.

    The parser expects mandatory folders for ground-truth and predictions,
    along with the specific view index to evaluate. Optional arguments control
    experiment prefixing, debugging behavior, and whether to pad ground-truth
    images prior to evaluation.

    Returns
    -------
    argparse.Namespace
        Parsed arguments including ``folder1`` (GT), ``folder2`` (predictions),
        ``view``, ``pre``, ``debug``, and ``pad``.
    """
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-f1", "--folder1", required=True, help="gt_folder")
    parser.add_argument("-f2", "--folder2", required=True, help="predict_folder")
    parser.add_argument("-v", "--view", default=1, type=int, help="view inputs")
    parser.add_argument("--pre", default="")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--pad", action="store_true")
    args = parser.parse_args()
    return args


def get_image_paths_current_dir(folder_path):
    """Enumerate image file paths in a directory (non-recursive, sorted).

    Files are filtered by a predefined set of common image extensions and
    returned as absolute paths sorted lexicographically.

    Parameters
    ----------
    folder_path : str
        Directory to scan for image files.

    Returns
    -------
    list of str
        Sorted list of absolute image file paths contained in the directory.
    """
    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".webp",
        ".jfif",
    }

    return sorted(
        [
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if os.path.splitext(f)[1].lower() in image_extensions
        ]
    )


def psnr_compute(
    input_data,
    results_data,
    mask_data=None,
    pad=False,
):
    """Compute the mean PSNR over paired images from two directories.

    For each filename present in the input (ground-truth) and result
    directories, this function reads the corresponding images, optionally
    applies a foreground mask (background is filled with ones), optionally
    performs centered padding on the ground-truth image, and evaluates PSNR
    per pair. The final result is the arithmetic mean of per-pair PSNR values.

    Parameters
    ----------
    input_data : str
        Directory containing ground-truth images.
    results_data : str
        Directory containing result/prediction images (for a specific view).
    mask_data : str or None, optional
        Directory containing mask images aligned with the ground-truth.
        If ``None``, no masking is applied.
    pad : bool, optional
        If ``True``, the ground-truth image is center-padded before evaluation.

    Returns
    -------
    tuple
        A 2-tuple ``(mean_psnr, per_id_psnrs)`` where ``mean_psnr`` is the
        average PSNR across all valid image pairs, and ``per_id_psnrs`` is a
        dictionary mapping the current identifier to a list of per-image PSNRs.

    Notes
    -----
    If the numbers of images in the two directories do not match, the function
    returns ``-1`` early. In that case, callers attempting tuple-unpacking
    should handle this condition to avoid errors.
    """

    gt_imgs = get_image_paths_current_dir(input_data)
    result_imgs = get_image_paths_current_dir(os.path.join(results_data))

    if mask_data is not None:
        mask_imgs = get_image_paths_current_dir(mask_data)
    else:
        mask_imgs = None

    if "visualization" in result_imgs[-1]:
        result_imgs = result_imgs[:-1]

    if len(gt_imgs) != len(result_imgs):
        return -1

    psnr_list = []
    cur_id = input_data.split("/")[-1]

    for mask_i, (gt, result) in tqdm(enumerate(zip(gt_imgs, result_imgs))):
        result_img = (cv2.imread(result, cv2.IMREAD_UNCHANGED) / 255.0).astype(
            np.float32
        )
        gt_img = (cv2.imread(gt, cv2.IMREAD_UNCHANGED) / 255.0).astype(np.float32)

        if mask_imgs is not None:
            mask_img = (
                cv2.imread(mask_imgs[mask_i], cv2.IMREAD_UNCHANGED) / 255.0
            ).astype(np.float32)
            mask_img = np.stack([mask_img] * 3, axis=-1)

        if pad:
            gt_img, _, _ = img_center_padding(gt_img)

        h, w, c = result_img.shape

        # scale_h = int(h * 512 / w)
        # gt_img = cv2.resize(gt_img, (512, scale_h), interpolation=cv2.INTER_AREA)
        # result_img = cv2.resize(
        #     result_img, (512, scale_h), interpolation=cv2.INTER_AREA
        # )

        if mask_imgs is not None:
            mask_img = cv2.resize(mask_img, (w, h), interpolation=cv2.INTER_AREA)
            gt_img = gt_img * mask_img + 1 - mask_img
            result_img = result_img * mask_img + 1 - mask_img

        psnr = compute_psnr(result_img, gt_img)

        psnr_list += [psnr]
        # Image.fromarray((gt_img * 255).astype(np.uint8)).save("gt.png")
        # Image.fromarray((result_img * 255).astype(np.uint8)).save("result.png")
    psnr = np.mean(psnr_list)

    curr_dict = {f"{cur_id}": psnr_list}

    return psnr, curr_dict


if __name__ == "__main__":

    opt = get_parse()

    input_folder = opt.folder1
    target_folder = opt.folder2
    view = opt.view
    mask_folder = None

    target_folder = target_folder[:-1] if target_folder[-1] == "/" else target_folder

    if mask_folder is not None:
        mask_folder = mask_folder[:-1] if mask_folder[-1] == "/" else mask_folder

    target_key = target_folder.split("/")[-2:]

    save_folder = os.path.join(
        f"./exps/metrics_{opt.pre}", "psnr_results", *target_key, f"view_{view:03d}"
    )
    os.makedirs(save_folder, exist_ok=True)

    items = os.listdir(input_folder)
    items = sorted(items)

    results_dict = dict()
    psnr_list = []
    psnr_metas = dict()
    cnt = 0

    for item in tqdm(items):
        if item == "helge_crop":
            continue

        input_item_folder = os.path.join(input_folder, item)
        if mask_folder is not None:
            mask_item_folder = os.path.join(mask_folder, item)
        else:
            mask_item_folder = None
        target_item_folder = os.path.join(
            target_folder, item, f"view_{view:03d}", "pred"
        )

        if os.path.exists(input_item_folder) and os.path.exists(target_item_folder):

            psnr, psnr_meta = psnr_compute(
                input_item_folder, target_item_folder, mask_item_folder, opt.pad
            )

            if psnr == -1:
                continue

            psnr_list.append(psnr)
            psnr_metas.update(psnr_meta)

            results_dict[item] = psnr
            if opt.debug and cnt == 4:
                break
        cnt += 1

    results_dict["all_mean"] = np.mean(psnr_list)
    write_json(os.path.join(save_folder, f"PSNR_{target_key[-1]}.json"), results_dict)
    write_json(
        os.path.join(save_folder, f"PSNR_{target_key[-1]}_metas.json"), psnr_metas
    )

    print(f"final psnr: {np.mean(psnr_list)}")
