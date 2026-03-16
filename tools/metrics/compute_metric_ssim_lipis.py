# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-03-03 10:28:47
# @Function      : easy to use SSIM and LPIPS metric

import os
import pdb
import shutil
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from prettytable import PrettyTable
from torch.utils.data import Dataset
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchvision import transforms
from tqdm import tqdm


class EvalDataset(Dataset):
    def __init__(self, gt_folder, pred_folder, height=1024):
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.height = height
        self.data = self.prepare_data()
        self.to_tensor = transforms.ToTensor()

    def extract_id_from_filename(self, filename):
        # find first number in filename
        start_i = None
        for i, c in enumerate(filename):
            if c.isdigit():
                start_i = i
                break
        if start_i is None:
            assert False, f"Cannot find number in filename {filename}"
        return filename[start_i : start_i + 8]

    def prepare_data(self):
        gt_files = scan_files_in_dir(self.gt_folder, postfix={".jpg", ".png"})

        gt_dict = {self.extract_id_from_filename(file.name): file for file in gt_files}
        pred_files = scan_files_in_dir(self.pred_folder, postfix={".jpg", ".png"})

        pred_files = list(filter(lambda x: "visualization" not in x.name, pred_files))

        tuples = []
        for pred_file in pred_files:
            pred_id = self.extract_id_from_filename(pred_file.name)
            if pred_id not in gt_dict:
                print(f"Cannot find gt file for {pred_file}")
            else:
                tuples.append((gt_dict[pred_id].path, pred_file.path))
        return tuples

    def resize(self, img):
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        gt_path, pred_path = self.data[idx]

        gt, pred = self.resize(Image.open(gt_path)), self.resize(Image.open(pred_path))
        if gt.height != self.height:
            gt = self.resize(gt)
        if pred.height != self.height:
            pred = self.resize(pred)
        gt = self.to_tensor(gt)
        pred = self.to_tensor(pred)
        return gt, pred


def get_parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--folder1", type=str, required=True)
    parser.add_argument("-f2", "--folder2", type=str, required=True)
    parser.add_argument("--pre", type=str, default="")
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


def write_json(path, x):
    """write a json file.

    Args:
        path (str): path to write json file.
        x (dict): dict to write.
    """
    import json

    with open(path, "w") as f:
        json.dump(x, f, indent=2)


def img_center_padding(img_np, pad_ratio=0.2):
    # Pads an input image onto a larger, centered canvas with a uniform white background.
    #
    # This function constructs a new canvas whose width and height are (1 + pad_ratio) times
    # those of the input image. The input image is placed at the center of this canvas,
    # and all remaining areas are filled with white pixels (i.e., pixel value 255 for uint8 images).
    #
    # Parameters
    # ----------
    # img_np : numpy.ndarray
    #     Input image array of shape (H, W, C), where H is the height, W is the width,
    #     and C is the number of channels. Typical dtype is uint8.
    # pad_ratio : float, optional
    #     Ratio by which each spatial dimension is increased. For example, pad_ratio=0.2
    #     will result in a canvas that is 20% larger in both height and width compared to the input image.
    #
    # Returns
    # -------
    # img_pad_np : numpy.ndarray
    #     The padded image, of the same dtype and number of channels as the input, with source
    #     content centered and surrounded by a white background.
    # offset_w : int
    #     Horizontal (column-wise) offset in the canvas at which the input image starts.
    # offset_h : int
    #     Vertical (row-wise) offset in the canvas at which the input image starts.
    #
    # Notes
    # -----
    # This procedure is commonly employed in image quality evaluation settings to standardize the
    # spatial dimensions across a dataset or to mitigate boundary effects in patch-based metrics.

    ori_w, ori_h = img_np.shape[:2]

    w = round((1 + pad_ratio) * ori_w)
    h = round((1 + pad_ratio) * ori_h)

    img_pad_np = (np.ones((w, h, 3), dtype=img_np.dtype) * 255).astype(np.uint8)
    offset_h, offset_w = (w - img_np.shape[0]) // 2, (h - img_np.shape[1]) // 2
    img_pad_np[
        offset_h : offset_h + img_np.shape[0] :, offset_w : offset_w + img_np.shape[1]
    ] = img_np

    return img_pad_np, offset_w, offset_h


def scan_files_in_dir(directory, postfix=None, progress_bar=None) -> list:
    # Scans all files in the given directory and its subdirectories, returning a list of entries.
    #
    # This function recursively traverses the directory tree rooted at the specified 'directory',
    # accumulating file entries whose extensions match the optional 'postfix' set. If a progress bar
    # instance (compatible with 'tqdm') is provided, it will be updated as each file is discovered.
    #
    # Parameters
    # ----------
    # directory : str
    #     Path to the root directory from which to begin scanning for files.
    # postfix : set of str or None, optional
    #     A set containing valid file extensions (e.g., {".jpg", ".png"}). If set to None,
    #     all files regardless of extension will be included in the results.
    # progress_bar : tqdm or None, optional
    #     An optional progress bar instance. If provided, it will display scanning progress
    #     as files are located. If None, a default 'tqdm' instance is constructed.
    #
    # Returns
    # -------
    # list of os.DirEntry
    #     A list of file entries (os.DirEntry objects) corresponding to files found in the
    #     directory tree, filtered by the supplied extension set if 'postfix' is not None.
    #
    # Notes
    # -----
    # The function is non-destructive and does not alter the filesystem. It is commonly used
    # for enumerating datasets of images or other media types within a directory hierarchy.

    file_list = []
    progress_bar = (
        tqdm(total=0, desc=f"Scanning", ncols=100)
        if progress_bar is None
        else progress_bar
    )
    for entry in os.scandir(directory):
        if entry.is_file():
            if postfix is None or os.path.splitext(entry.path)[1] in postfix:
                file_list.append(entry)
                progress_bar.total += 1
                progress_bar.update(1)
        elif entry.is_dir():
            file_list += scan_files_in_dir(
                entry.path, postfix=postfix, progress_bar=progress_bar
            )
    return file_list


def copy_resize_gt(gt_folder, height):
    # Resize all images in gt_folder so their height equals 'height', maintaining aspect ratio.
    # Copy the resized images to a new directory named "resize_{height}" alongside the original folder.
    # If the destination already contains a file with the same name, skip resizing that file.
    #
    # Parameters
    # ----------
    # gt_folder : str
    #     Directory containing ground-truth images to be resized.
    # height : int
    #     The target height to which all images will be resized, preserving aspect ratio.
    #
    # Returns
    # -------
    # new_folder : str
    #     The path to the output directory containing the resized images.

    new_folder = os.path.join(
        os.path.dirname(gt_folder[:-1] if gt_folder[-1] == "/" else gt_folder),
        f"resize_{height}",
    )
    if not os.path.exists(new_folder):
        os.makedirs(new_folder, exist_ok=True)
    for file in tqdm(os.listdir(gt_folder)):
        if os.path.exists(os.path.join(new_folder, file)):
            continue
        img = Image.open(os.path.join(gt_folder, file))
        img = np.asarray(img)
        img = Image.fromarray(img)
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        ssim_score += ssim(pred, gt) * batch_size
    return ssim_score / len(dataloader.dataset)


@torch.no_grad()
def lpips(dataloader):
    lpips_score = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to("cuda")
    score = 0
    for gt, pred in tqdm(dataloader, desc="Calculating LPIPS"):
        batch_size = gt.size(0)
        pred = pred.to("cuda")
        gt = gt.to("cuda")
        # LPIPS needs the images to be in the [-1, 1] range.
        gt = (gt * 2) - 1
        pred = (pred * 2) - 1
        score += lpips_score(gt, pred) * batch_size
    return score / len(dataloader.dataset)


def eval(pred_folder, gt_folder):
    """
    Evaluate SSIM and LPIPS metrics between predicted and ground-truth images.

    This function prepares the prediction and ground-truth folders,
    ensures that image heights match by resizing the ground-truth images if necessary,
    constructs an evaluation dataset and dataloader, computes SSIM and LPIPS
    metrics over all image pairs, and prints the results in a formatted table.

    Parameters
    ----------
    pred_folder : str
        Path to the folder containing predicted images.
    gt_folder : str
        Path to the folder containing ground-truth images.

    Returns
    -------
    ssim_ : float
        The mean Structural Similarity Index (SSIM) across all image pairs.
    lpips_ : float
        The mean Learned Perceptual Image Patch Similarity (LPIPS) across all image pairs.

    Note
    ----
    If the ground-truth images do not match the predicted images in height, this function
    creates a temporarily resized copy of the ground-truth folder and later deletes it.
    """

    pred_sample = os.listdir(pred_folder)[0]
    gt_sample = os.listdir(gt_folder)[0]

    img = Image.open(os.path.join(pred_folder, pred_sample))
    gt_img = Image.open(os.path.join(gt_folder, gt_sample))

    copy_folder = None
    if img.height != gt_img.height:
        title = "--" * 30 + "Resizing GT Images to height {img.height}" + "--" * 30
        print(title)
        gt_folder = copy_resize_gt(gt_folder, img.height)
        print("-" * len(title))
        copy_folder = gt_folder

    # Form dataset
    dataset = EvalDataset(gt_folder, pred_folder, img.height)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        num_workers=4,
        shuffle=False,
        drop_last=False,
    )

    # Calculate Metrics
    header = []
    row = []

    header += ["SSIM", "LPIPS"]
    ssim_ = ssim(dataloader).item()
    lpips_ = lpips(dataloader).item()
    row += [ssim_, lpips_]

    # Print Results
    print("GT Folder  : ", gt_folder)
    print("Pred Folder: ", pred_folder)
    table = PrettyTable()
    table.field_names = header
    table.add_row(row)

    if copy_folder is not None:
        shutil.rmtree(copy_folder)

    return ssim_, lpips_


if __name__ == "__main__":

    opt = get_parse()

    input_folder = opt.folder1
    target_folder = opt.folder2

    target_folder = target_folder[:-1] if target_folder[-1] == "/" else target_folder

    target_key = target_folder.split("/")[-2:]

    save_folder = os.path.join(
        f"./exps/metrics_{opt.pre}", "lpips_results", *target_key
    )
    os.makedirs(save_folder, exist_ok=True)

    items = os.listdir(input_folder)
    items = sorted(items)

    results_dict = defaultdict(dict)
    lpips_list = []
    ssim_list = []

    for item in items:

        input_item_folder = os.path.join(input_folder, item)
        target_item_folder = os.path.join(target_folder, item, "pred_rgb")

        if os.path.exists(input_item_folder) and os.path.exists(target_item_folder):

            ssim_, lpips_ = eval(input_item_folder, target_item_folder)

            if ssim_ == -1:
                continue

            lpips_list.append(lpips_)
            ssim_list.append(ssim_)

            results_dict[item]["lpips"] = lpips_
            results_dict[item]["ssim"] = ssim_
            if opt.debug:
                break
            print(results_dict)

    results_dict["all_mean"]["lpips"] = np.mean(lpips_list)
    results_dict["all_mean"]["ssim"] = np.mean(ssim_list)

    write_json(
        os.path.join(save_folder, f"lpips_ssim_{target_key[-1]}.json"), results_dict
    )

    print(f"final lpips: {np.mean(lpips_list)}")
    print(f"final ssim: {np.mean(ssim_list)}")
