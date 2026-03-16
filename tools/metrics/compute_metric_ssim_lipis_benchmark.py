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
    """Pad an image onto a larger centered canvas with a white background.

    The canvas is enlarged by a multiplicative factor of ``1 + pad_ratio``
    along both spatial dimensions, and the original image is placed at the
    center of the new canvas. The background is filled with 255 (white).

    Parameters
    ----------
    img_np : numpy.ndarray
        Input image array of shape ``(H, W, C)``.
    pad_ratio : float, optional
        Relative padding ratio applied to both height and width. For example,
        ``pad_ratio = 0.2`` yields a 20% enlargement.

    Returns
    -------
    tuple
        A 3-tuple ``(img_pad_np, offset_w, offset_h)`` where ``img_pad_np`` is
        the padded image (uint8), and ``offset_w``/``offset_h`` are the
        column/row offsets where the original image begins on the padded canvas.
    """

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
    """Recursively scan a directory for files with specific postfixes.

    This function walks through the directory tree and collects files whose
    extensions match the provided ``postfix`` set. A progress bar can be
    supplied or created internally to visualize the scanning process.

    Parameters
    ----------
    directory : str
        Root directory to scan.
    postfix : set or None, optional
        A set of file extensions to include (e.g., ``{'.jpg', '.png'}``). If
        ``None``, all files are included.
    progress_bar : tqdm.tqdm or None, optional
        External progress bar instance to update during scanning. If ``None``,
        a new one is created.

    Returns
    -------
    list
        A list of ``os.DirEntry`` objects corresponding to the matched files.
    """
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


class EvalDataset(Dataset):
    """Dataset for evaluating image metrics given GT and prediction folders.

    The dataset aligns prediction images with their corresponding ground-truth
    images by extracting an identifier from filenames (first contiguous digit
    sequence). Images are resized to a fixed height while preserving aspect
    ratio and returned as tensors in ``[0, 1]``.
    """

    def __init__(self, gt_folder, pred_folder, height=1024):
        """Initialize the evaluation dataset.

        Parameters
        ----------
        gt_folder : str
            Path to the folder containing ground-truth images.
        pred_folder : str
            Path to the folder containing predicted images.
        height : int, optional
            Target height for resizing images while maintaining aspect ratio.
        """
        self.gt_folder = gt_folder
        self.pred_folder = pred_folder
        self.height = height
        self.data = self.prepare_data()
        self.to_tensor = transforms.ToTensor()

    def extract_id_from_filename(self, filename):
        """Extract the first 8-digit identifier from a filename.

        The method searches for the first numeric character and returns the
        subsequent 8 characters assuming a fixed-length identifier.

        Parameters
        ----------
        filename : str
            The filename from which to extract the identifier.

        Returns
        -------
        str
            The extracted 8-character identifier.

        Raises
        ------
        AssertionError
            If no numeric character can be found in the filename.
        """
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
        """Prepare aligned GT-prediction file path pairs for evaluation.

        Ground-truth files are indexed by extracted identifiers, and prediction
        files are matched accordingly. Files containing the substring
        ``'visualization'`` are excluded from predictions.

        Returns
        -------
        list of tuple
            A list of tuples ``(gt_path, pred_path)``.
        """
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
        """Resize an image to the target height while preserving aspect ratio.

        Parameters
        ----------
        img : PIL.Image.Image
            The input image to resize.

        Returns
        -------
        PIL.Image.Image
            The resized image with height equal to ``self.height``.
        """
        w, h = img.size
        new_w = int(w * self.height / h)
        return img.resize((new_w, self.height), Image.LANCZOS)

    def __len__(self):
        """Return the number of aligned image pairs in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieve a pair of tensors corresponding to (GT, prediction).

        Images are loaded, resized to the target height, and converted to
        tensors in the range ``[0, 1]``.

        Parameters
        ----------
        idx : int
            Index of the pair to fetch.

        Returns
        -------
        tuple of torch.Tensor
            A tuple ``(gt, pred)`` with shapes ``(C, H, W)``.
        """
        gt_path, pred_path = self.data[idx]

        gt, pred = self.resize(Image.open(gt_path)), self.resize(Image.open(pred_path))
        if gt.height != self.height:
            gt = self.resize(gt)
        if pred.height != self.height:
            pred = self.resize(pred)
        gt = self.to_tensor(gt)
        pred = self.to_tensor(pred)
        return gt, pred


def copy_resize_gt(gt_folder, height):
    """Create a resized copy of GT images to match a target height.

    A new sibling folder named ``resize_{height}`` is created (if absent) and
    ground-truth images are copied into it after resizing the canvas if needed.

    Parameters
    ----------
    gt_folder : str
        Directory containing original ground-truth images.
    height : int
        Target height that GT images should match.

    Returns
    -------
    str
        Path to the folder containing the resized GT images.
    """
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
        # img, _, _ = img_center_padding(img)
        img = Image.fromarray(img)
        w, h = img.size
        img.save(os.path.join(new_folder, file))
    return new_folder


@torch.no_grad()
def ssim(dataloader):
    """Compute the mean Structural Similarity (SSIM) over a dataloader.

    The function expects batches of pairs ``(gt, pred)`` in ``[0, 1]`` and
    accumulates the SSIM score weighted by batch size.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding batches of ground-truth and prediction tensors.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the mean SSIM over the entire dataset.
    """
    ssim_score = 0
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
    for gt, pred in tqdm(dataloader, desc="Calculating SSIM"):
        batch_size = gt.size(0)
        gt, pred = gt.to("cuda"), pred.to("cuda")
        ssim_score += ssim(pred, gt) * batch_size
    return ssim_score / len(dataloader.dataset)


@torch.no_grad()
def lpips(dataloader):
    """Compute the mean LPIPS over a dataloader using a specified backbone.

    Images are converted from ``[0, 1]`` to ``[-1, 1]`` as required by LPIPS.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader yielding batches of ground-truth and prediction tensors.

    Returns
    -------
    torch.Tensor
        A scalar tensor containing the mean LPIPS over the entire dataset.
    """
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
    """Evaluate SSIM and LPIPS for a pair of folders (predictions vs GT).

    If the image heights in the two folders differ, ground-truth images are
    temporarily copied/resized to match the prediction height. Results are
    computed over aligned pairs, and any temporary folder is removed at the end.

    Parameters
    ----------
    pred_folder : str
        Directory containing prediction images.
    gt_folder : str
        Directory containing ground-truth images.

    Returns
    -------
    tuple of float
        A tuple ``(ssim, lpips)`` with scalar metric values.
    """
    # Check gt_folder has images with target height, resize if not
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


def get_parse():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-f1", "--folder1", type=str, required=True, help="gt folder")
    parser.add_argument(
        "-f2", "--folder2", type=str, required=True, help="target folder"
    )
    parser.add_argument("-v", "--view", type=int, help="view inputs")
    parser.add_argument("--pre", type=str, default="")
    parser.add_argument("--pad", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    opt = get_parse()

    input_folder = opt.folder1
    target_folder = opt.folder2
    view = opt.view

    target_folder = target_folder[:-1] if target_folder[-1] == "/" else target_folder
    target_key = target_folder.split("/")[-2:]

    save_folder = os.path.join(
        f"./exps/metrics_{opt.pre}", "lpips_results", *target_key, f"view_{view:03d}"
    )
    os.makedirs(save_folder, exist_ok=True)

    items = os.listdir(input_folder)
    items = sorted(items)

    results_dict = defaultdict(dict)
    lpips_list = []
    ssim_list = []

    for item in items:
        if item == "helge_crop":
            continue

        input_item_folder = os.path.join(input_folder, item)
        target_item_folder = os.path.join(
            target_folder, item, f"view_{view:03d}", "pred"
        )

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

    write_json(os.path.join(save_folder, f"lpips_ssim_{view:03d}.json"), results_dict)

    print("save path:", os.path.join(save_folder, f"lpips_ssim_{view:03d}.json"))

    print(f"final lpips: {np.mean(lpips_list)}")
    print(f"final ssim: {np.mean(ssim_list)}")
