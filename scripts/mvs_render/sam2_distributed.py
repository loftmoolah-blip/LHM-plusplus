# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Distributed SAM2 segmentation pipeline

import sys

sys.path.append(".")

import os
import pdb
import time
import traceback

import cv2
import numpy as np
from tqdm import tqdm

from engine.SegmentAPI.SAM import SAM2Seg, erode_and_dialted, eroded, fill_mask


def basename(path):

    pre_name = os.path.basename(path).split(".")[0]

    return pre_name


def multi_process(worker, items, **kwargs):
    """
    worker worker function to process items
    """

    nodes = kwargs["nodes"]
    dirs = kwargs["dirs"]

    bucket = int(
        np.ceil(len(items) / nodes)
    )  # avoid  last node  process too many items.

    print("Total Nodes:", nodes)
    print("Save Path:", dirs)
    rank = int(os.environ.get("RANK", 0))

    print("Current Rank:", rank)

    kwargs["RANK"] = rank

    if rank == nodes - 1:
        output_dir = worker(items[bucket * rank :], **kwargs)
    else:
        output_dir = worker(items[bucket * rank : bucket * (rank + 1)], **kwargs)

    if rank == 0 and nodes > 1:
        timesleep = int(kwargs.get("sleep", 7200))
        time.sleep(timesleep)  # one hour


def is_img(image_path):
    return os.path.isfile(image_path) and image_path.lower().endswith(
        (".png", ".jpg", ".jpeg", ".gif")
    )


def run_seg(items, **params):
    output_dir = params["dirs"]
    debug = params["debug"]
    model = params["model"]
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        items = items[:5]

    process_valid = []

    for item in tqdm(items, desc="Processing..."):

        if not os.path.isdir(item):
            continue

        print(f"processing img dir {item}")
        basename_folder = basename(item)
        files = os.listdir(item)
        files = [os.path.join(item, file) for file in files]

        img_files = list(filter(lambda x: is_img(x), files))

        save_folder = os.path.join(output_dir, basename_folder)
        os.makedirs(os.path.join(output_dir, basename_folder), exist_ok=True)

        try:
            for img_file in img_files:
                base_file_name = basename(img_file)

                out = model(img_path=img_file, bbox=None)
                alpha = fill_mask(out.masks)
                alpha = erode_and_dialted(
                    (alpha * 255).astype(np.uint8), kernel_size=3, iterations=3
                )
                save_path = os.path.join(save_folder, base_file_name + ".jpg")
                cv2.imwrite(save_path, alpha)
        except:
            continue

        process_valid.append(item)

    return output_dir


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True, help="input path")
    parser.add_argument("-o", "--output", required=True, help="output path")
    parser.add_argument("--nodes", default=1, type=int, help="how many workload?")
    parser.add_argument("--debug", action="store_true", help="debug tag")
    parser.add_argument("--txt", default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_parse()

    # catch avaliable items

    if opt.txt == None:
        available_items = os.listdir(opt.input)
        available_items = [os.path.join(opt.input, item) for item in available_items]
    else:
        available_items = []
        with open(opt.txt) as reader:
            for line in reader:
                available_items.append(line.strip())

            available_items = [
                os.path.join(opt.input, item.split(".")[0]) for item in available_items
            ]

    prior_seg = SAM2Seg(wo_supres=False)

    multi_process(
        worker=run_seg,
        items=available_items,
        dirs=opt.output,
        nodes=opt.nodes,
        debug=opt.debug,
        model=prior_seg,
    )
