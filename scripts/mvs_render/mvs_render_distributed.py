# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-08-31 10:02:15
# @Function      : Distributed MVS rendering pipeline

import sys

sys.path.append(".")

import os
import time
import traceback

import numpy as np
from tqdm import tqdm

from engine.MVSRender.mvs_render import MVSRender


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
        sleep_time = int(kwargs.get("timesleep", 3600))
        time.sleep(sleep_time)  # one hour


def run_mvsrender(items, **params):
    output_dir = params["dirs"]
    debug = params["debug"]
    os.makedirs(output_dir, exist_ok=True)

    if debug:
        items = items[:5]

    process_valid = []

    for item in tqdm(items, desc="Processing..."):
        if ".ply" not in item:
            continue

        ply_basename = basename(item)
        rendering_save_path = os.path.join(output_dir, ply_basename)

        if not debug and os.path.exists(rendering_save_path):
            continue

        try:
            # load ply
            renderer = MVSRender(item, radius=2.5)  # Radius is the rendering distance.

            # rendering ply
            renderer.rendering(rendering_save_path)
            print(f"rendering: {rendering_save_path}")
        except:
            traceback.print_exc()

        process_valid.append(item)

    return output_dir


def get_parse():
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-i", "--input", required=True, help="input path")
    parser.add_argument("-o", "--output", required=True, help="output path")
    parser.add_argument("--nodes", default=1, type=int, help="how many workload?")
    parser.add_argument("--debug", action="store_true", help="debug tag")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    opt = get_parse()

    # catch avaliable items
    available_items = os.listdir(opt.input)
    available_items = [os.path.join(opt.input, item) for item in available_items]

    multi_process(
        worker=run_mvsrender,
        items=available_items,
        dirs=opt.output,
        nodes=opt.nodes,
        debug=opt.debug,
    )
