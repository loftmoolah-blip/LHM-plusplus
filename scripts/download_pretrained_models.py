#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download prior models and LHM++ pretrained model weights to ./pretrained_models.
Skips items that already exist locally. Tries HuggingFace first, falls back to ModelScope.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.utils.model_card import (
    HuggingFace_MODEL_CARD,
    HuggingFace_Prior_MODEL_CARD,
    ModelScope_MODEL_CARD,
    ModelScope_Prior_MODEL_CARD,
)
from core.utils.model_download_utils import AutoModelQuery


def main():
    parser = argparse.ArgumentParser(
        description="Download LHM++ prior models and pretrained model weights."
    )
    parser.add_argument(
        "--prior",
        action="store_true",
        help="Download prior models only (e.g. LHMPP-Prior: human_model_files, voxel_grid, etc.)",
    )
    parser.add_argument(
        "--models",
        action="store_true",
        help="Download LHM++ pretrained model weights only (e.g. LHMPP-700M, LHMPP-700MC, LHMPPS-700M)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download both prior models and pretrained weights (default)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./pretrained_models",
        help="Directory to save downloaded models (default: ./pretrained_models)",
    )
    args = parser.parse_args()

    if not args.prior and not args.models and not args.all:
        args.all = True

    save_dir = args.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ROOT, save_dir)
    save_dir = os.path.normpath(save_dir)
    print(f"Save dir: {save_dir}")

    automodel = AutoModelQuery(save_dir=save_dir)

    if args.prior or args.all:
        prior_names = set(HuggingFace_Prior_MODEL_CARD) | set(ModelScope_Prior_MODEL_CARD)
        print(f"\n--- Downloading prior models: {prior_names} ---")
        result = automodel.download_all_prior_models()
        for name, path in result.items():
            print(f"  {name}: {path}")

    if args.models or args.all:
        main_names = set(HuggingFace_MODEL_CARD) | set(ModelScope_MODEL_CARD)
        print(f"\n--- Downloading pretrained models: {main_names} ---")
        for name in main_names:
            try:
                path = automodel.query(name)
                print(f"  {name}: {path}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
