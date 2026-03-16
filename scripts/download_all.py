#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click download: assets (motion_video), prior models, pretrained models.

Downloads everything needed to run the Gradio app and tests:
  1. motion_video (assets from Damo_XR_Lab/LHMPP-Assets)
  2. prior models (human_model_files, voxel_grid, BiRefNet, etc.)
  3. pretrained models (LHMPP-700M, LHMPP-700MC, LHMPPS-700M)

Skips items that already exist. Tries HuggingFace first for models, falls back to ModelScope.

Usage:
    python scripts/download_all.py
    python scripts/download_all.py --skip-asset --skip-models
    python scripts/download_all.py --save-dir ./pretrained_models --project-root .
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One-click download: assets, prior models, pretrained models."
    )
    parser.add_argument(
        "--skip-asset",
        action="store_true",
        help="Skip motion_video (assets) download",
    )
    parser.add_argument(
        "--skip-prior",
        action="store_true",
        help="Skip prior models download",
    )
    parser.add_argument(
        "--skip-models",
        action="store_true",
        help="Skip pretrained model weights download",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./pretrained_models",
        help="Directory for prior + pretrained models (default: ./pretrained_models)",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root for motion_video (default: .)",
    )
    parser.add_argument(
        "--force-asset",
        action="store_true",
        help="Re-download motion_video even if it exists",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ROOT, save_dir)
    save_dir = os.path.normpath(save_dir)

    project_root = args.project_root
    if not os.path.isabs(project_root):
        project_root = os.path.join(ROOT, project_root)
    project_root = os.path.normpath(project_root)

    print("=" * 60)
    print("LHM++ One-Click Download")
    print("=" * 60)

    # 1. Assets (motion_video)
    if not args.skip_asset:
        print("\n[1/3] Downloading assets (motion_video)...")
        try:
            from scripts.download_motion_video import motion_video_check

            motion_video_check(save_dir=project_root, force=args.force_asset)
            print("  OK: motion_video")
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)
    else:
        print("\n[1/3] Skipping assets (motion_video)")

    # 2. Prior models
    if not args.skip_prior:
        print("\n[2/3] Downloading prior models...")
        try:
            from core.utils.model_card import (
                HuggingFace_Prior_MODEL_CARD,
                ModelScope_Prior_MODEL_CARD,
            )
            from core.utils.model_download_utils import AutoModelQuery

            prior_names = set(HuggingFace_Prior_MODEL_CARD) | set(
                ModelScope_Prior_MODEL_CARD
            )
            automodel = AutoModelQuery(save_dir=save_dir)
            result = automodel.download_all_prior_models()
            for name, path in result.items():
                print(f"  {name}: {path}")
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)
    else:
        print("\n[2/3] Skipping prior models")

    # 3. Pretrained models
    if not args.skip_models:
        print("\n[3/3] Downloading pretrained model weights...")
        try:
            from core.utils.model_card import (
                HuggingFace_MODEL_CARD,
                ModelScope_MODEL_CARD,
            )
            from core.utils.model_download_utils import AutoModelQuery

            main_names = set(HuggingFace_MODEL_CARD) | set(ModelScope_MODEL_CARD)
            automodel = AutoModelQuery(save_dir=save_dir)
            for name in main_names:
                try:
                    path = automodel.query(name)
                    print(f"  {name}: {path}")
                except Exception as e:
                    print(f"  {name}: FAILED - {e}")
        except Exception as e:
            print(f"  FAILED: {e}")
            sys.exit(1)
    else:
        print("\n[3/3] Skipping pretrained models")

    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
