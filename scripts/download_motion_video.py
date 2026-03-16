#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download motion_video assets if not present locally.

Behavior:
- Check whether ./motion_video exists at the project root (or under --save-dir)
  and contains at least one non-hidden subdirectory;
- If missing (or --force is specified), download the LHMPP assets to a local
  cache and extract motion_video.tar into the target directory (project root
  by default).

Download strategy:
- Try HuggingFace first (huggingface_hub.snapshot_download);
- If HuggingFace is unavailable or fails, automatically fall back to
  ModelScope (modelscope.snapshot_download).

Dependencies:
- HuggingFace Hub: pip install huggingface_hub
- ModelScope: pip install modelscope

Usage:
    python scripts/download_motion_video.py
    python scripts/download_motion_video.py --save-dir .      # project root (default)
    python scripts/download_motion_video.py --force           # force re-download
"""

import argparse
import os
import shutil
import sys
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MODEL_ID = "Damo_XR_Lab/LHMPP-Assets"  # ModelScope asset repo ID
# Default HuggingFace asset repo ID (change to your actual repo if needed)
HF_MODEL_ID = "3DAIGC/LHMPP-Assets"


def has_motion_video(motion_dir: str) -> bool:
    """Check if motion_video dir exists and contains at least one case subdir."""
    if not os.path.isdir(motion_dir):
        return False
    subdirs = [
        d
        for d in os.listdir(motion_dir)
        if os.path.isdir(os.path.join(motion_dir, d)) and not d.startswith(".")
    ]
    return len(subdirs) > 0


def _download_assets(cache_dir: str) -> str:
    """
    Try to download motion assets.
    1) Try HuggingFace (huggingface_hub.snapshot_download)
    2) If unavailable or failed, fall back to ModelScope (modelscope.snapshot_download)
    """
    # 1. Try HuggingFace
    try:
        from huggingface_hub import snapshot_download as hf_snapshot

        print(f"Trying HuggingFace repo: {HF_MODEL_ID}")
        downloaded = hf_snapshot(repo_id=HF_MODEL_ID, cache_dir=cache_dir)
        print(f"HuggingFace assets downloaded to: {downloaded}")
        return downloaded
    except Exception as e:
        print(
            f"HuggingFace download failed or not available ({HF_MODEL_ID}): {e}\n"
            f"Falling back to ModelScope repo: {MODEL_ID}"
        )

    # 2. Fall back to ModelScope
    try:
        from modelscope import snapshot_download as ms_snapshot
    except ImportError:
        print("modelscope is required. Install with: pip install modelscope")
        raise ImportError("modelscope is required for motion_video download") from None

    cache_dir = os.path.join(ROOT, ".cache", "motion_assets")
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Downloading {MODEL_ID} to cache ...")
    downloaded = ms_snapshot(MODEL_ID, cache_dir=cache_dir)
    print(f"ModelScope assets downloaded to: {downloaded}")
    return downloaded


def motion_video_check(save_dir: str = ".", force: bool = False) -> None:
    """Check if motion_video exists; if not, download assets and extract motion_video."""
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ROOT, save_dir)
    target_motion = os.path.normpath(os.path.join(save_dir, "motion_video"))

    if has_motion_video(target_motion) and not force:
        return

    print("Motion video not found or invalid. Downloading...")

    cache_dir = os.path.join(ROOT, ".cache", "motion_assets")
    os.makedirs(cache_dir, exist_ok=True)
    downloaded = _download_assets(cache_dir)

    # Find and extract motion_video.tar
    tar_path = None
    for root, _, files in os.walk(downloaded):
        for f in files:
            if f.endswith(".tar") and "motion" in f.lower():
                tar_path = os.path.join(root, f)
                break
        if tar_path:
            break
    if not tar_path:
        # Fallback: motion_video/ dir directly
        src_motion = os.path.join(downloaded, "motion_video")
        if os.path.isdir(src_motion):
            os.makedirs(os.path.dirname(target_motion), exist_ok=True)
            if os.path.exists(target_motion):
                shutil.rmtree(target_motion)
            shutil.copytree(src_motion, target_motion)
            print("Motion video ready.")
            return
        raise FileNotFoundError(
            f"Model {MODEL_ID} does not contain motion_video.tar or motion_video/. "
            f"Downloaded to: {downloaded}"
        )

    os.makedirs(os.path.dirname(target_motion), exist_ok=True)
    print(f"Extracting {os.path.basename(tar_path)} to {os.path.dirname(target_motion)} ...")
    with tarfile.open(tar_path, "r") as tf:
        tf.extractall(os.path.dirname(target_motion))
    print("Motion video ready.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download motion_video from ModelScope model if not present."
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Parent directory for motion_video (default: . = project root)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if motion_video exists",
    )
    args = parser.parse_args()

    save_dir = args.save_dir
    if not os.path.isabs(save_dir):
        save_dir = os.path.join(ROOT, save_dir)
    target_motion = os.path.normpath(os.path.join(save_dir, "motion_video"))

    if has_motion_video(target_motion) and not args.force:
        print(f"motion_video already exists at {target_motion}, skip.")
        return

    try:
        motion_video_check(save_dir=args.save_dir, force=args.force)
    except Exception as e:
        print(f"Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
