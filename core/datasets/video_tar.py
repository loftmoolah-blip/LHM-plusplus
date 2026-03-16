# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-06 23:56:19
# @Function      : tar baseclass, used to I/O data.

import io
import json
import os
import tarfile
import time
from typing import Any, Dict, List, Optional, Set

import cv2
import numpy as np
from PIL import Image


def read_json(path: str) -> Dict[str, Any]:
    """Load a JSON file.

    Args:
        path: Path to the JSON file.

    Returns:
        Dictionary containing the JSON content.
    """
    with open(path, "r") as f:
        return json.load(f)


def write_json(path: str, data: Dict[str, Any]) -> None:
    """Write a dictionary to a JSON file.

    Args:
        path: Path to write the JSON file.
        data: Dictionary to write.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class TarFrameDataset:
    """Dataset for reading frame data from tar archives.

    This class provides efficient access to frame data stored in tar files,
    with support for images, masks, and parameter files.
    """

    def __init__(
        self,
        tar_path: str,
        format_list: Optional[List[str]] = None,
    ) -> None:
        """Initialize the TarFrameDataset.

        Args:
            tar_path: Path to the tar file.
            format_list: List of data types to load. Defaults to
                ['imgs_png', 'samurai_seg', 'smplx_params', 'flame_params'].
        """
        if format_list is None:
            format_list = ["imgs_png", "samurai_seg", "smplx_params", "flame_params"]

        self.tar_path = tar_path
        self.format_list = format_list
        self.index_map: Dict[int, Dict[str, Any]] = {}
        self.frame_items: List[int] = []
        self.valid_frame_items: Set[int] = set()
        self._build_index()

    def _build_index(self) -> None:
        """Build an index mapping frame numbers to file locations in the tar."""
        with tarfile.open(self.tar_path, "r") as tar:
            for member in tar.getmembers():
                if not member.isfile():
                    continue

                path_parts = member.name.split("/")
                if len(path_parts) < 2 or path_parts[-2] not in self.format_list:
                    continue

                frame_type = path_parts[-2]

                try:
                    frame_num = int(os.path.splitext(path_parts[-1])[0])
                except (ValueError, IndexError):
                    continue

                if frame_num not in self.index_map:
                    self.index_map[frame_num] = {"mask": [0, 0, 0]}

                if frame_type in ["imgs_png", "samurai_seg", "smplx_params"]:
                    if frame_type == "imgs_png":
                        self.index_map[frame_num]["mask"][0] = 1
                    elif frame_type == "samurai_seg":
                        self.index_map[frame_num]["mask"][1] = 1
                    elif frame_type == "smplx_params":
                        self.index_map[frame_num]["mask"][2] = 1

                self.index_map[frame_num][frame_type] = {"name": member.path}

        self.frame_items = list(self.index_map.keys())
        self.valid_frame_items = {
            frame_item
            for frame_item in self.frame_items
            if sum(self.index_map[frame_item]["mask"]) == 3
        }

    def __len__(self) -> int:
        """Return the number of valid frames."""
        return len(self.valid_frame_items)

    def __getitem__(self, frame_num: int) -> Dict[str, Any]:
        """Get frame data for a specific frame number.

        Args:
            frame_num: Frame number to retrieve.

        Returns:
            Dictionary containing frame data with keys:
                - imgs_png: RGB image as numpy array
                - samurai_seg: Segmentation mask as numpy array
                - smplx_params: SMPLX parameters as dictionary
                - flame_params: FLAME parameters as dictionary (optional)

        Raises:
            KeyError: If frame_num is not in valid_frame_items.
        """
        if frame_num not in self.valid_frame_items:
            raise KeyError(f"Frame {frame_num} not found in tar file")

        frame_data: Dict[str, Any] = {
            "imgs_png": None,
            "samurai_seg": None,
            "smplx_params": None,
            "flame_params": None,
        }

        with tarfile.open(self.tar_path, "r") as tf:
            for data_type in self.format_list:
                if data_type not in self.index_map[frame_num]:
                    continue

                info = self.index_map[frame_num][data_type]
                member_path = info["name"]
                file_member = tf.getmember(member_path)
                file_item = tf.extractfile(file_member)
                file_data = file_item.read()

                if data_type in ["imgs_png", "samurai_seg"]:
                    img_data = Image.open(io.BytesIO(file_data))
                    arr_data = np.asarray(img_data)
                    if data_type == "samurai_seg" and arr_data.ndim == 3:
                        arr_data = arr_data[:, :, 3]
                    frame_data[data_type] = arr_data
                elif data_type in ["smplx_params", "flame_params"]:
                    frame_data[data_type] = json.loads(file_data)

        return frame_data

    def get_frame_numbers(self) -> List[int]:
        """Get all frame numbers in sorted order.

        Returns:
            Sorted list of frame numbers.
        """
        return sorted(self.index_map.keys())

    def getattr(
        self,
        frame_num: int,
        data_type: str = "imgs_png",
    ) -> Dict[str, Any]:
        """Get a specific attribute for a frame.

        Args:
            frame_num: Frame number to retrieve.
            data_type: Type of data to retrieve. Must be one of:
                'imgs_png', 'samurai_seg', 'smplx_params', 'flame_params'.

        Returns:
            Dictionary with the requested data_type as key and the data as value.

        Raises:
            AssertionError: If data_type is not in the valid list.
        """
        assert data_type in ["imgs_png", "samurai_seg", "smplx_params", "flame_params"]

        frame_data: Dict[str, Any] = {}

        with tarfile.open(self.tar_path, "r") as tf:
            info = self.index_map[frame_num][data_type]
            member_path = info["name"]
            file_member = tf.getmember(member_path)
            file_item = tf.extractfile(file_member)
            file_data = file_item.read()

            if data_type in ["imgs_png", "samurai_seg"]:
                img_data = Image.open(io.BytesIO(file_data))
                arr_data = np.asarray(img_data)
                if data_type == "samurai_seg" and arr_data.ndim == 3:
                    arr_data = arr_data[:, :, 3]
                frame_data[data_type] = arr_data
            elif data_type in ["smplx_params", "flame_params"]:
                frame_data[data_type] = json.loads(file_data)

        return frame_data


if __name__ == "__main__":

    dataset = TarFrameDataset(
        "/mnt/workspaces/dataset/video_human_datasets/LHM_video_dataset/video_image_2024082101__-videos_clips__-data__-58004_0.tar"
    )

    save_dir = "debug/tar_data"
    img_dir = "debug/tar_data/imgs"
    mask_dir = "debug/tar_data/masks"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for i in range(len(dataset)):
        start = time.time()
        data = dataset[i + 1]
        img = data["imgs_png"]
        seg = data["samurai_seg"]

        cv2.imwrite(os.path.join(img_dir, f"{i:06d}.png"), img[..., ::-1])
        cv2.imwrite(os.path.join(mask_dir, f"{i:06d}.png"), seg)

        print(time.time() - start)
