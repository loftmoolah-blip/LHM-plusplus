# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Xiaodong Gu & Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-04-07 23:56:19
# @Function      : video_human_dataset, used tar to I/O data.

import math
from functools import partial

import torch

__all__ = ["MixerDataset"]


class MixerDataset(torch.utils.data.Dataset):
    """
    MixerDataset

    This dataset class enables the combination of multiple subsets (datasets) into a single dataset for training or evaluation.
    It constructs its internal subsets according to provided configuration, applying individual sample rates to each,
    and supports common interfaces of PyTorch datasets.
    The actual subsets are instantiated by dispatching to the correct dataset class according to their 'name' field.
    """

    def __init__(
        self,
        split: str,
        subsets: dict,
        **dataset_kwargs,
    ):
        for subset in subsets:
            if "processing_pipeline" in subset:
                pipeline_list = subset["processing_pipeline"]
                for pipeline in pipeline_list:
                    pipeline["val"] = dataset_kwargs.get("is_val", False)
        self.subsets = [
            self._dataset_fn(subset, split)(
                use_flame=subset["use_flame"],
                src_head_size=subset.get("src_head_size", 448),
                ref_img_size=subset.get("ref_img_size", 1),
                womask=subset.get("womask", False),
                image_sr_root=subset.get("image_sr_root", None),
                random_sample_ratio=subset.get(
                    "random_sample_ratio", 0.9
                ),  # view random sample ratio
                processing_pipeline=subset.get(
                    "processing_pipeline", None
                ),  # view random sample ratio
                **dataset_kwargs,
            )
            for subset in subsets
        ]
        self.virtual_lens = [
            math.ceil(subset_config["sample_rate"] * len(subset_obj))
            for subset_config, subset_obj in zip(subsets, self.subsets)
        ]

    @staticmethod
    def _dataset_fn(subset_config: dict, split: str):
        """
        Given a subset configuration dictionary and a split string, returns a function that instantiates
        the appropriate dataset class partially applied with its root_dirs and meta_path for the given split.

        Args:
            subset_config (dict): The configuration dictionary for one subset, containing keys like 'name', 'root_dirs', and 'meta_path'.
            split (str): The data split to use (e.g., 'train', 'val', 'test').

        Returns:
            Callable: A function that when called with additional arguments, constructs an instance of the requested dataset.
        """

        name = subset_config["name"]

        dataset_cls = None
        if name == "video_human_tar":
            from .video_human_dataset import VideoHumanDataset

            dataset_cls = VideoHumanDataset
        elif name == "video_human_a4o_tar":
            from .video_human_dataset_a4o import VideoHumanA4ODataset

            dataset_cls = VideoHumanA4ODataset
        elif name == "video_human_lhm_tar":
            from .video_human_lhm_dataset import VideoHumanLHMDataset

            dataset_cls = VideoHumanLHMDataset
        elif name == "video_human_lhm_a4o_tar":
            from .video_human_lhm_dataset_a4o import VideoHumanLHMA4ODataset

            dataset_cls = VideoHumanLHMA4ODataset
        elif name == "video_human_lhm_mesh_a4o_tar":
            from .video_human_lhm_dataset_a4o import VideoHumanMeshLHMA4ODataset

            dataset_cls = VideoHumanMeshLHMA4ODataset
        elif name == "video_human_mesh_a4o_tar":
            from .video_human_dataset_a4o import VideoHumanMeshA4ODataset

            dataset_cls = VideoHumanMeshA4ODataset
        elif name == "dna_a4o_tar":
            from .DNA_Rendering_A4O import DNARenderingDataset

            dataset_cls = DNARenderingDataset
        else:
            raise NotImplementedError(f"Dataset {name} not implemented")

        return partial(
            dataset_cls,
            root_dirs=subset_config["root_dirs"],
            meta_path=subset_config["meta_path"][split],
        )

    def __len__(self):
        return sum(self.virtual_lens)

    def __getitem__(self, idx):
        """
        Retrieves the data item at the given global index across all subsets.

        Args:
            idx (int): The global index of the data item to retrieve.

        Returns:
            dict or object: The data item from the corresponding subset at the computed local index.
        """
        subset_idx = 0
        virtual_idx = idx
        while virtual_idx >= self.virtual_lens[subset_idx]:
            virtual_idx -= self.virtual_lens[subset_idx]
            subset_idx += 1
        real_idx = virtual_idx % len(self.subsets[subset_idx])
        return self.subsets[subset_idx][real_idx]
