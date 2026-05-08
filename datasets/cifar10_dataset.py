"""
CIFAR-10 dataset loader with concept labels. Relies on create_dataset_cifar.py to have generated concept labels.

This module provides a custom DataLoader for the CIFAR-10 dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CIFAR10CBMDataset: Custom Dataset for CIFAR-10 with concept labels.

Functions:
    build_CIFAR10_CBM_datasets: Returns Dataset objects for training, validation, and testing splits.
"""

from __future__ import annotations
import torch
from torchvision import datasets
from typing import Any


def build_CIFAR10_CBM_datasets(
    datapath,
) -> tuple[CIFAR10CBMDataset, CIFAR10CBMDataset, CIFAR10CBMDataset]:
    datapath = datapath + "cifar10/"
    eval_dataset = CIFAR10CBMDataset(
        root=datapath,
        train=False,
        download=False,
    )
    image_datasets = {
        "train": CIFAR10CBMDataset(
            root=datapath,
            train=True,
            download=False,
        ),
        # avoid create 2 copies
        "val": eval_dataset,
        "test": eval_dataset,
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]


class CIFAR10CBMDataset(datasets.CIFAR10):
    def __init__(
        self,
        root: str,
        train: bool,
        download: bool = False,
        cache: bool = True,
    ) -> None:
        super(CIFAR10CBMDataset, self).__init__(
            root=root,
            train=train,
            download=download,
            transform=None,
        )

        self.cache = cache
        if train:
            concepts_path = root + "cifar10_train_concept_labels.pt"
        else:
            concepts_path = root + "cifar10_test_concept_labels.pt"

        concepts = torch.load(concepts_path, map_location="cpu", weights_only=True)
        self.concepts = concepts.float()

        # self.images: N, C, H, W
        self.images = torch.as_tensor(self.data, dtype=torch.uint8).permute(0, 3, 1, 2)
        self.labels = torch.as_tensor(self.targets, dtype=torch.long)
        if self.cache:
            self.images = self._share_memory_if_available(self.images.contiguous())
            self.labels = self._share_memory_if_available(self.labels)
            self.concepts = self._share_memory_if_available(self.concepts.contiguous())

        # drop original np arrays
        del self.data
        del self.targets

    def __getitem__(self, index: int) -> dict[str, Any]:  # type: ignore[override]
        X = self.images[index]
        target = self.labels[index].item()

        return {
            "img_code": index,
            "labels": target,
            "features": X,
            "concepts": self.concepts[index],
        }

    def get_concept_only_item(self, index: int) -> dict[str, Any]:
        return {
            "img_code": index,
            "labels": self.labels[index],
            "concepts": self.concepts[index],
        }

    def __len__(self) -> int:
        return self.labels.shape[0]

    @staticmethod
    def _share_memory_if_available(tensor: torch.Tensor) -> torch.Tensor:
        try:
            return tensor.share_memory_()
        except RuntimeError:
            return tensor


CIFAR10_CBM_dataloader = CIFAR10CBMDataset
get_CIFAR10_CBM_dataloader = build_CIFAR10_CBM_datasets
