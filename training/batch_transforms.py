"""Optional batch-level transforms applied after device transfer."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.transforms import v2


class CIFARBatchTransform(nn.Module):
    """CIFAR image preprocessing on the active training device.

    The dataset provides raw uint8 images in CHW format. This module converts a
    full batch to float, applies the CIFAR train/eval preprocessing, and returns
    224x224 tensors for the encoders.
    """

    def __init__(self) -> None:
        super().__init__()
        self.train_transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
                v2.Resize(size=(224, 224), antialias=True),
                v2.RandomHorizontalFlip(),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.eval_transform = v2.Compose(
            [
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize(size=(224, 224), antialias=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def forward(self, imgs: torch.Tensor, train: bool) -> torch.Tensor:
        if train:
            return self.train_transform(imgs)
        return self.eval_transform(imgs)


def create_batch_transform(config) -> nn.Module | None:
    if config.data.dataset == "cifar10":
        return CIFARBatchTransform()
    return None
