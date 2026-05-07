"""Generic epoch loops for staged training."""

from __future__ import annotations

import typing

import torch
import wandb
from tqdm import tqdm

from training.adapters import BatchTensors

if typing.TYPE_CHECKING:
    from omegaconf import DictConfig
    from torch.utils.data import DataLoader
    from torch.nn import Module
    from torchmetrics import Metric

    from training.adapters import CBMAdapter, SCBMAdapter
    from training.stages import TrainingStage


def move_batch_to_device(batch: dict, device: torch.device) -> BatchTensors:
    features = batch.get("features")
    return BatchTensors(
        features=None if features is None else features.to(device, non_blocking=True),
        targets=batch["labels"].to(device, non_blocking=True),
        concepts=batch["concepts"].to(device, non_blocking=True),
    )


def apply_batch_transform(
    batch: BatchTensors,
    batch_transform: Module | None,
    train: bool,
) -> BatchTensors:
    if batch_transform is None or batch.features is None:
        return batch
    batch.features = batch_transform(batch.features, train=train)
    return batch


def train_one_epoch(
    loader: DataLoader,
    adapter: CBMAdapter | SCBMAdapter,
    optimizer: torch.optim.Optimizer,
    stage: TrainingStage,
    metrics: Metric,
    epoch: int,
    device: torch.device,
    batch_transform: Module | None = None,
) -> None:
    adapter.prepare_train(stage)
    metrics.reset()

    last_batch_idx = -1
    for last_batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch = move_batch_to_device(batch, device)
        batch = apply_batch_transform(batch, batch_transform, train=True)
        output = adapter.forward_train(batch, epoch, stage)
        losses = adapter.compute_loss(output, batch)

        optimizer.zero_grad()
        adapter.backward_loss(losses, stage).backward()
        optimizer.step()

        adapter.update_metrics(metrics, losses, batch, output, validation=False)

    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()

    if last_batch_idx == -1:
        raise ValueError("Cannot train on an empty dataloader.")


def validate_one_epoch(
    loader: DataLoader,
    adapter: CBMAdapter | SCBMAdapter,
    metrics: Metric,
    epoch: int,
    config: DictConfig,
    device: torch.device,
    test: bool = False,
    concept_names_graph: list[str] | None = None,
    batch_transform: Module | None = None,
) -> None:
    adapter.model.eval()
    metrics.reset()

    last_batch_idx = -1
    with torch.no_grad():
        for last_batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch = move_batch_to_device(batch, device)
            batch = apply_batch_transform(batch, batch_transform, train=False)
            output = adapter.forward_eval(batch, epoch)

            if test:
                assert concept_names_graph is not None, (
                    "concept_names_graph must be provided for test plotting."
                )
                adapter.maybe_plot_test_batch(
                    output,
                    batch,
                    last_batch_idx,
                    len(loader),
                    config,
                    concept_names_graph,
                )

            losses = adapter.compute_loss(output, batch)
            adapter.update_metrics(metrics, losses, batch, output, validation=True)

    if last_batch_idx == -1:
        raise ValueError("Cannot validate on an empty dataloader.")

    metrics_dict = metrics.compute(validation=True, config=config)
    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = "Test: "

    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
