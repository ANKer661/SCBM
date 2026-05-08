"""
Staged training runner.

The runner coordinates setup and stage execution. Model-specific batch behavior
lives in training.adapters, while the epoch loops live in training.epoch.
"""

from __future__ import annotations
import os
import time
import typing
import uuid
from os.path import join
from pathlib import Path

import torch

from models.losses import create_loss
from models.factory import create_model
from datasets.datamodule import build_dataloaders
from training.adapters import create_adapter
from training.batch_transforms import create_batch_transform
from training.epoch import train_one_epoch, validate_one_epoch
from training.logging import finish_wandb, setup_wandb
from training.metrics import ConceptBottleneckMetrics
from training.optim import build_optimizer, build_scheduler
from training.stages import apply_freeze_policy, apply_stage_cleanup, build_stage_plan
from utils.freezing import freeze_module
from utils.utils import numerical_stability_check
from utils.utils import reset_random_seeds

if typing.TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from typing import Callable
    from training.adapters import CBMAdapter, SCBMAdapter
    from training.stages import TrainingStage
    from torchmetrics import Metric
    from omegaconf import DictConfig


class ExperimentRunner:
    """Coordinate setup, training, evaluation, and teardown for one experiment."""

    def __init__(self, config: DictConfig) -> None:
        self.config = config
        self.device: torch.device
        self.experiment_path = None

    def run(self) -> None:
        gen = reset_random_seeds(self.config.seed)
        self.device = self._setup_device()
        self.experiment_path = self._setup_experiment_path()
        setup_wandb(self.config, Path(__file__).absolute().parents[1])

        dataloaders = build_dataloaders(self.config, gen)
        train_loader = dataloaders.train
        target_train_loader = dataloaders.target_train
        val_loader = dataloaders.val
        test_loader = dataloaders.test
        concept_names_graph = self._get_concept_groups()

        model = self._setup_model(train_loader)
        batch_transform = create_batch_transform(self.config)
        if batch_transform is not None:
            batch_transform.to(self.device)
        loss_fn = create_loss(self.config)
        adapter = create_adapter(model, loss_fn, self.config)  # type: ignore
        metrics = ConceptBottleneckMetrics(self.config.data.num_concepts, self.device).to(
            self.device
        )

        if self.config.get("intervene_only", False):
            print("\nPERFORMING INTERVENTIONS:\n")
            intervention_start = time.perf_counter()
            intervene = self._select_intervention_function()
            intervene(
                train_loader,
                test_loader,
                model,
                metrics,
                0,
                self.config,
                loss_fn,
                self.device,
                batch_transform=batch_transform,
            )
            intervention_sec = time.perf_counter() - intervention_start
            print(f"[timing] intervention_sec={intervention_sec:.3f}", flush=True)
            finish_wandb()
            return None

        print(
            "TRAINING "
            + str(self.config.model.model)
            + ": "
            + str(self.config.model.concept_learning + "\n")
        )

        t_epochs = self._run_training(
            adapter,
            train_loader,
            target_train_loader,
            val_loader,
            metrics,
            batch_transform,
        )

        model.apply(freeze_module)
        if self.config.save_model:
            torch.save(model.state_dict(), join(self.experiment_path, "model.pth"))
            print("\nTRAINING FINISHED, MODEL SAVED!", flush=True)
        else:
            print("\nTRAINING FINISHED", flush=True)

        print("\nEVALUATION ON THE TEST SET:\n")
        validate_one_epoch(
            test_loader,
            adapter,
            metrics,
            t_epochs,
            self.config,
            self.device,
            test=True,
            concept_names_graph=concept_names_graph,
            batch_transform=batch_transform,
        )

        if self.config.train_only:
            finish_wandb()
            return None

        print("\nPERFORMING INTERVENTIONS:\n")
        intervene = self._select_intervention_function()
        intervene(
            train_loader,
            test_loader,
            model,
            metrics,
            t_epochs,
            self.config,
            loss_fn,
            self.device,
            batch_transform=batch_transform,
        )

        finish_wandb()
        return None

    def _setup_device(self) -> torch.device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("Using", torch.cuda.get_device_name(0))
        else:
            print("No GPU available")
        return device

    def _setup_experiment_path(self) -> Path:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        ex_name = "{}_{}".format(str(timestr), uuid.uuid4().hex[:5])
        experiment_path = (
            Path(self.config.experiment_dir)
            / self.config.model.model
            / self.config.data.dataset
            / ex_name
        )
        experiment_path.mkdir(parents=True)
        self.config.experiment_dir = str(experiment_path)
        print("Experiment path: ", experiment_path)
        return experiment_path

    def _setup_model(self, train_loader: DataLoader) -> torch.nn.Module:
        model = create_model(self.config)
        if self.config.model.get("cov_type") == "empirical":
            model.sigma_concepts = self._get_empirical_covariance(train_loader).to(
                self.device
            )
        elif self.config.model.get("cov_type") == "global":
            lower_triangle = self._get_empirical_covariance(train_loader).to(
                self.device
            )
            rows, cols = torch.tril_indices(
                row=self.config.data.num_concepts,
                col=self.config.data.num_concepts,
                offset=0,
            )
            model.sigma_concepts = torch.nn.Parameter(lower_triangle[rows, cols])
            diag_idx = rows == cols
            with torch.no_grad():
                model.sigma_concepts[diag_idx] = (
                    lower_triangle[rows, cols][diag_idx].expm1().clamp_min(1e-6).log()
                )

        model.to(self.device)
        return model

    def _get_empirical_covariance(self, dataloader: DataLoader) -> torch.Tensor:
        data = []
        for batch in dataloader:
            concepts = batch["concepts"]
            data.append(concepts)
        data = torch.cat(data)
        data_logits = torch.logit(0.05 + 0.9 * data)
        covariance = torch.cov(data_logits.transpose(0, 1))
        covariance = numerical_stability_check(covariance, device="cpu")
        return torch.linalg.cholesky(covariance)

    def _get_concept_groups(self) -> list[str]:
        config_data = self.config.data
        if config_data.dataset == "CUB":
            with open(
                os.path.join(
                    config_data.data_path, "CUB/CUB_200_2011/concept_names.txt"
                ),
                "r",
            ) as f:
                concept_names = []
                for line in f:
                    concept_names.append(line.replace("\n", "").split("::"))
            return [": ".join(name) for name in concept_names]

        if config_data.dataset == "cifar10":
            with open(
                os.path.join(config_data.data_path, "cifar10/cifar10_filtered.txt"),
                "r",
            ) as file:
                return [line.strip() for line in file]

        return [str(i) for i in range(config_data.num_concepts)]

    def _select_intervention_function(self) -> Callable:
        from interventions.evaluation import intervene_cbm_batch_first, intervene_scbm_batch_first

        if self.config.model.model == "cbm":
            return intervene_cbm_batch_first
        elif self.config.model.model == "scbm":
            return intervene_scbm_batch_first

        raise ValueError(f"Intervention not implemented for model {self.config.model.model}")

    def _run_training(
        self,
        adapter: SCBMAdapter | CBMAdapter,
        train_loader: DataLoader,
        target_train_loader: DataLoader | None,
        val_loader: DataLoader,
        metrics: Metric,
        batch_transform: torch.nn.Module | None,
    ) -> int:
        stages = build_stage_plan(self.config)
        for stage in stages:
            self._run_stage(
                stage,
                adapter,
                train_loader,
                target_train_loader,
                val_loader,
                metrics,
                batch_transform,
            )
        return stages[-1].epochs

    def _run_stage(
        self,
        stage: TrainingStage,
        adapter: SCBMAdapter | CBMAdapter,
        train_loader: DataLoader,
        target_train_loader: DataLoader | None,
        val_loader: DataLoader,
        metrics: Metric,
        batch_transform: torch.nn.Module | None,
    ) -> None:
        print(stage.message)
        apply_freeze_policy(adapter.model, stage.freeze_policy)

        optimizer = build_optimizer(self.config.model, adapter.model)
        lr_scheduler = build_scheduler(self.config.model, optimizer)
        stage_train_loader = self._select_train_loader(stage, train_loader, target_train_loader)
        for epoch in range(stage.epochs):
            epoch_start = time.perf_counter()
            val_sec = 0.0
            if epoch % stage.validate_every == 0:
                print("\nEVALUATION ON THE VALIDATION SET:\n")
                val_start = time.perf_counter()
                validate_one_epoch(
                    val_loader,
                    adapter,
                    metrics,
                    epoch,
                    self.config,
                    self.device,
                    batch_transform=batch_transform,
                )
                val_sec = time.perf_counter() - val_start
            train_start = time.perf_counter()
            train_one_epoch(
                stage_train_loader,
                adapter,
                optimizer,
                stage,
                metrics,
                epoch,
                self.device,
                batch_transform=batch_transform,
            )
            train_sec = time.perf_counter() - train_start
            lr_scheduler.step()
            epoch_sec = time.perf_counter() - epoch_start
            print(
                f"[timing] stage={stage.name} epoch={epoch + 1}/{stage.epochs} "
                f"train_sec={train_sec:.3f} val_sec={val_sec:.3f} "
                f"total_sec={epoch_sec:.3f}",
                flush=True,
            )

        apply_stage_cleanup(adapter.model, stage)

    def _select_train_loader(
        self,
        stage: TrainingStage,
        train_loader: DataLoader,
        target_train_loader: DataLoader | None,
    ) -> DataLoader:
        if (
            self.config.model.concept_learning == "autoregressive"
            and self.config.model.training_mode == "independent"
            and stage.mode == "t"
        ):
            if target_train_loader is None:
                raise ValueError("AR target training requires a concept-only train loader.")
            return target_train_loader

        return train_loader
