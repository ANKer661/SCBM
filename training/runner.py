"""
Staged training runner.

The runner coordinates setup and stage execution. Model-specific batch behavior
lives in training.adapters, while the epoch loops live in training.epoch.
"""

import time
import uuid
from os.path import join
from pathlib import Path

import torch

from models.losses import create_loss
from models.models import create_model
from training.adapters import create_adapter
from training.epoch import train_one_epoch, validate_one_epoch
from training.logging import finish_wandb, setup_wandb
from training.optim import build_optimizer, build_scheduler
from training.stages import apply_freeze_policy, apply_stage_cleanup, build_stage_plan
from utils.data import get_concept_groups, get_data, get_empirical_covariance
from utils.training import Custom_Metrics, freeze_module
from utils.utils import reset_random_seeds


class ExperimentRunner:
    """Coordinate setup, training, evaluation, and teardown for one experiment."""

    def __init__(self, config):
        self.config = config
        self.device = None
        self.experiment_path = None

    def run(self):
        gen = reset_random_seeds(self.config.seed)
        self.device = self._setup_device()
        self.experiment_path = self._setup_experiment_path()
        setup_wandb(self.config, Path(__file__).absolute().parents[1])

        train_loader, val_loader, test_loader = get_data(
            self.config,
            self.config.data,
            gen,
        )
        concept_names_graph = get_concept_groups(self.config.data)

        model = self._create_model(train_loader)
        loss_fn = create_loss(self.config)
        adapter = create_adapter(model, loss_fn, self.config)
        metrics = Custom_Metrics(self.config.data.num_concepts, self.device).to(
            self.device
        )

        print(
            "TRAINING "
            + str(self.config.model.model)
            + ": "
            + str(self.config.model.concept_learning + "\n")
        )

        t_epochs = self._run_training(
            adapter,
            train_loader,
            val_loader,
            metrics,
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
        )

        finish_wandb()
        return None

    def _setup_device(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("Using", torch.cuda.get_device_name(0))
        else:
            print("No GPU available")
        return device

    def _setup_experiment_path(self):
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

    def _create_model(self, train_loader):
        model = create_model(self.config)
        if self.config.model.get("cov_type") == "empirical":
            model.sigma_concepts = get_empirical_covariance(train_loader).to(
                self.device
            )
        elif self.config.model.get("cov_type") == "global":
            lower_triangle = get_empirical_covariance(train_loader).to(self.device)
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

    def _select_intervention_function(self):
        from utils.intervention import intervene_cbm, intervene_scbm

        if self.config.model.model == "cbm":
            return intervene_cbm
        return intervene_scbm

    def _run_training(
        self,
        adapter,
        train_loader,
        val_loader,
        metrics,
    ):
        stages = build_stage_plan(self.config)
        for stage in stages:
            self._run_stage(
                stage,
                adapter,
                train_loader,
                val_loader,
                metrics,
            )
        return stages[-1].epochs

    def _run_stage(
        self,
        stage,
        adapter,
        train_loader,
        val_loader,
        metrics,
    ):
        print(stage.message)
        apply_freeze_policy(adapter.model, stage.freeze_policy)

        optimizer = build_optimizer(self.config.model, adapter.model)
        lr_scheduler = build_scheduler(self.config.model, optimizer)
        for epoch in range(stage.epochs):
            if epoch % stage.validate_every == 0:
                print("\nEVALUATION ON THE VALIDATION SET:\n")
                validate_one_epoch(
                    val_loader,
                    adapter,
                    metrics,
                    epoch,
                    self.config,
                    self.device,
                )
            train_one_epoch(
                train_loader,
                adapter,
                optimizer,
                stage,
                metrics,
                epoch,
                self.device,
            )
            lr_scheduler.step()

        apply_stage_cleanup(adapter.model, stage)
