"""Experiment logging helpers."""

import os
from pathlib import Path

import wandb
from omegaconf import OmegaConf


def setup_wandb(config, project_root: Path) -> None:
    os.environ["WANDB_CACHE_DIR"] = os.path.join(
        project_root, "wandb", ".cache", "wandb"
    )
    print("Cache dir:", os.environ["WANDB_CACHE_DIR"])
    wandb.init(
        project=config.logging.project,
        reinit=True,
        entity=config.logging.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode=config.logging.mode,
        tags=[config.model.tag],
    )
    if config.logging.mode in ["online", "disabled"]:
        wandb.run.name = wandb.run.name.split("-")[-1] + "-" + config.experiment_name
    elif config.logging.mode == "offline":
        wandb.run.name = config.experiment_name
    else:
        raise ValueError("wandb needs to be set to online, offline or disabled.")


def finish_wandb() -> None:
    wandb.finish(quiet=True)
