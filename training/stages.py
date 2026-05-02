"""Training stage planning and freeze policies."""

from dataclasses import dataclass
from typing import Literal

from utils.freezing import freeze_module, unfreeze_module

StageMode = Literal["c", "t", "j"]
FreezePolicy = Literal["concept_pretrain", "concept_only", "target_only", "joint"]


@dataclass(frozen=True)
class TrainingStage:
    name: str
    mode: StageMode
    epochs: int
    validate_every: int
    freeze_policy: FreezePolicy
    message: str
    unfreeze_encoder_after: bool = False


def build_stage_plan(config) -> list[TrainingStage]:
    stages = []

    if (
        config.model.get("pretrain_concepts")
        and config.model.concept_learning == "autoregressive"
    ):
        stages.append(
            TrainingStage(
                name="ar_concept_pretrain",
                mode="c",
                epochs=config.model.p_epochs,
                validate_every=config.model.validate_per_epoch,
                freeze_policy="concept_pretrain",
                message="\nStarting concepts pre-training!\n",
                unfreeze_encoder_after=True,
            )
        )

    if config.model.training_mode in ("sequential", "independent"):
        stages.append(
            TrainingStage(
                name="concept",
                mode="c",
                epochs=config.model.c_epochs,
                validate_every=config.model.validate_per_epoch,
                freeze_policy="concept_only",
                message="\nStarting concepts training!\n",
            )
        )
        stages.append(
            TrainingStage(
                name="target",
                mode="t",
                epochs=config.model.t_epochs,
                validate_every=config.model.validate_per_epoch,
                freeze_policy="target_only",
                message="\nStarting target training!\n",
            )
        )
    elif config.model.training_mode == "joint":
        stages.append(
            TrainingStage(
                name="joint",
                mode="j",
                epochs=config.model.j_epochs,
                validate_every=config.model.validate_per_epoch,
                freeze_policy="joint",
                message="\nStarting joint training!\n",
            )
        )
    else:
        raise ValueError(
            "model.training_mode must be one of ['joint', 'sequential', "
            f"'independent'], got {config.model.training_mode!r}."
        )

    return stages


def apply_freeze_policy(model, policy: FreezePolicy) -> None:
    if policy == "concept_pretrain":
        model.freeze_c()
        model.encoder.apply(freeze_module)
    elif policy == "concept_only":
        model.freeze_c()
    elif policy == "target_only":
        model.freeze_t()
    elif policy == "joint":
        return
    else:
        raise ValueError(f"Unknown freeze policy: {policy!r}.")


def apply_stage_cleanup(model, stage: TrainingStage) -> None:
    if stage.unfreeze_encoder_after:
        model.encoder.apply(unfreeze_module)
