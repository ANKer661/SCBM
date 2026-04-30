"""
Training adapters that encapsulate model-specific behavior within a shared training loop.

The training loop calls a fixed sequence of methods on the adapter — prepare_train,
forward_train, compute_loss, backward_loss, update_metrics — without branching on model
type. Each adapter implements these methods according to its model's requirements.
"""

from dataclasses import dataclass

import torch


@dataclass
class BatchTensors:
    features: torch.Tensor
    targets: torch.Tensor
    concepts: torch.Tensor


@dataclass
class BatchOutput:
    concept_probs: torch.Tensor
    target_logits: torch.Tensor
    covariance: torch.Tensor | None = None


@dataclass
class LossOutput:
    target_loss: torch.Tensor
    concepts_loss: torch.Tensor
    total_loss: torch.Tensor
    precision_matrix_loss: torch.Tensor | None = None


class CBMAdapter:
    def __init__(self, model, loss_fn, config) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

    def prepare_train(self, stage) -> None:
        self.model.train()
        if self.config.model.training_mode in ("sequential", "independent"):
            if stage.mode == "c":
                self.model.head.eval()
            elif stage.mode == "t":
                self.model.encoder.eval()

    def forward_train(self, batch: BatchTensors, epoch: int, stage) -> BatchOutput:
        if self.config.model.training_mode == "independent" and stage.mode == "t":
            concept_probs, target_logits, _ = self.model(batch.features, epoch, batch.concepts)
        elif self.config.model.concept_learning == "autoregressive" and stage.mode == "c":
            concept_probs, target_logits, _ = self.model(
                batch.features, epoch, concepts_train_ar=batch.concepts
            )
        else:
            concept_probs, target_logits, _ = self.model(batch.features, epoch)

        return BatchOutput(concept_probs=concept_probs, target_logits=target_logits)

    def forward_eval(self, batch: BatchTensors, epoch: int) -> BatchOutput:
        concept_probs, target_logits, _ = self.model(batch.features, epoch, validation=True)
        if self.config.model.concept_learning == "autoregressive":
            concept_probs = torch.mean(concept_probs, dim=-1)
        return BatchOutput(concept_probs=concept_probs, target_logits=target_logits)

    def compute_loss(self, output: BatchOutput, batch: BatchTensors) -> LossOutput:
        target_loss, concepts_loss, total_loss = self.loss_fn(
            output.concept_probs,
            batch.concepts,
            output.target_logits,
            batch.targets,
        )
        return LossOutput(
            target_loss=target_loss,
            concepts_loss=concepts_loss,
            total_loss=total_loss,
        )

    def backward_loss(self, losses: LossOutput, stage) -> torch.Tensor:
        if stage.mode == "j":
            return losses.total_loss
        if stage.mode == "c":
            return losses.concepts_loss
        return losses.target_loss

    def update_metrics(
        self, metrics, losses: LossOutput, batch: BatchTensors, output: BatchOutput
    ) -> None:
        metrics.update(
            losses.target_loss,
            losses.concepts_loss,
            losses.total_loss,
            batch.targets,
            output.target_logits,
            batch.concepts,
            output.concept_probs,
        )

    def maybe_plot_test_batch(
        self,
        output,
        batch: BatchTensors,
        batch_idx: int,
        loader_len: int,
        config,
        concept_names_graph,
    ) -> None:
        return


class SCBMAdapter:
    def __init__(self, model, loss_fn, config) -> None:
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

    def prepare_train(self, stage) -> None:
        self.model.train()
        if self.config.model.training_mode in ("sequential", "independent"):
            if stage.mode == "c":
                self.model.head.eval()
            elif stage.mode == "t":
                self.model.encoder.eval()

    def forward_train(self, batch: BatchTensors, epoch: int, stage) -> BatchOutput:
        concepts_mcmc_probs, triang_cov, target_logits = self.model(
            batch.features, epoch, c_true=batch.concepts
        )
        return BatchOutput(
            concept_probs=concepts_mcmc_probs,
            target_logits=target_logits,
            covariance=triang_cov,
        )

    def forward_eval(self, batch: BatchTensors, epoch: int) -> BatchOutput:
        concepts_mcmc_probs, triang_cov, target_logits = self.model(
            batch.features, epoch, validation=True, c_true=batch.concepts
        )
        return BatchOutput(
            concept_probs=concepts_mcmc_probs,
            target_logits=target_logits,
            covariance=triang_cov,
        )

    def compute_loss(self, output: BatchOutput, batch: BatchTensors) -> LossOutput:
        target_loss, concepts_loss, precision_matrix_loss, total_loss = self.loss_fn(
            output.concept_probs,
            batch.concepts,
            output.target_logits,
            batch.targets,
            output.covariance,
        )
        return LossOutput(
            target_loss=target_loss,
            concepts_loss=concepts_loss,
            precision_matrix_loss=precision_matrix_loss,
            total_loss=total_loss,
        )

    def backward_loss(self, losses: LossOutput, stage) -> torch.Tensor:
        if stage.mode == "c":
            assert losses.precision_matrix_loss is not None
            return losses.concepts_loss + losses.precision_matrix_loss
        if stage.mode == "j":
            return losses.total_loss
        return losses.target_loss

    def update_metrics(
        self, metrics, losses: LossOutput, batch: BatchTensors, output: BatchOutput
    ) -> None:
        assert losses.precision_matrix_loss is not None
        metrics.update(
            losses.target_loss,
            losses.concepts_loss,
            losses.total_loss,
            batch.targets,
            output.target_logits,
            batch.concepts,
            output.concept_probs.mean(-1),
            prec_loss=losses.precision_matrix_loss,
        )

    def maybe_plot_test_batch(
        self,
        output,
        batch: BatchTensors,
        batch_idx: int,
        loader_len: int,
        config,
        concept_names_graph,
    ) -> None:
        if batch_idx % max(1, loader_len // 10) != 0:
            return

        try:
            from utils.plotting import compute_and_plot_heatmap

            cov = torch.matmul(
                output.covariance,
                torch.transpose(output.covariance, dim0=1, dim1=2),
            )
            corr = (cov[0] / cov[0].diag().sqrt()).transpose(dim0=0, dim1=1) / cov[0].diag().sqrt()
            compute_and_plot_heatmap(corr.cpu().numpy(), batch.concepts, concept_names_graph, config)
        except Exception:
            pass


def create_adapter(model, loss_fn, config) -> CBMAdapter | SCBMAdapter:
    if config.model.model == "cbm":
        return CBMAdapter(model, loss_fn, config)
    elif config.model.model == "scbm":
        return SCBMAdapter(model, loss_fn, config)
    else:
        raise ValueError(f"Unknown model type: {config.model.model}")
