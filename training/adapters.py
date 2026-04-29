"""Model-specific training adapters."""

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


class BaseAdapter:
    def __init__(self, model, loss_fn, config):
        self.model = model
        self.loss_fn = loss_fn
        self.config = config

    def prepare_train(self, stage):
        self.model.train()
        if self.config.model.training_mode in ("sequential", "independent"):
            if stage.mode == "c":
                self.model.head.eval()
            elif stage.mode == "t":
                self.model.encoder.eval()

    def backward_loss(self, losses: LossOutput, stage):
        if stage.mode == "j":
            return losses.total_loss
        if stage.mode == "c":
            return losses.concepts_loss
        return losses.target_loss

    def update_metrics(self, metrics, losses: LossOutput, batch: BatchTensors, output):
        kwargs = {}
        if losses.precision_matrix_loss is not None:
            kwargs["prec_loss"] = losses.precision_matrix_loss

        metrics.update(
            losses.target_loss,
            losses.concepts_loss,
            losses.total_loss,
            batch.targets,
            output.target_logits,
            batch.concepts,
            output.concept_probs,
            **kwargs,
        )

    def maybe_plot_test_batch(
        self,
        output,
        batch: BatchTensors,
        batch_idx: int,
        loader_len: int,
        config,
        concept_names_graph,
    ):
        return


class SCBMAdapter(BaseAdapter):
    def forward_train(self, batch: BatchTensors, epoch: int, stage):
        concepts_mcmc_probs, triang_cov, target_logits = self.model(
            batch.features, epoch, c_true=batch.concepts
        )
        return BatchOutput(
            concept_probs=concepts_mcmc_probs,
            target_logits=target_logits,
            covariance=triang_cov,
        )

    def forward_eval(self, batch: BatchTensors, epoch: int):
        concepts_mcmc_probs, triang_cov, target_logits = self.model(
            batch.features, epoch, validation=True, c_true=batch.concepts
        )
        return BatchOutput(
            concept_probs=concepts_mcmc_probs,
            target_logits=target_logits,
            covariance=triang_cov,
        )

    def compute_loss(self, output: BatchOutput, batch: BatchTensors):
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

    def backward_loss(self, losses: LossOutput, stage):
        if stage.mode == "c":
            assert losses.precision_matrix_loss is not None
            return losses.concepts_loss + losses.precision_matrix_loss
        return super().backward_loss(losses, stage)

    def update_metrics(self, metrics, losses: LossOutput, batch: BatchTensors, output):
        metrics_output = BatchOutput(
            concept_probs=output.concept_probs.mean(-1),
            target_logits=output.target_logits,
        )
        super().update_metrics(metrics, losses, batch, metrics_output)

    def maybe_plot_test_batch(
        self,
        output,
        batch: BatchTensors,
        batch_idx: int,
        loader_len: int,
        config,
        concept_names_graph,
    ):
        if batch_idx % max(1, loader_len // 10) != 0:
            return

        try:
            from utils.plotting import compute_and_plot_heatmap

            cov = torch.matmul(
                output.covariance,
                torch.transpose(output.covariance, dim0=1, dim1=2),
            )
            corr = (cov[0] / cov[0].diag().sqrt()).transpose(
                dim0=0, dim1=1
            ) / cov[0].diag().sqrt()
            matrix = corr.cpu().numpy()
            compute_and_plot_heatmap(matrix, batch.concepts, concept_names_graph, config)
        except Exception:
            pass


class CBMAdapter(BaseAdapter):
    def forward_train(self, batch: BatchTensors, epoch: int, stage):
        if self.config.model.training_mode == "independent" and stage.mode == "t":
            concept_probs, target_logits, _ = self.model(
                batch.features, epoch, batch.concepts
            )
        elif (
            self.config.model.concept_learning == "autoregressive"
            and stage.mode == "c"
        ):
            concept_probs, target_logits, _ = self.model(
                batch.features, epoch, concepts_train_ar=batch.concepts
            )
        else:
            concept_probs, target_logits, _ = self.model(batch.features, epoch)

        return BatchOutput(concept_probs=concept_probs, target_logits=target_logits)

    def forward_eval(self, batch: BatchTensors, epoch: int):
        concept_probs, target_logits, _ = self.model(
            batch.features, epoch, validation=True
        )
        if self.config.model.concept_learning == "autoregressive":
            concept_probs = torch.mean(concept_probs, dim=-1)
        return BatchOutput(concept_probs=concept_probs, target_logits=target_logits)

    def compute_loss(self, output: BatchOutput, batch: BatchTensors):
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


def create_adapter(model, loss_fn, config):
    if config.model.model == "cbm":
        return CBMAdapter(model, loss_fn, config)
    return SCBMAdapter(model, loss_fn, config)
