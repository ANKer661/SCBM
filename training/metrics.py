"""Metrics used by staged training."""

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from sklearn.metrics import jaccard_score
from torchmetrics import Metric

from utils.metrics import calc_concept_metrics, calc_target_metrics


class Custom_Metrics(Metric):
    """
    Track and compute losses, target metrics, and concept metrics per epoch.
    """

    def __init__(self, n_concepts: int, device: torch.device) -> None:
        super().__init__()
        self.n_concepts = n_concepts
        self.target_loss: torch.Tensor
        self.add_state("target_loss", default=torch.tensor(0.0, device=device))
        self.concepts_loss: torch.Tensor
        self.add_state("concepts_loss", default=torch.tensor(0.0, device=device))
        self.total_loss: torch.Tensor
        self.add_state("total_loss", default=torch.tensor(0.0, device=device))
        self.y_true: list
        self.add_state("y_true", default=[])
        self.y_pred_logits: list
        self.add_state("y_pred_logits", default=[])
        self.c_true: list
        self.add_state("c_true", default=[])
        self.c_pred_probs: list
        self.add_state("c_pred_probs", default=[])
        self.add_state("concepts_input", default=[])
        self.add_state("batch_features", default=[])
        self.cov_norm: torch.Tensor
        self.add_state("cov_norm", default=torch.tensor(0.0, device=device))
        self.add_state("n_samples", default=torch.tensor(0, dtype=torch.int, device=device))
        self.prec_loss: torch.Tensor
        self.add_state("prec_loss", default=torch.tensor(0.0, device=device))

    def update(
        self,
        target_loss: torch.Tensor,
        concepts_loss: torch.Tensor,
        total_loss: torch.Tensor,
        y_true: torch.Tensor,
        y_pred_logits: torch.Tensor,
        c_true: torch.Tensor,
        c_pred_probs: torch.Tensor,
        cov_norm: torch.Tensor | None = None,
        prec_loss: torch.Tensor | None = None,
    ) -> None:
        assert c_true.shape == c_pred_probs.shape

        n_samples = y_true.size(0)
        self.n_samples += n_samples
        self.target_loss += target_loss * n_samples
        self.concepts_loss += concepts_loss * n_samples
        self.total_loss += total_loss * n_samples
        self.y_true.append(y_true)
        self.y_pred_logits.append(y_pred_logits.detach())
        self.c_true.append(c_true)
        self.c_pred_probs.append(c_pred_probs.detach())
        if cov_norm is not None:
            self.cov_norm += cov_norm * n_samples
        if prec_loss is not None:
            self.prec_loss += prec_loss * n_samples

    def compute(self, validation: bool = False, config: DictConfig | None = None) -> dict:
        y_true = torch.cat(self.y_true, dim=0).cpu()
        c_true = torch.cat(self.c_true, dim=0).cpu()
        c_pred_probs = torch.cat(self.c_pred_probs, dim=0).cpu()
        y_pred_logits = torch.cat(self.y_pred_logits, dim=0).cpu()
        c_pred = c_pred_probs > 0.5
        if y_pred_logits.size(1) == 1:
            y_pred_probs = F.sigmoid(y_pred_logits.squeeze())
            y_pred = y_pred_probs > 0.5
        else:
            y_pred_probs = F.softmax(y_pred_logits, dim=1)
            y_pred = y_pred_logits.argmax(dim=-1)

        target_acc = (y_true == y_pred).sum() / self.n_samples
        concept_acc = (c_true == c_pred).sum() / (self.n_samples * self.n_concepts)
        complete_concept_acc = ((c_true == c_pred).sum(1) == self.n_concepts).sum() / self.n_samples
        target_jaccard = jaccard_score(y_true, y_pred, average="micro")
        concept_jaccard = jaccard_score(c_true, c_pred, average="micro")
        metrics = dict(
            {
                "target_loss": self.target_loss / self.n_samples,
                "prec_loss": self.prec_loss / self.n_samples,
                "concepts_loss": self.concepts_loss / self.n_samples,
                "total_loss": self.total_loss / self.n_samples,
                "y_accuracy": target_acc,
                "c_accuracy": concept_acc,
                "complete_c_accuracy": complete_concept_acc,
                "target_jaccard": target_jaccard,
                "concept_jaccard": concept_jaccard,
            }
        )

        if self.cov_norm != 0:
            metrics = metrics | {"covariance_norm": self.cov_norm / self.n_samples}

        if validation:
            c_pred_probs_list = []
            for j in range(self.n_concepts):
                c_pred_probs_list.append(
                    np.hstack(
                        (
                            np.expand_dims(1 - c_pred_probs[:, j], 1),
                            np.expand_dims(c_pred_probs[:, j], 1),
                        )
                    )
                )
            assert config is not None, "Config must be provided to compute validation metrics"
            y_metrics = calc_target_metrics(y_true.numpy(), y_pred_probs.numpy(), config.data)
            c_metrics, _ = calc_concept_metrics(c_true.numpy(), c_pred_probs_list, config.data)
            metrics = (
                metrics
                | {f"y_{k}": v for k, v in y_metrics.items()}
                | {f"c_{k}": v for k, v in c_metrics.items()}
            )

        return metrics
