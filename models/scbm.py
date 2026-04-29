"""Stochastic Concept Bottleneck Model."""

import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import MultivariateNormal, RelaxedBernoulli

from models.concept_backbones import build_encoder, build_head
from utils.training import freeze_module, unfreeze_module


class SCBM(nn.Module):
    """
    Stochastic Concept Bottleneck Model (SCBM) with Learned Covariance Matrix.

    This class implements a Stochastic Concept Bottleneck Model (SCBM) that extends concept prediction by incorporating
    a learned covariance matrix. The SCBM aims to capture the uncertainty and dependencies between concepts, providing
    a more robust and interpretable model for concept-based learning tasks.

    Key Features:
    - Predicts concepts along with a learned covariance matrix to model the relationships and uncertainties between concepts.
    - Supports various training modes and intervention strategies to improve model performance and interpretability.

    Args:
        config (dict): Configuration dictionary containing model and data settings.

    Noteworthy Attributes:
        training_mode (str): The training mode (e.g., "joint", "sequential", "independent").
        num_monte_carlo (int): The number of Monte Carlo samples for uncertainty estimation.
        straight_through (bool): Flag indicating whether to use straight-through gradients.
        curr_temp (float): The current temperature for the Gumbel-Softmax distribution.
        cov_type (str): The type of covariance matrix ("empirical", "global", or "amortized", where "empirical is fixed at start").

    Methods:
        forward(x, epoch, validation=False, c_true=None):
            Perform a forward pass through the model.
        intervene(c_mcmc_probs, c_mcmc_logits):
            Perform an intervention on the model's concept predictions.
    """

    def __init__(self, config) -> None:
        super(SCBM, self).__init__()

        # Configuration arguments
        config_model = config.model
        self.num_concepts = config.data.num_concepts
        self.num_classes = config.data.num_classes
        self.encoder_arch = config_model.encoder_arch
        self.head_arch = config_model.head_arch
        self.training_mode = config_model.training_mode
        self.concept_learning = config_model.concept_learning
        self.num_monte_carlo = config_model.num_monte_carlo
        self.straight_through = config_model.straight_through
        self.curr_temp = 1.0
        if self.training_mode == "joint":
            self.num_epochs = config_model.j_epochs
        else:
            self.num_epochs = config_model.t_epochs
        self.cov_type = config_model.cov_type

        self.init_temp = 1.0
        self.final_temp = 0.5
        self.temp_decay_rate = (math.log(self.final_temp) - math.log(self.init_temp)) / float(
            self.num_epochs
        )

        # Architectures
        # Encoder h(.)
        self.encoder, n_features, encoder_res = build_encoder(config, config_model)
        if encoder_res is not None:
            self.encoder_res = encoder_res

        self.mu_concepts = nn.Linear(n_features, self.num_concepts, bias=True)

        if self.cov_type == "global":
            self.sigma_concepts = nn.Parameter(
                torch.zeros(int(self.num_concepts * (self.num_concepts + 1) / 2))
            )  # Predict lower triangle of concept covariance
        elif self.cov_type == "empirical":
            self.sigma_concepts = torch.zeros(int(self.num_concepts * (self.num_concepts + 1) / 2))
        else:
            self.sigma_concepts = nn.Linear(
                n_features,
                int(self.num_concepts * (self.num_concepts + 1) / 2),
                bias=True,
            )
            self.sigma_concepts.weight.data *= (
                0.01  # To prevent exploding precision matrix at initialization
            )

        # Assume binary concepts
        self.act_c = nn.Sigmoid()

        # Link function g(.)
        if self.num_classes == 2:
            self.pred_dim = 1
        elif self.num_classes > 2:
            self.pred_dim = self.num_classes

        self.head = build_head(self.num_concepts, self.pred_dim, self.head_arch)

    def forward(
        self,
        x: torch.Tensor,
        epoch: int,
        validation: bool = False,
        return_full: bool = False,
        c_true: torch.Tensor | None = None,
    ):
        """
        Perform a forward pass through the Stochastic Concept Bottleneck Model (SCBM).

        This method performs a forward pass through the SCBM, predicting concept probabilities and logits for the target variable.

        Args:
            x (torch.Tensor): The input covariates. Shape: (batch_size, input_dims)
            epoch (int): The current epoch number.
            validation (bool, optional): Flag indicating whether this is a validation pass. Default is False.
            return_full (bool, optional): Flag indicating whether to also return mu of concept. Default is False.
            c_true (torch.Tensor, optional): The ground-truth concept values. Required for "independent" training mode. Default is None.

        Returns:
            tuple: A tuple containing:
                - c_mcmc_prob (torch.Tensor): MCMC samples for predicted concept probabilities. Shape: (batch_size, num_concepts, num_monte_carlo)
                - c_triang_cov (torch.Tensor): Cholesky decomposition of the concept logit covariance matrix. Shape: (batch_size, num_concepts, num_concepts)
                - y_pred_logits (torch.Tensor): Logits for the target variable. Shape: (batch_size, num_classes)
                - c_mu (torch.Tensor, optional): Predicted concept means. Shape: (batch_size, num_concepts). Returned if `return_full` is True.
        Notes:
            - The method first obtains intermediate representations from the encoder.
            - It then predicts the concept means and the Cholesky decomposition of the covariance matrix in the logit space.
            - The method samples from the predicted normal distribution to obtain concept logits and probabilities.
            - Depending on the training mode, it handles different strategies for sampling and backpropagation.
            - Finally, it predicts the target variable logits by averaging over multiple Monte Carlo samples.
        """

        # Get intermediate representations
        intermediate = self.encoder(x)

        # Get mu and cholesky decomposition of covariance
        c_mu = self.mu_concepts(intermediate)
        if self.cov_type == "global":
            assert isinstance(self.sigma_concepts, nn.Parameter), (
                "sigma_concepts must be a learnable parameter for global covariance."
            )
            c_sigma = self.sigma_concepts.repeat(c_mu.size(0), 1)
        elif self.cov_type == "empirical":
            assert isinstance(self.sigma_concepts, torch.Tensor), (
                "sigma_concepts must be a fixed tensor for empirical covariance."
            )
            c_sigma = self.sigma_concepts.unsqueeze(0).repeat(c_mu.size(0), 1, 1)
        else:
            assert isinstance(self.sigma_concepts, nn.Linear), (
                "sigma_concepts must be a linear layer for amortized covariance."
            )
            c_sigma = self.sigma_concepts(intermediate)

        if self.cov_type == "empirical":
            c_triang_cov = c_sigma
        else:
            # Fill the lower triangle of the covariance matrix with the values and make diagonal positive
            c_triang_cov = torch.zeros(
                (c_sigma.shape[0], self.num_concepts, self.num_concepts),
                device=c_sigma.device,
            )
            rows, cols = torch.tril_indices(row=self.num_concepts, col=self.num_concepts, offset=0)
            diag_idx = rows == cols
            c_triang_cov[:, rows, cols] = c_sigma
            c_triang_cov[:, range(self.num_concepts), range(self.num_concepts)] = (
                F.softplus(c_sigma[:, diag_idx]) + 1e-6
            )

        # Sample from predicted normal distribution
        c_dist = MultivariateNormal(c_mu, scale_tril=c_triang_cov)
        c_mcmc_logit = c_dist.rsample([self.num_monte_carlo]).movedim(
            0, -1
        )  # [batch_size,num_concepts,mcmc_size]
        c_mcmc_prob = self.act_c(c_mcmc_logit)

        # For all MCMC samples simultaneously sample from Bernoulli
        if validation or self.training_mode == "sequential":
            # No backpropagation necessary
            c_mcmc = torch.bernoulli(c_mcmc_prob)
        elif self.training_mode == "independent":
            assert c_true is not None, (
                "Ground-truth concepts must be provided for independent training mode."
            )
            c_mcmc = c_true.unsqueeze(-1).repeat(1, 1, self.num_monte_carlo).float()
        else:
            # Backpropagation necessary
            curr_temp = self.compute_temperature(epoch)
            dist = RelaxedBernoulli(temperature=curr_temp, probs=c_mcmc_prob)

            # Bernoulli relaxation
            mcmc_relaxed = dist.rsample()
            if self.straight_through:
                # Straight-Through Gumbel Softmax
                mcmc_hard = (mcmc_relaxed > 0.5) * 1
                c_mcmc = mcmc_hard - mcmc_relaxed.detach() + mcmc_relaxed
            else:
                c_mcmc = mcmc_relaxed

        y_pred_logits = self.compute_y_pred_logits(c_mcmc, c_mcmc_logit)

        # Return concept mu for interventions
        if return_full:
            return c_mcmc_prob, c_mu, c_triang_cov, y_pred_logits
        else:
            return c_mcmc_prob, c_triang_cov, y_pred_logits

    def compute_y_pred_logits(
        self, c_mcmc_probs: torch.Tensor, c_mcmc_logits: torch.Tensor
    ) -> torch.Tensor:
        # Pick the concept tensor: [B, C, M]
        x = c_mcmc_probs if self.concept_learning == "hard" else c_mcmc_logits
        B, C, M = x.shape

        # Run the head over all M samples at once: reshape to [B*M, C]
        x_flat = x.permute(0, 2, 1).reshape(B * M, C)  # [B*M, C]
        y_logits_flat = self.head(x_flat)  # [B*M, K] or [B*M, 1]

        if self.pred_dim == 1:
            # Binary: average Bernoulli probs then convert back to logits
            y_probs = torch.sigmoid(y_logits_flat).view(B, M, 1).mean(dim=1)  # [B, 1]
            y_pred_logits = torch.logit(y_probs, eps=1e-6)  # [B, 1]
            return y_pred_logits
        else:
            # Multiclass: compute log(mean softmax) in a numerically stable way:
            # log(mean_i p_i) == logsumexp(log p_i) - log(M), where log p_i = log_softmax(logits_i)
            y_log_probs = F.log_softmax(y_logits_flat, dim=-1).view(B, M, self.pred_dim)  # [B, M, K]
            y_pred_log_probs = torch.logsumexp(y_log_probs, dim=1) - math.log(M)  # [B, K]
            return y_pred_log_probs

    def intervene(self, c_mcmc_probs: torch.Tensor, c_mcmc_logits: torch.Tensor) -> torch.Tensor:
        y_pred_probs_i = 0
        c_hard = torch.bernoulli(c_mcmc_probs)
        for i in range(self.num_monte_carlo):
            if self.concept_learning == "soft":
                c_i = c_mcmc_logits[:, :, i]
            else:
                c_i = c_hard[:, :, i]

            y_pred_logits_i = self.head(c_i)
            if self.pred_dim == 1:
                y_pred_probs_i += torch.sigmoid(y_pred_logits_i)
            else:
                y_pred_probs_i += torch.softmax(y_pred_logits_i, dim=1)

        y_pred_probs = y_pred_probs_i / self.num_monte_carlo
        if self.pred_dim == 1:
            y_pred_logits = torch.logit(y_pred_probs, eps=1e-6)
        else:
            y_pred_logits = torch.log(y_pred_probs + 1e-6)

        return y_pred_logits

    def compute_temperature(self, epoch: int) -> float:
        curr_temp = max(self.init_temp * math.exp(self.temp_decay_rate * epoch), self.final_temp)
        self.curr_temp = curr_temp
        return curr_temp

    def freeze_c(self) -> None:
        self.head.apply(freeze_module)

    def freeze_t(self) -> None:
        self.head.apply(unfreeze_module)
        self.encoder.apply(freeze_module)
        self.mu_concepts.apply(freeze_module)
        if isinstance(self.sigma_concepts, nn.Linear):
            self.sigma_concepts.apply(freeze_module)
        else:
            self.sigma_concepts.requires_grad = False
