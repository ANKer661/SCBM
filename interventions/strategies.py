"""Strategies for encoding intervened concept values."""

from __future__ import annotations
import typing

import torch
from scipy.stats import chi2
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from utils.minimize_constraint import minimize_constr
from utils.utils import numerical_stability_check

if typing.TYPE_CHECKING:
    from torch.utils.data import DataLoader
    from omegaconf import DictConfig


def define_strategy(
    inter_strategy: str,
    train_loader: DataLoader,
    model: nn.Module,
    device: torch.device,
    config: DictConfig,
    batch_transform=None,
):
    """
    Return the intervention strategy that determines how the ground-truth intervened-on concept values are encoded in the model.
    This function selects and returns the appropriate intervention strategy based on the provided strategy name and model configuration.

    Args:
        inter_strategy (str): The name of the intervention strategy to use. Supported strategies are:
                              - "simple_perc": Uses the logits corresponding to 5% and 95%.
                              - "emp_perc": Uses the logits corresponding to 5th and 95th percentile of training predictions.
                              - "conf_interval_optimal": Uses a confidence interval-based optimal strategy for SCBMs.
        train_loader (torch.utils.data.DataLoader): DataLoader to obtain empirical percentiles of training data
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the computations on.
        config (dict): Configuration dictionary containing model and data settings.

    Returns:
        object: An instance of the selected intervention strategy class.

    Raises:
        NotImplementedError: If the specified strategy is not supported for the given model.

    Example:
        >>> strategy = define_strategy("simple_perc", train_loader, model, device, config)
        USING FOLLOWING STRATEGY: PercentileStrategy
    """

    if config.model.model == "cbm":
        if config.model.concept_learning in ("hard", "autoregressive", "embedding"):
            strategy = HardCBMStrategy()
        elif inter_strategy == "simple_perc":
            strategy = PercentileStrategy()
        elif inter_strategy == "emp_perc":
            strategy = EmpiricalPercentileStrategy(
                train_loader=train_loader,
                model=model,
                device=device,
                batch_transform=batch_transform,
            )
        else:
            raise NotImplementedError(
                f"No such strategy as {inter_strategy} defined for model {config.model.model}!"
            )
        print("USING FOLLOWING STRATEGY:", strategy.__class__.__name__)

    elif config.model.model == "scbm":
        strategy = SCBMConditionalStrategy(
            inter_strategy, train_loader, model, device, config, batch_transform=batch_transform
        )
        print("USING FOLLOWING STRATEGY:", strategy.interv_strat.__class__.__name__)
    else:
        raise NotImplementedError(
            f"No such strategy as {inter_strategy} defined for model {config.model.model}!"
        )

    return strategy


class SCBMConditionalStrategy:
    """
    A strategy for intervening on SCBM using the conditional normal distribution.

    This class defines a strategy for intervening on SCBM(Stochastic Concept Bottleneck Model)s.
    It supports different intervention strategies such as simple percentile, empirical percentile, and confidence interval optimal strategies.

    Args:
        inter_strategy (str): The name of the intervention strategy to use. Supported strategies are:
                              - "simple_perc": Uses the logits corresponding to 5% and 95%.
                              - "emp_perc": Uses the logits corresponding to 5th and 95th percentile of training predictions.
                              - "conf_interval_optimal": Uses a confidence interval-based optimal strategy for SCBMs.
        train_loader (torch.utils.data.DataLoader): DataLoader to obtain empirical percentiles of training data
        model (torch.nn.Module): The model to be evaluated.
        device (torch.device): The device to run the computations on.
        config (dict): Configuration dictionary containing model and data settings.
    """

    def __init__(self, inter_strategy, train_loader, model, device, config, batch_transform=None):
        self.num_monte_carlo = config.model.num_monte_carlo
        self.num_concepts = config.data.num_concepts
        self.act_c = nn.Sigmoid()
        if inter_strategy == "simple_perc":
            self.interv_strat = PercentileStrategy()
        elif inter_strategy == "emp_perc":
            self.interv_strat = EmpiricalPercentileStrategy(
                train_loader=train_loader,
                model=model,
                device=device,
                is_scbm=True,
                batch_transform=batch_transform,
            )
        elif inter_strategy == "conf_interval_optimal":
            self.interv_strat = ConfIntervalOptimalStrategy(level=config.model.level)
        elif inter_strategy == "conf_interval_optimal_fw":
            self.interv_strat = ConfIntervalOptimalFWStrategy(level=config.model.level)
        else:
            raise NotImplementedError(
                "No such strategy as",
                config.model.inter_strategy,
                "defined for model",
                config.model.model,
                "!",
            )

    def compute_intervention(self, c_mu, c_cov, c_true, c_mask):
        """
        Generate an intervention on an SCBM using the conditional normal distribution.

        First, this function computes the logits of the intervened-on concepts based on the intervention strategy.
        Then, using the predicted concept mean and covariance, it computes the conditional normal distribution, conditioned on
        the intervened-on concept logits. To this end, the order is permuted such that the intervened-on concepts form a block at the start.
        Finally, the method samples from the conditional normal distribution and permutes the results back to the original order.

        Args:
            c_mu (torch.Tensor): The predicted mean values of the concepts. Shape: (batch_size, num_concepts)
            c_cov (torch.Tensor): The predicted covariance matrix of the concepts. Shape: (batch_size, num_concepts, num_concepts)
            c_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts)
            c_mask (torch.Tensor): A mask indicating which concepts are intervened-on. Shape: (batch_size, num_concepts)

        Returns:
            tuple: A tuple containing the intervened-on concept means, covariances, MCMC sampled concept probabilities, and logits.
                    Note that the probabilities are set to 0/1 for the intervened-on concepts according to the ground-truth.
        """
        num_intervened = int(c_mask[0].sum().item())
        device = c_mask.device

        if num_intervened == 0:
            # No intervention
            interv_mu = c_mu
            interv_cov = c_cov
            # Sample from normal distribution
            dist = MultivariateNormal(interv_mu, covariance_matrix=interv_cov)
            mcmc_logits = dist.rsample([self.num_monte_carlo]).movedim(
                0, -1
            )  # [batch_size,bottleneck_size,mcmc_size]

        else:
            # Compute logits of intervened-on concepts
            c_intervened_logits = self.interv_strat.compute_intervened_logits(
                c_mu, c_cov, c_true, c_mask
            )

            ## Compute conditional normal distribution sample-wise
            # Permute covariance s.t. intervened-on concepts are a block at start
            indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
            perm_cov = c_cov.gather(1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2)))
            perm_cov = perm_cov.gather(2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1))
            perm_mu = c_mu.gather(1, indices)
            perm_c_intervened_logits = c_intervened_logits.gather(1, indices)

            # Compute mu and covariance conditioned on intervened-on concepts
            # Intermediate steps
            perm_intermediate_cov = torch.matmul(
                perm_cov[:, num_intervened:, :num_intervened],
                torch.inverse(perm_cov[:, :num_intervened, :num_intervened]),
            )
            perm_intermediate_mu = (
                perm_c_intervened_logits[:, :num_intervened] - perm_mu[:, :num_intervened]
            )
            # Mu and Cov
            perm_interv_mu = perm_mu[:, num_intervened:] + torch.matmul(
                perm_intermediate_cov, perm_intermediate_mu.unsqueeze(-1)
            ).squeeze(-1)
            perm_interv_cov = perm_cov[:, num_intervened:, num_intervened:] - torch.matmul(
                perm_intermediate_cov, perm_cov[:, :num_intervened, num_intervened:]
            )

            # Adjust for floating point errors in the covariance computation to keep it symmetric
            perm_interv_cov = numerical_stability_check(
                perm_interv_cov, device=device
            )  # Uncomment if Normal throws an error. Takes some time so maybe code it more smartly

            # Sample from conditional normal
            perm_dist = MultivariateNormal(perm_interv_mu, covariance_matrix=perm_interv_cov)
            perm_mcmc_logits = (
                perm_dist.rsample([self.num_monte_carlo]).movedim(0, -1).to(torch.float32)
            )  # [bottleneck_size-num_intervened,mcmc_size]

            # Concat logits of intervened-on concepts
            perm_mcmc_logits = torch.cat(
                (
                    perm_c_intervened_logits[:, :num_intervened]
                    .unsqueeze(-1)
                    .repeat(1, 1, self.num_monte_carlo),
                    perm_mcmc_logits,
                ),
                dim=1,
            )

            # Permute back into original form and store
            indices_reversed = torch.argsort(indices)
            mcmc_logits = perm_mcmc_logits.gather(
                1,
                indices_reversed.unsqueeze(2).expand(-1, -1, perm_mcmc_logits.size(2)),
            )

            # Return conditional mu&cov
            assert (
                torch.argsort(indices[:, num_intervened:])
                == torch.arange(len(perm_interv_mu[0][:]), device=device)
            ).all()  # Check that non-intervened concepts weren't permuted s.t. no permutation of interv_mu is needed
            interv_mu = perm_interv_mu
            interv_cov = perm_interv_cov

        assert not torch.isnan(mcmc_logits).any(), "mcmc_logits contains NaN"
        assert not torch.isnan(interv_mu).any(), "interv_mu contains NaN"
        assert not torch.isnan(interv_cov).any(), "interv_cov contains NaN"
        # Compute probabilities and set intervened-on probs to 0/1
        mcmc_probs = self.act_c(mcmc_logits)

        # Set intervened-on hard concepts to 0/1
        mcmc_probs = (c_true * c_mask).unsqueeze(2).repeat(1, 1, self.num_monte_carlo) + mcmc_probs * (
            1 - c_mask
        ).unsqueeze(2).repeat(1, 1, self.num_monte_carlo)

        return interv_mu, interv_cov, mcmc_probs, mcmc_logits


SCBM_Strategy = SCBMConditionalStrategy


class PercentileStrategy:
    # Set intervened concepts to 0.05 & 0.95 probabilities
    def __init__(self):
        pass

    def _compute_intervened_probs(self, c_true, c_mask):
        return (0.05 + 0.9 * c_true) * c_mask

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_probs(c_true, c_mask)
        c_intervened_logits = torch.logit(c_intervened_probs, eps=1e-6)
        return c_intervened_logits

    def compute_intervention_cbm(self, c_pred, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_probs(c_true, c_mask)
        c_intervened = c_intervened_probs + c_pred * (1 - c_mask)
        return c_intervened


class EmpiricalPercentileStrategy:
    # Set intervened concepts to 5th and 95th percentile of training distribution
    def __init__(
        self,
        train_loader: DataLoader,
        model: nn.Module,
        device: torch.device,
        is_scbm: bool = False,
        batch_transform=None,
    ) -> None:
        concept_pred = []
        with torch.no_grad():
            for _, batch in enumerate(train_loader):
                batch_features = batch["features"].to(device, non_blocking=True)
                if batch_transform is not None:
                    batch_features = batch_transform(batch_features, train=False)
                concepts_pred_probs, _, _ = model(batch_features, validation=True)
                if is_scbm:
                    # For SCBMs, we need to average over MCMC samples
                    concepts_pred_probs = concepts_pred_probs.mean(-1)
                concept_pred.append(concepts_pred_probs.detach())
        concept_pred = torch.cat(concept_pred, dim=0)
        self.concept_pred_percentiles = torch.quantile(
            concept_pred, q=torch.tensor([0.05, 0.95], device=device), dim=0
        )

    def _compute_intervened_perc(self, c_true, c_mask):
        c_true_pred_perc = torch.where(
            c_true == 1,
            self.concept_pred_percentiles[1, :],
            self.concept_pred_percentiles[0, :],
        )
        return c_true_pred_perc * c_mask

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_perc(c_true, c_mask)
        c_intervened_logits = torch.logit(c_intervened_probs, eps=1e-6)
        return c_intervened_logits

    def compute_intervention_cbm(self, c_pred, c_true, c_mask):
        c_intervened_probs = self._compute_intervened_perc(c_true, c_mask)
        c_intervened = c_intervened_probs + c_pred * (1 - c_mask)
        return c_intervened


class ConfIntervalOptimalStrategy:
    """
    A strategy for intervening on concepts using confidence interval bounds.

    Args:
        level (float, optional): The confidence level for the confidence interval.
    """

    # Set intervened concept logits to bounds of 90% confidence interval
    def __init__(self, level=0.9):
        self.level = level

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        """
        Compute the logits for the intervened-on concepts based on the confidence interval bounds.

        This method finds values that lie on the confidence region boundary and maximize the likelihood
        of the intervened concepts.

        Args:
            c_mu (torch.Tensor): The predicted mean values of the concepts. Shape: (batch_size, num_concepts)
            c_cov (torch.Tensor): The predicted covariance matrix of the concepts. Shape: (batch_size, num_concepts, num_concepts)
            c_true (torch.Tensor): The ground-truth concept values. Shape: (batch_size, num_concepts)
            c_mask (torch.Tensor): A mask indicating which concepts are intervened-on. Shape: (batch_size, num_concepts)

        Returns:
            torch.Tensor: The logits for the intervened-on concepts, rest filled with NaN. Shape: (batch_size, num_concepts)

        Step-by-step procedure:
            - The method first separates the intervened-on concepts from the others.
            - It finds a good initial point on the confidence region boundary, that is spanned in the logit space.
                It is defined as a vector with equal magnitude in each dimension, originating from c_mu and oriented
                in the direction of the ground truth. Thus, only the scale factor of this vector needs to be found
                s.t. it lies on the confidence region boundary.
            - It defines the confidence region bounds on the logits, as well as defining some objective and derivatives
              for faster optimization.
            - It performs sample-wise constrained optimization to find the intervention logits by minimizing the concept BCE
              while ensuring they lie within the boundary of the confidence region. The starting point from before is used as
              initialization. Note that this is done sequentially for each sample, and therefore very slow.
              The optimization problem also scales with the number of intervened-on concepts. There are certainly ways to make it much faster.
            - After having found the optimal points at the confidence region bound, it permutes determined concept logits back into the original order.

        """
        from torchmin import minimize

        # Find values that lie on confidence region ball
        # Approach: Find theta s.t.  Λn(θ)= −2(ℓ(θ)−ℓ(θ^))=χ^2_{1-α,n} and minimize concept loss of intervened concepts.
        # Note, theta^ is = mu, evaluated for the N(mu,Sigma) distribution, while theta is point on the boundary of the confidence region
        # Then, we make theta by arg min Concept BCE(θ) s.t. Λn(θ) <= holds with 1-α = self.level for theta~N(0,Sigma) (not fully correct explanation, but intuition).
        n_intervened = int(c_mask.sum(1)[0].item())
        # Separate intervened-on concepts from others
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
        perm_cov = c_cov.gather(1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2)))
        perm_cov = perm_cov.gather(2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1))
        marginal_interv_cov = perm_cov[:, :n_intervened, :n_intervened]
        marginal_interv_cov = numerical_stability_check(
            marginal_interv_cov.float(), device=marginal_interv_cov.device
        ).cpu()
        target = (c_true * c_mask).gather(1, indices)[:, :n_intervened].float().cpu()
        marginal_c_mu = c_mu.gather(1, indices)[:, :n_intervened].float().cpu()
        interv_direction = (
            ((2 * c_true - 1) * c_mask).gather(1, indices)[:, :n_intervened].float().cpu()
        )  # direction
        quantile_cutoff = chi2.ppf(q=self.level, df=n_intervened)

        # Finding good init point on confidence region boundary (each dim with equal magnitude)
        dist = MultivariateNormal(torch.zeros(n_intervened), marginal_interv_cov)
        loglikeli_theta_hat = dist.log_prob(torch.zeros(n_intervened))

        def conf_region(scale):
            loglikeli_theta_star = dist.log_prob(scale * interv_direction)
            log_likelihood_ratio = -2 * (loglikeli_theta_star - loglikeli_theta_hat)
            return ((quantile_cutoff - log_likelihood_ratio) ** 2).sum(-1)

        scale = minimize(
            conf_region,
            x0=torch.ones(c_mu.shape[0], 1),
            method="bfgs",
            max_iter=50,
            tol=1e-5,
        ).x
        scale = scale.abs()  # in case negative root was found (note that both give same log-likelihood as its point-symmetric around 0)
        x0 = marginal_c_mu + (interv_direction * scale)

        # Define bounds on logits
        lb_interv = torch.where(
            interv_direction > 0, marginal_c_mu + 1e-4, torch.tensor(float("-inf"))
        )
        ub_interv = torch.where(interv_direction < 0, marginal_c_mu - 1e-4, torch.tensor(float("inf")))

        # Define confidence region
        dist_logits = MultivariateNormal(marginal_c_mu, marginal_interv_cov)
        loglikeli_theta_hat = dist_logits.log_prob(marginal_c_mu)
        loglikeli_goal = -quantile_cutoff / 2 + loglikeli_theta_hat

        # Initialize variables
        cov_inverse = torch.linalg.inv(marginal_interv_cov)
        interv_vector = torch.empty_like(marginal_c_mu)

        #### Sample-wise constrained optimization (as there are no batched functions available out-of-the-box). Can surely be optimized
        for i in range(marginal_c_mu.shape[0]):
            # Define variables required for optimization
            dist_logits_uni = MultivariateNormal(marginal_c_mu[i], marginal_interv_cov[i])
            loglikeli_goal_uni = loglikeli_goal[i]
            target_uni = target[i]
            inverse = cov_inverse[i]
            marginal = marginal_c_mu[i]

            # Define minimization objective and jacobian
            def loglikeli_bern_uni(marginal_interv_vector):
                return F.binary_cross_entropy_with_logits(
                    input=marginal_interv_vector, target=target_uni, reduction="sum"
                )

            def jac_min_fct(x):
                return torch.sigmoid(x) - target_uni

            # Define confidence region constraint and its jacobian
            def conf_region_uni(marginal_interv_vector):
                loglikeli_theta_star = dist_logits_uni.log_prob(marginal_interv_vector)
                return loglikeli_theta_star - loglikeli_goal_uni

            def jac_constraint(x):
                return -(inverse @ (x - marginal).unsqueeze(-1)).squeeze(-1)

            # Wrapper for scipy "minimize" function
            # Find intervention logits by minimizing the concept BCE s.t. they still lie on the boundary of the confidence region
            minimum = minimize_constr(
                f=loglikeli_bern_uni,
                x0=x0[i],
                jac=jac_min_fct,
                method="SLSQP",
                constr={
                    "fun": conf_region_uni,
                    "lb": 0,
                    "ub": float("inf"),
                    "jac": jac_constraint,
                },
                bounds={"lb": lb_interv[i], "ub": ub_interv[i]},
                max_iter=50,
                tol=1e-4 * n_intervened,
            )
            interv_vector[i] = minimum.x

        # Permute intervened concept logits back into original order
        indices_reversed = torch.argsort(indices)
        interv_vector_unordered = torch.full_like(
            c_mu, float("nan"), device=c_mu.device, dtype=torch.float32
        )
        interv_vector_unordered[:, :n_intervened] = interv_vector
        c_intervened_logits = interv_vector_unordered.gather(1, indices_reversed)

        return c_intervened_logits


class ConfIntervalOptimalFWStrategy:
    """
    Batched Frank-Wolfe solver for SCBM confidence-interval interventions.

    This solves the same constrained problem as ConfIntervalOptimalStrategy,
    but avoids the sample-wise SciPy SLSQP loop. The original problem is
    written over intervened logits eta':

        minimize BCEWithLogits(eta', c_true)
        subject to (eta' - mu)^T Sigma^{-1} (eta' - mu) <= chi2
                   eta'_i >= mu_i if c_i = 1
                   eta'_i <= mu_i if c_i = 0

    We reparameterize eta' = mu + direction * u, where direction is +1 for
    positive concepts and -1 for negative concepts. Then the direction
    constraints become u >= 0, and the confidence region becomes an ellipsoid
    u^T A u <= chi2 with A = D Sigma^{-1} D, where D = diag(direction).

    Frank-Wolfe keeps every iterate inside this feasible ellipsoid by
    moving along convex combinations of feasible points.

    The linear minimization oracle below uses the closed-form ellipsoid
    direction -A^{-1} grad, then clamps to the positive orthant. The clamp is an
    approximation to the exact active-set solution for u >= 0.

    Args:
        level (float): The confidence level for the confidence interval. Default: 0.99.
        steps (int): The number of Frank-Wolfe iterations to perform. Default: 100.
        line_search_points (int): The number of points to evaluate in the line
            search between the current iterate and the oracle point.
            If set to 1, uses a standard diminishing step size instead. Default: 21.
        direction_eps (float): A small positive value to ensure the intervention
            direction is valid after clamping. Default: 1e-4.
    """

    def __init__(
        self,
        level: float = 0.99,
        steps: int = 100,
        line_search_points: int = 21,
        direction_eps: float = 1e-4,
    ) -> None:
        self.level = level
        self.steps = steps
        self.line_search_points = line_search_points
        self.direction_eps = direction_eps

    def compute_intervened_logits(
        self,
        c_mu: torch.Tensor,
        c_cov: torch.Tensor,
        c_true: torch.Tensor,
        c_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_intervened = int(c_mask.sum(1)[0].item())
        num_concepts = c_cov.size(1)

        # Move currently intervened concepts to the front so every sample has a
        # compact [B, num_intervened] optimization problem. The final result is
        # scattered back to the original concept order at the end.
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
        row_indices = indices.unsqueeze(2).expand(-1, -1, num_concepts)
        col_indices = indices.unsqueeze(1).expand(-1, num_concepts, -1)
        perm_cov = c_cov.gather(1, row_indices)
        perm_cov = perm_cov.gather(2, col_indices)
        marginal_cov = perm_cov[:, :num_intervened, :num_intervened]
        marginal_cov = numerical_stability_check(marginal_cov.float(), device=marginal_cov.device)
        target = (c_true * c_mask).gather(1, indices)[:, :num_intervened].float()
        marginal_mu = c_mu.gather(1, indices)[:, :num_intervened].float()

        # direction_i is +1 for c_i=1 and -1 for c_i=0
        # so the constraint u_i >= 0 express the "increasing constraint"
        direction = ((2 * c_true - 1) * c_mask).gather(1, indices)[:, :num_intervened].float()

        # The likelihood-ratio confidence region for a Gaussian is a
        # Mahalanobis ellipsoid with radius squared given by the chi-square
        # quantile for the number of intervened concepts.
        cutoff = float(chi2.ppf(q=self.level, df=num_intervened))
        cutoff_tensor = torch.tensor(cutoff, device=c_mu.device, dtype=marginal_mu.dtype)

        # In u-coordinates:
        #   eta' - mu = direction * u
        #   (eta' - mu)^T Sigma^{-1} (eta' - mu) = u^T A u
        # where A = D Sigma^{-1} D and D = diag(direction).
        cov_inverse = torch.linalg.inv(marginal_cov)
        A = (
            cov_inverse * direction.unsqueeze(1) * direction.unsqueeze(2)
        )  # (B, num_intervened, num_intervened)

        # start from the confidence-boundary point with equal positive
        # displacement in every intervened concept.
        ones = torch.ones_like(marginal_mu)  # (B, num_intervened)
        direction_quad = (ones.unsqueeze(1) @ A @ ones.unsqueeze(2)).squeeze(-1).squeeze(-1)  # (B,)
        init_scale = torch.sqrt(cutoff_tensor / (direction_quad + 1e-12))
        u = ones * init_scale.unsqueeze(1)  # (B, num_intervened)

        eye = torch.eye(num_intervened, device=c_mu.device, dtype=marginal_mu.dtype).expand_as(A)
        line_grid = torch.linspace(
            0,
            1,
            self.line_search_points,
            device=c_mu.device,
            dtype=marginal_mu.dtype,
        )

        for step_idx in range(self.steps):
            # gradient of BCEWithLogits(mu + direction * u, target) w.r.t. u
            eta = marginal_mu + direction * u
            grad_u = direction * (torch.sigmoid(eta) - target)

            # Frank-Wolfe linear minimization oracle on the ellipsoid.
            # without u >= 0 constraint, the minimizer is proportional to -A^{-1} grad.
            v = torch.linalg.solve(
                A + 1e-6 * eye,
                -grad_u.unsqueeze(-1),
            )
            # clamp negative coordinates to a small positive
            # value to keep the intervention direction valid
            v = torch.clamp(v.squeeze(-1), min=self.direction_eps)  # (B, num_intervened)

            # rescale to the boundary of confidence region
            v_quad = (v.unsqueeze(1) @ A @ v.unsqueeze(2)).squeeze(-1).squeeze(-1)  # (B,)
            v = v * torch.sqrt(cutoff_tensor / (v_quad + 1e-12)).unsqueeze(1)

            if self.line_search_points > 1:
                # Batched grid line search between the current feasible point
                # and the oracle point. Convex combinations stay feasible
                # because the ellipsoid plus orthant is a convex set.
                candidates = (1 - line_grid.view(1, -1, 1)) * u.unsqueeze(1) + line_grid.view(
                    1, -1, 1
                ) * v.unsqueeze(1)  # (B, line_search_points, num_intervened)
                candidate_eta = marginal_mu.unsqueeze(1) + direction.unsqueeze(1) * candidates
                candidate_loss = F.binary_cross_entropy_with_logits(
                    candidate_eta,
                    target.unsqueeze(1).expand_as(candidate_eta),
                    reduction="none",
                ).sum(dim=2)  # (B, line_search_points)
                best = candidate_loss.argmin(dim=1)
                u = candidates[torch.arange(u.shape[0], device=u.device), best]  # (B, num_intervened)
            else:
                # standard diminishing Frank-Wolfe step size fallback.
                step_size = 2.0 / (step_idx + 2.0)
                u = (1 - step_size) * u + step_size * v

        # convert back from u-coordinates to intervened logits eta' and
        # restore the original concept order
        intervened_eta = marginal_mu + direction * u
        indices_reversed = torch.argsort(indices)
        sorted_intervened_logits = torch.full_like(
            c_mu, float("nan"), device=c_mu.device, dtype=torch.float32
        )
        sorted_intervened_logits[:, :num_intervened] = intervened_eta
        return sorted_intervened_logits.gather(1, indices_reversed)


class HardCBMStrategy:
    # Set intervened concepts to 0 & 1
    def __init__(self) -> None:
        pass

    def compute_intervention_cbm(self, c_pred, c_true, c_mask):
        c_intervened = c_true * c_mask + c_pred * (1 - c_mask)
        return c_intervened
