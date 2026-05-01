"""Strategies for encoding intervened concept values."""

import torch
from scipy.stats import chi2
from torch import nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F

from utils.minimize_constraint import minimize_constr
from utils.utils import numerical_stability_check


def define_strategy(inter_strategy, train_loader, model, device, config):
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
            inter_strategy = HardCBMStrategy()
        elif inter_strategy == "simple_perc":
            inter_strategy = PercentileStrategy()
        elif inter_strategy == "emp_perc":
            inter_strategy = EmpiricalPercentileStrategy(
                train_loader=train_loader, model=model, device=device
            )
        print("USING FOLLOWING STRATEGY:", inter_strategy.__class__.__name__)

    elif config.model.model == "scbm":
        inter_strategy = SCBM_Strategy(
            inter_strategy, train_loader, model, device, config
        )
        print(
            "USING FOLLOWING STRATEGY:", inter_strategy.interv_strat.__class__.__name__
        )
    else:
        raise NotImplementedError(
            "No such strategy as",
            config.model.inter_strategy,
            "defined for model",
            config.model.model,
            "!",
        )

    return inter_strategy


class SCBM_Strategy:
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

    def __init__(self, inter_strategy, train_loader, model, device, config):
        self.num_monte_carlo = config.model.num_monte_carlo
        self.num_concepts = config.data.num_concepts
        self.act_c = nn.Sigmoid()
        if inter_strategy == "simple_perc":
            self.interv_strat = PercentileStrategy()
        elif inter_strategy == "emp_perc":
            self.interv_strat = EmpiricalPercentileStrategy(
                train_loader=train_loader, model=model, device=device, is_scbm=True
            )
        elif inter_strategy == "conf_interval_optimal":
            self.interv_strat = ConfIntervalOptimalStrategy(level=config.model.level)
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
        num_intervened = c_mask.sum(1)[0]
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
            perm_cov = c_cov.gather(
                1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2))
            )
            perm_cov = perm_cov.gather(
                2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
            )
            perm_mu = c_mu.gather(1, indices)
            perm_c_intervened_logits = c_intervened_logits.gather(1, indices)

            # Compute mu and covariance conditioned on intervened-on concepts
            # Intermediate steps
            perm_intermediate_cov = torch.matmul(
                perm_cov[:, num_intervened:, :num_intervened],
                torch.inverse(perm_cov[:, :num_intervened, :num_intervened]),
            )
            perm_intermediate_mu = (
                perm_c_intervened_logits[:, :num_intervened]
                - perm_mu[:, :num_intervened]
            )
            # Mu and Cov
            perm_interv_mu = perm_mu[:, num_intervened:] + torch.matmul(
                perm_intermediate_cov, perm_intermediate_mu.unsqueeze(-1)
            ).squeeze(-1)
            perm_interv_cov = perm_cov[
                :, num_intervened:, num_intervened:
            ] - torch.matmul(
                perm_intermediate_cov, perm_cov[:, :num_intervened, num_intervened:]
            )

            # Adjust for floating point errors in the covariance computation to keep it symmetric
            perm_interv_cov = numerical_stability_check(
                perm_interv_cov, device=device
            )  # Uncomment if Normal throws an error. Takes some time so maybe code it more smartly

            # Sample from conditional normal
            perm_dist = MultivariateNormal(
                perm_interv_mu, covariance_matrix=perm_interv_cov
            )
            perm_mcmc_logits = (
                perm_dist.rsample([self.num_monte_carlo])
                .movedim(0, -1)
                .to(torch.float32)
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

        assert (
            (mcmc_logits.isnan()).any()
            == (interv_mu.isnan()).any()
            == (interv_cov.isnan()).any()
            == False
        )
        # Compute probabilities and set intervened-on probs to 0/1
        mcmc_probs = self.act_c(mcmc_logits)

        # Set intervened-on hard concepts to 0/1
        mcmc_probs = (c_true * c_mask).unsqueeze(2).repeat(
            1, 1, self.num_monte_carlo
        ) + mcmc_probs * (1 - c_mask).unsqueeze(2).repeat(1, 1, self.num_monte_carlo)

        return interv_mu, interv_cov, mcmc_probs, mcmc_logits


class SCBMPercentileStrategy:
    # Set intervened concept logits to 0.05 & 0.95
    def __init__(self):
        pass

    def compute_intervened_logits(self, c_mu, c_cov, c_true, c_mask):
        c_intervened_probs = (0.05 + 0.9 * c_true) * c_mask
        c_intervened_logits = torch.logit(c_intervened_probs, eps=1e-6)
        return c_intervened_logits


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
    def __init__(self, train_loader, model, device, is_scbm=False):
        concept_pred = []
        with torch.no_grad():
            for k, batch in enumerate(train_loader):
                batch_features = batch["features"].to(device)
                concepts_pred_probs, _, _ = model(
                    batch_features, epoch=-1, validation=True
                )
                if is_scbm:
                    # For SCBMs, we need to average over MCMC samples
                    concepts_pred_probs = concepts_pred_probs.mean(-1)
                concept_pred.append(concepts_pred_probs.cpu())
        concept_pred = torch.cat(concept_pred, dim=0)
        self.concept_pred_percentiles = torch.quantile(
            concept_pred, q=torch.tensor([0.05, 0.95]), dim=0
        ).to(device)

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
        n_intervened = c_mask.sum(1)[0]
        # Separate intervened-on concepts from others
        indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
        perm_cov = c_cov.gather(1, indices.unsqueeze(2).expand(-1, -1, c_cov.size(2)))
        perm_cov = perm_cov.gather(
            2, indices.unsqueeze(1).expand(-1, c_cov.size(1), -1)
        )
        marginal_interv_cov = perm_cov[:, :n_intervened, :n_intervened]
        marginal_interv_cov = numerical_stability_check(
            marginal_interv_cov.float(), device=marginal_interv_cov.device
        ).cpu()
        target = (c_true * c_mask).gather(1, indices)[:, :n_intervened].float().cpu()
        marginal_c_mu = c_mu.gather(1, indices)[:, :n_intervened].float().cpu()
        interv_direction = (
            ((2 * c_true - 1) * c_mask)
            .gather(1, indices)[:, :n_intervened]
            .float()
            .cpu()
        )  # direction
        quantile_cutoff = chi2.ppf(q=self.level, df=n_intervened.cpu())

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
        scale = (
            scale.abs()
        )  # in case negative root was found (note that both give same log-likelihood as its point-symmetric around 0)
        x0 = marginal_c_mu + (interv_direction * scale)

        # Define bounds on logits
        lb_interv = torch.where(
            interv_direction > 0, marginal_c_mu + 1e-4, torch.tensor(float("-inf"))
        )
        ub_interv = torch.where(
            interv_direction < 0, marginal_c_mu - 1e-4, torch.tensor(float("inf"))
        )

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
            dist_logits_uni = MultivariateNormal(
                marginal_c_mu[i], marginal_interv_cov[i]
            )
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
                tol=1e-4 * n_intervened.cpu(),
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


class HardCBMStrategy:
    # Set intervened concepts to 0 & 1
    def __init__(self):
        pass

    def compute_intervention_cbm(self, c_pred, c_true, c_mask):
        c_intervened = c_true * c_mask + c_pred * (1 - c_mask)
        return c_intervened
