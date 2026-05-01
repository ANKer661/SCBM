"""Evaluation flows for interventions on SCBMs and baselines."""

import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import wandb

from interventions.policies import define_policy
from interventions.strategies import define_strategy
from utils.utils import numerical_stability_check


def _concat_stored_tensors(stored_tensors) -> list[torch.Tensor]:
    return [
        torch.cat([sublist[i] for sublist in stored_tensors], dim=0)
        for i in range(len(stored_tensors[0]))
    ]


def _make_intervention_loader(intervention_dataset, config) -> DataLoader:
    return DataLoader(
        intervention_dataset,
        batch_size=config.model.val_batch_size,
        num_workers=config.workers,
        shuffle=False,
    )


def _log_intervention_metrics(metrics, config, strategy, policy, num_intervened, define_metrics=False):
    metrics_dict = metrics.compute(validation=True, config=config)

    if define_metrics:
        wandb.define_metric("intervention/num_concepts_intervened")

    for key, value in metrics_dict.items():
        if define_metrics:
            wandb.define_metric(
                f"intervention_{strategy}_{policy}/{key}",
                step_metric="intervention/num_concepts_intervened",
            )
        wandb.log(
            {
                f"intervention_{strategy}_{policy}/{key}": value,
                "intervention/num_concepts_intervened": num_intervened,
            }
        )

    prints = f"Intervention on {num_intervened} concepts: "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()


def _build_intervention_components(strategy, policy, train_loader, model, device, config):
    try:
        intervention_policy = define_policy(policy)
        intervention_strategy = define_strategy(strategy, train_loader, model, device, config)
    except:
        print(
            f"Intervention strategy {strategy} with policy {policy} not implemented for model {config.model.model}."
        )
        return None, None

    return intervention_policy, intervention_strategy


def _collect_scbm_intervention_dataset(
    test_loader, model, metrics, epoch, config, loss_fn, device, intervention_strategy
):
    intervention_dataset_base = []
    intervention_dataset_fixed = []

    with torch.no_grad():
        for k, batch in enumerate(test_loader):
            batch_features, target_true = batch["features"].to(device), batch["labels"].to(device)
            concepts_true = batch["concepts"].to(device)
            (
                concepts_mcmc_probs,
                mu,
                triang_cov,
                target_pred_logits,
            ) = model(batch_features, epoch, validation=True, return_full=True)

            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
                concepts_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                triang_cov,
            )

            concepts_pred_probs = concepts_mcmc_probs.mean(-1)
            triang_cov = triang_cov.to(torch.float64)
            c_mu = mu.to(torch.float64)
            c_cov = torch.matmul(
                triang_cov,
                torch.transpose(triang_cov, dim0=1, dim1=2),
            )
            c_cov = numerical_stability_check(c_cov, device=device)
            c_cov_norm = torch.norm(c_cov) / (c_cov.numel() ** 0.5)

            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs,
                cov_norm=c_cov_norm,
                prec_loss=prec_loss,
            )

            (
                _,
                _,
                c_mcmc_probs,
                _,
            ) = intervention_strategy.compute_intervention(
                c_mu,
                c_cov,
                concepts_true,
                torch.zeros_like(concepts_true, device=concepts_true.device),
            )
            concepts_pred_probs = c_mcmc_probs.mean(-1)

            intervention_dataset_base.append(
                [
                    c_mu.cpu(),
                    c_cov.cpu(),
                    concepts_pred_probs.cpu(),
                ]
            )
            intervention_dataset_fixed.append(
                [
                    c_mu.cpu(),
                    c_cov.cpu(),
                    concepts_true.cpu(),
                    target_true.cpu(),
                ]
            )

    intervention_dataset_base = _concat_stored_tensors(intervention_dataset_base)
    intervention_dataset_fixed = _concat_stored_tensors(intervention_dataset_fixed)
    return intervention_dataset_base, intervention_dataset_fixed


def _run_scbm_intervention_step(
    intervention_dataset,
    intervention_dataset_fixed,
    intervention_policy,
    intervention_strategy,
    model,
    metrics,
    config,
    loss_fn,
    device,
):
    updated_intervention_dataset = []
    intervention_loader = _make_intervention_loader(intervention_dataset, config)

    with torch.no_grad():
        for k, batch in tqdm(enumerate(intervention_loader), leave=True, position=0):
            (
                c_mu,
                c_cov,
                concepts_pred_probs,
                concepts_mask,
                c_mu_original,
                c_cov_original,
                concepts_true,
                target_true,
            ) = [item.to(device) for item in batch]

            concepts_mask_new = intervention_policy.compute_intervention_mask(
                concepts_mask,
                concepts_pred_probs=concepts_pred_probs,
                mu=c_mu,
                cov=c_cov,
            )

            (
                c_interv_mu,
                c_interv_cov,
                c_mcmc_probs,
                c_mcmc_logits,
            ) = intervention_strategy.compute_intervention(
                c_mu_original,
                c_cov_original,
                concepts_true,
                concepts_mask_new,
            )

            target_pred_logits = model.intervene(c_mcmc_probs, c_mcmc_logits)

            target_loss, concepts_loss, prec_loss, total_loss = loss_fn(
                c_mcmc_probs,
                concepts_true,
                target_pred_logits,
                target_true,
                c_interv_cov,
                cov_not_triang=True,
            )

            concepts_interv_probs = c_mcmc_probs.mean(-1)
            c_norm = torch.norm(c_interv_cov) / (c_interv_cov.numel() ** 0.5)
            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_interv_probs,
                cov_norm=c_norm,
                prec_loss=prec_loss,
            )

            updated_intervention_dataset.append(
                [
                    c_interv_mu.cpu(),
                    c_interv_cov.cpu(),
                    concepts_interv_probs.cpu(),
                    concepts_mask_new.cpu(),
                ]
            )

    return TensorDataset(
        *_concat_stored_tensors(updated_intervention_dataset),
        *intervention_dataset_fixed,
    )


def _collect_cbm_intervention_dataset(test_loader, model, metrics, epoch, config, loss_fn, device):
    intervention_dataset_base = []

    with torch.no_grad():
        for k, batch in tqdm(enumerate(test_loader), leave=True, position=0):
            batch_features, target_true = batch["features"].to(device), batch["labels"].to(device)
            concepts_true = batch["concepts"].to(device)

            (
                concepts_pred_probs,
                target_pred_logits,
                concepts_hard,
            ) = model(batch_features, epoch, validation=True)
            if config.model.concept_learning == "autoregressive":
                concepts_pred_probs_m = torch.mean(concepts_pred_probs, dim=-1)
            else:
                concepts_pred_probs_m = concepts_pred_probs

            target_loss, concepts_loss, total_loss = loss_fn(
                concepts_pred_probs_m,
                concepts_true,
                target_pred_logits,
                target_true,
            )

            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_pred_probs_m,
            )
            intervention_dataset_base.append(
                [
                    concepts_pred_probs.cpu(),
                    concepts_true.cpu(),
                    target_true.cpu(),
                    batch_features.cpu(),
                    concepts_hard.cpu(),
                    concepts_pred_probs_m.cpu(),
                ]
            )

    return _concat_stored_tensors(intervention_dataset_base)


def _run_cbm_intervention_step(
    intervention_dataset_base,
    concepts_dataset_mask,
    intervention_policy,
    intervention_strategy,
    model,
    metrics,
    config,
    loss_fn,
    device,
):
    intervention_dataset = TensorDataset(*intervention_dataset_base, concepts_dataset_mask)
    intervention_loader = _make_intervention_loader(intervention_dataset, config)
    concepts_dataset_mask_new = []

    with torch.no_grad():
        for k, batch in tqdm(enumerate(intervention_loader), leave=True, position=0):
            (
                concepts_pred_probs,
                concepts_true,
                target_true,
                input_features,
                concepts_hard,
                concepts_pred_probs_m,
                concepts_mask,
            ) = [item.to(device) for item in batch]

            if config.model.concept_learning == "autoregressive":
                concepts_mask_new = intervention_policy.compute_intervention_mask(
                    concepts_mask,
                    concepts_pred_probs=concepts_pred_probs_m,
                )
                concept_probs, concepts_interv_probs = model.intervene_ar(
                    concepts_true.unsqueeze(-1).expand(-1, -1, concepts_hard.shape[-1]),
                    concepts_mask_new.unsqueeze(-1).expand(-1, -1, concepts_hard.shape[-1]),
                    input_features,
                )
                target_pred_logits = model.intervene(
                    concepts_interv_probs,
                    concepts_mask_new.unsqueeze(-1).expand(-1, -1, concepts_hard.shape[-1]),
                    input_features,
                    concepts_pred_probs,
                )
                concepts_interv_probs = torch.mean(concept_probs, dim=-1)
            else:
                concepts_mask_new = intervention_policy.compute_intervention_mask(
                    concepts_mask,
                    concepts_pred_probs=concepts_pred_probs,
                )
                concepts_interv_probs = intervention_strategy.compute_intervention_cbm(
                    concepts_pred_probs,
                    concepts_true,
                    concepts_mask_new,
                )
                target_pred_logits = model.intervene(
                    concepts_interv_probs,
                    concepts_mask_new,
                    input_features,
                    concepts_pred_probs,
                )

            target_loss, concepts_loss, total_loss = loss_fn(
                concepts_interv_probs,
                concepts_true,
                target_pred_logits,
                target_true,
            )

            metrics.update(
                target_loss,
                concepts_loss,
                total_loss,
                target_true,
                target_pred_logits,
                concepts_true,
                concepts_interv_probs,
            )
            concepts_dataset_mask_new.append(concepts_mask_new)

    return torch.cat(concepts_dataset_mask_new, dim=0).cpu()


def intervene_scbm(train_loader, test_loader, model, metrics, epoch, config, loss_fn, device):
    """
    Compute the efficacy of intervening on a model using different intervention strategies and policies for SCBMs.

    This function evaluates the efficacy of intervening on a model using various intervention strategies and policies.
    It performs interventions on the model's predicted concepts and computes the change in performance after intervention.
    The function logs the metrics at each step of the intervention process into wandb.
    Note that multiple comma-separated strategies and policies can be passed in the config file, and the function will
    iterate over all combinations.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data. Used for computing empirical percentiles.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The model to be evaluated.
        metrics (object): An object to track and compute metrics.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and data settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None
    """
    model.eval()
    policies = config.model.inter_policy.split(",")
    strategies = config.model.inter_strategy.split(",")
    num_interventions = min(200, config.data.num_concepts)

    # Intervening with different strategies
    first_intervention = True
    for strategy in strategies:
        # Intervening with different policies
        for policy in policies:
            intervention_policy, intervention_strategy = _build_intervention_components(
                strategy, policy, train_loader, model, device, config
            )
            if intervention_policy is None:
                continue

            ## One full model pass without interventions to set up the dataset required at each intervention step
            intervention_dataset_base, intervention_dataset_fixed = _collect_scbm_intervention_dataset(
                test_loader, model, metrics, epoch, config, loss_fn, device, intervention_strategy
            )

            # Calculate and log metrics
            _log_intervention_metrics(
                metrics, config, strategy, policy, num_intervened=0, define_metrics=first_intervention
            )
            first_intervention = False

            ## Computing intervention curves using stored concept predictions
            # Preparing dataset
            # Initializing concepts with 0's
            intervention_dataset = TensorDataset(
                *intervention_dataset_base,
                torch.zeros_like(intervention_dataset_fixed[-2]),
                *intervention_dataset_fixed,
            )

            # Performing interventions
            for num_intervened in range(1, num_interventions + 1):
                intervention_dataset = _run_scbm_intervention_step(
                    intervention_dataset,
                    intervention_dataset_fixed,
                    intervention_policy,
                    intervention_strategy,
                    model,
                    metrics,
                    config,
                    loss_fn,
                    device,
                )

                # Calculate and log metrics
                _log_intervention_metrics(
                    metrics, config, strategy, policy, num_intervened=num_intervened
                )
    return


def intervene_cbm(train_loader, test_loader, model, metrics, epoch, config, loss_fn, device):
    """
    Compute the efficacy of intervening on a model using different intervention strategies and policies for baselines.

    This function evaluates the efficacy of intervening on a model using various intervention strategies and policies.
    It performs interventions on the model's predicted concepts and computes the change in performance after intervention.
    The function logs the metrics at each step of the intervention process into wandb.
    Note that multiple comma-separated strategies and policies can be passed in the config file, and the function will
    iterate over all combinations.

    Args:
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data. Used for computing empirical percentiles.
        test_loader (torch.utils.data.DataLoader): DataLoader for the test data.
        model (torch.nn.Module): The model to be evaluated.
        metrics (object): An object to track and compute metrics.
        epoch (int): The current epoch number.
        config (dict): Configuration dictionary containing model and data settings.
        loss_fn (callable): The loss function used to compute losses.
        device (torch.device): The device to run the computations on.

    Returns:
        None
    """
    model.eval()
    policies = config.model.inter_policy.split(",")
    strategies = config.model.inter_strategy.split(",")
    num_interventions = min(200, config.data.num_concepts)
    if config.model.model == "cbm" and config.model.concept_learning in (
        "hard",
        "autoregressive",
        "embedding",
    ):
        strategies = ["hard"]
    # Intervening with different strategies
    first_intervention = True
    for strategy in strategies:
        # Intervening with different policies
        for policy in policies:
            intervention_policy, intervention_strategy = _build_intervention_components(
                strategy, policy, train_loader, model, device, config
            )
            if intervention_policy is None:
                continue

            # One full model pass without interventions
            intervention_dataset_base = _collect_cbm_intervention_dataset(
                test_loader, model, metrics, epoch, config, loss_fn, device
            )

            # Calculate and log metrics
            _log_intervention_metrics(
                metrics, config, strategy, policy, num_intervened=0, define_metrics=first_intervention
            )
            first_intervention = False

            # Computing intervention curves using stored concept predictions
            # Preparing dataset
            concepts_dataset_mask = torch.zeros_like(intervention_dataset_base[1])

            for num_intervened in range(1, num_interventions + 1):
                concepts_dataset_mask = _run_cbm_intervention_step(
                    intervention_dataset_base,
                    concepts_dataset_mask,
                    intervention_policy,
                    intervention_strategy,
                    model,
                    metrics,
                    config,
                    loss_fn,
                    device,
                )

                # Calculate and log metrics
                _log_intervention_metrics(
                    metrics, config, strategy, policy, num_intervened=num_intervened
                )
    return
