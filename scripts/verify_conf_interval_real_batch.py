"""Compare confidence-interval solvers on real SCBM forward outputs.

This is a verifier only. It builds the configured SCBM and dataloader, runs one
real batch through the model, then compares the current SciPy SLSQP intervention
solver with the production batched Frank-Wolfe strategy.

Usage:

python scripts/verify_conf_interval_real_batch.py \
  +model=SCBM \
  +data=synthetic \
  model.encoder_arch=FCNN \
  workers=2 \
  model.val_batch_size=512 \
  +verify.batch_size=512 \
  +verify.max_interventions=100 \
  +verify.steps=100 \
  +verify.device=cuda \
  +verify.reference=true \
  +verify.intervention_stride=10
"""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from scipy.stats import chi2

from datasets.datamodule import build_dataloaders
from interventions.strategies import ConfIntervalOptimalFWStrategy, ConfIntervalOptimalStrategy
from models.factory import create_model
from training.batch_transforms import create_batch_transform
from training.epoch import apply_batch_transform, move_batch_to_device
from utils.utils import numerical_stability_check, reset_random_seeds


def _log(message: str) -> None:
    print(f"[verify] {message}", flush=True)


def _get_empirical_covariance(dataloader) -> torch.Tensor:
    data = []
    for batch in dataloader:
        data.append(batch["concepts"])
    data = torch.cat(data)
    data_logits = torch.logit(0.05 + 0.9 * data)
    covariance = torch.cov(data_logits.transpose(0, 1))
    covariance = numerical_stability_check(covariance, device="cpu")
    return torch.linalg.cholesky(covariance)


def _setup_scbm(config: DictConfig, train_loader, device: torch.device) -> torch.nn.Module:
    model = create_model(config)
    if config.model.model != "scbm":
        raise ValueError("This verifier requires +model=SCBM.")

    if config.model.get("cov_type") == "empirical":
        model.sigma_concepts = _get_empirical_covariance(train_loader).to(device)
    elif config.model.get("cov_type") == "global":
        lower_triangle = _get_empirical_covariance(train_loader).to(device)
        rows, cols = torch.tril_indices(
            row=config.data.num_concepts,
            col=config.data.num_concepts,
            offset=0,
            device=device,
        )
        model.sigma_concepts = torch.nn.Parameter(lower_triangle[rows, cols])
        diag_idx = rows == cols
        with torch.no_grad():
            model.sigma_concepts[diag_idx] = (
                lower_triangle[rows, cols][diag_idx].expm1().clamp_min(1e-6).log()
            )

    checkpoint = config.get("verify", {}).get("checkpoint") if config.get("verify") else None
    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def _make_order(
    batch_size: int,
    num_concepts: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(0)
    orders = []
    for _ in range(batch_size):
        orders.append(torch.randperm(num_concepts, generator=generator, device=device))
    return torch.stack(orders)


def _make_prefix_mask(order: torch.Tensor, num_intervened: int) -> torch.Tensor:
    mask = torch.zeros_like(order, dtype=torch.float32)
    mask.scatter_(1, order[:, :num_intervened], 1.0)
    return mask


def _step_values(num_concepts: int, verify_config) -> list[int]:
    explicit_steps = verify_config.get("intervention_steps")
    if explicit_steps is not None:
        return [int(step) for step in explicit_steps]

    max_interventions = int(verify_config.get("max_interventions", num_concepts))
    max_interventions = min(max_interventions, num_concepts)
    stride = int(verify_config.get("intervention_stride", 1))
    values = list(range(0, max_interventions + 1, stride))
    if values[-1] != max_interventions:
        values.append(max_interventions)
    return values


def _resolve_output_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _run_loop_solver(
    name: str,
    solver,
    c_mu: torch.Tensor,
    c_cov: torch.Tensor,
    c_true: torch.Tensor,
    order: torch.Tensor,
    step_values: list[int],
    device: torch.device,
) -> tuple[float, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    outputs = []
    step_times = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_start = time.perf_counter()
    for num_intervened in step_values:
        c_mask = _make_prefix_mask(order, num_intervened)
        if num_intervened == 0:
            outputs.append(torch.full_like(c_mu, float("nan")).detach().cpu())
            step_times.append(0.0)
            continue
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_start = time.perf_counter()
        output = solver(c_mask)
        if device.type == "cuda":
            torch.cuda.synchronize()
        step_times.append(time.perf_counter() - step_start)
        outputs.append(output.detach().cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_sec = time.perf_counter() - total_start
    print(
        f"[verify] {name} loop total_sec={total_sec:.6f}, "
        f"mean_step_sec={sum(step_times) / len(step_times):.6f}, "
        f"steps={len(step_values)}",
        flush=True,
    )
    return total_sec, outputs[-1].to(device), torch.tensor(step_times), outputs


def _solver_diagnostics(
    logits: torch.Tensor,
    c_mu: torch.Tensor,
    c_cov: torch.Tensor,
    c_true: torch.Tensor,
    c_mask: torch.Tensor,
    level: float,
) -> dict[str, float]:
    num_intervened = int(c_mask.sum(1)[0].item())
    indices = torch.argsort(c_mask, dim=1, descending=True, stable=True)
    num_concepts = c_cov.size(1)
    row_indices = indices.unsqueeze(2).expand(-1, -1, num_concepts)
    col_indices = indices.unsqueeze(1).expand(-1, num_concepts, -1)

    perm_cov = c_cov.gather(1, row_indices)
    perm_cov = perm_cov.gather(2, col_indices)
    marginal_cov = perm_cov[:, :num_intervened, :num_intervened]
    marginal_cov = numerical_stability_check(marginal_cov.float(), device=marginal_cov.device)
    marginal_mu = c_mu.gather(1, indices)[:, :num_intervened].float()
    target = (c_true * c_mask).gather(1, indices)[:, :num_intervened].float()
    direction = ((2 * c_true - 1) * c_mask).gather(1, indices)[:, :num_intervened].float()
    pred = logits.gather(1, indices)[:, :num_intervened].float()

    delta = pred - marginal_mu
    cov_inverse = torch.linalg.inv(marginal_cov)
    maha = (delta.unsqueeze(1) @ cov_inverse @ delta.unsqueeze(2)).squeeze(-1).squeeze(-1)
    cutoff = float(chi2.ppf(q=level, df=num_intervened))
    objective = F.binary_cross_entropy_with_logits(pred, target, reduction="none").sum(dim=1)
    direction_margin = direction * delta

    return {
        "objective_mean": objective.mean().item(),
        "objective_max": objective.max().item(),
        "maha_mean": maha.mean().item(),
        "maha_max": maha.max().item(),
        "maha_cutoff": cutoff,
        "maha_violation_mean": torch.relu(maha - cutoff).mean().item(),
        "maha_violation_max": torch.relu(maha - cutoff).max().item(),
        "direction_min": direction_margin.min().item(),
        "direction_violation_count": (direction_margin < 0).sum().item(),
    }


def _write_loop_diff_curve(
    reference_outputs: list[torch.Tensor],
    candidate_outputs: list[torch.Tensor],
    order: torch.Tensor,
    c_mu: torch.Tensor,
    c_cov: torch.Tensor,
    c_true: torch.Tensor,
    step_values: list[int],
    csv_path: Path,
    level: float,
) -> list[dict[str, float]]:
    """Write per-step solver differences and optimization diagnostics."""
    order = order.cpu()
    c_mu = c_mu.detach().cpu()
    c_cov = c_cov.detach().cpu()
    c_true = c_true.detach().cpu()
    rows = []
    for num_intervened, reference, candidate in zip(
        step_values, reference_outputs, candidate_outputs
    ):
        if num_intervened == 0:
            rows.append(
                {
                    "num_intervened": num_intervened,
                    "abs_diff_mean": 0.0,
                    "abs_diff_median": 0.0,
                    "abs_diff_max": 0.0,
                    "slsqp_objective_mean": 0.0,
                    "slsqp_objective_max": 0.0,
                    "slsqp_maha_mean": 0.0,
                    "slsqp_maha_max": 0.0,
                    "slsqp_maha_cutoff": 0.0,
                    "slsqp_maha_violation_mean": 0.0,
                    "slsqp_maha_violation_max": 0.0,
                    "slsqp_direction_min": 0.0,
                    "slsqp_direction_violation_count": 0,
                    "fw_objective_mean": 0.0,
                    "fw_objective_max": 0.0,
                    "fw_maha_mean": 0.0,
                    "fw_maha_max": 0.0,
                    "fw_maha_cutoff": 0.0,
                    "fw_maha_violation_mean": 0.0,
                    "fw_maha_violation_max": 0.0,
                    "fw_direction_min": 0.0,
                    "fw_direction_violation_count": 0,
                }
            )
            continue
        mask = _make_prefix_mask(order, num_intervened)
        mask_bool = mask.bool()
        reference_valid = reference[mask_bool].view(reference.shape[0], -1)
        candidate_valid = candidate[mask_bool].view(candidate.shape[0], -1)
        diff = (candidate_valid - reference_valid).abs()
        reference_diag = _solver_diagnostics(reference, c_mu, c_cov, c_true, mask, level)
        candidate_diag = _solver_diagnostics(candidate, c_mu, c_cov, c_true, mask, level)
        rows.append(
            {
                "num_intervened": num_intervened,
                "abs_diff_mean": diff.mean().item(),
                "abs_diff_median": diff.median().item(),
                "abs_diff_max": diff.max().item(),
                **{f"slsqp_{key}": value for key, value in reference_diag.items()},
                **{f"fw_{key}": value for key, value in candidate_diag.items()},
            }
        )

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    _log(f"wrote per-step diff csv: {csv_path}")

    return rows


def _write_loop_timing_curve(
    step_values: list[int],
    candidate_step_times: torch.Tensor,
    csv_path: Path,
    reference_step_times: torch.Tensor | None = None,
) -> None:
    """Write per-step and cumulative solver timings."""
    candidate_step_times = candidate_step_times.cpu()
    candidate_cumulative = torch.cumsum(candidate_step_times, dim=0)
    if reference_step_times is not None:
        reference_step_times = reference_step_times.cpu()
        reference_cumulative = torch.cumsum(reference_step_times, dim=0)
    else:
        reference_cumulative = None

    rows = []
    for idx, num_intervened in enumerate(step_values):
        row = {
            "num_intervened": num_intervened,
            "fw_step_sec": candidate_step_times[idx].item(),
            "fw_cumulative_sec": candidate_cumulative[idx].item(),
        }
        if reference_step_times is not None:
            row["slsqp_step_sec"] = reference_step_times[idx].item()
            row["slsqp_cumulative_sec"] = reference_cumulative[idx].item()
        rows.append(row)

    fieldnames = ["num_intervened", "fw_step_sec", "fw_cumulative_sec"]
    if reference_step_times is not None:
        fieldnames.extend(["slsqp_step_sec", "slsqp_cumulative_sec"])

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _log(f"wrote per-step timing csv: {csv_path}")



@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    verify_config = config.get("verify", {})
    device_name = verify_config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    gen = reset_random_seeds(config.seed)

    # 1. Build the real experiment objects so the solver sees actual SCBM
    # outputs rather than random covariance matrices.
    _log("building dataloaders")
    dataloaders = build_dataloaders(config, gen)
    batch_transform = create_batch_transform(config)
    if batch_transform is not None:
        batch_transform.to(device)

    _log("building SCBM model")
    model = _setup_scbm(config, dataloaders.train, device)
    loader_name = verify_config.get("loader", "test")
    loader = getattr(dataloaders, loader_name)

    _log(f"loading first {loader_name} batch")
    batch = next(iter(loader))
    batch = move_batch_to_device(batch, device)
    batch = apply_batch_transform(batch, batch_transform, train=False)
    if batch.features is None:
        raise ValueError("SCBM verification requires feature tensors.")

    max_batch_size = verify_config.get("batch_size")
    if max_batch_size is not None:
        batch.features = batch.features[:max_batch_size]
        batch.concepts = batch.concepts[:max_batch_size]
        batch.targets = batch.targets[:max_batch_size]

    # 2. Run one SCBM forward pass and extract the intervention solver inputs:
    # c_mu is the concept-logit mean, c_cov is the concept-logit covariance.
    _log("running SCBM forward")
    with torch.no_grad():
        _, c_mu, c_triang_cov, _ = model(
            batch.features,
            validation=True,
            c_true=batch.concepts,
            return_full=True,
        )
        c_cov = torch.matmul(c_triang_cov, c_triang_cov.transpose(1, 2))

    c_true = batch.concepts.float()
    step_values = _step_values(config.data.num_concepts, verify_config)
    order = _make_order(c_mu.shape[0], config.data.num_concepts, device)

    run_reference = bool(verify_config.get("reference", True))
    reference_sec = 0.0
    reference_outputs = None
    reference_step_times = None
    print(
        "[verify] simulating intervention loop with steps: "
        f"{step_values[:5]}{'...' if len(step_values) > 5 else ''}",
        flush=True,
    )

    if run_reference:
        _log("running SciPy SLSQP reference loop")
        reference_strategy = ConfIntervalOptimalStrategy(level=config.model.level)

        def reference_solver(loop_mask):
            return reference_strategy.compute_intervened_logits(c_mu, c_cov, c_true, loop_mask)

        (
            reference_sec,
            _,
            reference_step_times,
            reference_outputs,
        ) = _run_loop_solver(
            "SciPy SLSQP reference",
            reference_solver,
            c_mu,
            c_cov,
            c_true,
            order,
            step_values,
            device,
        )
        print(
            f"[verify] reference first/last step sec: "
            f"{reference_step_times[0].item():.6f}/{reference_step_times[-1].item():.6f}",
            flush=True,
        )

    _log("running production Frank-Wolfe strategy loop")
    fw_strategy = ConfIntervalOptimalFWStrategy(
        level=config.model.level,
        steps=int(verify_config.get("steps", 100)),
        line_search_points=int(verify_config.get("line_search_points", 21)),
        direction_eps=float(verify_config.get("direction_eps", 1e-4)),
    )

    def fw_solver(loop_mask):
        return fw_strategy.compute_intervened_logits(c_mu, c_cov, c_true, loop_mask)

    (
        candidate_sec,
        _,
        candidate_step_times,
        candidate_outputs,
    ) = _run_loop_solver(
        "Production Frank-Wolfe strategy",
        fw_solver,
        c_mu,
        c_cov,
        c_true,
        order,
        step_values,
        device,
    )
    print(
        f"[verify] candidate first/last step sec: "
        f"{candidate_step_times[0].item():.6f}/{candidate_step_times[-1].item():.6f}",
        flush=True,
    )

    timing_prefix = (
        f"verification/conf_interval_timing_curve_"
        f"{config.data.dataset}_{config.model.cov_type}"
    )
    timing_csv_path = _resolve_output_path(
        verify_config.get("timing_curve_csv", f"{timing_prefix}.csv")
    )
    _write_loop_timing_curve(
        step_values,
        candidate_step_times,
        timing_csv_path,
        reference_step_times=reference_step_times,
    )

    if run_reference:
        # 4a. Compare every curve point, not just the final one. The CSV also
        # records objective and constraint diagnostics for both solvers.
        default_prefix = (
            f"verification/conf_interval_diff_curve_"
            f"{config.data.dataset}_{config.model.cov_type}"
        )
        csv_path = _resolve_output_path(
            verify_config.get("diff_curve_csv", f"{default_prefix}.csv")
        )
        diff_rows = _write_loop_diff_curve(
            reference_outputs,
            candidate_outputs,
            order,
            c_mu,
            c_cov,
            c_true,
            step_values,
            csv_path,
            config.model.level,
        )
        final_diff = diff_rows[-1]

    print(
        "Config: "
        f"dataset={config.data.dataset}, cov_type={config.model.cov_type}, "
        f"device={device}, loader={loader_name}, batch_size={c_mu.shape[0]}, "
        f"num_concepts={config.data.num_concepts}, "
        f"num_intervened={step_values[-1]}, "
        f"reference={run_reference}"
    )
    print(f"\nProduction Frank-Wolfe strategy sec: {candidate_sec:.6f}")
    if run_reference:
        print(f"SciPy SLSQP reference sec: {reference_sec:.6f}")
        print(
            "Final-step abs diff: "
            f"mean={final_diff['abs_diff_mean']:.6f}, "
            f"median={final_diff['abs_diff_median']:.6f}, "
            f"max={final_diff['abs_diff_max']:.6f}"
        )
        print(f"\nSpeedup candidate/reference: {reference_sec / max(candidate_sec, 1e-12):.2f}x")


if __name__ == "__main__":
    main()
