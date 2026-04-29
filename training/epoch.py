"""Generic epoch loops for staged training."""

from tqdm import tqdm
import torch
import wandb

from training.adapters import BatchTensors


def move_batch_to_device(batch, device):
    return BatchTensors(
        features=batch["features"].to(device),
        targets=batch["labels"].to(device),
        concepts=batch["concepts"].to(device),
    )


def train_one_epoch(loader, adapter, optimizer, stage, metrics, epoch, device):
    adapter.prepare_train(stage)
    metrics.reset()

    last_batch_idx = -1
    for last_batch_idx, batch in enumerate(
        tqdm(loader, desc=f"Epoch {epoch + 1}", position=0, leave=True)
    ):
        batch = move_batch_to_device(batch, device)
        output = adapter.forward_train(batch, epoch, stage)
        losses = adapter.compute_loss(output, batch)

        optimizer.zero_grad()
        adapter.backward_loss(losses, stage).backward()
        optimizer.step()

        adapter.update_metrics(metrics, losses, batch, output)

    metrics_dict = metrics.compute()
    wandb.log({f"train/{k}": v for k, v in metrics_dict.items()})
    prints = f"Epoch {epoch + 1}, Train     : "
    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    metrics.reset()

    if last_batch_idx == -1:
        raise ValueError("Cannot train on an empty dataloader.")


def validate_one_epoch(
    loader,
    adapter,
    metrics,
    epoch,
    config,
    device,
    test=False,
    concept_names_graph=None,
):
    adapter.model.eval()
    metrics.reset()

    last_batch_idx = -1
    with torch.no_grad():
        for last_batch_idx, batch in enumerate(
            tqdm(loader, desc=f"Epoch {epoch}", position=0, leave=True)
        ):
            batch = move_batch_to_device(batch, device)
            output = adapter.forward_eval(batch, epoch)

            if test:
                adapter.maybe_plot_test_batch(
                    output,
                    batch,
                    last_batch_idx,
                    len(loader),
                    config,
                    concept_names_graph,
                )

            losses = adapter.compute_loss(output, batch)
            adapter.update_metrics(metrics, losses, batch, output)

    if last_batch_idx == -1:
        raise ValueError("Cannot validate on an empty dataloader.")

    metrics_dict = metrics.compute(validation=True, config=config)
    if not test:
        wandb.log({f"validation/{k}": v for k, v in metrics_dict.items()})
        prints = f"Epoch {epoch}, Validation: "
    else:
        wandb.log({f"test/{k}": v for k, v in metrics_dict.items()})
        prints = "Test: "

    for key, value in metrics_dict.items():
        prints += f"{key}: {value:.3f} "
    print(prints)
    print()
    metrics.reset()
