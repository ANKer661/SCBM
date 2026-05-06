"""PyTorch dataloader construction."""

from dataclasses import dataclass

from torch.utils.data import DataLoader, Dataset

from datasets.cifar10_dataset import build_CIFAR10_CBM_datasets
from datasets.cifar100_dataset import build_CIFAR100_CBM_datasets
from datasets.CUB_dataset import build_CUB_datasets
from datasets.synthetic_dataset import get_synthetic_datasets


@dataclass(frozen=True)
class DatasetSplits:
    train: Dataset
    val: Dataset
    test: Dataset


def validate_data_path(config_data) -> None:
    if config_data.dataset == "synthetic":
        return
    if "data_path" not in config_data or config_data.data_path is None:
        raise ValueError(f"data.data_path must be set for dataset {config_data.dataset!r}.")


def build_datasets(config_base, config_data) -> DatasetSplits:
    validate_data_path(config_data)

    if config_data.dataset == "synthetic":
        print("SYNTHETIC DATASET")
        sim_type = None
        if "sim_type" in config_data:
            sim_type = config_data.sim_type
            print("SIMULATION TYPE: " + str(sim_type))
            if config_data.num_classes > 2:
                raise NotImplementedError(
                    "ERROR: Only binary classification is supported for synthetic data."
                )
        trainset, validset, testset = get_synthetic_datasets(
            num_vars=config_data.num_covariates,
            num_points=config_data.num_points,
            num_predicates=config_data.num_concepts,
            train_ratio=0.6,
            val_ratio=0.2,
            sim_type=sim_type,
            seed=config_base.seed,
        )
    elif config_data.dataset == "CUB":
        print("CUB DATASET")
        trainset, validset, testset = build_CUB_datasets(config_data)
    elif config_data.dataset == "cifar10":
        print("CIFAR-10 DATASET")
        trainset, validset, testset = build_CIFAR10_CBM_datasets(config_data.data_path)
    elif config_data.dataset == "cifar100":
        print("CIFAR-100 DATASET")
        trainset, validset, testset = build_CIFAR100_CBM_datasets(config_data.data_path)
    else:
        raise NotImplementedError("ERROR: Dataset not supported!")

    return DatasetSplits(train=trainset, val=validset, test=testset)


@dataclass(frozen=True)
class DataLoaders:
    train: DataLoader
    target_train: DataLoader | None
    val: DataLoader
    test: DataLoader


class ConceptOnlyDataset(Dataset):
    """Return labels/concepts without calling the image feature transform path."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore

    def __getitem__(self, index: int) -> dict:
        if hasattr(self.dataset, "get_concept_only_item"):
            return self.dataset.get_concept_only_item(index)  # type: ignore
        raise TypeError(f"{type(self.dataset).__name__} does not support concept-only batches.")


def should_build_target_train_loader(config) -> bool:
    return (
        config.model.concept_learning == "autoregressive"
        and config.model.training_mode == "independent"
    )


def build_dataloaders(config, gen) -> DataLoaders:
    datasets = build_datasets(config, config.data)

    train_loader = DataLoader(
        datasets.train,
        batch_size=config.model.train_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        drop_last=True,
        persistent_workers=config.workers > 0,
    )
    target_train_loader = None
    if should_build_target_train_loader(config):
        target_train_loader = DataLoader(
            ConceptOnlyDataset(datasets.train),
            batch_size=config.model.train_batch_size,
            shuffle=True,
            # Concept-only batches are cheap and avoid image transforms. Keeping
            # this loader in-process also avoids an extra persistent worker pool
            # during the target-head-only stage.
            num_workers=0,
            pin_memory=True,
            generator=gen,
            drop_last=True,
        )
    val_loader = DataLoader(
        datasets.val,
        batch_size=config.model.val_batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        generator=gen,
        persistent_workers=config.workers > 0,
    )
    test_loader = DataLoader(
        datasets.test,
        batch_size=config.model.val_batch_size,
        num_workers=config.workers,
        generator=gen,
    )

    return DataLoaders(
        train=train_loader,
        target_train=target_train_loader,
        val=val_loader,
        test=test_loader,
    )
