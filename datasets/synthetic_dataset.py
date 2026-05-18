"""
Functions for generating nonlinear synthetic data. It is used to generated the correlated concepts dataset.

This module provides functions to generate synthetic datasets with nonlinear relationships between covariates, concepts, and labels.
It includes utilities for creating random nonlinear mappings, generating synthetic data, and constructing dataset objects for training, validation, and testing.

Classes:
    SyntheticDataset: Custom Dataset class for the nonlinear synthetic data.

Functions:
    random_nonlin_map: Creates a random nonlinear function parameterized by an MLP.
    generate_synthetic_data_correlated_c: Generates a synthetic dataset with correlated concepts.
    get_synthetic_datasets: Constructs dataset objects for the synthetic data.
"""

import numpy as np
from numpy.random import multivariate_normal

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_low_rank_matrix

import torch
from torch.utils.data import Dataset


def random_nonlin_map(n_in, n_out, n_hidden, rank=100):
    """
    Create a random nonlinear function parameterized by an MLP.
    This is used to create the mapping between concepts and covariates.

    Args:
        n_in (int): Number of input features.
        n_out (int): Number of output features.
        n_hidden (int): Number of hidden units.
        rank (int, optional): Effective rank for the low-rank matrices. Default is 100.

    Returns:
        function: A random nonlinear mapping.
    """
    # Random MLP mapping
    W_0 = make_low_rank_matrix(n_in, n_hidden, effective_rank=rank)
    W_1 = make_low_rank_matrix(n_hidden, n_hidden, effective_rank=rank)
    W_2 = make_low_rank_matrix(n_hidden, n_out, effective_rank=rank)
    # No biases: ?
    b_0 = np.random.uniform(0, 0, (1, n_hidden))
    b_1 = np.random.uniform(0, 0, (1, n_hidden))
    b_2 = np.random.uniform(0, 0, (1, n_out))

    def nlin_map(x):
        h1 = np.maximum(0, x @ W_0 + np.tile(b_0, (x.shape[0], 1)))
        h2 = np.maximum(0, h1 @ W_1 + np.tile(b_1, (x.shape[0], 1)))

        return h2 @ W_2 + np.tile(b_2, (x.shape[0], 1))

    return nlin_map


def generate_synthetic_data_correlated_c(p: int, n: int, k: int, seed: int):
    """
    Generate a synthetic dataset with correlated concepts.

    Args:
        p (int): Number of covariates.
        n (int): Number of data points.
        k (int): Number of concepts.
        seed (int): Random generator seed.

    Returns:
        tuple: A tuple containing the design matrix (X), concept values (c), and labels (y).
    """
    # Generative process: x <-- z --> c --> y

    # Replicability
    np.random.seed(seed)

    # Generate concepts
    # Generate a k x 5 matrix of random values from a standard normal distribution
    W = np.random.randn(k, 10)

    # Compute the product W * W' (W times its transpose) and add a diagonal matrix of random values
    S = np.dot(W, W.T) + np.diag(np.random.rand(k))

    sigma = S

    try:
        torch.linalg.cholesky(torch.tensor(sigma))
    except RuntimeError:
        print("Matrix not positive definite")

    z = multivariate_normal(mean=np.zeros((k,)), cov=sigma, size=n)
    c = (z >= 0) * 1  # (n, k)

    # Generate balanced labels from concepts
    lin_weights = np.random.uniform(size=(1, k))
    lin_weights = np.tile(lin_weights, (n, 1))
    y = np.sum(lin_weights * c, 1, keepdims=True)
    tmp = np.median(y, 0)
    y = (y >= tmp) * 1  # (n, 1)

    # Nonlinear maps. The number of hidden units is fixed to 5 to make the task harder
    g = random_nonlin_map(
        n_in=k,
        n_out=p,
        n_hidden=10,
    )
    # Generate covariates
    # Concept groups get mapped to one
    X = g(z)  # (n, p)

    ss = StandardScaler()
    X = ss.fit_transform(X)
    # Add Gaussian noise after standardization to ensure similar effect on all covariates and have intuition on how much noise is added
    X = X + np.random.normal(0, 0.5, X.shape)
    ss = StandardScaler()
    X = ss.fit_transform(X)

    return X, c, y


class SyntheticDataset(Dataset):
    """
    Dataset class for the nonlinear synthetic data
    """

    def __init__(
        self,
        num_vars: int,
        num_points: int,
        num_predicates: int,
        sim_type: str | None = None,
        indices: np.ndarray | None = None,
        seed: int = 42,
        data: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> None:
        """
        Initialize the SyntheticDataset.

        Args:
            num_vars (int): Number of covariates.
            num_points (int): Number of data points.
            num_predicates (int): Number of concepts.
            sim_type (str, optional): Type of synthetic data to generate. Default is None.
            indices (numpy.ndarray, optional): Indices of the data points to be kept. Default is None.
            seed (int, optional): Random generator seed. Default is 42.
        """
        # Shall a partial predicate set be used?
        # self.predicate_idx = np.arange(0, num_predicates)
        if sim_type == "correlated_c":
            generate_synthetic_data = generate_synthetic_data_correlated_c
        else:
            raise ValueError("Simulation type not implemented!")

        if data is None:
            X, c, y = generate_synthetic_data(p=num_vars, n=num_points, k=num_predicates, seed=seed)
        else:
            X, c, y = data

        if indices is not None:
            X = X[indices]
            c = c[indices]
            y = y[indices]

        self.X = torch.as_tensor(X, dtype=torch.float32)  # (n, p)
        self.c = torch.as_tensor(c[:, :num_predicates], dtype=torch.float32)  # (n, k)
        self.y = torch.as_tensor(y[:, 0], dtype=torch.long)  # (n,)

    def __getitem__(self, index) -> dict[str, torch.Tensor]:
        """
        Returns points from the dataset

        @param index: index
        @return: a dictionary with the data; dict['features'] contains features, dict['label'] contains
        target labels, dict['concepts'] contains concepts
        """
        labels = self.y[index]
        concepts = self.c[index]
        features = self.X[index]

        return {"features": features, "labels": labels, "concepts": concepts}

    def get_concept_only_item(self, index) -> dict[str, torch.Tensor]:
        return {"labels": self.y[index], "concepts": self.c[index]}

    def __len__(self) -> int:
        return self.X.size(0)


def get_synthetic_datasets(
    num_vars: int,
    num_points: int,
    num_predicates: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    seed: int = 42,
    sim_type: str | None = None,
) -> tuple[SyntheticDataset, SyntheticDataset, SyntheticDataset]:
    """
    Construct dataset objects for the synthetic data.

    Args:
        num_vars (int): Number of covariates.
        num_points (int): Number of data points.
        num_predicates (int): Number of concepts.
        train_ratio (float, optional): Ratio of training set size. Default is 0.6.
        val_ratio (float, optional): Ratio of validation set size. Default is 0.2.
        seed (int, optional): Random generator seed. Default is 42.
        sim_type (str, optional): Type of synthetic data to generate. Default is None.

    Returns:
        tuple: Dataset objects for the training, validation, and test sets.
    """
    # Train-validation-test split
    indices_train, indices_valtest = train_test_split(
        np.arange(0, num_points), train_size=train_ratio, random_state=seed
    )
    indices_val, indices_test = train_test_split(
        indices_valtest,
        train_size=val_ratio / (1.0 - train_ratio),
        random_state=2 * seed,
    )

    if sim_type == "correlated_c":
        generate_synthetic_data = generate_synthetic_data_correlated_c
    else:
        raise ValueError("Simulation type not implemented!")

    data = generate_synthetic_data(
        p=num_vars,
        n=num_points,
        k=num_predicates,
        seed=seed,
    )

    # Datasets
    synthetic_datasets = {
        "train": SyntheticDataset(
            num_vars=num_vars,
            num_points=num_points,
            num_predicates=num_predicates,
            indices=indices_train,
            seed=seed,
            sim_type=sim_type,
            data=data,
        ),
        "val": SyntheticDataset(
            num_vars=num_vars,
            num_points=num_points,
            num_predicates=num_predicates,
            indices=indices_val,
            seed=seed,
            sim_type=sim_type,
            data=data,
        ),
        "test": SyntheticDataset(
            num_vars=num_vars,
            num_points=num_points,
            num_predicates=num_predicates,
            indices=indices_test,
            seed=seed,
            sim_type=sim_type,
            data=data,
        ),
    }

    return (
        synthetic_datasets["train"],
        synthetic_datasets["val"],
        synthetic_datasets["test"],
    )
