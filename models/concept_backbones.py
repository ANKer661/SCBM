"""Shared encoder and head builders for concept models."""

import os

import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models as tv_models


class FCNNEncoder(nn.Module):
    """
    Defining the concept encoder for the synthetic dataset.
    """

    def __init__(self, num_inputs: int, num_hidden: int, num_deep: int):
        super(FCNNEncoder, self).__init__()

        self.fc0 = nn.Linear(num_inputs, num_hidden)
        self.bn0 = nn.BatchNorm1d(num_hidden)
        self.fcs = nn.ModuleList([nn.Linear(num_hidden, num_hidden) for _ in range(num_deep)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(num_deep)])
        self.dp = nn.Dropout(0.05)

    def forward(self, x):
        z = self.bn0(self.dp(F.relu(self.fc0(x))))
        for bn, fc in zip(self.bns, self.fcs):
            z = bn(self.dp(F.relu(fc(z))))
        return z


def build_encoder(config, config_model):
    if config_model.encoder_arch == "FCNN":
        n_features = 256
        encoder = FCNNEncoder(num_inputs=config.data.num_covariates, num_hidden=n_features, num_deep=2)
        return encoder, n_features, None

    if config_model.encoder_arch == "resnet18":
        encoder_res = tv_models.resnet18(weights=None)
        encoder_res.load_state_dict(
            torch.load(
                os.path.join(config_model.model_directory, "resnet/resnet18-5c106cde.pth"),
                weights_only=False,
            )
        )
        n_features = encoder_res.fc.in_features
        encoder_res.fc = nn.Identity()  # type: ignore
        encoder = nn.Sequential(encoder_res)
        return encoder, n_features, encoder_res

    if config_model.encoder_arch == "simple_CNN":
        n_features = 256
        encoder = nn.Sequential(
            nn.Conv2d(3, 32, 5, 3),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Flatten(),
            nn.Linear(9216, n_features),
            nn.ReLU(),
        )
        return encoder, n_features, None

    raise NotImplementedError("ERROR: architecture not supported!")


def build_head(input_dim, pred_dim, head_arch):
    if head_arch == "linear":
        fc_y = nn.Linear(input_dim, pred_dim)
        return nn.Sequential(fc_y)

    fc1_y = nn.Linear(input_dim, 256)
    fc2_y = nn.Linear(256, pred_dim)
    return nn.Sequential(fc1_y, nn.ReLU(), fc2_y)
