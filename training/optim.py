"""Optimizer and scheduler construction for staged training."""

import torch

from utils.training import create_optimizer


def build_optimizer(config_model, model):
    return create_optimizer(config_model, model)


def build_scheduler(config_model, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config_model.decrease_every,
        gamma=1 / config_model.lr_divisor,
    )
