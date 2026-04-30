"""Optimizer and scheduler construction for staged training."""

import torch


def build_optimizer(config_model, model):
    """
    Parse the model configuration and return an optimizer for trainable parameters.
    """
    assert config_model.optimizer in [
        "sgd",
        "adam",
    ], "Only SGD and Adam optimizers are available!"

    optim_params = [
        {
            "params": filter(lambda p: p.requires_grad, model.parameters()),
            "lr": config_model.learning_rate,
            "weight_decay": config_model.weight_decay,
        }
    ]

    if config_model.optimizer == "sgd":
        return torch.optim.SGD(optim_params)
    return torch.optim.Adam(optim_params)


def build_scheduler(config_model, optimizer):
    return torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config_model.decrease_every,
        gamma=1 / config_model.lr_divisor,
    )
