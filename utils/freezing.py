"""Module parameter freezing helpers."""


def freeze_module(module) -> None:
    module.eval()
    for param in module.parameters():
        param.requires_grad = False


def unfreeze_module(module) -> None:
    module.train()
    for param in module.parameters():
        param.requires_grad = True
