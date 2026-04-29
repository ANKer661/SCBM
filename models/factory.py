"""Model factory."""

import torch

from models.cbm import CBM
from models.scbm import SCBM


def create_model(config):
    """
    Parse the configuration file and return a relevant model.
    """
    model = None
    if config.model.model not in ["cbm", "scbm"]:
        print("Could not create model with name ", config.model, "!")
        quit()
    if config.model.model == "cbm":
        model = CBM(config)
    elif config.model.model == "scbm":
        model = SCBM(config)
    if config.model.compile:
        model = torch.compile(model)
    return model
