"""Compatibility imports for concept models."""

from models.cbm import CBM
from models.factory import create_model
from models.scbm import SCBM

__all__ = ["CBM", "SCBM", "create_model"]
