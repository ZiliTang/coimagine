"""fit_capacity_v1: training-budget and model-width diagnostics on frozen data."""

from .existing import flatten, inventory, load_training_data
from .metrics import evaluate, loss_components

__all__ = ["evaluate", "flatten", "inventory", "load_training_data", "loss_components"]
