"""Learning probe for Stage 2: can a model discover the contact modes from data?

This package exists to answer one question, and only that question:

    given only what a robot can legally observe, can a learned model predict which
    joint action will work -- including the cases where the reason it fails is an
    invisible contact mode (the teammate is slipping, or the object is latched)?

Everything here is training-side. Deployment still uses the frozen protocol: the
features are built from the robot's own scan, its own feedback history and public
information. Privileged quantities are used **as labels and decoder targets only**,
never as inputs.

The environment is small and tabular, so the model is a hand-written NumPy MLP: no
heavy dependency, and the whole computation stays inspectable.
"""

from .features import (
    ACTION_IDS,
    FEATURE_DIM,
    feedback_counts,
    observation_vector,
    sample_vector,
)
from .data import FeasibilitySample, generate_dataset, sample_to_arrays
from .mlp import MultiHeadMLP, TrainHistory, train

__all__ = [
    "ACTION_IDS",
    "FEATURE_DIM",
    "FeasibilitySample",
    "MultiHeadMLP",
    "TrainHistory",
    "feedback_counts",
    "generate_dataset",
    "observation_vector",
    "sample_to_arrays",
    "sample_vector",
    "train",
]
