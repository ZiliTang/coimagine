"""Single-scale continuous world model and fixed-candidate diagnostics (task book
20261012_2007, phases B–D).

Everything here is new experiment code on top of the frozen ``continuous2d`` platform. The
platform itself is not modified: this package only reads observations/snapshots, builds
deterministic features, trains one plain MLP per seed, and evaluates fixed candidate
sequences with the new scalar score.
"""

from .contract import (
    SCORE_VERSION,
    TERMINATION_CLASSES,
    ScoreComponents,
    arena_diagonal,
    geometric_violation,
    score_components,
    termination_class,
    wrap_angle,
)
from .features import (
    INPUT_DIM,
    STATE_DIM,
    BoundaryState,
    apply_state_delta,
    encode_input,
    encode_state_target,
)
from .model import (
    JointWindowMLP,
    ModelConfig,
    WindowPrediction,
    load_model,
    predict_window,
    train_model,
)
from .records import (
    SPLIT_SCENES,
    CollectionConfig,
    collect_dataset,
    collect_episode,
    coverage,
    episode_seed,
)

__all__ = [
    "BoundaryState",
    "CollectionConfig",
    "INPUT_DIM",
    "JointWindowMLP",
    "ModelConfig",
    "SCORE_VERSION",
    "SPLIT_SCENES",
    "STATE_DIM",
    "ScoreComponents",
    "TERMINATION_CLASSES",
    "WindowPrediction",
    "apply_state_delta",
    "arena_diagonal",
    "collect_dataset",
    "collect_episode",
    "coverage",
    "encode_input",
    "encode_state_target",
    "episode_seed",
    "geometric_violation",
    "load_model",
    "predict_window",
    "score_components",
    "termination_class",
    "train_model",
    "wrap_angle",
]
