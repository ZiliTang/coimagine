"""Deployment-legal features for the learning probe.

Every input here is something a robot genuinely has: its own ray scan, its own
contact-feedback history, public robot poses and the public candidate template. The
true object pose and the true contact physics appear only as **labels** (see
:mod:`coimagine.learning.data`).
"""

from __future__ import annotations

from typing import Sequence, Tuple

from ..grid.dynamics import FEEDBACK_SYMBOLS
from ..grid.sensing import (
    KIND_FREE,
    KIND_OBJECT,
    KIND_ROBOT,
    KIND_WALL,
    OccupancyScan,
)
from ..grid.state import Cell

#: The public candidate template the probe scores, in a fixed order.
ACTION_IDS: Tuple[str, ...] = (
    "push:0:E",
    "push:1:E",
    "co_push:E",
    "probe:0:E",
    "release:0:E",
    "wait",
)

#: Cell kinds encoded in the scan plane. Index 0 of each cell's block means "unknown".
CELL_KINDS: Tuple[str, ...] = (KIND_FREE, KIND_WALL, KIND_OBJECT, KIND_ROBOT)

#: Grid side length; the scan plane is ``GRID * GRID`` cells with one block each.
GRID = 6
N_CELL_BLOCKS = len(CELL_KINDS) + 1  # +1 for "unknown"

OBSERVATION_DIM = GRID * GRID * N_CELL_BLOCKS + 2 + 2 * len(FEEDBACK_SYMBOLS)
FEATURE_DIM = OBSERVATION_DIM + len(ACTION_IDS)


def feedback_counts(history: Sequence[str]) -> Tuple[float, ...]:
    """Normalised histogram over the four feedback symbols."""

    counts = [0.0] * len(FEEDBACK_SYMBOLS)
    for symbol in history:
        if symbol in FEEDBACK_SYMBOLS:
            counts[FEEDBACK_SYMBOLS.index(symbol)] += 1.0
    total = sum(counts)
    if total:
        counts = [c / total for c in counts]
    return tuple(counts)


def observation_vector(
    scan: OccupancyScan,
    own_pose: Cell,
    feedback_a: Sequence[str],
    feedback_b: Sequence[str],
) -> Tuple[float, ...]:
    """The observation half of the feature vector, which does **not** include the action.

    Keeping the action out of this half is what lets the same encoder feed the
    privileged decoder heads (object pose, contact physics) and the action head.
    """

    plane = [0.0] * (GRID * GRID * N_CELL_BLOCKS)
    for (x, y), kind in scan.observed.items():
        if not (0 <= x < GRID and 0 <= y < GRID):
            continue
        block = CELL_KINDS.index(kind) + 1 if kind in CELL_KINDS else 0
        plane[(y * GRID + x) * N_CELL_BLOCKS + block] = 1.0
    features = list(plane)
    features.append(own_pose[0] / (GRID - 1))
    features.append(own_pose[1] / (GRID - 1))
    features.extend(feedback_counts(feedback_a))
    features.extend(feedback_counts(feedback_b))
    assert len(features) == OBSERVATION_DIM, (len(features), OBSERVATION_DIM)
    return tuple(features)


def action_vector(action_id: str) -> Tuple[float, ...]:
    return tuple(1.0 if action_id == candidate else 0.0 for candidate in ACTION_IDS)


def sample_vector(
    scan: OccupancyScan,
    own_pose: Cell,
    feedback_a: Sequence[str],
    feedback_b: Sequence[str],
    action_id: str,
) -> Tuple[float, ...]:
    return observation_vector(scan, own_pose, feedback_a, feedback_b) + action_vector(action_id)
