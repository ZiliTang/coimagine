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

#: The **public** event the environment broadcasts every step. This is the only channel
#: through which one robot can learn anything about the other's contribution, absent a
#: message: the teammate's own feedback symbols are private and must never be an input.
PUBLIC_SYMBOLS: Tuple[str, ...] = ("moved", "stalled")

#: Grid side length; the scan plane is ``GRID * GRID`` cells with one block each.
GRID = 6
N_CELL_BLOCKS = len(CELL_KINDS) + 1  # +1 for "unknown"

OBSERVATION_DIM = (
    GRID * GRID * N_CELL_BLOCKS + 2 + len(FEEDBACK_SYMBOLS) + len(PUBLIC_SYMBOLS)
)
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


def counts(history: Sequence[str], symbols: Sequence[str]) -> Tuple[float, ...]:
    """Normalised histogram of ``history`` over ``symbols``."""

    values = [0.0] * len(symbols)
    for symbol in history:
        if symbol in symbols:
            values[symbols.index(symbol)] += 1.0
    total = sum(values)
    if total:
        values = [v / total for v in values]
    return tuple(values)


def observation_vector(
    scan: OccupancyScan,
    own_pose: Cell,
    feedback_a: Sequence[str],
    public_events: Sequence[str],
) -> Tuple[float, ...]:
    """The observation half of the feature vector, which does **not** include the action.

    Contents, all deployment-legal for this robot: its own scan, its own pose, its own
    feedback history, and the **public** event history. The teammate's private feedback is
    deliberately absent -- it is not observable without a message, and including it would
    hand the learner the answer.
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
    features.extend(counts(public_events, PUBLIC_SYMBOLS))
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
