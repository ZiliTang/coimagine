"""Fixed candidate sequences for the diagnostic (CONTRACT section 4).

The generator uses only the initial public state and a seed. It never calls the model, never
rolls the true dynamics, never injects the acceptance witness, and never adapts templates to
model results. Half the sequences are uniform random over all 81 joint actions, half come
from fixed templates; the pool is deduplicated, deterministically shuffled, and nested
prefixes give the candidate sets.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .contract import (
    DIRECTION,
    MOVE_ACTIONS,
    N_ACTIONS_PER_ROBOT,
    N_JOINT_ACTIONS,
    PUSH_ACTIONS,
    WAIT_ACTION,
    GridMap,
    GridState,
    encode_joint,
)
from .data import qualified_pushers


@dataclass(frozen=True)
class CandidateConfig:
    max_sequences: int = 256
    random_fraction: float = 0.5
    seed: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {"max_sequences": self.max_sequences, "random_fraction": self.random_fraction, "seed": self.seed}


def _push_action(direction: Tuple[int, int]) -> int:
    return {DIRECTION[a]: a for a in PUSH_ACTIONS}[direction]


def _single_push(robot_id: int, direction: Tuple[int, int]) -> int:
    action = _push_action(direction)
    return encode_joint(action if robot_id == 0 else WAIT_ACTION, action if robot_id == 1 else WAIT_ACTION)


def _double_push(direction: Tuple[int, int]) -> int:
    action = _push_action(direction)
    return encode_joint(action, action)


def template_sequences(state: GridState, depth: int) -> List[Tuple[int, ...]]:
    """A small fixed family of interpretable sequences (deterministic from the state)."""

    sequences: List[Tuple[int, ...]] = []
    for robot_id, direction in qualified_pushers(state):
        action = _single_push(robot_id, direction)
        sequences.append(tuple([action] * depth))
    for direction in DIRECTION.values():
        sequences.append(tuple([_double_push(direction)] * depth))
    # move templates: each robot moves in a fixed direction, or both wait
    for action in MOVE_ACTIONS:
        sequences.append(tuple([encode_joint(action, WAIT_ACTION)] * depth))
        sequences.append(tuple([encode_joint(WAIT_ACTION, action)] * depth))
    sequences.append(tuple([encode_joint(WAIT_ACTION, WAIT_ACTION)] * depth))
    return sequences


def candidate_sequences(
    state: GridState,
    grid_map: GridMap,
    depth: int,
    k: int,
    seed: int,
    config: CandidateConfig | None = None,
) -> Tuple[Tuple[int, ...], ...]:
    """Nested candidate sets: prefixes of one fixed shuffled, deduplicated order."""

    config = config or CandidateConfig(seed=seed)
    rng = random.Random(seed)
    cap = min(k, config.max_sequences)
    if depth == 1:
        cap = min(cap, N_JOINT_ACTIONS)

    n_random = int(round(cap * config.random_fraction))
    n_template = cap - n_random

    templates = template_sequences(state, depth)
    rng.shuffle(templates)
    pool: List[Tuple[int, ...]] = []
    seen = set()
    for sequence in templates[:max(n_template, 0)]:
        if sequence not in seen:
            seen.add(sequence)
            pool.append(sequence)

    attempts = 0
    while len(pool) < cap and attempts < cap * 50:
        attempts += 1
        sequence = tuple(rng.randrange(N_JOINT_ACTIONS) for _ in range(depth))
        if sequence not in seen:
            seen.add(sequence)
            pool.append(sequence)
    # if the template share was short, top up with more random sequences
    rng.shuffle(pool)
    return tuple(pool[:cap])
