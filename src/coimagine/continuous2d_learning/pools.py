"""Fixed candidate pools for the ranking diagnostic (contract C2).

The pool depends only on the start's public observation and a seed. It never calls the
model, never rolls the true physics, never injects an exact witness, and never regenerates a
later-window parameter from a predicted or true future state.
"""

from __future__ import annotations

import json
import random
from typing import Dict, List, Sequence, Tuple

from ..continuous2d.candidates import JointCandidate, joint_candidates
from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.skills import Hold, JointSpec, MoveTo, PushContact, Observation

MAX_POOL = 128
REPEATED_ROOT_SHARE = 32
TEMPLATE_SHARE = 32


def root_library(observation: Observation, config: Continuous2DConfig) -> List[JointCandidate]:
    """The existing per-observation joint-candidate list, unchanged."""

    return joint_candidates(observation, config.candidates, config.control, config.physics)


def _spec_key(joint_spec: JointSpec) -> str:
    from ..continuous2d.skills import joint_spec_to_dicts

    return json.dumps(joint_spec_to_dicts(joint_spec), sort_keys=True)


def _sequence_key(sequence: Sequence[JointSpec]) -> str:
    return "||".join(_spec_key(spec) for spec in sequence)


def _push_options(candidates: Sequence[JointCandidate]) -> List[JointSpec]:
    return [candidate.skills for candidate in candidates if any(isinstance(skill, PushContact) for skill in candidate.skills)]


def _staging_options(candidates: Sequence[JointCandidate]) -> List[JointSpec]:
    """Joint specs in which both robots move to a staging point (the template's first window)."""

    return [
        candidate.skills
        for candidate in candidates
        if isinstance(candidate.skills[0], MoveTo) and isinstance(candidate.skills[1], MoveTo)
    ]


def _template_sequence(
    candidates: Sequence[JointCandidate], depth: int, rng: random.Random
) -> JointSpec | None:
    """Both sides ``move_to`` a staging point for one window, then ``push_contact``.

    Staging points and pushes come from the existing root library; the combination rule uses
    only public geometry.
    """

    moves = _staging_options(candidates)
    pushes = _push_options(candidates)
    if not moves or not pushes:
        return None
    staging = rng.choice(moves)
    sequence = [staging]
    for _ in range(depth - 1):
        sequence.append(rng.choice(pushes))
    return tuple(sequence)  # type: ignore[return-value]


def candidate_pool(
    observation: Observation,
    config: Continuous2DConfig,
    depth: int,
    seed: int,
    *,
    cap: int = MAX_POOL,
) -> List[Tuple[JointSpec, ...]]:
    """One frozen pool; smaller ``K`` are prefixes of this order."""

    root = root_library(observation, config)
    unique: List[JointSpec] = []
    seen_root = set()
    for candidate in root:
        key = _spec_key(candidate.skills)
        if key not in seen_root:
            seen_root.add(key)
            unique.append(candidate.skills)

    rng = random.Random(seed)
    pool: List[Tuple[JointSpec, ...]] = []
    seen = set()

    def _add(sequence: Sequence[JointSpec]) -> None:
        key = _sequence_key(sequence)
        if key not in seen:
            seen.add(key)
            pool.append(tuple(sequence))

    if depth == 1:
        shuffled = list(unique)
        rng.shuffle(shuffled)
        for spec in shuffled[:cap]:
            _add((spec,))
        return pool

    # 1. up to 32 distinct root joint skills, each repeated `depth` times
    sample = list(unique)
    rng.shuffle(sample)
    for spec in sample[:REPEATED_ROOT_SHARE]:
        _add(tuple([spec] * depth))

    # 2. up to 32 generic staging-then-push templates
    for _ in range(TEMPLATE_SHARE):
        template = _template_sequence(root, depth, rng)
        if template is not None:
            _add(template)

    # 3. fill with uniform samples from the root library; this also tops up a template
    #    shortfall, so the pool is never dominated by one repeated skill
    attempts = 0
    while len(pool) < cap and attempts < cap * 40 and unique:
        attempts += 1
        _add(tuple(rng.choice(unique) for _ in range(depth)))

    rng.shuffle(pool)
    return pool[:cap]
