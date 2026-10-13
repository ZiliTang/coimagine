"""Phase B root generation and joint-skill command sets.

Frozen by ``docs/option_prediction_mechanism_v1/CONTRACT.md`` section 2.1 (data unit and
scale) and by the task book section 3 (B1). Two things live here and nothing else:

1. **Roots.** A root is one legal physical initial state (object pose + both robot poses) in
   one of the six existing interaction classes. The existing generator
   ``coimagine.local_contact_option.roots.sample_root`` is reused unchanged; phase B only
   supplies a *new salt* (so the train/val/test seed families are disjoint from each other
   and from every earlier round) and a *distance rejection* that guarantees no two roots in
   different splits are geometric near-duplicates.

2. **Commands.** One root gets up to eight *joint skill commands*. A command is a genuinely
   different legal control choice from the frozen ranges ``ALPHA_CHOICES`` /
   ``SPEED_CHOICES`` of the existing generator -- not eight random seeds of the same action.
   Command selection uses only public information (the root seed and the frozen choice
   lists); it never consults object ground truth to pick "the best" command.

The eight perturbation ``VariantSpec``\\ s of the old round are **not** used here: the eight
action branches are the commands, and every phase-B execution uses
``VariantSpec("baseline")`` (no added range noise, no observation delay).
"""

from __future__ import annotations

import hashlib
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

import coimagine.local_contact_option.roots as _roots
from coimagine.local_contact_option.roots import (
    ALPHA_CHOICES,
    CLASSES,
    RootSpec,
    SPEED_CHOICES,
    sample_root,
)

# ---------------------------------------------------------------------------
# frozen phase-B parameters
# ---------------------------------------------------------------------------

#: New salt. ``sample_root`` mixes the module-level ``ROOT_SALT`` into every seed, so a new
#: salt gives train/val/test completely disjoint seed families (and phase B differs from the
#: old dev/test roots as well). The value is part of every ``roots.json`` artifact.
SALT_B = "option-prediction-mechanism-v1/phase_b"

#: Tier-1 scale: train 256 / val 64 / test 64 roots, at most 8 commands each (<= 3072).
SPLIT_ROOTS: Dict[str, int] = {"train": 256, "val": 64, "test": 64}
N_COMMANDS = 8

#: Geometric near-duplicate guard. Two roots are "the same initial-state family" when the
#: combined metric below is below ``MIN_ROOT_DISTANCE`` metres. The metric takes the maximum
#: of: object-centre distance, both robot-centre distances, and each angular difference
#: converted to metres by ``ANG_SCALE`` (a 0.5 m lever arm, i.e. the object's half-diagonal).
MIN_ROOT_DISTANCE = 0.02
ANG_SCALE = 0.5

CMD_SALT = 0xB0C0
#: Deterministic index stream for the 2-active-robot construction. Robot 0 walks this list
#: of eight distinct (alpha_idx, speed_idx) pairs; robot 1 walks a rotated copy.
_INDEX_PAIRS: Tuple[Tuple[int, int], ...] = tuple(
    (k % 3, (k // 3) % 3) for k in range(N_COMMANDS)
)


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


@contextmanager
def _new_salt(value: str):
    """Temporarily install the phase-B salt in the (unmodified) existing generator."""

    saved = _roots.ROOT_SALT
    _roots.ROOT_SALT = value
    try:
        yield
    finally:
        _roots.ROOT_SALT = saved


# ---------------------------------------------------------------------------
# roots
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingReport:
    split: str
    planned: int
    sampled: int
    per_class: Dict[str, int]
    sample_none: int          # indices where sample_root could not fill the cell
    leakage_rejected: int     # candidates rejected for being too close to an accepted root
    unfilled: Dict[str, int]  # per class: roots that could not be produced within the cap


def per_class_counts(total: int) -> Dict[str, int]:
    """Balanced six-class split of ``total``, extra roots to the earlier classes."""

    base, remainder = divmod(total, len(CLASSES))
    return {cls: base + (1 if index < remainder else 0)
            for index, cls in enumerate(CLASSES)}


def root_distance(a: RootSpec, b: RootSpec) -> float:
    """Combined geometric distance between two initial states, in metres.

    ``max`` over: object-centre distance, robot-centre distances, and each angular difference
    scaled by ``ANG_SCALE``. It is a *conservative* family test: near in any channel means
    near in the metric.
    """

    parts = [
        math.hypot(a.object_pose.x - b.object_pose.x, a.object_pose.y - b.object_pose.y),
        ANG_SCALE * abs(wrap_angle(a.object_pose.angle - b.object_pose.angle)),
    ]
    for robot_id in (0, 1):
        pa, pb = a.robot_poses[robot_id], b.robot_poses[robot_id]
        parts.append(math.hypot(pa.x - pb.x, pa.y - pb.y))
        parts.append(ANG_SCALE * abs(wrap_angle(pa.angle - pb.angle)))
    return max(parts)


def root_position_distance(a: RootSpec, b: RootSpec) -> float:
    """Position-only part of :func:`root_distance`, reported alongside it."""

    parts = [
        math.hypot(a.object_pose.x - b.object_pose.x, a.object_pose.y - b.object_pose.y),
    ]
    for robot_id in (0, 1):
        pa, pb = a.robot_poses[robot_id], b.robot_poses[robot_id]
        parts.append(math.hypot(pa.x - pb.x, pa.y - pb.y))
    return max(parts)


def _phase_b_root_id(split: str, interaction_class: str, index: int) -> str:
    return f"phase_b/{split}/{interaction_class}/{index:04d}"


def sample_split_roots(
    split: str,
    total: int,
    *,
    forbidden: Sequence[RootSpec] = (),
    min_distance: float = MIN_ROOT_DISTANCE,
    max_index_factor: int = 20,
) -> Tuple[List[RootSpec], SamplingReport]:
    """Deterministically sample ``total`` roots for one split, rejecting near-duplicates.

    ``forbidden`` holds roots already accepted in *other* splits; a candidate closer than
    ``min_distance`` to any of them is rejected and the next deterministic index is tried.
    The same test is applied within the split so one split does not repeat a family either.
    A ``sample_root`` miss (``None``) is counted, never retried at the same index, and never
    silently swallowed; the index cap bounds the work.
    """

    counts = per_class_counts(total)
    accepted: List[RootSpec] = []
    per_class: Dict[str, int] = {}
    sample_none = 0
    leakage_rejected = 0
    unfilled: Dict[str, int] = {}
    with _new_salt(SALT_B):
        for interaction_class in CLASSES:
            want = counts[interaction_class]
            got = 0
            index = 0
            index_cap = max(1, want * max_index_factor)
            while got < want and index < index_cap:
                candidate = sample_root(split, interaction_class, index)
                index += 1
                if candidate is None:
                    sample_none += 1
                    continue
                candidate = replace(
                    candidate,
                    root_id=_phase_b_root_id(split, interaction_class, index - 1),
                )
                if any(root_distance(candidate, other) < min_distance
                       for other in tuple(forbidden) + tuple(accepted)):
                    leakage_rejected += 1
                    continue
                accepted.append(candidate)
                got += 1
            per_class[interaction_class] = got
            if got < want:
                unfilled[interaction_class] = want - got
    report = SamplingReport(
        split=split, planned=total, sampled=len(accepted), per_class=per_class,
        sample_none=sample_none, leakage_rejected=leakage_rejected, unfilled=unfilled,
    )
    return accepted, report


def build_phase_b_roots(
    split_roots: Optional[Dict[str, int]] = None,
    *,
    min_distance: float = MIN_ROOT_DISTANCE,
) -> Tuple[Dict[str, List[RootSpec]], Dict[str, SamplingReport]]:
    """All splits, in the fixed order train -> val -> test.

    Train is built first, so every val/test root is checked against the accepted train roots;
    the check is symmetric, so the cross-split guarantee does not depend on the order.
    """

    requested = dict(split_roots or SPLIT_ROOTS)
    roots: Dict[str, List[RootSpec]] = {}
    reports: Dict[str, SamplingReport] = {}
    forbidden: List[RootSpec] = []
    for split in ("train", "val", "test"):
        total = int(requested.get(split, 0))
        accepted, report = sample_split_roots(
            split, total, forbidden=forbidden, min_distance=min_distance
        )
        roots[split] = accepted
        reports[split] = report
        forbidden.extend(accepted)
    return roots, reports


def leakage_summary(roots: Dict[str, Sequence[RootSpec]]) -> Dict[str, object]:
    """Minimum cross-split distance, computed with :func:`root_distance`."""

    splits = [name for name, values in roots.items() if values]
    pairs: List[Tuple[str, str, float, float, str, str]] = []
    for i, left in enumerate(splits):
        for right in splits[i + 1:]:
            worst = None
            for a in roots[left]:
                for b in roots[right]:
                    value = root_distance(a, b)
                    position = root_position_distance(a, b)
                    if worst is None or value < worst[0]:
                        worst = (value, position, a.root_id, b.root_id)
            if worst is not None:
                pairs.append((left, right, worst[0], worst[1], worst[2], worst[3]))
    return {
        "metric": "max(object-centre distance, robot-centre distances, "
                  f"{ANG_SCALE} * |wrapped angle differences|)",
        "min_distance_threshold": MIN_ROOT_DISTANCE,
        "pairs": [
            {"split_a": p[0], "split_b": p[1], "min_distance": p[2],
             "min_position_distance": p[3], "root_a": p[4], "root_b": p[5]}
            for p in pairs
        ],
    }


def root_records(roots: Sequence[RootSpec]) -> List[Dict[str, object]]:
    """A self-contained, JSON-serialisable description of one split's roots."""

    return [_root_record(root) for root in roots]


def _root_record(root: RootSpec) -> Dict[str, object]:
    return {
        "root_id": root.root_id,
        "split": root.split,
        "interaction_class": root.interaction_class,
        "template": root.template,
        "seed": root.seed,
        "active_robots": list(root.active_robots),
        "object_pose": [root.object_pose.x, root.object_pose.y, root.object_pose.angle],
        "robot_poses": [[p.x, p.y, p.angle] for p in root.robot_poses],
        # kept so a record can be reconstructed by the unmodified runner helper
        "meta": root_meta(root),
    }


def root_meta(root: RootSpec) -> Dict[str, object]:
    from coimagine.local_contact_option.runner import root_metadata

    payload = root_metadata(root)
    payload["alpha_rad"] = list(root.alpha)
    return payload


# ---------------------------------------------------------------------------
# joint skill commands
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JointCommand:
    index: int
    alpha: Tuple[Optional[float], Optional[float]]
    speed: Tuple[Optional[float], Optional[float]]
    active_robots: Tuple[int, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "command_index": self.index,
            "active_robots": list(self.active_robots),
            "alpha_rad": list(self.alpha),
            "alpha_deg": [None if a is None else round(math.degrees(a), 6) for a in self.alpha],
            "speed": list(self.speed),
            "active_duration": 2.0,
            "force_limit": 15.0,
        }

    def signature(self) -> Tuple:
        return tuple(
            (None if self.alpha[i] is None else round(self.alpha[i], 9),
             None if self.speed[i] is None else round(self.speed[i], 9))
            for i in (0, 1)
        )


def _rotate(values: Sequence[int], offset: int) -> List[int]:
    offset %= len(values)
    return list(values[offset:]) + list(values[:offset])


def commands_for_root(root: RootSpec, n: int = N_COMMANDS) -> List[JointCommand]:
    """``n`` distinct legal joint commands, deterministic in the root seed.

    * one active robot: the 3x3 grid of (alpha, speed) has nine legal points; ``n=8`` of them
      are kept (the dropped one is chosen by the root seed);
    * two active robots: eight distinct joint combinations. Robot 0 walks eight distinct
      (alpha, speed) pairs so the joints are distinct by construction; robot 1 walks a
      seed-rotated copy, so both robots vary their direction and their speed across the set.

    The per-root seed only rotates the mapping / picks the dropped grid point. No object
    ground truth is read, so the command set is a legal, deployable action vocabulary.
    """

    rng = random.Random(root.seed ^ CMD_SALT)
    alpha_perm = _rotate([0, 1, 2], rng.randrange(3))
    speed_perm = _rotate([0, 1, 2], rng.randrange(3))
    active = tuple(int(v) for v in root.active_robots)

    def build(pairs: Sequence[Tuple[int, int]], shift_a: int, shift_s: int) -> List[JointCommand]:
        out: List[JointCommand] = []
        for index, (alpha_index, speed_index) in enumerate(pairs):
            alpha: List[Optional[float]] = [None, None]
            speed: List[Optional[float]] = [None, None]
            for position, robot_id in enumerate(active):
                if position == 0:
                    a_idx = alpha_perm[alpha_index]
                    s_idx = speed_perm[speed_index]
                else:
                    a_idx = alpha_perm[(alpha_index + shift_a) % 3]
                    s_idx = speed_perm[(speed_index + shift_s) % 3]
                alpha[robot_id] = float(ALPHA_CHOICES[a_idx])
                speed[robot_id] = float(SPEED_CHOICES[s_idx])
            out.append(JointCommand(
                index=index, alpha=(alpha[0], alpha[1]), speed=(speed[0], speed[1]),
                active_robots=active,
            ))
        return out

    if len(active) == 1:
        grid = [(a, s) for a in range(3) for s in range(3)]
        drop = rng.randrange(len(grid))
        kept = [pair for position, pair in enumerate(grid) if position != drop]
        commands = build(kept, 0, 0)
    else:
        commands = build(_INDEX_PAIRS, rng.randrange(3), rng.randrange(3))

    if n != N_COMMANDS:
        commands = commands[:n]
    signatures = {command.signature() for command in commands}
    if len(signatures) != len(commands):
        raise RuntimeError(f"{root.root_id}: command generator produced duplicates")
    return commands


def commands_records(roots: Sequence[RootSpec], n: int = N_COMMANDS) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    for root in roots:
        out.append({
            "root_id": root.root_id,
            "split": root.split,
            "interaction_class": root.interaction_class,
            "active_robots": list(root.active_robots),
            "n_commands": len(commands_for_root(root, n)),
            "commands": [command.as_dict() for command in commands_for_root(root, n)],
        })
    return out


def apply_command(root: RootSpec, command: JointCommand) -> RootSpec:
    """The same physical initial state with this command's alpha/speed installed."""

    return replace(root, alpha=command.alpha, speed=command.speed)


def roots_sha256(records: Sequence[Dict[str, object]]) -> str:
    import json

    blob = json.dumps(list(records), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def command_choice_space() -> Dict[str, object]:
    return {
        "alpha_choices_rad": list(ALPHA_CHOICES),
        "alpha_choices_deg": [round(math.degrees(value), 3) for value in ALPHA_CHOICES],
        "speed_choices": list(SPEED_CHOICES),
        "n_commands_per_root": N_COMMANDS,
        "selection_rule": (
            "deterministic in root.seed only; one-active roots keep 8 of the 9 (alpha, speed) "
            "grid points, two-active roots use eight distinct joint pairs built from the same "
            "frozen ranges. No ground truth enters the selection."
        ),
    }
