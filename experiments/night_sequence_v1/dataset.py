"""Round-5 dataset: history window -> the two prediction heads, with all three mask families.

Masks (task book 3.2), kept strictly apart:

* ``token_valid``      -- one per window slot: the token is a real sampled instant, not padding.
* ``patch_valid``      -- the instant exists but the robot may see no fitted segment at all. Such a
                          slot is **never** turned into padding and never deleted; it only masks the
                          continuous geometry and still supervises visibility.
* ``target_available`` -- the requested future instant exists. With full per-period storage the next
  grid instant always exists, so head A never lacks a target; head B exists for every non-terminated
  window end. The flag is carried anyway so a truncated sequence cannot silently fabricate one.

Sampling follows task book 4.3: pick a train root uniformly, then one of its eight commands, then the
prediction instant -- 25% at t=0, otherwise uniformly among the remaining non-terminated instants.
Every model is given the same sampled example sequence for a given seed.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.sequences import EXIT_CODES  # noqa: E402
from experiments.night_sequence_v1.windows import (  # noqa: E402
    FEATURES,
    HISTORY,
    WINDOW,
    GEOMETRY_COLUMNS,
    POSE_COLUMNS,
    VALIDITY_COLUMNS,
    VELOCITY_COLUMNS,
)

DATA = ROOT / "outputs/night_sequence_v1"
TOKENS = DATA / "tokens"

#: each group: (name, width, kind). Continuous groups are standardised; binary use BCE on logits;
#: the exit class uses cross entropy.
GROUPS: Tuple[Tuple[str, int, str], ...] = (
    ("process_geometry", 8, "continuous"),
    ("process_pose", 6, "continuous"),
    ("process_velocity", 6, "continuous"),
    ("process_validity", 4, "binary"),
    ("process_exit", 1, "binary"),
    ("terminal_geometry", 8, "continuous"),
    ("terminal_pose", 6, "continuous"),
    ("terminal_velocity", 6, "continuous"),
    ("terminal_validity", 4, "binary"),
    ("terminal_remaining", 1, "continuous"),
    ("terminal_exit_class", len(EXIT_CODES), "multiclass"),
)
CONTINUOUS_GROUPS = [name for name, _w, kind in GROUPS if kind == "continuous"]


class Tokens:
    """Lazily loaded token arrays for one tier."""

    def __init__(self, splits: Sequence[str], directory: Path = TOKENS):
        self.arrays: Dict[str, Dict[str, np.ndarray]] = {}
        self.order: List[str] = []
        self.offsets: Dict[str, int] = {}
        total = 0
        for split in splits:
            payload = dict(np.load(directory / f"tokens_{split}.npz", allow_pickle=True))
            self.arrays[split] = payload
            self.offsets[split] = total
            size = len(payload["steps"])
            self.order.extend([split] * size)
            total += size
        self.size = total
        self.split_of = np.asarray(self.order)
        self.steps = np.concatenate([self.arrays[s]["steps"] for s in splits])
        self.roots = np.concatenate([self.arrays[s]["root_id"] for s in splits])
        self.records = np.concatenate([self.arrays[s]["record_id"] for s in splits])
        self.exit_class = np.concatenate([self.arrays[s]["exit_class"] for s in splits])
        self.T_exit = np.concatenate([self.arrays[s]["T_exit"] for s in splits])
        self.command_index = np.concatenate([self.arrays[s]["command_index"] for s in splits])
        self.token_arrays = {s: self.arrays[s]["tokens"] for s in splits}
        self.valid_arrays = {s: self.arrays[s]["token_valid"] for s in splits}
        self.terminal_arrays = {s: self.arrays[s]["terminal"] for s in splits}
        self.locate = {s: (np.where(self.split_of == s)[0] - self.offsets[s]) for s in splits}


def window_indices(step_count: int, end: int) -> Tuple[List[int], List[int]]:
    """The window for a prediction instant ``end``: O0 anchor + the last 32 real instants."""

    if end <= 0:
        return [0], []
    low = max(1, end - (HISTORY - 1))
    return [0] + list(range(low, end + 1)), []


def build_example(tokens: Tokens, index: int, end: int) -> Dict[str, np.ndarray]:
    """One example: padded window, masks, and the two heads' targets, all from the token array."""

    split = tokens.split_of[index]
    local = index - tokens.offsets[split]
    steps = int(tokens.steps[index])
    end = int(min(max(end, 0), steps - 2)) if steps >= 2 else 0
    block = tokens.token_arrays[split][local]
    terminal = tokens.terminal_arrays[split][local].astype(np.float32)
    indices, _dropped = window_indices(steps, end)
    window = np.zeros((WINDOW, FEATURES), dtype=np.float32)
    valid = np.zeros(WINDOW, dtype=np.float32)
    # left padding, so the anchor always sits at slot 0 and the most recent instant at the end
    offset = WINDOW - len(indices)
    for position, source in enumerate(indices):
        window[offset + position] = block[source]
        valid[offset + position] = 1.0
    next_token = block[end + 1].astype(np.float32)
    processed = {
        "process_geometry": np.concatenate([next_token[7:11], next_token[25:29]]).astype(np.float32),
        "process_pose": np.concatenate([next_token[0:3], next_token[18:21]]).astype(np.float32),
        "process_velocity": np.concatenate([next_token[3:6], next_token[21:24]]).astype(np.float32),
        "process_validity": np.array([next_token[6], next_token[14],
                                      next_token[24], next_token[32]], dtype=np.float32),
        "process_exit": np.array([1.0 if end + 1 == steps - 1 else 0.0], dtype=np.float32),
    }
    remaining = float(tokens.T_exit[index]) - float(end) * 0.05
    terminal_targets = {
        "terminal_geometry": np.concatenate([terminal[7:11], terminal[25:29]]).astype(np.float32),
        "terminal_pose": np.concatenate([terminal[0:3], terminal[18:21]]).astype(np.float32),
        "terminal_velocity": np.concatenate([terminal[3:6], terminal[21:24]]).astype(np.float32),
        "terminal_validity": np.array([terminal[6], terminal[14], terminal[24], terminal[32]],
                                      dtype=np.float32),
        "terminal_remaining": np.array([remaining], dtype=np.float32),
        "terminal_exit_class": np.array([int(tokens.exit_class[index])], dtype=np.int64),
    }
    # geometry masks come from the TRUE visibility at the target instant, never from a prediction
    process_geometry_mask = np.concatenate([
        np.repeat(next_token[6], 4), np.repeat(next_token[24], 4)]).astype(np.float32)
    terminal_geometry_mask = np.concatenate([
        np.repeat(terminal[6], 4), np.repeat(terminal[24], 4)]).astype(np.float32)
    masks = {
        "process_geometry": process_geometry_mask,
        "process_pose": np.ones(6, dtype=np.float32),
        "process_velocity": np.ones(6, dtype=np.float32),
        "process_validity": np.ones(4, dtype=np.float32),
        "process_exit": np.ones(1, dtype=np.float32),
        "terminal_geometry": terminal_geometry_mask,
        "terminal_pose": np.ones(6, dtype=np.float32),
        "terminal_velocity": np.ones(6, dtype=np.float32),
        "terminal_validity": np.ones(4, dtype=np.float32),
        "terminal_remaining": np.ones(1, dtype=np.float32),
        "terminal_exit_class": np.ones(1, dtype=np.float32),
    }
    return {"window": window, "token_valid": valid, "targets": {**processed, **terminal_targets},
            "masks": masks, "end": np.array([end]), "steps": np.array([steps]),
            "root_id": np.array([str(tokens.roots[index])], dtype=object),
            "record_id": np.array([str(tokens.records[index])], dtype=object)}


# ---------------------------------------------------------------------------
# normalisation: one scalar mean/std per group, fitted on train only
# ---------------------------------------------------------------------------


def fit_normalisation(tokens: Tokens, split: str = "d0", sample: int = 6000,
                      seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Per-group scalar mean/std over masked train values (task book 4.2)."""

    generator = np.random.default_rng(seed)
    rows = np.where(tokens.split_of == split)[0]
    rows = rows[generator.permutation(len(rows))[:sample]]
    collected: Dict[str, List[np.ndarray]] = {name: [] for name in CONTINUOUS_GROUPS}
    for index in rows:
        steps = int(tokens.steps[index])
        import random
        end = 0 if random.random() < 0.25 else random.randrange(0, max(1, steps - 1))
        example = build_example(tokens, int(index), end)
        for name in CONTINUOUS_GROUPS:
            mask = example["masks"][name]
            if mask.sum() > 0:
                collected[name].append(example["targets"][name][mask > 0])
    out: Dict[str, Dict[str, float]] = {}
    for name, blocks in collected.items():
        if not blocks:
            out[name] = {"mean": 0.0, "std": 1.0}
            continue
        values = np.concatenate(blocks).astype(np.float64)
        std = float(values.std())
        out[name] = {"mean": float(values.mean()), "std": std if std > 1e-9 else 1.0}
    return out


def to_torch(indices: Sequence[int], tokens: Tokens, ends: Sequence[int],
             normalisation: Dict[str, Dict[str, float]]):
    examples = [build_example(tokens, int(i), int(e)) for i, e in zip(indices, ends)]
    window = torch.tensor(np.stack([e["window"] for e in examples]), dtype=torch.float32)
    valid = torch.tensor(np.stack([e["token_valid"] for e in examples]), dtype=torch.float32)
    targets, masks = {}, {}
    for name, width, kind in GROUPS:
        raw = np.stack([e["targets"][name] for e in examples])
        mask = np.stack([e["masks"][name] for e in examples])
        if kind == "continuous":
            stats = normalisation[name]
            raw = (raw - stats["mean"]) / stats["std"]
        targets[name] = torch.tensor(raw, dtype=torch.float32 if kind != "multiclass"
                                     else torch.long)
        masks[name] = torch.tensor(mask, dtype=torch.float32)
    return window, valid, targets, masks


class RootSampler:
    """Uniform root -> uniform command -> prediction instant, with the root grouping precomputed.

    Rebuilding the per-root index lists on every batch is O(roots) per update and dominated the D2
    runs (16,384 roots); the grouping depends only on the dataset, so it is built once.
    """

    def __init__(self, lengths: np.ndarray, roots: np.ndarray):
        unique, inverse = np.unique(roots, return_inverse=True)
        order = np.argsort(inverse, kind="mergesort")
        sorted_inverse = inverse[order]
        boundaries = np.flatnonzero(np.r_[True, sorted_inverse[1:] != sorted_inverse[:-1], True])
        self.groups = [order[a:b] for a, b in zip(boundaries[:-1], boundaries[1:])]
        self.lengths = lengths

    def __len__(self) -> int:
        return len(self.groups)

    def sample(self, count: int, generator: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        indices = np.empty(count, dtype=np.int64)
        ends = np.empty(count, dtype=np.int64)
        for position in range(count):
            group = self.groups[generator.integers(len(self.groups))]
            choice = int(group[generator.integers(len(group))])
            steps = int(self.lengths[choice])
            if steps < 2 or generator.random() < 0.25:
                end = 0
            else:
                end = int(generator.integers(1, steps - 1))
            indices[position] = choice
            ends[position] = end
        return indices, ends


def sampler(lengths: np.ndarray, roots: np.ndarray, count: int,
            generator: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """One-off helper kept for tests and small callers; training uses :class:`RootSampler`."""

    return RootSampler(lengths, roots).sample(count, generator)
