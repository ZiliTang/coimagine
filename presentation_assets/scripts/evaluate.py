"""Round-5 baselines and evaluation.

Baselines (task book 6.1), both fitted on train only and both expressed in **original units**:

* ``P0`` hold: the fixed-dt head repeats the current visible geometry and own state; the terminal head
  repeats the current state, takes the remaining time from the train median of the same preset
  elapsed-time bin and the exit class from that bin's train majority. A start/current invisible
  segment falls back to train statistics and is reported as a fallback rate.
* ``P1`` grouped statistics: groups are (public joint command signature, elapsed-time bin of 0.25 s).
  Continuous quantities use the group median, the surface normal the mean unit vector, binary and
  class targets the group frequency. Groups with fewer than 20 samples fall back in the order
  command+time -> command -> global, and the lookup table is saved for inspection.

P1 frequencies are probabilities; the MLP/Transformer binary heads are logits. The evaluator is told
which it is and never applies a sigmoid to a probability.

Evaluation is split exactly as the task book demands: the **pre-execution** regime (t = 0, only O0 and
the candidate command) and the **in-execution** regime (a real observed prefix), the latter grouped by
preset elapsed-time bins. Sample counts and independent-root counts are reported everywhere.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_sequence_v1.dataset import (  # noqa: E402
    GROUPS,
    Tokens,
    build_example,
)
from experiments.night_sequence_v1.windows import (  # noqa: E402
    POSE_COLUMNS,
    VELOCITY_COLUMNS,
)
from experiments.night_sequence_v1.sequence_metrics import (  # noqa: E402
    action_sensitivity,
    percentiles,
    spearman,
)

DATA = ROOT / "outputs/night_sequence_v1"
ELAPSED_BIN = 0.25
MIN_GROUP = 20
KIND = {name: k for name, _w, k in GROUPS}
WIDTH = {name: width for name, width, _k in GROUPS}
ELAPSED_BINS = [round(i * ELAPSED_BIN, 3) for i in range(21)]


# ---------------------------------------------------------------------------
# evaluation sets (fixed once, reused by every model)
# ---------------------------------------------------------------------------


def fast_targets(tokens: Tokens, index: np.ndarray, end: np.ndarray) -> Dict[str, np.ndarray]:
    """The two heads' targets for a batch of (sequence, instant) pairs, straight from the tokens.

    Same definitions as ``dataset.build_example`` but vectorised: building each example through the
    window path would make the grouped baseline statistics millions of Python-level calls.
    """

    split_of = tokens.split_of[index]
    local = index - np.asarray([tokens.offsets[s] for s in split_of])
    nxt = np.stack([tokens.token_arrays[s][l, e + 1]
                    for s, l, e in zip(split_of, local, end)]).astype(np.float32)
    terminal = tokens.terminal_arrays[split_of[0]][:0]  # placeholder, replaced below
    terminal = np.stack([tokens.terminal_arrays[s][l] for s, l in zip(split_of, local)]).astype(np.float32)
    remaining = tokens.T_exit[index] - end * 0.05
    steps = tokens.steps[index]
    return {
        "process_geometry": np.concatenate([nxt[:, 7:11], nxt[:, 25:29]], axis=1),
        "process_pose": np.concatenate([nxt[:, 0:3], nxt[:, 18:21]], axis=1),
        "process_velocity": np.concatenate([nxt[:, 3:6], nxt[:, 21:24]], axis=1),
        "process_validity": np.stack([nxt[:, 6], nxt[:, 14], nxt[:, 24], nxt[:, 32]], axis=1),
        "process_exit": (end + 1 == steps - 1).astype(np.float32)[:, None],
        "terminal_geometry": np.concatenate([terminal[:, 7:11], terminal[:, 25:29]], axis=1),
        "terminal_pose": np.concatenate([terminal[:, 0:3], terminal[:, 18:21]], axis=1),
        "terminal_velocity": np.concatenate([terminal[:, 3:6], terminal[:, 21:24]], axis=1),
        "terminal_validity": np.stack([terminal[:, 6], terminal[:, 14], terminal[:, 24],
                                       terminal[:, 32]], axis=1),
        "terminal_remaining": remaining[:, None].astype(np.float32),
        "terminal_exit_class": tokens.exit_class[index].astype(np.int64),
    }


def fast_masks(tokens: Tokens, index: np.ndarray, end: np.ndarray) -> Dict[str, np.ndarray]:
    """Per-column supervision masks for a batch of (sequence, instant) pairs.

    The grouped baseline must respect exactly the same masks as the models: a slot whose patch is not
    visible has a zero-filled geometry target, and letting those zeros into a median silently drags
    the whole statistic toward zero.
    """

    split_of = tokens.split_of[index]
    local = index - np.asarray([tokens.offsets[s] for s in split_of])
    nxt = np.stack([tokens.token_arrays[s][l, e + 1]
                    for s, l, e in zip(split_of, local, end)]).astype(np.float32)
    terminal = np.stack([tokens.terminal_arrays[s][l]
                         for s, l in zip(split_of, local)]).astype(np.float32)
    ones8 = np.ones((len(index), 8), dtype=np.float32)
    geometry_process = np.concatenate(
        [np.repeat(nxt[:, 6:7], 4, axis=1), np.repeat(nxt[:, 24:25], 4, axis=1)], axis=1)
    geometry_terminal = np.concatenate(
        [np.repeat(terminal[:, 6:7], 4, axis=1), np.repeat(terminal[:, 24:25], 4, axis=1)], axis=1)
    return {
        "process_geometry": geometry_process,
        "terminal_geometry": geometry_terminal,
        "process_pose": np.ones((len(index), 6), dtype=np.float32),
        "terminal_pose": np.ones((len(index), 6), dtype=np.float32),
        "process_velocity": np.ones((len(index), 6), dtype=np.float32),
        "terminal_velocity": np.ones((len(index), 6), dtype=np.float32),
        "process_validity": np.ones((len(index), 4), dtype=np.float32),
        "terminal_validity": np.ones((len(index), 4), dtype=np.float32),
        "terminal_remaining": np.ones((len(index), 1), dtype=np.float32),
        "terminal_exit_class": np.ones(len(index), dtype=np.float32),
        "process_exit": np.ones((len(index), 1), dtype=np.float32),
        "_process_geometry_flat": geometry_process,
        "_terminal_geometry_flat": geometry_terminal,
        "_ones8": ones8,
    }


def command_signature(shard_command: Dict[str, float]) -> str:
    return "a0={}|a1={}|v0={}|v1={}|m={}".format(
        shard_command.get("alpha0"), shard_command.get("alpha1"),
        shard_command.get("speed0"), shard_command.get("speed1"),
        shard_command.get("active"))


def build_eval_set(tokens: Tokens, split: str, *, in_execution: int = 8192,
                   seed: int = 0) -> Dict[str, np.ndarray]:
    rows = np.where(tokens.split_of == split)[0]
    zero_rows = rows[np.array([True] * len(rows))]
    indices = np.concatenate([zero_rows, np.zeros(len(zero_rows), dtype=np.int64)])
    ends = np.concatenate([np.zeros(len(zero_rows), dtype=np.int64),
                           np.zeros(len(zero_rows), dtype=np.int64)])
    # stratified in-execution sample over elapsed-time bins
    generator = np.random.default_rng(seed)
    buckets: Dict[int, List[Tuple[int, int]]] = {}
    for index in rows:
        steps = int(tokens.steps[index])
        for end in range(1, max(2, steps - 1)):
            elapsed = end * 0.05
            bucket = int(elapsed / ELAPSED_BIN)
            buckets.setdefault(bucket, []).append((int(index), end))
    chosen: List[Tuple[int, int]] = []
    per_bucket = max(1, in_execution // max(1, len(buckets)))
    for bucket in sorted(buckets):
        pool = buckets[bucket]
        take = min(per_bucket, len(pool))
        picks = generator.choice(len(pool), size=take, replace=False)
        chosen.extend(pool[p] for p in picks)
    in_exec = np.asarray(chosen, dtype=np.int64)
    return {"zero_index": zero_rows, "zero_end": np.zeros(len(zero_rows), dtype=np.int64),
            "exec_index": in_exec[:, 0], "exec_end": in_exec[:, 1],
            "split": np.array([split])}


# ---------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------


def target_vector(tokens: Tokens, index: int, end: int) -> Dict[str, np.ndarray]:
    example = build_example(tokens, index, end)
    return example["targets"], example["masks"], example


def _current_tokens(tokens: Tokens, index: np.ndarray, end: np.ndarray) -> np.ndarray:
    split_of = tokens.split_of[index]
    local = index - np.asarray([tokens.offsets[s] for s in split_of])
    return np.stack([tokens.token_arrays[s][l, e]
                     for s, l, e in zip(split_of, local, end)]).astype(np.float32)


def hold_predictions_batch(tokens: Tokens, index: np.ndarray, end: np.ndarray,
                           remaining_table: Dict[int, float],
                           exit_table: Dict[int, int]) -> Dict[str, np.ndarray]:
    """P0 for a whole batch: hold the current visible geometry and own state."""

    current = _current_tokens(tokens, index, end)
    geometry = np.concatenate([current[:, 7:11], current[:, 25:29]], axis=1)
    pose = np.concatenate([current[:, 0:3], current[:, 18:21]], axis=1)
    velocity = np.concatenate([current[:, 3:6], current[:, 21:24]], axis=1)
    validity = np.stack([current[:, 6], current[:, 14], current[:, 24], current[:, 32]], axis=1)
    bins = (end * 0.05 / ELAPSED_BIN).astype(np.int64)
    remaining = np.asarray([remaining_table.get(int(b), 0.0) for b in bins], dtype=np.float32)
    exit_class = np.asarray([exit_table.get(int(b), 0) for b in bins], dtype=np.int64)
    return {
        "process_geometry": geometry, "process_pose": pose, "process_velocity": velocity,
        "process_validity": validity, "process_exit": np.zeros((len(index), 1), dtype=np.float32),
        "terminal_geometry": geometry, "terminal_pose": pose, "terminal_velocity": velocity,
        "terminal_validity": validity, "terminal_remaining": remaining[:, None],
        "terminal_exit_class": exit_class,
    }


def grouped_predictions_batch(statistics, tokens: Tokens, index: np.ndarray,
                              end: np.ndarray) -> Dict[str, np.ndarray]:
    """P1 for a whole batch, using the saved lookup tables."""

    signature = GroupedStatistics._signatures(tokens, index)
    bins = (end * 0.05 / ELAPSED_BIN).astype(np.int64)
    rows: Dict[str, List[np.ndarray]] = {name: [] for name, _w, _k in GROUPS}
    for position in range(len(index)):
        key_time = f"{signature[position]}@{int(bins[position])}"
        table = None
        for level, key in (("command_time", key_time), ("command", signature[position]),
                           ("global", "all")):
            candidate = statistics.tables[level].get(key)
            if candidate is not None and (level == "global"
                                          or statistics.counts[level].get(key, 0) >= MIN_GROUP):
                table = candidate
                break
        for name, _w, kind in GROUPS:
            value = table[name]
            rows[name].append(np.atleast_1d(value).astype(np.int64 if kind == "multiclass"
                                                          else np.float32))
    return {name: np.stack(values) for name, values in rows.items()}


def _hold_predictions(tokens: Tokens, index: int, end: int) -> Dict[str, np.ndarray]:
    example = build_example(tokens, index, end)
    split = tokens.split_of[index]
    local = index - tokens.offsets[split]
    block = tokens.token_arrays[split][local].astype(np.float32)
    terminal = tokens.terminal_arrays[split][local].astype(np.float32)
    current = block[end]
    out = {
        "process_geometry": np.concatenate([current[7:11], current[25:29]]),
        "process_pose": np.concatenate([current[0:3], current[18:21]]),
        "process_velocity": np.concatenate([current[3:6], current[21:24]]),
        "process_validity": np.array([current[6], current[14], current[24], current[32]]),
        "process_exit": np.zeros(1, dtype=np.float32),
        "terminal_geometry": np.concatenate([current[7:11], current[25:29]]),
        "terminal_pose": np.concatenate([current[0:3], current[18:21]]),
        "terminal_velocity": np.concatenate([current[3:6], current[21:24]]),
        "terminal_validity": np.array([current[6], current[14], current[24], current[32]]),
        "terminal_remaining": np.zeros(1, dtype=np.float32),
        "terminal_exit_class": np.zeros(1, dtype=np.int64),
    }
    return out


class GroupedStatistics:
    """P1: train statistics per (command signature, elapsed bin) with hierarchical fallback.

    Fitted on a bounded, uniformly sampled set of (sequence, instant) train pairs -- never on the
    test split, never grouped by the true exit instant, and always in **original units**.
    """

    MIN_GROUP = MIN_GROUP

    def __init__(self, tokens: Tokens, train_splits: Sequence[str], sample: int = 240000,
                 seed: int = 1):
        rows = np.where(np.isin(tokens.split_of, list(train_splits)))[0]
        generator = np.random.default_rng(seed)
        # sample instants directly: uniform over sequences then uniform over that sequence's instants
        count = min(sample, len(rows) * 8)
        chosen_rows = rows[generator.integers(0, len(rows), size=count)]
        steps = tokens.steps[chosen_rows]
        ends = (generator.random(count) * np.maximum(1, steps - 1)).astype(np.int64)
        ends = np.minimum(ends, np.maximum(0, steps - 2))
        targets = fast_targets(tokens, chosen_rows, ends)
        masks = fast_masks(tokens, chosen_rows, ends)
        self.masks = masks
        signature = self._signatures(tokens, chosen_rows)
        bins = (ends * 0.05 / ELAPSED_BIN).astype(np.int64)
        levels = {
            "command_time": np.asarray([f"{s}@{b}" for s, b in zip(signature, bins)]),
            "command": signature,
            "global": np.asarray(["all"] * count),
        }
        self.tables: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
        self.counts: Dict[str, Dict[str, int]] = {}
        for level, keys in levels.items():
            table: Dict[str, Dict[str, np.ndarray]] = {}
            counts: Dict[str, int] = {}
            order = np.argsort(keys, kind="mergesort")
            sorted_keys = keys[order]
            boundaries = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1], True])
            for start_index, stop_index in zip(boundaries[:-1], boundaries[1:]):
                key = str(sorted_keys[start_index])
                selection = order[start_index:stop_index]
                counts[key] = int(len(selection))
                table[key] = self._summarise(
                    {name: value[selection] for name, value in targets.items()},
                    {name: value[selection] for name, value in masks.items()})
            self.tables[level] = table
            self.counts[level] = counts

    @staticmethod
    def _signatures(tokens: Tokens, index: np.ndarray) -> np.ndarray:
        split_of = tokens.split_of[index]
        local = index - np.asarray([tokens.offsets[s] for s in split_of])
        command = np.stack([tokens.token_arrays[s][l, 0, 51:57]
                            for s, l in zip(split_of, local)]).astype(np.float32)
        return np.asarray([f"{round(float(c[0]), 6)}|{round(float(c[1]), 6)}|"
                           f"{round(float(c[2]), 6)}|{round(float(c[3]), 6)}|{int(c[4] > 0.5)}"
                           for c in command])

    def _summarise(self, targets: Dict[str, np.ndarray],
                   masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Group statistics **under the same supervision masks as the models** (round-6 fix).

        Continuous columns take the median of the rows where that column is supervised; a completely
        unsupervised column falls back to the global median rather than to zero.
        """

        out: Dict[str, np.ndarray] = {}
        for name, width, kind in GROUPS:
            stacked = np.asarray(targets[name], dtype=np.float64)
            column_mask = np.asarray(masks.get(name, np.ones_like(targets[name], dtype=np.float64)),
                                     dtype=np.float64)
            if kind == "continuous":
                if column_mask.ndim == 1:
                    column_mask = column_mask[:, None]
                summary = np.zeros(stacked.shape[1], dtype=np.float64)
                for column in range(stacked.shape[1]):
                    keep = column_mask[:, column] > 0
                    summary[column] = (float(np.median(stacked[keep, column])) if keep.any()
                                       else float(np.median(stacked[:, column])))
            elif kind == "binary":
                keep = column_mask > 0
                summary = np.asarray(
                    [float(stacked[keep[:, c], c].mean()) if keep[:, c].any() else 0.0
                     for c in range(stacked.shape[1])], dtype=np.float64)
                summary = (summary >= 0.5).astype(np.float32)
            else:
                values, counts = np.unique(stacked.astype(np.int64), return_counts=True)
                summary = np.asarray([values[np.argmax(counts)]])
            out[name] = np.asarray(summary, dtype=np.float64)
        # the surface normal is averaged as a unit vector over the supervised entries only
        for name in ("process_geometry", "terminal_geometry"):
            geometry_mask = np.asarray(masks[name], dtype=np.float64)
            for sin_index, cos_index in ((2, 3), (6, 7)):
                both = (geometry_mask[:, sin_index] > 0) & (geometry_mask[:, cos_index] > 0)
                if both.any():
                    vector = np.array([float(np.mean(targets[name][both, sin_index])),
                                       float(np.mean(targets[name][both, cos_index]))])
                else:
                    vector = np.array([out[name][sin_index], out[name][cos_index]])
                norm = float(np.hypot(*vector))
                out[name][sin_index] = vector[0] / norm if norm > 1e-6 else 0.0
                out[name][cos_index] = vector[1] / norm if norm > 1e-6 else 1.0
        return out

    def predict(self, tokens: Tokens, index: int, end: int) -> Dict[str, np.ndarray]:
        signature = str(self._signatures(tokens, np.asarray([index]))[0])
        key_time = f"{signature}@{int(end * 0.05 / ELAPSED_BIN)}"
        for level, key in (("command_time", key_time), ("command", signature), ("global", "all")):
            table = self.tables[level].get(key)
            if table is not None and (level == "global"
                                      or self.counts[level].get(key, 0) >= self.MIN_GROUP):
                return {name: np.array(value, copy=True) for name, value in table.items()}
        return {name: np.array(value, copy=True)
                for name, value in self.tables["global"]["all"].items()}


def evaluate_predictions(targets: Dict[str, np.ndarray], masks: Dict[str, np.ndarray],
                         predictions: Dict[str, np.ndarray], *, binary_kind: str,
                         groups: Dict[str, np.ndarray],
                         roots: Optional[Sequence[str]] = None) -> Dict[str, object]:
    """Continuous errors in original units, binary/class recall, grouped by the given keys."""

    if roots is None:
        roots = [f"row{i}" for i in range(len(predictions["terminal_geometry"]))]
    roots = np.asarray(roots)
    report: Dict[str, object] = {}
    geometry = {}
    for robot_id in (0, 1):
        base = 4 * robot_id
        value = masks["terminal_geometry"][:, base] > 0
        dx = np.abs(predictions["terminal_geometry"][:, base]
                    - targets["terminal_geometry"][:, base])
        dy = np.abs(predictions["terminal_geometry"][:, base + 1]
                    - targets["terminal_geometry"][:, base + 1])
        geometry[f"r{robot_id}_mid_2d"] = percentiles(np.hypot(dx, dy)[value].tolist())
        angle_pred = np.arctan2(predictions["terminal_geometry"][:, base + 3],
                                predictions["terminal_geometry"][:, base + 2])
        angle_true = np.arctan2(targets["terminal_geometry"][:, base + 3],
                                targets["terminal_geometry"][:, base + 2])
        difference = np.abs((angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi)
        geometry[f"r{robot_id}_normal_angle_deg"] = percentiles(
            np.degrees(difference)[value].tolist())
    report["terminal_geometry"] = geometry
    # the fixed-dt process head gets the same geometry treatment as the terminal head
    process_geometry = {}
    for robot_id in (0, 1):
        base = 4 * robot_id
        value = masks["process_geometry"][:, base] > 0
        dx = np.abs(predictions["process_geometry"][:, base]
                    - targets["process_geometry"][:, base])
        dy = np.abs(predictions["process_geometry"][:, base + 1]
                    - targets["process_geometry"][:, base + 1])
        process_geometry[f"r{robot_id}_mid_2d"] = percentiles(np.hypot(dx, dy)[value].tolist())
        angle_pred = np.arctan2(predictions["process_geometry"][:, base + 3],
                                predictions["process_geometry"][:, base + 2])
        angle_true = np.arctan2(targets["process_geometry"][:, base + 3],
                                targets["process_geometry"][:, base + 2])
        difference = np.abs((angle_pred - angle_true + np.pi) % (2 * np.pi) - np.pi)
        process_geometry[f"r{robot_id}_normal_angle_deg"] = percentiles(
            np.degrees(difference)[value].tolist())
    report["process_geometry"] = process_geometry
    # the fixed-dt horizon is 0.05 s, where a near-static robot barely moves; without a motion
    # breakdown the headline number is dominated by samples that are almost trivial to hold
    if "motion" in groups:
        motion = groups["motion"]
        per_motion = {}
        for value in sorted(set(motion.tolist())):
            selected = (motion == value) & (masks["process_geometry"][:, 0] > 0)
            entry = percentiles(np.hypot(
                predictions["process_geometry"][:, 0] - targets["process_geometry"][:, 0],
                predictions["process_geometry"][:, 1] - targets["process_geometry"][:, 1]
            )[selected].tolist())
            entry["samples"] = int(selected.sum())
            entry["independent_roots"] = int(len(set(roots[selected].tolist())))
            per_motion[str(value)] = entry
        report["process_geometry_by_motion"] = per_motion
    # continuous groups are reported **per column**: flattening metres together with radians would
    # mix units in one distribution and make the percentile meaningless
    for name, columns in (("terminal_pose", POSE_COLUMNS),
                          ("terminal_velocity", VELOCITY_COLUMNS),
                          ("process_pose", POSE_COLUMNS),
                          ("process_velocity", VELOCITY_COLUMNS)):
        value = masks[name] > 0
        difference = np.abs(predictions[name] - targets[name])
        report[name] = {column: percentiles(difference[value[:, index], index].tolist())
                        for index, column in enumerate(columns)}
        report[name]["all_columns_flattened"] = percentiles(
            difference[value].ravel().tolist())
    value = masks["terminal_remaining"] > 0
    report["terminal_remaining"] = percentiles(
        np.abs(predictions["terminal_remaining"][value]
               - targets["terminal_remaining"][value]).ravel().tolist())
    for name in ("terminal_validity", "process_validity"):
        per_column = {}
        probability = predictions[name] if binary_kind == "probability" else \
            1.0 / (1.0 + np.exp(-predictions[name]))
        for column, label in enumerate(["r0_patch_valid", "r0_contacting",
                                        "r1_patch_valid", "r1_contacting"]):
            value = masks[name][:, column] > 0
            predicted = probability[value, column] >= 0.5
            actual = targets[name][value, column] >= 0.5
            positives = int(actual.sum())
            true_positive = int((predicted & actual).sum())
            false_positive = int((predicted & ~actual).sum())
            false_negative = int((~predicted & actual).sum())
            true_negative = int((~predicted & ~actual).sum())
            negatives = int((~actual).sum())
            per_column[label] = {
                "n": int(value.sum()), "positives": positives, "negatives": negatives,
                "predicted_positives": int(predicted.sum()),
                "recall": float(true_positive / positives) if positives else None,
                "specificity": float(true_negative / negatives) if negatives else None,
                "precision": float(true_positive / max(1, true_positive + false_positive)),
                "confusion": {"true_negative": true_negative, "false_positive": false_positive,
                              "false_negative": false_negative, "true_positive": true_positive},
                "evidence_insufficient": positives < 10,
                "accuracy": float((predicted == actual).mean()) if value.any() else None}
        report[name] = per_column
    for name in ("terminal_exit_class", "process_exit"):
        if name == "process_exit":
            probability = predictions[name][:, 0] if binary_kind == "probability" else \
                1.0 / (1.0 + np.exp(-predictions[name][:, 0]))
            actual = targets[name][:, 0] >= 0.5
            predicted = probability >= 0.5
            report[name] = {"n": int(len(actual)), "positives": int(actual.sum()),
                            "predicted_positives": int(predicted.sum()),
                            "recall": float((predicted & actual).sum() / actual.sum())
                            if actual.sum() else None}
            continue
        predicted = (predictions[name].astype(int).reshape(-1) if predictions[name].ndim == 1
                     else np.argmax(predictions[name], axis=1).reshape(-1))
        actual = targets[name].astype(int).reshape(-1)
        classes = sorted(set(actual.tolist()) | set(predicted.tolist()))
        report[name] = {"support": {int(c): int((actual == c).sum()) for c in classes},
                        "recall": {int(c): (float(((predicted == c) & (actual == c)).sum()
                                                  / max(1, (actual == c).sum())))
                                   for c in classes},
                        "accuracy": float((predicted == actual).mean())}
    # grouped continuous error by elapsed bin / stage / configuration
    dx = np.abs(predictions["terminal_geometry"][:, 0] - targets["terminal_geometry"][:, 0])
    dy = np.abs(predictions["terminal_geometry"][:, 1] - targets["terminal_geometry"][:, 1])
    midpoint_error = np.hypot(dx, dy)
    visible = masks["terminal_geometry"][:, 0] > 0
    grouped = {}
    for key, values in groups.items():
        per_group = {}
        for group_value in sorted(set(values.tolist())):
            selected = (values == group_value) & visible
            entry = percentiles(midpoint_error[selected].tolist())
            entry["samples"] = int(selected.sum())
            entry["independent_roots"] = int(len(set(np.asarray(roots)[selected].tolist())))
            per_group[str(group_value)] = entry
        grouped[key] = per_group
    report["grouped"] = grouped
    return report


def save_predictions(path: Path, tokens: Tokens, index: np.ndarray, end: np.ndarray,
                     targets, masks, predictions, extra: Dict[str, np.ndarray]) -> None:
    payload = {"index": index, "end": end, "root_id": tokens.roots[index],
               "record_id": tokens.records[index]}
    for name in targets:
        payload[f"target_{name}"] = targets[name]
        payload[f"mask_{name}"] = masks[name]
        payload[f"pred_{name}"] = predictions[name]
    payload.update(extra)
    np.savez_compressed(path, **payload)
