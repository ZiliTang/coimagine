"""The two low-cost baselines of task book section B2, in original physical units.

P0 -- start-observation persistence. The terminal segment midpoint and normal copy the start
segment expressed in the same R_i0; pose change is zero; the residual linear velocity copies the
start velocity rotated into R_i0; visibility/contact copy the start flags; duration is the train
median and the exit class the train majority. When the *start* patch is invisible, P0 does not
quietly "succeed": the geometry is filled with the per-column train median, the slot is flagged as a
fallback, and scoring is also reported on the common subset where a true start patch exists.

P1 -- train statistics grouped by the full public joint command (including the active mask). It
exists to check whether the network actually uses the start observation or merely remembers what a
command usually produces. Continuous columns use the group's train median; the normal uses the mean
of the unit vectors renormalised, with a *fixed* fallback when the mean degenerates; discrete
targets use the group's train frequency / majority. An unseen command falls back to the global train
statistics and is counted.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Sequence, Tuple

import numpy as np

from experiments.observation_terminal_v1.dataset_terminal import VALIDITY_COLUMNS
from experiments.option_prediction_mechanism_v1.phase_b.dataset_b import EXIT_CLASSES


def _column_median(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-column median over the rows where the column is valid."""

    out = np.zeros(values.shape[1], dtype=np.float32)
    for column in range(values.shape[1]):
        valid = mask[:, column] > 0
        out[column] = float(np.median(values[valid, column])) if valid.any() else 0.0
    return out


def _fill_missing_columns(values: np.ndarray, mask: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            if mask[row, column] == 0:
                filled[row, column] = fallback[column]
    return filled


def _scatter(data, index: np.ndarray, values: Dict[str, np.ndarray],
             template: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Place split-length predictions into full-length arrays.

    The evaluation interface indexes predictions with positions into the whole dataset, so every
    model -- baselines included -- must return arrays aligned with ``data``, not with one split.
    """

    out: Dict[str, np.ndarray] = {}
    for name, block in template.items():
        filled = np.array(block, copy=True)
        filled[index] = values[name]
        out[name] = filled
    return out


def persistence_predictions(data, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """P0. Returns full-length predictions in original units plus its fallback bookkeeping."""

    index = np.where(data.split == split)[0]
    train = data.split == "train"
    mask = data.targets["terminal_observation_geometry"]["mask"]
    truth = data.targets["terminal_observation_geometry"]["y"]
    fallback = _column_median(truth[train], mask[train])
    geometry = _fill_missing_columns(data.persistence["terminal_observation_geometry"][index],
                                     data.persistence["geometry_mask"][index], fallback)
    fallback_rows = int(np.any(data.persistence["geometry_fallback"][index] > 0, axis=1).sum())
    common = np.ones(len(index), dtype=bool)
    for robot_id in (0, 1):
        common &= data.persistence["geometry_mask"][index][:, 4 * robot_id] > 0
    total = len(data.split)
    template = {
        "terminal_observation_geometry": np.zeros((total, 8), dtype=np.float32),
        "robot_pose_delta": np.zeros((total, 6), dtype=np.float32),
        "robot_residual_velocity": np.zeros((total, 6), dtype=np.float32),
        "duration": np.zeros((total, 1), dtype=np.float32),
        "observation_validity": np.zeros((total, 4), dtype=np.float32),
        "exit_class": np.zeros(total, dtype=np.int64),
    }
    values = {
        "terminal_observation_geometry": geometry,
        "robot_pose_delta": data.persistence["robot_pose_delta"][index],
        "robot_residual_velocity": data.persistence["robot_residual_velocity"][index],
        "duration": np.full((len(index), 1), data.train_duration_median, dtype=np.float32),
        "observation_validity": data.persistence["observation_validity"][index],
        "exit_class": np.full(len(index), data.train_exit_majority, dtype=np.int64),
    }
    return (_scatter(data, index, values, template), {"geometry_fallback_rows": fallback_rows,
        "common_subset_rows": int(common.sum()), "rows": int(len(index)),
        "common_subset_index": common})


def _command_group_stats(data, group: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Train statistics per command signature for one continuous group."""

    signatures = getattr(data, "command_signature")
    train = data.split == "train"
    truth = data.targets[group]["y"]
    mask = data.targets[group]["mask"]
    stats: Dict[str, Dict[str, np.ndarray]] = {}
    grouped: Dict[str, List[int]] = defaultdict(list)
    for row in np.where(train)[0]:
        grouped[signatures[row]].append(int(row))
    for signature, rows in grouped.items():
        rows_array = np.asarray(rows)
        stats[signature] = {"median": _column_median(truth[rows_array], mask[rows_array])}
        if group == "terminal_observation_geometry":
            stats[signature]["normal"] = _normal_statistic(truth[rows_array], mask[rows_array])
    return stats


def _normal_statistic(truth: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
    """Mean unit normal per robot, renormalised, with a fixed fallback when it degenerates."""

    out: Dict[str, np.ndarray] = {}
    for robot_id in (0, 1):
        sin_index, cos_index = 4 * robot_id + 2, 4 * robot_id + 3
        valid = (mask[:, sin_index] > 0) & (mask[:, cos_index] > 0)
        per_robot: Dict[str, np.ndarray] = {}
        for column in (sin_index, cos_index):
            per_robot[str(column)] = np.asarray(
                [float(np.mean(truth[valid, column]))] if valid.any() else [0.0],
                dtype=np.float32)
        vector = np.array([per_robot[str(sin_index)][0], per_robot[str(cos_index)][0]],
                          dtype=np.float64)
        norm = float(np.hypot(*vector))
        if norm < 1e-6:
            # fixed fallback: the frozen unit vector convention for a degenerate group
            vector = np.array([0.0, 1.0])
        else:
            vector = vector / norm
        out[str(robot_id)] = vector.astype(np.float32)
    return out


def command_grouped_predictions(data, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, object]]:
    """P1. Train statistics grouped by the public joint command."""

    index = np.where(data.split == split)[0]
    train = data.split == "train"
    signatures = getattr(data, "command_signature")
    global_geometry = _column_median(data.targets["terminal_observation_geometry"]["y"][train],
                                     data.targets["terminal_observation_geometry"]["mask"][train])
    geometry_stats = _command_group_stats(data, "terminal_observation_geometry")
    pose_stats = _command_group_stats(data, "robot_pose_delta")
    velocity_stats = _command_group_stats(data, "robot_residual_velocity")
    duration_stats = _command_group_stats(data, "duration")

    # global normal fallback for robots whose group statistic degenerates
    global_normal = _normal_statistic(data.targets["terminal_observation_geometry"]["y"][train],
                                      data.targets["terminal_observation_geometry"]["mask"][train])

    class_by_group: Dict[str, Counter] = defaultdict(Counter)
    grouped_validity: Dict[str, List[np.ndarray]] = defaultdict(list)
    for row in np.where(train)[0]:
        grouped_validity[signatures[row]].append(
            data.targets["observation_validity"]["y"][row])
        class_by_group[signatures[row]][int(data.targets["exit_class"]["y"][row])] += 1
    validity_stats = {signature: (np.mean(np.stack(rows), axis=0) >= 0.5).astype(np.float32)
                      for signature, rows in grouped_validity.items()}
    global_validity = (data.targets["observation_validity"]["y"][train].mean(axis=0) >= 0.5) \
        .astype(np.float32)
    global_class = int(np.bincount(data.targets["exit_class"]["y"][train]).argmax())

    n = len(index)
    geometry = np.zeros((n, 8), dtype=np.float32)
    pose = np.zeros((n, 6), dtype=np.float32)
    velocity = np.zeros((n, 6), dtype=np.float32)
    duration = np.zeros((n, 1), dtype=np.float32)
    validity = np.zeros((n, 4), dtype=np.float32)
    exit_class = np.zeros(n, dtype=np.int64)
    unseen = 0
    for position, row in enumerate(index):
        signature = signatures[row]
        if signature not in geometry_stats:
            unseen += 1
            geometry[position] = global_geometry
            pose[position] = 0.0
            velocity[position] = 0.0
            duration[position] = float(np.median(data.targets["duration"]["y"][train]))
            validity[position] = global_validity
            exit_class[position] = global_class
            continue
        geometry[position] = geometry_stats[signature]["median"]
        for robot_id in (0, 1):
            geometry[position, 4 * robot_id + 2:4 * robot_id + 4] = \
                geometry_stats[signature]["normal"][str(robot_id)]
        pose[position] = pose_stats[signature]["median"]
        velocity[position] = velocity_stats[signature]["median"]
        duration[position] = duration_stats[signature]["median"]
        validity[position] = validity_stats.get(signature, global_validity)
        counts = class_by_group.get(signature)
        exit_class[position] = counts.most_common(1)[0][0] if counts else global_class
    total = len(data.split)
    template = {
        "terminal_observation_geometry": np.zeros((total, 8), dtype=np.float32),
        "robot_pose_delta": np.zeros((total, 6), dtype=np.float32),
        "robot_residual_velocity": np.zeros((total, 6), dtype=np.float32),
        "duration": np.zeros((total, 1), dtype=np.float32),
        "observation_validity": np.zeros((total, 4), dtype=np.float32),
        "exit_class": np.zeros(total, dtype=np.int64),
    }
    values = {
        "terminal_observation_geometry": geometry,
        "robot_pose_delta": pose,
        "robot_residual_velocity": velocity,
        "duration": duration,
        "observation_validity": validity,
        "exit_class": exit_class,
    }
    return (_scatter(data, index, values, template), {"unseen_command_rows": unseen, "rows": n,
        "command_groups_train": len(geometry_stats),
        "validity_columns": VALIDITY_COLUMNS})
