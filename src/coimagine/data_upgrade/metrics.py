"""Shared evaluation metrics for phases B and D.

Everything here uses the fixed A4 tolerances (``CHAIN_POSITION_TOLERANCE`` /
``CHAIN_ANGLE_TOLERANCE``) and keeps surviving-chain and all-chain denominators separate.
Teacher-forced and recursive chains never share an accumulator.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import (
    CHAIN_ANGLE_TOLERANCE,
    CHAIN_POSITION_TOLERANCE,
    TERMINATION_CLASSES,
    geometric_violation,
    wrap_angle,
)
from ..continuous2d_learning.features import apply_state_delta
from ..continuous2d_learning.model import (
    JointWindowMLP,
    boundary_after,
    boundary_before,
    joint_spec_of,
    predict_window,
)
from ..continuous2d_learning.evaluation import predict_records
from ..continuous2d.config import default_config

GROUP_KEYS = (
    "all", "object_translation", "object_rotation", "stationary",
    "contact", "no_contact", "S0", "S1", "success", "failure",
)


def group_flags(record: Dict[str, object]) -> Dict[str, bool]:
    before, after = record["state"], record["next_state"]
    dx = float(after["object"][0]) - float(before["object"][0])
    dy = float(after["object"][1]) - float(before["object"][1])
    translation = math.hypot(dx, dy)
    rotation = abs(wrap_angle(float(after["object"][2]) - float(before["object"][2])))
    translated = translation > 0.01
    rotated = rotation > 0.02
    contact = bool(record["contact"]["had_robot_object_contact"])
    family = str(record["scene_id"]).split("_")[0]
    return {
        "all": True,
        "object_translation": translated,
        "object_rotation": rotated,
        "stationary": not translated and not rotated,
        "contact": contact,
        "no_contact": not contact,
        "S0": family == "S0",
        "S1": family == "S1",
        "success": int(record["success"]) == 1,
        "failure": str(record["termination_class"]) == "physical_failure",
    }


def _body_errors(predicted, actual):
    positions = [
        math.hypot(first.x - second.x, first.y - second.y)
        for first, second in zip(predicted.bodies(), actual.bodies())
    ]
    velocities = [
        math.hypot(first.vx - second.vx, first.vy - second.vy)
        for first, second in zip(predicted.bodies(), actual.bodies())
    ]
    angle = abs(wrap_angle(predicted.object.angle - actual.object.angle))
    return positions, velocities, angle


def evaluate_grouped(model: JointWindowMLP, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Teacher-forced single-step metrics, per group, with sample counts."""

    physics = default_config().physics
    outputs = predict_records(model, records)
    accum: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    termination_true: List[int] = []
    termination_pred: List[int] = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    invalid = 0
    for index, record in enumerate(records):
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        positions, velocities, angle = _body_errors(predicted, actual)
        result = geometric_violation(predicted.object, predicted.robots, scene, physics)
        invalid += int(bool(result["violation"] or not result["finite"]))
        true_index = TERMINATION_CLASSES.index(str(record["termination_class"]))
        pred_index = int(outputs["termination"][index])
        confusion[true_index, pred_index] += 1
        termination_true.append(true_index)
        termination_pred.append(pred_index)
        for group, flag in group_flags(record).items():
            if not flag:
                continue
            row = accum[group]
            row["object_position"].append(positions[0])
            row["robot0_position"].append(positions[1])
            row["robot1_position"].append(positions[2])
            row["object_velocity"].append(velocities[0])
            row["robot0_velocity"].append(velocities[1])
            row["robot1_velocity"].append(velocities[2])
            row["object_angle"].append(angle)
            row["invalid"].append(float(result["violation"] or not result["finite"]))
            row["wrong_termination"].append(float(true_index != pred_index))
    report: Dict[str, object] = {}
    for group in GROUP_KEYS:
        row = accum.get(group)
        if not row:
            report[group] = {"n": 0}
            continue
        report[group] = {
            "n": len(row["object_position"]),
            "object_position_error": float(np.mean(row["object_position"])),
            "robot0_position_error": float(np.mean(row["robot0_position"])),
            "robot1_position_error": float(np.mean(row["robot1_position"])),
            "object_velocity_error": float(np.mean(row["object_velocity"])),
            "robot0_velocity_error": float(np.mean(row["robot0_velocity"])),
            "robot1_velocity_error": float(np.mean(row["robot1_velocity"])),
            "object_angle_error": float(np.mean(row["object_angle"])),
            "invalid_rate": float(np.mean(row["invalid"])),
            "wrong_termination_rate": float(np.mean(row["wrong_termination"])),
        }
    return {
        "n": len(records),
        "groups": report,
        "termination_confusion": confusion.tolist(),
        "termination_classes": list(TERMINATION_CLASSES),
        "predicted_success": int(sum(1 for value in termination_pred if value == TERMINATION_CLASSES.index("success"))),
        "false_success": int(
            sum(
                1
                for true_value, pred_value in zip(termination_true, termination_pred)
                if pred_value == TERMINATION_CLASSES.index("success")
                and true_value != TERMINATION_CLASSES.index("success")
            )
        ),
        "invalid_rate": invalid / max(1, len(records)),
    }


def paired_persistence(model: JointWindowMLP, records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Per-sample paired difference: model error minus persistence error (negative = better)."""

    physics = default_config().physics
    outputs = predict_records(model, records)
    rows = []
    for index, record in enumerate(records):
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        model_positions, _, model_angle = _body_errors(predicted, actual)
        persistence_positions, _, persistence_angle = _body_errors(before, actual)
        rows.append(
            {
                "object_position": model_positions[0] - persistence_positions[0],
                "robot0_position": model_positions[1] - persistence_positions[1],
                "robot1_position": model_positions[2] - persistence_positions[2],
                "object_angle": model_angle - persistence_angle,
            }
        )
    out: Dict[str, object] = {"n": len(rows)}
    for key in ("object_position", "robot0_position", "robot1_position", "object_angle"):
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        out[key] = {
            "mean_difference": float(values.mean()) if values.size else float("nan"),
            "median_difference": float(np.median(values)) if values.size else float("nan"),
            "fraction_model_better": float(np.mean(values < 0)) if values.size else float("nan"),
        }
    return out


def chain_tolerance_metrics(
    model: JointWindowMLP,
    windows: Sequence[Dict[str, object]],
    *,
    splits: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Teacher-forced vs recursive chains over real episodes, with A4 fixed tolerances.

    The recursive chain is evaluated only over the windows where the true chain is alive, so
    a correct success absorption is never scored as an error. Teacher-forced and recursive
    accumulation is kept in separate arrays with explicit denominators.
    """

    physics = default_config().physics
    episodes: Dict[Tuple[str, str, str], List[Dict[str, object]]] = defaultdict(list)
    for record in windows:
        if splits is not None and record["split"] not in splits:
            continue
        episodes[(str(record["split"]), str(record["scene_id"]), str(record["episode_id"]))].append(record)
    for group in episodes.values():
        group.sort(key=lambda record: int(record["window_index"]))

    depth = max((len(group) for group in episodes.values()), default=0)
    # teacher accumulators
    t_err = [0.0] * depth
    t_ang = [0.0] * depth
    t_n = [0] * depth
    t_invalid = [0] * depth
    t_wrong_term = [0] * depth
    t_breach = [0] * depth
    # recursive accumulators
    r_err = [0.0] * depth
    r_ang = [0.0] * depth
    r_n = [0] * depth
    r_alive = [0] * depth
    r_invalid_at = [0] * depth
    r_wrong_term_at = [0] * depth
    r_breach_at = [0] * depth
    chains = 0
    terminated_at: List[Optional[int]] = []
    breach_at: List[Optional[int]] = []
    invalid_at: List[Optional[int]] = []
    wrong_term_at: List[Optional[int]] = []

    for _key, group in episodes.items():
        chains += 1
        scene = scene_by_name(str(group[0]["scene_id"]))
        true_len = len(group)
        chain_terminated: Optional[int] = None
        chain_breach: Optional[int] = None
        chain_invalid: Optional[int] = None
        chain_wrong_term: Optional[int] = None
        # teacher-forced
        for window, record in enumerate(group):
            before = boundary_before(record)
            actual = boundary_after(record)
            prediction = predict_window(model, scene, before, joint_spec_of(record), physics)
            violation = geometric_violation(
                prediction.next_state.object, prediction.next_state.robots, scene, physics
            )
            positions, _, angle = _body_errors(prediction.next_state, actual)
            position = max(positions)
            invalid = bool(violation["violation"] or not violation["finite"])
            true_index = TERMINATION_CLASSES.index(str(record["termination_class"]))
            pred_index = TERMINATION_CLASSES.index(prediction.termination)
            breach = position > CHAIN_POSITION_TOLERANCE or angle > CHAIN_ANGLE_TOLERANCE
            t_err[window] += position
            t_ang[window] += angle
            t_n[window] += 1
            t_invalid[window] += int(invalid)
            t_wrong_term[window] += int(pred_index != true_index)
            t_breach[window] += int(breach)
        # recursive, paired over the true chain's live windows
        state = boundary_before(group[0])
        for window, record in enumerate(group):
            r_alive[window] += 1
            prediction = predict_window(model, scene, state, joint_spec_of(record), physics)
            violation = geometric_violation(
                prediction.next_state.object, prediction.next_state.robots, scene, physics
            )
            actual = boundary_after(record)
            positions, _, angle = _body_errors(prediction.next_state, actual)
            position = max(positions)
            invalid = bool(violation["violation"] or not violation["finite"])
            true_index = TERMINATION_CLASSES.index(str(record["termination_class"]))
            pred_index = TERMINATION_CLASSES.index(prediction.termination)
            breach = position > CHAIN_POSITION_TOLERANCE or angle > CHAIN_ANGLE_TOLERANCE
            r_err[window] += position
            r_ang[window] += angle
            r_n[window] += 1
            r_invalid_at[window] += int(invalid)
            r_wrong_term_at[window] += int(pred_index != true_index)
            r_breach_at[window] += int(breach)
            if breach and chain_breach is None:
                chain_breach = window
            if invalid:
                chain_invalid = window
                chain_terminated = window
                break
            # a correct success absorption ends the chain without being an error
            if prediction.termination == "success" and true_index == TERMINATION_CLASSES.index("success"):
                chain_terminated = window
                break
            if pred_index != true_index:
                chain_wrong_term = window
                chain_terminated = window
                break
            state = prediction.next_state
        terminated_at.append(chain_terminated)
        breach_at.append(chain_breach)
        invalid_at.append(chain_invalid)
        wrong_term_at.append(chain_wrong_term)

    def _cumulative(values: List[Optional[int]], window: int) -> int:
        return sum(1 for value in values if value is not None and value <= window)

    def _rate(values: Sequence[int], denominators: Sequence[int]) -> List[float]:
        return [value / max(1, denominator) for value, denominator in zip(values, denominators)]

    return {
        "chains": chains,
        "depth": depth,
        "tolerances": {
            "position_m": CHAIN_POSITION_TOLERANCE,
            "angle_rad": CHAIN_ANGLE_TOLERANCE,
        },
        "teacher_forced": {
            "n_predictions": t_n,
            "position_error": [value / max(1, n) for value, n in zip(t_err, t_n)],
            "object_angle_error": [value / max(1, n) for value, n in zip(t_ang, t_n)],
            "invalid_rate": _rate(t_invalid, t_n),
            "wrong_termination_rate": _rate(t_wrong_term, t_n),
            "tolerance_breach_rate": _rate(t_breach, t_n),
        },
        "recursive_survivors": {
            "alive_at_window_start": r_alive,
            "n_predictions": r_n,
            "position_error": [value / max(1, n) for value, n in zip(r_err, r_n)],
            "object_angle_error": [value / max(1, n) for value, n in zip(r_ang, r_n)],
            "invalid_rate": _rate(r_invalid_at, r_n),
            "wrong_termination_rate": _rate(r_wrong_term_at, r_n),
            "tolerance_breach_rate": _rate(r_breach_at, r_n),
        },
        "all_chains": {
            "cumulative_termination_rate": [
                _cumulative(terminated_at, window) / max(1, chains) for window in range(depth)
            ],
            "cumulative_invalid_rate": [
                _cumulative(invalid_at, window) / max(1, chains) for window in range(depth)
            ],
            "cumulative_wrong_termination_rate": [
                _cumulative(wrong_term_at, window) / max(1, chains) for window in range(depth)
            ],
            "cumulative_tolerance_breach_rate": [
                _cumulative(breach_at, window) / max(1, chains) for window in range(depth)
            ],
        },
        "note": (
            "a correct success absorption is not an error; the recursive chain is compared "
            "only over windows where the true chain is alive"
        ),
    }
