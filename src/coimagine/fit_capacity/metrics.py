"""Unified evaluation metrics for fit_capacity_v1 (task book section 4).

One function does one batched forward pass and returns every required metric, so no caller
can accidentally report a training-log batch mean instead of a full-dataset evaluation.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from ..continuous2d.config import default_config
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import (
    TERMINATION_CLASSES,
    geometric_violation,
    wrap_angle,
)
from ..continuous2d_learning.features import apply_state_delta
from ..continuous2d_learning.model import (
    JointWindowMLP,
    _tensors,
    boundary_after,
    boundary_before,
    state_residual,
)

VIOLATION_PAIRS = ("robot_object", "robot_robot", "robot_wall", "object_wall")


def loss_components(
    outputs: Dict[str, torch.Tensor], targets: Tuple[torch.Tensor, ...]
) -> Tuple[torch.Tensor, ...]:
    """Per-sample state / hold / duration / termination losses (same convention as training)."""

    state_target, hold_target, duration_target, termination_target = targets
    state = (state_residual(outputs["state"], state_target) ** 2).mean(dim=1)
    hold = ((torch.sigmoid(outputs["hold"]) - hold_target) ** 2).squeeze(1)
    duration = ((torch.sigmoid(outputs["duration"]) - duration_target) ** 2).squeeze(1)
    termination = nn.functional.cross_entropy(
        outputs["termination"], termination_target, reduction="none"
    )
    return state, hold, duration, termination


def _percentile(values: Sequence[float], q: float) -> Optional[float]:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _group_of(record: Dict[str, object]) -> Dict[str, bool]:
    before, after = record["state"], record["next_state"]
    dx = float(after["object"][0]) - float(before["object"][0])
    dy = float(after["object"][1]) - float(before["object"][1])
    rotation = abs(wrap_angle(float(after["object"][2]) - float(before["object"][2])))
    moved = math.hypot(dx, dy) > 0.01
    turned = rotation > 0.02
    family = str(record["scene_id"]).split("_")[0]
    return {
        "all": True,
        "S0": family == "S0",
        "S1": family == "S1",
        "success": int(record["success"]) == 1,
        "non_success": int(record["success"]) != 1,
    }


def evaluate(
    model: JointWindowMLP,
    records: Sequence[Dict[str, object]],
    *,
    batch_size: int = 512,
) -> Dict[str, object]:
    """Full-dataset evaluation under the frozen metric contract."""

    if not records:
        return {"n": 0, "null_reason": "empty record set"}
    config = default_config()
    physics = config.physics
    tensors = _tensors(records)
    n = len(records)
    state_loss = np.zeros(n)
    hold_loss = np.zeros(n)
    duration_loss = np.zeros(n)
    termination_loss = np.zeros(n)
    object_position = np.zeros(n)
    robot_position = np.zeros((n, 2))
    object_angle = np.zeros(n)
    object_speed = np.zeros(n)
    robot_speed = np.zeros((n, 2))
    object_omega = np.zeros(n)
    persistence_object = np.zeros(n)
    persistence_angle = np.zeros(n)
    invalid = np.zeros(n, dtype=bool)
    non_finite = np.zeros(n, dtype=bool)
    penetration = np.zeros(n)
    penetration_by_pair = {name: np.zeros(n) for name in VIOLATION_PAIRS}
    predicted_class = np.zeros(n, dtype=np.int64)
    true_class = np.zeros(n, dtype=np.int64)
    violation_types: Dict[str, int] = defaultdict(int)
    predicted_success_and_illegal = 0

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            index = slice(start, min(start + batch_size, n))
            outputs = model(tensors[0][index])
            components = loss_components(outputs, tuple(t[index] for t in tensors[1:]))
            state_loss[start : index.stop] = components[0].numpy()
            hold_loss[start : index.stop] = components[1].numpy()
            duration_loss[start : index.stop] = components[2].numpy()
            termination_loss[start : index.stop] = components[3].numpy()
            delta = outputs["state"].numpy()
            termination = outputs["termination"].argmax(dim=1).numpy()
            for row, record_index in enumerate(range(start, min(start + batch_size, n))):
                record = records[record_index]
                scene = scene_by_name(str(record["scene_id"]))
                before = boundary_before(record)
                actual = boundary_after(record)
                predicted = apply_state_delta(before, delta[row], scene)
                object_position[record_index] = math.hypot(
                    predicted.object.x - actual.object.x, predicted.object.y - actual.object.y
                )
                for robot in (0, 1):
                    first, second = predicted.robots[robot], actual.robots[robot]
                    robot_position[record_index, robot] = math.hypot(first.x - second.x, first.y - second.y)
                    robot_speed[record_index, robot] = math.hypot(
                        first.vx - second.vx, first.vy - second.vy
                    )
                object_angle[record_index] = abs(wrap_angle(predicted.object.angle - actual.object.angle))
                object_speed[record_index] = math.hypot(
                    predicted.object.vx - actual.object.vx, predicted.object.vy - actual.object.vy
                )
                object_omega[record_index] = abs(predicted.object.omega - actual.object.omega)
                persistence_object[record_index] = math.hypot(
                    before.object.x - actual.object.x, before.object.y - actual.object.y
                )
                persistence_angle[record_index] = abs(
                    wrap_angle(before.object.angle - actual.object.angle)
                )
                result = geometric_violation(predicted.object, predicted.robots, scene, physics)
                invalid[record_index] = bool(result["violation"])
                non_finite[record_index] = not bool(result["finite"])
                penetration[record_index] = float(result["penetration"])
                for name in VIOLATION_PAIRS:
                    penetration_by_pair[name][record_index] = float(result["penetration_by_pair"].get(name, 0.0))
                for reason in result["reasons"]:
                    violation_types[str(reason)] += 1
                for pair in result["violating_pairs"]:
                    violation_types[f"pair:{pair}"] += 1
                predicted_class[record_index] = int(termination[row])
                true_class[record_index] = TERMINATION_CLASSES.index(str(record["termination_class"]))
                if int(termination[row]) == TERMINATION_CLASSES.index("success") and invalid[record_index]:
                    predicted_success_and_illegal += 1

    confusion = np.zeros((3, 3), dtype=np.int64)
    for true_value, predicted_value in zip(true_class, predicted_class):
        confusion[true_value, predicted_value] += 1
    success_index = TERMINATION_CLASSES.index("success")
    true_success = int(np.sum(true_class == success_index))
    predicted_success = int(np.sum(predicted_class == success_index))
    true_positive = int(np.sum((true_class == success_index) & (predicted_class == success_index)))

    groups: Dict[str, Dict[str, object]] = {}
    for name in ("all", "S0", "S1", "success", "non_success"):
        mask = np.asarray([_group_of(record)[name] for record in records], dtype=bool)
        if not mask.any():
            groups[name] = {"n": 0}
            continue
        groups[name] = {
            "n": int(mask.sum()),
            "object_position_error": float(object_position[mask].mean()),
            "object_position_error_p90": _percentile(object_position[mask].tolist(), 90),
            "robot0_position_error": float(robot_position[mask, 0].mean()),
            "robot1_position_error": float(robot_position[mask, 1].mean()),
            "object_angle_error": float(object_angle[mask].mean()),
            "invalid_rate": float(invalid[mask].mean()),
            "total_loss": float(
                (state_loss[mask] + hold_loss[mask] + duration_loss[mask] + termination_loss[mask]).mean()
            ),
        }
    by_scene: Dict[str, List[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_scene[str(record["scene_id"])].append(index)
    scene_equal_weight = {}
    for scene, indices in by_scene.items():
        scene_equal_weight[scene] = {
            "n": len(indices),
            "object_position_error": float(object_position[indices].mean()),
            "invalid_rate": float(invalid[indices].mean()),
            "total_loss": float(
                (state_loss[indices] + hold_loss[indices] + duration_loss[indices] + termination_loss[indices]).mean()
            ),
        }
    equal_weight_mean = {
        key: _mean([row[key] for row in scene_equal_weight.values()])
        for key in ("object_position_error", "invalid_rate", "total_loss")
    }

    return {
        "n": n,
        "loss": {
            "state": float(state_loss.mean()),
            "hold": float(hold_loss.mean()),
            "duration": float(duration_loss.mean()),
            "termination": float(termination_loss.mean()),
            "total": float((state_loss + hold_loss + duration_loss + termination_loss).mean()),
        },
        "position": {
            "object_mean": float(object_position.mean()),
            "object_p90": _percentile(object_position.tolist(), 90),
            "robot0_mean": float(robot_position[:, 0].mean()),
            "robot1_mean": float(robot_position[:, 1].mean()),
            "units": "m",
        },
        "angle": {
            "object_mean": float(object_angle.mean()),
            "object_p90": _percentile(object_angle.tolist(), 90),
            "units": "rad (wrapped)",
        },
        "velocity": {
            "object_speed_mean": float(object_speed.mean()),
            "robot0_speed_mean": float(robot_speed[:, 0].mean()),
            "robot1_speed_mean": float(robot_speed[:, 1].mean()),
            "object_omega_mean": float(object_omega.mean()),
            "units": "m/s and rad/s",
        },
        "legality": {
            "invalid_rate": float(invalid.mean()),
            "invalid_count": int(invalid.sum()),
            "non_finite_count": int(non_finite.sum()),
            "violation_types": dict(violation_types),
            "penetration_p50": _percentile(penetration.tolist(), 50),
            "penetration_p90": _percentile(penetration.tolist(), 90),
            "penetration_p99": _percentile(penetration.tolist(), 99),
            "penetration_by_pair_mean": {
                name: float(values.mean()) for name, values in penetration_by_pair.items()
            },
        },
        "termination": {
            "classes": list(TERMINATION_CLASSES),
            "confusion": confusion.tolist(),
            "true_success": true_success,
            "predicted_success": predicted_success,
            "success_recall": (true_positive / true_success) if true_success else None,
            "success_recall_denominator": true_success,
            "success_precision": (true_positive / predicted_success) if predicted_success else None,
            "success_precision_denominator": predicted_success,
            "predicted_success_and_illegal": predicted_success_and_illegal,
            "physical_failure_windows": int(np.sum(true_class == TERMINATION_CLASSES.index("physical_failure"))),
            "physical_failure_note": (
                "not validated" if int(np.sum(true_class == TERMINATION_CLASSES.index("physical_failure"))) == 0 else ""
            ),
        },
        "persistence": {
            "object_position_error": float(persistence_object.mean()),
            "object_angle_error": float(persistence_angle.mean()),
            "paired_object_position_difference": float((object_position - persistence_object).mean()),
            "paired_object_angle_difference": float((object_angle - persistence_angle).mean()),
            "fraction_model_better_position": float(np.mean(object_position < persistence_object)),
        },
        "groups": groups,
        "per_scene": scene_equal_weight,
        "scene_equal_weight_mean": equal_weight_mean,
    }
