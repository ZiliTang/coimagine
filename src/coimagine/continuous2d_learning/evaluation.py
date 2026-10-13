"""Phase B4: single-step test metrics and the state-persistence comparison.

Teacher-forced one-window predictions on the test split. This is a diagnostic of the model
interface and accuracy, not a planning result.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from ..continuous2d.config import default_config
from ..continuous2d.scenes import scene_by_name
from .contract import (
    DURATION_SCALE,
    HOLD_SCALE,
    TERMINATION_CLASSES,
    geometric_violation,
    wrap_angle,
)
from .features import (
    BoundaryState,
    apply_state_delta,
    encode_input,
)
from .model import (
    JointWindowMLP,
    boundary_after,
    boundary_before,
    joint_spec_of,
)
from .records import window_groups


def _batch_inputs(records: Sequence[Dict[str, object]]) -> torch.Tensor:
    physics = default_config().physics
    rows: List[List[float]] = []
    for record in records:
        scene = scene_by_name(str(record["scene_id"]))
        rows.append(encode_input(scene, boundary_before(record), joint_spec_of(record), physics))
    return torch.tensor(rows, dtype=torch.float32)


def predict_records(
    model: JointWindowMLP, records: Sequence[Dict[str, object]], *, batch_size: int = 512
) -> Dict[str, np.ndarray]:
    """One-step (teacher-forced) predictions for every record."""

    features = _batch_inputs(records)
    states: List[np.ndarray] = []
    holds: List[float] = []
    durations: List[float] = []
    terminations: List[int] = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            outputs = model(features[start : start + batch_size])
            states.append(outputs["state"].numpy())
            holds.extend((torch.sigmoid(outputs["hold"])[:, 0] * HOLD_SCALE).tolist())
            durations.extend((torch.sigmoid(outputs["duration"])[:, 0] * DURATION_SCALE).tolist())
            terminations.extend(
                outputs["termination"].argmax(dim=1).tolist()
            )
    return {
        "state_delta": np.concatenate(states, axis=0),
        "hold": np.asarray(holds),
        "duration": np.asarray(durations),
        "termination": np.asarray(terminations, dtype=np.int64),
    }


def _position_error(predicted: BoundaryState, actual: BoundaryState) -> Tuple[float, float, float]:
    def _one(a, b) -> float:
        return float(np.hypot(a.x - b.x, a.y - b.y))

    return (
        _one(predicted.object, actual.object),
        _one(predicted.robots[0], actual.robots[0]),
        _one(predicted.robots[1], actual.robots[1]),
    )


def _velocity_error(predicted: BoundaryState, actual: BoundaryState) -> Tuple[float, float, float]:
    def _one(a, b) -> float:
        return float(np.hypot(a.vx - b.vx, a.vy - b.vy))

    return (
        _one(predicted.object, actual.object),
        _one(predicted.robots[0], actual.robots[0]),
        _one(predicted.robots[1], actual.robots[1]),
    )


def evaluate_records(
    model: JointWindowMLP, records: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    physics = default_config().physics
    outputs = predict_records(model, records)
    groups = ("object_translation", "object_rotation", "stationary")
    accum: Dict[str, Dict[str, List[float]]] = {
        name: {"object_pos": [], "robot0_pos": [], "robot1_pos": [],
               "object_vel": [], "robot0_vel": [], "robot1_vel": [],
               "object_angle": [], "hold": [], "duration": [], "invalid": []}
        for name in (*groups, "all")
    }
    confusion = np.zeros((3, 3), dtype=np.int64)
    false_success = 0
    predicted_success = 0
    validity_reasons: Dict[str, int] = {}
    for index, record in enumerate(records):
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        violation = geometric_violation(predicted.object, predicted.robots, scene, physics)
        for reason in violation["reasons"]:
            validity_reasons[str(reason)] = validity_reasons.get(str(reason), 0) + 1
        pos = _position_error(predicted, actual)
        vel = _velocity_error(predicted, actual)
        angle = abs(wrap_angle(predicted.object.angle - actual.object.angle))
        hold_error = abs(float(outputs["hold"][index]) - float(record["success_counter_after"]))
        duration_error = abs(float(outputs["duration"][index]) - float(record["duration"]))
        flags = window_groups(record)
        labels = ["all"]
        for name in groups:
            if flags[name]:
                labels.append(name)
        for label in labels:
            row = accum[label]
            row["object_pos"].append(pos[0])
            row["robot0_pos"].append(pos[1])
            row["robot1_pos"].append(pos[2])
            row["object_vel"].append(vel[0])
            row["robot0_vel"].append(vel[1])
            row["robot1_vel"].append(vel[2])
            row["object_angle"].append(angle)
            row["hold"].append(hold_error)
            row["duration"].append(duration_error)
            row["invalid"].append(float(violation["violation"]))
        true_index = TERMINATION_CLASSES.index(str(record["termination_class"]))
        predicted_index = int(outputs["termination"][index])
        confusion[true_index, predicted_index] += 1
        if predicted_index == TERMINATION_CLASSES.index("success"):
            predicted_success += 1
            if true_index != TERMINATION_CLASSES.index("success"):
                false_success += 1

    def _mean(values: Sequence[float]) -> float:
        return float(np.mean(values)) if len(values) else float("nan")

    report: Dict[str, object] = {
        "n": len(records),
        "termination_confusion": confusion.tolist(),
        "termination_classes": list(TERMINATION_CLASSES),
        "predicted_success": predicted_success,
        "false_success": false_success,
        "false_success_rate": false_success / predicted_success if predicted_success else float("nan"),
        "raw_invalid_rate": _mean(accum["all"]["invalid"]),
        "invalid_reasons": validity_reasons,
        "groups": {},
    }
    for name, row in accum.items():
        report["groups"][name] = {
            "n": len(row["object_pos"]),
            "object_position_error": _mean(row["object_pos"]),
            "robot0_position_error": _mean(row["robot0_pos"]),
            "robot1_position_error": _mean(row["robot1_pos"]),
            "object_velocity_error": _mean(row["object_vel"]),
            "robot0_velocity_error": _mean(row["robot0_vel"]),
            "robot1_velocity_error": _mean(row["robot1_vel"]),
            "object_angle_error": _mean(row["object_angle"]),
            "hold_error": _mean(row["hold"]),
            "duration_error": _mean(row["duration"]),
            "invalid_rate": _mean(row["invalid"]),
        }
    return report


def persistence_report(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """The state-persistence reference: state unchanged, duration 2 s, no termination."""

    positions = []
    angles = []
    duration_errors = []
    hold_errors = []
    for record in records:
        before = boundary_before(record)
        actual = boundary_after(record)
        pos = _position_error(before, actual)
        positions.append(pos)
        angles.append(abs(wrap_angle(before.object.angle - actual.object.angle)))
        duration_errors.append(abs(DURATION_SCALE - float(record["duration"])))
        hold_errors.append(abs(0.0 - float(record["success_counter_after"])))
    array = np.asarray(positions)
    return {
        "n": len(records),
        "object_position_error": float(array[:, 0].mean()) if len(array) else float("nan"),
        "robot0_position_error": float(array[:, 1].mean()) if len(array) else float("nan"),
        "robot1_position_error": float(array[:, 2].mean()) if len(array) else float("nan"),
        "object_angle_error": float(np.mean(angles)) if angles else float("nan"),
        "duration_error": float(np.mean(duration_errors)) if duration_errors else float("nan"),
        "hold_error": float(np.mean(hold_errors)) if hold_errors else float("nan"),
    }
