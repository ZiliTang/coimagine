"""Phase D2: two separate test tables (original regression set and the new branch set)."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.config import default_config
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import TERMINATION_CLASSES, geometric_violation, wrap_angle
from ..continuous2d_learning.evaluation import predict_records
from ..continuous2d_learning.features import apply_state_delta
from ..continuous2d_learning.model import JointWindowMLP, boundary_after, boundary_before
from ..continuous2d_learning.rollout import distance_of_state
from ..continuous2d_learning.features import BoundaryState


def _spearman(first: Sequence[float], second: Sequence[float]) -> Optional[float]:
    if len(first) < 3:
        return None
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return None
    rank_a = np.argsort(np.argsort(a)).astype(np.float64)
    rank_b = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(rank_a, rank_b)[0, 1])


def evaluate_branch_set(
    model: JointWindowMLP, records: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Metrics on the new branch test set, aggregated per anchor first, then per scene."""

    config = default_config()
    physics = config.physics
    outputs = predict_records(model, records)
    predicted_success = 0
    true_success = 0
    true_positive = 0
    feasible_branches = 0
    rejected_feasible = 0
    object_errors: List[float] = []
    robot_errors: List[float] = [[], []]
    angle_errors: List[float] = []
    invalids = 0
    per_anchor: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for index, record in enumerate(records):
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        violation = geometric_violation(predicted.object, predicted.robots, scene, physics)
        invalid = bool(violation["violation"] or not violation["finite"])
        invalids += int(invalid)
        object_errors.append(math.hypot(predicted.object.x - actual.object.x, predicted.object.y - actual.object.y))
        for robot in (0, 1):
            first, second = predicted.robots[robot], actual.robots[robot]
            robot_errors[robot].append(math.hypot(first.x - second.x, first.y - second.y))
        angle_errors.append(abs(wrap_angle(predicted.object.angle - actual.object.angle)))
        predicted_class = TERMINATION_CLASSES[int(outputs["termination"][index])]
        is_success = int(record["success"]) == 1
        true_success += int(is_success)
        if predicted_class == "success":
            predicted_success += 1
            if is_success:
                true_positive += 1
        # a feasible branch is one whose true outcome is not a physical failure
        if str(record["termination_class"]) != "physical_failure":
            feasible_branches += 1
            if invalid:
                rejected_feasible += 1
        source = record.get("source", {}) or {}
        anchor = source.get("anchor_id")
        if anchor and source.get("branch_window") == 0:
            true_progress = distance_of_state(before, scene, config) - distance_of_state(actual, scene, config)
            predicted_progress = distance_of_state(before, scene, config) - distance_of_state(predicted, scene, config)
            per_anchor[str(anchor)].append(
                {
                    "true_progress": true_progress,
                    "predicted_progress": predicted_progress,
                    "invalid": invalid,
                }
            )
    correlations = []
    anchor_rows = []
    for anchor, rows in sorted(per_anchor.items()):
        rho = _spearman([row["true_progress"] for row in rows], [row["predicted_progress"] for row in rows])
        if rho is not None:
            correlations.append(rho)
        anchor_rows.append({"anchor_id": anchor, "n_actions": len(rows), "spearman": rho})
    return {
        "n": len(records),
        "object_position_error": float(np.mean(object_errors)) if object_errors else None,
        "robot0_position_error": float(np.mean(robot_errors[0])) if robot_errors[0] else None,
        "robot1_position_error": float(np.mean(robot_errors[1])) if robot_errors[1] else None,
        "object_angle_error": float(np.mean(angle_errors)) if angle_errors else None,
        "invalid_rate": invalids / max(1, len(records)),
        "true_success_windows": true_success,
        "predicted_success_windows": predicted_success,
        "success_recall": (true_positive / true_success) if true_success else None,
        "success_precision": (true_positive / predicted_success) if predicted_success else None,
        "success_recall_denominator": true_success,
        "success_precision_denominator": predicted_success,
        "feasible_branches": feasible_branches,
        "rejected_feasible_branches": rejected_feasible,
        "rejected_feasible_rate": (rejected_feasible / feasible_branches) if feasible_branches else None,
        "anchors": anchor_rows,
        "anchor_spearman_mean": float(np.mean(correlations)) if correlations else None,
        "anchor_spearman_n": len(correlations),
    }
