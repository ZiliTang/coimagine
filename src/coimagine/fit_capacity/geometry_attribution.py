"""C2: which predicted quantity drives the geometric violations, via offline pose swaps.

Only the frozen geometry checker is used. Velocities and the predicted termination class are
kept from the model output; the swaps are offline sensitivity probes, not reachable states and
never fed to a rollout.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..continuous2d.backend import BodyState
from ..continuous2d.config import default_config
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import PENETRATION_LIMIT, geometric_violation, wrap_angle
from ..continuous2d_learning.evaluation import predict_records
from ..continuous2d_learning.features import BoundaryState, apply_state_delta
from ..continuous2d_learning.model import JointWindowMLP, boundary_after, boundary_before

PAIRS = ("robot_object", "robot_robot", "robot_wall", "object_wall")
MODES = ("object_only", "robots_only", "all")


def _quantiles(values: Sequence[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"p50": None, "p90": None, "p99": None, "max": None, "n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)), "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)), "max": float(array.max()), "n": int(array.size),
    }


def _swap(predicted: BoundaryState, actual: BoundaryState, mode: str) -> BoundaryState:
    """Replace predicted poses by the true next poses; keep predicted velocities."""

    if mode == "object_only":
        obj = BodyState(actual.object.x, actual.object.y, actual.object.angle,
                        predicted.object.vx, predicted.object.vy, predicted.object.omega)
        robots = predicted.robots
    elif mode == "robots_only":
        obj = predicted.object
        robots = tuple(
            BodyState(a.x, a.y, a.angle, p.vx, p.vy, p.omega)
            for a, p in zip(actual.robots, predicted.robots)
        )
    else:
        obj = BodyState(actual.object.x, actual.object.y, actual.object.angle,
                        predicted.object.vx, predicted.object.vy, predicted.object.omega)
        robots = tuple(
            BodyState(a.x, a.y, a.angle, p.vx, p.vy, p.omega)
            for a, p in zip(actual.robots, predicted.robots)
        )
    return BoundaryState(object=obj, robots=robots, success_counter=predicted.success_counter)


def geometry_attribution(
    model: JointWindowMLP, records: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    config = default_config()
    physics = config.physics
    outputs = predict_records(model, records)
    pair_counts: Dict[str, int] = defaultdict(int)
    penetration = []
    illegal_flags: List[int] = []
    errors_when_violating = {"object_position": [], "robot0_position": [], "robot1_position": [],
                             "object_angle": []}
    errors_when_legal = {"object_position": [], "robot0_position": [], "robot1_position": [],
                         "object_angle": []}
    transition = {mode: {"illegal_to_legal": 0, "legal_to_illegal": 0, "illegal_stays": 0,
                         "legal_stays": 0} for mode in MODES}
    reference_modes_illegal = 0
    n = len(records)
    for index, record in enumerate(records):
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        actual = boundary_after(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        original = geometric_violation(predicted.object, predicted.robots, scene, physics)
        illegal = bool(original["violation"] or not original["finite"])
        illegal_flags.append(int(illegal))
        penetration.append(float(original["penetration"]))
        object_position = math.hypot(predicted.object.x - actual.object.x, predicted.object.y - actual.object.y)
        robot_positions = [
            math.hypot(first.x - second.x, first.y - second.y)
            for first, second in zip(predicted.robots, actual.robots)
        ]
        object_angle = abs(wrap_angle(predicted.object.angle - actual.object.angle))
        bucket = errors_when_violating if illegal else errors_when_legal
        bucket["object_position"].append(object_position)
        bucket["robot0_position"].append(robot_positions[0])
        bucket["robot1_position"].append(robot_positions[1])
        bucket["object_angle"].append(object_angle)
        if illegal:
            for name, value in original["penetration_by_pair"].items():
                if value > 0.0:
                    pair_counts[name] += 1
            for name in original["violating_pairs"]:
                pair_counts[f"above_limit:{name}"] += 1
        for mode in MODES:
            swapped = _swap(predicted, actual, mode)
            result = geometric_violation(swapped.object, swapped.robots, scene, physics)
            now_illegal = bool(result["violation"] or not result["finite"])
            key = ("illegal_to_legal" if not now_illegal else "illegal_stays") if illegal else (
                "legal_to_illegal" if now_illegal else "legal_stays"
            )
            transition[mode][key] += 1
            if mode == "all" and now_illegal:
                reference_modes_illegal += 1

    def _mean(values):
        return float(np.mean(values)) if values else None

    return {
        "n": n,
        "original_illegal_rate": _mean([float(v) for v in illegal_flags]),
        "pair_occurrence_rate": {name: count / max(1, n) for name, count in sorted(pair_counts.items())},
        "penetration_quantiles": _quantiles(penetration),
        "errors_when_violating": {key: _mean(values) for key, values in errors_when_violating.items()},
        "errors_when_legal": {key: _mean(values) for key, values in errors_when_legal.items()},
        "errors_when_violating_n": len(errors_when_violating["object_position"]),
        "errors_when_legal_n": len(errors_when_legal["object_position"]),
        "counterfactual": {
            mode: {
                **transition[mode],
                "illegal_rate_after": (
                    transition[mode]["illegal_stays"] + transition[mode]["legal_to_illegal"]
                ) / max(1, n),
            }
            for mode in MODES
        },
        "all_pose_reference_still_illegal": reference_modes_illegal,
        "all_pose_reference_illegal_rate": reference_modes_illegal / max(1, n),
        "penetration_limit": PENETRATION_LIMIT,
        "note": (
            "offline pose swaps with the frozen checker; mixed states are not physically "
            "reachable and are never sent to a rollout"
        ),
    }
