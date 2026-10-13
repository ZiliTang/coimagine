"""Phase B1: accept (or reject) the independent geometric check against the real engine.

The check is not assumed to be a correct oracle. It is applied to every *true* window end
state in the existing dataset and compared with the engine's own penetration bookkeeping.
Misclassified true states are saved one by one. Only ``GEOMETRY_BLOCKED`` if the
disagreement is systematic beyond numerical tolerance.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.backend import BodyState, PymunkBackend
from ..continuous2d.config import Continuous2DConfig, default_config
from ..continuous2d.scenes import scene_by_name
from ..continuous2d_learning.contract import PENETRATION_LIMIT, geometric_violation
from ..continuous2d_learning.features import BoundaryState

#: Differences below this are treated as numerical agreement, not a definition mismatch.
ENGINE_TOLERANCE = 1e-3


def boundary_from_record(record: Dict[str, object], key: str) -> BoundaryState:
    payload = record[key]
    return BoundaryState(
        object=BodyState(*[float(v) for v in payload["object"]]),  # type: ignore[arg-type,index]
        robots=tuple(  # type: ignore[arg-type]
            BodyState(*[float(v) for v in row]) for row in payload["robots"]  # type: ignore[arg-type]
        ),
        success_counter=float(record.get("success_counter_after", 0.0)),
    )


def _percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "max": float("nan"), "n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
        "n": int(array.size),
    }


def audit_true_states(
    windows: Sequence[Dict[str, object]],
    config: Optional[Continuous2DConfig] = None,
) -> Dict[str, object]:
    """Apply the checker to every true window end state, grouped by the true outcome."""

    config = config or default_config()
    groups: Dict[str, Dict[str, object]] = {}
    misclassified: List[Dict[str, object]] = []
    pair_values: Dict[str, List[float]] = defaultdict(list)
    reasons: Dict[str, int] = defaultdict(int)
    pair_violations: Dict[str, int] = defaultdict(int)

    for record in windows:
        truth = str(record["termination_class"])
        scene = scene_by_name(str(record["scene_id"]))
        state = boundary_from_record(record, "next_state")
        result = geometric_violation(state.object, state.robots, scene, config.physics)
        row = groups.setdefault(
            truth,
            {
                "windows": 0, "violations": 0, "out_of_bounds": 0, "penetration_limit": 0,
                "penetration": [], "by_pair": defaultdict(list),
            },
        )
        row["windows"] += 1
        if result["violation"]:
            row["violations"] += 1
            for reason in result["reasons"]:
                reasons[str(reason)] += 1
            for pair in result["violating_pairs"]:
                pair_violations[str(pair)] += 1
            misclassified.append(
                {
                    "episode_id": record["episode_id"],
                    "scene_id": record["scene_id"],
                    "split": record["split"],
                    "window_index": record["window_index"],
                    "termination_class": truth,
                    "termination": record["termination"],
                    "penetration": result["penetration"],
                    "penetration_by_pair": result["penetration_by_pair"],
                    "reasons": result["reasons"],
                    "engine_max_penetration_in_window": record["contact"]["max_penetration"],
                    "state": record["next_state"],
                }
            )
        row["penetration"].append(float(result["penetration"]))
        for name, value in result["penetration_by_pair"].items():
            row["by_pair"][name].append(float(value))
            pair_values[name].append(float(value))

    report: Dict[str, object] = {"groups": {}, "penetration_by_pair": {}, "violating_pairs": dict(pair_violations)}
    for truth, row in sorted(groups.items()):
        report["groups"][truth] = {
            "windows": row["windows"],
            "violations": row["violations"],
            "false_illegal_rate": row["violations"] / max(1, row["windows"]),
            "out_of_bounds": row["out_of_bounds"],
            "penetration_limit": row["penetration_limit"],
            "penetration": _percentiles(row["penetration"]),
            "by_pair": {name: _percentiles(values) for name, values in sorted(row["by_pair"].items())},
        }
    for name, values in sorted(pair_values.items()):
        report["penetration_by_pair"][name] = _percentiles(values)
    report["reasons"] = dict(reasons)
    report["misclassified"] = misclassified
    report["n_windows"] = len(windows)
    return report


def audit_engine_agreement(
    windows: Sequence[Dict[str, object]], config: Optional[Continuous2DConfig] = None
) -> Dict[str, object]:
    """Compare the checker with the engine **at the same instant** (full-step episodes)."""

    config = config or default_config()
    differences: List[float] = []
    per_pair_difference: Dict[str, List[float]] = defaultdict(list)
    near_threshold: List[Dict[str, object]] = []
    steps = 0
    for record in windows:
        if "steps" not in record:
            continue
        scene = scene_by_name(str(record["scene_id"]))
        for step in record["steps"]:  # type: ignore[union-attr]
            state = step["next_state"]
            boundary = BoundaryState(
                object=BodyState(*[float(v) for v in state["object"]]),
                robots=tuple(BodyState(*[float(v) for v in row]) for row in state["robots"]),
                success_counter=0.0,
            )
            result = geometric_violation(boundary.object, boundary.robots, scene, config.physics)
            engine = {name: float(value) for name, value in step["penetration_by_pair"].items()}
            steps += 1
            for name, value in result["penetration_by_pair"].items():
                engine_value = engine.get(name, 0.0)
                delta = value - engine_value
                per_pair_difference[name].append(delta)
                differences.append(abs(delta))
            if abs(result["penetration"] - float(step["penetration"])) > ENGINE_TOLERANCE or (
                PENETRATION_LIMIT * 0.5 <= result["penetration"] <= PENETRATION_LIMIT * 1.5
            ):
                near_threshold.append(
                    {
                        "episode_id": record["episode_id"],
                        "window_index": record["window_index"],
                        "step_index": step["step_index"],
                        "checker_penetration": result["penetration"],
                        "checker_by_pair": result["penetration_by_pair"],
                        "engine_penetration": float(step["penetration"]),
                        "engine_by_pair": engine,
                        "state": state,
                    }
                )
    array = np.asarray(differences, dtype=np.float64) if differences else np.zeros(0)
    report = {
        "steps": steps,
        "max_abs_difference": float(array.max()) if array.size else 0.0,
        "p99_abs_difference": float(np.percentile(array, 99)) if array.size else 0.0,
        "mean_abs_difference": float(array.mean()) if array.size else 0.0,
        "per_pair_mean_difference": {
            name: float(np.mean(values)) for name, values in sorted(per_pair_difference.items())
        },
        "near_threshold_cases": near_threshold,
        "tolerance": ENGINE_TOLERANCE,
    }
    report["systematic_disagreement"] = bool(
        report["max_abs_difference"] > ENGINE_TOLERANCE and report["mean_abs_difference"] > ENGINE_TOLERANCE
    )
    return report


def audit_predicted_states(
    windows: Sequence[Dict[str, object]],
    model,
    config: Optional[Continuous2DConfig] = None,
) -> Dict[str, object]:
    """Apply the same checker to one model's teacher-forced predicted next states."""

    from ..continuous2d_learning.features import apply_state_delta
    from ..continuous2d_learning.model import boundary_before, joint_spec_of
    from ..continuous2d_learning.rollout import boundary_from_dict  # noqa: F401  (documented import)

    config = config or default_config()
    physics = config.physics
    from ..continuous2d_learning.evaluation import predict_records

    outputs = predict_records(model, windows)
    groups: Dict[str, List[float]] = defaultdict(list)
    violating_types: Dict[str, int] = defaultdict(int)
    reasons: Dict[str, int] = defaultdict(int)
    per_group: Dict[str, Dict[str, object]] = {}
    for index, record in enumerate(windows):
        group = str(record["termination_class"])
        scene = scene_by_name(str(record["scene_id"]))
        before = boundary_before(record)
        predicted = apply_state_delta(before, outputs["state_delta"][index], scene)
        result = geometric_violation(predicted.object, predicted.robots, scene, physics)
        groups[group].append(float(result["penetration"]))
        if result["violation"]:
            for pair in result["violating_pairs"]:
                violating_types[str(pair)] += 1
            for reason in result["reasons"]:
                reasons[str(reason)] += 1
    for group, values in sorted(groups.items()):
        per_group[group] = _percentiles(values)
    all_values = [value for values in groups.values() for value in values]
    return {
        "overall": _percentiles(all_values),
        "by_true_outcome": per_group,
        "violating_pair_counts": dict(violating_types),
        "reason_counts": dict(reasons),
        "penetration_limit": PENETRATION_LIMIT,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))


def convention_probe(config: Optional[Continuous2DConfig] = None) -> Dict[str, object]:
    """Controlled probe of the engine's per-pair contact-distance sign convention.

    The frozen failure rule is ``max(0, -distance)`` over ``shapes_collide`` contacts. For
    Circle-vs-Segment and Polygon-vs-Polygon an overlap gives a negative distance, but for
    Polygon-vs-Segment it gives a **positive** distance, so object-into-arena-boundary
    overlap is invisible to the engine's own ``penetration_by_pair`` bookkeeping. The
    independent checker measures the physical overlap in all three cases.
    """

    config = config or default_config()
    physics = config.physics
    s0 = scene_by_name("S0_dev")
    s1 = scene_by_name("S1_dev")
    out: Dict[str, object] = {"engine_rule": "penetration = max(0, -contact.distance)"}
    rows = []
    for cy in (0.24, 0.20, 0.15):
        backend = PymunkBackend(s0, physics, config.config_hash())
        backend._write_states(
            BodyState(0.6, 1.0, 0.0, 0, 0, 0),
            (BodyState(0.6, cy, 0.0, 0, 0, 0), BodyState(2.6, 1.0, 0.0, 0, 0, 0)),
        )
        contacts = [
            {"distance": float(p.distance)}
            for wall in backend.static_shapes
            for p in backend.robot_shapes[0].shapes_collide(wall).points
        ]
        overlap = max(0.0, 0.05 - (cy - physics.robot_radius))
        rows.append(
            {
                "pair": "robot_boundary(circle-segment)", "geometric_overlap": overlap,
                "engine_contacts": contacts, "engine_penetration": backend.penetration_by_pair()["robot_wall"],
                "engine_records_overlap": backend.penetration_by_pair()["robot_wall"] > 0,
            }
        )
    for cy in (0.17, 0.14, 0.10):
        backend = PymunkBackend(s0, physics, config.config_hash())
        backend._write_states(
            BodyState(1.2, cy, 0.0, 0, 0, 0),
            (BodyState(0.5, 1.5, 0.0, 0, 0, 0), BodyState(2.5, 1.5, 0.0, 0, 0, 0)),
        )
        contacts = [
            {"distance": float(p.distance)}
            for wall in backend.static_shapes
            for p in backend.object_shape.shapes_collide(wall).points
        ]
        overlap = max(0.0, 0.05 - (cy - physics.object_height / 2.0))
        rows.append(
            {
                "pair": "object_boundary(polygon-segment)", "geometric_overlap": overlap,
                "engine_contacts": contacts, "engine_penetration": backend.penetration_by_pair()["object_wall"],
                "engine_records_overlap": backend.penetration_by_pair()["object_wall"] > 0,
            }
        )
    for cy in (0.80, 0.75, 0.70):
        backend = PymunkBackend(s1, physics, config.config_hash())
        backend._write_states(
            BodyState(1.60, cy, 0.0, 0, 0, 0),
            (BodyState(0.1, 0.1, 0.0, 0, 0, 0), BodyState(2.9, 1.9, 0.0, 0, 0, 0)),
        )
        contacts = [
            {"distance": float(p.distance)}
            for wall in backend.static_shapes
            for p in backend.object_shape.shapes_collide(wall).points
        ]
        wall_top = 0.65
        overlap = max(0.0, wall_top - (cy - physics.object_height / 2.0))
        rows.append(
            {
                "pair": "object_obstacle(polygon-polygon)", "geometric_overlap": overlap,
                "engine_contacts": contacts, "engine_penetration": backend.penetration_by_pair()["object_wall"],
                "engine_records_overlap": backend.penetration_by_pair()["object_wall"] > 0,
            }
        )
    out["rows"] = rows
    out["conclusion"] = (
        "engine records circle-segment and polygon-polygon overlap but not polygon-segment "
        "overlap; the independent checker uses the physical overlap for all three"
    )
    return out
