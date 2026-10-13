"""Exact replay of an existing phase-B execution, to read the observation at the exit instant T.

Why a replay is needed. The stage machine breaks out of its loop **before** the period's log row is
appended, so the last logged row is one control period *before* the option's exit. The recorded
terminal robot/object state, however, is captured after the loop, i.e. at the exit instant T. Using
the last logged row as "the terminal observation" would therefore label the record with sensing from
T - 0.05 s, and `_exit_block`'s "last available observation" fallback could reach even further back.

The replay does not create anything new:

* the same root (regenerated deterministically from the frozen salt and the recorded root id),
* the same eight-way joint command (taken from the record itself),
* the same snapshot / exact-restore path, the same controller, protocol, sensing and physics,
* **no extra stepping and no extra control call**: the executor is stopped at its own exit and its
  terminal observation set -- captured inside the same period the loop exited in -- is read.

Every replay is checked against the original record (exit reason, duration in steps, terminal object
and robot states); a mismatch is reported, never silently replaced.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.local_contact_option.contract import DELTA_PER_SPEED, FORCE_LIMIT  # noqa: E402
from coimagine.local_contact_option.option import OptionExecutor  # noqa: E402
from coimagine.local_contact_option.roots import (  # noqa: E402
    VARIANT_SPECS,
    apply_initial_velocities,
)
from coimagine.local_contact_option.runner import build_snapshot  # noqa: E402
from coimagine.local_contact_option.skill import LocalSurfacePush  # noqa: E402

#: the collector's frozen settings; imported lazily so this module stays importable without it
BASELINE_VARIANT = next(spec for spec in VARIANT_SPECS if spec.name == "baseline")


def frozen_option_config():
    from experiments.option_prediction_mechanism_v1.phase_b.collect_b import frozen_option_config as cfg
    return cfg()


def load_roots(scale: Dict[str, int]):
    """Regenerate the tier-2 roots deterministically (same salt, same counts)."""

    from experiments.option_prediction_mechanism_v1.phase_b import roots_b

    roots, reports = roots_b.build_phase_b_roots(scale)
    index = {}
    for split, values in roots.items():
        for root in values:
            index[root.root_id] = root
    return index, reports


def specs_from_record(record: dict) -> Dict[int, LocalSurfacePush]:
    """Rebuild the joint command exactly as recorded."""

    command = record["input"]["command"]
    alpha = command["alpha_rad"]
    speed = command["speed"]
    active = set(command["active_robots"])
    specs: Dict[int, LocalSurfacePush] = {}
    for robot_id in (0, 1):
        if robot_id not in active:
            specs[robot_id] = None  # type: ignore[assignment]
            continue
        specs[robot_id] = LocalSurfacePush(
            alpha=float(alpha[robot_id]), speed=float(speed[robot_id]),
            active_duration=float(command["active_duration"]),
            force_limit=float(command["force_limit"]),
            delta=DELTA_PER_SPEED * float(speed[robot_id]))
    return specs


def _close(a: Sequence[float], b: Sequence[float], tol: float = 1e-9) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def replay(record: dict, roots: Dict[str, object]) -> dict:
    """Replay one recorded execution and return its exit-instant observations plus a check."""

    root = roots.get(record["root_id"])
    if root is None:
        return {"record_id": record["record_id"], "ok": False, "reason": "root_not_regenerated"}
    option = frozen_option_config()
    base = default_config()
    snapshot, config, _initial_object, _initial_robots, _report, velocities = build_snapshot(
        root, BASELINE_VARIANT, base)
    env = Continuous2DEnv(config)
    env.restore(snapshot, exact=True)
    apply_initial_velocities(env, velocities)
    executor = OptionExecutor(option, "V")
    result = executor.run(env, root_id=root.root_id, variant=BASELINE_VARIANT.name,
                          joint_specs=specs_from_record(record),
                          active_robots=tuple(record["input"]["command"]["active_robots"]),
                          noise_rng=None)
    check = {
        "record_id": record["record_id"],
        "exit_reason": result.exit_reason == record["label"]["exit_class"],
        "duration_steps": result.wall_clock_steps == record["label"]["duration_steps"],
        "terminal_object": _close(result.terminal_object, record["truth"]["object_terminal_world"]),
        "terminal_robots": all(
            _close(actual, expected)
            for actual, expected in zip([state for state in result.terminal_robots],
                                        record["truth"]["robots_terminal_world"])),
    }
    last_row = result.rows[-1] if result.rows else None
    return {
        "record_id": record["record_id"], "root_id": record["root_id"],
        "ok": all(check[key] for key in
                  ("exit_reason", "duration_steps", "terminal_object", "terminal_robots")),
        "check": check,
        "exit_reason": result.exit_reason,
        "wall_clock_steps": result.wall_clock_steps,
        "terminal_object": list(result.terminal_object),
        "terminal_robots": [list(state) for state in result.terminal_robots],
        "primary_exit_reason": result.primary_exit_reason,
        "locally_settled": bool(result.locally_settled),
        "settled_observed": bool(result.settled_observed),
        "settled_after_abort": bool(result.settled_after_abort),
        "duration_seconds": result.wall_clock_steps * config.physics.control_dt,
        "terminal_observations": {
            str(robot_id): observation
            for robot_id, observation in (result.terminal_observations or {}).items()},
        "last_logged_period": None if last_row is None else {
            "time": last_row.get("time"), "stage": last_row.get("stage"),
            "observed": last_row.get("observed")},
    }


def observation_summary(observation) -> dict:
    """The parts of a whitelist observation the label extraction needs."""

    if observation is None:
        return {"present": False}
    patch = observation.patch
    return {
        "present": True,
        "own": [observation.own.x, observation.own.y, observation.own.angle,
                observation.own.vx, observation.own.vy, observation.own.omega],
        "contacting": bool(observation.contacting),
        "contact_point_robot": list(observation.contact_point_robot)
        if observation.contact_point_robot else None,
        "patch_valid": patch is not None and len(patch.points_robot) >= 3,
        "patch_distance": None if patch is None else float(patch.distance),
        "patch_points": 0 if patch is None else len(patch.points_robot),
        "patch_segment_robot": None if patch is None
        else [list(patch.segment_robot[0]), list(patch.segment_robot[1])],
        "patch_normal_robot": None if patch is None
        else list(_rotate(-observation.own.angle, patch.normal_world)),
        "patch_confidence": None if patch is None else float(patch.confidence),
        "sample_time": observation.sample_time,
        "time": observation.time,
    }


def _rotate(theta: float, vector: Sequence[float]) -> Tuple[float, float]:
    cos, sin = math.cos(theta), math.sin(theta)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)
