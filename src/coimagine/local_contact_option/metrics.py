"""The three metric layers of task book section 3.3.

Layer 1 -- was the execution clear and physically valid.
Layer 2 -- is the state transition more limited and more controllable.
Layer 3 -- is there a low-dimensional motion approximation.

The evaluator may use ground truth; the controller never sees any of it. In particular the
object's SE(2) displacement is expressed in the **initial object frame**, and that transform is
computed here, outside the controller.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .contract import PUSH_TARGET_ACTIVE

PUSH_SECONDS = PUSH_TARGET_ACTIVE
MICRO_LINEAR = 0.02
MICRO_ANGULAR = 0.03
MIN_MOTION_FOR_TWIST = 0.02
MIN_ROTATION_FOR_TWIST = 0.05


def wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def to_initial_frame(pose: Sequence[float], origin: Sequence[float]) -> Tuple[float, float, float]:
    """Express ``pose`` (x, y, angle) in the initial object frame given by ``origin``."""

    cos, sin = math.cos(-origin[2]), math.sin(-origin[2])
    dx, dy = pose[0] - origin[0], pose[1] - origin[1]
    return (dx * cos - dy * sin, dx * sin + dy * cos, wrap_angle(pose[2] - origin[2]))


def _stage_runs(rows: Sequence[Dict[str, object]]) -> List[Tuple[str, int, int]]:
    runs: List[Tuple[str, int, int]] = []
    for index, row in enumerate(rows):
        stage = str(row.get("phase", row.get("stage", "?")))
        if not runs or runs[-1][0] != stage:
            runs.append((stage, index, index))
        else:
            runs[-1] = (stage, runs[-1][1], index)
    return runs


# ---------------------------------------------------------------------------
# layer 3: constant body-frame twist
# ---------------------------------------------------------------------------


def _basis(omega: float, t: float) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if abs(omega) < 1e-6:
        return (t, 0.0), (0.0, t)
    a = math.sin(omega * t) / omega
    b = -(1.0 - math.cos(omega * t)) / omega
    c = (1.0 - math.cos(omega * t)) / omega
    d = math.sin(omega * t) / omega
    return (a, c), (b, d)


def fit_constant_twist(
    times: Sequence[float],
    positions: Sequence[Tuple[float, float]],
    angles: Sequence[float],
    *,
    omega_range: float = 3.0,
    omega_steps: int = 601,
) -> Optional[Dict[str, float]]:
    """Least-squares constant body-frame twist ``(vx, vy, omega)``.

    A straight line is the ``omega = 0`` special case of the same model, so the fit never
    assumes a circular arc. The angular rate is found by a scan, and ``(vx, vy)`` follow in
    closed form for each candidate rate.
    """

    count = len(times)
    if count < 4:
        return None
    best: Optional[Dict[str, float]] = None
    for step in range(omega_steps):
        omega = -omega_range + 2.0 * omega_range * step / (omega_steps - 1)
        aa = ab = bb = ax = ay = bx = by = 0.0
        for t, (px, py), _theta in zip(times, positions, angles):
            a, b = _basis(omega, t)
            aa += a[0] * a[0] + a[1] * a[1]
            ab += a[0] * b[0] + a[1] * b[1]
            bb += b[0] * b[0] + b[1] * b[1]
            ax += a[0] * px + a[1] * py
            bx += b[0] * px + b[1] * py
        det = aa * bb - ab * ab
        if abs(det) < 1e-12:
            continue
        vx = (ax * bb - bx * ab) / det
        vy = (aa * bx - ab * ax) / det
        position_error = 0.0
        angle_error = 0.0
        for t, (px, py), theta in zip(times, positions, angles):
            a, b = _basis(omega, t)
            mx = a[0] * vx + b[0] * vy
            my = a[1] * vx + b[1] * vy
            position_error += (mx - px) ** 2 + (my - py) ** 2
            angle_error += wrap_angle(omega * t - theta) ** 2
        position_error = math.sqrt(position_error / count)
        angle_error = math.sqrt(angle_error / count)
        score = position_error + 0.10 * angle_error
        if best is None or score < best["score"]:
            best = {"vx": vx, "vy": vy, "omega": omega, "score": score,
                    "position_rmse": position_error, "angle_rmse": angle_error}
    return best


def fit_and_extrapolate(
    times: Sequence[float],
    positions: Sequence[Tuple[float, float]],
    angles: Sequence[float],
) -> Optional[Dict[str, float]]:
    """Fit on the first half and extrapolate over the second half."""

    count = len(times)
    if count < 8:
        return None
    half = count // 2
    fit = fit_constant_twist(times[:half], positions[:half], angles[:half])
    if fit is None:
        return None
    vx, vy, omega = fit["vx"], fit["vy"], fit["omega"]
    position_error = 0.0
    angle_error = 0.0
    for t, (px, py), theta in zip(times[half:], positions[half:], angles[half:]):
        a, b = _basis(omega, t)
        mx = a[0] * vx + b[0] * vy
        my = a[1] * vx + b[1] * vy
        position_error += (mx - px) ** 2 + (my - py) ** 2
        angle_error += wrap_angle(omega * t - theta) ** 2
    n = count - half
    return {
        "vx": vx, "vy": vy, "omega": omega,
        "fit_position_rmse": fit["position_rmse"], "fit_angle_rmse": fit["angle_rmse"],
        "extrapolation_position_rmse": math.sqrt(position_error / n),
        "extrapolation_angle_rmse": math.sqrt(angle_error / n),
        "fit_samples": half, "extrapolation_samples": n,
    }


# ---------------------------------------------------------------------------
# layer 1 + 2
# ---------------------------------------------------------------------------


def execution_metrics(
    result,
    *,
    initial_object: Sequence[float],
    initial_robots: Sequence[Sequence[float]],
    force_limit: float,
    physics,
    option,
) -> Dict[str, object]:
    rows = result.rows
    if not rows:
        return {"empty": True}
    # the terminal state comes from the executor's own snapshot at the moment of exit, so a
    # sparse or truncated per-period log can never change it
    if getattr(result, "terminal_object", None) is not None:
        final_object = result.terminal_object
        final_robots = result.terminal_robots
    else:  # pragma: no cover - every arm now supplies a terminal snapshot
        final_object = rows[-1]["true"]["object"]
        final_robots = rows[-1]["true"]["robots"]

    # stage-resolved object motion, in the initial object frame
    stage_runs = _stage_runs(rows)
    stage_motion: Dict[str, Dict[str, float]] = {}
    for stage, first, last in stage_runs:
        before = rows[max(0, first - 1)]["true"]["object"] if first > 0 else initial_object
        after = rows[last]["true"]["object"]
        relative_before = to_initial_frame(before, initial_object)
        relative_after = to_initial_frame(after, initial_object)
        entry = stage_motion.setdefault(stage, {"linear": 0.0, "angle": 0.0, "periods": 0})
        entry["linear"] += math.hypot(relative_after[0] - relative_before[0],
                                      relative_after[1] - relative_before[1])
        entry["angle"] += abs(wrap_angle(relative_after[2] - relative_before[2]))
        entry["periods"] += last - first + 1

    relative_object = to_initial_frame(final_object, initial_object)
    displacement = math.hypot(relative_object[0], relative_object[1])
    rotation = abs(relative_object[2])
    final_speed = math.hypot(final_object[3], final_object[4])
    final_omega = abs(final_object[5])

    saturations = 0
    periods = 0
    contact_periods = 0
    force_magnitudes: List[float] = []
    for row in rows:
        forces = row.get("force", [[0.0, 0.0], [0.0, 0.0]])
        for force in forces:
            magnitude = math.hypot(force[0], force[1])
            force_magnitudes.append(magnitude)
            if magnitude >= 0.99 * force_limit:
                saturations += 1
            periods += 1
        observed = row.get("observed", {})
        if any(bool(observed.get(str(robot_id), {}).get("contact")) for robot_id in (0, 1)):
            contact_periods += 1
    if rows[0].get("stage") == "old_window":
        for row in rows:
            for force in row.get("force", []):
                magnitude = math.hypot(force[0], force[1])
                if magnitude >= 0.99 * force_limit:
                    saturations += 1

    # local settle verdict against the ground-truth final state
    settled_observed = result.exit_reason == "settled_observed"
    settled_true = final_speed < option.settle_linear and final_omega < option.settle_angular

    trajectory = [
        (row["time"], tuple(row["true"]["object"][:2]), row["true"]["object"][2])
        for row in rows if row.get("stage") in ("push", "Push", "PUSH")
    ]
    twist = None
    if trajectory and len(trajectory) >= 8:
        times = [t - trajectory[0][0] for t, _p, _a in trajectory]
        positions: List[Tuple[float, float]] = []
        angles = []
        for _t, point, angle in trajectory:
            relative = to_initial_frame((point[0], point[1], angle), initial_object)
            positions.append((relative[0], relative[1]))
            angles.append(relative[2])
        if (max(p[0] for p in positions) - min(p[0] for p in positions)) ** 2 + \
           (max(p[1] for p in positions) - min(p[1] for p in positions)) ** 2 > MIN_MOTION_FOR_TWIST ** 2 \
           or (max(angles) - min(angles)) > MIN_ROTATION_FOR_TWIST:
            twist = fit_and_extrapolate(times, positions, angles)

    robot_relative = [
        to_initial_frame(pose, initial_object) for pose in final_robots
    ]
    robot_start = [to_initial_frame(pose, initial_object) for pose in initial_robots]
    robot_travel = [
        math.hypot(after[0] - before[0], after[1] - before[1])
        for after, before in zip(robot_relative, robot_start)
    ]

    max_penetration = 0.0
    overshoot = 0
    push_periods = 0
    push_contact_periods = 0
    commanded = 0.0
    for row in rows:
        max_penetration = max(max_penetration, float(row.get("penetration", 0.0)))
        if str(row.get("stage")) == "push":
            if any(bool(row.get("observed", {}).get(str(robot_id), {}).get("contact"))
                   for robot_id in (0, 1)):
                push_contact_periods += 1
            push_periods += 1
            observed = row.get("observed", {})
            for robot_id in ("0", "1"):
                entry = observed.get(robot_id, {})
                if entry.get("stage") == "push" and entry.get("reference"):
                    commanded = max(commanded, math.hypot(*entry["reference"]))
            speed = math.hypot(row["true"]["robots"][0][3], row["true"]["robots"][0][4])
            if commanded > 0.0 and speed > 1.5 * commanded + 0.05:
                overshoot += 1

    return {
        "empty": False,
        "exit_reason": result.exit_reason,
        "max_penetration": max_penetration,
        "speed_overshoot_ratio": overshoot / max(1, push_periods),
        "push_periods": push_periods,
        "push_contact_ratio": push_contact_periods / max(1, push_periods),
        "displacement_per_active_second": (
            displacement / max(1e-6, max((float(row.get("active_contact", {}).get("0", 0.0))
                                          for row in rows), default=0.0))
        ),
        "aborted_by": result.aborted_by,
        "wall_clock_steps": result.wall_clock_steps,
        "wall_clock_seconds": result.wall_clock_steps * physics.control_dt,
        "stage_seconds": dict(result.stage_seconds),
        "active_contact_time": max(
            float(row.get("active_contact", {}).get("0", 0.0)) for row in rows
        ) if rows and "active_contact" in rows[0] else None,
        "push_wall_time": max(
            float(row.get("push_wall", {}).get("0", 0.0)) for row in rows
        ) if rows and "push_wall" in rows[0] else None,
        "reached_effective_push": _reached_effective_push(rows),
        "contact_period_ratio": contact_periods / max(1, len(rows)),
        "force_saturation_ratio": saturations / max(1, periods),
        "max_force": max(force_magnitudes) if force_magnitudes else 0.0,
        "displacement": displacement,
        "rotation": rotation,
        "stage_motion": stage_motion,
        "final_speed": final_speed,
        "final_omega": final_omega,
        "final_kinetic": 0.5 * physics.object_mass * final_speed ** 2,
        "settled_observed": settled_observed,
        "settled_true": settled_true,
        "false_stable": bool(settled_observed and not settled_true),
        "relative_object": list(relative_object),
        "relative_robots": [list(row) for row in robot_relative],
        "robot_travel": robot_travel,
        "micro_motion": bool(displacement < MICRO_LINEAR and rotation < MICRO_ANGULAR),
        "flags": list(result.flags),
        "push_completed": bool(getattr(result, "push_completed", False)),
        "settled_after_abort": bool(getattr(result, "settled_after_abort", False)),
        "effective_push_seconds": getattr(result, "effective_push_seconds", None),
        # the evaluator-side terminal kinematics of *both* robots, not only the object
        "terminal_robot_speeds": [
            math.hypot(float(row[3]), float(row[4])) for row in final_robots
        ],
        "terminal_robot_omegas": [abs(float(row[5])) for row in final_robots],
        "approach_seconds": float(result.stage_seconds.get("approach", 0.0))
        + float(result.stage_seconds.get("ready", 0.0)),
        "push_interval": getattr(result, "push_intervals", None),
        "delay_stats": getattr(result, "delay_stats", None),
        "twist": twist,
    }


def _reached_effective_push(rows: Sequence[Dict[str, object]]) -> bool:
    if not rows or "active_contact" not in rows[0]:
        return False
    return max(float(row["active_contact"].get("0", 0.0)) for row in rows) >= PUSH_SECONDS - 1e-6


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------


def quantile(values: Sequence[float], q: float) -> Optional[float]:
    values = sorted(value for value in values if value is not None)
    if not values:
        return None
    if len(values) == 1:
        return float(values[0])
    position = q * (len(values) - 1)
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return float(values[low])
    return float(values[low] + (values[high] - values[low]) * (position - low))


def summarise(values: Sequence[float]) -> Dict[str, Optional[float]]:
    values = [value for value in values if value is not None]
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None}
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "median": quantile(values, 0.5),
        "p90": quantile(values, 0.90),
        "p95": quantile(values, 0.95),
    }


def cluster_bootstrap(
    per_root: Dict[str, List[float]], *, samples: int, seed: int
) -> Dict[str, Optional[float]]:
    """Bootstrap the mean over **roots**, so the eight variants are not treated as independent."""

    import random

    roots = sorted(per_root)
    if not roots:
        return {"mean": None, "low": None, "high": None, "roots": 0}
    rng = random.Random(seed)
    means = []
    root_means = [sum(per_root[root]) / len(per_root[root]) for root in roots if per_root[root]]
    if not root_means:
        return {"mean": None, "low": None, "high": None, "roots": 0}
    for _ in range(samples):
        draw = [root_means[rng.randrange(len(root_means))] for _ in range(len(root_means))]
        means.append(sum(draw) / len(draw))
    means.sort()
    return {
        "mean": sum(root_means) / len(root_means),
        "low": means[int(0.025 * len(means))],
        "high": means[min(len(means) - 1, int(0.975 * len(means)))],
        "roots": len(root_means),
    }
