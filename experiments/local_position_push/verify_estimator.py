"""Verify the surface-speed estimator against ground truth.

    python experiments/local_position_push/verify_estimator.py

The settle test compares the estimator against ``settle_surface`` (0.02 m/s), so the estimator
must be quiet when the surface is not moving *relative to the robot* and loud when it is.

Every control period records the matched point speeds (both aggregators are recomputed from the
same match set, so the comparison is paired) together with the physically correct reference: the
object's normal velocity relative to the robot, ``(v_object - v_robot) . n``.

The run is driven past its settle exit and then held with the protocol's brake law for a while,
so the genuinely static regime -- the one the settle test actually has to judge -- is covered.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.local_contact_option import option as option_module  # noqa: E402
from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.control import (  # noqa: E402
    LocalControlState,
    brake_force,
    world_points,
)
from coimagine.local_contact_option.option import (  # noqa: E402
    OptionExecutor,
    build_observation,
)
from coimagine.local_contact_option.roots import build_roots  # noqa: E402
from coimagine.local_contact_option.skill import LocalSurfacePush  # noqa: E402

CONFIG = default_config()
OPTION = OptionConfig(kv=25.0, a_ref_max=3.0)
MATCH_TOLERANCE = 0.05
THRESHOLD = OPTION.settle_surface
HOLD_PERIODS = 400
CONTROL_DT = CONFIG.physics.control_dt
BINS = ((0.0, 0.002), (0.002, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.05), (0.05, 1e9))


def matched_speeds(state, patch, pose, sample_time, now) -> Optional[List[float]]:
    """Diagnostic duplicate of the estimator's match set; never used by the controller."""

    if patch is None or sample_time is None:
        return None
    if (state.previous_world_points is None or state.previous_pose is None
            or state.previous_sample_time is None):
        return None
    dt = sample_time - state.previous_sample_time
    if dt <= 0.0:
        return None
    current = world_points(patch, pose)
    normal = patch.normal_world
    speeds: List[float] = []
    for point in current:
        best, best_distance = None, MATCH_TOLERANCE
        for candidate in state.previous_world_points:
            distance = math.hypot(point[0] - candidate[0], point[1] - candidate[1])
            if distance < best_distance:
                best_distance, best = distance, candidate
        if best is None:
            continue
        delta = (point[0] - best[0], point[1] - best[1])
        speeds.append((delta[0] * normal[0] + delta[1] * normal[1]) / dt)
    return speeds


def truth_normal_speed(env: Continuous2DEnv, patch) -> Optional[float]:
    if patch is None:
        return None
    object_state = env.backend.object_state()
    robot = env.backend.robot_states()[0]
    normal = patch.normal_world
    relative = (object_state.vx - robot.vx, object_state.vy - robot.vy)
    return relative[0] * normal[0] + relative[1] * normal[1]


def median_abs(values: Sequence[float]) -> float:
    ordered = sorted(abs(value) for value in values)
    return ordered[len(ordered) // 2]


def rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(value * value for value in values) / len(values))


def controlled_cases() -> List[dict]:
    cases = {
        "moving at 0.10 m/s, 5 clean matches": [0.10] * 5,
        "static, 5 clean matches": [0.0] * 5,
        "static, 1 of 5 points mis-associated (one ray spacing = 0.28 m/s)":
            [0.0, 0.0, 0.0, 0.0, 0.28],
        "static, 2 of 5 points mis-associated": [0.0, 0.0, 0.0, 0.28, 0.28],
        "symmetric -/+ 0.20 m/s (the case the RMS was introduced for)":
            [0.20, -0.20, 0.20, -0.20],
    }
    return [{"case": name, "median_abs": median_abs(values), "rms": rms(values),
             "median_says_rest": median_abs(values) < THRESHOLD,
             "rms_says_rest": rms(values) < THRESHOLD}
            for name, values in cases.items()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    roots = build_roots("dev", args.per_class)
    rows: List[dict] = []
    environment: List[Continuous2DEnv] = []
    original = option_module.update_surface_estimate

    def wrapper(state, patch, pose, sample_time, now, **kwargs):
        speeds = matched_speeds(state, patch, pose, sample_time, now)
        result = original(state, patch, pose, sample_time, now, **kwargs)
        if speeds and environment:
            rows.append({"speeds": speeds,
                         "truth_normal": truth_normal_speed(environment[0], patch)})
        return result

    option_module.update_surface_estimate = wrapper
    try:
        for root in roots:
            env = Continuous2DEnv(CONFIG)
            env.reset(root.scene())
            environment.append(env)
            specs = {robot_id: LocalSurfacePush(**root.local_parameters(robot_id))
                     for robot_id in root.active_robots}
            OptionExecutor(OPTION, "V").run(
                env, root_id=root.root_id, variant="verify", joint_specs=specs,
                active_robots=root.active_robots)
            state = LocalControlState()
            state.start(alpha=0.0, speed=0.0, force_limit=OPTION.force_limit, kv=OPTION.kv,
                        a_ref_max=OPTION.a_ref_max)
            for _ in range(HOLD_PERIODS):
                observation = build_observation(
                    env, OPTION, 0, stage="settle", teammate=None, skill={})
                original(state, observation.patch,
                         (observation.own.x, observation.own.y, observation.own.angle),
                         observation.sample_time, observation.time)
                force = brake_force(observation, state, CONTROL_DT)
                env.step_control((force, (0.0, 0.0)))
                if env.termination is not None:
                    break
    finally:
        option_module.update_surface_estimate = original

    buckets: Dict[str, List[Tuple[float, float]]] = {
        f"{low:.3f}-{high:.3f}": [] for low, high in BINS}
    for row in rows:
        truth = row["truth_normal"]
        if truth is None:
            continue
        values = row["speeds"]
        entry = (median_abs(values), rms(values))
        magnitude = abs(truth)
        for low, high in BINS:
            if low <= magnitude < high:
                buckets[f"{low:.3f}-{high:.3f}"].append(entry)
                break

    table = {}
    for key, values in buckets.items():
        if not values:
            table[key] = {"n": 0}
            continue
        table[key] = {
            "n": len(values),
            "median_reading_p50": _q(sorted(v[0] for v in values), 0.5),
            "rms_reading_p50": _q(sorted(v[1] for v in values), 0.5),
            "median_says_moving": sum(1 for v in values if v[0] >= THRESHOLD) / len(values),
            "rms_says_moving": sum(1 for v in values if v[1] >= THRESHOLD) / len(values),
        }
    quiet = [v for key in ("0.000-0.002", "0.002-0.005", "0.005-0.010", "0.010-0.020")
             for v in buckets[key]]
    moving = buckets["0.050-1000000000.000"]
    payload = {
        "threshold": THRESHOLD, "periods": len(rows),
        "bins_truth_normal_speed": table,
        "truth_below_threshold": {
            "n": len(quiet),
            "median_false_alarm": sum(1 for v in quiet if v[0] >= THRESHOLD) / max(1, len(quiet)),
            "rms_false_alarm": sum(1 for v in quiet if v[1] >= THRESHOLD) / max(1, len(quiet)),
        },
        "truth_above_005": {
            "n": len(moving),
            "median_detection": (sum(1 for v in moving if v[0] >= THRESHOLD) / len(moving))
            if moving else None,
            "rms_detection": (sum(1 for v in moving if v[1] >= THRESHOLD) / len(moving))
            if moving else None,
        },
        "controlled_cases": controlled_cases(),
        "note": "both aggregators are recomputed from the same match set, so the comparison is "
                "paired; the truth reference is (v_object - v_robot) . n",
    }
    out = Path(args.out) if args.out else (
        ROOT / "outputs/local_position_push_v1/local-position-push-v1/estimator_audit.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({key: value for key, value in payload.items() if key != "controlled_cases"},
                     indent=1, sort_keys=True))
    print("controlled cases (threshold 0.02 m/s):")
    for row in payload["controlled_cases"]:
        print(f"  {row['case']}\n     median={row['median_abs']:.4f} "
              f"says_rest={row['median_says_rest']}   rms={row['rms']:.4f} "
              f"says_rest={row['rms_says_rest']}")
    print("ESTIMATOR_AUDIT_OK")
    return 0


def _q(ordered: Sequence[float], fraction: float) -> float:
    if not ordered:
        return 0.0
    position = fraction * (len(ordered) - 1)
    low = int(math.floor(position))
    high = min(len(ordered) - 1, low + 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


if __name__ == "__main__":
    raise SystemExit(main())
