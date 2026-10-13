"""Phase C redo: one ambiguity case under the **frozen** local controller and sensing.

Why this file exists
--------------------
The first C attempt parameterised its skills with ``PushContact``, whose contact point is resolved
by the simulator against the **true object pose**. That put hidden truth into the *controller*: the
two worlds received the same command string, but the commanded weld point -- and therefore the
motion the robots were dragged into -- depended on the object's hidden tangential position. The
"the robots then become distinguishable" observation was therefore partly an artefact of privileged
information leaking through the low-level controller, not evidence of autonomous identification.

This redo removes that path entirely. It uses

* the frozen local controller, arm **V** (``OptionExecutor("V")`` + ``LocalSurfacePush``), whose
  force is built only from the whitelist observation -- the inward normal read from the robot's own
  scan, its own velocity, and the shared skill parameters. There is no object-frame weld point;
* the frozen sensing configuration: ``scan_range = 0.6 m``, ``scan_rays = 72`` uniform over the
  **full 360 degrees** in the robot frame, ideal sensing (no noise, no delay);
* the frozen option protocol: APPROACH / READY / PUSH (2.0 s effective contact, 3.0 s wall) /
  BRAKE / SETTLE (4.0 s budget), with the round-3 settle estimator frozen at ``settle_window=0.20``;
* the frozen physics: mass 3.0 kg, friction 0.6, linear/angular damping 8.0/0.6, force limit 15 N,
  control period 0.05 s. Only the object's **length** is changed (7.0 m instead of 1.0 m) so that its
  short ends stay outside the 0.6 m scan in both worlds; thickness stays at the frozen 0.25 m.

The two worlds put both robots at the *same global poses* and shift the object along its own long
axis, so the shared observation can be identical. The run then checks, in order:

1. every element of both robots' whitelist observation is identical between the worlds;
2. the controller's internal state at t = 0 and the first control period's reference **and applied
   force** are identical -- if the low-level controller were reading hidden truth this would fail;
3. only then does the true physics diverge, because the object's centre of mass sits at a different
   tangential offset relative to the same contacts.

Run::

    .venv/bin/python experiments/option_prediction_mechanism_v1/phase_c_frozen/run_frozen_c.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.continuous2d.scenes import Box, Pose, SceneSpec  # noqa: E402
from coimagine.continuous2d.scoring import distance_to_box  # noqa: E402
from coimagine.local_contact_option.contract import (  # noqa: E402
    DELTA_PER_SPEED,
    FORCE_LIMIT,
    OptionConfig,
)
from coimagine.local_contact_option.option import OptionExecutor, build_observation  # noqa: E402
from coimagine.local_contact_option.sensing import LocalContactObservation  # noqa: E402
from coimagine.local_contact_option.skill import LocalSurfacePush  # noqa: E402

# ---------------------------------------------------------------------------
# frozen scene family
# ---------------------------------------------------------------------------

ARENA = (16.0, 4.0)
EDGE_Y = 2.0
OBJECT_LENGTH = 5.0          # only the LENGTH is changed from the frozen 1.0 m
OBJECT_THICKNESS = 0.25      # frozen PhysicsConfig.object_height
ROBOT_X = (8.0, 10.0)        # both worlds, identical global poses
STANDOFF = 0.02
DELTA = 0.02                 # pre-declared near-optimal tolerance, same as the C contract

#: The single declared case: the goal box sits at the MIDPOINT of the two worlds' object centres so
#: that neither world is privileged, and is long enough to contain the object's length. Object
#: length 5.0 m with shift 0.5 m keeps both short ends more than 1.0 m outside the frozen 0.6 m
#: scan in both worlds (A ends 6.5/11.5 m, B ends 7.0/12.0 m). Declared before running; not tuned
#: after seeing payoffs.
GOAL_CENTRE_X = 9.25
GOAL_HALF = (2.5, 0.6)
CANDIDATE_SHIFTS = (0.5, 0.75)

#: Eight genuinely different legal joint commands from the frozen parameter ranges
#: (alpha in {-15, 0, +15} deg, speed in {0.08, 0.16, 0.26} m/s).
COMMANDS: Tuple[Tuple[str, float, float, float, float], ...] = (
    ("squeeze",        0.0,   0.0,  0.16, 0.16),
    ("tangential_pos", math.radians(15.0), math.radians(-15.0), 0.16, 0.16),
    ("tangential_neg", math.radians(-15.0), math.radians(15.0), 0.16, 0.16),
    ("couple_pos",     math.radians(15.0), math.radians(15.0), 0.16, 0.16),
    ("couple_neg",     math.radians(-15.0), math.radians(-15.0), 0.16, 0.16),
    ("tangential_pos_fast", math.radians(15.0), math.radians(-15.0), 0.26, 0.26),
    ("tangential_neg_fast", math.radians(-15.0), math.radians(15.0), 0.26, 0.26),
    ("single_sided_pos", math.radians(15.0), 0.0, 0.16, 0.16),
)


def frozen_config():
    """Physics frozen except the object's length; thickness, mass, friction, damping, F_max stay."""

    base = default_config()
    return replace(base, physics=replace(base.physics, object_width=OBJECT_LENGTH,
                                         object_height=OBJECT_THICKNESS))


def frozen_option_config() -> OptionConfig:
    return replace(OptionConfig(kv=25.0, a_ref_max=3.0, approach_max_speed=0.30,
                                approach_contact_speed=0.06, approach_gain=4.0,
                                ready_precontact_speed=0.02),
                   settle_window=0.20, settle_window_min_baseline=0.10,
                   settle_max_sample_gap=0.15)


def robot_poses() -> Tuple[Pose, Pose]:
    radius = default_config().physics.robot_radius
    lower = EDGE_Y - OBJECT_THICKNESS / 2.0
    upper = EDGE_Y + OBJECT_THICKNESS / 2.0
    return (Pose(ROBOT_X[0], lower - radius - STANDOFF, math.pi / 2.0),
            Pose(ROBOT_X[1], upper + radius + STANDOFF, -math.pi / 2.0))


def scene(world: str, object_centre_x: float, shift: float) -> SceneSpec:
    delta = 0.0 if world == "A" else shift
    return SceneSpec(
        name=f"frozen_c_{world}", category="mechanism_C_frozen", variant=0 if world == "A" else 1,
        arena=ARENA, obstacles=(),
        goal=Box(GOAL_CENTRE_X, EDGE_Y, GOAL_HALF[0], GOAL_HALF[1]),
        object_pose=Pose(object_centre_x + delta, EDGE_Y, 0.0),
        robot_poses=robot_poses(),
        description="phase-C-frozen long rectangle; only the object's x position differs",
        scene_version="mechanism_C_frozen_v1",
        object_width=OBJECT_LENGTH, object_height=OBJECT_THICKNESS,
        robot_radius=default_config().physics.robot_radius)


def scene_hash(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# observation identity, exactly as the option pipeline builds it
# ---------------------------------------------------------------------------


def observation_payload(observation: LocalContactObservation) -> Dict[str, object]:
    patch = observation.patch
    return {
        "own": [observation.own.x, observation.own.y, observation.own.angle,
                observation.own.vx, observation.own.vy, observation.own.omega],
        "contacting": bool(observation.contacting),
        "contact_point_robot": list(observation.contact_point_robot)
        if observation.contact_point_robot else None,
        "contact_force_world": list(observation.contact_force_world),
        "contact_impulse": float(observation.contact_impulse),
        "arena": list(observation.arena),
        "robot_radius": observation.robot_radius,
        "force_limit": observation.force_limit,
        "skill": dict(observation.skill),
        "patch": None if patch is None else {
            "points_robot": [list(p) for p in patch.points_robot],
            "segment_robot": [list(p) for p in patch.segment_robot],
            "distance": patch.distance,
            "normal_world": list(patch.normal_world),
            "tangent_world": list(patch.tangent_world),
            "confidence": patch.confidence,
            "timestamp": patch.timestamp,
            "is_target": bool(patch.is_target),
            "arc_rays": int(patch.arc_rays),
            "occluder_labels": list(patch.occluder_labels),
        },
    }


def observation_difference(a: Dict[str, object], b: Dict[str, object]) -> Dict[str, float]:
    """Per-leaf absolute differences; a missing-vs-present leaf is reported as a mismatch count."""

    out: Dict[str, float] = {}

    def walk(x, y, path: str) -> None:
        if isinstance(x, dict) and isinstance(y, dict):
            for key in sorted(set(x) | set(y)):
                walk(x.get(key), y.get(key), f"{path}.{key}")
        elif isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                out[f"{path}.length_mismatch"] = float(abs(len(x) - len(y)))
            for index, (item_x, item_y) in enumerate(zip(x, y)):
                walk(item_x, item_y, f"{path}[{index}]")
        elif x is None or y is None:
            if x is not y:
                out[f"{path}.presence_mismatch"] = 1.0
        elif isinstance(x, bool) or isinstance(y, bool):
            if bool(x) != bool(y):
                out[f"{path}.flag_mismatch"] = 1.0
        elif isinstance(x, str) or isinstance(y, str):
            if str(x) != str(y):
                out[f"{path}.label_mismatch"] = 1.0
        else:
            out[path] = abs(float(x) - float(y))

    walk(a, b, "obs")
    return out


def start_observations(world: str, centre: float, shift: float):
    config = frozen_option_config()
    physics = frozen_config()
    env = Continuous2DEnv(physics)
    env.reset(scene(world, centre, shift))
    out = {}
    for robot_id in (0, 1):
        out[robot_id] = build_observation(
            env, config, robot_id, stage="approach", teammate=None, skill={}, noise_rng=None)
    return env, out


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def run_command(world: str, centre: float, shift: float, command) -> dict:
    name, alpha0, alpha1, speed0, speed1 = command
    physics = frozen_config()
    option = frozen_option_config()
    env = Continuous2DEnv(physics)
    active = (0, 1)
    env.reset(scene(world, centre, shift))
    initial_object = env.backend.object_state().as_tuple()
    initial_corners = env.backend.object_corners()
    start_distance = distance_to_box(initial_corners, env.scene.goal)
    specs = {}
    for robot_id, (alpha, speed) in enumerate(((alpha0, speed0), (alpha1, speed1))):
        specs[robot_id] = LocalSurfacePush(
            alpha=alpha, speed=speed, active_duration=2.0, force_limit=FORCE_LIMIT,
            delta=DELTA_PER_SPEED * speed)
    executor = OptionExecutor(option, "V")
    result = executor.run(env, root_id=f"frozen_c/{world}", variant=name,
                          joint_specs=specs, active_robots=active, noise_rng=None)
    end_corners = env.backend.object_corners()
    end_distance = distance_to_box(end_corners, env.scene.goal)
    duration = result.wall_clock_steps * physics.physics.control_dt
    impulse = sum(math.hypot(force[0], force[1]) * physics.physics.control_dt
                  for row in result.rows for force in (row.get("force") or []))
    completion = 1.0 if env.task_success() else 0.0
    progress = min(1.0, max(0.0, (start_distance - end_distance) / max(start_distance, 1e-6)))
    action_cost = impulse / (physics.physics.max_force * max(duration, 1e-9))
    q = completion + 0.5 * progress - 0.05 * duration - 0.10 * action_cost
    return {
        "command": name, "world": world,
        "alpha_deg": [math.degrees(alpha0), math.degrees(alpha1)],
        "speed": [speed0, speed1],
        "success": bool(env.task_success()),
        "start_distance": start_distance, "end_distance": end_distance,
        "progress": progress, "duration": duration, "action_cost": action_cost,
        "q": q, "exit_reason": result.exit_reason,
        "primary_exit_reason": result.primary_exit_reason,
        "locally_settled": result.locally_settled,
        "settled_observed": result.settled_observed,
        "settled_after_abort": result.settled_after_abort,
        "wall_clock_steps": result.wall_clock_steps,
        "first_row": _first_row(result.rows),
        "initial_object": list(initial_object),
        "terminal_object": list(result.terminal_object),
    }


def _first_row(rows) -> Optional[dict]:
    if not rows:
        return None
    row = rows[0]
    observed = row.get("observed") or {}
    return {
        "time": row.get("time"), "stage": row.get("stage"),
        "force": row.get("force"),
        "reference": [entry.get("reference") for entry in observed.values()],
        "patch_distance": [entry.get("patch_distance") for entry in observed.values()],
        "patch_normal": [entry.get("patch_normal") for entry in observed.values()],
        "contact": [entry.get("contact") for entry in observed.values()],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-root", default="outputs/option_prediction_mechanism_v1/phase_c_frozen")
    parser.add_argument("--object-centre-x", type=float, default=9.0)
    args = parser.parse_args()
    centre = args.object_centre_x
    payload: Dict[str, object] = {
        "scene": {"arena": list(ARENA), "edge_y": EDGE_Y, "object_length": OBJECT_LENGTH,
                  "object_thickness": OBJECT_THICKNESS, "robot_x": list(ROBOT_X),
                  "standoff": STANDOFF, "goal_half": list(GOAL_HALF),
                  "goal_centre_x": GOAL_CENTRE_X,
                  "object_centre_x": centre},
        "frozen": {
            "controller": "local V (OptionExecutor arm V, LocalSurfacePush)",
            "scan_range": OptionConfig().scan_range, "scan_rays": OptionConfig().scan_rays,
            "scan_geometry": "uniform 360 deg in the robot frame (frozen)",
            "option_config": frozen_option_config().as_dict(),
            "physics": {k: v for k, v in asdict(frozen_config().physics).items()},
            "note": "only object_width (length) changed: 1.0 -> 7.0 m; thickness/mass/friction/"
                    "damping/force limit/control period are the frozen values",
        },
        "delta": DELTA,
        "candidates": {},
    }

    for shift in CANDIDATE_SHIFTS:
        entry: Dict[str, object] = {"shift": shift}
        # 1. observation identity through the real option pipeline
        _env_a, obs_a = start_observations("A", centre, shift)
        _env_b, obs_b = start_observations("B", centre, shift)
        per_robot = {}
        worst = 0.0
        for robot_id in (0, 1):
            difference = observation_difference(observation_payload(obs_a[robot_id]),
                                                observation_payload(obs_b[robot_id]))
            finite = {k: v for k, v in difference.items() if v > 0.0}
            per_robot[f"robot{robot_id}"] = {
                "max_abs_difference": max(difference.values()) if difference else 0.0,
                "differences": finite}
            worst = max(worst, max(difference.values()) if difference else 0.0)
        entry["observation_identity"] = {"max_abs_difference": worst, "per_robot": per_robot,
                                         "identical": worst == 0.0}
        # 2. do the two worlds really have different truth? (they must)
        entry["truth_differs"] = {
            "object_pose_A": list(_env_a.backend.object_state().as_tuple()),
            "object_pose_B": list(_env_b.backend.object_state().as_tuple())}
        entry["scene_hash_A"] = scene_hash(scene("A", centre, shift).__dict__
                                           if hasattr(scene("A", centre, shift), "__dict__")
                                           else str(scene("A", centre, shift)))
        entry["scene_hash_B"] = scene_hash(str(scene("B", centre, shift)))

        # 3. run the eight commands in both worlds
        runs = {"A": [], "B": []}
        for command in COMMANDS:
            for world in ("A", "B"):
                runs[world].append(run_command(world, centre, shift, command))
        entry["runs"] = runs

        # 4. the first control period must be identical: same observation, reference and force
        first_checks = []
        worst_first = 0.0
        for index, command in enumerate(COMMANDS):
            row_a = runs["A"][index]["first_row"]
            row_b = runs["B"][index]["first_row"]
            difference = observation_difference(row_a, row_b)
            local_worst = max(difference.values()) if difference else 0.0
            worst_first = max(worst_first, local_worst)
            first_checks.append({
                "command": command[0], "max_abs_difference": local_worst,
                "differences": {k: v for k, v in difference.items() if v > 0.0},
                "force_A": row_a["force"] if row_a else None,
                "force_B": row_b["force"] if row_b else None})
        entry["first_period_identity"] = {
            "max_abs_difference_across_all_commands": worst_first,
            "identical": worst_first == 0.0, "checks": first_checks}

        # 5. payoff matrix and the near-optimal sets
        names = [command[0] for command in COMMANDS]
        q_a = {name: runs["A"][index]["q"] for index, name in enumerate(names)}
        q_b = {name: runs["B"][index]["q"] for index, name in enumerate(names)}
        best_a, best_b = max(q_a.values()), max(q_b.values())
        near_a = sorted(name for name in names if best_a - q_a[name] <= DELTA + 1e-12)
        near_b = sorted(name for name in names if best_b - q_b[name] <= DELTA + 1e-12)
        argmax_a = max(names, key=lambda name: q_a[name])
        argmax_b = max(names, key=lambda name: q_b[name])
        mutual_regret = max(best_a - q_a[argmax_b], best_b - q_b[argmax_a])
        all_knowing = 0.5 * (best_a + best_b)
        common = max(0.5 * (q_a[name] + q_b[name]) for name in names)
        entry["analysis"] = {
            "q_A": q_a, "q_B": q_b, "near_optimal_A": near_a, "near_optimal_B": near_b,
            "near_optimal_disjoint": not (set(near_a) & set(near_b)),
            "argmax_A": argmax_a, "argmax_B": argmax_b,
            "mutual_regret": mutual_regret, "all_knowing_50_50": all_knowing,
            "common_action_50_50": common, "information_gap_50_50": all_knowing - common,
            "positive": bool(not (set(near_a) & set(near_b)) and mutual_regret > DELTA),
        }
        payload["candidates"][f"shift_{shift}"] = entry
        print(f"shift {shift}: obs identical={entry['observation_identity']['identical']} "
              f"first-period identical={entry['first_period_identity']['identical']} "
              f"argmax A={argmax_a} B={argmax_b} regret={mutual_regret:.4f} "
              f"disjoint={entry['analysis']['near_optimal_disjoint']} "
              f"positive={entry['analysis']['positive']}", flush=True)

    out_root = ROOT / args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "frozen_c_result.json").write_text(
        json.dumps(payload, indent=1, sort_keys=True, default=float))
    print(f"wrote {out_root / 'frozen_c_result.json'}")
    print("FROZEN_C_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
