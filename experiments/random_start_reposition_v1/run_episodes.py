"""S3: continuous joint episodes with mixed skills, one shared clock, plus clips.

The high-level controller used here is a **scripted demo policy**, written only to produce a
continuous mixed-skill sequence and an event log. It is not the learned policy, it is not
claimed to be optimal, and it reads only the legal observation whitelist plus the public goal:

* if no valid surface fragment is visible, ask PUSH(0 deg, 0.16 m/s): its first stage is the
  bounded local seek;
* otherwise measure the unit direction from this robot to the public goal centre and compare it
  with the direction the object would be pushed in (the estimated inward surface normal,
  rotated by the push offset -15/0/+15 deg);
* if the best available push is within 30 deg of the goal direction, PUSH;
* otherwise RE POSITION clockwise or counter-clockwise, whichever rotates the contact normal
  towards the goal direction, 0.50 m when the misalignment is large and 0.25 m otherwise;
* if a measured surface return is already inside the goal circle, WAIT_BRAKE.

The random baseline is a uniform draw over the 225 joint actions and exists only to give the
learning round a reference.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import layout, render  # noqa: E402
from coimagine.random_start.spec import ROBOT_ACTIONS, ROBOT_RADIUS  # noqa: E402

OUT = ROOT / "outputs/random_start_reposition_v1"
PUSH_INDEX = {(alpha, speed): index for index, alpha, speed in
              ((action.index, action.alpha_deg, action.speed) for action in ROBOT_ACTIONS
               if action.kind == "push")}
REPOSITION_INDEX = {(action.direction, action.distance): action.index for action in ROBOT_ACTIONS
                    if action.kind == "reposition"}
WAIT_INDEX = {action.duration: action.index for action in ROBOT_ACTIONS if action.kind == "wait"}


def _unit(vector: Tuple[float, float]) -> Tuple[float, float]:
    length = math.hypot(vector[0], vector[1])
    return (0.0, 0.0) if length < 1e-9 else (vector[0] / length, vector[1] / length)


def _rotate(vector: Tuple[float, float], angle: float) -> Tuple[float, float]:
    cos, sin = math.cos(angle), math.sin(angle)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def scripted_choice(observation, goal_xy: Tuple[float, float]) -> Tuple[int, Dict[str, object]]:
    """One robot's choice, from its own observation and the public goal only."""

    patch = observation.patch
    if patch is None or not patch.valid:
        return PUSH_INDEX[(0.0, 0.16)], {"reason": "no valid surface fragment -> seek/approach"}
    nearest_goal = min(math.dist(point, goal_xy) for point in patch.points_world)
    if nearest_goal <= 0.15:
        return WAIT_INDEX[0.5], {"reason": f"measured surface point is {nearest_goal:.3f} m from "
                                          "the goal centre -> stop pressing and wait"}
    goal_direction = _unit((goal_xy[0] - observation.own.x, goal_xy[1] - observation.own.y))
    inward = patch.normal_world
    best = None
    for alpha in (-15.0, 0.0, 15.0):
        pushed = _rotate(inward, math.radians(alpha))
        angle = math.degrees(math.acos(max(-1.0, min(1.0,
            pushed[0] * goal_direction[0] + pushed[1] * goal_direction[1]))))
        if best is None or angle < best[0]:
            best = (angle, alpha, pushed)
    angle, alpha, pushed = best
    if angle <= 30.0:
        speed = 0.26 if angle <= 10.0 else 0.16
        return PUSH_INDEX[(alpha, speed)], {
            "reason": f"best push offset {alpha:+.0f} deg is {angle:.1f} deg off the goal "
                      f"direction", "alignment_deg": angle}
    cross = inward[0] * goal_direction[1] - inward[1] * goal_direction[0]
    direction = "CCW" if math.degrees(math.atan2(cross, inward[0] * goal_direction[0]
                                                 + inward[1] * goal_direction[1])) > 0 else "CW"
    distance = 0.50 if angle > 50.0 else 0.25
    return REPOSITION_INDEX[(direction, distance)], {
        "reason": f"normals are {angle:.1f} deg off; orbit {direction} by {distance:.2f} m",
        "alignment_deg": angle}


def scripted_joint(environment: E.RandomStartEnv) -> Tuple[int, int, Dict[str, object]]:
    now = environment.total_periods * E.CONTROL_DT
    observations = {robot_id: environment.observe(robot_id, environment.total_periods, now)
                    for robot_id in (0, 1)}
    choices = {}
    for robot_id in (0, 1):
        action, info = scripted_choice(observations[robot_id], environment.instance.goal_centre)
        choices[robot_id] = (action, info)
    # if both want to push in almost opposite directions, the worse-aligned one holds instead
    if (ROBOT_ACTIONS[choices[0][0]].kind == "push"
            and ROBOT_ACTIONS[choices[1][0]].kind == "push"):
        first, second = observations[0], observations[1]
        if first.patch is not None and second.patch is not None:
            dot = (first.patch.normal_world[0] * second.patch.normal_world[0]
                   + first.patch.normal_world[1] * second.patch.normal_world[1])
            if dot < -0.3:
                worse = 1 if choices[0][1].get("alignment_deg", 0) <= choices[1][1].get(
                    "alignment_deg", 0) else 0
                choices[worse] = (WAIT_INDEX[0.5],
                                  {"reason": "the two push directions oppose; this robot holds"})
    return choices[0][0], choices[1][0], {str(key): value[1] for key, value in choices.items()}


def run_episode(instance, policy: str, seed: int = 0,
                record: bool = True) -> Dict[str, object]:
    environment = E.RandomStartEnv(instance)
    rng = np.random.default_rng(seed)
    decisions: List[Dict[str, object]] = []
    while environment.termination is None:
        if policy == "scripted":
            first, second, info = scripted_joint(environment)
        elif policy == "random":
            first, second = int(rng.integers(0, 15)), int(rng.integers(0, 15))
            info = {"0": {"reason": "uniform random"}, "1": {"reason": "uniform random"}}
        elif policy == "reposition_only":
            first, second = int(rng.choice([9, 10, 11, 12])), int(rng.choice([9, 10, 11, 12]))
            info = {"0": {"reason": "reposition-only ablation"},
                    "1": {"reason": "reposition-only ablation"}}
        else:
            raise KeyError(policy)
        decisions.append({"skill_index": environment.skills_used + 1,
                          "action": [ROBOT_ACTIONS[first].label, ROBOT_ACTIONS[second].label],
                          "choice_reason": info})
        environment.joint_step(first, second)
    success = environment.success
    return {
        "instance": instance.name, "map": instance.map, "policy": policy, "seed": seed,
        "success": bool(success), "termination": environment.termination,
        "final_goal_distance": float(environment.object_distance()),
        "start_goal_distance": float(instance_start_distance(instance)),
        "skills_used": int(environment.skills_used),
        "periods": int(environment.total_periods),
        "time": float(environment.total_periods) * E.CONTROL_DT,
        "return": float(environment.reward_total),
        "reward_components": dict(environment.reward_components),
        "exit_reasons": dict(Counter(
            entry["exit_reason"] for boundary in environment.boundaries
            for entry in boundary["skill_ends"].values() if entry)),
        "decisions": decisions,
        "environment": environment if record else None,
    }


def instance_start_distance(instance) -> float:
    return math.dist((instance.object_pose.x, instance.object_pose.y), instance.goal_centre)


def clustered_rate(rows: Sequence[Dict[str, object]], bootstrap: int = 2000,
                   seed: int = 0) -> Dict[str, object]:
    """Mean success over instances, with a bootstrap CI clustered by instance."""

    by_instance: Dict[str, List[float]] = {}
    for row in rows:
        by_instance.setdefault(str(row["instance"]), []).append(1.0 if row["success"] else 0.0)
    values = np.asarray([float(np.mean(group)) for group in by_instance.values()])
    rng = np.random.default_rng(seed)
    draws = [float(np.mean(rng.choice(values, size=len(values), replace=True)))
             for _ in range(bootstrap)]
    return {"instances": int(values.size), "episodes": len(rows),
            "mean_success": float(values.mean()),
            "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--train-instances", type=int, default=16)
    parser.add_argument("--validation-instances", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    train, _ = layout.build_split("L0", "train", count=args.train_instances)
    validation, _ = layout.build_split("L0", "validation", count=args.validation_instances)

    runs: List[Dict[str, object]] = []
    for instance in train:
        runs.append(run_episode(instance, "scripted"))
    for instance in validation:
        runs.append(run_episode(instance, "scripted"))
    for instance in validation:
        for seed in range(args.seeds):
            runs.append(run_episode(instance, "random", seed=seed))
    for instance in validation[:4]:
        runs.append(run_episode(instance, "reposition_only", seed=0))

    stripped = [{key: value for key, value in row.items() if key != "environment"}
                for row in runs]
    summary = {}
    for policy in ("scripted", "random", "reposition_only"):
        for split, instances in (("train", train), ("validation", validation)):
            subset = [row for row in stripped if row["policy"] == policy
                      and row["instance"] in {instance.name for instance in instances}]
            if subset:
                summary[f"{policy}/{split}"] = clustered_rate(subset)
                summary[f"{policy}/{split}"]["median_final_distance"] = float(np.median(
                    [row["final_goal_distance"] for row in subset]))
                summary[f"{policy}/{split}"]["median_skills"] = float(np.median(
                    [row["skills_used"] for row in subset]))
    (out / "results" / "continuous_episodes.json").write_text(json.dumps(
        {"summary": summary, "runs": stripped}, indent=1))

    # ---- clips (fixed selection rules) --------------------------------------
    clips: Dict[str, object] = {}
    with_env = [row for row in runs if row["environment"] is not None]
    successes = [row for row in with_env if row["policy"] == "scripted" and row["success"]]
    failures = [row for row in with_env if row["policy"] == "scripted" and not row["success"]]
    video_dir = out / "videos" / "continuous"
    picks = []
    if successes:
        picks.append(("best_success", max(successes, key=lambda row: row["return"])))
    if len(successes) > 1:
        picks.append(("typical_success", sorted(successes, key=lambda row: row["skills_used"])[
            len(successes) // 2]))
    if failures:
        picks.append(("worst_failure", min(failures, key=lambda row: row["final_goal_distance"])))
    for label, row in picks:
        environment = row["environment"]
        instance = next(item for item in train + validation if item.name == row["instance"])
        title = (f"{label}: {row['policy']} policy, {row['skills_used']} joint skills, "
                 f"{row['time']:.1f} s, success={row['success']}, "
                 f"final distance {row['final_goal_distance']:.2f} m")
        meta = {"name": instance.name, "map": instance.map, "arena": list(instance.arena),
                "obstacles": [{"cx": box.cx, "cy": box.cy, "half_w": box.half_w,
                               "half_h": box.half_h} for box in instance.obstacles],
                "title": title}
        step = max(1, len(environment.rows) // 300)
        clips[label] = render.render_episode(environment.rows, meta, video_dir,
                                             f"{label}_{row['instance']}", fps=20, step=step)
        render.write_keyframes(environment.rows, meta, video_dir, f"{label}_{row['instance']}")
        (out / "logs" / f"{label}_{row['instance']}_events.json").write_text(json.dumps(
            {"instance": instance.as_dict(), "policy": row["policy"],
             "decisions": row["decisions"], "exit_reasons": row["exit_reasons"],
             "boundaries": environment.boundaries, "return": row["return"]}, indent=1,
            default=str))
    print(json.dumps({"summary": {key: {"mean_success": round(value["mean_success"], 3),
                                        "ci95": [round(value["ci95"][0], 3),
                                                 round(value["ci95"][1], 3)],
                                        "instances": value["instances"]}
                               for key, value in summary.items()},
                      "clips": sorted(clips)}, indent=1))
    print("EPISODES_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
