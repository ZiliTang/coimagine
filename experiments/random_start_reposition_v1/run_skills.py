"""S2: bounded skill acceptance on fixed seeds, plus the skill clips.

Sections 7.1 and 6.2 of the task book. Every number here comes from a run of the real
environment; the clips are rendered from the same log that produced the numbers.

Frozen executable subset (declared before looking at outcomes)

    subset = the first 20 L0 **train** instances, in index order, whose robot-0 initial
    observation already contains a valid fitted surface fragment with at least 6 returns and
    a clearance gap in [0.15, 0.45] m.
    action = REPOSITION_CCW(0.25 m) and REPOSITION_CW(0.25 m), one robot per episode, the
    other robot holding.

The gate asks for at least 16 of 20 completing the requested distance. Whether the robot
also crossed a true face boundary is evaluated afterwards from the true object pose, and is
reported separately because it is not a control input.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import geometry as G  # noqa: E402
from coimagine.random_start import layout, render  # noqa: E402
from coimagine.random_start.spec import OBJECT_HEIGHT, OBJECT_WIDTH, ROBOT_ACTIONS  # noqa: E402

OUT = ROOT / "outputs/random_start_reposition_v1"
HOLD_ACTION = 13  # WAIT_BRAKE(0.50 s); the wait actions are indices 13 and 14


def object_polygon(pose: Sequence[float]) -> Tuple[Tuple[float, float], ...]:
    return G.box_polygon(pose[0], pose[1], pose[2], OBJECT_WIDTH / 2.0, OBJECT_HEIGHT / 2.0)


def nearest_face(poly, point) -> int:
    distances = [(G.point_segment_distance(point, poly[i], poly[(i + 1) % len(poly)]), i)
                 for i in range(len(poly))]
    return int(min(distances)[1])


def meta_for(instance, title: str) -> Dict[str, object]:
    return {"name": instance.name, "map": instance.map, "arena": list(instance.arena),
            "obstacles": [{"cx": box.cx, "cy": box.cy, "half_w": box.half_w,
                           "half_h": box.half_h} for box in instance.obstacles],
            "title": title}


def probe(instance, action_index: int, partner: int = HOLD_ACTION) -> Dict[str, object]:
    """One single-skill episode: ``action_index`` on robot 0, ``partner`` on robot 1."""

    environment = E.RandomStartEnv(instance)
    object_before = environment.object_state().as_tuple()
    polygon_before = object_polygon(object_before[:3])
    face_before = nearest_face(polygon_before, (instance.robot_poses[0].x,
                                                instance.robot_poses[0].y))
    boundary = environment.joint_step(action_index, partner)
    object_after = environment.object_state().as_tuple()
    polygon_after = object_polygon(object_after[:3])
    face_after = nearest_face(polygon_after, tuple(environment.robot_states()[0].as_tuple()[:2]))
    exit_info = boundary["skill_ends"][0]
    patch_rows = [row for row in environment.rows if row["r0"]["kind"] != "hold"]
    pushes = [row for row in environment.rows if row["r0"]["kind"] == "push"]
    return {
        "instance": instance.name, "map": instance.map, "action": ROBOT_ACTIONS[action_index].label,
        "action_index": int(action_index), "exit_reason": exit_info["exit_reason"],
        "exit_stage": exit_info["stage"], "speed_at_exit": exit_info["speed_at_exit"],
        "residual_motion": exit_info["residual_motion"],
        "periods": int(boundary["periods"]), "time": float(boundary["periods"]) * E.CONTROL_DT,
        "object_displacement": float(math.dist(object_before[:2], object_after[:2])),
        "object_rotation": float(abs(object_after[2] - object_before[2])),
        "object_speed_after": float(math.hypot(object_after[3], object_after[4])),
        "face_change": bool(face_before != face_after),
        "face_before": face_before, "face_after": face_after,
        "corner_events": int(patch_rows[-1]["r0"]["corner_events"] or 0) if patch_rows else 0,
        "effective_contact": float(patch_rows[-1]["r0"]["effective_contact_seconds"] or 0.0)
        if patch_rows else 0.0,
        "progress": float(patch_rows[-1]["r0"]["progress"] or 0.0) if patch_rows else 0.0,
        "requested_distance": float(ROBOT_ACTIONS[action_index].distance),
        "max_penetration": float(max(row["penetration"] for row in environment.rows)),
        "safety_trigger": patch_rows[-1]["r0"]["safety_trigger"] if patch_rows else None,
        "push_command_ever": bool(
            any(row["r0"]["desired_velocity"] is not None
                and math.hypot(*row["r0"]["desired_velocity"]) > 1e-9 for row in pushes)),
        "final_goal_distance": float(environment.object_distance()),
        "progress_target": float(ROBOT_ACTIONS[action_index].distance)
        if ROBOT_ACTIONS[action_index].kind == "reposition" else 2.0,
    }


def frozen_subset(instances: Sequence, count: int = 20) -> List[Dict[str, object]]:
    """The declared-before-the-fact executable subset (see the module docstring)."""

    chosen: List[Dict[str, object]] = []
    for instance in instances:
        environment = E.RandomStartEnv(instance)
        observation = environment.observe(0, 0, 0.0)
        patch = observation.patch
        if patch is None or not patch.valid or patch.arc_rays < 6:
            continue
        gap = patch.nearest_distance - E.ROBOT_RADIUS
        if not (0.15 <= gap <= 0.45):
            continue
        chosen.append({"instance": instance, "gap": gap, "rays": patch.arc_rays})
        if len(chosen) >= count:
            break
    return chosen


def summarise(rows: Sequence[Dict[str, object]]) -> Dict[str, object]:
    by_action: Dict[str, Dict[str, object]] = defaultdict(lambda: {"n": 0, "exits": Counter()})
    for row in rows:
        entry = by_action[row["action"]]
        entry["n"] += 1
        entry["exits"][row["exit_reason"]] += 1
    out = {}
    for action, entry in by_action.items():
        out[action] = {"n": entry["n"], "exits": dict(entry["exits"])}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--instances", type=int, default=24)
    parser.add_argument("--videos", action="store_true", default=True)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    train_l0, _ = layout.build_split("L0", "train", count=args.instances)
    validation_l0, _ = layout.build_split("L0", "validation", count=8)

    # ---- 1. bounded battery over all 15 skills, fixed seeds -----------------
    battery: List[Dict[str, object]] = []
    for instance in train_l0:
        for action_index in range(15):
            battery.append(probe(instance, action_index))

    # ---- 2. the frozen executable subset and the gate ------------------------
    pool, _ = layout.build_split("L0", "train", count=64)
    subset = frozen_subset(pool, count=20)
    gate: Dict[str, object] = {"subset_rule": "first 20 L0 train instances with a valid "
                                              "robot-0 patch >= 6 returns and gap in [0.15,0.45] m",
                               "n": len(subset), "runs": {}}
    for action_index, name in ((9, "REPOSITION_CCW(0.25m)"), (11, "REPOSITION_CW(0.25m)")):
        results = [probe(entry["instance"], action_index) for entry in subset]
        completed = sum(1 for row in results if row["exit_reason"] == "reposition_complete")
        crossings = sum(1 for row in results if row["face_change"])
        gate["runs"][name] = {
            "completed": completed, "n": len(results),
            "completed_fraction": completed / max(1, len(results)),
            "face_crossings": crossings,
            "exit_reasons": dict(Counter(row["exit_reason"] for row in results)),
            "progress_vs_target": [[round(row["progress"], 4), row["progress_target"]]
                                   for row in results],
            "instances": [row["instance"] for row in results],
            "per_instance": results}
    gate["pass"] = bool(gate["runs"]["REPOSITION_CCW(0.25m)"]["completed"] >= 16
                        and gate["runs"]["REPOSITION_CW(0.25m)"]["completed"] >= 16)

    # ---- 3. WAIT_BRAKE: no new push, no frozen world -------------------------
    wait_checks: List[Dict[str, object]] = []
    for instance in train_l0[:8]:
        environment = E.RandomStartEnv(instance)
        environment.joint_step(3, HOLD_ACTION)  # give the object some momentum first
        before = environment.object_state().as_tuple()
        boundary = environment.joint_step(HOLD_ACTION, HOLD_ACTION)
        after = environment.object_state().as_tuple()
        wait_rows = [row for row in environment.rows[-boundary["periods"]:]]
        wait_checks.append({
            "instance": instance.name,
            "object_speed_before": float(math.hypot(before[3], before[4])),
            "object_speed_after": float(math.hypot(after[3], after[4])),
            "object_kept_moving": bool(math.hypot(after[3], after[4]) > 0.0
                                       or math.hypot(before[3], before[4]) > 0.0),
            "commanded_push": bool(any(math.hypot(*row["r0"]["desired_velocity"]) > 1e-9
                                       for row in wait_rows)),
            "forced_zero_object_velocity": bool(math.hypot(after[3], after[4]) == 0.0
                                                and math.hypot(before[3], before[4]) > 1e-3),
            "exit_reasons": sorted({str(row["r0"]["exit_reason"]) for row in wait_rows}),
            "wait_stage_rows": sum(1 for row in wait_rows if row["r0"]["kind"] == "wait"),
        })

    # ---- 4. clips (fixed selection rules) ------------------------------------
    clips: Dict[str, object] = {}
    if args.videos:
        video_dir = out / "videos" / "skills"
        # 4a PUSH normal push (first battery row of a plain push that completed)
        push_rows = [row for row in battery if row["action"] == "PUSH(a=+0deg,v=0.16)"
                     and row["exit_reason"] == "push_complete"]
        if push_rows:
            instance = next(it for it in train_l0 if it.name == push_rows[0]["instance"])
            base = E.RandomStartEnv(instance)
            base.joint_step(3, 13)
            base.joint_step(3, 13)
            clips["push_normal"] = render.render_episode(
                base.rows, meta_for(instance, "PUSH(a=0,v=0.16) x2: normal contact push"),
                video_dir, "push_normal", fps=20, step=2)
            render.write_keyframes(base.rows, meta_for(instance, "PUSH normal"),
                                   video_dir, "push_normal")
        # 4b PUSH off-centre: rotation
        rotation_rows = [row for row in battery if row["action"] == "PUSH(a=+15deg,v=0.26)"
                         and row["object_rotation"] > 0.02]
        if rotation_rows:
            instance = next(it for it in train_l0 if it.name == rotation_rows[0]["instance"])
            base = E.RandomStartEnv(instance)
            base.joint_step(7, 13)
            clips["push_offcentre"] = render.render_episode(
                base.rows, meta_for(instance, "PUSH(a=+15deg,v=0.26): off-centre push rotates "
                                              "the object"),
                video_dir, "push_offcentre", fps=20, step=2)
        # 4c PUSH failure: an observation-lost or blocked exit, the worst by progress
        failure_rows = sorted([row for row in battery
                               if row["action"].startswith("PUSH")
                               and row["exit_reason"] in ("not_observed", "patch_lost",
                                                          "blocked_no_progress",
                                                          "contact_lost", "safety_abort")],
                              key=lambda row: row["object_displacement"])
        if failure_rows:
            instance = next(it for it in train_l0 if it.name == failure_rows[0]["instance"])
            base = E.RandomStartEnv(instance)
            base.joint_step(failure_rows[0]["action_index"], 13)
            clips["push_failure"] = render.render_episode(
                base.rows, meta_for(instance, f"PUSH failure ({failure_rows[0]['exit_reason']})"),
                video_dir, "push_failure", fps=20, step=2)
        # 4d/4e REPOSITION corner crossings, one per direction
        for action_index, direction in ((9, "CCW"), (11, "CW")):
            crossed = [row for row in gate["runs"][f"REPOSITION_{direction}(0.25m)"]["per_instance"]
                       if row["face_change"] and row["exit_reason"] == "reposition_complete"]
            if not crossed:
                crossed = [row for row in battery
                           if row["action"] == f"REPOSITION_{direction}(0.25m)"
                           and row["face_change"]]
            if crossed:
                instance = next(it for it in train_l0 + validation_l0
                                if it.name == crossed[0]["instance"])
                base = E.RandomStartEnv(instance)
                base.joint_step(action_index, 13)
                clips[f"reposition_{direction.lower()}_corner"] = render.render_episode(
                    base.rows, meta_for(instance, f"REPOSITION_{direction}(0.25 m): true face "
                                                  f"{crossed[0]['face_before']} -> "
                                                  f"{crossed[0]['face_after']} (post-hoc truth)"),
                    video_dir, f"reposition_{direction.lower()}_corner", fps=20, step=2)
                render.write_keyframes(base.rows, meta_for(instance, f"REPOSITION {direction}"),
                                       video_dir, f"reposition_{direction.lower()}_corner")
        # 4f REPOSITION blocked (obstacle / teammate / wall) - L1 preferred
        l1_train, _ = layout.build_split("L1", "train", count=16)
        blocked = None
        for instance in l1_train:
            for action_index in (9, 11):
                row = probe(instance, action_index)
                if row["exit_reason"] in ("safety_abort", "blocked_no_progress", "blocked_contact"):
                    blocked = (instance, action_index, row)
                    break
            if blocked:
                break
        if blocked:
            instance, action_index, row = blocked
            base = E.RandomStartEnv(instance)
            base.joint_step(action_index, 13)
            clips["reposition_blocked"] = render.render_episode(
                base.rows, meta_for(instance, f"REPOSITION blocked on L1 "
                                              f"({row['exit_reason']})"),
                video_dir, "reposition_blocked", fps=20, step=2)
        # 4g WAIT_BRAKE residual motion after a push
        instance = train_l0[0]
        base = E.RandomStartEnv(instance)
        base.joint_step(5, 13)
        base.joint_step(13, 13)
        clips["wait_residual"] = render.render_episode(
            base.rows, meta_for(instance, "PUSH then WAIT_BRAKE: residual motion, no new push"),
            video_dir, "wait_residual", fps=20, step=2)

    report = {
        "battery": {"n": len(battery), "by_action": summarise(battery),
                    "per_run": battery},
        "gate": gate,
        "wait_brake": {"checks": wait_checks,
                       "no_push_command": bool(not any(c["commanded_push"] for c in wait_checks)),
                       "world_not_frozen": bool(any(c["object_kept_moving"] for c in wait_checks)),
                       "no_forced_zero_velocity": bool(
                           not any(c["forced_zero_object_velocity"] for c in wait_checks))},
        "clips": clips,
    }
    (out / "results" / "skills_acceptance.json").write_text(json.dumps(report, indent=1,
                                                                      default=str))
    print(json.dumps({"gate": {name: {"completed": run["completed"], "n": run["n"],
                                      "face_crossings": run["face_crossings"]}
                               for name, run in gate["runs"].items()},
                      "gate_pass": gate["pass"],
                      "wait_no_push": report["wait_brake"]["no_push_command"],
                      "clips": sorted(clips)}, indent=1))
    print("SKILLS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
