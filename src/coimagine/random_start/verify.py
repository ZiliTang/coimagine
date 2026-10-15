"""Section 7.2 semantics checks, run on the real environment.

Each check returns ``{"ok": bool, "detail": ...}``. The checks are deliberately adversarial:

* the information-boundary check changes geometry the robot cannot see (an object beyond the
  1.20 m range) and requires the legal observation *and* the command to be unchanged, with a
  positive control that a visible change does move them;
* the range check re-derives every ray by brute-force sampling instead of analytic geometry;
* the one-world check counts physics advances, not loop iterations;
* the reward check recomputes the return from the raw log with a separate implementation.
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..continuous2d.scenes import Box, Pose, SceneSpec
from . import env as E
from . import geometry as G
from . import layout, lidar
from .lidar import CATEGORY_NONE, CATEGORY_OBJECT, CATEGORY_STATIC, CATEGORY_TEAMMATE
from .skills import make_skill
from .spec import (
    ARENA,
    CONTROL_DT,
    LIDAR_RANGE,
    OBJECT_HEIGHT,
    OBJECT_WIDTH,
    ROBOT_ACTIONS,
    ROBOT_RADIUS,
)

Vec = Tuple[float, float]


def _synthetic(name: str, object_centre: Vec, object_angle: float,
               robots: Sequence[Pose], goal: Vec, obstacles=()) -> layout.InstanceSpec:
    return layout.InstanceSpec(
        name=name, map="L0", split="test", index=0, seed=0, arena=ARENA,
        obstacles=tuple(obstacles), object_pose=Pose(object_centre[0], object_centre[1],
                                                     object_angle),
        robot_poses=(robots[0], robots[1]), goal_centre=goal)


def check_information_boundary() -> Dict[str, object]:
    """Invisible geometry must not reach the controller; visible geometry must."""

    robots = (Pose(0.40, 1.10, 0.0), Pose(0.40, 0.60, 0.0))
    # the far pair is invisible to *both* robots (beyond the 1.20 m range); the near pair is
    # visible to both and differs only in the hidden object orientation
    far_a = _synthetic("far_a", (2.90, 1.10), 0.0, robots, (1.60, 1.10))
    far_b = _synthetic("far_b", (2.70, 0.90), 1.10, robots, (1.60, 1.10))
    near_a = _synthetic("near_a", (1.40, 1.10), 0.0, robots, (2.90, 2.00))
    near_b = _synthetic("near_b", (1.40, 1.10), 0.35, robots, (2.90, 2.00))

    def fingerprint(instance) -> Dict[str, object]:
        """Robot 0's own legal measurement of the world plus the command it produces."""

        environment = E.RandomStartEnv(instance)
        observation = environment.observe(0, 0, 0.0)
        force = None
        for action in ROBOT_ACTIONS:
            skill = make_skill(action)
            force, info = skill.step(observation, CONTROL_DT)
            if action.kind == "push":
                break
        patch = observation.patch
        return {"distances": tuple(observation.scan.distances),
                "categories": tuple(observation.scan.category),
                "patch": None if patch is None else (round(patch.distance, 9),
                                                     tuple(round(value, 9)
                                                           for value in patch.normal_world)),
                "force": force,
                "object_returns": int(sum(1 for index, valid in enumerate(observation.scan.valid)
                                          if valid and observation.scan.category[index]
                                          == CATEGORY_OBJECT))}

    def identical(first, second) -> bool:
        return bool(first["distances"] == second["distances"]
                    and first["categories"] == second["categories"]
                    and first["patch"] == second["patch"]
                    and np.allclose(first["force"], second["force"]))

    first, second = fingerprint(far_a), fingerprint(far_b)
    third, fourth = fingerprint(near_a), fingerprint(near_b)
    return {"ok": bool(identical(first, second) and not identical(third, fourth)),
            "invisible_object_returns": [first["object_returns"], second["object_returns"]],
            "invisible_measurement_identical": identical(first, second),
            "invisible_command_identical": bool(np.allclose(first["force"], second["force"])),
            "positive_control_visible_change_moves_measurement": not identical(third, fourth),
            "visible_object_returns": [third["object_returns"], fourth["object_returns"]]}


def check_lidar_against_brute_force(instances: Sequence[layout.InstanceSpec],
                                    stride: float = 0.001) -> Dict[str, object]:
    """Every ray is re-derived by marching along it; analytic and sampled hits must agree."""

    def first_hit(origin: Vec, direction: Vec, arena, obstacles, object_poly, robots,
                  robot_id: int, limit: float):
        steps = int(limit / stride)
        for step in range(steps + 1):
            distance = step * stride
            point = (origin[0] + distance * direction[0], origin[1] + distance * direction[1])
            # boundary: the wall surface is the arena rectangle inset by the capsule radius
            if not (0.05 <= point[0] <= arena[0] - 0.05 and 0.05 <= point[1] <= arena[1] - 0.05):
                return distance, CATEGORY_STATIC
            for polygon in obstacles:
                if G.point_in_convex(point, polygon):
                    return distance, CATEGORY_STATIC
            if G.point_in_convex(point, object_poly):
                return distance, CATEGORY_OBJECT
            for other_id, robot in enumerate(robots):
                if other_id == robot_id:
                    continue
                if math.dist(point, (robot[0], robot[1])) <= ROBOT_RADIUS:
                    return distance, CATEGORY_TEAMMATE
        return None

    worst_distance = 0.0
    mismatched_classes = 0
    checked = 0
    for instance in instances[:4]:
        states = [(pose.x, pose.y, pose.angle, 0.0, 0.0, 0.0) for pose in instance.robot_poses]
        obstacles = tuple(G.box_polygon(box.cx, box.cy, box.angle, box.half_w, box.half_h)
                          for box in instance.obstacles)
        object_poly = G.box_polygon(instance.object_pose.x, instance.object_pose.y,
                                    instance.object_pose.angle, OBJECT_WIDTH / 2.0,
                                    OBJECT_HEIGHT / 2.0)
        for robot_id in (0, 1):
            scan = lidar.scan_lidar(instance.arena, obstacles,
                                    (instance.object_pose.x, instance.object_pose.y,
                                     instance.object_pose.angle), states, robot_id, 0, 0.0)
            for index in range(len(scan.valid)):
                checked += 1
                direction = (math.cos(scan.ray_angles[index]), math.sin(scan.ray_angles[index]))
                expected = first_hit(scan.origin, direction, instance.arena, obstacles,
                                     object_poly, states, robot_id, LIDAR_RANGE)
                if expected is None:
                    if scan.valid[index]:
                        # the analytic hit must not be a phantom: only allow the last stride
                        worst_distance = max(worst_distance,
                                             abs(scan.distances[index] - LIDAR_RANGE))
                    continue
                if not scan.valid[index]:
                    mismatched_classes += 1
                    continue
                worst_distance = max(worst_distance, abs(scan.distances[index] - expected[0]))
                if scan.category[index] != expected[1]:
                    mismatched_classes += 1
    return {"ok": bool(worst_distance <= 3 * stride and mismatched_classes == 0),
            "rays_checked": checked, "worst_distance_error_m": worst_distance,
            "stride_m": stride, "class_mismatches": mismatched_classes}


def check_fit_is_from_returns(instances: Sequence[layout.InstanceSpec]) -> Dict[str, object]:
    """The fitted normal must equal the least-squares normal of the returned points."""

    worst_angle = 0.0
    worst_rms = 0.0
    samples = 0
    for instance in instances[:6]:
        states = [(pose.x, pose.y, pose.angle, 0.0, 0.0, 0.0) for pose in instance.robot_poses]
        for robot_id in (0, 1):
            scan = lidar.scan_lidar(instance.arena, (), (instance.object_pose.x,
                                                        instance.object_pose.y,
                                                        instance.object_pose.angle),
                                    states, robot_id, 0, 0.0)
            for patch in lidar.scan_patches(scan, (states[robot_id][0], states[robot_id][1])):
                if not patch.valid:
                    continue
                points = np.asarray(patch.points_robot)
                centred = points - points.mean(axis=0)
                u, s, vt = np.linalg.svd(centred, full_matrices=False)
                direction = vt[0]
                normal_robot = np.asarray([-direction[1], direction[0]])
                if normal_robot @ points.mean(axis=0) < 0:
                    normal_robot = -normal_robot
                cos, sin = math.cos(scan.heading), math.sin(scan.heading)
                normal_world = np.asarray([normal_robot[0] * cos - normal_robot[1] * sin,
                                           normal_robot[0] * sin + normal_robot[1] * cos])
                angle = math.degrees(math.acos(max(-1.0, min(1.0,
                    float(np.dot(normal_world, patch.normal_world))))))
                worst_angle = max(worst_angle, angle)
                residual = centred @ np.asarray([direction[0], direction[1]])
                perpendicular = centred @ normal_robot
                worst_rms = max(worst_rms, float(np.sqrt((perpendicular ** 2).mean())))
                samples += 1
    return {"ok": bool(worst_angle < 1e-6 and worst_rms <= 0.005 + 1e-12),
            "valid_patches": samples, "worst_normal_angle_deg_vs_svd": worst_angle,
            "worst_rms_m": worst_rms,
            "note": "the fitted normal is reproduced by an independent SVD of the same returns; "
                    "the true object orientation is never read by the fitter"}


def check_one_world_per_period(instance: layout.InstanceSpec) -> Dict[str, object]:
    environment = E.RandomStartEnv(instance)
    tick_before = environment.env.tick
    boundary = environment.joint_step(3, 9)
    return {"ok": bool(environment.world_advances == boundary["periods"]
                       and environment.env.tick - tick_before == boundary["periods"]),
            "periods": boundary["periods"], "world_advances": environment.world_advances,
            "env_tick_delta": environment.env.tick - tick_before,
            "note": "both robots' forces act on the same single advance of the same world"}


def check_tick_and_time_axis(instance: layout.InstanceSpec) -> Dict[str, object]:
    environment = E.RandomStartEnv(instance)
    boundary = environment.joint_step(1, 13)
    bad = []
    for row in environment.rows:
        if row["tick"] != row["period"]:
            bad.append(("tick_period", row["period"], row["tick"]))
        if abs(row["time"] - row["period"] * CONTROL_DT) > 1e-12:
            bad.append(("time_axis", row["period"], row["time"]))
    frames = len(environment.rows)
    return {"ok": bool(not bad and frames == boundary["periods"]),
            "frames": frames, "periods": boundary["periods"],
            "problems": bad[:5],
            "note": "n control periods produce exactly n stored frames in this log; the initial "
                    "observation O0 is the boundary observation the first skill read, and the "
                    "terminal frame is the last period's observation"}


def check_terminal_frame_shared(instance: layout.InstanceSpec) -> Dict[str, object]:
    """A skill's terminal frame is the next skill's first frame; nothing is duplicated."""

    environment = E.RandomStartEnv(instance)
    first = environment.joint_step(3, 13)
    boundary_vector = environment.boundary_vector()
    second = environment.joint_step(5, 13)
    entries = environment.history
    first_of_second = entries[int(first["periods"])]["observation"]
    return {"ok": bool(len(entries) == environment.total_periods
                       and np.array_equal(boundary_vector, first_of_second)),
            "history_entries": len(entries), "periods": int(environment.total_periods),
            "boundary_vector_matches_next_skill_first_frame": bool(
                np.array_equal(boundary_vector, first_of_second)),
            "note": "frames tile the episode: n periods store n frames, the skill boundary frame "
                    "appears exactly once, and a branch reuses the parent's last frame"}


def check_sample_control_order(instance: layout.InstanceSpec) -> Dict[str, object]:
    """Each period senses the state at the *start* of that period, then applies control."""

    environment = E.RandomStartEnv(instance)
    initial = [state.as_tuple() for state in environment.robot_states()]
    environment.joint_step(1, 13)
    rows = environment.rows
    problems = []
    previous_robots = initial
    for index, row in enumerate(rows):
        for robot_id in (0, 1):
            origin = row[f"r{robot_id}"]["scan"]["origin"]
            expected = previous_robots[robot_id][:2]
            if math.dist(origin, expected) > 1e-9:
                problems.append({"period": row["period"], "robot": robot_id,
                                 "scan_origin": origin, "state_before": list(expected)})
        previous_robots = row["true"]["robots"]
    return {"ok": bool(not problems), "periods": len(rows), "problems": problems[:3],
            "note": "the scan of period k is taken from the true state at the start of period k, "
                    "and the row stores the state after the control of that period"}


def check_cross_skill_continuity(instance: layout.InstanceSpec) -> Dict[str, object]:
    environment = E.RandomStartEnv(instance)
    first = environment.joint_step(3, 13)
    last_before = environment.rows[-1]
    second = environment.joint_step(3, 13)
    first_after = environment.rows[first["periods"]]
    gaps = []
    for robot_id in (0, 1):
        before = last_before["true"]["robots"][robot_id]
        after = first_after["true"]["robots"][robot_id]
        gap = math.dist(before[:2], after[:2])
        speed = math.hypot(before[3], before[4])
        gaps.append({"robot": robot_id, "position_step_m": gap,
                     "speed_before": speed, "allowed_m": speed * CONTROL_DT + 0.02})
    object_before = last_before["true"]["object"]
    object_after = first_after["true"]["object"]
    object_step = math.dist(object_before[:2], object_after[:2])
    return {"ok": bool(all(entry["position_step_m"] <= entry["allowed_m"] for entry in gaps)
                       and object_step <= 0.1),
            "robot_steps": gaps, "object_step_m": object_step,
            "note": "positions and velocities carry over; only the controllers' internal state "
                    "is rebuilt at a skill boundary"}


def check_hold_across_skills_and_wait(instance: layout.InstanceSpec) -> Dict[str, object]:
    """A hand-placed terminal state: the hold must accumulate inside WAIT_BRAKE."""

    goal = instance.goal_centre
    placed = replace(instance, name=instance.name + "_at_goal",
                     object_pose=Pose(goal[0], goal[1], instance.object_pose.angle))
    environment = E.RandomStartEnv(placed)
    boundary = environment.joint_step(12, 12)
    success_rows = [row for row in environment.rows if row["hold_seconds"] > 0]
    return {"ok": bool(environment.success and boundary["periods"] <= 12
                       and len(success_rows) >= 9),
            "success": bool(environment.success), "periods": boundary["periods"],
            "hold_rows": len(success_rows),
            "max_hold": max((row["hold_seconds"] for row in environment.rows), default=0.0),
            "note": "the low-speed hold is an environment-level timer; WAIT_BRAKE can complete "
                    "it because it never issues a push command"}


def check_reward_recomputation(instances: Sequence[layout.InstanceSpec]) -> Dict[str, object]:
    """Success, budget-exhausted and ordinary-skill returns, recomputed from the log."""

    outcomes = []
    for instance in instances:
        environment = E.RandomStartEnv(instance)
        for action_index in (3, 9, 12):
            if environment.termination is not None:
                break
            environment.joint_step(action_index, action_index)
        recomputed = E.recompute_episode_reward(environment.rows, environment.boundaries,
                                                (instance.object_pose.x, instance.object_pose.y))
        outcomes.append({
            "instance": instance.name,
            "termination": environment.termination,
            "environment_total": environment.reward_total,
            "recomputed_total": recomputed["total"],
            "difference": abs(environment.reward_total - recomputed["total"]),
            "success": environment.success,
            "budget": environment.termination == E.TERMINATION_BUDGET,
        })
    # a forced success episode and a forced budget episode
    goal = instances[0].goal_centre
    placed = replace(instances[0], name="forced_success",
                     object_pose=Pose(goal[0], goal[1], instances[0].object_pose.angle))
    forced = E.RandomStartEnv(placed)
    forced.joint_step(12, 12)
    forced_recomputed = E.recompute_episode_reward(forced.rows, forced.boundaries,
                                                   (goal[0], goal[1]))
    outcomes.append({"instance": "forced_success", "termination": forced.termination,
                     "environment_total": forced.reward_total,
                     "recomputed_total": forced_recomputed["total"],
                     "difference": abs(forced.reward_total - forced_recomputed["total"]),
                     "success": True, "budget": False})

    budgeted = E.RandomStartEnv(instances[0], E.EnvConfig(skill_budget=2, time_budget=1e9))
    for action_index in (12, 12):
        if budgeted.termination is None:
            budgeted.joint_step(action_index, action_index)
    budget_recomputed = E.recompute_episode_reward(
        budgeted.rows, budgeted.boundaries,
        (instances[0].object_pose.x, instances[0].object_pose.y))
    outcomes.append({"instance": "forced_budget", "termination": budgeted.termination,
                     "environment_total": budgeted.reward_total,
                     "recomputed_total": budget_recomputed["total"],
                     "difference": abs(budgeted.reward_total - budget_recomputed["total"]),
                     "success": budgeted.success, "budget": True})
    worst = max(outcome["difference"] for outcome in outcomes)
    return {"ok": bool(worst < 1e-9
                       and any(outcome["success"] for outcome in outcomes)
                       and any(outcome["budget"] for outcome in outcomes)),
            "worst_difference": worst, "episodes": outcomes}


def check_branch_restore(instance: layout.InstanceSpec) -> Dict[str, object]:
    environment = E.RandomStartEnv(instance)
    environment.joint_step(3, 13)
    # a *terminated* parent may not be extended, checked on a budget-1 environment
    finished = E.RandomStartEnv(instance, E.EnvConfig(skill_budget=1))
    finished.joint_step(3, 3)
    refused = False
    try:
        finished.joint_step(3, 3)
    except RuntimeError:
        refused = True
    state = environment.export_branch_state()
    total_before = environment.total_periods
    first_branch = environment.joint_step(5, 12)
    rows_after_first = [copy.deepcopy(row) for row in environment.rows]
    environment.restore_branch_state(state)
    second_branch = environment.joint_step(5, 12)
    rows_after_second = environment.rows
    same_periods = int(first_branch["periods"]) == int(second_branch["periods"])
    same = bool(len(rows_after_first) == len(rows_after_second)
                and all(rows_after_first[index]["true"] == rows_after_second[index]["true"]
                        for index in range(len(rows_after_first))))
    return {"ok": bool(same and same_periods
                       and environment.total_periods == total_before + first_branch["periods"]
                       and refused),
            "replay_identical": same, "same_periods": same_periods,
            "periods_restored": environment.total_periods,
            "branch_periods": int(first_branch["periods"]),
            "terminated_parent_refused": refused,
            "terminated_parent_termination": finished.termination,
            "note": "the parent's last frame is shared, not duplicated: the branch's first "
                    "period advances from it"}


def run_all(instances: Sequence[layout.InstanceSpec]) -> Dict[str, object]:
    checks = {
        "information_boundary": check_information_boundary(),
        "lidar_vs_brute_force": check_lidar_against_brute_force(instances),
        "fit_is_from_returns": check_fit_is_from_returns(instances),
        "one_world_per_period": check_one_world_per_period(instances[0]),
        "tick_and_time_axis": check_tick_and_time_axis(instances[0]),
        "terminal_frame_shared": check_terminal_frame_shared(instances[0]),
        "sample_control_order": check_sample_control_order(instances[0]),
        "cross_skill_continuity": check_cross_skill_continuity(instances[0]),
        "hold_across_skills_and_wait": check_hold_across_skills_and_wait(instances[0]),
        "reward_recomputation": check_reward_recomputation(instances[:3]),
        "branch_restore": check_branch_restore(instances[0]),
    }
    return {"checks": checks,
            "passed": int(sum(1 for check in checks.values() if check["ok"])),
            "total": len(checks),
            "all_ok": bool(all(check["ok"] for check in checks.values()))}
