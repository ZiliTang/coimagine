"""Phase C physics acceptance: the checks with physical meaning, plus short traces.

    python experiments/continuous2d/run_physics_acceptance.py --out outputs/migration_v1/<run>

Writes ``physics_acceptance.json`` and the trajectories/figures it produces under the run
directory. Every check is run on the real backend; nothing here is asserted from theory.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import sys
import time
from pathlib import Path

from coimagine.continuous2d import Hold, MoveTo, PushContact, default_config, default_scenes
from coimagine.continuous2d.backend import snapshot_equivalence
from coimagine.continuous2d.environment import (
    TERMINATION_NUMERIC,
    TERMINATION_OUT_OF_BOUNDS,
    TERMINATION_PENETRATION,
    Continuous2DEnv,
)
from coimagine.continuous2d.mpc import plan, run_mpc_episode
from coimagine.continuous2d.render import render_episode
from coimagine.continuous2d.skills import JointSpec

ROOT = Path(__file__).resolve().parents[2]


def _state(env: Continuous2DEnv):
    return (
        env.backend.object_state().as_tuple(),
        tuple(state.as_tuple() for state in env.backend.robot_states()),
    )


def _flatten(value) -> tuple:
    """Flatten nested snapshots to their numeric leaves (skip termination strings/None)."""

    if isinstance(value, (tuple, list)):
        out = []
        for item in value:
            out.extend(_flatten(item))
        return tuple(out)
    if isinstance(value, bool):
        return (float(value),)
    if isinstance(value, (int, float)):
        return (float(value),)
    return ()


def check_scene_validity(config) -> dict:
    results = {}
    env = Continuous2DEnv(config)
    for scene in default_scenes():
        env.reset(scene.name)
        results[scene.name] = {
            "initial_contacts": len(env.backend.contacts()),
            "initial_penetration": env.backend.max_penetration(),
            "initial_penetration_by_pair": env.backend.penetration_by_pair(),
            "gap_width": scene.gap_width,
        }
    return results


def check_static_object(config) -> dict:
    """No robot contact: a resting object must not move because a control was issued."""

    env = Continuous2DEnv(config)
    env.reset("S0_dev")
    before = env.backend.object_state().as_tuple()
    for _ in range(40):
        env.step_control(((0.0, 8.0), (0.0, 8.0)))
    after = env.backend.object_state().as_tuple()
    return {
        "before": before,
        "after": after,
        "max_displacement": max(abs(a - b) for a, b in zip(before, after)),
    }


def check_centred_vs_offcentre(config) -> dict:
    """Centre and off-centre pushes must have distinguishable translation/rotation responses."""

    env = Continuous2DEnv(config)
    outcomes = {}
    # Push the west end face east; the only difference is how far the contact point sits
    # from the object's long axis, which is exactly the lever arm that adds rotation.
    for label, contact_y in (("centred", 0.0), ("offcentre", 0.10)):
        env.reset("S0_dev")
        start = env.backend.object_state()
        stage = (start.x - 0.5 - config.physics.robot_radius - 0.02, start.y + contact_y)
        env.execute_joint_skill((MoveTo(stage), Hold()))
        before = env.backend.object_state()
        spec = PushContact(0, (-0.5, contact_y), (1.0, 0.0), config.physics.max_force)
        result = env.execute_joint_skill((spec, Hold()))
        final = result.final.object
        outcomes[label] = {
            "delta_x": final.x - before.x,
            "delta_y": final.y - before.y,
            "delta_angle": final.angle - before.angle,
            "termination": result.termination,
        }
    return outcomes


def check_force_limit_and_no_teleport(config) -> dict:
    """Commanded forces stay under the cap and a robot cannot pass through the object."""

    env = Continuous2DEnv(config)
    env.reset("S0_dev")
    object_cells_before = env.backend.object_state().as_tuple()
    # Drive robot 0 straight at the object's centre with the largest legal force.
    target = (env.backend.object_state().x, env.backend.object_state().y)
    result = env.execute_joint_skill((MoveTo(target), Hold()))
    applied = [math.hypot(*step.applied_control[0]) for step in result.steps]
    robot_final = result.final.robots[0]
    object_state = result.final.object
    # The robot's centre must stay outside the object's footprint.
    distance_to_box = _distance_to_rotated_box(
        robot_final.x, robot_final.y, object_state, config.physics.object_width, config.physics.object_height
    )
    return {
        "max_applied_force": max(applied) if applied else 0.0,
        "force_limit": config.physics.max_force,
        "robot_final": robot_final.as_tuple(),
        "object_moved": max(abs(a - b) for a, b in zip(object_cells_before, object_state.as_tuple())),
        "robot_box_distance": distance_to_box,
        "penetration": result.diagnostics["final_penetration"],
    }


def _distance_to_rotated_box(x, y, object_state, width, height) -> float:
    cos, sin = math.cos(-object_state.angle), math.sin(-object_state.angle)
    dx, dy = x - object_state.x, y - object_state.y
    local_x = dx * cos - dy * sin
    local_y = dx * sin + dy * cos
    clamped_x = max(-width / 2, min(width / 2, local_x))
    clamped_y = max(-height / 2, min(height / 2, local_y))
    return math.hypot(local_x - clamped_x, local_y - clamped_y)


def check_push_is_not_failure(config) -> dict:
    """Contact pushes end in window_complete/success, not in a numerical failure."""

    env = Continuous2DEnv(config)
    env.reset("S0_dev")
    spec = PushContact(0, (-0.5, 0.0), (1.0, 0.0), config.physics.max_force)
    result = env.execute_joint_skill((spec, Hold()))
    return {
        "termination": result.termination,
        "is_numeric_failure": result.termination
        in (TERMINATION_NUMERIC, TERMINATION_OUT_OF_BOUNDS, TERMINATION_PENETRATION),
    }


def check_penetration(config) -> dict:
    env = Continuous2DEnv(config)
    env.reset("S0_dev")
    worst = 0.0
    spec = PushContact(0, (-0.5, 0.0), (1.0, 0.0), config.physics.max_force)
    result = env.execute_joint_skill((spec, Hold()))
    for step in result.steps:
        worst = max(worst, step.penetration)
    return {
        "max_penetration": worst,
        "penetration_by_pair": env.backend.penetration_by_pair(),
        "penetration_limit": config.physics.penetration_limit,
        "object_width": config.physics.object_width,
        "note": "tolerance is recorded against the object size, not required to be zero",
    }


def check_repeatability(config) -> dict:
    """Same initial state and same control prefix must reproduce within tolerance."""

    prefix = [
        ((6.0, 1.0), (0.0, 0.0)),
        ((6.0, 1.0), (0.0, 0.0)),
        ((3.0, -1.0), (2.0, 0.0)),
        ((0.0, 0.0), (0.0, 0.0)),
    ]
    runs = []
    for _ in range(2):
        env = Continuous2DEnv(config)
        env.reset("S0_dev")
        for control in prefix:
            env.step_control(control)
        runs.append(_state(env))
    difference = max(
        abs(a - b) for a, b in zip(_flatten(runs[0]), _flatten(runs[1]))
    )
    return {"max_difference": difference, "state": runs[0][0]}


def check_snapshot_equivalence(config) -> dict:
    env = Continuous2DEnv(config)
    env.reset("S0_dev")
    for _ in range(8):
        env.step_control(((5.0, 0.0), (0.0, 0.0)))
    snapshot = env.export_state()
    equivalence = snapshot_equivalence(env.backend, snapshot, ((5.0, 0.0), (0.0, 0.0)))
    # development-only approximate clone vs exact prefix replay, four more control steps
    env.restore(snapshot, exact=False)
    for _ in range(4):
        env.step_control(((5.0, 0.0), (0.0, 0.0)))
    approximate_future = _state(env)
    env.restore(snapshot, exact=True)
    for _ in range(4):
        env.step_control(((5.0, 0.0), (0.0, 0.0)))
    exact_future = _state(env)
    future_difference = max(
        abs(a - b) for a, b in zip(_flatten(approximate_future), _flatten(exact_future))
    )
    return {
        "one_step_approximate_vs_exact": equivalence,
        "four_step_approximate_vs_exact_max_difference": future_difference,
        "exact_restore_deviation": env.last_restore_deviation,
    }


def check_success_hold_restore(config, seconds: float = 0.40, extra: float = 0.15) -> dict:
    """S1.4-1: in-goal hold survives export/restore across a window boundary."""

    from coimagine.continuous2d.scenes import Pose, S0_DEVELOPMENT

    goal = S0_DEVELOPMENT.goal
    custom = dataclasses.replace(
        S0_DEVELOPMENT,
        name="S0_hold_probe",
        object_pose=Pose(goal.cx, goal.cy, 0.0),
    )
    steps = int(round(seconds / config.physics.control_dt))
    total = int(round((seconds + extra + 0.2) / config.physics.control_dt))
    original = Continuous2DEnv(config)
    original.reset(custom)
    for _ in range(steps):
        original.step_control(((0.0, 0.0), (0.0, 0.0)))
    snapshot = original.export_state()
    restored = Continuous2DEnv(config)
    restored.restore(snapshot, exact=True)

    counters = {"original": [], "restored": []}
    success_tick = {"original": None, "restored": None}
    for tick in range(total):
        for label, env in (("original", original), ("restored", restored)):
            env.step_control(((0.0, 0.0), (0.0, 0.0)))
            counters[label].append(round(env.success_counter, 9))
            if success_tick[label] is None and env.task_success():
                success_tick[label] = env.tick
    same_counter = counters["original"] == counters["restored"]
    return {
        "hold_before_snapshot": snapshot.success_counter,
        "same_counter_sequence": same_counter,
        "same_success_tick": success_tick["original"] == success_tick["restored"],
        "success_tick": success_tick,
        "counter_sequence": counters["original"],
    }


def check_contact_prefix_restore(config) -> dict:
    """S1.4-2: exact restore of a contact-heavy prefix, including near-wall contact."""

    outcomes = {}
    for label, scene, spec in (
        ("offcentre_push", "S0_dev", PushContact(0, (-0.5, 0.12), (1.0, 0.0), config.physics.max_force)),
        ("object_near_wall", "S1_dev", PushContact(0, (0.28, 0.12), (0.0, -1.0), config.physics.max_force)),
    ):
        original = Continuous2DEnv(config)
        # prefix inside one window boundary: execute windows, then compare continued windows
        original.reset(scene)
        prefix_windows = 2
        for _ in range(prefix_windows):
            if original.termination is None:
                original.execute_joint_skill((spec, Hold()))
        snapshot = original.export_state()
        restored = Continuous2DEnv(config)
        restored.restore(snapshot, exact=True)
        original_results, restored_results = [], []
        for _ in range(2):
            if original.termination is None:
                original_results.append(original.execute_joint_skill((spec, Hold())))
            if restored.termination is None:
                restored_results.append(restored.execute_joint_skill((spec, Hold())))
        max_difference = 0.0
        for first, second in zip(original_results, restored_results):
            max_difference = max(
                max_difference,
                max(
                    abs(a - b)
                    for a, b in zip(_flatten(first.final.as_key()), _flatten(second.final.as_key()))
                ),
            )
        outcomes[label] = {
            "scene": scene,
            "prefix_windows": prefix_windows,
            "prefix_length": len(snapshot.prefix),
            "restore_deviation": restored.last_restore_deviation,
            "continued_max_difference": max_difference,
            "contacts_at_snapshot": [
                str(entry["pair"]) for entry in original.backend.contacts()
            ],
        }
    return outcomes


def check_live_planner_isolation(config) -> dict:
    """S1.4-3: planning never changes the live world or its subsequent response."""

    live = Continuous2DEnv(config)
    planner = Continuous2DEnv(config)
    live.reset("S1_dev")
    before = live.export_state()
    scratch = Continuous2DEnv(config)
    first = scratch.rollout(before, [(PushContact(0, (0.28, 0.12), (0.0, -1.0), config.physics.max_force), Hold())])
    scratch.rollout(before, [(PushContact(0, (-0.5, 0.0), (1.0, 0.0), config.physics.max_force), Hold())])
    after = live.export_state()
    untouched = before.as_key() == after.as_key() and live.termination is None

    # branch pollution: the same branch twice gives identical results
    again = scratch.rollout(before, [(PushContact(0, (0.28, 0.12), (0.0, -1.0), config.physics.max_force), Hold())])
    branch_equal = first.final.as_key() == again.final.as_key()

    # the live world continues normally after planning
    response = live.execute_joint_skill((Hold(), Hold()))
    return {
        "live_unchanged_after_planning": untouched,
        "branches_independent": branch_equal,
        "live_after_planning_termination": response.termination,
        "live_tick": live.tick,
    }


def check_wall_penetration_coverage(config) -> dict:
    """S3.2: robot/object versus static wall and boundary contacts are actually observed."""

    env = Continuous2DEnv(config)
    env.reset("S1_dev")
    spec = PushContact(0, (0.28, 0.12), (0.0, -1.0), config.physics.max_force)
    pairs_seen = set()
    for _ in range(6):
        if env.termination is not None:
            break
        result = env.execute_joint_skill((spec, Hold()))
        for step in result.steps:
            for point in step.contacts:
                pairs_seen.add(str(point[0]))
        pairs_seen.update(env.backend.penetration_by_pair().keys())
    # A direct wall probe: drive the object into the passage wall.
    direct = Continuous2DEnv(config)
    direct.reset("S1_dev")
    wall_spec = PushContact(0, (0.5, 0.0), (1.0, 0.0), config.physics.max_force)
    for _ in range(4):
        if direct.termination is not None:
            break
        direct.execute_joint_skill((wall_spec, Hold()))
    wall_contacts = [entry for entry in direct.backend.contacts() if str(entry["pair"]).endswith("wall")]
    return {
        "pairs_seen_in_s1_push": sorted(pairs_seen),
        "wall_contacts_detected": len(wall_contacts),
        "penetration_by_pair": direct.backend.penetration_by_pair(),
        "penetration_limit": config.physics.penetration_limit,
    }



def check_no_cross_contamination(config) -> dict:
    """Two candidate simulations must not pollute each other or the source state."""

    env = Continuous2DEnv(config)
    scratch = Continuous2DEnv(config)
    env.reset("S1_dev")
    snapshot = env.export_state()
    first = scratch.rollout(snapshot, [(PushContact(0, (0.28, 0.12), (0.0, -1.0), 15.0), Hold())])
    scratch.rollout(snapshot, [(PushContact(0, (-0.5, 0.0), (1.0, 0.0), 15.0), Hold())])
    again = scratch.rollout(snapshot, [(PushContact(0, (0.28, 0.12), (0.0, -1.0), 15.0), Hold())])
    source_after = env.export_state()
    difference = max(
        abs(a - b) for a, b in zip(_flatten(first.final.as_key()), _flatten(again.final.as_key()))
    )
    return {
        "rollout_max_difference": difference,
        "source_unchanged": source_after.as_key() == snapshot.as_key(),
    }


def check_success_needs_hold(config) -> dict:
    """Success requires in-goal plus low speed for the configured time, not one frame."""

    env = Continuous2DEnv(config)
    env.reset("S0_dev")
    # drive the object into the goal, then verify a moving in-goal object is not success
    spec = PushContact(0, (-0.5, 0.0), (1.0, 0.0), config.physics.max_force)
    result = env.execute_joint_skill((spec, Hold()))
    in_goal_at_end = env.object_in_goal()
    slow_at_end = env.object_slow()
    return {
        "in_goal_at_end": in_goal_at_end,
        "slow_at_end": slow_at_end,
        "success_counter": env.success_counter,
        "success_hold_time": config.physics.success_hold_time,
        "termination": result.termination,
    }


def witness(config, out_dir: Path) -> dict:
    """S1 feasibility witness: replay the accepted plan's executed joint specs open-loop.

    The sequence is a fixed list of (contact point, face normal, hold) taken from the
    accepted depth-3 plan and replayed with no re-planning. It shows the physical
    "adjust orientation, then pass" sequence is realizable. A hand-designed single-robot
    two-phase rule was tried separately and did **not** complete; that is recorded in the
    report rather than hidden.
    """

    env = Continuous2DEnv(config)
    episode = run_mpc_episode(env, "S1_dev", config, depth=3, beam=3, seed=0)
    specs = [
        (getattr(entry, "results", None) or (entry,))[0].joint_spec
        for entry in episode.trace
    ]
    env.reset("S1_dev")
    replay_success = False
    replay_windows = 0
    for spec in specs:
        result = env.execute_joint_skill(spec)
        replay_windows += 1
        if result.success:
            replay_success = True
            break
    paths = render_episode(
        episode, "S1_dev", config, str(out_dir / "witness"), stem="s1_prepare_then_pass", every=4
    )
    return {
        "planner_success": episode.success,
        "planner_windows": episode.windows,
        "replayed_specs": [spec[0].label + " || " + spec[1].label for spec in specs],
        "replay_success": replay_success,
        "replay_windows": replay_windows,
        "figures": paths,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="outputs/migration_v1/physics_acceptance")
    args = parser.parse_args()
    config = default_config()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    report = {
        "config_hash": config.config_hash(),
        "scene_validity": check_scene_validity(config),
        "static_object": check_static_object(config),
        "centred_vs_offcentre": check_centred_vs_offcentre(config),
        "force_limit_and_no_teleport": check_force_limit_and_no_teleport(config),
        "push_is_not_failure": check_push_is_not_failure(config),
        "penetration": check_penetration(config),
        "repeatability": check_repeatability(config),
        "snapshot_equivalence": check_snapshot_equivalence(config),
        "no_cross_contamination": check_no_cross_contamination(config),
        "success_needs_hold": check_success_needs_hold(config),
        "s1_hold_restore": check_success_hold_restore(config),
        "s1_contact_prefix_restore": check_contact_prefix_restore(config),
        "s1_live_planner_isolation": check_live_planner_isolation(config),
        "wall_penetration_coverage": check_wall_penetration_coverage(config),
        "s1_witness": witness(config, out_dir),
        "wall_clock": None,
    }
    report["wall_clock"] = time.time() - started

    checks = {
        "scenes_start_without_robot_object_or_robot_robot_contact": all(
            entry["initial_penetration_by_pair"]["robot_object"] == 0.0
            and entry["initial_penetration_by_pair"]["robot_robot"] == 0.0
            for entry in report["scene_validity"].values()
        ),
        "initial_wall_penetration_recorded": all(
            "robot_wall" in entry["initial_penetration_by_pair"]
            for entry in report["scene_validity"].values()
        ),
        "static_object_does_not_self_move": report["static_object"]["max_displacement"] < 1e-9,
        "offcentre_push_rotates": abs(
            report["centred_vs_offcentre"]["offcentre"]["delta_angle"]
        ) > abs(report["centred_vs_offcentre"]["centred"]["delta_angle"]) + 0.05,
        "force_respects_limit": report["force_limit_and_no_teleport"]["max_applied_force"]
        <= config.physics.max_force + 1e-9,
        "robot_never_inside_object": report["force_limit_and_no_teleport"]["robot_box_distance"] > 0.0,
        "push_not_counted_as_failure": not report["push_is_not_failure"]["is_numeric_failure"],
        "repeatable": report["repeatability"]["max_difference"] < 1e-9,
        "exact_restore_bit_identical": all(
            value == 0.0 for value in report["snapshot_equivalence"]["exact_restore_deviation"].values()
        )
        if report["snapshot_equivalence"]["exact_restore_deviation"]
        else False,
        "success_hold_survives_restore": report["s1_hold_restore"]["same_counter_sequence"]
        and report["s1_hold_restore"]["same_success_tick"],
        "contact_prefix_restore_exact": all(
            entry["continued_max_difference"] == 0.0
            for entry in report["s1_contact_prefix_restore"].values()
        ),
        "live_planner_isolated": report["s1_live_planner_isolation"][
            "live_unchanged_after_planning"
        ]
        and report["s1_live_planner_isolation"]["branches_independent"],
        "wall_contacts_observed": report["wall_penetration_coverage"]["wall_contacts_detected"] > 0,
        "candidates_do_not_cross_contaminate": report["no_cross_contamination"][
            "rollout_max_difference"
        ] < 1e-9
        and report["no_cross_contamination"]["source_unchanged"],
        "s1_witness_replays": report["s1_witness"]["replay_success"],
    }
    report["checks"] = checks
    (out_dir / "physics_acceptance.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(checks, indent=2))
    for name, value in report["scene_validity"].items():
        print(f"  {name}: contacts={value['initial_contacts']} pen={value['initial_penetration']:.4f} pair={ {k: round(v,4) for k, v in value['initial_penetration_by_pair'].items() if v>0} } gap={value['gap_width']}")
    print("S1 witness:", report["s1_witness"]["planner_success"], report["s1_witness"]["replay_success"], report["s1_witness"]["replay_windows"])
    print(f"wall_clock={report['wall_clock']:.1f}s output={out_dir}")
    all_ok = all(checks.values())
    print("PHYSICS_ACCEPTANCE_OK" if all_ok else "PHYSICS_ACCEPTANCE_FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
