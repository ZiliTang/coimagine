"""Targeted regression tests for the protocol-fix round (S1–S3).

These pin the semantics the review found broken: the success-hold timer and termination
survive export/restore; the live world is isolated from planning; success is absorbed
rather than re-expanded; failures are terminal; costs are not double counted; and the
schema-v2 data contract keeps both robots and full-precision skills.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

from coimagine.continuous2d import Continuous2DEnv, Hold, MoveTo, PushContact, default_config
from coimagine.continuous2d import mpc as mpc_module
from coimagine.continuous2d.data import SCHEMA_VERSION, segments, window_records
from coimagine.continuous2d.mpc import (
    PLANNER_NO_FEASIBLE,
    PlanResult,
    _lookahead,
    plan,
    run_mpc_episode,
    run_random_episode,
)
from coimagine.continuous2d.scenes import S0_DEVELOPMENT, Pose
from coimagine.continuous2d.skills import (
    Hold as HoldSkill,
    joint_spec_from_dicts,
    joint_spec_to_dicts,
    skill_from_dict,
    skill_to_dict,
)

CONFIG = default_config()


def _in_goal_scene():
    goal = S0_DEVELOPMENT.goal
    return dataclasses.replace(
        S0_DEVELOPMENT, name="S0_hold_probe", object_pose=Pose(goal.cx, goal.cy, 0.0)
    )


# ---------------------------------------------------------------------------
# S1.4 targeted tests
# ---------------------------------------------------------------------------


def test_success_hold_survives_snapshot_restore() -> None:
    """Hold 0.40 s, snapshot, restore; original and restored succeed at the same step."""

    scene = _in_goal_scene()
    original = Continuous2DEnv(CONFIG)
    original.reset(scene)
    steps = int(round(0.40 / CONFIG.physics.control_dt))
    for _ in range(steps):
        original.step_control(((0.0, 0.0), (0.0, 0.0)))
    assert original.success_counter == pytest.approx(0.40)
    snapshot = original.export_state()
    assert snapshot.success_counter == pytest.approx(0.40)

    restored = Continuous2DEnv(CONFIG)
    restored.restore(snapshot, exact=True)
    assert restored.success_counter == pytest.approx(0.40)

    original_counters, restored_counters = [], []
    original_success_tick = restored_success_tick = None
    for _ in range(12):
        original.step_control(((0.0, 0.0), (0.0, 0.0)))
        restored.step_control(((0.0, 0.0), (0.0, 0.0)))
        original_counters.append(round(original.success_counter, 9))
        restored_counters.append(round(restored.success_counter, 9))
        if original_success_tick is None and original.task_success():
            original_success_tick = original.tick
        if restored_success_tick is None and restored.task_success():
            restored_success_tick = restored.tick
    assert original_counters == restored_counters
    assert original_success_tick == restored_success_tick is not None
    # The accumulated 0.40 s is not thrown away: success lands within a few steps of the
    # snapshot, not after a fresh full 0.50 s hold. (Float accumulation makes the exact
    # step 11 rather than 10.)
    assert steps < original_success_tick <= steps + 3


@pytest.mark.parametrize(
    "scene,spec",
    [
        ("S0_dev", PushContact(0, (-0.5, 0.12), (1.0, 0.0), 15.0)),
        ("S1_dev", PushContact(0, (0.28, 0.12), (0.0, -1.0), 15.0)),
    ],
)
def test_contact_prefix_exact_restore(scene, spec) -> None:
    """A contact-heavy prefix restores exactly, including near-wall contact."""

    original = Continuous2DEnv(CONFIG)
    original.reset(scene)
    for _ in range(2):
        if original.termination is None:
            original.execute_joint_skill((spec, HoldSkill()))
    snapshot = original.export_state()
    restored = Continuous2DEnv(CONFIG)
    restored.restore(snapshot, exact=True)
    assert restored.last_restore_deviation == {
        "position": 0.0,
        "angle": 0.0,
        "linear_velocity": 0.0,
        "angular_velocity": 0.0,
    }
    assert len(snapshot.prefix) > 0
    for _ in range(2):
        if original.termination is None:
            first = original.execute_joint_skill((spec, HoldSkill()))
            second = restored.execute_joint_skill((spec, HoldSkill()))
            assert first.final.as_key() == second.final.as_key()


def test_live_env_is_isolated_from_planning() -> None:
    live = Continuous2DEnv(CONFIG)
    planner = Continuous2DEnv(CONFIG)
    live.reset("S1_dev")
    before = live.export_state()
    episode = run_mpc_episode(
        live, "S1_dev", CONFIG, depth=1, beam=3, seed=0, planner_env=planner, horizon=1
    )
    # Planning ran on the planner only; the live env advanced by exactly one executed window.
    assert episode.windows == 1
    assert live.tick == len(episode.trace[0].steps)
    # A second, independent live environment starting from the same scene is unaffected.
    other = Continuous2DEnv(CONFIG)
    other.reset("S1_dev")
    assert other.export_state().as_key() == before.as_key()


def test_terminated_environment_does_not_advance() -> None:
    scene = _in_goal_scene()
    env = Continuous2DEnv(CONFIG)
    env.reset(scene)
    result = env.execute_joint_skill((HoldSkill(), HoldSkill()))
    assert result.success and env.termination == "success"
    tick = env.tick
    again = env.execute_joint_skill((MoveTo((0.5, 0.5)), HoldSkill()))
    assert again.duration == 0.0 and len(again.steps) == 0
    assert again.diagnostics["already_terminated"] is True
    assert env.tick == tick
    observation, diagnostics = env.step_control(((15.0, 0.0), (15.0, 0.0)))
    assert diagnostics["already_terminated"] is True
    assert env.tick == tick


# ---------------------------------------------------------------------------
# S2.4 targeted tests
# ---------------------------------------------------------------------------


def test_success_is_absorbed_at_depth_three() -> None:
    scene = _in_goal_scene()
    env = Continuous2DEnv(CONFIG)
    env.reset(scene)
    result = plan(env, env.export_state(), CONFIG, depth=3, beam=3)
    assert result.feasible
    # The first window already completes the task: deeper layers must not extend it.
    assert result.predicted_windows == 1
    assert result.predicted_success_time is not None
    assert result.predicted_success_time < CONFIG.physics.window_steps * CONFIG.physics.control_dt
    assert result.predicted_success_time == pytest.approx(0.55, abs=0.06)


class _FakeResult:
    def __init__(self, snapshot, *, success, termination, time_used, usage):
        self.final = snapshot
        self.success = success
        self.termination = termination
        self.cost_components = {"time": time_used, "force_usage": usage, "steps": 40.0}


def test_failed_node_is_not_expanded_and_cost_is_not_double_counted(monkeypatch) -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S0_dev")
    snapshot = env.export_state()

    def fake_evaluate(scratch, snap, specs, *, root_candidate_id, cumulative_time=0.0,
                      cumulative_usage=0.0, prefix_substeps=1):
        # Candidate 0 always fails; the others advance by exactly one window each.
        if root_candidate_id == 0 and cumulative_time == 0.0:
            return (
                _FakeResult(snap, success=False, termination="penetration_limit", time_used=2.0, usage=1.0),
                (0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 0.0),
                "penetration_limit", None, None, 0,
            )
        return (
            _FakeResult(snap, success=False, termination="window_complete", time_used=1.0, usage=0.5),
            (0.0, 0.0, 0.0, -1.0, 0.0, -(cumulative_usage + 0.5), -float(root_candidate_id)),
            None, None, None, 0,
        )

    monkeypatch.setattr(mpc_module, "evaluate", fake_evaluate)
    tree = _lookahead(env, snapshot, CONFIG, depth=2, beam=3)
    assert tree["failure_distribution"].get("penetration_limit", 0) >= 1
    # No surviving node has the failed root; it was not expanded into a success.
    assert all(not node.termination for node in tree["entries"])
    assert all(node.root_candidate_id != 0 or node.cumulative_time != 2.0 for node in tree["entries"])
    # Each surviving two-window node counted each window once: 2 x 1.0 s and 2 x 0.5 usage.
    assert all(node.cumulative_time == pytest.approx(2.0) for node in tree["entries"])
    assert all(node.cumulative_usage == pytest.approx(1.0) for node in tree["entries"])


def test_no_feasible_plan_ends_the_episode(monkeypatch) -> None:
    def infeasible_plan(scratch, snapshot, config, *, depth, beam=3):
        return PlanResult(
            first=None, sequence=(), key=None, feasible=False, termination=PLANNER_NO_FEASIBLE,
            nodes=1, simulated_steps=0, prefix_replay_steps=0, wall_clock=0.0,
            candidates_at_root=1, failure_distribution={"penetration_limit": 1},
        )

    monkeypatch.setattr(mpc_module, "plan", infeasible_plan)
    live = Continuous2DEnv(CONFIG)
    episode = run_mpc_episode(live, "S0_dev", CONFIG, depth=1, beam=3, seed=0)
    assert episode.no_feasible_plan
    assert episode.termination == PLANNER_NO_FEASIBLE
    assert episode.windows == 0
    assert episode.live_physics_steps == 0
    assert live.tick == 0  # the live world never executed a failed candidate


# ---------------------------------------------------------------------------
# S3 targeted tests
# ---------------------------------------------------------------------------


def test_schema_v2_records_both_robots_and_structured_skills() -> None:
    episode = run_random_episode(Continuous2DEnv(CONFIG), "S0_dev", CONFIG, seed=0, horizon=2)
    records = window_records("ep", "S0_dev", 0, episode, CONFIG)
    record = records[0]
    assert record["schema_version"] == SCHEMA_VERSION == 2
    assert len(record["applied_control"]) == 2 * len(record["steps"])
    assert {entry[1] for entry in record["applied_control"]} == {0, 1}
    for step in record["steps"]:
        assert len(step["applied_control"]) == 2
        assert step["state"]["robots"][0] != step["next_state"]["robots"][0]
    for payload in record["joint_skill_specs"]:
        assert "kind" in payload
    assert record["state"]["boundary"] == "before_window"
    assert record["next_state"]["boundary"] == "after_window"


def test_skill_round_trip_keeps_close_non_integer_parameters() -> None:
    first = PushContact(0, (0.1234567891, -0.9876543219), (0.7071067811865476, -0.7071067811865476), 14.999999)
    second = PushContact(0, (0.1234567892, -0.9876543219), (0.7071067811865476, -0.7071067811865476), 14.999999)
    assert skill_from_dict(skill_to_dict(first)) == first
    assert skill_from_dict(skill_to_dict(second)) == second
    assert skill_from_dict(skill_to_dict(first)) != skill_from_dict(skill_to_dict(second))
    move = MoveTo((0.10000001, 0.20000002))
    assert skill_from_dict(skill_to_dict(move)) == move
    joint = (first, HoldSkill())
    assert joint_spec_from_dicts(joint_spec_to_dicts(joint)) == joint


def test_window_replays_from_the_same_snapshot_with_structured_skills() -> None:
    episode = run_random_episode(Continuous2DEnv(CONFIG), "S0_dev", CONFIG, seed=0, horizon=2)
    record = window_records("ep", "S0_dev", 0, episode, CONFIG)[0]
    snapshot = episode.trace[0].initial
    spec = joint_spec_from_dicts(record["joint_skill_specs"])
    replay = Continuous2DEnv(CONFIG)
    replay.restore(snapshot, exact=True)
    result = replay.execute_joint_skill(spec)
    assert [state.as_tuple() for state in result.final.robots] == [
        tuple(row) for row in record["next_state"]["robots"]
    ]
    assert result.final.object.as_tuple() == tuple(record["next_state"]["object"])


def test_segments_stay_inside_one_episode_and_do_not_extend_the_terminal() -> None:
    # S0_v4 depth1 completes deterministically under the frozen closeout scene library.
    episode = run_mpc_episode(
        Continuous2DEnv(CONFIG), "S0_v4", CONFIG, depth=1, beam=3, seed=0
    )
    assert episode.success
    records = window_records("ep-a", "S0_v4", 0, episode, CONFIG)
    assert records[-1]["success"] == 1  # terminal is the last recorded window
    for earlier, later in zip(records, records[1:]):
        assert earlier["next_state"]["robots"] == later["state"]["robots"]
        assert earlier["next_state"]["object"] == later["state"]["object"]
        assert earlier["t"] + len(earlier["steps"]) == later["t"]
    if len(records) >= 2:
        clips = segments(records, 2)
        assert all(len(clip) == 2 for clip in clips)
    mixed = [dict(records[0], episode_id="a"), dict(records[-1], episode_id="b")]
    with pytest.raises(ValueError):
        segments(mixed, 1)


def test_wall_contacts_are_detected_and_bounded() -> None:
    env = Continuous2DEnv(CONFIG)
    env.reset("S1_dev")
    spec = PushContact(0, (0.5, 0.0), (1.0, 0.0), CONFIG.physics.max_force)
    seen_wall = False
    for _ in range(4):
        if env.termination is not None:
            break
        result = env.execute_joint_skill((spec, HoldSkill()))
        if any(str(point[0]).endswith("wall") for step in result.steps for point in step.contacts):
            seen_wall = True
    assert seen_wall
    pairs = env.backend.penetration_by_pair()
    assert set(pairs) == {"robot_object", "robot_robot", "robot_wall", "object_wall"}
    assert all(math.isfinite(value) for value in pairs.values())
