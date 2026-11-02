"""Joint timing/dose contracts; no desired final object displacement is manufactured."""
from functools import partial
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.env import RandomStartEnv
from coimagine.random_start.effect_push import ACTIONS, VERSION, EffectPushConfig, EffectPushSkill, make_effect_skill
from experiments.option_testbench.effect_push_eval import make_case, instance_from_dict, joint_cases, run_one


def config(**kwargs):
    kwargs.setdefault("translation_completion", "normal_reference")  # frozen v4 protocol
    return EffectPushConfig(endpoint_tracking=True, fast_approach=True, **kwargs)


def make_world(case, factory=None):
    return RandomStartEnv(instance_from_dict(case["instance"]),
                          skill_factory=factory or partial(make_effect_skill, config=config()),
                          action_table=ACTIONS, action_version=VERSION)


@pytest.fixture(scope="module")
def mixed():
    case = make_case("mixed_protocol", cooperative=True)
    case["actions"] = [10, 14]
    world = make_world(case)
    boundary = world.joint_step(*case["actions"])
    return world, boundary


def test_scales_keep_independent_normal_reference_budgets(mixed):
    w, b = mixed
    assert b["execution"]["normal_reference_pair"]
    assert b["execution"]["normal_speed_references"] == pytest.approx([.024, .12])
    assert b["execution"]["normal_reference_budget"] == pytest.approx([.1008, .504])
    assert all(v["exit_reason"] == "normal_reference_complete" for v in b["skill_ends"].values())
    assert abs(b["skill_ends"][0]["time"] - b["skill_ends"][1]["time"]) < 1.
    # Explicitly a nominal reference contract, not two contradictory object goals.
    assert w.object_state().y - w.instance.object_pose.y != pytest.approx(.1, abs=.02)


def test_preparation_and_contact_gates_are_causal(mixed):
    w, b = mixed
    events = {e["event"]: e["time"] for e in b["execution"]["events"]}
    assert events["contact_released"] > events["staging_released"]
    for row in w.rows:
        if row["execution"]["phase"] != "execute":
            assert row["execution"]["normal_reference_budget"] == [0., 0.]
    # A side's budget increments only for its own observed contact during active execution.
    previous = [0., 0.]
    for row in w.rows:
        dose = row["execution"]["normal_reference_budget"]
        for i, rate in enumerate((.024, .12)):
            if dose[i] > previous[i] + 1e-9:
                assert row[f"r{i}"]["contacting_object"]
                assert dose[i] - previous[i] == pytest.approx(rate * spec.CONTROL_DT)
        previous = dose


def test_scale_does_not_throttle_tangential_preparation(mixed):
    w, _ = mixed
    row = w.rows[0]
    components = []
    for i in (0, 1):
        info = row[f"r{i}"]
        n = info["effect"]["face_normal"]
        v = info["desired_velocity"]
        components.append(abs(-n[1] * v[0] + n[0] * v[1]))
    assert components[0] > 0
    assert components[0] == pytest.approx(components[1])


def test_force_limits_and_finished_side_hold_are_recorded(mixed):
    w, b = mixed
    for row in w.rows:
        assert all(math.hypot(*force) <= spec.MAX_FORCE + 1e-9 for force in row["forces"])
    assert b["execution"]["post_exit"] == "hold_brake"
    assert any(row["execution"]["active"] == [True, False] for row in w.rows)


def test_failed_preparation_does_not_deadlock_peer():
    case = next(c for c in joint_cases() if c["name"] == "peer_direction_unavailable")
    w = make_world(case)
    b = w.joint_step(*case["actions"])
    assert b["skill_ends"][0]["exit_reason"] == "peer_prepare_failed"
    assert b["skill_ends"][1]["exit_reason"] == "direction_unavailable"
    assert b["end_time"] <= .15
    assert b["reward"]["time_cost"] < 0
    assert b["termination"] is None


def test_short_scale_waits_under_shared_bounded_deadline():
    class DelayedReady(EffectPushSkill):
        def step(self, obs, dt):
            force, info = super().step(obs, dt)
            if obs.robot_id == 1 and obs.time < 9. and not self.joint_active and not self.done:
                self.ready = False
                self.stage = info["stage"] = "approach"
                self._mark_progress()  # synthetic legitimate ongoing preparation for this timing test
            return force, info
    case = make_case("delayed_ready_unit", cooperative=True)
    case["actions"] = [10, 14]
    w = make_world(case, lambda a, f: DelayedReady(a, f, config()))
    b = w.joint_step(*case["actions"])
    assert b["execution"]["requested_time_budgets"][0] < 9.
    assert next(e["time"] for e in b["execution"]["events"] if e["event"] == "staging_released") >= 9.
    assert all(v["exit_reason"] == "normal_reference_complete" for v in b["skill_ends"].values())
    assert b["end_time"] <= b["execution"]["effective_window_seconds"]


def test_active_side_failure_does_not_cancel_peer_or_restart_finished_side():
    class InjectedExit(EffectPushSkill):
        active_since = None
        def step(self, obs, dt):
            force, info = super().step(obs, dt)
            if self.joint_active:
                if self.active_since is None:
                    self.active_since = obs.time
                if obs.robot_id == 0 and obs.time - self.active_since >= .3 and not self.done:
                    self._finish("injected_unit_exit", obs)
            return force, info
    case = make_case("active_exit_unit", cooperative=True)
    case["actions"] = [10, 14]
    w = make_world(case, lambda a, f: InjectedExit(a, f, config()))
    b = w.joint_step(*case["actions"])
    assert b["skill_ends"][0]["exit_reason"] == "injected_unit_exit"
    assert b["skill_ends"][1]["exit_reason"] == "normal_reference_complete"
    end = b["skill_ends"][0]["time"]
    assert all(not row["execution"]["active"][0] for row in w.rows if row["time"] > end + spec.CONTROL_DT)


def test_different_directions_keep_parameters_without_vector_goal():
    case = next(c for c in joint_cases() if c["name"] == "N_E_0.5_0.1")
    r = run_one(case, True, {"endpoint_tracking": True, "fast_approach": True,
                            "translation_completion": "normal_reference"}, "joint_execution")
    assert r["metric"]["fulfilled"] is None
    assert r["boundary"]["action_index"] == [14, 0]
    assert r["boundary"]["execution"]["normal_speed_references"] == pytest.approx([.12, .024])
    assert r["boundary"]["execution"]["normal_reference_pair"]


def test_rotation_angles_are_not_reinterpreted_as_metres():
    case = next(c for c in joint_cases() if c["name"] == "rotation_mixed_angles")
    w = make_world(case)
    b = w.joint_step(*case["actions"])
    assert not b["execution"]["normal_reference_pair"]
    assert b["execution"]["normal_reference_budget"] == [0., 0.]
    assert b["execution"]["normal_speed_references"] == [.12, .12]


def test_incompatible_execution_versions_are_rejected():
    case = make_case("mixed_versions", cooperative=True)
    case["actions"] = [10, 14]
    factory = lambda a, f: EffectPushSkill(a, f, config(joint_execution="legacy_v3" if a.index == 10 else "proportional_normal_v1"))
    w = make_world(case, factory)
    with pytest.raises(ValueError, match="same execution protocol"):
        w.joint_step(*case["actions"])


def test_unknown_joint_protocol_is_rejected():
    with pytest.raises(ValueError, match="unknown joint execution"):
        config(joint_execution="implicit_merge")


def test_unreachable_preparation_exits_without_spending_the_full_window():
    case = next(c for c in joint_cases() if c["name"] == "same_face_N_NW")
    w = make_world(case)
    b = w.joint_step(*case["actions"])
    assert all(v["exit_reason"] == "joint_prepare_stalled" for v in b["skill_ends"].values())
    assert b["end_time"] < 10.
    assert b["execution"]["normal_reference_budget"] == [0., 0.]


def test_opposed_long_references_can_finish_without_object_motion():
    case = next(c for c in joint_cases() if c["name"] == "N_S_0.5_0.5")
    w = make_world(case)
    b = w.joint_step(*case["actions"])
    assert all(v["exit_reason"] == "normal_reference_complete" for v in b["skill_ends"].values())
    assert w.object_state().x == pytest.approx(w.instance.object_pose.x)
    assert w.object_state().y == pytest.approx(w.instance.object_pose.y)
