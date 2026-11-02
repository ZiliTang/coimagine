"""Post-threshold following is a separate, bounded phase, not extra push credit."""
from dataclasses import asdict, replace
from functools import partial
import json
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.effect_push import (
    ACTIONS, VERSION, EffectPushConfig, EffectPushSkill, estimate_face, make_effect_skill)
from coimagine.random_start.env import RandomStartEnv
from experiments.option_testbench.effect_push_eval import instance_from_dict, post_follow_cases


def config(**kwargs):
    return EffectPushConfig(endpoint_tracking=True, fast_approach=True,
                            same_face_allocation=True, post_push="contact_follow", **kwargs)


def execute(name, cfg=None, factory=None):
    case = next(c for c in post_follow_cases() if c["name"] == name)
    w = RandomStartEnv(instance_from_dict(case["instance"]), action_table=ACTIONS,
                       action_version=VERSION,
                       skill_factory=factory or partial(make_effect_skill, config=cfg or config()))
    return w, w.joint_step(*case["actions"])


@pytest.fixture(scope="module")
def mixed():
    return execute("same_N_0.1_0.5")


def test_threshold_switches_directly_to_follow_without_intermediate_brake(mixed):
    w, b = mixed
    a_time = b["execution"]["drive_complete_times"][0]
    row = next(r for r in w.rows if r["r0"].get("effect", {}).get("drive_complete_time") == a_time)
    assert row["r0"]["stage"] == "follow"
    assert row["r0"]["desired_velocity"][1] > 0
    assert b["execution"]["drive_complete_times"][0] < b["execution"]["drive_complete_times"][1]
    assert all(e["exit_reason"] == "normal_travel_complete" for e in b["skill_ends"].values())


def test_following_travel_is_recorded_separately_and_primary_counter_freezes(mixed):
    w, b = mixed
    rows = [r["r0"]["effect"] for r in w.rows if r["r0"].get("effect") and r["r0"]["effect"]["follow_started"]]
    assert len(rows) > 20
    assert all(e["normal_travel"] == rows[0]["normal_travel"] for e in rows)
    assert all(e["normal_reference_budget"] == rows[0]["normal_reference_budget"] for e in rows)
    assert .1 <= rows[0]["normal_travel"] < .12
    assert b["execution"]["follow_normal_travel"][0] > .1
    assert b["execution"]["follow_path_length"][0] >= b["execution"]["follow_normal_travel"][0]


def test_follow_speed_and_command_force_are_bounded(mixed):
    w, _ = mixed
    for r in w.rows:
        i = r["r0"]
        if i["stage"] != "follow":
            continue
        n, f = i["effect"]["face_normal"], i["force"]
        assert math.hypot(*i["desired_velocity"]) <= config().follow_speed + 1e-9
        assert math.hypot(*f) <= config().follow_force + 1e-9
        assert abs(sum(x*y for x, y in zip(n, f))) <= config().follow_normal_force + 1e-9


def test_follow_contact_point_does_not_keep_correcting_nominal_heading(mixed):
    w, _ = mixed
    offsets = [r["r0"]["effect"]["offset"] for r in w.rows if r["r0"]["stage"] == "follow"]
    assert max(offsets) - min(offsets) < 1e-10


def test_joint_braking_waits_for_preceding_completion_flags(mixed):
    w, b = mixed
    event = next(e for e in b["execution"]["events"] if e["event"] == "follow_stop_released")
    assert event["time"] >= max(b["execution"]["drive_complete_times"]) + spec.CONTROL_DT - 1e-9
    assert not event["peer_failure"]
    assert all(r["r0"]["stage"] != "brake" for r in w.rows if r["time"] <= event["time"])


@pytest.mark.parametrize("name,expected", [("N_E_0.5_0.1", "follow_contact_lost")])
def test_other_face_loss_is_an_explicit_failure(name, expected):
    _, b = execute(name)
    assert b["skill_ends"][1]["exit_reason"] == expected
    assert b["skill_ends"][0]["exit_reason"] == "normal_travel_complete"


@pytest.mark.parametrize("limits,expected", [({"follow_max_seconds": .1}, "follow_time_limit"),
                                            ({"follow_max_travel": .001}, "follow_travel_limit")])
def test_follow_budgets_are_enforced(limits, expected):
    _, b = execute("same_N_0.1_0.5", config(**limits))
    assert b["skill_ends"][0]["exit_reason"] == expected


def test_peer_failure_releases_remaining_follower_for_braking():
    class FailActive(EffectPushSkill):
        def step(self, obs, dt):
            force, info = super().step(obs, dt)
            if obs.robot_id == 1 and obs.time >= 4.5 and not self.done:
                self._finish("injected_active_failure", obs)
                info.update(done=True, stage="done", exit_reason=self.exit_reason)
            return force, info
    _, b = execute("same_N_0.1_0.5", factory=lambda a, f: FailActive(a, f, config()))
    assert b["skill_ends"][1]["exit_reason"] == "injected_active_failure"
    event = next(e for e in b["execution"]["events"] if e["event"] == "follow_stop_released")
    assert event["peer_failure"]
    assert b["skill_ends"][0]["exit_reason"] == "normal_travel_complete"


@pytest.mark.parametrize("name", ["rotation_mixed_angles", "scale_0.50"])
def test_solo_and_rotation_do_not_enter_follow(name):
    w, b = execute(name)
    assert b["execution"]["post_exit"] == "hold_brake"
    assert not any(r[f"r{i}"]["stage"] == "follow" for r in w.rows for i in (0, 1))


def test_follow_command_ignores_peer_private_kinematics_and_hidden_ray_labels(mixed):
    w, _ = mixed
    # Input counterfactual at one ordinary observation, no scene truth API passed.
    obs = w.boundary_observations()[0]
    altered = replace(obs, teammate=replace(obs.teammate, x=-999., y=888.),
                      scan=replace(obs.scan, raw_labels=("changed",)*len(obs.scan.valid)))
    a, b = EffectPushSkill(ACTIONS[10], config=config()), EffectPushSkill(ACTIONS[10], config=config())
    for skill in (a, b):
        face = estimate_face(obs, obs.patch, config())
        assert face is not None
        skill.initial_face = skill.face = face
        skill.initial_normal = skill.last_normal = face.normal
        skill.reference_centre = face.centre
        skill.follow_started = skill.follow_enabled = skill.joint_pair = skill.joint_active = True
        skill.stage = "follow"
    fa, ia = a.step(obs, spec.CONTROL_DT)
    fb, ib = b.step(altered, spec.CONTROL_DT)
    assert ia["stage"] == ib["stage"] == "follow"
    assert fa == pytest.approx(fb)
    assert ia["effect"] == ib["effect"]


def test_mixed_post_push_protocols_are_rejected():
    factory = lambda a, f: EffectPushSkill(a, f, replace(config(),
        post_push="contact_follow" if a.index == 10 else "hold_brake"))
    with pytest.raises(ValueError, match="same post-push protocol"):
        execute("same_N_0.1_0.5", factory=factory)


def test_saved_pose_audit_checks_both_counters_and_detects_tampering(mixed):
    from experiments.option_testbench.post_push_follow_audit import audit
    w, b = mixed
    case = next(c for c in post_follow_cases() if c["name"] == "same_N_0.1_0.5")
    dx, dy = w.object_state().x-w.instance.object_pose.x, w.object_state().y-w.instance.object_pose.y
    record = json.loads(json.dumps({"case": case, "rows": w.rows, "boundary": b,
        "config": asdict(config()), "manifest_actions": [asdict(a) for a in ACTIONS],
        "metric": {"case": case["name"], "variant": True,
                   "exits": [b["skill_ends"][i]["exit_reason"] for i in (0, 1)],
                   "normal_reference_budget": b["execution"]["normal_reference_budget"],
                   "actual_displacement_xy": [dx, dy], "actual_effect": dy,
                   "seconds": b["periods"]*spec.CONTROL_DT,
                   "rotation_deg": math.degrees(w.object_state().angle-w.instance.object_pose.angle),
                   "max_penetration": max(r["penetration"] for r in w.rows)}}))
    result = audit(record)
    assert result["independent_follow_normal_travel_m"][0] > .2
    assert result["max_reconstruction_error_m"] < 1e-8
    row = next(row for row in record["rows"] if row["r0"]["stage"] == "follow")
    row["r0"]["effect"]["normal_travel"] += .1
    with pytest.raises(AssertionError):
        audit(record)


@pytest.mark.parametrize("kwargs", [{"post_push": "unknown"}, {"follow_force": 20},
                                    {"follow_normal_force": 4}, {"follow_max_seconds": 0},
                                    {"follow_speed": float("nan")},
                                    {"translation_completion": "normal_reference", "post_push": "contact_follow"}])
def test_invalid_follow_protocols_and_limits_are_rejected(kwargs):
    with pytest.raises(ValueError):
        EffectPushConfig(**kwargs)
