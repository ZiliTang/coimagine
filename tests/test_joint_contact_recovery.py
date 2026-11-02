"""Gap hysteresis and causal joint reapproach; physical contact stays separately measured."""
from dataclasses import asdict, replace
import json
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.effect_push import ACTIONS, VERSION, EffectPushSkill, estimate_face
from coimagine.random_start.env import RandomStartEnv
from experiments.option_testbench.effect_push_eval import (
    ForcePulseSkill, comparison_config, instance_from_dict, recovery_cases, run_one)


def execute(name, **overrides):
    case = next(c for c in recovery_cases() if c["name"] == name)
    values = asdict(comparison_config("joint_recovery", True, {}))
    values.update(overrides)
    feedback = values.pop("feedback")
    return run_one(case, feedback, values, "joint_recovery")


@pytest.fixture(scope="module", params=["same_N_0.1_0.5_outward_pulse", "N_E_0.5_0.1_outward_pulse"])
def recovered(request):
    return execute(request.param)


def test_real_force_pulse_causes_joint_pause_and_successful_resume(recovered):
    b = recovered["boundary"]
    events = b["execution"]["events"]
    start = next(e for e in events if e["event"] == "recovery_started")
    end = next(e for e in events if e["event"] == "recovery_resumed")
    assert 0 < end["time"]-start["time"] < recovered["config"]["recovery_timeout"]
    assert b["execution"]["recovery_attempts"] == 1
    assert recovered["metric"]["exits"] == ["normal_travel_complete"]*2
    assert any(row["r0"]["stage"] == row["r1"]["stage"] == "recover" for row in recovered["rows"])


def test_recovery_excludes_active_and_follow_credit_but_records_real_path(recovered):
    rows = recovered["rows"]
    for a, b in zip(rows, rows[1:]):
        for i in (0, 1):
            if a[f"r{i}"]["stage"] == "recover":
                before, after = a[f"r{i}"]["effect"], b[f"r{i}"]["effect"]
                assert after["normal_travel"] == pytest.approx(before["normal_travel"])
                assert after["follow_normal_travel"] == pytest.approx(before["follow_normal_travel"])
                if b[f"r{i}"]["stage"] == "recover":
                    assert after["normal_reference_budget"] == before["normal_reference_budget"]
    side = recovered["case"]["disturbance"]["robot"]
    assert recovered["boundary"]["execution"]["recovery_path_length"][side] > .005


def test_recovery_has_no_propulsion_bias_and_retains_actuator_limits(recovered):
    for row in recovered["rows"]:
        for i in (0, 1):
            info = row[f"r{i}"]
            assert math.hypot(*row["forces"][i]) <= spec.MAX_FORCE + 1e-9
            if info["stage"] == "recover":
                n, raw = info["effect"]["face_normal"], info["desired_velocity"]
                vn = sum(x*y for x, y in zip(n, raw))
                assert 0 <= vn + 1e-9 <= recovered["config"]["recovery_approach_speed"] + 2e-9
                if info["gap"] <= recovered["config"]["recovery_resume_gap"]:
                    assert vn == pytest.approx(0.)


def test_resume_uses_preceding_ready_flags_with_hysteresis_hold(recovered):
    end = next(e for e in recovered["boundary"]["execution"]["events"] if e["event"] == "recovery_resumed")
    ready = [r for r in recovered["rows"] if end["time"]-.1-1e-9 <= r["time"] <= end["time"]+1e-9]
    assert len(ready) >= 2
    for r in ready[-2:]:
        assert all(r[f"r{i}"]["effect"]["recovery_ready"] for i in (0, 1))
        assert all(r[f"r{i}"]["gap"] <= recovered["config"]["recovery_resume_gap"] for i in (0, 1))
    resumed = next(r for r in recovered["rows"] if r["time"] > end["time"]+1e-9)
    assert resumed["r0"]["stage"] == ("follow" if recovered["case"]["disturbance"]["robot"] == 0 else "push")
    assert resumed["r1"]["stage"] == ("push" if recovered["case"]["disturbance"]["robot"] == 0 else "follow")


@pytest.mark.parametrize("name", ["N_E_0.5_0.1", "N_E_0.1_0.5"])
def test_small_gaps_do_not_abort_and_are_not_mislabelled_physical_contact(name):
    r = execute(name)
    assert r["metric"]["exits"] == ["normal_travel_complete"]*2
    assert r["boundary"]["execution"]["recovery_attempts"] == 0
    side = 1 if name == "N_E_0.5_0.1" else 0
    rows = [v[f"r{side}"] for v in r["rows"] if v[f"r{side}"]["stage"] == "follow"]
    assert sum(v["contacting_object"] for v in rows)/len(rows) < .2
    assert all(v["gap"] <= r["config"]["recovery_enter_gap"] for v in rows)


def test_recovery_timeout_is_explicit_and_stops_both_sides():
    r = execute("same_N_0.1_0.5_outward_pulse", recovery_timeout=.1)
    assert r["metric"]["exits"] == ["joint_recovery_timeout"]*2


def test_recovery_attempt_budget_stops_repeated_requests():
    cfg = replace(comparison_config("joint_recovery", True, {}), recovery_max_attempts=1)
    class RepeatRequest(EffectPushSkill):
        def step(self, obs, dt):
            force, info = super().step(obs, dt)
            if obs.time >= 3. and self.stage in ("push", "follow") and not self.done:
                self.recovery_requested = True  # synthetic repeated report, a budget unit test
            return force, info
    case = next(c for c in recovery_cases() if c["name"] == "same_N_0.1_0.5")
    w = RandomStartEnv(instance_from_dict(case["instance"]), action_table=ACTIONS, action_version=VERSION,
                       skill_factory=lambda a, f: RepeatRequest(a, f, cfg))
    b = w.joint_step(*case["actions"])
    assert all(e["exit_reason"] == "joint_recovery_attempt_limit" for e in b["skill_ends"].values())


def test_reapproach_motion_is_accounted_even_without_binary_contact():
    cfg = comparison_config("joint_recovery", True, {})
    case = next(c for c in recovery_cases() if c["name"] == "same_N_0.1_0.5")
    w = RandomStartEnv(instance_from_dict(case["instance"]))
    obs = w.boundary_observations()[0]
    face = estimate_face(obs, obs.patch, cfg)
    skill = EffectPushSkill(ACTIONS[10], config=cfg)
    skill.travel_previous_recovery = True
    skill.travel_previous_normal = face.normal
    skill.travel_previous_time = obs.time
    skill.step_vector = (.01, .02)
    skill._measure_normal_travel(replace(obs, time=obs.time+spec.CONTROL_DT, contacting_object=False), face, spec.CONTROL_DT)
    assert skill.normal_travel == skill.follow_normal_travel == 0.
    assert skill.recovery_normal_travel == pytest.approx(sum(x*y for x, y in zip(skill.step_vector, face.normal)))
    assert skill.recovery_path_length == pytest.approx(math.hypot(.01, .02))


def test_saved_pose_recovery_audit_detects_tampered_ledger(recovered):
    from experiments.option_testbench.joint_contact_recovery_audit import audit
    record = json.loads(json.dumps(recovered))
    record["manifest_actions"] = [asdict(a) for a in ACTIONS]
    result = audit(record)
    assert len(result["recoveries"]) == 1
    row = next(r for r in record["rows"] if r["r0"]["stage"] == "recover")
    row["r0"]["effect"]["recovery_normal_travel"] += .1
    with pytest.raises(AssertionError):
        audit(record)


def test_joint_recovery_rejects_different_thresholds():
    cfg = comparison_config("joint_recovery", True, {})
    case = next(c for c in recovery_cases() if c["name"] == "same_N_0.1_0.5")
    w = RandomStartEnv(instance_from_dict(case["instance"]), action_table=ACTIONS, action_version=VERSION,
        skill_factory=lambda a, f: EffectPushSkill(a, f, replace(cfg, recovery_enter_gap=.02 if a.index==10 else .01)))
    with pytest.raises(ValueError, match="share thresholds"):
        w.joint_step(*case["actions"])


def test_peer_failure_during_recovery_stops_the_remaining_side():
    cfg = comparison_config("joint_recovery", True, {})
    case = next(c for c in recovery_cases() if c["name"] == "N_E_0.5_0.1_outward_pulse")
    class FailRecover(ForcePulseSkill):
        def step(self, obs, dt):
            force, info = super().step(obs, dt)
            if obs.robot_id == 0 and self.stage == "recover" and not self.done:
                self._finish("injected_recovery_failure", obs)
                info.update(stage="done", done=True, exit_reason=self.exit_reason)
            return force, info
    w = RandomStartEnv(instance_from_dict(case["instance"]), action_table=ACTIONS, action_version=VERSION,
        skill_factory=lambda a, f: FailRecover(a, f, cfg, case["disturbance"]))
    b = w.joint_step(*case["actions"])
    assert b["skill_ends"][0]["exit_reason"] == "injected_recovery_failure"
    assert b["skill_ends"][1]["exit_reason"] == "peer_recovery_failed"


@pytest.mark.parametrize("changes", [{"contact_recovery": "unknown"}, {"recovery_enter_gap": .002},
                                     {"recovery_max_attempts": 0}, {"recovery_max_attempts": 1.5},
                                     {"recovery_timeout": float("nan")}, {"post_push": "hold_brake"}])
def test_invalid_recovery_contracts_rejected(changes):
    from coimagine.random_start.effect_push import EffectPushConfig
    values = asdict(comparison_config("joint_recovery", True, {}));values.update(changes)
    with pytest.raises(ValueError):
        EffectPushConfig(**values)
