"""Own-pose normal odometry, separately from commanded dose and object effect."""
from dataclasses import replace
from functools import partial
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.effect_push import (
    ACTIONS, VERSION, EffectPushConfig, EffectPushSkill, estimate_face, make_effect_skill)
from coimagine.random_start.env import RandomStartEnv
from experiments.option_testbench.effect_push_eval import (
    instance_from_dict, make_case, normal_travel_cases)


def config(**kwargs):
    return EffectPushConfig(endpoint_tracking=True, fast_approach=True,
                            same_face_allocation=True, **kwargs)


def world(case, factory=None):
    return RandomStartEnv(instance_from_dict(case["instance"]), action_table=ACTIONS,
                          action_version=VERSION,
                          skill_factory=factory or partial(make_effect_skill, config=config()))


def counter():
    obs = world(make_case("counter")).boundary_observations()[0]
    face = estimate_face(obs, obs.patch, config())
    skill = EffectPushSkill(ACTIONS[12], config=config())
    skill._note_position(obs)
    skill.travel_previous_time = obs.time
    skill.travel_previous_normal = (0., 1.)
    skill.travel_previous_active = skill.travel_previous_contact = True
    return skill, replace(obs, contacting_object=True, time=obs.time + spec.CONTROL_DT), face


@pytest.mark.parametrize("dx,dy,expected", [(0.04, 0., 0.), (0., .02, .02),
                                           (.03, -.01, -.01)])
def test_signed_projection_credits_no_tangential_motion(dx, dy, expected):
    skill, obs, face = counter()
    obs = replace(obs, own=replace(obs.own, x=obs.own.x + dx, y=obs.own.y + dy))
    skill._note_position(obs)
    assert skill._measure_normal_travel(obs, face, spec.CONTROL_DT) == pytest.approx(expected)


def test_projection_uses_preceding_normal_and_not_changed_current_normal():
    skill, obs, face = counter()
    skill.step_vector = (.02, 0.)
    rotated = replace(face, normal=(math.sin(.1), math.cos(.1)))
    assert skill._measure_normal_travel(obs, rotated, spec.CONTROL_DT) == 0.


@pytest.mark.parametrize("missing", ["previous_contact", "current_contact", "active",
                                      "time_gap", "face", "normal_jump"])
def test_unobserved_or_preparation_intervals_are_not_credited(missing):
    skill, obs, face = counter()
    skill.step_vector = (0., .1)
    if missing == "previous_contact":
        skill.travel_previous_contact = False
    elif missing == "current_contact":
        obs = replace(obs, contacting_object=False)
    elif missing == "active":
        skill.travel_previous_active = False
    elif missing == "time_gap":
        obs = replace(obs, time=obs.time + 10 * spec.CONTROL_DT)
    elif missing == "face":
        face = None
    else:
        face = replace(face, normal=(1., 0.))
    assert skill._measure_normal_travel(obs, face, spec.CONTROL_DT) == 0.


def test_counter_does_not_use_estimated_object_centre_or_teammate_kinematics():
    skill, obs, face = counter()
    skill.step_vector = (.03, .02)
    face = replace(face, centre=(999., -888.), centre_observable=False)
    obs = replace(obs, teammate=replace(obs.teammate, x=-900., y=800.))
    assert skill._measure_normal_travel(obs, face, spec.CONTROL_DT) == pytest.approx(.02)


def test_braking_recoil_below_target_resumes_push_instead_of_claiming_completion(monkeypatch):
    skill, obs, face = counter()
    monkeypatch.setattr(skill, "_observe_face", lambda _: face)
    skill.config = replace(skill.config, endpoint_tracking=False)
    skill.stage = "brake"
    skill.normal_travel = skill.action.distance - .01
    skill.settled_seconds = skill.config.settle_seconds
    skill.initial_face = skill.face = face
    skill.initial_normal = skill.last_normal = face.normal
    skill.reference_centre = face.centre
    _, info = skill.step(obs, spec.CONTROL_DT)
    assert not skill.done
    assert info["stage"] == "push"
    assert info["effect"]["status"] == "normal_travel_trimming"


def test_backward_travel_is_subtracted_from_accumulated_distance():
    skill, obs, face = counter()
    skill.normal_travel = .12
    skill.step_vector = (0., -.03)
    skill._measure_normal_travel(obs, face, spec.CONTROL_DT)
    assert skill.normal_travel == pytest.approx(.09)


@pytest.fixture(scope="module")
def executions():
    names = {"scale_0.10", "scale_0.50", "same_N_0.5_0.5", "same_N_0.1_0.5", "N_S_0.5_0.5"}
    result = {}
    for case in normal_travel_cases():
        if case["name"] in names:
            w = world(case)
            result[case["name"]] = w, w.joint_step(*case["actions"])
    return result


@pytest.mark.parametrize("name,target", [("scale_0.10", .1), ("scale_0.50", .5),
                                          ("same_N_0.5_0.5", .5)])
def test_single_and_dual_translation_complete_measured_distance(executions, name, target):
    w, b = executions[name]
    assert b["skill_ends"][0]["exit_reason"] == "normal_travel_complete"
    travel = b["execution"]["normal_travel"][0]
    assert target <= travel < target + .025  # bounded braking overshoot in these fixed cases
    assert travel != pytest.approx(b["execution"]["normal_reference_budget"][0])
    if name.startswith("same_"):
        assert b["execution"]["normal_travel_pair"]
        assert b["skill_ends"][1]["exit_reason"] == "normal_travel_complete"
    # The boundary feature carries the same own-odometry contract, including Hold.
    assert w._last_progress[0] == pytest.approx((travel, target))


def test_mixed_scales_exit_independently_without_claiming_equal_finish_times(executions):
    _, b = executions["same_N_0.1_0.5"]
    assert b["skill_ends"][0]["exit_reason"] == "normal_travel_complete"
    assert b["skill_ends"][0]["time"] < b["skill_ends"][1]["time"]
    assert b["execution"]["normal_speed_references"] == pytest.approx([.024, .12])
    assert b["execution"]["normal_travel"][0] >= .1
    # Peer dynamics can still fail after one side exits; success is not manufactured.
    if b["skill_ends"][1]["exit_reason"] == "normal_travel_complete":
        assert b["execution"]["normal_travel"][1] >= .5


def test_opposed_commands_cannot_complete_without_actual_normal_travel(executions):
    _, b = executions["N_S_0.5_0.5"]
    assert all(e["exit_reason"] != "normal_travel_complete" for e in b["skill_ends"].values())
    assert max(abs(x) for x in b["execution"]["normal_travel"]) < .02


def test_mixed_translation_completion_modes_are_rejected():
    case = make_case("mixed_units", cooperative=True)
    case["actions"] = [10, 14]
    factory = lambda a, f: EffectPushSkill(a, f, config(
        translation_completion="normal_reference" if a.index == 10 else "normal_travel"))
    with pytest.raises(ValueError, match="same completion measure"):
        world(case, factory).joint_step(*case["actions"])


def test_unknown_completion_mode_is_rejected():
    with pytest.raises(ValueError):
        config(translation_completion="hidden_object_truth")
