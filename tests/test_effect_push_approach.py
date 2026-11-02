"""Preparation speed, braking handshake and real option effect regressions."""
from dataclasses import replace
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.effect_push import ACTIONS, EffectPushConfig, EffectPushSkill
from experiments.option_testbench.effect_push_eval import make_case, run_one, instance_from_dict
from coimagine.random_start.env import RandomStartEnv


def candidate():
    return EffectPushSkill(ACTIONS[12], config=EffectPushConfig(endpoint_tracking=True, fast_approach=True))


def test_free_approach_is_faster_but_contact_retains_slow_cap():
    skill = candidate()
    assert skill._approach_lateral(.8, 0., .10, False) == pytest.approx(.20)
    assert abs(skill._approach_lateral(.8, 0., .01, False)) <= .08
    assert abs(skill._approach_lateral(.8, 0., .10, True)) <= .08


def test_approach_profile_brakes_near_target_and_is_mirror_symmetric():
    skill = candidate()
    far = skill._approach_lateral(.8, .1, .1, False)
    near = skill._approach_lateral(.01, .08, .1, False)
    assert 0 <= near < far
    assert near == pytest.approx(-skill._approach_lateral(-.01, -.08, .1, False))
    assert abs(near) <= math.sqrt(2 * skill.config.approach_brake_accel * .01)
    assert skill._approach_lateral(0., .08, .1, False) == 0


def test_close_position_does_not_commit_push_with_high_lateral_velocity():
    case = make_case("staging_velocity")
    case["instance"]["robots"][0]["y"] += .07  # sensed edge gap now 3 cm
    obs = RandomStartEnv(instance_from_dict(case["instance"])).boundary_observations()[0]
    obs = replace(obs, own=replace(obs.own, vx=.12))
    skill = candidate()
    skill.approach_blend = 1.  # arrived here after a preceding fast lateral approach
    skill.step(obs, spec.CONTROL_DT)
    assert abs(skill.tangential_error) < .025
    assert skill.stage == "approach" and not skill.contact_seen


def test_small_initial_correction_keeps_previous_controller():
    case = make_case("small_correction", cooperative=True, tilt=math.radians(8))
    obs = RandomStartEnv(instance_from_dict(case["instance"])).boundary_observations()[0]
    old = EffectPushSkill(ACTIONS[12], config=EffectPushConfig(endpoint_tracking=True))
    new = candidate()
    old.cooperative = new.cooperative = True
    force0, _ = old.step(obs, spec.CONTROL_DT)
    force1, _ = new.step(obs, spec.CONTROL_DT)
    assert new.approach_blend == 0
    assert force0 == pytest.approx(force1)


def test_fast_path_respects_information_boundary_and_force_ramp():
    case = make_case("far_offset", offset=.3)
    obs = RandomStartEnv(instance_from_dict(case["instance"])).boundary_observations()[0]
    a, b = candidate(), candidate()
    altered = replace(obs, scan=replace(obs.scan, raw_labels=("hidden",) * len(obs.scan.valid)))
    force, info = a.step(obs, spec.CONTROL_DT)
    force2, info2 = b.step(altered, spec.CONTROL_DT)
    assert force == pytest.approx(force2)
    assert info["effect"] == info2["effect"]
    assert math.hypot(*info["reference_velocity"]) <= spec.A_REF_MAX * spec.CONTROL_DT + 1e-9
    assert math.hypot(*force) <= spec.MAX_FORCE


def test_short_rotation_completes_without_extending_its_budget():
    record = run_one(make_case("short_rotation", rotation=1, angle=.1), True,
                     {"endpoint_tracking": True, "fast_approach": True}, "fast_approach")
    metric = record["metric"]
    assert metric["exit"] == "effect_complete" and metric["fulfilled"]
    assert metric["seconds"] < 6.4
    assert metric["max_penetration"] < .01


@pytest.mark.parametrize("field,value", [("approach_tangential_speed", 0),
                                         ("approach_brake_accel", float("nan")),
                                         ("approach_velocity_damping", -1)])
def test_invalid_approach_limits_are_rejected(field, value):
    with pytest.raises(ValueError):
        EffectPushConfig(**{field: value})
