"""Effect-PUSH geometry, feedback signs, information boundary and real short executions."""
from dataclasses import replace
from functools import partial
import math

import numpy as np
import pytest

from coimagine.random_start import env as E, spec
from coimagine.random_start.effect_push import (
    ACTIONS, DISTANCES, DIRECTIONS, VERSION, WAIT_INDEX, EffectAction, EffectPushConfig,
    EffectPushSkill, action_id, action_mask, type_scale_mask, availability,
    estimate_face, make_effect_skill, tangent)
from experiments.option_testbench.effect_push_eval import make_case, instance_from_dict


def world(case=None, **kwargs):
    instance = instance_from_dict((case or make_case("test"))["instance"])
    # These tests freeze the original object-effect completion contract. New
    # paired-reference semantics are exercised in test_joint_execution.py.
    factory = partial(make_effect_skill, config=EffectPushConfig(joint_execution="legacy_v3", translation_completion="object_effect"))
    return E.RandomStartEnv(instance, skill_factory=factory,
                            action_table=ACTIONS, action_version=VERSION, **kwargs)


def test_codebook_explicit_units_and_legacy_menu_preserved():
    assert len(ACTIONS) == 66
    assert [a.index for a in ACTIONS] == list(range(66))
    assert len({a.label for a in ACTIONS}) == 66
    assert len(spec.ROBOT_ACTIONS) == 28
    assert len({(a.type_index, a.scale_index) for a in ACTIONS}) == 66
    assert all(action_id(a.type_index, a.scale_index) == a.index for a in ACTIONS)
    with pytest.raises(ValueError):
        action_id(13, 4)
    for a in ACTIONS[:40]:
        assert a.distance in DISTANCES and a.distance <= .5
        assert a.scale_kind == "distance"
    for a in ACTIONS[40:50]:
        assert a.scale_kind == "angle" and a.rotation_angle > 0 and a.distance == 0


@pytest.mark.parametrize("direction", range(8))
def test_world_normal_mask_tracks_robot_heading_and_surface_fit(direction):
    case = make_case("heading", direction=direction)
    w = world(case)
    obs = w.boundary_observations()[0]
    mask = action_mask(obs)
    assert all(mask[direction * 5:direction * 5 + 5])
    assert not any(mask[((direction + 4) % 8) * 5:((direction + 4) % 8) * 5 + 5])
    # World direction availability comes from the fit; it is not a robot-forward label.
    robots = list(w.instance.robot_poses)
    robots[0] = replace(robots[0], angle=robots[0].angle + .7)
    rotated = E.RandomStartEnv(replace(w.instance, robot_poses=tuple(robots)))
    assert action_mask(rotated.boundary_observations()[0])[direction * 5 + 2]


def test_mask_and_command_ignore_simulator_ray_labels():
    obs = world().boundary_observations()[0]
    altered = replace(obs, scan=replace(obs.scan, raw_labels=("hidden_changed",) * len(obs.scan.valid)))
    assert np.array_equal(action_mask(obs), action_mask(altered))
    first, second = EffectPushSkill(ACTIONS[12]), EffectPushSkill(ACTIONS[12])
    f0, i0 = first.step(obs, spec.CONTROL_DT)
    f1, i1 = second.step(altered, spec.CONTROL_DT)
    assert f0 == pytest.approx(f1)
    assert i0["effect"] == i1["effect"]
    structured = type_scale_mask(obs)
    assert structured.shape == (14, 5)
    assert not structured[10:, 4].any()
    assert all(structured[a.type_index, a.scale_index] == action_mask(obs)[a.index] for a in ACTIONS)


def test_both_shadow_boundaries_do_not_get_filled_by_shape_prior():
    obs = world().boundary_observations()[0]
    # Synthetic adversarial scan: every neighbour of either endpoint is an occluder.
    scan = replace(obs.scan, valid=(True,) * len(obs.scan.valid),
                   category=(3,) * len(obs.scan.category), distances=(.1,) * len(obs.scan.distances))
    assert estimate_face(replace(obs, scan=scan), obs.patch, EffectPushConfig()) is None


def test_heading_error_moves_point_in_correct_local_torque_direction():
    obs = world().boundary_observations()[0]
    face = estimate_face(obs, obs.patch, EffectPushConfig())
    skill = EffectPushSkill(ACTIONS[12])
    skill.heading_error = .1  # counterclockwise drift -> clockwise corrective torque
    skill._choose_contact(face, spec.CONTROL_DT)
    offset = np.asarray(skill.contact_target) - np.asarray(face.point)
    assert np.dot(offset, tangent(face.normal)) > 0
    torque = offset[0] * face.normal[1] - offset[1] * face.normal[0]
    assert torque < 0
    assert abs(skill.point_offset) <= skill.config.point_rate * spec.CONTROL_DT


def test_missing_tangential_geometry_cannot_manufacture_translation_progress():
    obs = world().boundary_observations()[0]
    face = estimate_face(obs, obs.patch, EffectPushConfig())
    skill = EffectPushSkill(ACTIONS[12])
    skill.reference_centre = face.centre
    displaced = replace(face, centre=(face.centre[0] + .2, face.centre[1] + .3),
                        centre_observable=False)
    skill._update_effect(displaced, spec.CONTROL_DT)
    assert skill.effect_progress == 0 and skill.lateral_error == 0


def test_effect_origin_is_not_reset_at_first_contact(monkeypatch):
    obs = world().boundary_observations()[0]
    # Isolate the effect integrator downstream of the geometry estimator.
    skill = EffectPushSkill(ACTIONS[12], config=EffectPushConfig(endpoint_tracking=False))
    skill.step(obs, spec.CONTROL_DT)
    reference = skill.reference_centre
    face = replace(skill.face, centre=(reference[0], reference[1] + .03))
    monkeypatch.setattr(skill, "_observe_face", lambda _: face)
    skill.stage = "push"
    skill.step(replace(obs, contacting_object=True), spec.CONTROL_DT)
    assert skill.reference_centre == reference
    assert skill.effect_progress == pytest.approx(.03)


def test_predicted_braking_never_counts_as_observed_completion():
    obs = world().boundary_observations()[0]
    skill = EffectPushSkill(ACTIONS[40])
    skill.step(obs, spec.CONTROL_DT)
    skill.stage = "brake"
    skill.effect_progress = .05
    skill.rotation = .05
    skill.settled_seconds = skill.config.settle_seconds
    skill.step(replace(obs, contacting_object=True), spec.CONTROL_DT)
    assert not skill.done
    assert skill.stage == "push" and skill.rotation_trimming


def test_short_partial_view_has_explicit_expiry(monkeypatch):
    obs = world().boundary_observations()[0]
    # Exercise the pre-existing line-only fallback independently of endpoint updates.
    skill = EffectPushSkill(ACTIONS[12], config=EffectPushConfig(endpoint_tracking=False))
    skill.step(obs, spec.CONTROL_DT)
    partial_face = replace(skill.face, centre_observable=False, evidence="history_line_only")
    monkeypatch.setattr(skill, "_observe_face", lambda _: partial_face)
    for _ in range(20):
        skill.step(obs, spec.CONTROL_DT)
        if skill.done:
            break
    assert skill.exit_reason == "effect_unobservable"
    assert skill.effect_progress == 0


def test_small_rotation_of_same_visible_face_is_allowed():
    w = world()
    obs = w.boundary_observations()[0]
    skill = EffectPushSkill(ACTIONS[12])
    skill.step(obs, spec.CONTROL_DT)
    instance = replace(w.instance, object_pose=replace(w.instance.object_pose, angle=.04))
    moved = E.RandomStartEnv(instance).boundary_observations()[0]
    skill.step(moved, spec.CONTROL_DT)
    assert not skill.done
    assert skill.face is not None


@pytest.mark.parametrize("kwargs", [{"speed": 0}, {"nominal_angle_deg": 100},
                                    {"heading_gain": float("nan")}, {"point_rate": -1}])
def test_invalid_control_configuration_is_explicit(kwargs):
    with pytest.raises(ValueError):
        EffectPushConfig(**kwargs)


def test_cooperative_push_uses_separate_contact_roles():
    case = make_case("pair", cooperative=True)
    w = world(case)
    b = w.joint_step(*case["actions"])
    assert all(v["exit_reason"] == "effect_complete" for v in b["skill_ends"].values())
    first = next(r for r in w.rows if r["r0"].get("effect") and r["r1"].get("effect"))
    e0, e1 = first["r0"]["effect"], first["r1"]["effect"]
    assert e0["role_offset"] * e1["role_offset"] < 0
    assert .28 <= w.object_state().y - w.instance.object_pose.y <= .34
    assert max(r["penetration"] for r in w.rows) < .01


def test_other_face_is_not_silently_selected():
    obs = world().boundary_observations()[0]
    skill = EffectPushSkill(ACTIONS[12])
    skill.step(obs, spec.CONTROL_DT)
    wrong = replace(obs.patch, normal_world=(1., 0.), tangent_world=(0., 1.))
    _, info = skill.step(replace(obs, patch=wrong, patches=(wrong,)), spec.CONTROL_DT)
    assert skill.exit_reason == "effect_face_lost"
    assert info["effect"]["auto_reposition"] is False


def test_direction_rejection_is_charged_and_does_not_end_task():
    w = world(make_case("unavailable", negative="opposite"))
    b = w.joint_step(32, WAIT_INDEX)  # PUSH_S from a north-pushing face
    assert b["skill_ends"][0]["exit_reason"] == "direction_unavailable"
    assert w.total_periods > 0 and b["reward"]["time_cost"] < 0
    assert b["termination"] is None
    assert w.object_state().x == pytest.approx(w.instance.object_pose.x)
    assert w.object_state().y == pytest.approx(w.instance.object_pose.y)


@pytest.mark.parametrize("index", [10, 14, 40, 45])
def test_real_short_effect_execution(index):
    w = world()
    initial = w.object_state()
    b = w.joint_step(index, WAIT_INDEX)
    assert b["skill_ends"][0]["exit_reason"] == "effect_complete"
    a = ACTIONS[index]
    final = w.object_state()
    if a.effect_kind == "translate":
        actual = ((final.x - initial.x) * a.effect_direction[0]
                  + (final.y - initial.y) * a.effect_direction[1])
        assert a.distance - .02 <= actual <= a.distance + .04
        assert abs(final.x - initial.x) < .04
    else:
        assert a.rotation_angle - .03 <= a.rotation_sign * (final.angle - initial.angle) <= a.rotation_angle + .08
    assert max(r["penetration"] for r in w.rows) < .01
    assert w.total_periods == w.world_advances


def test_action_parameters_are_checked_during_branch_restore():
    w = world()
    state = w.export_branch_state()
    changed = list(ACTIONS)
    changed[0] = replace(changed[0], distance=.5)  # same label, different physical request
    other = E.RandomStartEnv(w.instance, skill_factory=make_effect_skill,
                             action_table=changed, action_version=VERSION)
    with pytest.raises(ValueError, match="parameters"):
        other.restore_branch_state(state)
    legacy = E.RandomStartEnv(w.instance)
    with pytest.raises(ValueError, match="codebook"):
        legacy.restore_branch_state(state)
    world().restore_branch_state(state)
