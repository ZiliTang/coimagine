"""Own-scan reservations, selective clearance, and a real same-face contact regression."""
from dataclasses import replace
from functools import partial
import math

import pytest

from coimagine.random_start import spec
from coimagine.random_start.env import RandomStartEnv
from coimagine.random_start.effect_push import ACTIONS, VERSION, EffectPushConfig, EffectPushSkill, estimate_face, make_effect_skill
from coimagine.random_start.same_face import SameFaceReservation, dot, sub
from experiments.option_testbench.effect_push_eval import allocation_cases, joint_cases, make_case, instance_from_dict


def sample():
    obs = RandomStartEnv(instance_from_dict(make_case("reservation", cooperative=True)["instance"])).boundary_observations()[0]
    face = estimate_face(obs, obs.patch, EffectPushConfig(endpoint_tracking=True))
    return obs, face


def test_contact_target_stays_in_reserved_order_and_outside_peer_clearance():
    obs, face = sample()
    reserve = SameFaceReservation()
    target = reserve.project(face, obs, face.point, spec.CONTROL_DT)
    assert target.status == "allocated" and target.side == -1
    t = (-face.normal[1], face.normal[0])
    assert dot(sub(target.peer, target.target), t) >= target.target_separation - 1e-9
    assert target.interval[0] <= dot(target.target, t) <= target.interval[1]
    # Even when an unconstrained direction correction requests the peer's side,
    # its projection cannot cross the reserved separator.
    other = reserve.project(face, replace(obs, time=.05), target.peer, spec.CONTROL_DT)
    assert other.side == target.side
    assert dot(sub(other.peer, other.target), t) >= other.target_separation - 1e-9


def test_working_clearance_relaxes_only_after_a_second_valid_sample():
    obs, face = sample()
    reserve = SameFaceReservation()
    first = reserve.project(face, obs, face.point, spec.CONTROL_DT)
    second = reserve.project(face, replace(obs, time=.05), face.point, spec.CONTROL_DT)
    assert first.avoidance_margin == spec.OBSTACLE_AVOID_MARGIN
    assert second.avoidance_margin == pytest.approx(.04)
    assert second.target_separation == pytest.approx(.41)


def test_fast_closure_retains_a_larger_buffer():
    obs, face = sample()
    reserve = SameFaceReservation()
    reserve.project(face, obs, face.point, spec.CONTROL_DT)
    reserve.closing_speed = .4
    closed = reserve.project(face, replace(obs, time=.05), face.point, spec.CONTROL_DT)
    assert closed.avoidance_margin > .04
    assert closed.avoidance_margin <= spec.OBSTACLE_AVOID_MARGIN


def test_peer_loss_cannot_keep_reduced_clearance_indefinitely():
    obs, face = sample()
    reserve = SameFaceReservation()
    reserve.project(face, obs, face.point, spec.CONTROL_DT)
    briefly = reserve.project(face, replace(obs, time=.05, teammate_nearest=None), face.point, spec.CONTROL_DT)
    assert briefly.avoidance_margin == spec.OBSTACLE_AVOID_MARGIN
    lost = reserve.project(face, replace(obs, time=.30, teammate_nearest=None), face.point, spec.CONTROL_DT)
    assert lost.status == "peer_unobserved"


def test_narrow_face_is_rejected_instead_of_overlapping_targets():
    obs, face = sample()
    narrow = replace(face, length=.25)
    result = SameFaceReservation().project(narrow, obs, face.point, spec.CONTROL_DT)
    assert result.status == "no_room" and result.target is None


def test_normal_centre_estimate_bias_cannot_remove_robot_separator():
    obs, face = sample()
    # Move the independently inferred object centre 5 cm toward the peer.
    t = (-face.normal[1], face.normal[0])
    biased = replace(face, point=(face.point[0]+.05*t[0], face.point[1]+.05*t[1]))
    r = SameFaceReservation().project(biased, obs, biased.point, spec.CONTROL_DT)
    assert r.status == "allocated"
    separator = .5 * (dot((obs.own.x, obs.own.y), t) + dot(r.peer, t))
    assert dot(r.target, t) <= separator - r.target_separation / 2 + 1e-9


def test_adjacent_face_does_not_enable_relaxed_same_face_avoidance():
    case = next(c for c in joint_cases() if c["name"] == "N_E_0.5_0.1")
    obs = RandomStartEnv(instance_from_dict(case["instance"])).boundary_observations()[0]
    cfg = EffectPushConfig(endpoint_tracking=True, fast_approach=True)
    old = EffectPushSkill(ACTIONS[14], config=cfg)
    new = EffectPushSkill(ACTIONS[14], config=replace(cfg, same_face_allocation=True))
    old.cooperative = new.cooperative = True
    a, _ = old.step(obs, spec.CONTROL_DT)
    b, info = new.step(obs, spec.CONTROL_DT)
    assert info["effect"]["same_face"]["status"] == "not_same_face"
    assert a == pytest.approx(b)


def test_allocator_ignores_privileged_labels_and_peer_private_pose():
    obs, _ = sample()
    cfg = EffectPushConfig(endpoint_tracking=True, fast_approach=True, same_face_allocation=True, translation_completion="normal_reference")
    a, b = EffectPushSkill(ACTIONS[14], config=cfg), EffectPushSkill(ACTIONS[14], config=cfg)
    a.cooperative = b.cooperative = True
    altered = replace(obs, scan=replace(obs.scan, raw_labels=("secret",)*len(obs.scan.valid)),
                      teammate=replace(obs.teammate, x=999., y=-999., vx=99.))
    fa, ia = a.step(obs, spec.CONTROL_DT)
    fb, ib = b.step(altered, spec.CONTROL_DT)
    assert fa == pytest.approx(fb) and ia["effect"] == ib["effect"]


def test_equal_gap_same_face_different_directions_establish_contact():
    case = next(c for c in allocation_cases() if c["name"] == "same_face_N_NW_equal_gap")
    cfg = EffectPushConfig(endpoint_tracking=True, fast_approach=True, same_face_allocation=True, translation_completion="normal_reference")
    w = RandomStartEnv(instance_from_dict(case["instance"]),
                       skill_factory=partial(make_effect_skill, config=cfg), action_table=ACTIONS, action_version=VERSION)
    b = w.joint_step(*case["actions"])
    assert all(v["exit_reason"] == "normal_reference_complete" for v in b["skill_ends"].values())
    for row in w.rows:
        a, p = row["true"]["robots"]
        assert math.dist(a[:2], p[:2]) > 2 * spec.ROBOT_RADIUS
        assert max(math.hypot(*f) for f in row["forces"]) <= spec.MAX_FORCE + 1e-9


@pytest.mark.parametrize("kwargs", [{"clearance": 0}, {"clearance": .2}, {"target_pad": float("nan")}])
def test_invalid_spacing_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError):
        SameFaceReservation(**kwargs)
