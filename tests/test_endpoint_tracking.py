"""Observed endpoint evidence, temporal causality and controller integration."""
from dataclasses import replace
import math

import numpy as np
import pytest

from coimagine.random_start import lidar, spec
from coimagine.random_start.endpoint_tracking import EndpointTracker, observe_endpoints, interpolate_face
from coimagine.random_start.effect_push import ACTIONS, EffectPushConfig, EffectPushSkill
from experiments.option_testbench.effect_push_eval import make_case, instance_from_dict
from coimagine.random_start.env import RandomStartEnv


def observation(case=None):
    w = RandomStartEnv(instance_from_dict((case or make_case("endpoint_test"))["instance"]))
    return w.boundary_observations()[0]


def test_single_face_measures_intervals_not_last_return_corners():
    obs = observation()
    measured = observe_endpoints(obs.scan, obs.patch, obs.patches)
    assert len(measured) == 2 and all(m.observed for m in measured)
    assert all(m.kind == "silhouette_interval" and m.uncertainty > .01 for m in measured)
    t = obs.patch.tangent_world
    truth = sorted((np.dot((2., 1.875), t), np.dot((3., 1.875), t)))
    for m, q in zip(measured, truth):
        assert np.dot(m.lower, t) <= q <= np.dot(m.upper, t)


def test_two_fitted_faces_refine_the_common_physical_corner():
    scan = lidar.scan_lidar((5., 4.), (), (2.5, 2., 0.),
                            ((3.3, 1.6, math.atan2(.4, -.8)), (4.4, .5, 0.)), 0, 0, 0.)
    patches = lidar.scan_patches(scan, scan.origin)
    corners = [m for patch in patches for m in observe_endpoints(scan, patch, patches)
               if m.kind == "two_face_intersection"]
    assert len(corners) == 2
    assert all(math.dist(m.point, (3., 1.875)) < 1e-6 for m in corners)


def test_shadow_boundaries_are_not_physical_endpoints():
    obs = observation(make_case("pair", cooperative=True))
    measured = observe_endpoints(obs.scan, obs.patch, obs.patches)
    assert sum(m.observed for m in measured) == 1
    assert any(m.kind == "shadow_censored" and m.point is None for m in measured)
    result = EndpointTracker().update(obs.scan, obs.patch, obs.patches, 1., 0.)
    assert result.observed_count == 1 and result.status == "observed"
    assert sum(p is not None for p in result.endpoints) == 1


def test_synthetic_points_preserve_real_support_and_confidence():
    obs = observation()
    before = (obs.patch.arc_rays, obs.patch.confidence, obs.scan)
    points = interpolate_face(obs.patch, 4)
    assert len(points) == 4 * (obs.patch.arc_rays - 1) + 1
    assert before == (obs.patch.arc_rays, obs.patch.confidence, obs.scan)
    plane = np.mean(np.asarray(obs.patch.points_world) @ np.asarray(obs.patch.normal_world))
    assert max(abs(np.dot(p, obs.patch.normal_world) - plane) for p in points) < 1e-10


def test_duplicate_observation_is_idempotent_and_labels_are_unused():
    obs = observation()
    a, b = EndpointTracker(), EndpointTracker()
    x = a.update(obs.scan, obs.patch, obs.patches, 1., 0.)
    assert a.update(obs.scan, obs.patch, obs.patches, 1., 0.) is x
    changed = replace(obs.scan, raw_labels=("hidden_changed",) * len(obs.scan.valid))
    assert b.update(changed, obs.patch, obs.patches, 1., 0.) == x


def test_both_endpoint_predictions_expire_and_are_not_observations():
    obs = observation()
    tracker = EndpointTracker(max_age=.2)
    tracker.update(obs.scan, obs.patch, obs.patches, 1., 0.)
    occluded = replace(obs.scan, valid=(True,) * 72, category=(3,) * 72, distances=(.01,) * 72)
    predicted = tracker.update(replace(occluded, tick=1), obs.patch, obs.patches, 1., .05)
    assert predicted.status == "predicted" and predicted.observed_count == 0
    assert predicted.age == pytest.approx(.05)
    assert tracker.update(replace(occluded, tick=5), obs.patch, obs.patches, 1., .25) is None


def test_no_prior_history_cannot_fill_two_hidden_endpoints():
    obs = observation()
    hidden = replace(obs.scan, valid=(True,) * 72, category=(3,) * 72, distances=(.01,) * 72)
    assert EndpointTracker().update(hidden, obs.patch, obs.patches, 1., 0.) is None


def test_history_is_reset_on_face_change_or_clock_reversal():
    obs = observation()
    tracker = EndpointTracker()
    tracker.update(obs.scan, obs.patch, obs.patches, 1., 1.)
    tracker.states[0].velocity = (.2, .2)
    tracker.update(replace(obs.scan, tick=2), obs.patch, obs.patches, 1., .5)
    assert tracker.states[0].velocity == (0., 0.)
    # A 90-degree face is not associated with the previous endpoint identities.
    wrong = replace(obs.patch, normal_world=(1., 0.))
    tracker.update(replace(obs.scan, tick=3), wrong, (wrong,), 1., .55)
    assert tracker.last_normal == (1., 0.)


def test_endpoint_anchor_sets_contact_distance_and_privileged_labels_do_not_change_force():
    obs = observation(make_case("pair", cooperative=True))
    config = EffectPushConfig(endpoint_tracking=True)
    a, b = EffectPushSkill(ACTIONS[12], config=config), EffectPushSkill(ACTIONS[12], config=config)
    a.cooperative = b.cooperative = True
    altered = replace(obs, scan=replace(obs.scan, raw_labels=("secret",) * 72))
    fa, ia = a.step(obs, spec.CONTROL_DT)
    fb, ib = b.step(altered, spec.CONTROL_DT)
    assert fa == pytest.approx(fb) and ia["effect"] == ib["effect"]
    e = ia["effect"]
    assert e["endpoint"]["observed_count"] == 1
    anchor = e["endpoint"]["anchor"]
    assert math.dist(anchor, e["contact_target"]) == pytest.approx(e["anchor_distance"])


def test_predicted_endpoint_never_credits_full_translational_effect():
    obs = observation()
    skill = EffectPushSkill(ACTIONS[12], config=EffectPushConfig(endpoint_tracking=True))
    skill.step(obs, spec.CONTROL_DT)
    hidden = replace(obs.scan, tick=1, valid=(True,) * 72, category=(3,) * 72, distances=(.01,) * 72)
    skill.step(replace(obs, scan=hidden, time=.05), spec.CONTROL_DT)
    assert not skill.face.centre_observable
    assert skill.effect_progress == 0 and skill.exit_reason != "effect_complete"


@pytest.mark.parametrize("value", [0., float("nan"), -1.])
def test_invalid_tracker_time_constants_are_rejected(value):
    with pytest.raises(ValueError):
        EndpointTracker(tau=value)


def test_world_frame_tracking_rejects_robot_ego_motion_as_object_motion():
    tracker = EndpointTracker()
    centers = []
    for i in range(60):
        yaw = math.pi / 2 + math.radians(3) * math.sin(i * .13)
        scan = lidar.scan_lidar((5., 4.), (), (2.5, 2., 0.),
                                ((2.5, 1.595, yaw), (4.4, .5, 0.)), 0, i, i * .05)
        patches = lidar.scan_patches(scan, scan.origin)
        result = tracker.update(scan, patches[0], patches, 1., i * .05)
        centers.append(result.midpoint)
    assert max(math.dist(c, (2.5, 1.875)) for c in centers) < .001
