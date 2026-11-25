from copy import deepcopy

import numpy as np
import pytest

from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend
from experiments.planar_tasks.authoring_v2.four_stage.prediction import JointModel, feature, compact_row
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary, observation
from tools.cap_unknown_safety import fixture


@pytest.fixture
def template():
    world, _, _ = replay_boundary(fixture(False), [])
    p = PublicFrontend().prepare(observation(world))
    return dict(action="push_r0_0_4", public_context=p["public_context"],
                semantic=p["symbols"], next_semantic=deepcopy(p["symbols"]),
                process=dict(r0=[0., .01, 0., .01], r1=[0., 0., 0., 0.], target_boundary=0.))


def rows(template, prefix, count):
    return [{**deepcopy(template), "room": prefix+str(i)} for i in range(count)]


def test_small_calibration_does_not_claim_calibrated_range(template):
    model = JointModel.fit(rows(template, "fit", 4), rows(template, "cal", 4))
    p = model.predict(template["public_context"], template["action"])
    assert p["supported"]
    assert not p["calibrated"]


def test_room_balanced_calibration_and_motion_enclosure(template):
    calibration = rows(template, "cal", 10)
    calibration[-1]["process"]["r0"][1] = .03
    model = JointModel.fit(rows(template, "fit", 4), calibration)
    forecast = model.predict(template["public_context"], template["action"])
    assert forecast["calibrated"]
    assert forecast["support_rooms"] == 4
    assert forecast["branches"][0]["motion"]["r0"][1] >= .03 - 1e-12


def test_raw_scan_is_a_real_predictor_input(template):
    before = deepcopy(template["public_context"])
    changed = deepcopy(before)
    changed["observation"]["robots"][0]["scan"]["distances"][0] += .2
    assert before["symbols"] == changed["symbols"]
    assert not np.array_equal(feature(before), feature(changed))


def test_same_room_repeated_rows_do_not_inflate_support(template):
    same = rows(template, "same", 1) * 20
    model = JointModel.fit(same)
    p = model.predict(template["public_context"], template["action"])
    assert p["support_rooms"] == 1
    assert not p["supported"]


def test_fit_calibration_overlap_rejected(template):
    with pytest.raises(ValueError, match="overlap"):
        JointModel.fit(rows(template, "same", 4), rows(template, "same", 10))


def test_unseen_boolean_calibration_result_becomes_unknown(template):
    cal = rows(template, "cal", 10)
    cal[-1]["next_semantic"]["seen"] = False
    model = JointModel.fit(rows(template, "fit", 4), cal)
    p = model.predict(template["public_context"], template["action"])
    assert all(b["updates"]["seen"] is None for b in p["branches"])


def test_large_pose_shift_is_not_diluted_by_similar_raw_rays(template):
    model = JointModel.fit(rows(template, "fit", 4), rows(template, "cal", 10))
    query = deepcopy(template["public_context"])
    query["symbols"]["r0_rx"] += .5
    assert not model.predict(query, template["action"])["supported"]


def test_compacted_public_features_reproduce_raw_fit(template):
    raw = JointModel.fit(rows(template, "fit", 4), rows(template, "cal", 10))
    compact = JointModel.fit([compact_row(r, raw.host) for r in rows(template, "fit", 4)],
                            [compact_row(r, raw.host) for r in rows(template, "cal", 10)])
    assert compact.packet == raw.packet
    assert compact.predict(template["public_context"], template["action"]) == raw.predict(
        template["public_context"], template["action"])


def test_residual_units_use_only_fit_rooms_and_keep_joint_coverage(template):
    fit = rows(template, "fit", 6)
    for i, r in enumerate(fit):
        r["next_semantic"]["stagnation"] = 10.*i
    calibration = rows(template, "cal", 10)
    a = JointModel.fit(fit, calibration, normalization="fit_residual_q90")
    calibration[-1]["process"]["target_boundary"] = .04
    b = JointModel.fit(fit, calibration, normalization="fit_residual_q90")
    assert a.packet["error_scales"] == b.packet["error_scales"]
    p = b.predict(template["public_context"], template["action"])
    assert p["calibrated"]
    assert p["branches"][0]["motion"]["target_boundary"] >= .04-1e-12


def test_shorter_contained_path_does_not_count_as_motion_miss(template):
    training = rows(template, "fit", 4)
    for r in training:
        r["process"] = dict(r0=[-.1,.1,-.1,.1],r1=[-.1,.1,-.1,.1],target_boundary=.1)
    calibration = rows(template, "cal", 10)
    m = JointModel.fit(training, calibration)
    assert m.packet["calibration"][template["action"]]["padding"] == 0.
    prediction = m.predict(template["public_context"], template["action"])
    assert prediction["branches"][0]["motion"]["target_boundary"] == .1


def test_which_robot_is_moving_is_not_diluted_by_raw_ray_count(template):
    template["public_context"]["symbols"]["speed"] = .2
    template["public_context"]["observation"]["robots"][1]["own"]["vx"] = .2
    model = JointModel.fit(rows(template,"fit",4),rows(template,"cal",10))
    query = deepcopy(template["public_context"])
    query["observation"]["robots"][1]["own"]["vx"] = 0.
    query["observation"]["robots"][0]["own"]["vx"] = .2
    assert not model.predict(query,template["action"])["supported"]


def test_public_switching_stage_is_a_distinct_supported_mode(template):
    m=JointModel.fit(rows(template,"fit",4),rows(template,"cal",10),
                     mode_profile="contact_and_public_stage_v2",calibration_scope="action_mode")
    query=deepcopy(template["public_context"])
    assert m.predict(query,template["action"])["calibrated"]
    query["observation"]["robots"][0]["execution"]["stage"]="switching"
    prediction=m.predict(query,template["action"])
    assert not prediction["supported"]
    assert not prediction["calibrated"]


def test_between_prefix_query_gets_explicit_semantic_uncertainty(template):
    m=JointModel.fit(rows(template,"fit",4),rows(template,"cal",10),semantic_distance_expansion=True)
    query=deepcopy(template["public_context"])
    query["observation"]["robots"][0]["own"]["vx"]+=.02
    p=m.predict(query,template["action"])
    assert p["supported"] and p["calibrated"]
    interval=p["branches"][0]["updates"]["gx"]
    assert interval[1]-interval[0]>=.1-1e-10
    assert p["branches"][0]["motion"]["target_boundary"]==0.
