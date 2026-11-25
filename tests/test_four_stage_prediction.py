from copy import deepcopy

import numpy as np
import pytest

from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend
from experiments.planar_tasks.authoring_v2.four_stage.prediction import JointModel, feature
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
