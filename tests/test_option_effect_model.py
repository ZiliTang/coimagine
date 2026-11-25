import json

import pytest

from experiments.planar_tasks.authoring_v2.option_effect_model import EffectModel


def features(**changes):
    public = dict(pusher_static_gap_m=.35, peer_gap_m=.7, peer_along_m=.2, peer_lateral_m=-.1,
                  target_ray_fraction=.25, push_speed_mps=.4, goal_along_m=1.1,
                  peer_yield=0, next_face_offered=1)
    return dict(public, **changes)


def row(room, succeeded=True, dose=.1, advance=.09, **changes):
    return dict(room=room, progress_m=dose, features=features(**changes),
                outcome=dict(forward_m=advance, lateral_m=-advance, yaw_rad=.01,
                             peer_motion_m=.015, object_static=False,
                             endpoint_followup_available=True, followup_completed=succeeded))


def test_equal_mass_per_room_and_joint_outcome_dependence():
    rows = [row("many", advance=.1)] * 12 + [row("single", False, advance=-.1)]
    model = EffectModel.fit(rows, min_rooms=2)
    pred = model.predict(.1, features())
    assert pred["status"] == "local"
    assert pred["room_count"] == pred["dose_support_rooms"] == pred["local_support_rooms"] == 2
    assert pred["evidence"] == "empirical"
    assert pred["failure_rate"] == pytest.approx(.5)
    assert pred["mean"]["forward_m"] == pytest.approx(0)
    assert pred["variance"]["forward_m"] == pytest.approx(.01)
    assert pred["interval"]["forward_m"] == [-.1, .1]
    # A legal menu at the endpoint does not guarantee physical completion.
    assert pred["mean"]["endpoint_followup_available"] == 1
    for sample in pred["samples"]:
        outcome = sample["outcome"]
        assert outcome["lateral_m"] == -outcome["forward_m"]
        assert outcome["followup_completed"] == (outcome["forward_m"] > 0)


def test_fixed_neighbourhood_ignores_remote_context_and_another_dose():
    rows = [
        row("one"), row("two", False, peer_gap_m=.73),
        row("three", False, peer_gap_m=.86), row("far", False, peer_gap_m=1.3),
        row("flag", False, peer_yield=1), row("one", False, peer_gap_m=.74),
        row("other dose", False, dose=.2),
    ]
    model = EffectModel.fit(rows, min_rooms=2, nearest_rooms=2)
    pred = model.predict(.1, features())
    assert pred["local_support_rooms"] == 3
    assert [s["room"] for s in pred["samples"]] == ["one", "two"]
    assert pred["failure_rate"] == .5
    pooled = model.predict(.1, features(peer_yield=1))
    assert pooled["status"] == "pooled"
    assert pooled["room_count"] == 5
    assert pooled["local_support_rooms"] == 1
    unsupported = model.predict(.2, features())
    assert unsupported["status"] == "unknown"
    assert unsupported["dose_support_rooms"] == 1
    assert unsupported["failure_rate"] is None
    assert not unsupported["samples"]


def test_tied_cutoff_does_not_use_room_spelling_and_frozen_rows_roundtrip():
    rows = [row("z"), row("a", False), row("b", False)]
    model = EffectModel.fit(rows, min_rooms=2, nearest_rooms=2)
    baseline = model.predict(.1, features())
    assert baseline["room_count"] == 3
    assert baseline["failure_rate"] == pytest.approx(2 / 3)
    assert model.to_dict() == EffectModel.fit(reversed(rows), min_rooms=2, nearest_rooms=2).to_dict()
    frozen = json.loads(json.dumps(model.to_dict()))
    restored = EffectModel.from_dict(frozen)
    assert restored.predict(.1, features()) == baseline
    frozen["rows"][0]["outcome"]["forward_m"] = 32
    rows[0]["features"]["goal_along_m"] = 32
    baseline["samples"][0]["outcome"]["forward_m"] = 32
    assert restored.predict(.1, features()) == model.predict(.1, features())
    frozen["public_scales"]["peer_gap_m"] *= 2
    with pytest.raises(ValueError, match="incompatible"):
        EffectModel.from_dict(frozen)


def test_current_public_inputs_cannot_include_truth_or_the_observed_followup():
    sample = row("r")
    sample.update(room_seed=293, end_truth={"tag": "private"})
    model = EffectModel.fit([sample], min_rooms=1)
    snapshot = model.to_dict()
    assert not {"room_seed", "end_truth"} & set(snapshot["rows"][0])
    for public in [features(hidden_box_x=.5), features(followup_completed=True),
                   features(peer_along_m=float("inf")), features(target_ray_fraction=1.3),
                   features(next_face_offered=None)]:
        with pytest.raises(ValueError):
            model.predict(.1, public)
    with pytest.raises(ValueError):
        EffectModel.fit([row("r", succeeded=None)])


def test_no_data_never_becomes_evidence_of_safety():
    pred = EffectModel.fit([]).predict(.2, features())
    assert pred["status"] == "unknown"
    assert pred["mean"] == {} and pred["failure_rate"] is None
