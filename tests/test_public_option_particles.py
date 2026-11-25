import pytest

from experiments.planar_tasks.authoring_v2.public_option_particles import PublicOptionParticles


PUBLIC = dict(pusher_static_gap_m=0.60, peer_gap_m=0.71, peer_along_m=-0.30,
              peer_lateral_m=0.15, target_ray_fraction=0.30, push_speed_mps=0.15,
              goal_along_m=1.60, peer_yield=1, next_face_offered=1)
ACTION = dict(a0=5, a1=46, ticks=10, target_id="task-object")


def donor(room, anchor="early", **fields):
    return dict(dict(room=room, anchor_id=anchor, public_history=f"{room}:{anchor}",
                     features=PUBLIC, original_action=ACTION), **fields)


def query(pool, simulate, **changes):
    fields = dict(features=PUBLIC, original_action=ACTION, public_history="q", progress_m=.20)
    return pool.probe(**dict(fields, **changes), simulate=simulate)


def endpoint(**changes):
    return dict(dict(prefix_ticks=65, push_ticks=31, followup_ticks=0,
                     forward_m=.16, lateral_m=-.04, yaw_rad=.03, peer_motion_m=.05,
                     object_static=False, endpoint_followup_available=True,
                     endpoint_terminal=None, push_reason="review_switch", progress_reached=True),
                **changes)


def test_chooses_two_distinct_rooms_in_public_order_and_counts_the_work():
    records = [donor("z"), donor("b", "late"), donor("a", "late"), donor("a"), donor("c")]
    pool = PublicOptionParticles(records)
    calls = []

    def simulate(room, anchor, dose):
        calls.append((room, anchor, dose))
        return endpoint(prefix_ticks=100 if anchor == "early" else 900,
                        push_ticks=22 if room == "a" else 40,
                        forward_m=.09 if room == "a" else .25, object_static=room == "b")

    result = query(pool, simulate, public_history="b:late", progress_m=.10,
                   original_action=dict(ACTION, ticks=35, target_id="another-room-object"))
    assert calls == [("a", "early", .10), ("b", "late", .10)]
    assert result["status"] == "simulated" and result["compatible_room_count"] == 4
    assert (result["calls"], result["prefix_ticks"], result["candidate_ticks"], result["followup_ticks"]) == (
        2, 1000, 62, 0)
    assert [m["exact_public_history_match"] for m in result["matches"]] == [False, True]
    assert [(s["room"], s["outcome"]["forward_m"], s["outcome"]["object_static"]) for s in result["samples"]] == [
        ("a", .09, False), ("b", .25, True)]
    assert all("followup_completed" not in s["outcome"] for s in result["samples"])
    assert result["samples"][0]["outcome"]["endpoint_followup_available"]
    calls.clear()
    changed = query(PublicOptionParticles(reversed(records)), simulate, public_history="z:early")
    assert calls == [("a", "early", .20), ("b", "late", .20)]
    assert all(not m["exact_public_history_match"] for m in changed["matches"])


@pytest.mark.parametrize(("field", "value"), [
    ("pusher_static_gap_m", .49), ("peer_gap_m", .69), ("peer_along_m", .01),
    ("peer_lateral_m", -.01), ("target_ray_fraction", .299), ("push_speed_mps", .149),
    ("goal_along_m", 1.80), ("peer_yield", 0), ("next_face_offered", 0),
])
def test_each_public_cell_coordinate_is_required(field, value):
    pool = PublicOptionParticles([donor("a"), donor("b", features=dict(PUBLIC, **{field: value}))])
    result = query(pool, lambda *args: pytest.fail("fewer than two compatible rooms"))
    assert result["status"] == "insufficient" and result["compatible_room_count"] == 1
    assert result["calls"] == result["prefix_ticks"] == result["candidate_ticks"] == 0
    assert result["samples"] == []


@pytest.mark.parametrize("action", [dict(ACTION, a0=4), dict(ACTION, a0=9),
                                    dict(ACTION, a1=44), dict(ACTION, a0=46, a1=5)])
def test_joint_code_preserves_speed_direction_and_both_roles(action):
    pool = PublicOptionParticles([donor("a"), donor("a", "late"), donor("b", original_action=action)])
    result = query(pool, lambda *args: pytest.fail("duplicate anchors must not count twice"))
    assert result["calls"] == 0 and [m["room"] for m in result["matches"]] == ["a"]


def test_rejects_privileged_query_inputs_and_non_endpoint_simulations():
    with pytest.raises(ValueError, match="public donor"):
        PublicOptionParticles([donor("a", hidden_state=(1, 2))])
    pool = PublicOptionParticles([donor("a"), donor("b")])
    with pytest.raises(ValueError, match="public input"):
        query(pool, None, features=dict(PUBLIC, object_static=True))
    with pytest.raises(ValueError, match="endpoint"):
        query(pool, lambda *args: endpoint(followup_completed=True))
    with pytest.raises(ValueError, match="40-tick"):
        query(pool, lambda *args: endpoint(push_ticks=41))
    with pytest.raises(ValueError, match="progress"):
        query(pool, None, progress_m=.30)
