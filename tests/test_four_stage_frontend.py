from copy import deepcopy

import pytest

from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend, schema, OPTIONS
from experiments.planar_tasks.authoring_v2.four_stage.compiler import Extractor
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary, observation
from tools.cap_unknown_safety import fixture


def initial(blocked=False):
    world, _, _ = replay_boundary(fixture(blocked), [])
    return observation(world)


def test_public_symbols_ignore_unexported_truth_and_registry_is_complete():
    o = initial()
    altered = deepcopy(o)
    altered["truth"] = dict(target_position=[1000, -1000], obstacle_at_goal=True)
    a = PublicFrontend().prepare(o)
    b = PublicFrontend().prepare(altered)
    assert a["symbols"] == b["symbols"]
    assert set(a["symbols"]) == set(schema()["metrics"])
    assert len(a["commands"]) == 69


def test_same_public_observation_cannot_reveal_occluded_obstacle():
    empty, blocked = initial(False), initial(True)
    assert empty == blocked
    a, b = PublicFrontend().prepare(empty), PublicFrontend().prepare(blocked)
    assert a["symbols"] == b["symbols"]
    assert a["reports"] == b["reports"]
    assert any(a["symbols"]["unknown:" + name] for name in OPTIONS if name.startswith("push"))


def test_registered_prediction_values_do_not_mutate_observation_memory():
    f = PublicFrontend()
    packet = f.prepare(initial())
    values = f.register(packet)
    values["seen"] = not values["seen"]
    assert values["seen"] != packet["symbols"]["seen"]
    before = deepcopy(packet["symbols"])
    code = 'def decide(z):\n    candidates = []\n    candidates.append(("brake", -1))\n    return candidates\n'
    p = Extractor(code, schema())
    p.native_candidates(before)
    assert before == packet["symbols"]


def test_both_pushers_are_in_motion_envelope():
    f = PublicFrontend()
    p = f.prepare(initial())
    both = p["regions"]["push_both_0_4"]
    single = p["regions"]["push_r0_0_4"]
    assert both["robots"][1].area > single["robots"][1].area
    assert both["target"].equals(single["target"])


def test_history_cannot_be_initialized_at_an_arbitrary_midpoint():
    o = initial()
    o["tick"] = 50
    with pytest.raises(ValueError, match="complete public prefix"):
        PublicFrontend().prepare(o)


def test_evidence_cache_preserves_frozen_query_semantics_and_invalidates():
    from shapely.geometry import Point
    from experiments.planar_tasks.authoring_v2.symbolic_unknown.evidence import EvidenceMap
    from experiments.planar_tasks.authoring_v2.four_stage.evidence import CachedEvidenceMap
    plain = EvidenceMap(minimum_static_component_area=.04)
    fast = CachedEvidenceMap(minimum_static_component_area=.04)
    o = initial()
    for m in (plain, fast):
        m.add_initial_certificate(Point(1, 1).buffer(.1), source="declared_initial_separation")
        m.update(o)
    for region in (Point(1, 1).buffer(.05), Point(4, 4).buffer(.5), Point(2, 2).buffer(.4)):
        assert plain.query(region) == fast.query(region)
    for m in (plain, fast):
        m.add_initial_certificate(Point(4, 4).buffer(.6), source="declared_initial_separation")
    assert plain.query(Point(4, 4).buffer(.1)) == fast.query(Point(4, 4).buffer(.1))
    for m in (plain, fast):
        m.forget_outside(Point(1, 1).buffer(.2))
    assert plain.query(Point(4, 4).buffer(.1)) == fast.query(Point(4, 4).buffer(.1))
