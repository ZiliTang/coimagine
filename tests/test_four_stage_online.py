from copy import deepcopy

import pytest

from experiments.planar_tasks.authoring_v2.four_stage.native_frontend import NativeFrontend
from experiments.planar_tasks.authoring_v2.four_stage.online import OnlineRollout, compatibility, hypotheses
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary, observation
from experiments.planar_tasks.relational_search.contract import digest
from tools.cap_unknown_safety import fixture


def root(hidden=False):
    world, _, _ = replay_boundary(fixture(hidden), [])
    frontend = NativeFrontend()
    prepared = frontend.prepare(observation(world))
    return frontend, prepared


def test_public_hypotheses_do_not_change_with_unseen_obstacle():
    _, a = root(False)
    _, b = root(True)
    assert a["observation"] == b["observation"]
    assert list(hypotheses(a["public_context"])) == list(hypotheses(b["public_context"]))


def test_all_observed_hits_must_be_explained():
    _, p = root()
    o = p["observation"]
    changed = deepcopy(o)
    scan = changed["robots"][0]["scan"]
    index = next(i for i, v in enumerate(scan["valid"]) if v)
    scan["distances"][index] += .2
    assert compatibility(o, o)["compatible"]
    assert not compatibility(o, changed)["compatible"]


def test_nominal_rollout_cannot_self_certify_or_mutate_public_root():
    frontend, p = root()
    before = digest(p["public_context"])
    result = OnlineRollout(frontend, per_query=1).run(
        p["public_context"], "brake", p["commands"]["brake"])
    assert not result["errors"]
    assert not result["admissible_evidence"]
    assert result["cost"]["branches"] == 1
    assert result["cost"]["ticks"] == 44
    assert result["branches"]
    assert digest(frontend.last_packet["public_context"]) == before


def test_budget_and_wrong_root_fail_closed():
    frontend, p = root()
    rollout = OnlineRollout(frontend, max_ticks=0)
    result = rollout.run(p["public_context"], "brake", p["commands"]["brake"])
    assert result["incomplete"] and result["cost"]["branches"] == 0
    changed = deepcopy(p["public_context"])
    changed["symbols"]["gx"] += .1
    with pytest.raises(ValueError, match="current public boundary"):
        rollout.run(changed, "brake", p["commands"]["brake"])


def test_multiple_observations_are_geometry_history_not_fake_commands():
    world,_,_=replay_boundary(fixture(False),[])
    frontend=NativeFrontend();p=frontend.prepare(observation(world))
    world.step(p["commands"]["brake"]);frontend.record_command("brake")
    p=frontend.prepare(observation(world))
    assert len(p["public_context"]["history"])==2
    result=OnlineRollout(frontend,per_query=1).run(p["public_context"],"brake",p["commands"]["brake"])
    assert not result["errors"]
    assert result["cost"]["branches"]==1
