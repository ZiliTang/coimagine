from pathlib import Path

import pytest

from experiments.planar_tasks.authoring_v2.four_stage import author_round1
from experiments.planar_tasks.authoring_v2.four_stage.carrier import build,verify,checked_boundary
from experiments.planar_tasks.authoring_v2.four_stage.frontend import PublicFrontend,OPTIONS
from experiments.planar_tasks.authoring_v2.four_stage.native_frontend import NativeFrontend
from experiments.planar_tasks.authoring_v2.four_stage.native_geometry import build as build_native
from experiments.planar_tasks.authoring_v2.four_stage.gateway import Gateway
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary,observation
from tools.cap_unknown_safety import fixture


def test_dependency_registration_preserves_candidate_scores_and_selection():
    build_native()
    source=Path(author_round1.__file__).read_text()
    full,lean=verify(build(source)),verify(build(source,lean=True))
    world,_,_=replay_boundary(fixture(False),[])
    a=NativeFrontend();b=NativeFrontend(semantic_names=lean.host["metrics"])
    for _ in range(4):
        o=observation(world);pa,pb=a.prepare(o),b.prepare(o)
        assert len(pb["symbols"]) < len(pa["symbols"])
        assert {k:pa["symbols"][k] for k in pb["symbols"]} == pb["symbols"]
        assert full.native_candidates(pa["symbols"]) == lean.native_candidates(pb["symbols"])
        da,db=Gateway(full,mode="author").decide(pa),Gateway(lean,mode="author").decide(pb)
        assert da["selected"]==db["selected"]
        assert [(c["candidate"]["option"],c["verdict"]["allowed"]) for c in da["checks"]] == [(c["candidate"]["option"],c["verdict"]["allowed"]) for c in db["checks"]]
        packet=checked_boundary(build(source,lean=True),pb,{db["selected"]:True})
        assert len(packet["memory"]["four_stage_values"])==len(lean.host["metrics"])
        world.step(db["command"]);a.record_command(da["selected"]);b.record_command(db["selected"])


def test_shared_envelopes_match_original_body_bounds():
    build_native()
    world,_,_=replay_boundary(fixture(False),[])
    o=observation(world);a=PublicFrontend();b=NativeFrontend()
    pa,pb=a.prepare(o),b.prepare(o)
    for name in OPTIONS:
        for ga,gb in zip(pa["regions"][name]["robots"],pb["regions"][name]["robots"]):
            assert ga.equals(gb)
        assert pa["regions"][name]["target"].equals(pb["regions"][name]["target"])


def test_uncomputed_historical_geometry_cannot_use_a_new_observation():
    build_native()
    source=build(Path(author_round1.__file__).read_text(),lean=True);program=verify(source)
    f=NativeFrontend(semantic_names=program.host["metrics"])
    world,_,_=replay_boundary(fixture(False),[])
    old=f.prepare(observation(world))
    world.step(dict(target_id=None,a0=44,a1=44,ticks=4));f.prepare(observation(world))
    with pytest.raises(ValueError,match="after its public boundary"):
        old["reports"]["push_both_0_8"]


def test_verified_history_fast_path_matches_whitelist_and_preserves_old_packet():
    from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory,current_event
    from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
    from experiments.planar_tasks.relational_search.contract import digest
    build_native()
    source=build(Path(author_round1.__file__).read_text(),lean=True);program=verify(source)
    a=NativeFrontend(semantic_names=program.host["metrics"])
    b=NativeFrontend(semantic_names=program.host["metrics"])
    world=Environment(fixture(False),EventConfig(mode="duration"),fast_feedback=True,
                      record_details=False,history_factory=MemoryHistory)
    previous=None
    for _ in range(3):
        pa=a.prepare_history(world.history);pb=b.prepare(current_event(world.history)["observation"])
        assert pa["symbols"]==pb["symbols"]
        if previous:
            assert digest(previous[0])==previous[1]
        previous=(pa["observation"],digest(pa["observation"]))
        world.step(dict(target_id=None,a0=44,a1=44,ticks=4))
    with pytest.raises(TypeError,match="verified"):
        a.prepare_history({"observation":pa["observation"],"truth":{"x":1}})
