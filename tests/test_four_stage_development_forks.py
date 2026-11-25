from copy import deepcopy

from experiments.planar_tasks.authoring_v2.four_stage.development_forks import fork_call
from experiments.planar_tasks.authoring_v2.four_stage.physical import command_and_stop,state
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary,observation
from experiments.planar_tasks.relational_search.contract import digest
from tools.cap_unknown_safety import fixture


def test_development_fork_preserves_parent_and_matches_same_prefix_replay(tmp_path):
    commands=[dict(target_id=None,a0=52,a1=44,ticks=4)]*3
    world,_,_=replay_boundary(fixture(False),commands)
    root=digest(observation(world));truth=deepcopy(state(world))
    action=dict(target_id=None,a0=48,a1=44,ticks=4)
    def probe():
        measured=command_and_stop(world,action)
        return dict(status="complete",process=measured["process"],endpoint=measured["endpoint"],final=state(world))
    branched=fork_call(probe,tmp_path/"branch.json.gz")
    assert branched["child_exit"]==0
    assert digest(observation(world))==root
    assert state(world)==truth
    replay,_,_=replay_boundary(fixture(False),commands)
    measured=command_and_stop(replay,action)
    assert branched["process"]==measured["process"]
    assert digest(branched["endpoint"])==digest(measured["endpoint"])
    assert digest(branched["final"])==digest(state(replay))


def test_contiguous_brake_matches_reissued_brake_at_every_physics_tick():
    commands=[dict(target_id=None,a0=52,a1=44,ticks=4)]*3
    a,_,_=replay_boundary(fixture(False),commands)
    b,_,_=replay_boundary(fixture(False),commands)
    command=dict(target_id=None,a0=48,a1=44,ticks=4)
    slow=command_and_stop(a,command)
    fast=command_and_stop(b,command,brake_chunk=40)
    assert fast["process"]==slow["process"]
    slow_states=[s for row in slow["stopping"] for s in row["samples"]]
    fast_states=[s for row in fast["stopping"] for s in row["samples"]]
    assert slow_states==fast_states


def test_contiguous_brake_preserves_contacting_box_physics():
    f=fixture(False,robot_gap=.11)
    a,_,_=replay_boundary(f,[])
    command=dict(target_id=observation(a)["goal"]["target_id"],a0=1,a1=1,ticks=4)
    prefix=[command]*8
    a,_,_=replay_boundary(f,prefix);b,_,_=replay_boundary(f,prefix)
    slow=command_and_stop(a,command);fast=command_and_stop(b,command,brake_chunk=40)
    assert slow["root"]["goal_gain"]>0
    assert fast["process"]==slow["process"]
    assert [s for row in slow["stopping"] for s in row["samples"]] == [s for row in fast["stopping"] for s in row["samples"]]
