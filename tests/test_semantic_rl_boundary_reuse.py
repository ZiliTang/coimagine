"""Boundary ownership, legal-support parity and independent authored inputs."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import pytest
from experiments.planar_tasks.semantic_rl.boundary_actions import BoundaryActions
from experiments.planar_tasks.semantic_rl.memory_flow import MemoryHistory,current_event
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.environment import Environment
from experiments.planar_tasks.semantic_rl.trusted_native import TrustedStreamProgram,owned_copy
from experiments.planar_tasks.relational_search.contract import validate_action,digest
from experiments.planar_tasks.relational_search.motion import global_pairs
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture


def outcome(call):
    try:return ('ok',call())
    except (TypeError,ValueError,KeyError) as e:return (type(e).__name__,str(e))


def test_index_matches_complete_public_menu_and_invalid_commands():
    o=Environment(seed_fixture()).history.events[0]['observation'];index=BoundaryActions(o)
    tree=ActionTree(o);fast=ActionTree(o,_projected=True,_action_index=index)
    assert tree.choices({})==fast.choices({})
    for p in tree.choices({}):
        assert tree.choices(p)==fast.choices(p)
        for q in tree.choices(p):
            assert tree.choices(q)==fast.choices(q)
            for r in tree.choices(q):
                assert tree.choices(r)==fast.choices(r)
                assert tree.complete(r)==fast.complete(r)
                for ticks in (1,40):
                    a=dict(r,ticks=ticks)
                    assert index.validate(a,o)==validate_action(a,o)
    base=dict(target_id=None,a0=44,a1=44,ticks=1)
    invalid=[{},dict(base,ticks=True),dict(base,ticks=0),dict(base,ticks=41),dict(base,target_id='hidden'),
             dict(base,target_id=[]),dict(base,a0=-1),dict(base,a0=True),dict(base,a1=1000),dict(base,extra=1)]
    for a in invalid:assert outcome(lambda:validate_action(a,o))==outcome(lambda:index.validate(a,o))
    with pytest.raises(ValueError,match='different boundary'):index.validate(base,deepcopy(o))
    o['tick']+=1
    with pytest.raises(ValueError,match='different boundary'):index.validate(base,o)


def test_new_boundary_rebuilds_index_and_external_mutation_does_not_change_menu():
    o=Environment(seed_fixture()).history.events[0]['observation'];h=MemoryHistory();h.append(o)
    old=current_event(h)['observation'];index=h.action_index
    move=next(p for p in old['global_legal_pairs'] if p != [44,44])
    a=dict(target_id=None,a0=move[0],a1=move[1],ticks=1)
    o['global_legal_pairs'].clear();h.events[0]['observation']['global_legal_pairs'].clear()
    assert index.validate(a,old)==a
    new=deepcopy(old);new['tick']+=1;new['global_legal_pairs']=[[44,44]]
    h.append(new,dict(target_id=None,a0=44,a1=44,ticks=1))
    now=current_event(h)['observation']
    assert h.action_index is not index
    with pytest.raises(ValueError):h.action_index.validate(a,now)
    with pytest.raises(ValueError,match='different boundary'):index.validate(a,now)
    assert index.validate(a,old)==a # immutable previous boundary remains meaningful


def test_environment_adopts_own_projection_but_public_append_still_checks(monkeypatch):
    from experiments.planar_tasks.semantic_rl import memory_flow
    def reject(*args):raise RuntimeError('external projection entered')
    monkeypatch.setattr(memory_flow,'public_observation',reject)
    # Environment.public() has already performed the whitelist/validation.
    env=Environment(seed_fixture(),history_factory=MemoryHistory,record_details=False)
    assert current_event(env.history)['observation']['tick']==0
    with pytest.raises(RuntimeError,match='external projection'):MemoryHistory().append({})


def test_trusted_callbacks_cannot_mutate_host_or_each_other():
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    p=TrustedStreamProgram(source,trusted_source_sha256=digest(source))
    def update(memory,event):
        memory['count']+=1;event['observation']['tick']=999
        event['observation']['menu'][0][0]=999
        return memory
    def context(memory,o):
        assert o['tick']==3 and o['menu']==[[1,2]]
        memory['count']=999;o['menu'].clear()
        return dict(values=[1.],state=[3.,0.,0.],proposals=[])
    p.native_module=SimpleNamespace(initial_memory=lambda:dict(count=0),update_memory=update,prepare_context=context,
                                   STATE_NAMES=['a','b','c'],NAMES=['duration_match'])
    request=dict(op='advance',memory=dict(count=4),event=dict(observation=dict(tick=3,menu=[[1,2]]),executed_ticks=1))
    before=deepcopy(request);result=p.request(request)
    assert request==before and result['memory']==dict(count=5)
    assert result['state']==[3.,0.,0.]
    shared=[];value=dict(a=shared,b=shared);copy=owned_copy(value)
    assert copy['a'] is copy['b'] and copy['a'] is not shared
    copy['a'].append(1);assert shared==[]


def test_static_menu_template_returns_independent_lists():
    first=global_pairs();expected=first.copy();first.clear()
    assert global_pairs()==expected and len(expected)==len(set(expected))
