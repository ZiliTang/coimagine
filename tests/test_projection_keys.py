"""Owned packet transfer preserves validation, isolation and candidate association."""
from copy import deepcopy
from pathlib import Path

import pytest
from experiments.planar_tasks.relational_search import contract as C
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.semantic_rl.environment import Environment


def poison(value):
    if isinstance(value, dict):
        for v in list(value.values()):poison(v)
        value['unexpected_private_field'] = 'changed'
    elif isinstance(value, list):
        for v in value:poison(v)
        value.append('changed')


def test_environment_public_packet_owns_all_mutable_values():
    world = Environment(seed_fixture(),record_details=False)
    env = world.env
    packet = env.public();expected = deepcopy(packet);audit = env.offline_audit()
    poison(packet)
    assert env.offline_audit() == audit
    assert env.public() == expected


def test_external_projection_still_copies_and_owned_projection_still_filters():
    raw = Environment(seed_fixture(),record_details=False).env.public()
    raw['secret'] = 7
    raw['robots'][0]['own']['mass'] = 77
    raw['robots'][0]['scan']['hidden_geometry'] = [[999,999]]
    raw['robots'][0]['execution']['private'] = 42
    raw['entities'][0]['true_pose'] = [999,999]
    expected = deepcopy(raw)
    public = C.public_observation(raw)
    transferred = C._owned_public_observation(deepcopy(raw))
    assert public == transferred
    assert 'secret' not in transferred
    assert 'mass' not in transferred['robots'][0]['own']
    assert 'hidden_geometry' not in transferred['robots'][0]['scan']
    assert 'private' not in transferred['robots'][0]['execution']
    assert 'true_pose' not in transferred['entities'][0]
    poison(public)
    assert raw == expected


def outcome(fn, value):
    try:return ('ok',fn(value))
    except (TypeError,ValueError,KeyError) as e:return (type(e).__name__,str(e))


@pytest.mark.parametrize('bad', ['tick','future','duplicate','labels','goal','global_menu','entity_menu','nan'])
def test_owned_projection_has_identical_rejections(bad):
    raw=Environment(seed_fixture(),record_details=False).env.public()
    if bad=='tick':raw['tick']=True
    elif bad=='future':raw['entities'][0]['last_seen_tick']=raw['tick']+1
    elif bad=='duplicate':raw['entities'].append(deepcopy(raw['entities'][0]))
    elif bad=='labels':raw['robots'][0]['scan']['entity_ids'].pop()
    elif bad=='goal':raw['goal']['radius']=-1
    elif bad=='global_menu':raw['global_legal_pairs'].append([999,999])
    elif bad=='entity_menu':raw['entity_legal_pairs']['hidden']=[[44,44]]
    elif bad=='nan':raw['robots'][0]['scan']['distances'][0]=float('nan')
    slow=outcome(C.public_observation,deepcopy(raw))
    fast=outcome(C._owned_public_observation,deepcopy(raw))
    assert slow==fast and slow[0]!='ok'


def test_stream_result_lookup_is_order_independent_and_does_not_encode_reply(monkeypatch):
    from experiments.planar_tasks.semantic_rl import features as F
    from experiments.planar_tasks.semantic_rl.actions import ActionTree
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    f=F.Features(source,execution_backend='trusted_inprocess',trusted_source_sha256=C.digest(source))
    try:
        h=f.new_history();h.append(Environment(seed_fixture(),record_details=False).env.public())
        tree=ActionTree(h.events[-1]['observation'])
        expected=f.panel(h,tree,{})
        original=f.worker.evaluate
        def reversed_reply(*a,**kw):
            r=original(*a,**kw)
            r['actions']=list(reversed(r['actions']));r['values']=list(reversed(r['values']))
            return r
        monkeypatch.setattr(f.worker,'evaluate',reversed_reply)
        calls=[];canonical=F.canonical
        def counted(a):calls.append(a);return canonical(a)
        monkeypatch.setattr(F,'canonical',counted)
        actual=f.panel(h,tree,{})
        assert actual==expected
        # Only the unchanged worker mapping requires canonical action keys.
        assert len(calls)==len(actual['representatives'])
    finally:f.close()
