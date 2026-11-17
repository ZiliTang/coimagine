"""Real isolation, mutation and long-history equivalence of feature runtime."""
from copy import deepcopy
import pytest
from experiments.planar_tasks.relational_search.contract import History
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.semantic_rl.environment import Environment
from experiments.planar_tasks.semantic_rl.actions import ActionTree, FIELDS
from experiments.planar_tasks.semantic_rl.features import Features

SOURCE='''import math
NAMES=['count','ticks','clean','counter']
counter=0
def initial_memory():return {'count':0}
def update_memory(m,e):
 m['count']+=1
 e['observation']['tick']=-1
 return m
def propose_actions(m,o):
 o['tick']=-1
 return [dict(target_id=None,a0=44,a1=44,ticks=4)]
def semantic_features(m,o,a,p):
 global counter
 counter+=1
 clean=float('secret' not in o and o['tick']>=0 and 'damaged' not in m and p[0]['ticks']==4)
 m['damaged']=1;o['tick']=-100;p[0]['ticks']=99
 return dict(count=m['count'],ticks=a['ticks'],clean=clean,counter=counter)
'''


def test_window_mutation_and_namespace_equivalence():
    o=Environment(seed_fixture()).history.events[0]['observation']
    # Smaller public menu for the adversarial test; full menus covered by artifact audit.
    o=deepcopy(o);o['entities']=[];o['entity_legal_pairs']={}
    o['legal_pairs']=o['global_legal_pairs']=[[44,44]]
    fast=Features(SOURCE,fast_embedding=True,runtime='cached_fork')
    old=Features(SOURCE,fast_embedding=True)
    try:
        for length in (1,64,65):
            events=[]
            for i in range(length):
                obs=deepcopy(o);obs['tick']=i;obs['secret']=123
                events.append(dict(observation=obs,action=None if i==0 else dict(target_id=None,a0=44,a1=44,ticks=1)))
            h=History(events[-64:]);tree=ActionTree(h.events[-1]['observation']);prefix={}
            for key in FIELDS:
                p=fast.panel(h,tree,prefix)
                assert p==old.panel(h,tree,prefix)
                assert all(row[2]==1 for row in p['semantic'])
                prefix=p['choices'][0]
            # Content mutation must invalidate the cache even with same identities.
            h.events[-1]['observation']['tick']+=1
            assert fast.panel(h,tree,{})==old.panel(h,tree,{})
    finally:fast.close()


def test_worker_failure_is_bounded_and_does_not_poison_next_query():
    h=Environment(seed_fixture()).history;tree=ActionTree(h.events[-1]['observation'])
    bad=SOURCE.replace("m['count']+=1","raise RuntimeError('author failure')")
    f=Features(bad,runtime='cached_fork')
    try:
        with pytest.raises(ValueError,match='author failure'):f.panel(h,tree,{})
        assert f.calls==1 and f.worker.process is None
    finally:f.close()


def test_hidden_extensions_never_cross_isolated_boundary():
    h=Environment(seed_fixture()).history;h.events[-1]['observation']['secret']=999
    f=Features(SOURCE,runtime='cached_fork')
    try:
        p=f.panel(h,ActionTree(h.events[-1]['observation']),{})
        assert all(row[2]==1 for row in p['semantic'])
    finally:f.close()


def test_cached_runtime_rejects_io_dependencies_and_hung_author_exits():
    from experiments.planar_tasks.semantic_rl.fast_features import FastProgram
    for extra in ('\nimport os','\nopen("/tmp/leak","w")','\ngetattr({},"__class__")'):
        with pytest.raises(ValueError,match='cached runtime'):Features(SOURCE+extra,runtime='cached_fork')
    h=Environment(seed_fixture()).history;tree=ActionTree(h.events[-1]['observation'])
    source=SOURCE.replace("m['count']+=1",'while True:pass')
    f=Features(source,runtime='cached_fork');f.worker.timeout=.4
    try:
        with pytest.raises(TimeoutError):f.panel(h,tree,{})
        assert f.worker.process is None
    finally:f.close()


def test_history_digest_cache_does_not_hide_inplace_mutation():
    from experiments.planar_tasks.relational_search.contract import digest
    h=Environment(seed_fixture()).history
    assert h.sha256==digest(h.events)
    original=h.sha256
    h.events[-1]['observation']['goal']['xy'][0]+=.1
    assert h.sha256==digest(h.events) and h.sha256!=original
