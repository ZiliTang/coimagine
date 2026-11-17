"""Real native compilation/loading, public boundary, bounds and CAP integration."""
from pathlib import Path
from copy import deepcopy
import pytest
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree,FIELDS
from experiments.planar_tasks.semantic_rl.environment import Environment
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source,validate_source

SOURCE=Path('experiments/planar_tasks/semantic_rl/typed_example.py').read_text()


def test_compiled_batch_prefix_missing_and_hidden_input_projection():
    source=SOURCE.replace("contact = sum(bool(r['contacting_object']) for r in observation['robots']) / 2.0",
                          "contact = float('secret' in observation)")
    h=Environment(seed_fixture()).history;h.events[-1]['observation']['secret']=123
    tree=ActionTree(h.events[-1]['observation']);f=Features(source,fast_embedding=True)
    prefix={}
    try:
        assert f.runtime=='typed_batch'
        with f.decision(h):
            for key in FIELDS:
                p=f.panel(h,tree,prefix)
                assert all(row[0]==0 for row in p['semantic'])
                assert all((row[1] is None)==('ticks' not in choice) for row,choice in zip(p['semantic'],p['choices']))
                prefix=p['choices'][0]
        assert f.worker.compilation['build']['boundscheck']
        assert f.worker.compilation['build']['native_memoryview_signature_verified']
        assert f.recommendation(h)['raw_first']['ticks']==20
    finally:f.close()


def test_compile_cache_and_directive_rejection():
    p,m=compile_source(SOURCE);q,n=compile_source(SOURCE)
    assert p==q and m==n and (p/'semantics.so').is_file()
    for bad in [SOURCE+'\nimport os', '# cython: boundscheck=False\n'+SOURCE,
                SOURCE.replace('i: cython.Py_ssize_t','i: cython.Py_ssize_t\n    cython.cast(cython.int, 0)')]:
        with pytest.raises(ValueError):validate_source(bad)


def test_const_input_cannot_be_written_and_bounds_remain_checked():
    bad=SOURCE.replace('out[i, 0] = context[1]','context[0] = 1.0')
    with pytest.raises(ValueError,match='compilation failed'):compile_source(bad)
    bad=SOURCE.replace('out[i, 0] = context[1]','out[i, 0] = context[9999]')
    f=Features(bad);h=Environment(seed_fixture()).history
    try:
        with pytest.raises(ValueError,match='IndexError'):f.panel(h,ActionTree(h.events[-1]['observation']),{})
        assert f.worker.process is None
    finally:f.close()


def test_default_author_campaign_compiles_meters_and_freezes(tmp_path):
    from experiments.planar_tasks.authoring_v2.semantic_rl import SemanticRLCampaign
    from experiments.planar_tasks.relational_search.pipeline import read
    c=SemanticRLCampaign.create(tmp_path/'new',execution_mode='duration',feature_runtime='typed_batch')
    assert read(c.path/'campaign.json')['stage1']['feature_runtime']=='typed_batch'
    assert any('TYPED_SEMANTICS' in p.name for p in (c.path/'knowledge').iterdir())
    with pytest.raises(ValueError,match='SEMANTIC_API'):c.submit('NAMES=["x"]','bad legacy format')
    s=c.submit(SOURCE,'neutral native format test')
    assert (c.path/'semantics'/f'{s["id"]}.so').is_file()
    r=c.test(s['id'],'open_transport',max_ticks=3,max_decisions=2)
    assert r['ticks']==3 and r['error'] is None
    assert c.state['used']['feature_batches']==3 # rejected submission, accepted schema, one CAP query
    assert not c.state['reservations']
    assert c.freeze()['status']=='verified'


def test_typed_memory_window_and_inplace_history_mutation():
    from experiments.planar_tasks.relational_search.contract import History
    source=SOURCE.replace('return {}',"return {'count': 0}").replace('    return memory',
        "    memory['count'] += 1\n    return memory").replace(
        "contact = sum(bool(r['contacting_object']) for r in observation['robots']) / 2.0",
        "contact = float(memory['count']) + observation['tick']/1000.")
    o=Environment(seed_fixture()).history.events[0]['observation']
    o=deepcopy(o);o['entities']=[];o['entity_legal_pairs']={}
    o['legal_pairs']=o['global_legal_pairs']=[[44,44]]
    f=Features(source)
    try:
        for length in (1,64,65):
            events=[]
            for i in range(length):
                obs=deepcopy(o);obs['tick']=i
                events.append(dict(observation=obs,action=None if i==0 else dict(target_id=None,a0=44,a1=44,ticks=1)))
            h=History(events[-64:])
            for changed in (0,1):
                if changed:h.events[-1]['observation']['tick']+=1
                p=f.panel(h,ActionTree(h.events[-1]['observation']),{})
                assert all(row[0]==min(length,64)+(length-1+changed)/1000. for row in p['semantic'])
    finally:f.close()
