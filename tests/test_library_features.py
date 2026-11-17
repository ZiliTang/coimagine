import numpy as np
from pathlib import Path
from experiments.planar_tasks.semantic_rl.library_features import LibraryFeatures
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.authoring_v2.random_room_tasks import generate
from experiments.planar_tasks.relational_search import pipeline_execution as X

def test_flat_library_preserves_support_and_causal_inputs():
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    groups={'candidate':{'duration_match':'A'},'state':{'contact_fraction':'D','blocked_fraction':'D','observed_entities':'D'}}
    entries=[dict(name=n,source=source,groups=groups) for n in ['one','two']]
    f=LibraryFeatures(entries,protocol(EventConfig(mode='duration')))
    try:
        scene,boxes=X.fixture(generate()['cases'][0]['fixture']);o=X.RecordedEnv(scene,boxes).public()
        h=f.new_history();h.append(o);tree=ActionTree(o);prefix={}
        with f.decision(h):
            for _ in range(4):
                p=f.panel(h,tree,prefix);assert p['choices']==tree.choices(prefix)
                assert len(p['attribution_context'])==len(f.layout['context_groups'])
                assert p['attribution_candidates'].shape[1]==len(f.layout['candidate_groups'])
                assert len(p['proposals'])==1
                prefix=p['choices'][0]
        old=f.members[0].worker.history_hash
        f.panel(h,tree,{})
        assert f.members[0].worker.history_hash==old
        assert f.layout['members']==['one','two']
    finally:f.close()

def test_geometry_only_projection_does_not_need_author_execution():
    from experiments.planar_tasks.semantic_rl.library_features import GeometryOnlyLibraryFeatures
    from experiments.planar_tasks.semantic_rl.input_attribution_v2 import Scorer
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    groups={'candidate':{'duration_match':'A'},'state':{'contact_fraction':'D','blocked_fraction':'D','observed_entities':'D'}}
    es=[dict(name=n,source=source,groups=groups) for n in ['one','two']]
    f=LibraryFeatures(es,protocol(EventConfig(mode='duration')))
    try:
        scene,boxes=X.fixture(generate()['cases'][0]['fixture']);o=X.RecordedEnv(scene,boxes).public();h=f.new_history();h.append(o);tree=ActionTree(o)
        p=f.panel(h,tree,{});g=GeometryOnlyLibraryFeatures(es,f.action_protocol,f.layout);q=g.panel(h,tree,{})
        model=Scorer(f.layout,'G',(np.zeros(len(p['attribution_context'])),np.ones(len(p['attribution_context']))),(np.zeros(p['attribution_candidates'].shape[1]),np.ones(p['attribution_candidates'].shape[1])),f.sha256,f.action_protocol)
        for key,kind in [('attribution_context','context'),('attribution_candidates','candidate')]:
            np.testing.assert_array_equal(model.normalize(p[key],kind).numpy(),model.normalize(q[key],kind).numpy())
        assert p['choices']==q['choices'] and f.sha256==g.sha256 and g.calls==0
    finally:f.close()

def test_geometry_only_rollout_protocol():
    from experiments.planar_tasks.semantic_rl.library_features import GeometryOnlyLibraryFeatures
    from experiments.planar_tasks.semantic_rl.input_attribution_v2 import Scorer
    from experiments.planar_tasks.semantic_rl.runner import rollout
    source=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()
    groups={'candidate':{'duration_match':'A'},'state':{'contact_fraction':'D','blocked_fraction':'D','observed_entities':'D'}}
    es=[dict(name='neutral',source=source,groups=groups)];cfg=EventConfig(mode='duration');f=LibraryFeatures(es,protocol(cfg))
    fixture=generate()['cases'][0]['fixture']
    try:
        scene,boxes=X.fixture(fixture);o=X.RecordedEnv(scene,boxes).public();h=f.new_history();h.append(o);p=f.panel(h,ActionTree(o),{})
        g=GeometryOnlyLibraryFeatures(es,f.action_protocol,f.layout)
        model=Scorer(f.layout,'G',(np.zeros(len(p['attribution_context'])),np.ones(len(p['attribution_context']))),(np.zeros(p['attribution_candidates'].shape[1]),np.ones(p['attribution_candidates'].shape[1])),f.sha256,f.action_protocol)
        result,_=rollout(fixture,g.source,model=model,features=g,config=cfg,max_ticks=1,max_decisions=1,deterministic=True,fast_feedback=True)
        assert result['summary']['ticks']==1 and result['summary']['error'] is None
        assert result['semantic_protocol']['geometry_only_projection']
    finally:f.close()
