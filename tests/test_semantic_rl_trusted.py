from pathlib import Path
from copy import deepcopy
import pytest
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree,FIELDS
from experiments.planar_tasks.semantic_rl.environment import Environment
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.relational_search.contract import digest,canonical
SOURCE=Path('experiments/planar_tasks/semantic_rl/stream_example.py').read_text()


def test_trusted_backend_is_opt_in_and_source_pinned():
    assert Features(SOURCE).execution_backend=='isolated'
    with pytest.raises(ValueError,match='digest'):
        Features(SOURCE,execution_backend='trusted_inprocess')
    with pytest.raises(ValueError,match='digest'):
        Features(SOURCE,execution_backend='trusted_inprocess',trusted_source_sha256='wrong')


def test_host_native_panels_and_memory_match_isolation():
    a=Features(SOURCE);b=Features(SOURCE,execution_backend='trusted_inprocess',trusted_source_sha256=digest(SOURCE))
    ha=a.new_history();hb=b.new_history();o=Environment(seed_fixture()).history.events[0]['observation']
    try:
        for i in range(4):
            obs=deepcopy(o);obs['tick']=i
            for e in obs['entities']:e['last_seen_tick']=i
            action=None if i==0 else dict(target_id=None,a0=44,a1=44,ticks=1)
            ha.append(obs,action);hb.append(obs,action);tree=ActionTree(obs)
            with a.decision(ha),b.decision(hb):
                prefix={}
                for k in FIELDS:
                    x=a.panel(ha,tree,prefix);y=b.panel(hb,tree,prefix)
                    assert canonical(x)==canonical(y)
                    prefix=x['choices'][0]
            assert canonical(a.worker.memory)==canonical(b.worker.memory)
        assert b.worker.process is None and b.worker.compilation['host_soabi'].startswith('cpython-311')
        assert a.calls==b.calls==20
        snap=b.snapshot();b.restore(snap,hb)
        assert canonical(a.recommendation(ha))==canonical(b.recommendation(hb))
    finally:a.close();b.close()
