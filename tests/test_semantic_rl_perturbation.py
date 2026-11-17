from collections import Counter
from copy import deepcopy
from pathlib import Path
import pytest
from experiments.planar_tasks.authoring_v2.semantic_rl_perturbation import make_panels,train_schedule,PerturbationCampaign
from experiments.planar_tasks.relational_search.pipeline import save,read,Campaign,unpack
from experiments.planar_tasks.relational_search.contract import digest


def test_panels_reproducible_disjoint_balanced_and_geometry_only():
    panels=make_panels();assert panels==make_panels()
    hashes=[]
    for split in ('train','test'):
        rows=panels[split]['cases'];assert len(rows)==12
        assert set(Counter(r['family'] for r in rows).values())=={4}
        assert Counter(r['level'] for r in rows)=={'small':6,'medium':6}
        assert all(r['fixture_sha256']==digest(r['fixture']) for r in rows)
        hashes.extend(r['fixture_sha256'] for r in rows)
    assert len(set(hashes))==24
    order=train_schedule(panels['train']['cases'])
    assert len(order)==24 and set(Counter(order).values())=={2}
    assert not set(order)&{r['id'] for r in panels['test']['cases']}


def test_split_gate_rejects_test_relabeling_mutation_and_unsealed_evaluation(tmp_path):
    panels=make_panels()
    for name,p in panels.items():save(tmp_path/'panels'/f'{name}.json',p)
    c=object.__new__(PerturbationCampaign);c.path=tmp_path
    c.validate_case(panels['train']['cases'][0],True)
    held=panels['test']['cases'][0]
    with pytest.raises(ValueError,match='split'):c.validate_case(held,True)
    relabeled=deepcopy(held);relabeled['split']='train'
    with pytest.raises(ValueError,match='unregistered'):c.validate_case(relabeled,True)
    changed=deepcopy(panels['train']['cases'][0]);changed['fixture']['object']['x']+=.01
    with pytest.raises(ValueError,match='changed'):c.validate_case(changed,True)
    with pytest.raises(FileNotFoundError,match='evaluation_seal'):
        c.run_case(held,'cap')


def test_real_short_evaluation_settles_actual_ticks_and_keeps_failure(tmp_path,monkeypatch):
    from experiments.planar_tasks.authoring_v2 import semantic_rl_perturbation as module
    parent=Campaign.create(tmp_path/'parent')
    state=read(parent.path/'campaign.json');state['fixture_profile']='stage1-wall-anchor-v2';save(parent.path/'campaign.json',state)
    source='''NAMES=['constant']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=4)]
def semantic_features(m,o,a,p):return dict(constant=0.)
'''
    (parent.path/'semantics').mkdir();(parent.path/'semantics/s000.py').write_text(source)
    (parent.path/'semantic_fits/f000').mkdir(parents=True)
    # Campaign admission now validates the initialization's execution protocol.
    from experiments.planar_tasks.semantic_rl.features import Features
    from experiments.planar_tasks.semantic_rl.actions import ActionTree
    from experiments.planar_tasks.semantic_rl.policy import Policy
    from experiments.planar_tasks.semantic_rl.environment import Environment
    from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
    import torch
    torch.set_num_threads(1)
    h=Environment(seed_fixture()).history;features=Features(source)
    root=features.panel(h,ActionTree(h.events[-1]['observation']),{})
    Policy([root],features.sha256).save(parent.path/'semantic_fits/f000/policy.pt',{})
    parent.freeze()
    c=PerturbationCampaign.create(tmp_path/'child',parent.path)
    (c.path/'models').mkdir();(c.path/'models/final.pt').write_bytes(b'sealed fixture')
    save(c.path/'evaluation_seal.json',dict(checkpoint_sha256=module.sha(c.path/'models/final.pt')))
    monkeypatch.setattr(module,'MAX_TICKS',4);monkeypatch.setattr(module,'MAX_DECISIONS',1)
    case=read(c.path/'panels/test.json')['cases'][0]
    run,result,batch=c.run_case(case,'cap')
    state=read(c.path/'campaign.json')
    assert run['ticks']==4 and run['status']=='incomplete' and not batch
    assert result['comparison_cost_seconds']==180. and result['summary']['task_cost_seconds'] is None
    assert state['used']['execution_ticks']==4 and state['used']['feature_batches']==1
    assert not state['reservations']
    assert c.freeze()['status']=='verified'
