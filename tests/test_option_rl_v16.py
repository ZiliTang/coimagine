"""Exact continuation and developer retention bookkeeping."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from experiments.option_rl_v16 import train as T
from experiments.option_rl_v16.retention import update


def test_resume_matches_uninterrupted_training_model_optimizer_stats_and_rng(tmp_path,monkeypatch):
    original=T.make_env
    def short(instance,**kwargs):
        env=original(instance,**kwargs);env.config=replace(env.config,time_budget=.6);return env
    monkeypatch.setattr(T,'make_env',short)
    kw=dict(episodes=1,workers=1,hidden=16,epochs=1,eval_roots=0,hours=.02,seed=3,coupled=True)
    T.train(out=tmp_path/'whole',iterations=3,**kw)
    T.train(out=tmp_path/'first',iterations=1,**kw)
    T.train(out=tmp_path/'resumed',iterations=2,resume_from=tmp_path/'first/policy.pt',**kw)
    a=torch.load(tmp_path/'whole/policy.pt',weights_only=False);b=torch.load(tmp_path/'resumed/policy.pt',weights_only=False)
    for name in a['state_dict']:torch.testing.assert_close(a['state_dict'][name],b['state_dict'][name],rtol=0,atol=0)
    for kind in a['optimizer']:
        for idx,state in a['optimizer'][kind]['state'].items():
            for key,value in state.items():torch.testing.assert_close(value,b['optimizer'][kind]['state'][idx][key],rtol=0,atol=0)
    np.testing.assert_array_equal(a['obs_mean'],b['obs_mean']);np.testing.assert_array_equal(a['obs_var'],b['obs_var'])
    assert a['obs_count']==b['obs_count'] and a['next_scene_ordinal']==b['next_scene_ordinal']==3
    assert torch.equal(a['torch_rng'],b['torch_rng']) and torch.equal(a['sampling_rng'],b['sampling_rng'])
    assert [r['iteration'] for r in b['history']]==[1,2,3]
    assert (tmp_path/'resumed/snapshots/iteration_000001.pt').exists()
    # An earlier snapshot must not inherit later evaluation evidence stored in
    # its parent directory, even though all files are physically available.
    (tmp_path/'resumed/evaluations/iteration_999999.json').write_text('{"development_training":{"roots":[]}}')
    T.train(out=tmp_path/'earlier',iterations=2,resume_from=tmp_path/'resumed/snapshots/iteration_000001.pt',**kw)
    assert not (tmp_path/'earlier/evaluations/iteration_999999.json').exists()
    c=torch.load(tmp_path/'earlier/policy.pt',weights_only=False)
    for name in a['state_dict']:torch.testing.assert_close(a['state_dict'][name],c['state_dict'][name],rtol=0,atol=0)


def test_resume_rejects_training_change_before_sampling(tmp_path):
    # Reuse a real tiny checkpoint through a separate fixture would duplicate a
    # rollout; constructor-level model mismatch fails before source/scene work.
    p=T.Policy(944,16,coupled=True)
    data=dict(T.metadata(),model_config=p.model_config,rollout_config=dict(context='extended',hold_seconds=.8),
        hidden=16,history=[dict(iteration=1)])
    source=tmp_path/'parent.pt';torch.save(data,source)
    with pytest.raises(ValueError,match='architecture/rollout'):
        T.train(out=tmp_path/'bad',iterations=1,resume_from=source,coupled=False,hidden=16)
    assert not (tmp_path/'bad').exists()


def test_retention_counts_loss_regain_and_task_groups_once(tmp_path):
    def evaluation(a,b):
        return dict(development_training=dict(roots=[dict(name='contact_0',success=a,progress=float(a)),dict(name='dual_0',success=b,progress=float(b))]))
    r=update(tmp_path,40,evaluation(True,False));assert r['retention_gained']==1
    r=update(tmp_path,45,evaluation(False,True));assert r['retention_lost']==1 and r['retention_gained']==1
    assert r['dev_single_success']==0 and r['dev_dual_success']==1
    r=update(tmp_path,45,evaluation(False,True));assert r['retention_lost']==r['retention_gained']==0
    r=update(tmp_path,50,evaluation(True,True));assert r['retention_gained']==1
