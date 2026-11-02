"""Identical episodes, masks, log-probs and values under ordered spawn sampling."""
import numpy as np
import torch
from experiments.option_rl_v10 import parallel,scenes,train
from experiments.option_rl_v10.interface import Policy,make_env,observe
from experiments.option_rl_v45.train import RunningMeanStd


def test_serial_and_spawn_workers_are_identical_and_frozen():
    torch.set_num_threads(1);torch.manual_seed(9)
    instances=scenes.panel(3,2,'train');dim=len(observe(make_env(instances[0]))[0])
    policy=Policy(dim,16);stats=RunningMeanStd(dim);snap=parallel.snapshot(policy,stats)
    jobs=[dict(instance=i,snapshot=snap,seed=parallel.behaviour_seed(2,i.index),collect_raw=True,
               diagnostic_time_budget=.6) for i in instances]
    rng_before=torch.get_rng_state().clone()
    serial=parallel.collect(None,jobs)
    assert torch.equal(rng_before,torch.get_rng_state())
    with parallel.pool(2) as executor:
        spawned=parallel.collect(executor,jobs);repeated=parallel.collect(executor,jobs)
    for a,b,c in zip(serial,spawned,repeated):
        for key in ('name','scene_hash','return','success','progress','periods','actions','policy_snapshot'):
            assert a['summary'][key]==b['summary'][key]==c['summary'][key]
        assert np.array_equal(a['raw_observations'],b['raw_observations'])
        for x,y,z in zip(a['steps'],b['steps'],c['steps']):
            assert x.a0==y.a0==z.a0 and x.a1==y.a1==z.a1
            assert x.logp==y.logp==z.logp and x.value==y.value==z.value
            assert np.array_equal(x.obs,y.obs) and np.array_equal(x.mask0,y.mask0) and np.array_equal(x.mask1,y.mask1)
    assert parallel.snapshot(policy,stats)['hash']==snap['hash'] and stats.count==1e-4
    data=[s for r in spawned for s in r['steps']]
    report=train.ppo_update(policy,torch.optim.Adam(policy.parameters(),lr=1e-4),data,epochs=1)
    assert report['initial_logp_max_error']<2e-5
