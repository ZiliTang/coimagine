import copy
import math
import shutil

import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import assemble, Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_learner import checkpoint, train_online_round
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.long_online import PartitionedStore, behavior_entropy, train_round
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel


def shard(path, n, offset):
    rng = np.random.default_rng(offset)
    arrays = dict(x=rng.normal(size=(n,12)).astype(np.float32),
        candidates=rng.normal(size=(n,7,85)).astype(np.float32), sources=np.zeros((n,7,7),np.float32),
        valid=np.ones((n,7),bool), chosen=np.arange(n)%7, duration=np.ones(n,np.int64),
        y=np.ones(n,np.float32), cap=np.zeros(n,np.int64), root=np.full(n,offset,np.int64),
        boundary=np.arange(n,dtype=np.int32), reward=np.zeros(n,np.float32), done=np.zeros(n,bool))
    arrays['x'][0,7] = np.nan
    arrays['reward'][-1] = 1.
    arrays['done'][-1] = True
    np.savez(path,**arrays)
    return dict(path=str(path),boundaries=n)


def test_partitioned_replay_preserves_order_and_episode_next_states(tmp_path):
    shards = [shard(tmp_path/'a.npz',3,1), shard(tmp_path/'b.npz',5,2)]
    assemble(tmp_path/'whole',shards)
    assemble(tmp_path/'a',shards[:1])
    assemble(tmp_path/'b',shards[1:])
    whole = Store(tmp_path/'whole')
    blocks = PartitionedStore([tmp_path/'a',tmp_path/'b'])
    ids = np.array([7,0,2,3,2,6,1,4,5])
    for following in (False,True):
        for key,expected in whole.batch(ids,following).items():
            np.testing.assert_array_equal(blocks.batch(ids,following)[key],expected)
    for key in ('reward','done','chosen'):
        np.testing.assert_array_equal(blocks.arrays[key],whole.arrays[key])


def test_entropy_matches_actual_epsilon_greedy_distribution():
    assert behavior_entropy(1,.05) == 0
    assert behavior_entropy(7,0) == 0
    assert abs(behavior_entropy(7,1)-math.log(7)) < 1e-12
    probabilities = np.array([.96,.01,.01,.01,.01])
    assert abs(behavior_entropy(5,.05)+np.sum(probabilities*np.log(probabilities))) < 1e-12


def test_instrumented_resume_matches_original_update_exactly(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(5)
    parent, baseline, new = [tmp_path/name for name in ('parent','baseline','new')]
    a,b = shard(tmp_path/'a.npz',7,4), shard(tmp_path/'b.npz',9,6)
    assemble(parent/'arrays/train',[a,b])
    assemble(tmp_path/'a',[a])
    assemble(tmp_path/'b',[b])
    assemble(tmp_path/'whole',[a,b])
    shutil.copytree(parent/'arrays',baseline/'arrays')
    layout = dict(width=12,trace_width=7,trace_widths=[1]*7)
    q = QModel(layout,np.zeros(12),np.ones(12),'GLOBAL',np.zeros(85),np.ones(85))
    opt = torch.optim.Adam(q.parameters(),lr=3e-4)
    # Populate Adam moments so the check covers a real optimizer continuation.
    loss = sum(p.square().sum() for p in q.parameters())
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)
    target = copy.deepcopy(q)
    rng = np.random.default_rng(17)
    rng.integers(20,size=9)
    path = parent/'models/GLOBAL_s17/online_r4.pt'
    checkpoint(path,q,opt,17,1500,'Q',rng,online_updates=6000,target_model=target.state_dict())
    other = baseline/'models/GLOBAL_s17/online_r4.pt'
    other.parent.mkdir(parents=True)
    shutil.copyfile(path,other)
    train_online_round(baseline,'GLOBAL',17,5,tmp_path/'whole',updates=1)
    train_round(parent,new,17,5,[tmp_path/'a',tmp_path/'b'],updates=1)
    old = torch.load(baseline/'models/GLOBAL_s17/online_r5.pt',weights_only=False)
    actual = torch.load(new/'models/GLOBAL_s17/online_r5.pt',weights_only=False)
    def same(a,b):
        if isinstance(a,torch.Tensor):
            assert torch.equal(a,b)
        elif isinstance(a,dict):
            assert a.keys() == b.keys()
            for k in a:same(a[k],b[k])
        elif isinstance(a,(list,tuple)):
            assert len(a) == len(b)
            for v,w in zip(a,b):same(v,w)
        else:
            assert a == b
    for key in ('model','optimizer','target_model','numpy_rng','torch_rng','online_updates'):
        same(old[key],actual[key])
