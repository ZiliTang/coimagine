import copy
import shutil

import numpy as np
import torch

from test_cap_trace_value_long import shard
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import assemble, Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_learner import checkpoint
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.long_online import PartitionedStore
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import ReplayCache
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.fast_learner import train_round
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.semantic_rl.environment import Environment
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.fast_runtime import AuditedCachedHistory
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory


def test_cached_history_matches_public_projection_and_fork():
    observation = Environment(seed_fixture()).env.public()
    observation['private_extension'] = 123
    reference, cached = StreamHistory(), AuditedCachedHistory()
    for history in (reference, cached):
        history.append(observation)
    observation['tick'] = 1
    assert cached.sha256 == reference.sha256
    assert cached.events == reference.events
    assert 'private_extension' not in cached.events[0]['observation']

    fork = cached.fork()
    action = dict(target_id=None, a0=44, a1=44, ticks=1)
    for history in (reference, cached, fork):
        history.append(observation, action)
    assert fork.sha256 == cached.sha256 == reference.sha256
    assert fork.events == cached.events == reference.events


def assert_same(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_same(x,y)
    else:assert a==b


def test_resident_replay_keeps_nan_payload_signed_zero_and_next_index(tmp_path):
    rows=[shard(tmp_path/'a.npz',7,1),shard(tmp_path/'b.npz',5,2)]
    assemble(tmp_path/'source',rows)
    x=np.load(tmp_path/'source/x.npy',mmap_mode='r+')
    x.view(np.uint32)[0,:5]=[0x80000000,0x7fc01234,0xffc00001,0x00000001,0x7f800000]
    x.flush()
    original=Store(tmp_path/'source')
    packed=ReplayCache(tmp_path/'packed').get(tmp_path/'source')
    indices=np.array([0,6,7,11,0,3])
    for next_state in (False,True):
        a,b=original.batch(indices,next_state),packed.batch(indices,next_state)
        for key in a:assert a[key].tobytes()==b[key].tobytes()


def test_resident_multithread_resume_matches_mmap_across_target_sync(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rows=[shard(tmp_path/'a.npz',7,4),shard(tmp_path/'b.npz',9,6)]
    assemble(tmp_path/'source',rows)
    layout=dict(width=12,trace_width=7,trace_widths=[1]*7)
    q=QModel(layout,np.zeros(12),np.ones(12),'GLOBAL',np.zeros(85),np.ones(85))
    optimizer=torch.optim.Adam(q.parameters(),lr=3e-4)
    target=copy.deepcopy(q)
    previous=tmp_path/'previous.pt'
    checkpoint(previous,q,optimizer,17,247,'Q',np.random.default_rng(17),
               online_updates=12247,target_model=target.state_dict())
    train_round(previous,tmp_path/'mmap',17,9,Store(tmp_path/'source'),
                PartitionedStore([tmp_path/'source']),updates=9)
    cache=ReplayCache(tmp_path/'packed')
    torch.set_num_threads(4)
    try:
        train_round(previous,tmp_path/'resident',17,9,cache.get(tmp_path/'source'),
                    cache.online([tmp_path/'source']),updates=9)
    finally:torch.set_num_threads(1)
    a=torch.load(tmp_path/'mmap/models/GLOBAL_s17/online_r9.pt',weights_only=False)
    b=torch.load(tmp_path/'resident/models/GLOBAL_s17/online_r9.pt',weights_only=False)
    for key in ('model','optimizer','target_model','numpy_rng','torch_rng','online_updates'):
        assert_same(a[key],b[key])
