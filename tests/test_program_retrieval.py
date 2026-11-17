from contextlib import contextmanager
from types import SimpleNamespace
import numpy as np
import pytest

from experiments.planar_tasks.semantic_rl import program_retrieval as R
from experiments.planar_tasks.authoring_v2.program_retrieval_baselines import sample_boundaries,select_roots


def test_initial_outcomes_unknown_and_failed_neighbors():
    index=R.RetrievalIndex([[0],[10]],[[0,np.nan,1],[1,0,0]],['a','b','c'],{},evidence='initial_outcomes')
    assert index.select([0])['cap']=='c'
    assert index.select([0])['scores'][1] is None
    assert index.select([10])['cap']=='a'
    failed=R.RetrievalIndex([[0],[10]],[[0,0],[0,1]],['a','b'],{},evidence='initial_outcomes')
    assert failed.select([0])['fallback']
    assert failed.select([0])['cap']=='b'


def test_prefix_votes_unknown_is_not_failure_and_ties_stable():
    index=R.RetrievalIndex([[0],[1],[2]],[[1,np.nan],[np.nan,1],[np.nan,1]],['a','b'],{},evidence='successful_prefix',k=3)
    assert index.select([1])['cap']=='b'
    assert index.select([1])['score_kind']=='success_vote'
    tied=R.RetrievalIndex([[0],[0]],[[1,np.nan],[np.nan,1]],['a','b'],{},evidence='successful_prefix')
    assert tied.select([0])['cap']=='a'


def test_missing_features_roundtrip_and_tamper_detection(tmp_path):
    index=R.RetrievalIndex([[np.nan,0],[1,np.nan]],[[1,0],[0,1]],['a','b'],{},evidence='initial_outcomes')
    assert np.isfinite(index.encoded).all()
    dest=tmp_path/'index';index.save(dest)
    copy=R.RetrievalIndex.load(dest)
    assert copy.select([np.nan,0])==index.select([np.nan,0])
    with pytest.raises(ValueError):index.select([np.inf,0])
    with (dest/'arrays.npz').open('ab') as f:f.write(b'tamper')
    with pytest.raises(ValueError,match='integrity'):R.RetrievalIndex.load(dest)


def test_no_test_labels_and_bounded_balanced_prefix_sampling():
    with pytest.raises(ValueError,match='development only'):
        select_roots([dict(split='fresh_test',movable_count=2,case_id='secret')])
    rows=[dict(split='newdev',movable_count=n,case_id=f'{n}_{j}') for n in [2,3] for j in range(3)]
    assert select_roots(rows,4)==['2_0','3_0','2_1','3_1']
    assert sample_boundaries(2,16)==[0,1]
    assert len(sample_boundaries(100,16))==16
    assert sample_boundaries(100,16)[-1]==99


@pytest.mark.parametrize('mode',['initial','boundary'])
def test_routing_memory_tracks_real_actions_and_initial_is_locked(monkeypatch,mode):
    class Member:
        def __init__(self,source,**kwargs):
            self.source=source;self.seen=[];self.worker=self
        @contextmanager
        def decision(self,h):
            if not self.seen or self.seen[-1][0]!=h.index:self.seen.append((h.index,h.actual))
            yield
        def recommendation(self):
            return dict(first_legal=True,raw_first=dict(source=self.source,actual=self.seen[-1][1]))
        def close(self):pass
    class Encoder:
        def __init__(self,*args):pass
        def advance(self,h):return np.array([h.index])
    class Router:
        names=['a','b'];representation='snapshot';identity='test'
        metadata=dict(sources={'a':R.digest('A'),'b':R.digest('B')},action_protocol={})
        def select(self,q):return dict(cap='a' if q[0]==0 else 'b')
    monkeypatch.setattr(R,'Features',Member);monkeypatch.setattr(R,'GeometryEncoder',Encoder)
    f=R.RoutedFeatures([dict(name='a',source='A'),dict(name='b',source='B')],Router(),{},mode=mode)
    h=SimpleNamespace(episode='one',index=0,sha256='0',actual=None)
    assert f.recommendation(h)['raw_first']['source']=='A'
    h.index=1;h.sha256='1';h.actual='EXTERNAL_EXECUTED_ACTION'
    packet=f.recommendation(h)
    assert packet['raw_first']['source']==('A' if mode=='initial' else 'B')
    assert packet['raw_first']['actual']=='EXTERNAL_EXECUTED_ACTION'
    assert all(m.seen==[(0,None),(1,'EXTERNAL_EXECUTED_ACTION')] for m in f.members)
    assert f.trace[-1]['switched']==(mode=='boundary')
    h.index=3
    with pytest.raises(ValueError,match='skipped'):f.recommendation(h)
