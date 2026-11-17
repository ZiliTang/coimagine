import numpy as np
import pytest
import torch
from experiments.planar_tasks.semantic_rl import input_attribution as A


def model(arm):
    layout=dict(context_groups=['G','D','A','common'],candidate_groups=['D','A','G','common'],version=A.VERSION)
    stats=(np.zeros(4),np.ones(4));torch.manual_seed(17)
    return A.Scorer(layout,arm,stats,stats,'source',{'unit':'ticks'})


@pytest.mark.parametrize('arm',A.ARMS)
def test_mask_blocks_value_and_missingness(arm):
    m=model(arm);c=np.array([1.,2.,3.,4.]);x=np.array([[1.,2.,3.,4.],[4.,3.,2.,1.]])
    before=m.logits(c,x)
    for i,g in enumerate(m.layout['context_groups']):
        if g!='common' and g not in arm:c[i]=np.nan
    for i,g in enumerate(m.layout['candidate_groups']):
        if g!='common' and g not in arm:x[:,i]=[np.nan,1e8]
    torch.testing.assert_close(before,m.logits(c,x),rtol=0,atol=0)


def test_factorized_affine_matches_concatenation_values_and_gradients():
    m=model('GDA');c=torch.randn(2,8);x=torch.randn(5,8);owners=torch.tensor([0,0,1,1,1])
    y=m.logits(c,x,owners,normalized=True)
    weight=torch.cat([m.context_linear.weight,m.candidate_linear.weight],1)
    expected=m.tail(torch.nn.functional.linear(torch.cat([c[owners],x],1),weight,m.candidate_linear.bias)).flatten()
    torch.testing.assert_close(y,expected,atol=1e-6,rtol=1e-6)
    first=torch.autograd.grad(y.sum(),list(m.parameters()),retain_graph=True)
    second=torch.autograd.grad(expected.sum(),list(m.parameters()))
    for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)


def test_save_load_preserves_mask_and_logits(tmp_path):
    m=model('GA');p=tmp_path/'m.pt';m.save(p,{'seed':17});n=A.Scorer.load(p)
    c=np.array([1.,None,3.,4.]);x=np.array([[1.,2.,3.,4.],[4.,3.,2.,1.]])
    torch.testing.assert_close(m.logits(c,x),n.logits(c,x),rtol=0,atol=0)


def test_geometry_history_incremental_and_episode_reset(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(A,'current_event',lambda h:h.event)
    monkeypatch.setattr(A,'observation_vector',lambda e:np.array(e,dtype=float))
    s=A.GeometryHistory();h=SimpleNamespace(episode='a',index=0,sha256='a0',event=[1,2])
    s.advance(h);v=s.vector.copy();s.advance(h);np.testing.assert_array_equal(v,s.vector)
    h.index=1;h.sha256='a1';h.event=[3,4];s.advance(h)
    np.testing.assert_array_equal(s.vector[:4],[1,2,2,3])
    h.index=3
    with pytest.raises(ValueError,match='skipped'):s.advance(h)
    h.episode='b';h.index=0;h.sha256='b0';h.event=[5,6];s.advance(h)
    np.testing.assert_array_equal(s.vector[:4],[5,6,5,6])


def test_training_uses_only_selected_boundary_normalization():
    layout=dict(context_groups=['G','D','A','common'],candidate_groups=['D','A','G','common'],version=A.VERSION)
    ds=dict(contexts=np.array([[1.,2.,3.,4.],[1e9]*4]),candidates=np.tile(np.array([[0.,1.,2.,3.],[1.,2.,3.,4.]]),(8,1)),offsets=np.array([[0,2,4,6,8],[8,10,12,14,16]]),targets=np.zeros((2,4),dtype=np.int64))
    m,curve=A.train(ds,[0],layout,'G','source',{},updates=2)
    np.testing.assert_array_equal(m.context_mean.numpy(),[1,2,3,4]);assert curve[-1]['update']==2
