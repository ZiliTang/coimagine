import numpy as np
import pytest
import torch
from experiments.planar_tasks.semantic_rl import input_attribution as V1
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as A


def model(arm,norm=True):
    layout=dict(context_groups=['G','D','A','common'],candidate_groups=['D','A','G','common'],version=A.VERSION)
    stats=(np.zeros(4),np.ones(4));torch.manual_seed(17)
    return A.Scorer(layout,arm,stats,stats,'source',{'unit':'ticks'},projection_norm=norm)


@pytest.mark.parametrize('arm',A.ARMS)
def test_mask_blocks_removed_values_and_missingness(arm):
    m=model(arm);c=np.array([1.,2.,3.,4.]);x=np.array([[1.,2.,3.,4.],[4.,3.,2.,1.]])
    before=m.logits(c,x)
    for i,g in enumerate(m.layout['context_groups']):
        if g!='common' and g not in arm:c[i]=np.nan
    for i,g in enumerate(m.layout['candidate_groups']):
        if g!='common' and g not in arm:x[:,i]=[np.nan,1e8]
    torch.testing.assert_close(before,m.logits(c,x),rtol=0,atol=0)


@pytest.mark.parametrize('norm',[True,False])
def test_save_load_and_factored_gradients(tmp_path,norm):
    m=model('GA',norm);p=tmp_path/'m.pt';m.save(p,{'seed':17});n=A.Scorer.load(p)
    c=torch.randn(2,8);x=torch.randn(5,8);owners=torch.tensor([0,0,1,1,1])
    y=m.logits(c,x,owners,normalized=True)
    torch.testing.assert_close(y,n.logits(c,x,owners,normalized=True),rtol=0,atol=0)
    expected=m.tail(m.context_norm(m.context_linear(c[owners]))+m.candidate_norm(m.candidate_linear(x))).flatten()
    torch.testing.assert_close(y,expected,atol=1e-6,rtol=1e-6)
    first=torch.autograd.grad(y.sum(),list(m.parameters()),retain_graph=True)
    second=torch.autograd.grad(expected.sum(),list(m.parameters()))
    for a,b in zip(first,second):torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-6)


def test_duration_encoding_complete_support_and_no_unrevealed_values():
    x=np.zeros((41,248));x[1:,86]=1;x[1:,87]=np.arange(1,41)*.05
    x[0,87]=123. # unrevealed representative must not leak
    y=A.append_duration_encoding(x)
    np.testing.assert_array_equal(y[:,:248],x)
    np.testing.assert_array_equal(y[1:,248:],np.eye(40));assert not y[0,248:].any()
    bad=x.copy();bad[1,87]=2.05
    with pytest.raises(ValueError,match='outside'):A.append_duration_encoding(bad)


def test_disabled_repairs_reproduce_v1():
    m=model('GDA',False);torch.manual_seed(17)
    s=(np.zeros(4),np.ones(4));old=V1.Scorer(m.layout,'GDA',s,s,'source',{'unit':'ticks'})
    c=np.arange(4);x=np.arange(12).reshape(3,4)
    torch.testing.assert_close(m.logits(c,x),old.logits(c,x),rtol=0,atol=0)


def test_fixed_projection_norm_controls_large_scale():
    m=model('G');c=torch.randn(8,8);x=torch.randn(8,8)
    with torch.no_grad():m.context_linear.weight.mul_(1e5)
    z=m.context_norm(m.context_linear(c))+m.candidate_norm(m.candidate_linear(x))
    assert z.square().mean()<4.01
    assert float((z.abs()>3).float().mean())<.15


def test_training_normalization_does_not_use_other_rows():
    m=model('G');ds=dict(contexts=np.array([[1.,2.,3.,4.],[1e9]*4]),candidates=np.tile(np.array([[0.,1.,2.,3.],[1.,2.,3.,4.]]),(8,1)),offsets=np.array([[0,2,4,6,8],[8,10,12,14,16]]),targets=np.zeros((2,4),dtype=np.int64))
    trained,curve=A.train(ds,[0],m.layout,'G','source',{},updates=2)
    np.testing.assert_array_equal(trained.context_mean.numpy(),[1,2,3,4]);assert curve[-1]['update']==2
