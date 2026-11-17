import numpy as np
import pytest
import torch
from experiments.planar_tasks.semantic_rl import gd_attribution as GD
from experiments.planar_tasks.semantic_rl import library_learning as L
from experiments.planar_tasks.semantic_rl.input_attribution_v2 import Scorer


def make(cls=GD.GDScorer, arm='GD'):
    layout=dict(context_groups=['G','D','A','common'],candidate_groups=['D','A','G','common'])
    stats=(np.zeros(4),np.ones(4))
    torch.manual_seed(17)
    return cls(layout,arm,stats,stats,'source',{})


def test_identical_parameters_and_gd_mask():
    gd=make();full=make(Scorer,'GDA')
    for a,b in zip(gd.parameters(),full.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert gd.context_mask.tolist()==[1,1,0,1]
    assert gd.candidate_mask.tolist()==[1,0,1,1]
    x=np.arange(8).reshape(2,4).astype(float)
    assert GD.verify_A_mask(gd,x,x)['A_value_and_missingness_invariant']
    before=gd.normalize(x,'context');x[:,1]+=1
    assert not torch.equal(before,gd.normalize(x,'context'))


def test_checkpoint_roundtrip(tmp_path):
    m=make();p=tmp_path/'model.pt';m.save(p,{})
    loaded=GD.GDScorer.load(p)
    assert loaded.arm=='GD'
    for k,v in m.state_dict().items():
        torch.testing.assert_close(v,loaded.state_dict()[k],rtol=0,atol=0)


def test_training_and_constructor_restore(monkeypatch):
    m=make();old=L.Scorer
    ds=dict(contexts=np.ones((1,4)),candidates=np.tile(np.arange(8).reshape(2,4),(4,1)),
            offsets=np.array([[0,2,4,6,8]]),targets=np.zeros((1,4),dtype=int))
    trained,curve=GD.train(ds,[0],m.layout,'GD','source',{},updates=2)
    assert trained.arm=='GD' and curve[-1]['update']==2 and L.Scorer is old
    def fail(*args,**kwargs):
        assert L.Scorer is GD.GDScorer
        raise RuntimeError('expected')
    monkeypatch.setattr(L,'train',fail)
    with pytest.raises(RuntimeError,match='expected'):GD.train()
    assert L.Scorer is old
