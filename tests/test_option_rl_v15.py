"""Joint correlation without changing initialization, legality or costs."""
import numpy as np
import pytest
import torch
from experiments.option_rl_v14.interface import Policy as Independent
from experiments.option_rl_v15.interface import Policy,ACTIONS


def test_coupled_and_independent_start_exactly_equal_including_rng():
    torch.set_num_threads(1);torch.manual_seed(4);a=Independent(944,32)
    rng_a=torch.get_rng_state().clone()
    torch.manual_seed(4);b=Policy(944,32,coupled=True)
    assert torch.equal(rng_a,torch.get_rng_state())
    x=torch.randn(7,944);x[:,760+2]=1
    masks=[np.ones((7,46),bool) for _ in (0,1)];masks[0][0,:32]=False;masks[1][1,:32]=False
    for a0 in [None,torch.full((7,),44)]:
        da,va=a.distributions(x,masks,a0=a0);db,vb=b.distributions(x,masks,a0=a0)
        torch.testing.assert_close(va,vb,rtol=0,atol=0)
        for u,v in zip(da,db):torch.testing.assert_close(u.probs,v.probs,rtol=0,atol=0)


def test_joint_scores_can_correlate_brakes_and_preserve_scale_ratios():
    p=Policy(944,16,coupled=True);x=torch.zeros(1,944);m=[np.ones(46,bool)]*2
    with torch.no_grad():p.joint_types.bias[12*14+12]=4.
    a,_=p.distributions(x,m,a0=torch.tensor([44]));b,_=p.distributions(x,m,a0=torch.tensor([8]))
    assert a[1].probs[0,44]>b[1].probs[0,44]*5
    a_ratio=a[1].probs[0,8:12]/a[1].probs[0,8:12].sum()
    b_ratio=b[1].probs[0,8:12]/b[1].probs[0,8:12].sum()
    torch.testing.assert_close(a_ratio,b_ratio)


def test_exact_joint_entropy_logprob_and_finite_coupling_gradient():
    torch.manual_seed(7);p=Policy(944,16,coupled=True)
    with torch.no_grad():p.joint_types.weight.normal_(0,.1)
    x=torch.randn(4,944);m=[np.ones((4,46),bool)]*2
    ds,_=p.distributions(x,m);joint=[]
    for k in range(46):
        d,_=p.distributions(x,m,a0=torch.full((4,),k));joint.append(ds[0].probs[:,k,None]*d[1].probs)
    joint=torch.stack(joint,1)
    torch.testing.assert_close(joint.sum((1,2)),torch.ones(4))
    assert (joint[:,~p.legal_pairs]==0).all()
    entropy=-(joint*joint.clamp_min(1e-30).log()).sum((1,2))
    torch.testing.assert_close(entropy,sum(d.entropy() for d in ds),atol=3e-6,rtol=1e-5)
    actions,logp,_=p.act(x,m,generator=torch.Generator().manual_seed(2))
    torch.testing.assert_close(logp,joint[torch.arange(4),actions[0],actions[1]].log())
    (-logp.mean()-.02*entropy.mean()).backward()
    assert torch.isfinite(p.joint_types.weight.grad).all() and p.joint_types.weight.grad.norm()>0
