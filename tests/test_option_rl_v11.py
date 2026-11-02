import math
from dataclasses import replace
import numpy as np
import torch
from experiments.option_rl_v11 import interface as I,controller as C,scenes
from experiments.option_rl_v11.train import run_episode,ppo_update
from experiments.option_rl_v45.train import RunningMeanStd

def test_joint_follow_masks_and_exact_entropy_and_gradients():
 torch.set_num_threads(1);torch.manual_seed(1);p=I.Policy(8,32);x=torch.zeros(128,8);m=[np.ones(62,bool)]*2
 acts,lp,_=p.act(x,m,generator=torch.Generator().manual_seed(1))
 for a,b in zip(*[v.tolist() for v in acts]):
  assert I.ACTIONS[a].kind!='follow' or I.ACTIONS[b].kind=='push'
  assert I.ACTIONS[b].kind!='follow' or I.ACTIONS[a].kind=='push'
 ds,_=p.distributions(x,m,a0=acts[0]);torch.testing.assert_close(lp,ds[0].log_prob(acts[0])+ds[1].log_prob(acts[1]))
 probs=[]
 for a in range(62):
  d,_=p.distributions(x[:1],[np.ones(62,bool)]*2,a0=torch.tensor([a]));probs.append(d[0].probs[:,a,None]*d[1].probs)
 joint=torch.cat(probs,0).flatten();assert abs(float(joint.sum())-1)<1e-5
 expected=-(joint[joint>0]*joint[joint>0].log()).sum()
 torch.testing.assert_close((ds[0].entropy()+ds[1].entropy())[0],expected)
 # A peer with only WAIT/FOLLOW makes the FOLLOW row entirely invalid.
 masks=np.zeros((128,62),bool);masks[:,60:]=True
 ds,_=p.distributions(x,[np.ones(62,bool),masks],a0=torch.zeros(128,dtype=torch.long))
 loss=-ds[0].log_prob(torch.zeros(128,dtype=torch.long)).mean()-.02*sum(d.entropy().mean() for d in ds)
 loss.backward();assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)
 assert not ds[0].probs[:,61].any()

def test_fresh_episode_and_ppo_probability_reconstruction():
 torch.set_num_threads(1);torch.manual_seed(1);inst=scenes.scene(1,0);w=I.make_env(inst);p=I.Policy(len(I.observe(w)[0]),32);n=RunningMeanStd(len(I.observe(w)[0]))
 def short(i):
  e=I.make_env(i);e.config=replace(e.config,time_budget=8.);return e
 steps,s=run_episode(p,inst,n,generator=torch.Generator().manual_seed(9),update_stats=True,env_factory=short)
 assert s['initial_geometry']['legal'] and s['eligible_for_ppo']
 stats=ppo_update(p,torch.optim.Adam(p.parameters(),lr=3e-4),steps,epochs=1,minibatch=32)
 assert stats['initial_logp_max_error']<2e-5
 I.validate_metadata(I.metadata())

def test_new_profile_matches_recommended_single_push():
 from experiments.option_testbench.effect_push_eval import make_case,ACTIONS as OLD
 from experiments.option_testbench.follow_peer_eval import instance_from_dict
 c=make_case('single_N_0.3_offset_0.18',distance=.3,offset=.18)
 c['actions']=[next(a.index for a in I.ACTIONS if a.label==OLD[i].label) for i in c['actions']]
 inst=instance_from_dict(c['instance']);w=I.make_env(inst)
 w.joint_step(*c['actions'],record=True)
 assert w.termination is None
 assert w.rows[-1]['time']==4.35 or abs(w.rows[-1]['time']-4.35)<1e-8
 assert w.rows[-1]['r0']['exit_reason']=='normal_travel_complete'
 assert len(I.ACTIONS)==62 and C.CONFIG.speed==.32
