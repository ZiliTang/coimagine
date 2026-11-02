from dataclasses import replace
import numpy as np
import torch
from experiments.option_rl_v12 import interface as I,controller as C,scenes
from experiments.option_rl_v12.train import run_episode,ppo_update
from experiments.option_rl_v45.train import RunningMeanStd

def test_joint_mask_logprob_and_gradients_with_persistence_prior():
 torch.set_num_threads(1);torch.manual_seed(1);p=I.Policy(822,32);x=torch.zeros(128,822);x[:,760+2]=1;x[:,791+12]=1
 m=[np.ones(46,bool)]*2;acts,lp,_=p.act(x,m,generator=torch.Generator().manual_seed(1))
 for a,b in zip(acts[0].tolist(),acts[1].tolist()):
  assert I.ACTIONS[a].kind!='follow' or I.ACTIONS[b].kind=='push'
  assert I.ACTIONS[b].kind!='follow' or I.ACTIONS[a].kind=='push'
 ds,_=p.distributions(x,m,a0=acts[0]);torch.testing.assert_close(lp,ds[0].log_prob(acts[0])+ds[1].log_prob(acts[1]))
 loss=-lp.mean()-.02*sum(d.entropy().mean() for d in ds);loss.backward();assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)
 I.validate_metadata(I.metadata())

def test_short_episode_ppo_and_bounded_legal_sequence_dataset(tmp_path):
 torch.set_num_threads(1);torch.manual_seed(2);inst=scenes.scene(0,0);w=I.make_env(inst);p=I.Policy(822,32);n=RunningMeanStd(822)
 def short(i):
  e=I.make_env(i);e.config=replace(e.config,time_budget=4.);return e
 steps,s=run_episode(p,inst,n,generator=torch.Generator().manual_seed(4),update_stats=True,env_factory=short,dataset_dir=tmp_path)
 stats=ppo_update(p,torch.optim.Adam(p.parameters(),lr=3e-4),steps,epochs=1,minibatch=32);assert stats['initial_logp_max_error']<2e-5
 f=list(tmp_path.glob('*.npz'));assert len(f)==1
 with np.load(f[0],allow_pickle=False) as d:
  assert len(d['observations'])==len(d['actions'])+1 and d['observations'].shape[1]==822
  assert abs(float(d['rewards'].sum())-s['return'])<1e-5
