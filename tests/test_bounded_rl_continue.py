import torch
from test_semantic_rl_parameter_policy import fixture
from experiments.planar_tasks.semantic_rl.bounded_policy import BoundedPolicy,bounded_ppo_update
from experiments.planar_tasks.semantic_rl.parameter_policy import state_digest
from experiments.planar_tasks.authoring_v2.bounded_rl_continue import restore


def equal(a,b):
 if isinstance(a,torch.Tensor):return torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b


def test_adam_continuation_matches_uninterrupted_update(fixture,tmp_path):
 h,f,s=fixture;p=BoundedPolicy.from_supervisor(s,direction_radius=1)
 opt=torch.optim.Adam(p.parameters(),lr=3e-4)
 def episode(model,seed):
  tr=model.decide(h,f,generator=torch.Generator().manual_seed(seed),temperature=2.)
  tr.update(periods=1,reward=-.05,terminal=True)
  return [dict(steps=[tr],bootstrap=0.)]
 bounded_ppo_update(p,opt,episode(p,1),epochs=1,seed=17)
 p.save(tmp_path/'p.pt',{});torch.save(dict(optimizer=opt.state_dict()),tmp_path/'o.pt')
 q,other=restore(tmp_path/'p.pt',tmp_path/'o.pt',f.source)
 assert equal(opt.state_dict(),other.state_dict()) and state_digest(p)==state_digest(q)
 e=episode(p,2);a=bounded_ppo_update(p,opt,e,epochs=1,seed=18);b=bounded_ppo_update(q,other,e,epochs=1,seed=18)
 assert a==b and state_digest(p)==state_digest(q) and equal(opt.state_dict(),other.state_dict())
