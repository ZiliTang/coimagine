from copy import deepcopy
import pytest
import torch
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.parameter_policy import DurationPolicy,state_digest,duration_ppo_update
from experiments.planar_tasks.semantic_rl.batched_ppo import prepare,evaluate_prepared
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture

SOURCE="""NAMES=['duration']
def initial_memory():return {}
def update_memory(m,e):return m
def propose_actions(m,o):return [dict(target_id=None,a0=44,a1=44,ticks=20)]
def semantic_features(m,o,a,p):return dict(duration=a['ticks']/40)
"""

@pytest.fixture
def fixture():
 torch.set_num_threads(1);torch.manual_seed(17)
 env=Environment(seed_fixture(),EventConfig(mode='duration'),fast_feedback=True)
 f=Features(SOURCE,action_protocol=protocol(EventConfig(mode='duration')))
 root=f.panel(env.history,ActionTree(env.history.events[-1]['observation']),{})
 model=Policy([root],f.sha256)
 yield env.history,f,model
 f.close()

def test_structure_immutable_despite_actor_changes_and_sampling(fixture):
 h,f,s=fixture;p=DurationPolicy.from_supervisor(s);sha=state_digest(s)
 teacher=s.decide(h,f,deterministic=True)['action']
 with torch.no_grad():
  for v in p.semantic.parameters():v.add_(torch.randn_like(v)*.5)
 gen=torch.Generator().manual_seed(42)
 for _ in range(4):
  tr=p.decide(h,f,generator=gen,temperature=2.)
  assert all(tr['action'][k]==teacher[k] for k in ('target_id','a0','a1'))
  assert 1<=tr['action']['ticks']<=40 and abs(tr['action']['ticks']-teacher['ticks'])<=4
  assert [len(x['choices']) for x in tr['panels'][:3]]==[1,1,1]
  assert state_digest(p.supervisor)==sha
  lp=evaluate_prepared(p,prepare(p,[tr]))[0][0]
  assert float(lp.detach())==pytest.approx(tr['logp'],abs=2e-5)

def test_zero_radius_and_checkpoint_cannot_load_as_unrestricted(tmp_path,fixture):
 h,f,s=fixture;p=DurationPolicy.from_supervisor(s,radius=0)
 assert p.decide(h,f,deterministic=True)['action']==s.decide(h,f,deterministic=True)['action']
 p.save(tmp_path/'restricted.pt',dict(test=True));q,meta=DurationPolicy.load(tmp_path/'restricted.pt',f.sha256)
 assert q.decide(h,f,deterministic=True)['action']==p.decide(h,f,deterministic=True)['action']
 assert meta['test']
 with pytest.raises(ValueError,match='incompatible'):Policy.load(tmp_path/'restricted.pt',f.sha256)
 with pytest.raises(ValueError,match='semantic'):DurationPolicy.load(tmp_path/'restricted.pt','wrong')

def test_update_preserves_supervisor_and_separates_gradient_clipping(fixture,monkeypatch):
 h,f,s=fixture;p=DurationPolicy.from_supervisor(s)
 tr=p.decide(h,f,deterministic=True);tr.update(periods=1,reward=-.05,terminal=True)
 before=state_digest(p.supervisor);calls=[];original=torch.nn.utils.clip_grad_norm_
 def capture(parameters,*args,**kwargs):
  params=list(parameters);calls.append({id(x) for x in params});return original(params,*args,**kwargs)
 monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',capture)
 opt=torch.optim.Adam(p.parameters(),lr=3e-4)
 result=duration_ppo_update(p,opt,[dict(steps=[tr],bootstrap=0.)],epochs=1)
 assert result['updates']==1 and result['logp_recompute_error']<2e-5
 assert len(calls)==2 and not (calls[0]&calls[1])
 assert calls[0]=={id(x) for m in (p.semantic,p.residual) for x in m.parameters()}
 assert calls[1]=={id(x) for x in p.critic.parameters()}
 assert state_digest(p.supervisor)==before
 bad=deepcopy(tr);bad['panels'][0]['semantic']*=2;bad['panels'][0]['raw']*=2
 with pytest.raises(ValueError,match='unfrozen'):duration_ppo_update(p,opt,[dict(steps=[bad],bootstrap=0.)])
