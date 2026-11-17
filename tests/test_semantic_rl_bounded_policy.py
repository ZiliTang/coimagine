import pytest,torch,random
from types import MethodType
from experiments.planar_tasks.semantic_rl.bounded_policy import BoundedPolicy,compatible,action_parameters,bounded_ppo_update
from experiments.planar_tasks.semantic_rl.parameter_policy import DurationPolicy,state_digest
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.batched_ppo import prepare,evaluate_prepared
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.relational_search.pipeline_execution import seed_fixture
from experiments.planar_tasks.authoring_v2.random_curriculum import build
from experiments.planar_tasks.authoring_v2.semantic_rl import development_fixtures
from test_semantic_rl_parameter_policy import fixture


def test_direction_speed_and_family_contract():
 assert compatible(0,3,0) and not compatible(4,3,0)
 assert compatible(7,0,1) and compatible(31,0,1) and not compatible(8,0,1)
 assert compatible(78,47,1) and not compatible(55,47,1)
 assert not compatible(47,0,1) and not compatible(46,44,1)
 assert compatible(32,34,1) and not compatible(35,34,1)
 assert compatible(43,43,1) and not compatible(40,43,1)
 with pytest.raises(ValueError):action_parameters(45)

@pytest.mark.parametrize('direction',[0,1])
def test_zero_update_greedy_parity_save_and_probabilities(fixture,tmp_path,direction):
 h,f,s=fixture;p=BoundedPolicy.from_supervisor(s,direction_radius=direction)
 assert p.decide(h,f,deterministic=True)['action']==s.decide(h,f,deterministic=True)['action']
 gen=torch.Generator().manual_seed(78)
 for _ in range(3):
  tr=p.decide(h,f,generator=gen,temperature=2.);a,q=tr['action'],tr['supervisor_action']
  assert a['target_id']==q['target_id']
  assert all(compatible(a[k],q[k],direction) for k in ('a0','a1'))
  assert abs(a['ticks']-q['ticks'])<=4
  assert float(evaluate_prepared(p,prepare(p,[tr]))[0][0].detach())==pytest.approx(tr['logp'],abs=2e-5)
 p.save(tmp_path/'p.pt',{});loaded,_=BoundedPolicy.load(tmp_path/'p.pt',f.sha256)
 assert loaded.decide(h,f,deterministic=True)['action']==p.decide(h,f,deterministic=True)['action']
 with pytest.raises(ValueError):DurationPolicy.load(tmp_path/'p.pt',f.sha256)
 with pytest.raises(ValueError):Policy.load(tmp_path/'p.pt',f.sha256)


def test_move_conditional_support_is_legal_and_update_preserves_supervisor(fixture):
 h,f,s=fixture;anchor=dict(target_id=None,a0=47,a1=63,ticks=20)
 def forced(self,history,features,**kwargs):
  kwargs['teacher']=anchor;return Policy._decide(self,history,features,**kwargs)
 s._decide=MethodType(forced,s);p=BoundedPolicy.from_supervisor(s,direction_radius=1)
 gen=torch.Generator().manual_seed(1);before=state_digest(p.supervisor)
 with torch.no_grad():
  for param in p.semantic.parameters():param.add_(torch.randn_like(param))
 tr=p.decide(h,f,generator=gen,temperature=2.)
 assert len(tr['panels'][1]['choices'])>4
 from experiments.planar_tasks.relational_search.contract import validate_action
 validate_action(tr['action'],h.events[-1]['observation'])
 for panel in tr['panels']:
  for a in panel['representatives']:validate_action(a,h.events[-1]['observation'])
 assert all(compatible(tr['action'][k],anchor[k],1) for k in ('a0','a1'))
 tr.update(periods=1,reward=-.05,terminal=True)
 out=bounded_ppo_update(p,torch.optim.Adam(p.parameters(),lr=3e-4),[dict(steps=[tr],bootstrap=0)],epochs=1)
 assert out['updates']==1 and out['logp_recompute_error']<2e-5
 assert before==state_digest(p.supervisor)


def test_curriculum_reproducible_no_outcome_filter_and_disjoint():
 bases=[dict(id=k,family=k,fixture=f) for k,f in development_fixtures('stage1-wall-anchor-v2').items()]
 a=build(bases,rounds=2);b=build(bases,rounds=2);assert a==b
 train=a['train']['cases'];assert len(train)==32
 from collections import Counter
 assert Counter(r['level'] for r in train)==dict(anchor=8,small=16,medium=8)
 hashes={k:{r['fixture_sha256'] for r in v['cases']} for k,v in a.items()}
 assert not hashes['train']&hashes['test'] and not hashes['train']&hashes['validation'] and not hashes['validation']&hashes['test']
 assert len(a['validation']['cases'])==24 and len(a['test']['cases'])==48
 assert all('success' not in r for v in a.values() for r in v['cases'])
