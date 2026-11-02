"""Effect-menu integration, conditional probabilities, legal resets and PPO semantics."""
from dataclasses import asdict,replace
import json,math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from coimagine.random_start import env as E,layout,spec
from coimagine.random_start.effect_push import EffectPushConfig,make_effect_skill
from experiments.option_rl_v10 import scenes,train
from experiments.option_rl_v10.interface import (
    ACTIONS,PAIR_IDS,CONFIG,Policy,make_env,observe,metadata,validate_metadata)
from experiments.option_rl_v45.train import RunningMeanStd

torch.set_num_threads(1)


def test_menu_has_no_rectangular_id_assumption_and_all_factories_align():
    assert len(ACTIONS)==68 and PAIR_IDS.shape==(14,5)
    assert PAIR_IDS[12,4]==PAIR_IDS[13,4]==-1
    assert ACTIONS[58].kind=='reposition' and ACTIONS[58].distance==1.2
    for action in ACTIONS:
        assert PAIR_IDS[action.type_index,action.scale_index]==action.index
        skill=make_effect_skill(action,config=CONFIG)
        assert skill.action==action
        if action.kind=='reposition':assert skill.peer_detour=='outer_arc' and skill.boundary_mode=='rounded_scan'
        if action.kind=='follow':assert skill.completion_clock=='maintenance_duration'
        if action.kind=='push':assert skill.config==CONFIG


def test_only_valid_global_id_is_sampled_for_every_menu_entry():
    p=Policy(3,16);x=torch.zeros(1,3)
    for a in ACTIONS:
        mask=np.zeros(68,dtype=bool);mask[a.index]=True
        ids,logp,_=p.act(x,(mask,mask))
        assert [int(i.item()) for i in ids]==[a.index,a.index]
        assert logp.item()==pytest.approx(0,abs=1e-6)
    with pytest.raises(ValueError):p.act(x,(np.zeros(68,dtype=bool),)*2)


def test_greedy_maximizes_conditional_joint_probability_and_masks_ragged_scales():
    p=Policy(3,8)
    with torch.no_grad():
        for param in p.parameters():param.zero_()
        for t in p.types:t.bias.fill_(-20);t.bias[0]=.1;t.bias[1]=0
        for s in p.scales:s.bias[5]=10
    masks=(np.ones(68,dtype=bool),)*2
    ids,_,_=p.act(torch.zeros(1,3),masks,deterministic=True)
    assert [int(a.item()) for a in ids]==[5,5] # type 0 is more likely, type 1 has the best joint action
    for s in p.scales:
        with torch.no_grad():s.bias[12*5+4]=10000
    only_wait=np.array([a.kind=='wait' for a in ACTIONS])
    ds,_=p.distributions(torch.zeros(1,3),(only_wait,only_wait))
    assert ds[0].probs.sum().item()==pytest.approx(1)
    assert ds[0].probs[0,:60].sum().item()==0
    assert torch.isfinite(ds[0].entropy()).all()


def test_sampled_and_evaluated_logprob_agree_and_masked_gradients_are_finite():
    torch.manual_seed(2);p=Policy(3,16);x=torch.randn(16,3)
    m=np.ones((16,68),dtype=bool);m[:,0:10]=False
    gen=torch.Generator().manual_seed(71)
    actions,lp,v=p.act(x,(m,m),generator=gen)
    ds,_=p.distributions(x,(m,m))
    assert torch.allclose(lp,sum(d.log_prob(a) for d,a in zip(ds,actions)))
    (-(lp.mean())+.5*v.square().mean()-.02*sum(d.entropy().mean() for d in ds)).backward()
    assert all(torch.isfinite(p.grad).all() for p in p.parameters() if p.grad is not None)


def test_legal_dynamic_scenes_reproducible_and_disjoint_from_fixed_panels():
    roots=[scenes.scene(7,i) for i in range(1000)]
    for r in roots:
        a=scenes.geometry_audit(r)
        assert a['legal'] and min(a['robot_object_gaps'])>=.06
        assert max(a['robot_object_gaps'])<=.37 and min(a['robot_wall_gaps'])>=.05
        assert a['robot_robot_gap']>=.05 and a['object_wall_gap']>=.25-1e-10
    np.random.default_rng(8).random(100);torch.rand(30)
    assert layout.content_hash(scenes.scene(7,3))==layout.content_hash(roots[3])
    assert layout.audit_split_content({'train':roots,'validation':scenes.panel(7,64,'validation'),
                                      'test':scenes.panel(7,64,'test')})['ok']


def test_known_old_overlap_roots_are_still_reproducible_and_not_silently_rewritten():
    from experiments.option_rl_v45.scenes import sample_scenes
    roots=sample_scenes(2,16,'test')
    invalid=[r.index for r in roots if 'robot_object_overlap' in scenes.geometry_audit(r)['reasons']]
    assert invalid==[5,11,12]


def test_new_environment_rejects_bad_initial_geometry_before_execution():
    from experiments.option_rl_v45.scenes import sample_scenes
    bad=sample_scenes(2,16,'test')[5]
    with pytest.raises(ValueError,match='illegal initial geometry'):make_env(bad)


def test_metadata_roundtrip_and_legacy_checkpoint_rejection(tmp_path):
    data=json.loads(json.dumps(metadata()));validate_metadata(data)
    with pytest.raises(ValueError):validate_metadata({'action_count':28})
    data['actions'][58]['distance']=1.
    with pytest.raises(ValueError):validate_metadata(data)


def test_follow_uses_declared_maintenance_time_with_legacy_default_unchanged():
    from experiments.option_testbench.effect_push_eval import instance_from_dict,reposition_scale_cases
    case=next(c for c in reposition_scale_cases() if c['name']=='scale2_rp_CCW_follow_1.20_6s')
    instance=instance_from_dict(case['instance'])
    action=next(a for a in ACTIONS if a.kind=='follow' and a.duration==2.)
    wait=next(a for a in ACTIONS if a.kind=='wait' and a.duration==2.)
    world=make_env(instance);world.joint_step(action.index,wait.index,record=True)
    rows=[r['r0'] for r in world.rows if r['r0']['kind']=='follow']
    assert rows[-1]['follow_time']==pytest.approx(2.,abs=1e-7)
    assert rows[-1]['exit_reason']=='follow_complete'
    assert make_effect_skill(action).completion_clock=='legacy_window'


def test_unchanged_policy_ppo_ratio_and_finite_actor_critic_gradient():
    p=Policy(3,16);stats=RunningMeanStd(3);steps=[]
    for i in range(6):
        obs=np.array([i*.1,.1,-.1],dtype=np.float32);m=np.ones(68,dtype=bool)
        with torch.no_grad():a,lp,v=p.act(torch.tensor(obs)[None,:],(m,m))
        steps.append(train.Step(obs,int(a[0].item()),int(a[1].item()),lp.item(),v.item(),-.1,m,m,i==5,2))
    result=train.ppo_update(p,torch.optim.Adam(p.parameters(),lr=1e-4),steps,epochs=2)
    assert result['initial_logp_max_error']<2e-5 and result['updates']==2
    assert 0<result['actor_trunk_grad_norm']<float('inf')
    assert 0<result['critic_trunk_grad_norm']<float('inf')


def test_undiscounted_returns_do_not_depend_on_option_segmentation():
    def st(reward,terminal,periods):return SimpleNamespace(reward=reward,terminal=terminal,periods=periods,value=0)
    a,_=train.returns_and_advantages([st(1,False,40),st(-3,True,1760)])
    b,_=train.returns_and_advantages([st(-2,True,1800)])
    assert a[0]==b[0]==-2


def test_short_exits_cannot_trigger_old_120_decision_cutoff(monkeypatch):
    instance=scenes.scene(4,1);waitmask=np.array([a.kind=='wait' for a in ACTIONS])
    class ClockEnv:
        def __init__(self):self.config=SimpleNamespace(time_budget=90.);self.total_periods=0;self.termination=None
        def object_distance(self):return 1.
        def joint_step(self,*args,**kwargs):
            self.total_periods+=1
            if self.total_periods==1800:self.termination=E.TERMINATION_BUDGET
            return dict(periods=1,reward={'total':-2. if self.termination else 0.},termination=self.termination,
                        success=False,skill_ends={i:{'exit_reason':'safety_abort'} for i in (0,1)})
    monkeypatch.setattr(train,'observe',lambda _: (np.zeros(3,dtype=np.float32),waitmask,waitmask))
    steps,summary=train.run_episode(Policy(3,8),instance,RunningMeanStd(3),env_factory=lambda _:ClockEnv())
    assert len(steps)==1800 and summary['seconds']==90. and steps[-1].terminal
    assert train.returns_and_advantages(steps)[0][0]==-2


def test_real_rollout_preserves_frozen_reward_and_anomaly_is_quarantined(monkeypatch):
    instance=scenes.scene(1,0);p=Policy(len(observe(make_env(instance))[0]),8)
    def short(instance):
        w=make_env(instance);w.config=replace(w.config,time_budget=.1);return w
    steps,s=train.run_episode(p,instance,RunningMeanStd(p.body[0].in_features),env_factory=short,deterministic=True)
    assert s['termination'] is not None and sum(x.reward for x in steps)==pytest.approx(s['return'])
    assert sum(s['components'].values())==pytest.approx(s['return'])
    class AnomalyEnv:
        config=SimpleNamespace(time_budget=90.);total_periods=0;termination=None
        def object_distance(self):return 1.
        def joint_step(self,*args,**kwargs):
            self.total_periods=1;self.termination=E.TERMINATION_PENETRATION
            return dict(periods=1,reward={'total':-.002},termination=self.termination,success=False,
                        skill_ends={i:{'exit_reason':'safety_abort'} for i in (0,1)})
    monkeypatch.setattr(train,'observe',lambda _: (np.zeros(3,dtype=np.float32),np.ones(68,dtype=bool),np.ones(68,dtype=bool)))
    _,s=train.run_episode(Policy(3,8),instance,RunningMeanStd(3),env_factory=lambda _:AnomalyEnv())
    assert not s['eligible_for_ppo'] and not s['success'] and s['return']==-.002
