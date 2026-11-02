"""Physical-time credit, true actor boundaries, and separated policy/value checks."""
from dataclasses import replace
import math
from pathlib import Path
import numpy as np
import pytest
import torch
from experiments.option_rl_v14 import interface as I,scenes
from experiments.option_rl_v14.train import Step,returns_and_advantages,run_episode,ppo_update
from experiments.option_rl_v14.mode_env import EXTRA_PER_ROBOT,EXTRA_FIELDS
from experiments.option_rl_v45.train import RunningMeanStd


def step(reward,value=0.,terminal=False,periods=4):
    return Step(np.zeros(822,np.float32),44,44,0.,value,reward,np.ones(46,bool),np.ones(46,bool),terminal,periods)


def test_mc_does_not_cross_task_terminals_and_bootstraps_only_sampling_cuts():
    targets,adv=returns_and_advantages([step(1,3,True),step(2,4,False)],method='mc',bootstrap=5.)
    np.testing.assert_allclose(targets,[1,7]);np.testing.assert_allclose(adv,[-2,3])


def test_gae_hand_worked_and_variable_physical_duration():
    steps=[step(1,2,False,4),step(3,4,True,16)]
    targets,adv=returns_and_advantages(steps,trace_seconds=2.)
    np.testing.assert_allclose(adv,[3-math.exp(-.2/2),-1],rtol=1e-6)
    np.testing.assert_allclose(targets,adv+[2,4])
    a=returns_and_advantages([step(0,0,False,4),step(1,0,True)],trace_seconds=2)[1][0]
    b=returns_and_advantages([step(0,0,False,16),step(1,0,True)],trace_seconds=2)[1][0]
    assert a==pytest.approx(math.exp(-.2/2));assert b==pytest.approx(math.exp(-.8/2))


@pytest.mark.parametrize('encoder',['flat','scan'])
@pytest.mark.parametrize('autoreg',[False,True])
def test_joint_distribution_exact_entropy_legality_and_gradients(encoder,autoreg):
    torch.set_num_threads(1);torch.manual_seed(4)
    p=I.Policy(822,32,encoder=encoder,autoregressive=autoreg);x=torch.zeros(3,822)
    masks=[np.ones((3,46),bool) for _ in (0,1)];masks[0][1,:32]=False
    ds,v=p.distributions(x,masks);prob0=ds[0].probs
    joint=[]
    for i in range(46):
        d,_=p.distributions(x,masks,a0=torch.full((3,),i))
        joint.append(prob0[:,i,None]*d[1].probs)
    joint=torch.stack(joint,1)
    torch.testing.assert_close(joint.sum((1,2)),torch.ones(3))
    assert (joint[:,~p.legal_pairs]==0).all()
    exact=-(joint*joint.clamp_min(1e-30).log()).sum((1,2))
    torch.testing.assert_close(exact,sum(d.entropy() for d in ds),atol=3e-6,rtol=1e-5)
    acts,lp,v=p.act(x,masks,generator=torch.Generator().manual_seed(8))
    torch.testing.assert_close(lp,joint[torch.arange(3),acts[0],acts[1]].log())
    (-lp.mean()-.02*sum(d.entropy().mean() for d in ds)+v.square().mean()).backward()
    assert all(torch.isfinite(q.grad).all() for q in p.parameters() if q.grad is not None)


def test_autoregressive_head_can_correlate_ordinary_pushes():
    torch.set_num_threads(1);p=I.Policy(822,16,autoregressive=True)
    with torch.no_grad():p.conditional[-1].weight.normal_(0,.2)
    x=torch.zeros(1,822);m=[np.ones(46,bool)]*2
    first,_=p.distributions(x,m,a0=torch.tensor([0]));second,_=p.distributions(x,m,a0=torch.tensor([8]))
    assert not torch.allclose(first[1].probs[:,:32],second[1].probs[:,:32])


def test_actor_critic_have_disjoint_optimizers_and_critic_cannot_move_actor():
    p=I.Policy(822,16);actor=p.actor_parameters();critic=p.critic_parameters()
    assert not {id(q) for q in actor}&{id(q) for q in critic}
    assert {id(q) for q in actor+critic}=={id(q) for q in p.parameters()}
    original=[q.clone() for q in actor];opt=torch.optim.Adam(critic,lr=.01)
    ds,v=p.distributions(torch.ones(2,822),[np.ones(46,bool)]*2)
    opt.zero_grad();(v-10).square().mean().backward();opt.step()
    assert all(torch.equal(a,b) for a,b in zip(original,actor))
    assert all(q.grad is None for q in actor)


def test_shared_and_separate_start_with_identical_functions():
    torch.manual_seed(17);a=I.Policy(822,32,separate=False)
    torch.manual_seed(17);b=I.Policy(822,32,separate=True)
    x=torch.randn(4,822);m=[np.ones(46,bool)]*2
    da,va=a.distributions(x,m);db,vb=b.distributions(x,m)
    torch.testing.assert_close(va,vb,rtol=0,atol=0)
    for d,e in zip(da,db):torch.testing.assert_close(d.probs,e.probs,rtol=0,atol=0)


def test_feedback_has_explicit_pauses_and_never_reads_object_truth():
    env=I.make_env(scenes.scene(0,0));fields=env.observation_fields();vector,_,_=I.observe(env)
    assert len(fields)==len(vector)==822+2*EXTRA_PER_ROBOT
    env.slots[0].pause_reason='perception_wait'
    env.object_state=lambda:(_ for _ in ()).throw(AssertionError('hidden object state'))
    x=env.feedback_vector()
    assert x[EXTRA_FIELDS.index('phase_perception_wait')]==1
    assert x[EXTRA_FIELDS.index('perception_wait')]==1


class WaitPolicy:
    rollout_config=dict(context='extended',hold_seconds=.8)
    def act(self,x,masks,**kwargs):
        return [torch.tensor([44]),torch.tensor([44])],torch.tensor([0.]),torch.tensor([0.])


def test_held_action_is_one_actor_sample_with_dense_replay_and_true_reward(tmp_path):
    inst=scenes.scene(0,0)
    def short(i):
        e=I.make_env(i);e.config=replace(e.config,time_budget=1.65);return e
    dim=len(I.observe(short(inst))[0]);norm=RunningMeanStd(dim)
    steps,summary=run_episode(WaitPolicy(),inst,norm,env_factory=short,collect_raw=True,dataset_dir=tmp_path)
    assert len(steps)==3 and len(summary['execution_samples'])==9
    assert sum(s.periods for s in steps)==33 and steps[-1].terminal
    assert [s.periods for s in steps]==[16,16,1]
    assert sum(s.reward for s in steps)==pytest.approx(summary['return'])
    with np.load(next(tmp_path.glob('*.npz'))) as data:
        assert len(data['actions'])==9 and len(data['observations'])==10
        assert data['actor_decision'].sum()==3
        assert data['rewards'].sum()==pytest.approx(summary['return'],abs=1e-6)
        assert data['dt'].sum()==pytest.approx(1.65)


def test_new_observation_wrapper_preserves_old_healthy_physics():
    from experiments.option_rl_v13.controller import make_env as original
    inst=scenes.scene(0,4);a=original(inst);b=I.make_env(inst)
    for cmd in [(44,44)]*3+[(40,44)]*4:
        ra=a.joint_step(*cmd,record=True);rb=b.joint_step(*cmd,record=True)
        assert ra['reward']==rb['reward']
        assert a.rows[-1]['true']==b.rows[-1]['true'] and a.rows[-1]['forces']==b.rows[-1]['forces']


def test_held_push_reward_matches_independent_raw_physics_recomputation():
    from coimagine.random_start.env import recompute_episode_reward
    from experiments.option_rl_v14.learnability import proposals,instance_from_dict
    inst=instance_from_dict(proposals()[0]['instance']);worlds=[]
    def factory(i):
        e=I.make_env(i);e.config=replace(e.config,time_budget=2.4);worlds.append(e);return e
    class Push(WaitPolicy):
        def act(self,x,masks,**kwargs):
            return [torch.tensor([10]),torch.tensor([44])],torch.tensor([0.]),torch.tensor([0.])
    p=Push();e=I.make_env(inst);norm=RunningMeanStd(len(I.observe(e)[0]))
    steps,summary=run_episode(p,inst,norm,env_factory=factory,record=True)
    w=worlds[0]
    raw=recompute_episode_reward(w.rows,w.boundaries,(inst.object_pose.x,inst.object_pose.y),w.config)
    assert abs(summary['initial_distance']-summary['final_distance'])>.01
    assert sum(s.reward for s in steps)==pytest.approx(raw['total'],abs=1e-10)
    assert sum(s.periods for s in steps)==len(w.rows)


def test_new_local_failure_releases_actor_before_hold_budget():
    inst=scenes.scene(0,0)
    def factory(i):
        e=I.make_env(i);e.config=replace(e.config,time_budget=1.6);original=e.joint_step;calls=0
        def joint(*args,**kwargs):
            nonlocal calls
            result=original(*args,**kwargs);calls+=1
            if calls==1:
                e.slots[0].skill.done=True;e.slots[0].skill.exit_reason='prepare_stalled'
            return result
        e.joint_step=joint;return e
    e=I.make_env(inst);steps,_=run_episode(WaitPolicy(),inst,RunningMeanStd(len(I.observe(e)[0])),env_factory=factory)
    assert steps[0].periods==4
    assert sum(s.periods for s in steps)==32
