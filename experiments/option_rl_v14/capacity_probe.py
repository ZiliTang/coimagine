"""Supervised representability diagnostic, isolated from PPO and its checkpoints.

Offline witness labels may use truth. Inputs are legal observation/feedback only.
Fitting witness frames is not an RL result or a test of generalization.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from experiments.option_rl_v14 import interface as I
from experiments.option_rl_v14.render_episodes import instance_from_dict,render_episodes
from experiments.option_rl_v14.train import evaluate
from experiments.option_rl_v45.train import RunningMeanStd
from experiments.option_rl_v14.live import atomic_json


def run(panel,out,updates=600):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);panel=Path(panel)
    declared=json.loads(panel.read_text());torch.set_num_threads(1);torch.manual_seed(0);rng=np.random.default_rng(0)
    observations=[];masks=[[],[]];actions=[];invalid=[];scenes=[]
    for c in declared['cases']:
        inst=instance_from_dict(c['instance']);scenes.append(inst);env=I.make_env(inst)
        witness=json.loads((panel.parent/'logs'/f'{inst.name}.json').read_text())
        for k,cmd in enumerate(witness['commands']):
            obs,m0,m1=I.observe(env)
            if m0[cmd[0]] and m1[cmd[1]]:
                observations.append(obs);masks[0].append(m0);masks[1].append(m1);actions.append(cmd)
            else:invalid.append(dict(scene=inst.name,frame=k,command=cmd))
            env.joint_step(*cmd)
        if not env.success:raise RuntimeError('witness replay did not reproduce success')
    raw=np.stack(observations);norm=RunningMeanStd(raw.shape[1]);norm.update(raw)
    encoded=norm.normalise(raw);encoded[:,760:]=raw[:,760:]
    x=torch.tensor(encoded);a=torch.tensor(actions);mm=[torch.tensor(np.stack(m)) for m in masks]
    policy=I.Policy(raw.shape[1],256);policy.rollout_config=dict(context='extended',hold_seconds=.2)
    optimizer=torch.optim.Adam(policy.actor_parameters(),lr=3e-4);losses=[];started=time.monotonic()
    # Balance the few genuine switching/braking labels against long RP plateaus.
    groups={}
    for k,cmd in enumerate(actions):groups.setdefault(tuple(I.ACTIONS[j].kind for j in cmd),[]).append(k)
    buckets=list(groups.values())
    for n in range(updates):
        idx=torch.tensor([rng.choice(buckets[rng.integers(len(buckets))]) for _ in range(128)])
        ds,_=policy.distributions(x[idx],[m[idx] for m in mm],a0=a[idx,0])
        loss=-sum(d.log_prob(a[idx,i]) for i,d in enumerate(ds)).mean()
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.actor_parameters(),1.);optimizer.step()
        if n%50==0:losses.append(dict(update=n+1,loss=float(loss.detach())))
    with torch.no_grad():
        pred,_,_=policy.act(x,mm,deterministic=True)
        correct=(pred[0]==a[:,0])&(pred[1]==a[:,1]);accuracy=float(correct.float().mean())
    train=evaluate(policy,norm,scenes)
    perturbations=[instance_from_dict(d) for d in declared['perturbation_instances']]
    dev=evaluate(policy,norm,perturbations)
    atomic_json(out/'training_evaluation.json',train);atomic_json(out/'perturbation_evaluation.json',dev)
    render_episodes(policy,norm,scenes,out,count=4)
    torch.save(dict(state_dict=policy.state_dict(),model_config=policy.model_config,rollout_config=policy.rollout_config,
        obs_mean=norm.mean,obs_var=norm.var,obs_count=norm.count,scope='supervised capacity probe ONLY; not PPO initialization'),out/'probe.pt')
    result=dict(scope='privileged offline witness labels; legal current-frame inputs; supervised fit diagnostic, not RL',
        retained_frames=len(actions),invalid_label_frames=len(invalid),invalid_labels=invalid,updates=updates,
        frame_greedy_joint_accuracy=accuracy,closed_loop_train_success=train['success_rate'],
        closed_loop_perturbation_success=dev['success_rate'],losses=losses,seconds=time.monotonic()-started,
        no_ppo_weights_changed=True)
    atomic_json(out/'result.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--panel',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--updates',type=int,default=600)
    run(**vars(p.parse_args()))
