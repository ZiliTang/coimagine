"""Frozen-policy stopping, credit and braking counterfactual diagnostics.

Object truth is logged/used to select OFFLINE fork points only. The sampled
policy always receives the existing legal observation and controller feedback.
Each fork replays commands from reset; no unsupported controller snapshot.
"""
import argparse,hashlib,json,math,time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path
import numpy as np
import torch
from experiments.option_rl_v14.interface import ACTIONS,make_env
from experiments.option_rl_v14.learnability import load_panel
from experiments.option_rl_v14.train import load_checkpoint,run_episode,returns_and_advantages
from experiments.option_rl_v14.live import atomic_json


def truth(env):
    o=env.object_state();gx,gy=env.instance.goal_centre;distance=math.hypot(gx-o.x,gy-o.y)
    return dict(period=env.total_periods,time=env.total_periods*.05,distance=distance,
        linear_speed=math.hypot(o.vx,o.vy),angular_speed=abs(o.omega),
        closing_speed=((gx-o.x)*o.vx+(gy-o.y)*o.vy)/max(distance,1e-8))


def episode_job(job):
    torch.set_num_threads(1)
    p,n,_=load_checkpoint(job['checkpoint']);env=make_env(job['instance'],context=p.rollout_config['context'])
    states={};original=env.joint_step
    def logged(*args,**kwargs):
        states[env.total_periods]=truth(env)
        return original(*args,**kwargs)
    env.joint_step=logged
    steps,summary=run_episode(p,job['instance'],n,generator=torch.Generator().manual_seed(job['seed']),
        deterministic=job['greedy'],env_factory=lambda _:env)
    x=torch.tensor(np.stack([s.obs for s in steps]));mm=[torch.tensor(np.stack([getattr(s,f'mask{i}') for s in steps])) for i in (0,1)]
    with torch.no_grad():
        ds,_=p.distributions(x,mm,a0=torch.full((len(steps),),44))
        brake=(ds[0].probs[:,44]*ds[1].probs[:,44]).numpy()
    _,gae=returns_and_advantages(steps);_,mc=returns_and_advantages(steps,method='mc')
    records=[];period=0
    for k,s in enumerate(steps):
        row=dict(states[period],a0=s.a0,a1=s.a1,labels=[ACTIONS[s.a0].label,ACTIONS[s.a1].label],
            both_brake=s.a0==44 and s.a1==44,one_brake=s.a0==44 or s.a1==44,
            brake_pair_probability=float(brake[k]),gae=float(gae[k]),mc=float(mc[k]),
            value=s.value,reward=s.reward,dt=s.periods*.05)
        records.append(row);period+=s.periods
    return dict(name=summary['name'],seed=job['seed'],greedy=job['greedy'],success=summary['success'],
        return_=summary['return'],progress=summary['progress'],termination=summary['termination'],
        records=records,actions=summary['actions'])


def replay_prefix(env,actions,count):
    for action in actions[:count]:
        start=env.total_periods
        while env.total_periods-start<round(action['dt']/.05):
            b=env.joint_step(action['a0'],action['a1'])
            if env.termination:raise RuntimeError('fork prefix terminated')
        if env.total_periods-start!=round(action['dt']/.05):raise RuntimeError('prefix clock mismatch')


def forks(instance,episode):
    out=[];selected=set()
    for threshold in (.30,.18):
        idx=next((i for i,r in enumerate(episode['records']) if r['distance']<threshold and r['closing_speed']>.01),None)
        if idx is None or idx in selected:continue
        selected.add(idx);env=make_env(instance);replay_prefix(env,episode['actions'],idx)
        state=truth(env);expected=episode['records'][idx]
        if abs(state['distance']-expected['distance'])>1e-9:raise RuntimeError('prefix physical mismatch')
        while not env.termination:env.joint_step(44,44)
        out.append(dict(name=instance.name,threshold=threshold,fork_time=state['time'],fork_distance=state['distance'],
            fork_speed=state['linear_speed'],baseline_success=episode['success'],brake_success=env.success,
            baseline_return=episode['return_'],brake_return=env.reward_total,
            final_distance=env.object_distance(),seconds=env.total_periods*.05,
            prefix_physical_match=True,offline_truth_selected_fork=True))
    return out


def aggregate(episodes):
    all_records=[dict(r,episode_success=e['success']) for e in episodes for r in e['records']]
    groups={
        'approaching_near_goal':[r for r in all_records if r['distance']<.35 and r['closing_speed']>.01],
        'inside_goal_moving':[r for r in all_records if r['distance']<.2 and r['linear_speed']>.02],
        'inside_goal_slow':[r for r in all_records if r['distance']<.2 and r['linear_speed']<=.02],
    }
    result={}
    for name,records in groups.items():
        result[name]=dict(count=len(records),both_brake=sum(r['both_brake'] for r in records),
            brake_pair_probability=float(np.mean([r['brake_pair_probability'] for r in records])) if records else None)
        for choice in ('brake','other'):
            sub=[r for r in records if r['both_brake']==(choice=='brake')]
            result[name][choice]=dict(count=len(sub),success_episode_records=sum(r['episode_success'] for r in sub),
                mean_gae=float(np.mean([r['gae'] for r in sub])) if sub else None,
                mean_mc=float(np.mean([r['mc'] for r in sub])) if sub else None)
    return result


def run(checkpoint,panel,out,samples=64,workers=8):
    out=Path(out);out.mkdir(parents=True,exist_ok=False);(out/'episodes').mkdir();start=time.monotonic()
    train,_=load_panel(panel)
    jobs=[dict(checkpoint=str(Path(checkpoint).resolve()),instance=i,greedy=True,seed=0) for i in train]
    jobs += [dict(checkpoint=str(Path(checkpoint).resolve()),instance=train[k%len(train)],greedy=False,seed=10000+k) for k in range(samples)]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        episodes=list(pool.map(episode_job,jobs))
    greedy=episodes[:len(train)];stochastic=episodes[len(train):]
    branches=[]
    for i,e in zip(train,greedy):branches+=forks(i,e)
    for k,e in enumerate(episodes):atomic_json(out/'episodes'/f'{k:03d}.json',e)
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
        scope='frozen final policy; fresh diagnostic on-policy samples; truth used only for offline grouping/forks',
        greedy_count=len(greedy),greedy_success=sum(e['success'] for e in greedy),
        stochastic_count=len(stochastic),stochastic_success=sum(e['success'] for e in stochastic),
        stochastic=aggregate(stochastic),greedy=aggregate(greedy),forks=branches,seconds=time.monotonic()-start)
    atomic_json(out/'summary.json',result);print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--panel',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=int,default=64);p.add_argument('--workers',type=int,default=8)
    run(**vars(p.parse_args()))
