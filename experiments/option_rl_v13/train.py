"""PPO readiness baseline using versioned effect options; historical run9 is untouched.

Centralized sharing of the two local observations; not a finite-bit policy. Gamma=1
matches the frozen environment cost. Physical anomaly episodes are recorded and
quarantined from PPO, never used as cheap terminal returns. All eval roots remain
in the reported denominator. This is a new baseline, not a run9 continuation.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib,json,math,sys,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2]
for p in (ROOT,ROOT/'src'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
from coimagine.random_start import env as E,layout,spec
from experiments.option_rl_v45.train import RunningMeanStd
from experiments.option_rl_v13 import scenes
from experiments.option_rl_v13.interface import ACTIONS,Policy,make_env,observe,metadata,validate_metadata
from experiments.option_rl_v13.live import atomic_json,status as live_status,stop_reason
from experiments.option_rl_v13.monitor import write_dashboard

# Every joint option advances >=1 control tick. This cap cannot precede the 90 s budget.
MAX_SKILLS=math.ceil(E.EnvConfig().time_budget/spec.CONTROL_DT)+1

@dataclass
class Step:
    obs:np.ndarray
    a0:int
    a1:int
    logp:float
    value:float
    reward:float
    mask0:np.ndarray
    mask1:np.ndarray
    terminal:bool
    periods:int


def run_episode(policy,instance,normalizer,*,generator=None,deterministic=False,update_stats=False,
                record=False,env_factory=make_env,collect_raw=False,dataset_dir=None,policy_iteration=0):
    env=env_factory(instance);initial=max(env.object_distance(),1e-6)
    steps=[];decisions=[];reward=0.;components=Counter();exits=Counter();last=None;raw_observations=[];model_durations=[];model_rewards=[]
    for _ in range(math.ceil(env.config.time_budget/spec.CONTROL_DT)+1):
        obs,m0,m1=observe(env)
        if collect_raw:raw_observations.append(obs.copy())
        if update_stats:normalizer.update(obs[None,:])
        encoded=normalizer.normalise(obs)
        encoded[760:]=obs[760:]  # legal controller flags/scaled commands retain exact meaning
        with torch.no_grad():
            actions,logp,value=policy.act(torch.tensor(encoded)[None,:],(m0,m1),
                                          deterministic=deterministic,generator=generator)
        a0,a1=(int(a.item()) for a in actions)
        if dataset_dir is not None and not collect_raw:raw_observations.append(obs.copy())
        boundary=env.joint_step(a0,a1,record=record);last=boundary
        if boundary['periods']<1:raise RuntimeError('zero-time option cannot enter PPO')
        terminal=boundary['termination'] is not None
        model_durations.append(boundary['periods']*spec.CONTROL_DT);model_rewards.append(boundary['reward']['total'])
        steps.append(Step(encoded,a0,a1,float(logp.item()),float(value.item()),float(boundary['reward']['total']),
                          m0.copy(),m1.copy(),terminal,int(boundary['periods'])))
        decisions.append(dict(time=env.total_periods*spec.CONTROL_DT,a0=a0,a1=a1,
                              labels=[ACTIONS[a0].label,ACTIONS[a1].label],
                              exits=[boundary['skill_ends'][i]['exit_reason'] for i in (0,1)],
                              phases=[boundary['skill_ends'][i]['stage'] for i in (0,1)],events=boundary['execution'].get('events',[])))
        for i in (0,1):exits[boundary['skill_ends'][i]['exit_reason']]+=1
        reward+=boundary['reward']['total']
        for k in ('progress','distance_cost','time_cost','success_bonus','failure_penalty'):
            components[k]+=boundary['reward'].get(k,0.)
        if terminal:break
    if env.termination is None:raise RuntimeError('physical-horizon guard exhausted without task termination')
    eligible=env.termination not in E.ANOMALY_TERMINATIONS
    summary=dict(name=instance.name,scene_hash=layout.content_hash(instance),return_=float(reward),
        success=bool(last['success']),progress=float(np.clip((initial-env.object_distance())/initial,-1.,1.)),
        initial_distance=initial,final_distance=env.object_distance(),seconds=env.total_periods*spec.CONTROL_DT,
        periods=env.total_periods,decisions=len(steps),termination=env.termination,
        eligible_for_ppo=eligible,components=dict(components),skill_exits=dict(exits),
        actions=decisions,initial_geometry=scenes.geometry_audit(instance))
    summary['return']=summary.pop('return_')
    if dataset_dir is not None and instance.split=='train':
        from experiments.option_rl_v13.dataset import save_episode
        save_episode(dataset_dir,instance.index,raw_observations+[env.boundary_vector()],[[d['a0'],d['a1']] for d in decisions],model_durations,model_rewards,scene_name=instance.name,policy_iteration=policy_iteration)
    if record:summary['rows']=env.rows
    if collect_raw:summary['raw_observations']=np.stack(raw_observations)
    return steps,summary


def returns_and_advantages(steps):
    targets=np.zeros(len(steps),dtype=np.float32);running=0.
    for i in range(len(steps)-1,-1,-1):
        running=steps[i].reward+(0. if steps[i].terminal else running)
        targets[i]=running
    return targets,targets-np.asarray([s.value for s in steps],dtype=np.float32)


def lifecycle_rates(summaries):
    records=[d for s in summaries for d in s['actions']]
    n=max(1,2*len(records));states=[(e,p) for d in records for e,p in zip(d['exits'],d['phases'])]
    return dict(controller_failed_rate=sum(e!='continuing' for e,p in states)/n,
        peer_wait_rate=sum(p=='peer_wait' for e,p in states)/n,
        perception_wait_rate=sum(p=='perception_wait' for e,p in states)/n,
        failed_or_wait_rate=sum(e!='continuing' or p in ('peer_wait','perception_wait') for e,p in states)/n,
        changed_intent_retry_rate=sum(d['events'].count('changed_intent_retry_requested') for d in records)/n)


def ppo_update(policy,optimizer,steps,*,epochs=4,minibatch=256,entropy_coef=.02,clip=.2,value_coef=.5):
    if not steps:return {'updates':0,'status':'no_valid_rollouts'}
    targets,advantages=returns_and_advantages(steps)
    if advantages.std()>1e-8:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    obs=torch.tensor(np.stack([s.obs for s in steps]));a=[torch.tensor([getattr(s,f'a{i}') for s in steps]) for i in (0,1)]
    masks=[torch.tensor(np.stack([getattr(s,f'mask{i}') for s in steps])) for i in (0,1)]
    old=torch.tensor([s.logp for s in steps]);target=torch.tensor(targets);adv=torch.tensor(advantages)
    stats=Counter();n=len(steps)
    # A contract check before any update: sampled and recomputed probabilities must agree.
    with torch.no_grad():
        ds,_=policy.distributions(obs,masks,a0=a[0]);recomputed=sum(d.log_prob(v) for d,v in zip(ds,a))
        error=float((recomputed-old).abs().max())
        if error>2e-5:raise RuntimeError(f'rollout/PPO log-prob mismatch: {error}')
    for _ in range(epochs):
        for idx in torch.randperm(n).split(minibatch):
            ds,value=policy.distributions(obs[idx],[m[idx] for m in masks],a0=a[0][idx])
            logp=sum(d.log_prob(v[idx]) for d,v in zip(ds,a));ratio=(logp-old[idx]).exp()
            stats['approx_kl']+=float((ratio-1-(logp-old[idx])).mean().detach())
            stats['clip_fraction']+=float(((ratio-1).abs()>clip).float().mean().detach())
            actor=-torch.min(ratio*adv[idx],ratio.clamp(1-clip,1+clip)*adv[idx]).mean()
            critic=(value-target[idx]).square().mean();entropy=sum(d.entropy().mean() for d in ds)
            loss=actor+value_coef*critic-entropy_coef*entropy
            optimizer.zero_grad();loss.backward()
            if stats['updates']==0:
                trunk=list(policy.body.parameters())
                # Separate terms are diagnostic only; reconstruct their graph once.
                d2,v2=policy.distributions(obs[idx],[m[idx] for m in masks],a0=a[0][idx])
                lp2=sum(d.log_prob(v[idx]) for d,v in zip(d2,a));r2=(lp2-old[idx]).exp()
                pl=-torch.min(r2*adv[idx],r2.clamp(1-clip,1+clip)*adv[idx]).mean()
                vl=value_coef*(v2-target[idx]).square().mean()
                for name,term in (('actor_trunk_grad_norm',pl),('critic_trunk_grad_norm',vl)):
                    gs=torch.autograd.grad(term,trunk,retain_graph=True,allow_unused=True)
                    stats[name]=math.sqrt(sum(float(g.square().sum()) for g in gs if g is not None))
            grad=nn.utils.clip_grad_norm_(policy.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(grad):raise RuntimeError('nonfinite PPO loss/gradient')
            optimizer.step();stats['policy_loss']+=float(actor.detach());stats['value_loss']+=float(critic.detach())
            stats['entropy']+=float(entropy.detach());stats['grad_norm_max']=max(stats['grad_norm_max'],float(grad));stats['updates']+=1
    for k in ('policy_loss','value_loss','entropy','approx_kl','clip_fraction'):stats[k]/=stats['updates']
    return dict(stats,initial_logp_max_error=error)


def evaluate(policy,normalizer,instances,*,stochastic=False,seed=0,on_root=None,should_stop=None,executor=None):
    generator=torch.Generator().manual_seed(seed);summaries=[]
    if executor is not None:
        from experiments.option_rl_v13.parallel import evaluation_summaries
        summaries=evaluation_summaries(executor,policy,normalizer,instances,stochastic=stochastic,
            seed=seed,on_root=on_root,should_stop=should_stop)
    else:
        for instance in instances:
            if should_stop is not None and should_stop():break
            _,s=run_episode(policy,instance,normalizer,deterministic=not stochastic,generator=generator)
            summaries.append(s)
            if on_root is not None:on_root(len(summaries),len(instances),s)
    def mean(key):return float(np.mean([s[key] for s in summaries])) if summaries else None
    bins={}
    for name,lo,hi in [('near',0.,.75),('middle',.75,1.5),('far',1.5,float('inf'))]:
        sub=[s for s in summaries if lo<=s['initial_distance']<hi]
        bins[name]=dict(count=len(sub),success_rate=float(np.mean([s['success'] for s in sub])) if sub else None,
                       mean_progress=float(np.mean([s['progress'] for s in sub])) if sub else None)
    outside=[s for s in summaries if s['initial_distance']>spec.GOAL_RADIUS]
    complete=len(summaries)==len(instances)
    return dict(success_rate=mean('success') if complete else None,
        mean_progress=mean('progress') if complete else None,mean_return=mean('return') if complete else None,
        complete=complete,expected_roots=len(instances),completed_roots=len(summaries),
        initially_inside_goal=len(summaries)-len(outside),outside_goal_count=len(outside),
        outside_goal_success=float(np.mean([s['success'] for s in outside])) if outside and complete else None,
        anomalies=sum(not s['eligible_for_ppo'] for s in summaries),roots=summaries,distance_bins=bins,
        lifecycle=lifecycle_rates(summaries))


def load_checkpoint(path):
    data=torch.load(path,weights_only=False,map_location='cpu');validate_metadata(data)
    policy=Policy(data['obs_dim'],data['hidden']);policy.load_state_dict(data['state_dict']);policy.eval()
    normalizer=RunningMeanStd(data['obs_dim']);normalizer.mean=data['obs_mean'];normalizer.var=data['obs_var'];normalizer.count=data['obs_count']
    return policy,normalizer,data


def train(*,out,iterations=10000,episodes=8,hours=1.,seed=0,eval_roots=64,eval_every=10,
          hidden=256,epochs=4,lr=3e-4,entropy=.02,render_every=0,workers=1,resume_from=None):
    if min(iterations,episodes,eval_every,hidden,epochs,workers)<=0 or hours<=0 or eval_roots<0:
        raise ValueError('positive budgets and nonnegative eval roots required')
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('new baseline requires an empty output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'evaluations').mkdir();(out/'rollouts').mkdir()
    torch.set_num_threads(1);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+701)
    validations=scenes.panel(seed,eval_roots,'validation');tests=scenes.panel(seed,eval_roots,'test')
    split_audit=layout.audit_split_content({'validation':validations,'test':tests})
    if not split_audit['ok']:raise ValueError('fixed panels overlap')
    frozen={layout.content_hash(i) for i in validations+tests}
    manifest=dict(metadata(),dense_dataset='5Hz boundary sequences, every32 train roots / last256 selected episodes, atomic NPZ; 10Hz intra-mode interface available but not enabled; world-model training not enabled',scene_version=scenes.VERSION,seed=seed,training_scenes='one fresh indexed root per episode attempt',
        evaluation_scenes='fixed new legal roots; different from run9',goal_distribution='uniform full random; no annealing or BC',
        gamma=1.,anomaly_rule='record all; quarantine physical-anomaly trajectories from PPO; all eval roots count',
        normalization_protocol='serial-bootstrap-then-batch-frozen-v1' if workers>1 else 'legacy-online-v1',
        max_attempts_per_iteration=episodes*3,config=locals_config(iterations,episodes,hours,eval_roots,eval_every,hidden,epochs,lr,entropy,render_every,workers),
        validation_instances=[i.as_dict() for i in validations],test_instances=[i.as_dict() for i in tests],split_audit=split_audit)
    sources=[Path(__file__),Path(__file__).with_name('interface.py'),Path(__file__).with_name('scenes.py'),Path(__file__).with_name('monitor.py'),Path(__file__).with_name('live.py'),Path(__file__).with_name('parallel.py')]
    sources += [ROOT/'src/coimagine/random_start'/f'{name}.py' for name in ('env','spec','skills','effect_push','boundary_follow','endpoint_tracking','same_face','lidar','progress_guard','efficient_profile','peer_clearance')]
    sources += [Path(__file__).with_name('dataset.py'),Path(__file__).with_name('mode_env.py'),Path(__file__).with_name('push_controller.py'),Path(__file__).with_name('controller.py'),ROOT/'src/coimagine/continuous2d/environment.py',ROOT/'src/coimagine/continuous2d/backend.py',ROOT/'src/coimagine/continuous2d/config.py',ROOT/'src/coimagine/local_contact_option/control.py']
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest['reward_config']=E.EnvConfig(lidar_backend='numpy_batch',peer_clearance_recovery='boundary_v1').as_dict()
    resumed=None
    if resume_from is not None:
        resume_from=Path(resume_from).resolve();resumed=torch.load(resume_from,map_location='cpu',weights_only=False)
        validate_metadata(resumed)
        if resumed['hidden']!=hidden:raise ValueError('resume hidden dimension mismatch')
        manifest['resume']=dict(path=str(resume_from),sha256=hashlib.sha256(resume_from.read_bytes()).hexdigest(),
            parent_iteration=resumed['history'][-1]['iteration'] if resumed['history'] else 0,
            next_scene_ordinal=resumed['next_scene_ordinal'],parent_manifest=resumed['manifest'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    first=scenes.scene(seed,0);obs_dim=len(observe(make_env(first))[0]);normalizer=RunningMeanStd(obs_dim)
    policy=Policy(obs_dim,hidden);optimizer=torch.optim.Adam(policy.parameters(),lr=lr)
    history=[];ordinal=0;started=time.monotonic();best=None;status='complete'
    if resumed is not None:
        if resumed['obs_dim']!=obs_dim:raise ValueError('resume observation schema mismatch')
        policy.load_state_dict(resumed['state_dict']);optimizer.load_state_dict(resumed['optimizer'])
        normalizer.mean=np.asarray(resumed['obs_mean']).copy();normalizer.var=np.asarray(resumed['obs_var']).copy();normalizer.count=resumed['obs_count']
        ordinal=resumed['next_scene_ordinal'];torch.set_rng_state(resumed['torch_rng']);generator.set_state(resumed['sampling_rng'])
    def save(path):
        data=dict(metadata(),state_dict=policy.state_dict(),optimizer=optimizer.state_dict(),obs_dim=obs_dim,hidden=hidden,
                  obs_mean=normalizer.mean,obs_var=normalizer.var,obs_count=normalizer.count,history=history,
                  next_scene_ordinal=ordinal,torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state(),manifest=manifest)
        temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
    def persist():
        atomic_json(out/'history.json',history)
        write_dashboard(history,out/'dashboard.html',subtitle=f'{out} | continuous46-5hz | dynamic legal | gamma=1')
    live_status(out,phase='starting',iteration=0,episodes=episodes)
    persist()
    from experiments.option_rl_v13.parallel import pool
    executor=pool(workers) if workers>1 else None
    try:
        for iteration in range(1,iterations+1):
            if stop_reason(out):status='requested_stop';break
            iteration_started=time.monotonic();valid=[];attempts=[]
            raw_observations=[]
            if executor is not None and normalizer.count>1e-3:
                from experiments.option_rl_v13.parallel import training_batch
                valid,attempts,ordinal,raw_observations=training_batch(executor,policy,normalizer,
                    seed=seed,ordinal=ordinal,episodes=episodes,frozen=frozen,out=out,iteration=iteration,workers=workers)
            else:
                for attempt in range(episodes*3):
                    if stop_reason(out):break
                    live_status(out,phase='rollout',iteration=iteration,episode=len(valid),episodes=episodes,attempt=attempt)
                    instance=scenes.scene(seed,ordinal);ordinal+=1
                    if layout.content_hash(instance) in frozen:raise RuntimeError('dynamic training root duplicates fixed holdout')
                    steps,summary=run_episode(policy,instance,normalizer,generator=generator,update_stats=True,dataset_dir=out/'world_model_replay',policy_iteration=iteration)
                    atomic_json(out/'rollouts'/f'{instance.index:07d}.json',dict(instance=instance.as_dict(),summary=summary))
                    attempts.append(summary)
                    if summary['eligible_for_ppo']:valid.append((steps,summary))
                    live_status(out,phase='rollout',iteration=iteration,episode=len(valid),episodes=episodes,attempt=attempt+1)
                    if len(valid)>=episodes:break
            if len(valid)<episodes:
                status='requested_stop' if stop_reason(out) else 'insufficient_valid_rollouts'
                save(out/'checkpoint.pt');break
            rollout_wall_seconds=time.monotonic()-iteration_started
            live_status(out,phase='ppo',iteration=iteration,episodes=episodes,episode=len(valid))
            stats=ppo_update(policy,optimizer,[s for ep,_ in valid for s in ep],epochs=epochs,entropy_coef=entropy)
            if raw_observations:normalizer.update(np.concatenate(raw_observations,axis=0))
            outside=[s for s in attempts if s['initial_distance']>spec.GOAL_RADIUS]
            all_actions=[a for summary in attempts for d in summary['actions'] for a in (d['a0'],d['a1'])]
            mix=Counter(ACTIONS[a].effect_kind if ACTIONS[a].kind=='push' else ACTIONS[a].kind for a in all_actions)
            all_exits=Counter()
            lengths=[]
            for summary in attempts:
                all_exits.update(summary['skill_exits']);previous=0.
                for d in summary['actions']:lengths.append(d['time']-previous);previous=d['time']
            entry=dict(iteration=iteration,explore_return=float(np.mean([s['return'] for s in attempts])),
                explore_success=float(np.mean([s['success'] for s in attempts])),explore_progress=float(np.mean([s['progress'] for s in attempts])),
                attempted_episodes=len(attempts),valid_episodes=len(valid),anomaly_episodes=len(attempts)-len(valid),
                train_anomaly_rate=(len(attempts)-len(valid))/len(attempts),
                train_initial_inside_goal=len(attempts)-len(outside),
                train_outside_goal_success=float(np.mean([s['success'] for s in outside])) if outside else None,
                steps=sum(len(ep) for ep,_ in valid),seconds=time.monotonic()-started,success_rate=None,mean_progress=None,**stats)
            for kind in ('translate','rotate','reposition','follow','wait'):entry['action_'+kind]=mix[kind]/max(1,len(all_actions))
            entry['blocked_exit_rate']=sum(v for k,v in all_exits.items() if k.startswith('blocked') or k=='safety_abort' or 'stalled' in k)/max(1,sum(all_exits.values()))
            mode_decisions=[d for summary in attempts for d in summary['actions']]
            entry['mode_switch_request_rate']=sum(d['events'].count('mode_change_requested') for d in mode_decisions)/max(1,2*len(mode_decisions))
            entry['mode_switch_commit_rate']=sum(d['events'].count('mode_change_committed') for d in mode_decisions)/max(1,2*len(mode_decisions))
            entry['switching_state_rate']=sum(p=='switching' for d in mode_decisions for p in d['phases'])/max(1,2*len(mode_decisions))
            entry.update(lifecycle_rates(attempts))
            entry['stalled_exit_rate']=sum(v for k,v in all_exits.items() if 'stalled' in k)/max(1,sum(all_exits.values()))
            entry['long_option_rate']=sum(t>=10 for t in lengths)/max(1,len(lengths))
            entry['option_mean_seconds']=float(np.mean(lengths)) if lengths else 0.
            entry['option_p95_seconds']=float(np.percentile(lengths,95)) if lengths else 0.
            entry['rollout_wall_seconds']=rollout_wall_seconds
            entry['sample_periods_per_wall_second']=sum(s['periods'] for s in attempts)/max(rollout_wall_seconds,1e-9)
            entry['short_option_rate']=sum(t<=.15+1e-9 for t in lengths)/max(1,len(lengths))
            for key in ('progress','distance_cost','time_cost','success_bonus','failure_penalty'):
                entry['reward_'+key]=float(np.mean([s['components'][key] for s in attempts]))
            entry['iteration_seconds']=time.monotonic()-iteration_started
            history.append(entry)
            live_status(out,phase='iteration_saved',iteration=iteration,episode=len(valid),episodes=episodes)
            persist();save(out/'checkpoint.pt')
            score=None
            if validations and not stop_reason(out) and (iteration==1 or iteration%eval_every==0):
                def progress(done,total,root):
                    live_status(out,phase='validation',iteration=iteration,episode=len(valid),episodes=episodes,
                                eval_done=done,eval_total=total,eval_root=root['name'])
                progress(0,len(validations),{'name':''})
                persist()
                evaluation=evaluate(policy,normalizer,validations,on_root=progress,should_stop=lambda:stop_reason(out),executor=executor)
                atomic_json(out/'evaluations'/f'iteration_{iteration:06d}.json',evaluation)
                if evaluation['complete']:
                    entry.update(success_rate=evaluation['success_rate'],mean_progress=evaluation['mean_progress'],
                        eval_return=evaluation['mean_return'],eval_anomalies=evaluation['anomalies'],
                        eval_anomaly_rate=evaluation['anomalies']/len(validations),
                        eval_outside_goal_success=evaluation['outside_goal_success'],eval_initial_inside_goal=evaluation['initially_inside_goal'])
                    for name,data in evaluation['distance_bins'].items():
                        entry[f'val_{name}_success']=data['success_rate'];entry[f'val_{name}_progress']=data['mean_progress']
                    entry.update({'eval_'+k:v for k,v in evaluation['lifecycle'].items()})
                    score=(evaluation['success_rate'],evaluation['mean_progress'])
            entry['seconds']=time.monotonic()-started;entry['iteration_seconds']=time.monotonic()-iteration_started
            persist();save(out/'checkpoint.pt')
            if score is not None and (best is None or score>best):best=score;save(out/'best.pt')
            if render_every and not stop_reason(out) and (iteration==1 or iteration%render_every==0) and validations:
                live_status(out,phase='render',iteration=iteration,episode=len(valid),episodes=episodes)
                from experiments.option_rl_v13.render_episodes import render_episodes
                render_episodes(policy,normalizer,validations,out,count=4)
            print(json.dumps(entry),flush=True)
            if stop_reason(out):status='requested_stop';break
            if time.monotonic()-started>=hours*3600:status='wall_budget_reached';break
    finally:
        if executor is not None:executor.shutdown(wait=True,cancel_futures=True)
    save(out/'policy.pt')
    result=dict(status=status,iterations=len(history),seconds=time.monotonic()-started,next_scene_ordinal=ordinal,
                best_validation=best,checkpoint=str(out/'policy.pt'),stop_reason=stop_reason(out))
    atomic_json(out/'result.json',result)
    live_status(out,phase=status,iteration=len(history),episodes=episodes,result='result.json')
    persist();return result


def locals_config(iterations,episodes,hours,eval_roots,eval_every,hidden,epochs,lr,entropy,render_every,workers=1):
    return dict(iterations=iterations,episodes=episodes,hours=hours,eval_roots=eval_roots,eval_every=eval_every,
                hidden=hidden,epochs=epochs,lr=lr,entropy=entropy,render_every=render_every,workers=workers)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--iterations',type=int,default=10000)
    p.add_argument('--episodes',type=int,default=8);p.add_argument('--hours',type=float,default=1.)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--eval-roots',type=int,default=64)
    p.add_argument('--eval-every',type=int,default=10);p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--entropy',type=float,default=.02)
    p.add_argument('--render-every',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--resume-from',type=Path)
    args=vars(p.parse_args());print(json.dumps(train(**args)),flush=True)
if __name__=='__main__':main()
