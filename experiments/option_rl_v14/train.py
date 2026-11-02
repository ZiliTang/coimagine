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
from experiments.option_rl_v14 import scenes
from experiments.option_rl_v14.interface import ACTIONS,Policy,make_env,observe,metadata,validate_metadata
from experiments.option_rl_v14.live import atomic_json,status as live_status,stop_reason
from experiments.option_rl_v14.monitor import write_dashboard

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
                record=False,env_factory=None,collect_raw=False,dataset_dir=None,policy_iteration=0,sample_ordinal=None):
    config=getattr(policy,'rollout_config',dict(context='extended',hold_seconds=.8))
    env=(env_factory(instance) if env_factory else make_env(instance,context=config['context']))
    initial=max(env.object_distance(),1e-6);repeats=round(config['hold_seconds']/.2)
    steps=[];decisions=[];execution=[];reward=0.;components=Counter();exits=Counter();last=None
    raw=[];dense_obs=[];dense_actions=[];dense_dt=[];dense_rewards=[];decision_flags=[]
    for _ in range(math.ceil(env.config.time_budget/.2)+1):
        obs,m0,m1=observe(env)
        if update_stats:normalizer.update(obs[None,:])
        encoded=normalizer.normalise(obs);encoded[760:]=obs[760:]
        with torch.no_grad():
            actions,logp,value=policy.act(torch.tensor(encoded)[None,:],(m0,m1),deterministic=deterministic,generator=generator)
        a0,a1=(int(a.item()) for a in actions);step_reward=0.;periods=0;events=[]
        prior_failed=[slot.skill.done for slot in env.slots]
        for k in range(repeats):
            before,_,_=observe(env)
            if collect_raw:raw.append(before.copy())
            dense_obs.append(before);dense_actions.append([a0,a1]);decision_flags.append(k==0)
            boundary=env.joint_step(a0,a1,record=record);last=boundary
            if boundary['periods']<1:raise RuntimeError('zero-time action')
            dt=boundary['periods']*spec.CONTROL_DT;dense_dt.append(dt);dense_rewards.append(boundary['reward']['total'])
            periods+=boundary['periods'];step_reward+=boundary['reward']['total'];events+=boundary['execution'].get('events',[])
            rec=dict(time=env.total_periods*spec.CONTROL_DT,dt=dt,a0=a0,a1=a1,labels=[ACTIONS[a0].label,ACTIONS[a1].label],
                exits=[boundary['skill_ends'][i]['exit_reason'] for i in (0,1)],
                phases=[boundary['skill_ends'][i]['stage'] for i in (0,1)],events=boundary['execution'].get('events',[]))
            execution.append(rec)
            for i in (0,1):exits[rec['exits'][i]]+=1
            for key in ('progress','distance_cost','time_cost','success_bonus','failure_penalty'):components[key]+=boundary['reward'].get(key,0.)
            newly_failed=any(slot.skill.done and not was for slot,was in zip(env.slots,prior_failed))
            if boundary['termination'] is not None or newly_failed:break
        terminal=last['termination'] is not None
        steps.append(Step(encoded,a0,a1,float(logp.item()),float(value.item()),float(step_reward),m0.copy(),m1.copy(),terminal,periods))
        decisions.append(dict(rec,dt=periods*spec.CONTROL_DT,events=events))
        reward+=step_reward
        if terminal:break
    if env.termination is None:raise RuntimeError('physical-horizon guard exhausted')
    eligible=env.termination not in E.ANOMALY_TERMINATIONS
    summary=dict(name=instance.name,scene_hash=layout.content_hash(instance),success=bool(last['success']),
        progress=float(np.clip((initial-env.object_distance())/initial,-1.,1.)),initial_distance=initial,final_distance=env.object_distance(),
        seconds=env.total_periods*spec.CONTROL_DT,periods=env.total_periods,decisions=len(steps),termination=env.termination,
        eligible_for_ppo=eligible,components=dict(components),skill_exits=dict(exits),actions=decisions,execution_samples=execution,
        initial_geometry=scenes.geometry_audit(instance))
    summary['return']=float(reward)
    if dataset_dir is not None and instance.split=='train':
        from experiments.option_rl_v14.dataset import save_episode
        save_episode(dataset_dir,instance.index if sample_ordinal is None else sample_ordinal,dense_obs+[env.boundary_vector()],dense_actions,dense_dt,dense_rewards,
            decision_flags=decision_flags,scene_name=instance.name,policy_iteration=policy_iteration,context=config['context'])
    if record:summary['rows']=env.rows
    if collect_raw:summary['raw_observations']=np.stack(raw)
    return steps,summary


def returns_and_advantages(steps,*,method='gae',trace_seconds=10.,bootstrap=0.):
    if method not in ('mc','gae'):raise ValueError('unknown advantage estimator')
    if trace_seconds<=0:raise ValueError('positive physical trace time required')
    values=np.asarray([s.value for s in steps],dtype=np.float64)
    advantages=np.zeros(len(steps),dtype=np.float64);targets=np.zeros(len(steps),dtype=np.float64)
    running=float(bootstrap);trace=0.
    for i in range(len(steps)-1,-1,-1):
        step=steps[i];live=not step.terminal
        if method=='mc':
            running=step.reward+(running if live else 0.);targets[i]=running;advantages[i]=running-values[i]
        else:
            next_value=(values[i+1] if i+1<len(steps) else float(bootstrap)) if live else 0.
            delta=step.reward+next_value-values[i]
            lam=math.exp(-step.periods*spec.CONTROL_DT/trace_seconds)
            trace=delta+(lam*trace if live else 0.);advantages[i]=trace;targets[i]=trace+values[i]
    return targets.astype(np.float32),advantages.astype(np.float32)


def lifecycle_rates(summaries):
    records=[d for s in summaries for d in s.get('execution_samples',s['actions'])]
    n=max(1,2*len(records));states=[(e,p) for d in records for e,p in zip(d['exits'],d['phases'])]
    return dict(controller_failed_rate=sum(e!='continuing' for e,p in states)/n,
        peer_wait_rate=sum(p=='peer_wait' for e,p in states)/n,
        perception_wait_rate=sum(p=='perception_wait' for e,p in states)/n,
        failed_or_wait_rate=sum(e!='continuing' or p in ('peer_wait','perception_wait') for e,p in states)/n,
        changed_intent_retry_rate=sum(d['events'].count('changed_intent_retry_requested') for d in records)/n)


def ppo_update(policy,optimizer,steps,*,epochs=4,minibatch=256,entropy_coef=.02,clip=.2,value_coef=.5,
               advantage='gae',trace_seconds=10.,target_kl=.03):
    if not steps:return {'updates':0,'status':'no_valid_rollouts'}
    targets,advantages=returns_and_advantages(steps,method=advantage,trace_seconds=trace_seconds)
    stats=Counter(advantage_std=float(advantages.std()),return_target_std=float(targets.std()))
    if advantages.std()>1e-8:advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
    obs=torch.tensor(np.stack([s.obs for s in steps]));a=[torch.tensor([getattr(s,f'a{i}') for s in steps]) for i in (0,1)]
    masks=[torch.tensor(np.stack([getattr(s,f'mask{i}') for s in steps])) for i in (0,1)]
    old=torch.tensor([s.logp for s in steps]);target=torch.tensor(targets);adv=torch.tensor(advantages)
    with torch.no_grad():
        ds,_=policy.distributions(obs,masks,a0=a[0]);lp=sum(d.log_prob(v) for d,v in zip(ds,a))
        error=float((lp-old).abs().max())
        if error>2e-5:raise RuntimeError(f'rollout/PPO log-prob mismatch: {error}')
    stop=False
    for epoch in range(epochs):
        for idx in torch.randperm(len(steps)).split(minibatch):
            ds,value=policy.distributions(obs[idx],[m[idx] for m in masks],a0=a[0][idx])
            logp=sum(d.log_prob(v[idx]) for d,v in zip(ds,a));ratio=(logp-old[idx]).exp()
            kl=float((ratio-1-(logp-old[idx])).mean().detach())
            if target_kl and kl>1.5*target_kl:
                stats['kl_early_stop']=1.;stop=True;break
            actor=-torch.min(ratio*adv[idx],ratio.clamp(1-clip,1+clip)*adv[idx]).mean()
            critic=(value-target[idx]).square().mean();entropy=sum(d.entropy().mean() for d in ds)
            actor_loss=actor-entropy_coef*entropy;critic_loss=value_coef*critic
            if policy.critic_body is not None:
                optimizer['actor'].zero_grad();optimizer['critic'].zero_grad()
                actor_loss.backward();critic_loss.backward()
                ag=nn.utils.clip_grad_norm_(policy.actor_parameters(),1.)
                cg=nn.utils.clip_grad_norm_(policy.critic_parameters(),1.)
                if not all(torch.isfinite(v) for v in (actor_loss,critic_loss,ag,cg)):raise RuntimeError('nonfinite PPO')
                optimizer['actor'].step();optimizer['critic'].step();grad=max(float(ag),float(cg))
            else:
                optimizer['shared'].zero_grad()
                ga=torch.autograd.grad(actor_loss,list(policy.parameters()),retain_graph=True,allow_unused=True)
                gc=torch.autograd.grad(critic_loss,list(policy.parameters()),retain_graph=True,allow_unused=True)
                ag=math.sqrt(sum(float(g.square().sum()) for g in ga if g is not None))
                cg=math.sqrt(sum(float(g.square().sum()) for g in gc if g is not None))
                (actor_loss+critic_loss).backward()
                grad=float(nn.utils.clip_grad_norm_(policy.parameters(),1.))
                if not math.isfinite(grad):raise RuntimeError('nonfinite shared gradient')
                optimizer['shared'].step()
            stats['actor_grad_norm']+=float(ag);stats['critic_grad_norm']+=float(cg)
            stats['approx_kl']+=kl;stats['clip_fraction']+=float(((ratio-1).abs()>clip).float().mean().detach())
            stats['policy_loss']+=float(actor.detach());stats['value_loss']+=float(critic.detach());stats['entropy']+=float(entropy.detach())
            stats['grad_norm_max']=max(stats['grad_norm_max'],grad);stats['updates']+=1
        if stop:break
    for key in ('actor_grad_norm','critic_grad_norm','approx_kl','clip_fraction','policy_loss','value_loss','entropy'):
        stats[key]/=max(1,stats['updates'])
    with torch.no_grad():
        _,v=policy.distributions(obs,masks,a0=a[0]);var=float(target.var(unbiased=False))
        stats['explained_variance']=1-float((target-v).var(unbiased=False))/var if var>1e-8 else 0.
    stats['actor_parameter_grad_norm']=stats['actor_grad_norm'];stats['critic_parameter_grad_norm']=stats['critic_grad_norm']
    return dict(stats,initial_logp_max_error=error)


def evaluate(policy,normalizer,instances,*,stochastic=False,seed=0,on_root=None,should_stop=None,executor=None):
    generator=torch.Generator().manual_seed(seed);summaries=[]
    if executor is not None:
        from experiments.option_rl_v14.parallel import evaluation_summaries
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
    policy=Policy(data['obs_dim'],data['hidden'],**data['model_config']);policy.rollout_config=data['rollout_config'];policy.load_state_dict(data['state_dict']);policy.eval()
    normalizer=RunningMeanStd(data['obs_dim']);normalizer.mean=data['obs_mean'];normalizer.var=data['obs_var'];normalizer.count=data['obs_count']
    return policy,normalizer,data


def train(*,out,iterations=10000,episodes=8,hours=1.,seed=0,eval_roots=64,eval_every=10,
          hidden=256,epochs=4,lr=3e-4,entropy=.02,render_every=0,workers=1,resume_from=None,
          separate=True,advantage='gae',trace_seconds=10.,hold_seconds=.8,context='extended',encoder='flat',
          autoregressive=False,target_kl=.03,panel='dynamic',panel_path=None):
    if min(iterations,episodes,eval_every,hidden,epochs,workers)<=0 or hours<=0 or eval_roots<0:
        raise ValueError('positive budgets and nonnegative eval roots required')
    if hold_seconds<.2 or abs(hold_seconds/.2-round(hold_seconds/.2))>1e-8:raise ValueError('hold must be a multiple of .2 seconds')
    if resume_from is not None:raise ValueError('v14 diagnostics start fresh; cross-variant resume disabled')
    from functools import partial
    env_factory=partial(make_env,context=context)
    model_config=dict(separate=separate,encoder=encoder,autoregressive=autoregressive)
    rollout_config=dict(context=context,hold_seconds=hold_seconds)
    out=Path(out).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('new baseline requires an empty output directory')
    out.mkdir(parents=True,exist_ok=True);(out/'evaluations').mkdir();(out/'rollouts').mkdir()
    torch.set_num_threads(1);torch.manual_seed(seed);generator=torch.Generator().manual_seed(seed+701)
    validations=scenes.panel(seed,eval_roots,'validation');tests=scenes.panel(seed,eval_roots,'test')
    dev_train=None
    if panel=='micro':
        from experiments.option_rl_v14.learnability import load_panel
        dev_train,validations=load_panel(panel_path)
        tests=[]
    elif panel!='dynamic':raise ValueError('unknown training panel')
    split_audit=layout.audit_split_content(dict(validation=validations,test=tests,**({'development_train':dev_train} if dev_train else {})))
    if not split_audit['ok']:raise ValueError('fixed panels overlap')
    frozen={layout.content_hash(i) for i in validations+tests}
    manifest=dict(metadata(),dense_dataset='5Hz boundary sequences, every31 episode occurrences / last256 selected episodes, atomic NPZ; 10Hz intra-mode interface available but not enabled; world-model training not enabled',scene_version=scenes.VERSION,seed=seed,training_scenes=('fixed developer overfit panel; not generalization' if panel=='micro' else 'one fresh indexed root per episode attempt'),
        evaluation_scenes='fixed new legal roots; different from run9',goal_distribution='uniform full random; no annealing or BC',
        gamma=1.,anomaly_rule='record all; quarantine physical-anomaly trajectories from PPO; all eval roots count',
        normalization_protocol='serial-bootstrap-then-batch-frozen-v1' if workers>1 else 'legacy-online-v1',
        max_attempts_per_iteration=episodes*3,config=locals_config(iterations,episodes,hours,eval_roots,eval_every,hidden,epochs,lr,entropy,render_every,workers),
        validation_instances=[i.as_dict() for i in validations],test_instances=[i.as_dict() for i in tests],split_audit=split_audit)
    manifest['config'].update(model_config=model_config,rollout_config=rollout_config,advantage=advantage,trace_seconds=trace_seconds,target_kl=target_kl,panel=panel,panel_path=str(panel_path) if panel_path else None)
    manifest['model_config']=model_config;manifest['rollout_config']=rollout_config
    manifest['normalization_protocol']='reset warmup then frozen per-batch; update once after PPO'
    if panel=='micro':
        manifest.update(scene_version='declared-developer-panel-v1',goal_distribution='fixed 0.45m developer goals; no task reward changes',evaluation_scenes='fixed 1cm tangent perturbations; development only')
    manifest['development_train_instances']=[i.as_dict() for i in dev_train] if dev_train else []
    sources=[Path(__file__),Path(__file__).with_name('interface.py'),Path(__file__).with_name('scenes.py'),Path(__file__).with_name('monitor.py'),Path(__file__).with_name('live.py'),Path(__file__).with_name('parallel.py')]
    sources += [ROOT/'src/coimagine/random_start'/f'{name}.py' for name in ('env','spec','skills','effect_push','boundary_follow','endpoint_tracking','same_face','lidar','progress_guard','efficient_profile','peer_clearance')]
    sources += [Path(__file__).with_name('dataset.py'),Path(__file__).with_name('mode_env.py'),ROOT/'experiments/option_rl_v13/push_controller.py',Path(__file__).with_name('controller.py'),ROOT/'src/coimagine/continuous2d/environment.py',ROOT/'src/coimagine/continuous2d/backend.py',ROOT/'src/coimagine/continuous2d/config.py',ROOT/'src/coimagine/local_contact_option/control.py']
    sources += [ROOT/'experiments/option_rl_v13'/f'{name}.py' for name in ('controller','mode_env','interface')]
    sources += [Path(__file__).with_name('learnability.py'),ROOT/'experiments/option_rl_v45/train.py']
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest['reward_config']=E.EnvConfig(lidar_backend='numpy_batch',peer_clearance_recovery='boundary_v1').as_dict()
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    first=dev_train[0] if dev_train else scenes.scene(seed,0);obs_dim=len(observe(env_factory(first))[0]);normalizer=RunningMeanStd(obs_dim)
    warmup=dev_train or [scenes.scene(seed,i) for i in range(64)]
    normalizer.update(np.stack([observe(env_factory(i))[0] for i in warmup]))
    policy=Policy(obs_dim,hidden,**model_config);policy.rollout_config=rollout_config
    optimizer=({'actor':torch.optim.Adam(policy.actor_parameters(),lr=lr),'critic':torch.optim.Adam(policy.critic_parameters(),lr=lr)}
        if separate else {'shared':torch.optim.Adam(policy.parameters(),lr=lr)})
    history=[];ordinal=0;started=time.monotonic();best=None;status='complete'
    def save(path):
        data=dict(metadata(),state_dict=policy.state_dict(),optimizer={k:v.state_dict() for k,v in optimizer.items()},model_config=model_config,rollout_config=rollout_config,obs_dim=obs_dim,hidden=hidden,
                  obs_mean=normalizer.mean,obs_var=normalizer.var,obs_count=normalizer.count,history=history,
                  next_scene_ordinal=ordinal,torch_rng=torch.get_rng_state(),sampling_rng=generator.get_state(),manifest=manifest)
        temp=path.with_suffix('.tmp');torch.save(data,temp);temp.replace(path)
    def persist():
        atomic_json(out/'history.json',history)
        write_dashboard(history,out/'dashboard.html',subtitle=f'{out} | v14 | dense5Hz / held actor | gamma=1')
    live_status(out,phase='starting',iteration=0,episodes=episodes)
    persist()
    from experiments.option_rl_v14.parallel import pool
    executor=pool(workers) if workers>1 else None
    try:
        for iteration in range(1,iterations+1):
            if stop_reason(out):status='requested_stop';break
            iteration_started=time.monotonic();valid=[];attempts=[]
            raw_observations=[]
            if executor is not None and normalizer.count>1e-3:
                from experiments.option_rl_v14.parallel import training_batch
                valid,attempts,ordinal,raw_observations=training_batch(executor,policy,normalizer,
                    seed=seed,ordinal=ordinal,episodes=episodes,frozen=frozen,out=out,iteration=iteration,workers=workers,development_instances=dev_train)
            else:
                for attempt in range(episodes*3):
                    if stop_reason(out):break
                    live_status(out,phase='rollout',iteration=iteration,episode=len(valid),episodes=episodes,attempt=attempt)
                    instance=(dev_train[ordinal%len(dev_train)] if dev_train else scenes.scene(seed,ordinal));ordinal+=1
                    if layout.content_hash(instance) in frozen:raise RuntimeError('dynamic training root duplicates fixed holdout')
                    steps,summary=run_episode(policy,instance,normalizer,generator=generator,update_stats=False,collect_raw=True,dataset_dir=out/'world_model_replay',policy_iteration=iteration,sample_ordinal=ordinal-1)
                    raw_observations.append(summary.pop('raw_observations'))
                    atomic_json(out/'rollouts'/f'{ordinal-1:07d}.json',dict(instance=instance.as_dict(),summary=summary))
                    attempts.append(summary)
                    if summary['eligible_for_ppo']:valid.append((steps,summary))
                    live_status(out,phase='rollout',iteration=iteration,episode=len(valid),episodes=episodes,attempt=attempt+1)
                    if len(valid)>=episodes:break
            if len(valid)<episodes:
                status='requested_stop' if stop_reason(out) else 'insufficient_valid_rollouts'
                save(out/'checkpoint.pt');break
            rollout_wall_seconds=time.monotonic()-iteration_started
            live_status(out,phase='ppo',iteration=iteration,episodes=episodes,episode=len(valid))
            stats=ppo_update(policy,optimizer,[s for ep,_ in valid for s in ep],epochs=epochs,entropy_coef=entropy,advantage=advantage,trace_seconds=trace_seconds,target_kl=target_kl)
            if raw_observations:normalizer.update(np.concatenate(raw_observations,axis=0))
            outside=[s for s in attempts if s['initial_distance']>spec.GOAL_RADIUS]
            all_actions=[a for summary in attempts for d in summary.get('execution_samples',summary['actions']) for a in (d['a0'],d['a1'])]
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
            mode_decisions=[d for summary in attempts for d in summary.get('execution_samples',summary['actions'])]
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
            entry['dense_samples']=sum(len(s['execution_samples']) for s in attempts)
            entry['simulated_seconds']=sum(s['seconds'] for s in attempts)
            entry['persistence_bias']=float(policy.persistence_bias.detach())
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
                if dev_train:
                    dev_eval=evaluate(policy,normalizer,dev_train,executor=executor)
                    evaluation['development_training']=dev_eval
                    entry.update(dev_train_success=dev_eval['outside_goal_success'],dev_train_progress=dev_eval['mean_progress'])
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
                from experiments.option_rl_v14.render_episodes import render_episodes
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
    p.add_argument('--hidden',type=int,default=256)
    p.add_argument('--shared',dest='separate',action='store_false')
    p.add_argument('--advantage',choices=['mc','gae'],default='gae');p.add_argument('--trace-seconds',type=float,default=10.)
    p.add_argument('--hold-seconds',type=float,default=.8);p.add_argument('--target-kl',type=float,default=.03)
    p.add_argument('--context',choices=['legacy','extended'],default='extended')
    p.add_argument('--encoder',choices=['flat','scan'],default='flat');p.add_argument('--autoregressive',action='store_true')
    p.add_argument('--panel',choices=['dynamic','micro'],default='dynamic');p.add_argument('--panel-path',type=Path)
    args=vars(p.parse_args());print(json.dumps(train(**args)),flush=True)
if __name__=='__main__':main()
