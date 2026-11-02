"""Persistent spawn workers, frozen per-batch policy/normalizer, ordered PPO data."""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,multiprocessing as mp,os,time
import numpy as np
import torch
from experiments.option_rl_v15.interface import Policy
from experiments.option_rl_v15 import scenes
from experiments.option_rl_v15.live import atomic_json,status,stop_reason
from coimagine.random_start import layout

_CACHE={}


def worker_init():
    torch.set_num_threads(1)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass


def pool(workers):
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    return ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=worker_init)


def snapshot(policy,normalizer):
    state={k:v.detach().cpu().numpy().copy() for k,v in policy.state_dict().items()}
    digest=hashlib.sha256()
    for k,v in state.items():digest.update(k.encode());digest.update(v.tobytes())
    digest.update(normalizer.mean.tobytes());digest.update(normalizer.var.tobytes())
    digest.update(repr(policy.model_config).encode());digest.update(repr(policy.rollout_config).encode())
    return dict(state=state,hidden=policy.hidden,model_config=policy.model_config,rollout_config=policy.rollout_config,obs_dim=len(normalizer.mean),
        mean=normalizer.mean.copy(),var=normalizer.var.copy(),count=normalizer.count,hash=digest.hexdigest())


def behaviour_seed(seed,index,stream=0):
    return int(np.random.SeedSequence([seed,index,909,stream]).generate_state(1,dtype=np.uint64)[0]%(1<<63))


def rollout_job(job):
    from experiments.option_rl_v15.train import run_episode
    from experiments.option_rl_v45.train import RunningMeanStd
    started=time.monotonic();s=job['snapshot']
    if _CACHE.get('hash')!=s['hash']:
        with torch.random.fork_rng(devices=[]):policy=Policy(s['obs_dim'],s['hidden'],**s['model_config']);policy.rollout_config=s['rollout_config']
        policy.load_state_dict({k:torch.from_numpy(v.copy()) for k,v in s['state'].items()});policy.eval()
        stats=RunningMeanStd(s['obs_dim']);stats.mean=s['mean'].copy();stats.var=s['var'].copy();stats.count=s['count']
        _CACHE.update(hash=s['hash'],policy=policy,stats=stats)
    kwargs={}
    if job.get('dataset_dir'):kwargs.update(dataset_dir=job['dataset_dir'],policy_iteration=job.get('iteration',0),sample_ordinal=job.get('sample_ordinal'))
    if job.get('diagnostic_time_budget') is not None:
        from dataclasses import replace
        from experiments.option_rl_v15.interface import make_env
        def factory(instance):
            env=make_env(instance);env.config=replace(env.config,time_budget=job['diagnostic_time_budget']);return env
        kwargs['env_factory']=factory
    steps,summary=run_episode(_CACHE['policy'],job['instance'],_CACHE['stats'],
        generator=torch.Generator().manual_seed(job['seed']),deterministic=job.get('deterministic',False),
        update_stats=False,collect_raw=job.get('collect_raw',False),**kwargs)
    raw=summary.pop('raw_observations',None)
    summary.update(worker_pid=os.getpid(),behaviour_seed=job['seed'],policy_snapshot=s['hash'])
    return dict(steps=steps,summary=summary,raw_observations=raw,wall_seconds=time.monotonic()-started)


def collect(executor,jobs,*,on_result=None,should_stop=None,timeout=300):
    if executor is None:
        results=[]
        for job in jobs:
            if should_stop and should_stop():break
            value=rollout_job(job);results.append(value)
            if on_result:on_result(len(results),len(jobs),value)
        return results
    futures={executor.submit(rollout_job,j):i for i,j in enumerate(jobs)}
    values=[None]*len(jobs);done=0
    for f in as_completed(futures,timeout=timeout):
        if should_stop and should_stop():
            for pending in futures:pending.cancel()
        if f.cancelled():continue
        value=f.result();values[futures[f]]=value;done+=1
        if on_result:on_result(done,len(jobs),value)
    # Completion order only affects heartbeat. PPO batch order is the declared root order.
    return [v for v in values if v is not None]


def training_batch(executor,policy,normalizer,*,seed,ordinal,episodes,frozen,out,iteration,workers,development_instances=None):
    valid=[];attempts=[];raw=[];snap=snapshot(policy,normalizer)
    while len(valid)<episodes and len(attempts)<episodes*3 and not stop_reason(out):
        count=min(episodes-len(valid),episodes*3-len(attempts));instances=[]
        for _ in range(count):
            instance=(development_instances[ordinal%len(development_instances)] if development_instances else scenes.scene(seed,ordinal));ordinal+=1
            if layout.content_hash(instance) in frozen:raise RuntimeError('parallel training root duplicates fixed holdout')
            instances.append(instance)
        jobs=[dict(instance=i,snapshot=snap,seed=behaviour_seed(seed,ordinal-count+j),collect_raw=True,dataset_dir=str(out/'world_model_replay'),iteration=iteration,sample_ordinal=ordinal-count+j) for j,i in enumerate(instances)]
        prior=len(valid);completed_valid=0
        def progress(done,total,result):
            nonlocal completed_valid
            completed_valid+=int(result['summary']['eligible_for_ppo'])
            status(out,phase='parallel_rollout',iteration=iteration,episode=prior+completed_valid,episodes=episodes,
                   jobs_done=done,jobs_total=total,workers=workers)
        status(out,phase='parallel_rollout',iteration=iteration,episode=prior,episodes=episodes,jobs_done=0,jobs_total=count,workers=workers)
        results=collect(executor,jobs,on_result=progress)
        for attempt_offset,(instance,result) in enumerate(zip(instances,results)):
            summary=result['summary'];atomic_json(out/'rollouts'/f'{ordinal-count+attempt_offset:07d}.json',dict(instance=instance.as_dict(),summary=summary))
            attempts.append(summary)
            if summary['eligible_for_ppo']:
                valid.append((result['steps'],summary));raw.append(result['raw_observations'])
    return valid,attempts,ordinal,raw


def evaluation_summaries(executor,policy,normalizer,instances,*,stochastic,seed,on_root,should_stop):
    snap=snapshot(policy,normalizer)
    jobs=[dict(instance=i,snapshot=snap,seed=behaviour_seed(seed,i.index,1),deterministic=not stochastic) for i in instances]
    def progress(done,total,result):
        if on_root:on_root(done,total,result['summary'])
    return [r['summary'] for r in collect(executor,jobs,on_result=progress,should_stop=should_stop)]
