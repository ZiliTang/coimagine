"""16-process synchronous PPO with immutable rollout versions and live metrics.

Explicit invocation only. The parent is a frozen perturbation campaign; only its
TRAIN panel is admitted. Test outcomes never enter sampling or model selection.
"""
import argparse
import ast
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from dataclasses import dataclass,asdict
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import random
import shutil
import time
import traceback
import torch
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save,pack,unpack
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.schema import Schema,migrate
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.fast_features import VERSION as FEATURE_RUNTIME_VERSION
from experiments.planar_tasks.semantic_rl.batched_ppo import ppo_batch_update
from experiments.planar_tasks.semantic_rl.memory_flow import compact_step
from experiments.planar_tasks.semantic_rl.save_timing import SaveTiming


@dataclass(frozen=True)
class Config:
    workers:int=16
    rounds:int=128
    epochs:int=4
    minibatch_size:int=64
    max_ticks:int=1800
    max_decisions:int=600
    episode_seconds:float=600.
    wall_seconds:float=21600.
    temperature:float=2.
    seed:int=2026111404
    semantic_capacity:int=64
    feature_runtime:str="cached_fork"
    execution_mode:str="progress"
    execution_backend:str="isolated"
    detail_trials:int=0
    detail_every:int=0

    def __post_init__(self):
        for name in ('workers','rounds','epochs','minibatch_size','max_ticks','max_decisions','semantic_capacity'):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:raise ValueError('positive integer '+name)
        if self.feature_runtime not in ('legacy','cached_fork','typed_batch','typed_stream'):raise ValueError('feature runtime')
        if self.execution_backend not in ('isolated','trusted_inprocess'):raise ValueError('semantic execution backend')
        EventConfig(mode=self.execution_mode)
        for name in ('detail_trials','detail_every'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:raise ValueError('nonnegative integer '+name)
        if self.workers>32 or not 1<=self.semantic_capacity<=128:raise ValueError('worker/capacity limit')
        if not all(0<v<float('inf') for v in (self.episode_seconds,self.wall_seconds,self.temperature)):
            raise ValueError('positive finite time/temperature')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def names_from_source(source):
    for node in ast.parse(source).body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in node.targets):
            return list(ast.literal_eval(node.value))
    raise ValueError('parallel author contract requires literal NAMES')


def worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def collect(job):
    """Separate process, local physics/feature state, only public policy inputs."""
    p=Path(job['path']);source=Path(job['source']).read_text()
    # Missing keys mean an old, explicitly replayed disk-based request.
    memory=job.get('batch_transport','disk')=='memory'
    if job.get('batch_transport','disk') not in ('disk','memory'):raise ValueError('batch transport')
    details=job.get('record_details',not memory)
    if type(details) is not bool:raise ValueError('boolean record_details required')
    io=SaveTiming()
    io.log(job['id']+' start',details=details,batch_transport='memory' if memory else 'disk')
    if digest(source)!=job['semantic_sha256'] or sha(job['checkpoint'])!=job['checkpoint_sha256']:
        raise ValueError('rollout version mismatch')
    model,_=Policy.load(job['checkpoint'],job['semantic_sha256'])
    features=Features(source,schema=Schema(tuple(model.schema['slots'])),fast_embedding=True,
                      runtime=job.get('feature_runtime','legacy'),action_protocol=model.action_protocol,
                      execution_backend=job.get('execution_backend','isolated'),
                      trusted_source_sha256=job.get('trusted_source_sha256'))
    io.call(save,p/'worker.json',dict(pid=os.getpid(),started=time.time(),policy_version=job['policy_version']))
    def progress(v):
        if v['index']%5==0 or v['reason']=='terminal':io.call(save,p/'progress.json',dict(v,updated=time.time(),pid=os.getpid()))
    try:
        result,batch=rollout(job['case']['fixture'],source,model=model,features=features,config=EventConfig(mode=job.get('execution_mode','progress')),
               max_ticks=job['max_ticks'],max_decisions=job['max_decisions'],wall_seconds=job['episode_seconds'],
               seed=job['seed'],temperature=job['temperature'],on_progress=progress,fast_feedback=features.runtime!='legacy',
               record_details=details,training_fastpath=memory)
    finally:
        features.close()
    result.update(case_id=job['case']['id'],split='train',policy_version=job['policy_version'],
                  checkpoint_sha256=job['checkpoint_sha256'],feature_batches=features.calls,feature_timing=features.timing)
    if features.runtime in ('typed_batch','typed_stream'):
        result.update(feature_runtime=features.runtime,feature_compilation=features.worker.compilation,
                      semantic_execution_backend=features.execution_backend)
    if details:io.call(pack,p/'result.json.gz',result,compresslevel=1)
    if details or not memory:io.call(pack,p/'batch.json.gz',batch,compresslevel=1)
    summary=dict(result['summary'],id=job['id'],case_id=job['case']['id'],family=job['case']['family'],split='train',
                 policy_version=job['policy_version'],checkpoint_sha256=job['checkpoint_sha256'],
                 feature_batches=features.calls,feature_timing=features.timing,pid=os.getpid(),
                 record_details=details,batch_transport='memory' if memory else 'disk',save_seconds=io.seconds)
    io.call(save,p/'summary.json',summary)
    summary['save_seconds']=io.seconds
    io.log(job['id']+' complete',details=details,batch_transport=summary['batch_transport'])
    if memory:
        return dict(summary=summary,episode=dict(steps=[compact_step(s) for s in batch] if details else batch,
                                                bootstrap=summary['bootstrap']))
    return summary


def record_trial(cfg,number):
    return number<cfg.detail_trials or bool(cfg.detail_every and number%cfg.detail_every==0)


def create(path,parent,cfg,*,resume_from=None):
    parent=Path(parent).resolve();parent_seal=Campaign(parent).verify_frozen()
    source=(parent/'inputs/source.py').read_text();panel=read(parent/'panels/train.json')
    if not panel['cases'] or any(c['split']!='train' or digest(c['fixture'])!=c['fixture_sha256'] for c in panel['cases']):
        raise ValueError('only registered training fixtures admitted')
    if cfg.execution_backend=='trusted_inprocess':
        from experiments.planar_tasks.semantic_rl.typed_compiler import stream_config
        stream_config(source)
    old,meta=Policy.load(parent/'inputs/initial.pt',digest(source))
    expected=protocol(EventConfig(mode=cfg.execution_mode)) if cfg.execution_mode=='duration' else None
    if old.action_protocol != expected:raise ValueError('parent action-scale protocol mismatch; recollect and initialize first')
    model,migration=migrate(old,names_from_source(source),digest(source),capacity=cfg.semantic_capacity)
    continuation=None
    if resume_from is not None:
        from .semantic_rl_resume import load_resume
        model,continuation,resume_model,resume_optimizer=load_resume(resume_from,source,panel,cfg,model)
    n=cfg.rounds*cfg.workers
    c=Campaign.create(path,limits=dict(attempts=n,resets=n,execution_ticks=n*cfg.max_ticks,rounds=cfg.rounds,fits=0,author_turns=0),
                      model=dict(provider='none',name='synchronous semantic PPO'))
    p=c.path;(p/'inputs').mkdir();(p/'models').mkdir();(p/'runs').mkdir()
    (p/'inputs/source.py').write_text(source);shutil.copyfile(parent/'inputs/initial.pt',p/'inputs/legacy_initial.pt')
    save(p/'panels/train.json',panel);save(p/'inputs/migration.json',migration)
    model.save(p/'models/initial.pt',dict(meta,migration=migration))
    if continuation:
        shutil.copyfile(resume_model,p/'models/initial.pt')
        shutil.copyfile(resume_optimizer,p/'inputs/resume_optimizer.pt')
    from .semantic_rl_resume import schedule as make_schedule
    episode_offset=continuation['episode_offset'] if continuation else 0
    schedule=make_schedule(panel['cases'],cfg.seed,n+episode_offset)[episode_offset:]
    save(p/'inputs/provenance.json',dict(parent=str(parent),parent_seal=parent_seal,source_sha256=sha(p/'inputs/source.py'),
           panel_sha256=digest(panel),initial_checkpoint_sha256=sha(p/'models/initial.pt'),schema=model.schema,
           selection='fixed final checkpoint; no validation/test selection',continuation=continuation))
    from experiments.planar_tasks.semantic_rl.typed_compiler import source_api,STREAM_API,VERSION as TYPED_VERSION
    effective_runtime=('typed_stream' if source_api(source)==STREAM_API else 'typed_batch') if source_api(source) else cfg.feature_runtime
    s=read(p/'campaign.json');s.update(profile='semantic-ppo-parallel-v3',feature_runtime_version='typed-stream-v1' if effective_runtime=='typed_stream' else TYPED_VERSION if effective_runtime=='typed_batch' else FEATURE_RUNTIME_VERSION,action_protocol=expected,config=dict(asdict(cfg),feature_runtime=effective_runtime),runs=[],training_schedule=schedule[:n],
          completed_rounds=0,phase='ready',completion=None,deadline=time.time()+cfg.wall_seconds,continuation=continuation)
    s['used']['feature_batches']=0;s['limits']['feature_batches']=n*((5*cfg.max_decisions+2) if effective_runtime=='typed_stream' else (4*cfg.max_decisions+1))
    save(p/'campaign.json',s);save(p/'metrics.json',dict(rounds=[],updates=[]))
    return c,model


def run(path,parent,cfg,*,resume_from=None,on_round=None):
    torch.set_num_threads(1)
    io=SaveTiming()
    io.log('training start',batch_transport='memory',detail_trials=cfg.detail_trials,detail_every=cfg.detail_every)
    setup_started=time.perf_counter()
    c,model=create(path,parent,cfg,resume_from=resume_from);p=c.path;state=read(p/'campaign.json');provenance=read(p/'inputs/provenance.json')
    io.log('initialization complete',setup_seconds=time.perf_counter()-setup_started,
           setup_includes_source_snapshot_and_model=True)
    cases={r['id']:r for r in read(p/'panels/train.json')['cases']}
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);metrics=dict(rounds=[],updates=[],diagnostics=[]);checkpoint='models/initial.pt'
    continuation=state.get('continuation') or {}
    round_offset=continuation.get('round_offset',0);episode_offset=continuation.get('episode_offset',0)
    update_offset=continuation.get('optimizer_step_offset',0)
    if continuation:
        restored=torch.load(p/'inputs/resume_optimizer.pt',map_location='cpu',weights_only=True)
        optimizer.load_state_dict(restored['optimizer']);torch.set_rng_state(restored['rng'])
    started=time.monotonic();completion='complete'
    def write(*args,**kwargs):return io.call(save,*args,**kwargs)
    try:
        # spawn isolates module globals in the physics controller; one torch thread each.
        with ProcessPoolExecutor(max_workers=cfg.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as pool:
            for index in range(cfg.rounds):
                if state['deadline']-time.time()<cfg.episode_seconds+30:
                    completion='incomplete_wall_budget';break
                c.verify_sources()
                if sha(p/'inputs/source.py')!=provenance['source_sha256'] or digest(read(p/'panels/train.json'))!=provenance['panel_sha256']:
                    raise ValueError('immutable input changed')
                state.update(phase='collecting',active_round=index,policy_checkpoint=checkpoint)
                cksha=sha(p/checkpoint);jobs=[];batch_start=time.monotonic()
                for worker in range(cfg.workers):
                    number=index*cfg.workers+worker;rid=f'a{number:05d}';directory=p/'runs'/rid;directory.mkdir()
                    cid=state['training_schedule'][number];job=dict(id=rid,path=str(directory),source=str(p/'inputs/source.py'),
                         checkpoint=str(p/checkpoint),checkpoint_sha256=cksha,semantic_sha256=model.semantic_sha256,
                         policy_version=index+round_offset,case=cases[cid],seed=cfg.seed+episode_offset+number,max_ticks=cfg.max_ticks,max_decisions=cfg.max_decisions,
                         episode_seconds=cfg.episode_seconds,temperature=cfg.temperature,feature_runtime=state['config']['feature_runtime'],execution_mode=cfg.execution_mode,
                         batch_transport='memory',record_details=record_trial(cfg,number),
                         execution_backend=cfg.execution_backend,trusted_source_sha256=model.semantic_sha256 if cfg.execution_backend=='trusted_inprocess' else None)
                    write(directory/'request.json',job);jobs.append(job)
                    state['runs'].append(dict(id=rid,status='running',case_id=cid,family=cases[cid]['family'],split='train',policy_version=index+round_offset))
                    feature_reserve=5*cfg.max_decisions+2 if state['config']['feature_runtime']=='typed_stream' else 4*cfg.max_decisions+1
                    state['reservations'][rid]=dict(execution_ticks=cfg.max_ticks,feature_batches=feature_reserve)
                    for key,amount in dict(attempts=1,resets=1,execution_ticks=cfg.max_ticks,feature_batches=feature_reserve).items():state['used'][key]+=amount
                write(p/'campaign.json',state)
                futures={pool.submit(collect,j):j for j in jobs};completed=[];failed=[];buffers={}
                while futures:
                    done,_=wait(futures,timeout=1,return_when=FIRST_COMPLETED)
                    for future in done:
                        job=futures.pop(future);rid=job['id']
                        try:
                            payload=future.result();rec=payload['summary']
                            buffers[rid]=payload['episode']
                            del payload
                            if rec['checkpoint_sha256']!=cksha or rec['policy_version']!=index+round_offset:raise ValueError('stale worker result')
                        except Exception as exc:
                            rec=dict(id=rid,case_id=job['case']['id'],status='infrastructure_error',error=str(exc),policy_version=index+round_offset)
                            write(p/'runs'/rid/'error.json',rec);failed.append(rec)
                        else:
                            reservation=state['reservations'].pop(rid)
                            state['used']['execution_ticks']+=rec['ticks']-reservation['execution_ticks']
                            state['used']['feature_batches']+=rec['feature_batches']-reservation['feature_batches']
                            completed.append(rec)
                        state['runs'][index*cfg.workers+int(rid[1:])%cfg.workers]=rec
                        write(p/'campaign.json',state)
                    if done:
                        done.clear()
                        del future
                if failed:raise RuntimeError('worker failure; reservations retained, no retry or PPO for partial batch')
                collection_seconds=time.monotonic()-batch_start;state['phase']='updating';write(p/'campaign.json',state)
                episodes=[]
                for rec in sorted(completed,key=lambda r:r['id']):
                    if rec['eligible_for_ppo']:
                        episodes.append(buffers.pop(rec['id']))
                buffers.clear()
                count_before=len(metrics['updates'])
                last_metrics_write=time.monotonic()
                def on_update(row):
                    nonlocal last_metrics_write
                    metrics['updates'].append(dict(row,optimizer_step=len(metrics['updates'])+1,round=index,policy_version=index+round_offset,
                        cumulative_optimizer_step=update_offset+len(metrics['updates'])+1))
                    if time.monotonic()-last_metrics_write>=2.:
                        write(p/'metrics.json',metrics);last_metrics_write=time.monotonic()
                update=ppo_batch_update(model,optimizer,episodes,epochs=cfg.epochs,minibatch_size=cfg.minibatch_size,
                                        seed=cfg.seed+round_offset+index,on_update=on_update)
                checkpoint=f'models/round_{index:04d}.pt';io.call(model.save,p/checkpoint,dict(round=index,config=asdict(cfg),source='parallel PPO'))
                io.call(torch.save,dict(optimizer=optimizer.state_dict(),rng=torch.get_rng_state()),p/'models'/f'optimizer_{index:04d}.pt')
                total_decisions=sum(r['decisions'] for r in completed);total_ticks=sum(r['ticks'] for r in completed)
                metrics['rounds'].append(dict(round=index,episodes=len(completed),success=sum(r['success'] for r in completed),
                     success_rate=sum(r['success'] for r in completed)/len(completed),wall_truncations=sum(r['stop']=='wall_budget' for r in completed),
                     tool_errors=sum(bool(r['error']) for r in completed),ticks=total_ticks,decisions=total_decisions,
                     collection_seconds=collection_seconds,decisions_per_second=total_decisions/collection_seconds,
                     ticks_per_second=total_ticks/collection_seconds,mean_cost=sum(r['task_cost_seconds'] if r['success'] else 180. for r in completed)/len(completed),
                     updates=update['updates'],logp_recompute_error=update['logp_recompute_error'],early_stop_kl=update['early_stop_kl'],
                     update_seconds=update['wall_seconds'],elapsed_seconds=time.monotonic()-started,
                     total_optimizer_steps=len(metrics['updates']),checkpoint=checkpoint,checkpoint_sha256=sha(p/checkpoint),
                     detail_trials=sum(r['record_details'] for r in completed),
                     worker_save_seconds=sum(r['save_seconds'] for r in completed),host_save_seconds=io.seconds,
                     feature_sandbox_seconds=sum(r['feature_timing']['sandbox_seconds'] for r in completed),
                     feature_embedding_seconds=sum(r['feature_timing']['embedding_seconds'] for r in completed)))
                state['completed_rounds']=index+1;state['used']['rounds']=index+1
                metrics['rounds'][-1].update(global_round=round_offset+index,
                    cumulative_episodes=episode_offset+(index+1)*cfg.workers,
                    cumulative_optimizer_steps=update_offset+len(metrics['updates']),
                    family_outcomes={family:dict(episodes=sum(r['family']==family for r in completed),
                        success=sum(r['family']==family and r['success'] for r in completed)) for family in {r['family'] for r in completed}})
                for key in ('recommended','duration_only_deviation','motion_deviation'):
                    metrics['rounds'][-1][key+'_fraction']=sum(r.get('choice_counts',{}).get(key,0) for r in completed)/max(1,total_decisions)
                write(p/'metrics.json',metrics);write(p/'campaign.json',state)
                episodes.clear()
                if on_round is not None:
                    rng_state=torch.get_rng_state();diagnostic_started=time.monotonic()
                    try:
                        diagnostic=on_round(p,p/checkpoint,round_offset+index+1)
                        if diagnostic is not None:metrics['diagnostics'].append(diagnostic)
                    finally:torch.set_rng_state(rng_state)
                    metrics['rounds'][-1]['diagnostic_seconds']=time.monotonic()-diagnostic_started
                    write(p/'metrics.json',metrics)
                io.log('round '+str(index+1),details=sum(r['record_details'] for r in completed),worker_save_seconds=sum(r['save_seconds'] for r in completed),host_cost='cumulative')
                print('round',index+1,'success',sum(r['success'] for r in completed),'/',len(completed),'steps',len(metrics['updates']),flush=True)
        io.call(shutil.copyfile,p/checkpoint,p/'models/final.pt')
        state.update(phase='finished',completion=completion,final_checkpoint_sha256=sha(p/'models/final.pt'))
        write(p/'campaign.json',state)
        freeze_started=time.perf_counter();result=c.freeze()
        io.log('training complete',freeze_seconds=time.perf_counter()-freeze_started,
               freeze_includes_hashing_and_manifest=True)
        return result
    except BaseException as exc:
        state.update(phase='failed',completion='incomplete_error');write(p/'campaign.json',state)
        write(p/'FAILURE.json',dict(error=str(exc),traceback=traceback.format_exc(),at=time.time()))
        io.log('training failed')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--parent',type=Path,required=True)
    parser.add_argument('--resume-from',type=Path)
    for name,field in Config.__dataclass_fields__.items():parser.add_argument('--'+name.replace('_','-'),type=field.type,default=field.default)
    args=vars(parser.parse_args());out=args.pop('out');parent=args.pop('parent');resume_from=args.pop('resume_from')
    os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
    print(run(out,parent,Config(**args),resume_from=resume_from),flush=True)


if __name__=='__main__':main()
