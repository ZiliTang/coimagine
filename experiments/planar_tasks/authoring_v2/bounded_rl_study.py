"""Explicit bounded two-arm random-initial-state pilot; no provider calls."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
os.environ['COIMAGINE_LIDAR_KERNEL']='native'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import argparse,shutil,time,traceback
import torch
from .random_curriculum import build
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save,pack,unpack
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.policy import Policy
from experiments.planar_tasks.semantic_rl.bounded_policy import BoundedPolicy,compatible,action_parameters,bounded_ppo_update
from experiments.planar_tasks.semantic_rl.parameter_policy import state_digest
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.schema import Schema
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl.memory_flow import compact_step
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory
from experiments.planar_tasks.semantic_rl.save_timing import SaveTiming

BASE=Path('outputs/semantic_rl');HOST=Path('outputs/semantic_rl_authors/random_parameter_20261116_01')
PREVIOUS=BASE/'local_cap_integration_20261116_01';OLD=BASE/'duration_stream_continuation_parent_20261115_01'
PARENT=BASE/'random_parameter_parent_20261116_01';EVAL=BASE/'random_parameter_evaluation_20261116_01';VALIDATION=BASE/'random_parameter_validation_20261116_01'
ARMS={'speed':0,'direction':1};FEATURES=None

def features(source,model):return Features(source,schema=Schema(tuple(model.schema['slots'])),fast_embedding=True,action_protocol=model.action_protocol,execution_backend='trusted_inprocess',trusted_source_sha256=digest(source))

def make(path,n,profile):
 c=Campaign.create(path,limits=dict(attempts=n,resets=n,execution_ticks=n*1800,rounds=32,fits=0,author_turns=0),model=dict(provider='none',name=profile))
 with c.locked():c.state.update(profile=profile,runs=[],phase='ready');c._save()
 return c

def prepare():
 torch.set_num_threads(1);seal=Campaign(PREVIOUS).verify_frozen();state=read(PREVIOUS/'campaign.json')
 roots=read(OLD/'panels/roots.json')['cases'];train=read(OLD/'panels/train.json')['cases'];oldtest=read(OLD/'panels/test.json')['cases']
 anchors=[dict(q,split='anchor') for q in roots+train if q['id'].removeprefix('root_') in state['tasks'] or q['id'] in state['tasks']]
 assert len(anchors)==13
 panels=build(anchors,exclude=[q['fixture_sha256'] for q in roots+train+oldtest])
 p=make(PARENT,0,'bounded-random-initial-state-protocol-v1');(PARENT/'inputs').mkdir()
 shutil.copyfile(PREVIOUS/'semantics/s000.py',PARENT/'inputs/source.py');shutil.copyfile(PREVIOUS/'semantic_fits/f000/policy.pt',PARENT/'inputs/supervisor.pt')
 for split,v in panels.items():save(PARENT/'panels'/f'{split}.json',v)
 save(PARENT/'panels/anchors.json',dict(cases=anchors))
 protocol=dict(parent_seal=seal,arms=ARMS,workers=16,rounds=32,episodes_per_arm=512,sampling='per16 episodes:4anchor,8small,4medium; fixed seeds; geometry-only rejection; identical draws for both arms',parameters=dict(speed='all public speed levels within same family',direction='MOVE/PUSH0 or +/-45degrees; ROTATE sign fixed',ticks='supervisor +/-4 bounded1..40',frozen='target_id and each robot family'),seeds=dict(train=2026111604,validation=2026111605,test=2026111606,ppo=2026111607),validation=24,test=48,anchors=13,max_ticks=1800,max_decisions=300,episode_seconds=180,arm_wall_seconds=2400,epochs=4,minibatch_size=64,lr=.0003,temperature=2.,residual_coef=.05,stop='anchor successes below11/13 at two consecutive8-round diagnostics; errors stop; validation logged, never selects best',selection='fixed last complete or registered stop checkpoint; no test access until both arms frozen')
 save(PARENT/'protocol.json',protocol)
 source=(PARENT/'inputs/source.py').read_text();sup,_=Policy.load(PARENT/'inputs/supervisor.pt',digest(source));f=features(source,sup);actors=[BoundedPolicy.from_supervisor(sup,direction_radius=x) for x in (0,1)];checked=0
 try:
  for r in state['runs']:
   if not r.get('imported'):continue
   data=unpack(PREVIOUS/'runs'/r['id']/'result.json.gz');h=StreamHistory()
   for index,row in enumerate(data['public_rows'][:-1]):
    h.append(row['observation'],row['action'])
    with torch.no_grad():expected=sup.decide(h,f,deterministic=True)['action']
    assert expected==data['public_rows'][index+1]['action']
    for actor in actors:assert actor.decide(h,f,deterministic=True)['action']==expected
    checked+=1
 finally:f.close()
 save(PARENT/'acceptance.json',dict(initial_greedy_boundaries_per_arm=checked,exact_supervisor_match=True))
 with p.locked():p.state.update(phase='finished',completion='complete',protocol=protocol);p._save()
 save(HOST/'PREPARED.json',dict(seal=p.freeze(),protocol=protocol));print('PREPARED',checked,flush=True)

def worker(job):
 global FEATURES
 torch.set_num_threads(1);source=Path(job['source']).read_text();loader=Policy if job['kind']=='supervised' else BoundedPolicy
 model,_=loader.load(job['checkpoint'],digest(source))
 if FEATURES is None:FEATURES=features(source,model)
 audits=[]
 if job['kind']!='supervised':
  original=model.decide
  def audited(*a,**kw):
   tr=original(*a,**kw);audits.append((tr['action'],tr['supervisor_action']));return tr
  model.decide=audited
 r,steps=rollout(job['fixture'],source,model=model,features=FEATURES,config=EventConfig(mode='duration'),max_ticks=1800,max_decisions=300,wall_seconds=180,seed=job['seed'],deterministic=not job['training'],temperature=2. if job['training'] else 1.,fast_feedback=True,training_fastpath=True,record_details=job['details'])
 if audits:
  violations=sum(a['target_id']!=q['target_id'] or abs(a['ticks']-q['ticks'])>4 or any(not compatible(a[k],q[k],model.direction_radius) for k in ('a0','a1')) for a,q in audits)
  if violations:raise ValueError('bounded parameter violation')
  r['summary']['parameter_audit']=dict(decisions=len(audits),violations=violations,duration_changed=sum(a['ticks']!=q['ticks'] for a,q in audits),speed_changed=sum(any(action_parameters(a[k])[2]!=action_parameters(q[k])[2] for k in ('a0','a1')) for a,q in audits),direction_changed=sum(any(action_parameters(a[k])[1]!=action_parameters(q[k])[1] for k in ('a0','a1')) for a,q in audits))
  if job['details']:r['parameter_choices']=[dict(action=a,supervisor=q) for a,q in audits]
 if not job['training']:steps=[]
 elif job['details']:steps=[compact_step(t) for t in steps]
 return r,steps

def collect(c,cases,checkpoint,kind,pool,io,*,training=False,round_number=0,all_details=False):
 jobs=[];offset=len(read(c.path/'campaign.json')['runs'])
 with c.locked():
  for i,case in enumerate(cases):
   n=offset+i;rid=f'a{n:04d}';p=c.path/'runs'/rid;p.mkdir(parents=True)
   details=all_details or training and (n<3 or n%128==0)
   req=dict(id=rid,case_id=case['id'],family=case['family'],split=case['split'],level=case['level'],condition=kind,round=round_number,record_details=details,status='running',fixture_sha256=case['fixture_sha256'])
   job=dict(id=rid,source=str(PARENT/'inputs/source.py'),checkpoint=str(checkpoint),kind=kind,fixture=case['fixture'],seed=2026111607+(round_number*16+i if training else i),training=training,details=details)
   c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(req);io.call(save,p/'request.json',req);jobs.append((job,req))
  c._save()
 pending={pool.submit(worker,j):(j,r) for j,r in jobs};rows={};episodes={}
 for future in as_completed(pending):
  job,req=pending[future];r,steps=future.result();rid=job['id'];row=dict(r['summary'],**{k:v for k,v in req.items() if k!='status'})
  if job['details']:io.call(pack,c.path/'runs'/rid/'result.json.gz',r,compresslevel=1)
  io.call(save,c.path/'runs'/rid/'summary.json',row)
  with c.locked():
   c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][rid];c.state['runs'][int(rid[1:])]=row;c._save()
  if row['error'] or row['stop']=='wall_budget':raise RuntimeError('execution error/truncation: '+rid)
  rows[rid]=row;episodes[rid]=dict(steps=steps,bootstrap=r['summary']['bootstrap'])
 ids=sorted(rows);return [rows[k] for k in ids],[episodes[k] for k in ids]

def summary(rows):
 return dict(episodes=len(rows),success=sum(r['success'] for r in rows),mean_cost=sum(r['task_cost_seconds'] if r['success'] else 180 for r in rows)/len(rows))

def finish(c,completion):
 with c.locked():c.state.update(phase='finished',completion=completion);c._save()
 return c.freeze()

def run():
 torch.set_num_threads(1);parent_seal=Campaign(PARENT).verify_frozen();protocol=read(PARENT/'protocol.json');source=(PARENT/'inputs/source.py').read_text()
 sup,_=Policy.load(PARENT/'inputs/supervisor.pt',digest(source));train=read(PARENT/'panels/train.json')['cases'];val=read(PARENT/'panels/validation.json')['cases'];anchors=read(PARENT/'panels/anchors.json')['cases'];io=SaveTiming()
 d=make(VALIDATION,24+2*4*37,'bounded-parameter-validation-v1');finished={}
 with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('spawn')) as pool:
  rows,_=collect(d,val,PARENT/'inputs/supervisor.pt','supervised',pool,io);save(VALIDATION/'baseline.json',summary(rows));print('BASELINE VALIDATION',summary(rows),flush=True)
  for name,direction_radius in ARMS.items():
   path=BASE/f'random_parameter_{name}_20261116_01';c=make(path,512,'semantic-ppo-parallel-bounded-random-v1');(path/'inputs').mkdir();(path/'models').mkdir()
   shutil.copyfile(PARENT/'inputs/source.py',path/'inputs/source.py');save(path/'inputs/provenance.json',dict(parent_seal=parent_seal,schema=sup.schema));save(path/'protocol.json',protocol)
   model=BoundedPolicy.from_supervisor(sup,direction_radius=direction_radius);checkpoint=path/'models/initial.pt';model.save(checkpoint,dict(arm=name));optimizer=torch.optim.Adam(model.parameters(),lr=.0003)
   metrics=dict(rounds=[],updates=[],diagnostics=[]);save(path/'metrics.json',metrics)
   with c.locked():c.state.update(config=dict(protocol,direction_radius=direction_radius,feature_runtime='typed_stream',execution_mode='duration'),completed_rounds=0,training_schedule=[x['id'] for x in train]);c._save()
   start=time.monotonic();completion='complete';previous_bad=False
   try:
    for r in range(32):
     if time.monotonic()-start>2100:completion='incomplete_wall_admission';break
     with c.locked():c.state.update(phase='collecting',active_round=r,policy_checkpoint=str(checkpoint.relative_to(path)));c._save()
     t=time.monotonic();rows,episodes=collect(c,train[r*16:(r+1)*16],checkpoint,name,pool,io,training=True,round_number=r);collection=time.monotonic()-t
     t=time.monotonic();update=bounded_ppo_update(model,optimizer,episodes,epochs=4,minibatch_size=64,seed=2026111607+r);update_seconds=time.monotonic()-t
     for u in update['curve']:metrics['updates'].append(dict(u,round=r,optimizer_step=len(metrics['updates'])+1,grad_norm=u['actor_grad_norm']))
     checkpoint=path/'models'/f'round_{r:04d}.pt';io.call(model.save,checkpoint,dict(arm=name,round=r));io.call(torch.save,dict(optimizer=optimizer.state_dict()),path/'models'/f'optimizer_{r:04d}.pt')
     ticks=sum(x['ticks'] for x in rows);decisions=sum(x['decisions'] for x in rows)
     point=dict(summary(rows),round=r,success_rate=sum(x['success'] for x in rows)/16,ticks=ticks,decisions=decisions,collection_seconds=collection,update_seconds=update_seconds,ticks_per_second=ticks/collection,decisions_per_second=decisions/collection,updates=update['updates'],total_optimizer_steps=len(metrics['updates']),logp_recompute_error=update['logp_recompute_error'],early_stop_kl=update['early_stop_kl'],wall_truncations=0,tool_errors=0,elapsed_seconds=time.monotonic()-start,checkpoint=str(checkpoint.relative_to(path)),parameter_changes={k:sum(x['parameter_audit'][k] for x in rows) for k in ['duration_changed','speed_changed','direction_changed','violations']})
     metrics['rounds'].append(point)
     with c.locked():c.state['completed_rounds']=r+1;c.state['used']['rounds']=r+1;c._save()
     io.call(save,path/'metrics.json',metrics);io.log(name+' round'+str(r+1),details=sum(x['record_details'] for x in rows));print('ROUND',name,r+1,point['success'],'/16',flush=True)
     if (r+1)%8==0:
      rr,_=collect(d,anchors+val,checkpoint,name,pool,io,round_number=r+1);a=summary(rr[:13]);v=summary(rr[13:]);point=dict(round=r+1,anchors=a,validation=v)
      metrics['diagnostics'].append(point);save(path/'metrics.json',metrics);print('DIAGNOSTIC',name,point,flush=True)
      bad=a['success']<11
      if bad and previous_bad:completion='stopped_anchor_regression';break
      previous_bad=bad
    shutil.copyfile(checkpoint,path/'models/final.pt');finished[name]=dict(path=str(path),completion=completion,seal=finish(c,completion),rounds=c.state['completed_rounds'])
   except BaseException:
    with c.locked():c.state.update(phase='failed',completion='incomplete_error');c._save()
    raise
  validation_seal=finish(d,'complete');save(HOST/'TRAIN_FINISHED.json',dict(arms=finished,validation_seal=validation_seal))
  # Test fixtures are loaded only after both final training artifacts are frozen.
  test=read(PARENT/'panels/test.json')['cases'];e=make(EVAL,3*61,'bounded-parameter-fresh-test-v1');allrows=[]
  models={'supervised':PARENT/'inputs/supervisor.pt',**{n:Path(v['path'])/'models/final.pt' for n,v in finished.items()}}
  for name,checkpoint in models.items():
   rows,_=collect(e,anchors+test,checkpoint,name,pool,io,all_details=True);allrows+=rows;print('FINAL',name,'anchors',summary(rows[:13]),'test',summary(rows[13:]),flush=True)
  save(EVAL/'comparison.json',allrows);save(EVAL/'selection.json',dict(arms=finished,parent_seal=parent_seal,test='fresh independent48initials; not used by optimizer or checkpoint selection'))
  save(HOST/'FINISHED.json',dict(arms=finished,validation_seal=validation_seal,evaluation_seal=finish(e,'complete')));io.log('finished')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);args=parser.parse_args();HOST.mkdir(parents=True,exist_ok=True)
 try:prepare() if args.command=='prepare' else run()
 except BaseException as exc:save(HOST/(args.command+'_FAILURE.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
