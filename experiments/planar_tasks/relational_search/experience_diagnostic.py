"""Three-layer witnessed-candidate diagnostic, separate from frozen author runs."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from itertools import islice
from pathlib import Path
import hashlib
import json
import time
from .pipeline import Campaign,read,save,pack,unpack,ROOT
from . import pipeline_execution as X
from .contract import History,canonical,digest,validate_action
from .experience_candidates import ExperienceCandidates
from .live_scorer import FittedScorer
from .live_search import Budget,candidates
from .live_execution import run
from .predictor import NominalPredictor


def write_result(path,result):
 path.mkdir(parents=True,exist_ok=True)
 save(path/'summary.json',result['summary']);pack(path/'public.json.gz',result['public_rows'])
 pack(path/'replay.json.gz',result['replay']);save(path/'events.json',result['events'])
 if 'decisions' in result:pack(path/'decisions.json.gz',result['decisions'])
 if 'commands' in result:save(path/'commands.json',result['commands'])


def retrieval_control(value,db,max_ticks=900,max_decisions=225):
 s,b=X.fixture(value);env=X.RecordedEnv(s,b,overrides=value['overrides']);h=History();h.append(env.public())
 rows=[dict(command_index=0,observation=h.events[-1]['observation'],action=None,history_hash=h.sha256)]
 commands=[];decisions=[];events=[];error=None;start=time.monotonic()
 for i in range(max_decisions):
  if env.termination is not None:break
  retrieved=db.retrieve(h);action=retrieved[0]['action'] if retrieved else dict(target_id=None,a0=44,a1=44,ticks=4)
  if env.total_periods+action['ticks']>max_ticks:break
  try:
   validate_action(action,h.events[-1]['observation']);env.command_index=i;events.append(env.step_command(action))
  except ValueError as exc:error=str(exc);break
  decisions.append(dict(real_tick=h.events[-1]['observation']['tick'],retrieved=retrieved,executed_action=action,real_end_tick=env.total_periods))
  commands.append(action);h.append(env.public(),action);rows.append(dict(command_index=len(commands),action=action,observation=h.events[-1]['observation'],history_hash=h.sha256))
 return dict(summary=dict(status='tool_error' if error else 'success' if env.success else 'task_failure',success=env.success,ticks=env.total_periods,task_cost_seconds=None if error else env.total_periods*.05 if env.success else 180.,error=error,decisions=len(decisions),prediction_ticks=0,prediction_calls=0,wall_seconds=time.monotonic()-start),public_rows=rows,commands=commands,decisions=decisions,events=events,replay=dict(scene=X.scene_metadata(s,b),frames=env.frames))


def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();dest=a.output.resolve();dest.mkdir(parents=True,exist_ok=False)
 old=Campaign(ROOT/'outputs/relational_pipeline/sol_closed_loop_20261113_01');verification=old.verify_frozen()
 source=(old.path/'semantics/s003.py').read_text();fitted=read(old.path/'fits/f003.json');budget=Budget()
 specs=[('seed','r000','a000'),('angle','r002','a001')]
 protocol=dict(source_campaign=str(old.path),source_verification=verification,semantic_id='s003',fit_id='f003',condition='S',cases=specs,
  layers=['exact successful command replay','all witnessed boundaries: coverage/rank/one-step nominal prediction','actual feedback: generated-only search, retrieved-candidate search, nearest-history direct-action control'],
  search=asdict(budget),max_ticks=900,max_decisions=225,episode_wall_seconds=900,reserved_root_calls=2,
  library='Both successful demonstrations; legal observation-only descriptor; last 3 observations + last 2 actual commands; no time/index/true state; 16 nearest records, at most 4 unique legal actions; train-on-replay diagnostic, NOT generalization.',
  scorer='Frozen S; no refit and no new model calls. Direct retrieval control uses no scorer or predictor, separately labeled.',
  counts='Each simulation charged; layer 2 reuses recorded histories, with separate nominal one-step calls. Exact requested command durations retained.',
  created=time.time())
 save(dest/'protocol.json',protocol)
 sources={}
 for p in sorted(set(ROOT.glob('experiments/**/*.py'))|set(ROOT.glob('src/coimagine/**/*.py'))):
  rel=str(p.relative_to(ROOT));data=p.read_bytes();sources[rel]=hashlib.sha256(data).hexdigest();q=dest/'sources'/rel;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
 save(dest/'sources.json',sources);save(dest/'f003.json',fitted);(dest/'s003.py').write_text(source)
 usage=dict(execution_ticks=0,prediction_ticks=0,prediction_calls=0,resets=0,scoring_batches=0)
 def status(stage,**extra):save(dest/'progress.json',dict(stage=stage,usage=usage,updated=time.time(),**extra));print(stage,json.dumps(extra),flush=True)
 cases=[];witnesses=[]
 for name,rid,aid in specs:
  p=old._apath(rid,aid);value=read(old._rpath(rid)/'fixture.json');commands=read(p/'request.json')['commands'];original=unpack(p/'public.json.gz')
  status('layer1_replay',case=name);r=X.execute(value,commands);write_result(dest/'layer1'/name,r)
  usage['execution_ticks']+=r['summary']['ticks'];usage['resets']+=1
  same=[row['history_hash'] for row in r['public_rows']]==[row['history_hash'] for row in original]
  save(dest/'layer1'/name/'comparison.json',dict(exact_public_history=same,original=rid+'/'+aid,success=r['summary']['success']))
  if not same or not r['summary']['success']:raise RuntimeError('witness not reproducible')
  cases.append((name,value,commands,r['public_rows']));witnesses.append((name,r['public_rows'],commands))
 db=ExperienceCandidates.from_trajectories(witnesses);save(dest/'experience_database.json',dict(records=db.records,scale=db.scale.tolist(),max_actions=db.max_actions,neighbors=db.neighbors))
 for name,value,commands,rows in cases:
  h=History();records=[];scorer=FittedScorer(source,fitted,'S')
  for i,row in enumerate(rows[:-1]):
   h.append(row['observation'],row['action']);teacher=commands[i];retrieved=db.retrieve(h)
   generated=list(islice(candidates(h.events[-1]['observation']),72));merged=[];seen=set()
   for action in [x['action'] for x in retrieved]+generated:
    k=canonical(action)
    if k not in seen:merged.append(action);seen.add(k)
   merged=merged[:72]
   union=list({canonical(x):x for x in generated+merged+[teacher]}.values());scores=scorer(h,union);qmap=dict(zip(map(canonical,union),scores));q=qmap[canonical(teacher)]
   def rank(aa):
    above=sum(qmap[canonical(x)]>q+1e-9 for x in aa);ties=sum(abs(qmap[canonical(x)]-q)<=1e-9 for x in aa)
    top=max(aa,key=lambda x:qmap[canonical(x)])
    return dict(present=teacher in aa,rank=above+1,ties=ties,top=top,top_score=qmap[canonical(top)],teacher_score=q,top1_tied=above==0)
   prediction={};model=None
   try:
    model=NominalPredictor(h,ticks=40,calls=1,seconds=15);response=model.predict(model.root,teacher)
    prediction={k:response[k] for k in ('ticks','terminal','success','charged_ticks')};prediction['termination']=model.record()['events'][-1].get('nominal_termination')
   except Exception as exc:prediction['error']=str(exc)
   finally:
    if model is not None:
     record=model.record();used=record['usage']['used'];usage['prediction_ticks']+=used['ticks'];usage['prediction_calls']+=used['calls'];model.close()
   records.append(dict(boundary=i,tick=row['observation']['tick'],history_hash=h.sha256,teacher=teacher,retrieved=retrieved,retrieved_teacher=any(x['action']==teacher for x in retrieved),nearest_is_teacher=bool(retrieved and retrieved[0]['action']==teacher),generated=rank(generated),merged=rank(merged),prediction=prediction))
   usage['scoring_batches']+=1
   if i%20==0:status('layer2_boundary',case=name,boundary=i,total=len(rows)-1)
  save(dest/'layer2'/name/'boundaries.json',records)
  save(dest/'layer2'/name/'summary.json',dict(boundaries=len(records),generated_coverage=sum(r['generated']['present'] for r in records),merged_coverage=sum(r['merged']['present'] for r in records),retrieval_coverage=sum(r['retrieved_teacher'] for r in records),nearest_correct=sum(r['nearest_is_teacher'] for r in records),teacher_top1_tied=sum(r['merged']['top1_tied'] for r in records),teacher_unique_top1=sum(r['merged']['top1_tied'] and r['merged']['ties']==1 for r in records),nominal_failures=sum(bool(r['prediction'].get('error')) or (r['prediction'].get('terminal',False) and not r['prediction'].get('success',False)) for r in records)))
 # Same budgets on the two search arms; retrieved protected calls are INCLUDED in 12 calls/600 ticks.
 for name,value,commands,rows in cases:
  for arm in ('generated_only','retrieved_search','nearest_history_control'):
   status('layer3_start',case=name,arm=arm)
   if arm=='nearest_history_control':result=retrieval_control(value,db)
   else:
    def progress(n,tick,d):status('layer3_boundary',case=name,arm=arm,decisions=n,real_tick=tick)
    result=run(value,source,fitted,condition='S',max_ticks=900,max_decisions=225,budget=budget,wall_seconds=900,on_progress=progress,experience=db if arm=='retrieved_search' else None,reserved_root_calls=2 if arm=='retrieved_search' else 0)
   write_result(dest/'layer3'/name/arm,result)
   usage['execution_ticks']+=result['summary']['ticks'];usage['resets']+=1
   usage['prediction_ticks']+=result['summary']['prediction_ticks'];usage['prediction_calls']+=result['summary']['prediction_calls']
   status('layer3_finished',case=name,arm=arm,summary=result['summary'])
 summary=dict(protocol=protocol,usage=usage,layer1={n:read(dest/'layer1'/n/'comparison.json') for n,*_ in cases},layer2={n:read(dest/'layer2'/n/'summary.json') for n,*_ in cases},layer3={n:{arm:read(dest/'layer3'/n/arm/'summary.json') for arm in ('generated_only','retrieved_search','nearest_history_control')} for n,*_ in cases},old_campaign_verification=old.verify_frozen())
 save(dest/'summary.json',summary);status('complete');print(json.dumps(summary,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
