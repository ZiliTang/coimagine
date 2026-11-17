"""Bounded real-prefix handoff acceptance and teacher compatibility mapping."""
from . import bounded_rl_study as S
from pathlib import Path
import argparse,time,shutil,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
import torch
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.scale import protocol

BASE=S.BASE;OLD=BASE/'local_cap_loop_20261116_01';OUT=BASE/'handoff_entry_20261116_02'
HOST=Path('outputs/semantic_rl_authors/handoff_entry_20261116_02')
TASKS=['train_open_transport_03','train_clear_transport_01','train_clear_transport_02']
CUTS=[8,16,24];TEACHERS=['s000','s002']


def serialized_equal(a,b):
 return S.digest(a)==S.digest(b)


def prepare():
 seal=S.Campaign(OLD).verify_frozen();old=S.read(OLD/'campaign.json')
 c=S.Campaign.create(OUT,limits=dict(attempts=36,resets=36,execution_ticks=64800,rounds=0,fits=0,author_turns=0),model=dict(provider='none',name='frozen CAP handoff acceptance'))
 jobs=[]
 for task in TASKS:
  fixture=S.read(OLD/'tasks'/task/'fixture.json');S.save(OUT/'tasks'/task/'fixture.json',fixture)
  for donor in TEACHERS:
   r=next(r for r in old['runs'] if r.get('task')==task and r.get('sid')==donor)
   data=S.unpack(OLD/'runs'/r['id']/'result.json.gz');commands=[x['action'] for x in data['public_rows'][1:]]
   for cut in CUTS:
    assert len(commands)>cut
    for receiver in TEACHERS:
     jobs.append(dict(task=task,donor=donor,receiver=receiver,cut=cut,prefix=commands[:cut],reference_id=r['id'],fixture=fixture,split='development' if task!=TASKS[-1] else 'engineering_holdout_known_initial'))
 for sid in TEACHERS:
  p=OUT/'inputs'/f'{sid}.py';p.parent.mkdir(exist_ok=True);shutil.copyfile(OLD/'semantics'/f'{sid}.py',p)
 S.save(OUT/'jobs.json',jobs)
 config=dict(parent_seal=seal,max_episodes=36,max_ticks=1800,max_decisions=300,episode_wall_seconds=180,workers=4,cuts=CUTS,scope='engineering compatibility on known historical training initial states; not fresh combination generalization',split_rule='open03+clear01 development; clear02 known-initial engineering holdout; no fitting or repair in this stage',selection='all donor/receiver pairs and predeclared cuts; no success filtering',prefix='actual replay with receiver memory updated from actual history; total horizon unchanged; all prefix ticks charged')
 S.save(OUT/'protocol.json',config)
 with c.locked():c.state.update(profile='real-prefix-handoff-v1',runs=[],phase='ready',protocol=config);c._save()
 S.save(HOST/'PREPARED.json',config);print('PREPARED36',flush=True)


def worker(job):
 torch.set_num_threads(1);source=(OUT/'inputs'/f"{job['receiver']}.py").read_text()
 f=Features(source,fast_embedding=True,action_protocol=protocol(S.EventConfig(mode='duration')),execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source))
 try:r,_=S.rollout(job['fixture'],source,features=f,config=S.EventConfig(mode='duration'),prefix=job['prefix'],max_ticks=1800,max_decisions=300,wall_seconds=180,fast_feedback=True,record_details=True)
 finally:f.close()
 ref=S.unpack(OLD/'runs'/job['reference_id']/'result.json.gz');cut=job['cut']
 def visible(rows):return [dict(action=x['action'],observation=x['observation']) for x in rows]
 prefix_equal=visible(r['public_rows'][:cut+1])==visible(ref['public_rows'][:cut+1])
 frames=[x for x in ref['replay']['frames'] if x['tick']<=r['prefix_replay']['ticks']]
 actual=[x for x in r['replay']['frames'] if x['tick']<=r['prefix_replay']['ticks']]
 audit=dict(prefix_public_equal=prefix_equal,prefix_physics_equal=serialized_equal(actual,frames),same_teacher=job['donor']==job['receiver'])
 if audit['same_teacher']:
  audit.update(full_public_equal=visible(r['public_rows'])==visible(ref['public_rows']),full_physics_equal=serialized_equal(r['replay']['frames'],ref['replay']['frames']))
 r['handoff']=dict({k:v for k,v in job.items() if k not in ['fixture','prefix']},audit=audit)
 return r


def run():
 c=S.Campaign(OUT);c.verify_sources();jobs=S.read(OUT/'jobs.json');started=time.monotonic();io=S.SaveTiming()
 with c.locked():
  c.state['phase']='running'
  for i,j in enumerate(jobs):
   rid=f'a{i:04d}';c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);req=dict(id=rid,case_id=j['task'],split=j['split'],donor=j['donor'],receiver=j['receiver'],cut=j['cut'],status='running',record_details=True)
   c.state['runs'].append(req);S.save(OUT/'runs'/rid/'request.json',dict(req,prefix=j['prefix'],reference_id=j['reference_id']))
  c._save()
 try:
  failures=[]
  with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn')) as pool:
   pending={pool.submit(worker,j):i for i,j in enumerate(jobs)}
   for future in as_completed(pending):
    i=pending[future];rid=f'a{i:04d}';r=future.result();a=r['handoff']['audit'];req=S.read(OUT/'runs'/rid/'request.json');summary=dict(r['summary'],**{k:v for k,v in req.items() if k not in ['status','prefix','reference_id']},handoff_audit=a,prefix_replay=r['prefix_replay']);io.call(S.pack,OUT/'runs'/rid/'result.json.gz',r);io.call(S.save,OUT/'runs'/rid/'summary.json',summary)
    with c.locked():c.state['runs'][i]=summary;c.state['used']['execution_ticks']-=1800-summary['ticks'];del c.state['reservations'][rid];c._save()
    print(rid,summary['case_id'],summary['donor'],summary['receiver'],summary['cut'],summary['success'],a,flush=True)
    if not all(v for k,v in a.items() if k!='same_teacher') or r['summary']['error'] or r['summary']['stop']=='wall_budget':failures.append(rid)
  if failures:raise RuntimeError('handoff acceptance failed: '+','.join(failures))
  rows=S.read(OUT/'campaign.json')['runs'];report=dict(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows),elapsed_seconds=time.monotonic()-started,scope='known initial-state handoff acceptance; no new CAP, no fit, no PPO')
  S.save(OUT/'result.json',report);S.save(HOST/'FINISHED.json',dict(seal=S.finish(c,'complete'),episodes=len(rows)));io.log('handoff-finished')
 except BaseException:
  with c.locked():c.state.update(phase='failed',completion='incomplete_error');c._save()
  raise

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);a=p.parse_args();HOST.mkdir(parents=True,exist_ok=True)
 try:prepare() if a.command=='prepare' else run()
 except BaseException as exc:S.save(HOST/(a.command+'_FAILURE.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
