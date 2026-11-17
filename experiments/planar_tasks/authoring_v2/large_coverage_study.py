"""Outcome-independent 216+216 coverage experiment, frozen teachers and semantics."""
from . import coverage_distillation_study as T
from . import random_curriculum as RC
from experiments.planar_tasks.semantic_rl.coverage_large_fit import compact,streaming_policy
from experiments.planar_tasks.semantic_rl.coverage_fit import fit_updates
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from collections import Counter
import argparse,hashlib,math,multiprocessing as mp,random,shutil,time,traceback
import numpy as np
S=T.S;B=S.BASE;HOST=Path('outputs/semantic_rl_authors/large_coverage_20261116_01')
PARENT=B/'large_coverage_parent_20261116_01';COLLECT=B/'large_coverage_collection_20261116_01';FIT=B/'large_coverage_fit_20261116_01';EVAL=B/'large_coverage_evaluation_20261116_01'
PREVIOUS=B/'coverage_distillation_parent_20261116_01';PREVFIT=B/'coverage_distillation_fit_20261116_01';LONG=T.LONG;OLD=T.OLD;WORKERS=16;CACHE={};RUNNER=None
BOUNDS={**RC.BOUNDS,'wide':{k:2*v for k,v in RC.BOUNDS['medium'].items()}};RC.BOUNDS=BOUNDS

def prepare():
 seals={str(p):S.Campaign(p).verify_frozen() for p in [PREVIOUS,PREVFIT,LONG,OLD]}
 registry={r['id']:r for r in S.read('docs/research/Policy_Registry.json')['entries']}
 anchors=S.read(LONG/'panels/anchors.json')['cases'];seen={c['fixture_sha256'] for c in anchors}
 for path in [LONG,PREVIOUS,B/'random_parameter_parent_20261116_01',B/'duration_stream_continuation_parent_20261115_01']:
  for panel in (path/'panels').glob('*.json'):
   for case in S.read(panel).get('cases',[]):seen.add(case['fixture_sha256'])
 excluded=sorted(seen);groups={f:[c for c in anchors if c['family']==f] for f in sorted({c['family'] for c in anchors})};panels={}
 for split,seed in [('train',2026111631),('test',2026111632)]:
  rng=random.Random(seed);cases=[]
  for family in groups:
   for level in ('small','medium','wide'):
    for _ in range(24):cases.append(RC.draw(rng.choice(groups[family]),level,rng,identifier=f'{split}_{len(cases):04d}',split=split,seen=seen))
  panels[split]=dict(seed=seed,cases=cases,bounds=BOUNDS)
 assert len({c['fixture_sha256'] for p in panels.values() for c in p['cases']})==432
 c=T.make(PARENT,0);(PARENT/'inputs').mkdir()
 for name,rid in [('base','CAP-001.v4'),('repair','CAP-002.v2'),('semantics','SEM-001.v1')]:
  r=registry[rid];assert T.C.sha(r['artifact'])==r['file_sha256'];shutil.copyfile(r['artifact'],PARENT/'inputs'/f'{name}.py')
 for name,rid in [('original','SUP-002.v1'),('smallfit','SUP-002.v3')]:
  r=registry[rid];assert T.C.sha(r['artifact'])==r['file_sha256'];shutil.copyfile(r['artifact'],PARENT/'inputs'/f'{name}.pt')
 for name in ['frozen_runner.py','original_dataset.json.gz']:shutil.copyfile(PREVIOUS/'inputs'/name,PARENT/'inputs'/name)
 for name,p in panels.items():S.save(PARENT/'panels'/f'{name}.json',p)
 S.save(PARENT/'panels/anchors.json',dict(cases=anchors))
 p=dict(input_seals=seals,workers=WORKERS,train_cases=216,test_cases=216,excluded_fixture_hashes=excluded,bounds=BOUNDS,max_episodes=2402,max_ticks=4323600,
  episode_ticks=1800,episode_decisions=300,episode_wall_seconds=180,collection_admission_seconds=3600,
  data='original13 plus one lowest-cost successful CAP per random train case, base wins ties; no selection on learner failure; teacher failures remain in all evaluation denominators',
  fits=dict(seed=17,updates='min(20000,max(6000,100*ceil(expanded_boundaries/32)))',arms=['control','expanded'],optimizer='Adam .003; same loss/residual .05; from scratch',wall_seconds_per_arm=1800),
  selection='fixed final, no test checkpoint selection',scope='single optimizer seed; larger random population reduces sampling noise, not all statistical or optimization bias')
 S.save(PARENT/'protocol.json',p);S.save(HOST/'PREPARED.json',dict(seal=S.finish(c,'complete'),protocol=p))
 print('PREPARED432 distinct new initial states;16workers;max2402executions',flush=True)

def worker(job):
 global RUNNER
 S.torch.set_num_threads(1)
 if RUNNER is None:RUNNER=T.load_frozen_runner(PARENT/'inputs/frozen_runner.py')
 name=job['condition'];cap=name in ('base','repair')
 if name not in CACHE:
  source=(PARENT/'inputs'/f'{name if cap else "semantics"}.py').read_text()
  model=None if cap else S.Policy.load(PARENT/'inputs'/f'{name}.pt' if name in ('original','smallfit') else FIT/'models'/f'{name}.pt',S.digest(source))[0]
  features=S.Features(source,schema=S.Schema(tuple(model.schema['slots'])) if model else None,fast_embedding=True,action_protocol=T.protocol(S.EventConfig(mode='duration')),execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source))
  CACHE[name]=(source,model,features)
 source,model,f=CACHE[name]
 r,_=RUNNER(job['fixture'],source,model=model,features=f,config=S.EventConfig(mode='duration'),max_ticks=1800,max_decisions=300,wall_seconds=180,deterministic=True,seed=job['seed'],fast_feedback=True,training_fastpath=False,record_details=True)
 return r

def collect(c,cases,name,pool,io):
 offset=len(S.read(c.path/'campaign.json')['runs']);start=time.monotonic();pending={};cursor=0;rows=[]
 while pending or cursor<len(cases):
  while len(pending)<WORKERS and cursor<len(cases) and time.monotonic()-start<3600:
   case=cases[cursor];i=offset+cursor;rid=f'a{i:04d}';req=dict(id=rid,case_id=case['id'],family=case['family'],level=case['level'],split=case['split'],fixture_sha256=case['fixture_sha256'],condition=name,record_details=True,status='running')
   with c.locked():c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(req);c.state.update(phase='collecting',active_condition=name);c._save()
   io.call(S.save,c.path/'runs'/rid/'request.json',req);pending[pool.submit(worker,dict(case,condition=name,seed=2026111633+cursor))]=(req,i);cursor+=1
  if not pending:break
  done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
  for future in done:
   req,i=pending.pop(future);r=future.result();assert r['fixture_sha256']==req['fixture_sha256']
   row=dict(r['summary'],**{k:v for k,v in req.items() if k!='status'})
   io.call(S.pack,c.path/'runs'/req['id']/'result.json.gz',r,compresslevel=1);io.call(S.save,c.path/'runs'/req['id']/'summary.json',row)
   with c.locked():c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][req['id']];c.state['runs'][i]=row;c._save()
   rows.append(row)
   if row['error'] or row['stop']=='wall_budget':raise RuntimeError('execution error '+req['id'])
   if len(rows)%48==0:io.log(name,completed=len(rows),success=sum(x['success'] for x in rows),elapsed_seconds=round(time.monotonic()-start,2))
 if len(rows)!=len(cases):raise TimeoutError('collection admission')
 io.log(name+' complete',episodes=len(rows),success=sum(x['success'] for x in rows));return sorted(rows,key=lambda x:x['id'])

def fit(selected):
 c=T.make(FIT,0,fits=2);(FIT/'models').mkdir();source=(PARENT/'inputs/semantics.py').read_text();original_raw=S.unpack(PARENT/'inputs/original_dataset.json.gz');sup,_=S.Policy.load(PARENT/'inputs/original.pt',S.digest(source));features=S.features(source,sup)
 # Check compact/streaming preprocessing against unchanged legacy math before large fits.
 a,_=fit_updates(original_raw[:4],features,updates=2);b,_=fit_updates(compact(original_raw[:4]),features,updates=2,policy_factory=streaming_policy)
 delta=max(float((v-b.state_dict()[k]).abs().max()) for k,v in a.state_dict().items());assert delta<2e-6,delta
 S.save(FIT/'preprocessing_check.json',dict(updates=2,max_tensor_error=delta,tolerance=2e-6));del a,b
 original=compact(original_raw);del original_raw;data=list(original);metadata=[];started=time.monotonic()
 try:
  for i,r in enumerate(selected):
   result=S.unpack(COLLECT/'runs'/r['id']/'result.json.gz');result['split']='train';ts=T.demonstration_traces(result,features,split='train',mode='duration');small=compact(ts);data.extend(small)
   hh=hashlib.sha256()
   for tr in small:
    for p in tr['panels']:
     hh.update(p['semantic'].tobytes());hh.update(p['raw'].tobytes())
   metadata.append(dict(case_id=r['case_id'],teacher=r['condition'],run=r['id'],boundaries=len(small),feature_sha256=hh.hexdigest()))
   del result,ts,small
   if (i+1)%16==0:print('FEATURES',i+1,'/',len(selected),'boundaries',len(data),flush=True)
  updates=min(20000,max(6000,100*math.ceil(len(data)/32)))
  S.save(FIT/'data.json',dict(original_trajectories=13,original_boundaries=407,new=metadata,total_boundaries=len(data),updates_per_arm=updates,feature_batches=features.calls,feature_seconds=time.monotonic()-started,semantic_sha256=S.digest(source)))
  print('DATA',len(data),'boundaries;updates_per_arm',updates,flush=True)
  for name,traces in [('control',original),('expanded',data)]:
   with c.locked():c._charge(fits=1);c.state.update(phase='fitting',active_fit=name);c._save()
   def progress(row):
    S.save(FIT/f'{name}_progress.json',row)
    if row['update']%1000==0:print('FIT',name,row,flush=True)
   model,curve=fit_updates(traces,features,updates=updates,wall_seconds=1800,on_progress=progress,policy_factory=streaming_policy)
   correct=0;oldcorrect=0
   with S.torch.no_grad():
    for i,tr in enumerate(traces):
     match=all(int(model.distribution(p)[0].logits.argmax())==j for p,j in zip(tr['panels'],tr['indices']));correct+=match;oldcorrect+=match and i<407
   model.save(FIT/'models'/f'{name}.pt',dict(seed=17,updates=updates,boundaries=len(traces),source='large random coverage, fixed semantics'))
   S.save(FIT/f'{name}_curve.json',curve);S.save(FIT/f'{name}_fit.json',dict(updates=updates,boundaries=len(traces),matches=correct,old_matches=oldcorrect,last=curve[-1]));print('FIT_DONE',name,correct,'/',len(traces),flush=True)
 finally:features.close()
 S.save(FIT/'timing.json',dict(total_seconds=time.monotonic()-started));S.save(HOST/'FIT_FINISHED.json',dict(seal=S.finish(c,'complete')))

def stats(rows):
 groups={};names=sorted({r['condition'] for r in rows})
 for group in ('anchor','train','test','small','medium','wide','open_transport','wall_release','clear_transport','prior_range'):
  sub=[r for r in rows if r['split']==group] if group in ('anchor','train','test') else [r for r in rows if r['split']=='test' and (r['level'] in ('small','medium') if group=='prior_range' else group in (r['level'],r['family']))]
  ids={r['case_id'] for r in sub};sets={n:{r['case_id'] for r in sub if r['condition']==n and r['success']} for n in names};union=sets['base']|sets['repair'];g=dict(n=len(ids),teacher_union=len(union),conditions={},paired={})
  for n in names:
   rr=[r for r in sub if r['condition']==n]
   if rr:g['conditions'][n]=dict(n=len(rr),success=sum(r['success'] for r in rr),mean_cost=sum(r['task_cost_seconds'] if r['success'] else 180 for r in rr)/len(rr),retained_union=len(sets[n]&union),lost_union=sorted(union-sets[n]),outside_union=sorted(sets[n]-union))
  for n in ['smallfit','control','expanded']:
   if n not in g['conditions']:continue
   for ref in ['original','repair','control']:
    if ref==n or ref not in g['conditions']:continue
    g['paired'][n+'_vs_'+ref]=dict(gained=sorted(sets[n]-sets[ref]),lost=sorted(sets[ref]-sets[n]))
  groups[group]=g
 return groups

def uncertainty(rows):
 test=[r for r in rows if r['split']=='test'];cases=sorted({r['case_id'] for r in test});idx={(r['case_id'],r['condition']):r for r in test};strata={}
 for cid in cases:
  r=idx[cid,'original'];strata.setdefault((r['family'],r['level']),[]).append(cid)
 result={}
 for name,ref in [('smallfit','original'),('expanded','original'),('expanded','control'),('expanded','repair')]:
  rng=np.random.default_rng(2026111634);samples=np.zeros(10000);observed=0
  for ids in strata.values():
   delta=np.array([int(idx[c,name]['success'])-int(idx[c,ref]['success']) for c in ids]);observed+=delta.sum();samples+=delta[rng.integers(0,len(ids),(10000,len(ids)))].sum(axis=1)
  result[name+'_vs_'+ref]=dict(difference=observed/len(cases),paired_stratified_bootstrap_95=list(np.quantile(samples/len(cases),[.025,.975])),replicates=10000,scope='fixed models; sampled initial distribution; not optimizer seed uncertainty')
 return result

def run():
 started=time.monotonic();S.torch.set_num_threads(1);S.Campaign(PARENT).verify_frozen();train=S.read(PARENT/'panels/train.json')['cases'];anchors=S.read(PARENT/'panels/anchors.json')['cases'];io=S.SaveTiming();c=T.make(COLLECT,648);collected=[]
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
  for name in ('base','repair','original'):collected+=collect(c,train,name,pool,io)
 selected=[]
 for case in train:
  choices=[r for r in collected if r['case_id']==case['id'] and r['condition'] in ('base','repair') and r['success']]
  if choices:selected.append(min(choices,key=lambda r:(r['ticks'],r['condition']!='base')))
 S.save(COLLECT/'selection.json',dict(selected=selected,uncovered=[case['id'] for case in train if case['id'] not in {r['case_id'] for r in selected}],rule='all random train cases; shortest successful CAP, base tie'))
 S.save(HOST/'COLLECTION_FINISHED.json',dict(seal=S.finish(c,'complete'),selected=len(selected)));print('SELECTED',len(selected),'random cases',flush=True)
 fit(selected);S.Campaign(FIT).verify_frozen()
 test=S.read(PARENT/'panels/test.json')['cases'];e=T.make(EVAL,1754);evaluation=[]
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
  for name in ('control','expanded'):evaluation+=collect(e,anchors+train+test,name,pool,io)
  for name in ('base','repair','original','smallfit'):evaluation+=collect(e,test,name,pool,io)
 refs=[]
 for case in S.read(T.C.OUT/'comparison.json')['paired']:
  if case['split']!='anchor':continue
  for name,old in [('base','base'),('repair','repair'),('original','supervised')]:
   info=case['runs'][old];r=S.read(Path(info['campaign'])/'runs'/info['run']/'summary.json');refs.append(dict(r,condition=name,imported=True,origin=info))
 allrows=collected+evaluation+refs;report=dict(groups=stats(allrows),uncertainty=uncertainty(allrows),rows=allrows,new_episodes=len(collected)+len(evaluation),new_ticks=sum(r['ticks'] for r in collected+evaluation),elapsed_seconds=time.monotonic()-started,save_seconds=io.seconds,selected=len(selected))
 S.save(EVAL/'comparison.json',report);seal=S.finish(e,'complete');report['seals']={str(p):S.Campaign(p).verify_frozen() for p in [PARENT,COLLECT,FIT,EVAL]};S.save(HOST/'RESULT.json',report);S.save(HOST/'FINISHED.json',dict(status='complete',seal=seal,new_episodes=report['new_episodes']));io.log('finished');print('FINAL',{k:{n:v['success'] for n,v in g['conditions'].items()} for k,g in report['groups'].items() if k in ('anchor','train','test')},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);args=p.parse_args();HOST.mkdir(parents=True,exist_ok=True)
 try:prepare() if args.command=='prepare' else run()
 except BaseException as exc:S.save(HOST/f'{args.command}_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
