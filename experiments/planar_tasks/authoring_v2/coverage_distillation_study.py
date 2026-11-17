"""One controlled coverage-driven data expansion, fixed semantics, no CAP/PPO edits."""
from . import bounded_rl_study as S
from . import cap_coverage_study as C
from .random_curriculum import draw,BOUNDS
from experiments.planar_tasks.semantic_rl.coverage_fit import fit_updates
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces,warm_start
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.trusted_native import compile_host
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import Counter,defaultdict
from types import SimpleNamespace
import argparse,copy,hashlib,importlib.util,math,multiprocessing as mp,random,shutil,time,traceback

B=S.BASE;HOST=Path('outputs/semantic_rl_authors/coverage_distillation_20261116_01')
PARENT=B/'coverage_distillation_parent_20261116_01';DEV=B/'coverage_distillation_development_20261116_01';FIT=B/'coverage_distillation_fit_20261116_01';EVAL=B/'coverage_distillation_evaluation_20261116_01'
OLD=B/'local_cap_integration_20261116_01';LONG=B/'random_parameter_direction_long_parent_20261116_01'
UPDATES=3000;WORKERS=4;CACHE={};RUNNER=None

def load_frozen_runner(path):
 # Loading frozen source must not add a __pycache__ file to its sealed inventory.
 spec=importlib.util.spec_from_file_location('experiments.planar_tasks.semantic_rl._coverage_distillation_frozen_runner',path)
 module=importlib.util.module_from_spec(spec)
 exec(compile(Path(path).read_text(),str(path),'exec'),module.__dict__)
 return module.rollout

def make(p,n,fits=0):
 c=S.Campaign.create(p,limits=dict(attempts=n,resets=n,execution_ticks=n*1800,rounds=0,fits=fits,author_turns=0),model=dict(provider='none',name='coverage-driven supervised distillation'))
 with c.locked():c.state.update(profile='coverage-distillation-v1',runs=[],phase='ready');c._save()
 return c

def prepare():
 seals={str(p):S.Campaign(p).verify_frozen() for p in [OLD,LONG,C.OUT]}
 registry={r['id']:r for r in S.read('docs/research/Policy_Registry.json')['entries']}
 anchors=S.read(LONG/'panels/anchors.json')['cases'];seen={c['fixture_sha256'] for c in anchors}
 excluded=[]
 for path in [LONG,B/'random_parameter_parent_20261116_01',B/'duration_stream_continuation_parent_20261115_01']:
  for panel in (path/'panels').glob('*.json'):
   for c in S.read(panel).get('cases',[]):seen.add(c['fixture_sha256']);excluded.append(c['fixture_sha256'])
 groups={f:[c for c in anchors if c['family']==f] for f in sorted({c['family'] for c in anchors})};panels={}
 for split,seed in [('development',2026111621),('holdout',2026111622)]:
  rng=random.Random(seed);cases=[]
  for family in groups:
   for level in ('small','medium'):
    for _ in range(4):cases.append(draw(rng.choice(groups[family]),level,rng,identifier=f'{split}_{len(cases):04d}',split='train' if split=='development' else 'test',seen=seen))
  panels[split]=dict(seed=seed,cases=cases,bounds=BOUNDS)
 assert not {c['fixture_sha256'] for c in panels['development']['cases']}&{c['fixture_sha256'] for c in panels['holdout']['cases']}
 c=make(PARENT,0);(PARENT/'inputs').mkdir()
 for key,rid in [('base','CAP-001.v4'),('repair','CAP-002.v2'),('semantics','SEM-001.v1')]:
  entry=registry[rid];assert C.sha(entry['artifact'])==entry['file_sha256'];shutil.copyfile(entry['artifact'],PARENT/'inputs'/f'{key}.py')
 shutil.copyfile(registry['SUP-002.v1']['artifact'],PARENT/'inputs/original.pt')
 shutil.copyfile(C.OUT/'inputs/frozen_runner.py',PARENT/'inputs/frozen_runner.py')
 shutil.copyfile(OLD/'semantic_fits/f000/dataset.json.gz',PARENT/'inputs/original_dataset.json.gz')
 for name,panel in panels.items():S.save(PARENT/'panels'/f'{name}.json',panel)
 S.save(PARENT/'panels/anchors.json',dict(cases=anchors))
 original=[r for r in S.read(OLD/'campaign.json')['runs'] if r.get('imported')]
 assert len(original)==13
 S.save(PARENT/'original_demos.json',original)
 p=dict(input_seals=seals,updates=UPDATES,seed=17,workers=WORKERS,max_new_demos=12,fit_wall_seconds=600,collection_admission_seconds=1200,
   protocol='duration',max_ticks=1800,max_decisions=300,episode_wall_seconds=180,excluded_fixture_hashes=sorted(set(excluded)),
   selection='teacher success and original learner failure; family/level round-robin; lowest-cost successful teacher, base wins ties',
   coverage='observed stage/action/geometry buckets, not exhaustive source branch or state-space coverage',
   arms=dict(control='original13 from scratch',expanded='original13 plus at most12 development teacher success trajectories'),
   no_new_semantics=True,no_ppo=True,heldout='fixed before sampling; outcomes not queried until both fitted checkpoints frozen')
 S.save(PARENT/'protocol.json',p);S.save(HOST/'PREPARED.json',dict(seal=S.finish(c,'complete'),protocol=p))
 print('PREPARED development24 holdout24; max266 executions; two3000-update fits',flush=True)

def worker(job):
 global RUNNER
 S.torch.set_num_threads(1)
 if RUNNER is None:
  RUNNER=load_frozen_runner(PARENT/'inputs/frozen_runner.py')
 condition=job['condition'];cap=condition in ('base','repair');key=condition
 if key not in CACHE:
  source=(PARENT/'inputs'/f'{condition if cap else "semantics"}.py').read_text()
  model=None if cap else S.Policy.load(PARENT/'inputs/original.pt' if condition=='original' else FIT/'models'/f'{condition}.pt',S.digest(source))[0]
  f=S.Features(source,schema=S.Schema(tuple(model.schema['slots'])) if model else None,fast_embedding=True,action_protocol=protocol(S.EventConfig(mode='duration')),execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source))
  CACHE[key]=(source,model,f)
 source,model,f=CACHE[key]
 # Canonical history required for demonstration integrity verification.
 r,_=RUNNER(job['fixture'],source,model=model,features=f,config=S.EventConfig(mode='duration'),max_ticks=1800,max_decisions=300,wall_seconds=180,deterministic=True,seed=job['seed'],fast_feedback=True,training_fastpath=False,record_details=True)
 return r

def collect(c,cases,condition,pool,io):
 jobs=[];offset=len(S.read(c.path/'campaign.json')['runs']);started=time.monotonic()
 # At most four in-flight jobs; no hidden backlog beyond admission deadline.
 pending={};cursor=0;rows=[]
 from concurrent.futures import wait,FIRST_COMPLETED
 while pending or cursor<len(cases):
  while len(pending)<WORKERS and cursor<len(cases) and time.monotonic()-started<1200:
   case=cases[cursor];i=offset+cursor;rid=f'a{i:04d}';req=dict(id=rid,case_id=case['id'],family=case['family'],level=case['level'],split=case['split'],fixture_sha256=case['fixture_sha256'],condition=condition,record_details=True,status='running')
   with c.locked():c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(req);c.state['phase']='collecting';c._save()
   io.call(S.save,c.path/'runs'/rid/'request.json',req);pending[pool.submit(worker,dict(case,condition=condition,seed=2026111623+cursor))]=(req,i);cursor+=1
  if not pending:break
  done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
  for future in done:
   req,i=pending.pop(future);r=future.result();assert r['fixture_sha256']==req['fixture_sha256']
   row=dict(r['summary'],**{k:v for k,v in req.items() if k!='status'})
   io.call(S.pack,c.path/'runs'/req['id']/'result.json.gz',r,compresslevel=1);io.call(S.save,c.path/'runs'/req['id']/'summary.json',row)
   with c.locked():c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][req['id']];c.state['runs'][i]=row;c._save()
   rows.append(row)
   if row['error'] or row['stop']=='wall_budget':raise RuntimeError('execution error '+req['id'])
 if len(rows)!=len(cases):raise TimeoutError('collection admission budget')
 io.log(condition,episodes=len(rows),success=sum(r['success'] for r in rows));return sorted(rows,key=lambda r:r['id'])

def choose(rows,maximum=12):
 grouped=defaultdict(dict)
 for r in rows:grouped[r['case_id']][r['condition']]=r
 buckets=defaultdict(list)
 for cid,rs in sorted(grouped.items()):
  if rs['original']['success']:continue
  valid=[rs[k] for k in ('base','repair') if rs[k]['success']]
  if not valid:continue
  best=min(valid,key=lambda r:(r['ticks'],r['condition']!='base'))
  buckets[best['family'],best['level']].append(best)
 selected=[]
 while len(selected)<maximum and any(buckets.values()):
  for k in sorted(buckets):
   if buckets[k] and len(selected)<maximum:selected.append(buckets[k].pop(0))
 return selected

def behavior_keys(result,source_path):
 directory,_=compile_host(source_path.read_text());spec=importlib.util.spec_from_file_location('_coimagine_semantics',directory/'semantics.so');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 memory=m.initial_memory();keys=Counter();phases=Counter();edges=Counter();previous=None
 for i,row in enumerate(result['public_rows'][:-1]):
  o=row['observation'];event=dict(copy.deepcopy(row),executed_ticks=None if i==0 else o['tick']-result['public_rows'][i-1]['observation']['tick']);memory=m.update_memory(memory,event);phase=m.plan(memory,o)[4];a=result['public_rows'][i+1]['action'];target=o['goal']['target_id']
  pts=next((e['points'] for e in o['entities'] if e['id']==target),[])
  def sign(x):return -1 if x<-.2 else 1 if x>.2 else 0
  if pts:
   cx=(min(p[0] for p in pts)+max(p[0] for p in pts))/2;cy=(min(p[1] for p in pts)+max(p[1] for p in pts))/2
   geometry=(sign(o['goal']['xy'][0]-cx),sign(o['goal']['xy'][1]-cy),*[(sign(r['own']['x']-cx),sign(r['own']['y']-cy)) for r in o['robots']])
  else:geometry=('unknown',)
  def intent(x):
   if x<32:return ('PUSH',x//4)
   if 32<=x<=37:return ('ROTATE',int(x>=35))
   if x>=47:return ('MOVE',(x-47)//4)
   return ('OTHER',x)
  key=repr((phase,'global' if a['target_id'] is None else 'goal' if a['target_id']==target else 'other',intent(a['a0']),intent(a['a1']),geometry))
  keys[key]+=1;phases[phase]+=1
  if previous is not None and phase!=previous:edges[previous+'->'+phase]+=1
  previous=phase
 return dict(keys=dict(keys),phases=dict(phases),transitions=dict(edges))

def coverage(selected,rows):
 original=defaultdict(Counter);expanded=defaultdict(Counter);probe=defaultdict(Counter);details=[]
 for r in S.read(PARENT/'original_demos.json'):
  teacher='base' if r['origin']['source_id']=='s000' else 'repair';d=behavior_keys(S.unpack(OLD/'runs'/r['id']/'result.json.gz'),PARENT/'inputs'/f'{teacher}.py');original[teacher].update(d['keys']);details.append(dict(origin='original',run=r['id'],teacher=teacher,**d))
 for name in ('base','repair'):expanded[name].update(original[name])
 chosen={r['id'] for r in selected}
 for r in rows:
  if r['condition'] not in ('base','repair') or not r['success']:continue
  name=r['condition'];d=behavior_keys(S.unpack(DEV/'runs'/r['id']/'result.json.gz'),PARENT/'inputs'/f'{name}.py');probe[name].update(d['keys'])
  if r['id'] in chosen:expanded[name].update(d['keys'])
  details.append(dict(origin='development',run=r['id'],teacher=name,selected=r['id'] in chosen,**d))
 report={}
 for name in ('base','repair'):
  total=sum(probe[name].values());report[name]=dict(original_buckets=len(original[name]),expanded_buckets=len(expanded[name]),probe_buckets=len(probe[name]),original_probe_unique_covered=len(set(original[name])&set(probe[name])),expanded_probe_unique_covered=len(set(expanded[name])&set(probe[name])),probe_decisions=total,original_probe_decisions_covered=sum(v for k,v in probe[name].items() if k in original[name]),expanded_probe_decisions_covered=sum(v for k,v in probe[name].items() if k in expanded[name]))
 return dict(scope='finite observable behavior buckets; no exhaustive branch-coverage claim',teachers=report,details=details)

def fit(selected):
 c=make(FIT,0,fits=2);(FIT/'models').mkdir();source=(PARENT/'inputs/semantics.py').read_text();original=S.unpack(PARENT/'inputs/original_dataset.json.gz');assert len(original)==407
 sup,_=S.Policy.load(PARENT/'inputs/original.pt',S.digest(source));features=S.features(source,sup)
 expanded=list(original);newinfo=[];start=time.monotonic()
 try:
  for r in selected:
   data=S.unpack(DEV/'runs'/r['id']/'result.json.gz');data['split']='train'
   ts=demonstration_traces(data,features,split='train',mode='duration');expanded.extend(ts);newinfo.append(dict(run=r['id'],case_id=r['case_id'],teacher=r['condition'],boundaries=len(ts)))
   print('FEATURES',r['case_id'],len(ts),'total',len(expanded),flush=True)
  S.save(FIT/'data.json',dict(original_trajectories=13,original_boundaries=407,new=newinfo,total_boundaries=len(expanded),semantic_sha256=S.digest(source)))
  # Focused equivalence check against the existing warm-start math: 2 updates.
  a,_=warm_start(original[:4],features,epochs=2,seed=17);b,_=fit_updates(original[:4],features,updates=2,seed=17)
  assert all(S.torch.equal(v,b.state_dict()[k]) for k,v in a.state_dict().items())
  S.save(FIT/'fit_equivalence.json',dict(updates=2,all_model_tensors_exact=True))
  del a,b
  for name,data in [('control',original),('expanded',expanded)]:
   with c.locked():c._charge(fits=1);c.state.update(phase='fitting',active_fit=name);c._save()
   def progress(row):
    S.save(FIT/f'{name}_progress.json',row)
    if row['update']%500==0:print('FIT',name,row,flush=True)
   model,curve=fit_updates(data,features,updates=UPDATES,wall_seconds=600,on_progress=progress)
   match=0
   with S.torch.no_grad():
    for tr in data:match+=all(int(model.distribution(p)[0].logits.argmax())==i for p,i in zip(tr['panels'],tr['indices']))
   model.save(FIT/'models'/f'{name}.pt',dict(updates=UPDATES,data_boundaries=len(data),seed=17,source='coverage-driven supervised pilot',original_model='SUP-002.v1'))
   S.save(FIT/f'{name}_curve.json',curve);S.save(FIT/f'{name}_fit.json',dict(updates=UPDATES,boundaries=len(data),teacher_forced_action_matches=match,last=curve[-1]))
   print('FIT_DONE',name,match,'/',len(data),flush=True)
 finally:features.close()
 S.save(FIT/'timing.json',dict(total_seconds=time.monotonic()-start));seal=S.finish(c,'complete');S.save(HOST/'FIT_FINISHED.json',dict(seal=seal))


def compare(rows):
 groups={}
 for split in ('anchor','train','test'):
  sub=[r for r in rows if r['split']==split];cases=sorted({r['case_id'] for r in sub});index={(r['case_id'],r['condition']):r for r in sub};conditions=sorted({r['condition'] for r in sub})
  sets={name:{cid for cid in cases if (cid,name) in index and index[cid,name]['success']} for name in conditions}
  union=sets.get('base',set())|sets.get('repair',set())
  groups[split]=dict(n=len(cases),conditions={name:dict(evaluated=sum((cid,name) in index for cid in cases),success=len(sets[name]),successful_cases=sorted(sets[name])) for name in conditions},teacher_union=len(union),learned={name:dict(retained_union=sorted(sets[name]&union),lost_union=sorted(union-sets[name]),outside_union=sorted(sets[name]-union)) for name in ('original','control','expanded') if name in sets},paired=[dict(case_id=cid,success={name:bool(index[cid,name]['success']) if (cid,name) in index else None for name in conditions}) for cid in cases])
 return groups

def run():
 start=time.monotonic();S.torch.set_num_threads(1);S.Campaign(PARENT).verify_frozen();development=S.read(PARENT/'panels/development.json')['cases'];anchors=S.read(PARENT/'panels/anchors.json')['cases'];io=S.SaveTiming();d=make(DEV,72)
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
  rows=[]
  for name in ('base','repair','original'):rows+=collect(d,development,name,pool,io)
 selected=choose(rows);S.save(DEV/'selection.json',dict(rule=S.read(PARENT/'protocol.json')['selection'],selected=selected));S.save(DEV/'coverage.json',coverage(selected,rows));S.save(HOST/'DEVELOPMENT_FINISHED.json',dict(seal=S.finish(d,'complete'),selected=len(selected)))
 print('SELECTED',[(r['case_id'],r['condition']) for r in selected],flush=True)
 if not selected:
  S.save(HOST/'FINISHED.json',dict(status='complete_no_development_gap',elapsed_seconds=time.monotonic()-start));return
 fit(selected)
 # Only now read held-out fixtures for execution; fit is already sealed.
 S.Campaign(FIT).verify_frozen();heldout=S.read(PARENT/'panels/holdout.json')['cases'];e=make(EVAL,194);newrows=[]
 with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
  for name in ('control','expanded'):newrows+=collect(e,anchors+development+heldout,name,pool,io)
  for name in ('base','repair','original'):newrows+=collect(e,heldout,name,pool,io)
 # Original anchor reference is from the fully paired frozen coverage study.
 oldcompare=S.read(C.OUT/'comparison.json')['paired'];refs=[]
 for case in oldcompare:
  if case['split']!='anchor':continue
  for name,old in [('base','base'),('repair','repair'),('original','supervised')]:
   info=case['runs'][old];r=S.read(Path(info['campaign'])/'runs'/info['run']/'summary.json');refs.append(dict(r,condition=name,imported=True,origin=info))
 combined=rows+newrows+refs;result=compare(combined);S.save(EVAL/'comparison.json',dict(groups=result,rows=combined));seal=S.finish(e,'complete')
 report=dict(status='complete',selected=[dict(case_id=r['case_id'],teacher=r['condition'],run=r['id']) for r in selected],groups=result,new_episodes=len(rows)+len(newrows),new_ticks=sum(r['ticks'] for r in rows+newrows),elapsed_seconds=time.monotonic()-start,save_seconds=io.seconds,seals={str(p):S.Campaign(p).verify_frozen() for p in [PARENT,DEV,FIT,EVAL]})
 S.save(HOST/'RESULT.json',report);S.save(HOST/'FINISHED.json',dict(status='complete',seal=seal));io.log('finished')
 print('FINAL',{k:{n:v['success'] for n,v in g['conditions'].items()} for k,g in result.items()},flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run']);args=p.parse_args();HOST.mkdir(parents=True,exist_ok=True)
 try:prepare() if args.command=='prepare' else run()
 except BaseException as exc:S.save(HOST/f'{args.command}_FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise
