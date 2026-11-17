"""Eight predeclared fits on the fixed 150-demo corpus; no new CAP or RL."""
from . import large_coverage_study as L
from experiments.planar_tasks.semantic_rl import input_attribution as I
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,ctypes,gc,hashlib,json,multiprocessing as mp,random,shutil,subprocess,sys,time,traceback
import numpy as np

S=L.S;T=L.T;B=S.BASE
HOST=Path('outputs/semantic_rl_authors/input_attribution_20261117_01')
PARENT=B/'input_attribution_parent_20261117_01'
DATA=B/'input_attribution_data_20261117_01'
FIT=B/'input_attribution_fit_20261117_01'
EVAL=B/'input_attribution_evaluation_20261117_01'
WORKERS=16;UPDATES=16800;CACHE={};RUNNER=None
CONDITIONS=[f'{a}_{n}' for n in (50,150) for a in I.ARMS]


def file_record(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())


def prepare():
    manifest=Path('docs/research/CAP_Coverage_150_v1_20261116.json');fixed=S.read(manifest)
    for r in fixed['artifacts'].values():assert file_record(Path(r['path']))['sha256']==r['sha256']
    seals={str(p):S.Campaign(p).verify_frozen() for p in (L.PARENT,L.COLLECT,L.FIT,L.EVAL,T.OLD)}
    c=T.make(PARENT,0);(PARENT/'inputs').mkdir();(PARENT/'panels').mkdir()
    shutil.copyfile(manifest,PARENT/'inputs/dataset_manifest.json')
    for name in ('semantics.py','frozen_runner.py'):shutil.copyfile(L.PARENT/'inputs'/name,PARENT/'inputs'/name)
    for name in ('train','test','anchors'):shutil.copyfile(L.PARENT/'panels'/f'{name}.json',PARENT/'panels'/f'{name}.json')
    original=[r for r in S.read(T.OLD/'campaign.json')['runs'] if r.get('imported')];assert len(original)==13
    demos=[dict(id=f'original_{r["id"]}',original=True,artifact=file_record(T.OLD/'runs'/r['id']/'result.json.gz')) for r in original]
    selected=S.read(L.COLLECT/'selection.json')['selected'];buckets=defaultdict(list)
    for r in selected:buckets[r['family'],r['level']].append(r)
    rng=random.Random(2026111701)
    for k in sorted(buckets):rng.shuffle(buckets[k])
    small=[]
    while len(small)<37:
        for k in sorted(buckets):
            if buckets[k] and len(small)<37:small.append(buckets[k].pop())
    small_ids={r['case_id'] for r in small}
    demos += [dict(id=r['case_id'],original=False,family=r['family'],level=r['level'],artifact=file_record(L.COLLECT/'runs'/r['id']/'result.json.gz')) for r in selected]
    for d in demos:d['in50']=d['original'] or d['id'] in small_ids
    assert len(demos)==150 and sum(d['in50'] for d in demos)==50
    S.save(PARENT/'demonstrations.json',demos)
    train=S.read(PARENT/'panels/train.json')['cases'];diagnostic=[]
    for k in sorted({(r['family'],r['level']) for r in train}):diagnostic += [r for r in train if (r['family'],r['level'])==k][:3]
    S.save(PARENT/'panels/train_diagnostic.json',dict(cases=diagnostic))
    protocol=dict(version=I.VERSION,input_seals=seals,dataset_manifest=file_record(manifest),conditions=CONDITIONS,subset_seed=2026111701,
        arms={'G':'public geometry/history summary plus action encoding','GA':'G plus CAP advice and internal program-control signals','DA':'descriptive semantics plus CAP control signals; no geometry input','GDA':'all three'},
        control_names=sorted(I.CONTROL_NAMES),semantic_scope='A conservatively includes phase/route progress; D excludes them',
        history='G: first boundary, all-prefix mean/min/max, last8 measurement vectors, all72 scan ranges/categories; D/A: original8-frame semantic memory; both causally updated from identical legal observations/actions; G remains lossy, not a full-history learner',
        scorer='same padded input slots and factorized affine32/tanh/affine32/tanh/affine1; inactive values AND missing flags zeroed after scaling',
        training=dict(updates=UPDATES,seed=17,batch_size=32,lr=.003,gradient_clip=1.,residual_penalty=0.,normalization='training subset only',final='fixed final, no checkpoint selection',wall_seconds=1800),
        comparison_note='Feature-attribution diagnostic scorer, not a numerical reproduction of prior two-branch residual model; no causal comparison to its136/216 score. Nominal parameter count is matched; active input columns differ. All arms compute and then mask the full feature set, so runtime is not an optimized deployment-cost comparison.',
        evaluation=dict(anchors=13,train_diagnostic=27,seen_test=216,workers=WORKERS,episode_ticks=1800,episode_decisions=300,episode_wall_seconds=180,max_episodes=2048,max_ticks=3686400,condition_admission_seconds=3600),
        report='same-case paired stratified bootstrap10000; fixed seen diagnostic panel; one optimizer seed; no new-blind or semantic-sufficiency claim',
        no_new_cap=True,no_rl=True,no_new_initial_states=True)
    S.save(PARENT/'protocol.json',protocol);S.save(HOST/'PREPARED.json',dict(seal=S.finish(c,'complete'),protocol=protocol));print('PREPARED8 fits /2048evaluation episodes',flush=True)


def build_features():
    S.torch.set_num_threads(1);S.Campaign(PARENT).verify_frozen();c=T.make(DATA,0)
    source=(PARENT/'inputs/semantics.py').read_text()
    f=I.AttributionFeatures(source,fast_embedding=True,action_protocol=T.protocol(S.EventConfig(mode='duration')),execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source))
    contexts=[];candidates=[];offsets=[];targets=[];meta=[];cursor=0;started=time.monotonic()
    try:
        for j,d in enumerate(S.read(PARENT/'demonstrations.json')):
            p=Path(d['artifact']['path']);assert file_record(p)==d['artifact'];r=S.unpack(p);r['split']='train'
            ts=demonstration_traces(r,f,split='train',mode='duration')
            for k,tr in enumerate(ts):
                contexts.append(tr['panels'][0]['attribution_context']);off=[cursor]
                for panel in tr['panels']:
                    assert np.array_equal(panel['attribution_context'],contexts[-1],equal_nan=True)
                    x=panel['attribution_candidates'];candidates.append(x);cursor+=len(x);off.append(cursor)
                offsets.append(off);targets.append(tr['indices']);meta.append(dict(demo=d['id'],boundary=k,original=d['original'],in50=d['in50'],history_hash=tr['history_hash']))
            del r,ts
            if (j+1)%15==0:print('FEATURES',j+1,'/150 boundaries',len(contexts),flush=True)
        assert len(contexts)==5351 and sum(x['original'] for x in meta)==407
        np.save(DATA/'contexts.npy',np.asarray(contexts,dtype=np.float64));del contexts
        np.save(DATA/'candidates.npy',np.concatenate(candidates).astype(np.float64));del candidates
        np.save(DATA/'offsets.npy',np.asarray(offsets,dtype=np.int64));np.save(DATA/'targets.npy',np.asarray(targets,dtype=np.int64))
        S.save(DATA/'metadata.json',meta);S.save(DATA/'layout.json',f.layout)
        S.save(DATA/'extraction.json',dict(boundaries=len(meta),candidate_rows=cursor,seconds=time.monotonic()-started,original=407,subset50=sum(x['in50'] for x in meta),source_sha=S.digest(source),action_protocol=f.action_protocol))
    finally:f.close()
    S.save(HOST/'DATA_FINISHED.json',dict(seal=S.finish(c,'complete')));print('FEATURES_FROZEN',len(meta),flush=True)


def dataset():return {k:np.load(DATA/(k+'.npy'),mmap_mode='r') for k in ('contexts','candidates','offsets','targets')}


def fit_one(condition):
    # Each fit is a fresh process, releasing normalizers/tensors before collection.
    S.torch.set_num_threads(1);arm,n=condition.split('_');n=int(n)
    meta=S.read(DATA/'metadata.json');ids=[i for i,r in enumerate(meta) if n==150 or r['in50']]
    layout=S.read(DATA/'layout.json');ex=S.read(DATA/'extraction.json');ds=dataset();out=FIT/condition;out.mkdir()
    c=S.Campaign(FIT)
    with c.locked():
        c._charge(fits=1);c.state.update(phase='fitting',active_fit=condition);c._save()
    def progress(row):
        S.save(out/'progress.json',row)
        if row['update']%2000==0:print('FIT',condition,row,flush=True)
    model,curve=I.train(ds,ids,layout,arm,ex['source_sha'],ex['action_protocol'],updates=UPDATES,wall_seconds=1800,progress=progress)
    S.save(out/'curve.json',curve)
    checks=dict(train=I.accuracy(model,ds,ids),original=I.accuracy(model,ds,list(range(407))))
    if n==50:
        chosen=set(ids);checks['not_in_subset']=I.accuracy(model,ds,[i for i in range(len(meta)) if i not in chosen])
    S.save(out/'accuracy.json',checks)
    metadata=dict(condition=condition,trajectories=n,boundaries=len(ids),updates=UPDATES,seed=17,parameters=sum(p.numel() for p in model.parameters()))
    model.save(out/'model.pt',metadata);S.save(out/'metadata.json',metadata)
    restored=I.Scorer.load(out/'model.pt');assert all(S.torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    print('FIT_DONE',condition,checks['train']['exact_actions'],'/',len(ids),flush=True)


def worker(job):
    global RUNNER
    S.torch.set_num_threads(1)
    if RUNNER is None:RUNNER=T.load_frozen_runner(PARENT/'inputs/frozen_runner.py')
    name=job['condition']
    if name not in CACHE:
        # Do not accumulate eight feature runtimes across conditions.
        for _,f in CACHE.values():f.close()
        CACHE.clear()
        model=I.Scorer.load(FIT/name/'model.pt');source=(PARENT/'inputs/semantics.py').read_text()
        f=I.AttributionFeatures(source,fast_embedding=True,action_protocol=model.action_protocol,execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source));CACHE[name]=(model,f)
    model,f=CACHE[name]
    r,_=RUNNER(job['fixture'],f.source,model=model,features=f,config=S.EventConfig(mode='duration'),max_ticks=1800,max_decisions=300,wall_seconds=180,deterministic=True,seed=2026111702,fast_feedback=True,training_fastpath=False,record_details=True)
    return r


def collect(c,cases,name,pool,io):
    offset=len(S.read(c.path/'campaign.json')['runs']);start=time.monotonic();pending={};cursor=0;rows=[]
    while pending or cursor<len(cases):
        while len(pending)<WORKERS and cursor<len(cases) and time.monotonic()-start<3600:
            case=cases[cursor];i=offset+cursor;rid=f'a{i:04d}';req=dict(id=rid,case_id=case['id'],family=case['family'],level=case['level'],split=case['split'],fixture_sha256=case['fixture_sha256'],condition=name,record_details=True,status='running')
            with c.locked():c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(req);c.state.update(phase='collecting',active_condition=name);c._save()
            io.call(S.save,c.path/'runs'/rid/'request.json',req);pending[pool.submit(worker,dict(case,condition=name))]=(req,i);cursor+=1
        if not pending:break
        done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
        for future in done:
            req,i=pending.pop(future);r=future.result();assert r['fixture_sha256']==req['fixture_sha256']
            row=dict(r['summary'],**{k:v for k,v in req.items() if k!='status'})
            io.call(S.pack,c.path/'runs'/req['id']/'result.json.gz',r,compresslevel=1);io.call(S.save,c.path/'runs'/req['id']/'summary.json',row)
            with c.locked():c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][req['id']];c.state['runs'][i]=row;c._save()
            rows.append(row)
            if row['error'] or row['stop']=='wall_budget':raise RuntimeError('attribution execution failure '+req['id'])
            if len(rows)%64==0:io.log(name,completed=len(rows),success=sum(x['success'] for x in rows),elapsed_seconds=round(time.monotonic()-start,2))
    if len(rows)!=len(cases):raise TimeoutError('collection admission')
    io.log(name+' complete',episodes=len(rows),success=sum(r['success'] for r in rows));return rows


def summarize(rows):
    result={}
    for split in ('anchor','train','test'):
        rr=[r for r in rows if r['split']==split];g={}
        for name in CONDITIONS:
            sub=[r for r in rr if r['condition']==name]
            g[name]=dict(n=len(sub),success=sum(r['success'] for r in sub),family={f:dict(n=sum(x['family']==f for x in sub),success=sum(x['success'] for x in sub if x['family']==f)) for f in sorted({r['family'] for r in rr})})
        result[split]=g
    test=[r for r in rows if r['split']=='test'];index={(r['case_id'],r['condition']):r for r in test}
    strata=defaultdict(list)
    for r in test:
        if r['condition']==CONDITIONS[0]:strata[r['family'],r['level']].append(r['case_id'])
    pairs=[(f'{a}_{n}',f'{b}_{n}') for n in (50,150) for a,b in [('GA','G'),('GDA','GA'),('GDA','DA')]]+[(f'{a}_150',f'{a}_50') for a in I.ARMS]
    uncertainty={}
    for a,b in pairs:
        rng=np.random.default_rng(2026111703);samples=np.zeros(10000);gained=[];lost=[]
        for ids in strata.values():
            delta=np.array([int(index[c,a]['success'])-int(index[c,b]['success']) for c in ids]);samples+=delta[rng.integers(0,len(ids),(10000,len(ids)))].sum(1)
            gained += [c for c,d in zip(ids,delta) if d==1];lost += [c for c,d in zip(ids,delta) if d==-1]
        uncertainty[a+'_vs_'+b]=dict(gained=gained,lost=lost,difference=(len(gained)-len(lost))/216,interval=list(np.quantile(samples/216,[.025,.975])))
    return dict(groups=result,uncertainty=uncertainty,scope='fixed seen diagnostic panel; single seed; unadjusted exploratory paired intervals')


def run():
    start=time.monotonic();S.Campaign(PARENT).verify_frozen()
    subprocess.run([sys.executable,'-m',__spec__.name,'extract'],check=True)
    S.Campaign(DATA).verify_frozen();T.make(FIT,0,fits=len(CONDITIONS))
    for name in CONDITIONS:subprocess.run([sys.executable,'-m',__spec__.name,'fit','--condition',name],check=True)
    S.save(HOST/'FIT_FINISHED.json',dict(seal=S.finish(S.Campaign(FIT),'complete')))
    S.Campaign(FIT).verify_frozen()
    cases=sum([S.read(PARENT/'panels'/f'{k}.json')['cases'] for k in ('anchors','train_diagnostic','test')],[]);assert len(cases)==256
    e=T.make(EVAL,len(cases)*len(CONDITIONS));io=S.SaveTiming();rows=[]
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
        for name in CONDITIONS:rows+=collect(e,cases,name,pool,io)
    result=summarize(rows);result.update(rows=rows,new_episodes=len(rows),new_ticks=sum(r['ticks'] for r in rows),elapsed_seconds=time.monotonic()-start,save_seconds=io.seconds)
    S.save(EVAL/'comparison.json',result);S.finish(e,'complete')
    result['seals']={str(p):S.Campaign(p).verify_frozen() for p in (PARENT,DATA,FIT,EVAL)}
    S.save(HOST/'RESULT.json',result);S.save(HOST/'FINISHED.json',dict(status='complete',episodes=len(rows)))
    print('FINAL',result['groups'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('prepare','extract','fit','run'));p.add_argument('--condition',choices=CONDITIONS);args=p.parse_args();HOST.mkdir(exist_ok=True,parents=True)
    try:
        if args.command=='prepare':prepare()
        elif args.command=='extract':build_features()
        elif args.command=='fit':fit_one(args.condition)
        else:run()
    except BaseException as exc:S.save(HOST/(args.command+'_FAILURE.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
