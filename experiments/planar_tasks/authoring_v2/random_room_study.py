"""Frozen random-room CAP library -> flat supervised integration -> heldout runs."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,json,hashlib,math,multiprocessing as mp,shutil,time,traceback
import numpy as np
import torch
from .random_room_author import BASE as AUTHOR,HOST
from .random_room_tasks import OUT as SCENES
from experiments.planar_tasks.relational_search.pipeline import ROOT,Campaign,read,save,pack,unpack
from experiments.planar_tasks.relational_search.contract import digest,validate_action,History
from experiments.planar_tasks.semantic_rl.library_features import LibraryFeatures,GeometryOnlyLibraryFeatures
from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory,VERSION as STREAM_VERSION
from experiments.planar_tasks.semantic_rl.memory_flow import current_event
from experiments.planar_tasks.semantic_rl.actions import ActionTree,FIELDS
from experiments.planar_tasks.semantic_rl.environment import EventConfig
from experiments.planar_tasks.semantic_rl.scale import protocol
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I
from experiments.planar_tasks.semantic_rl.library_learning import train as library_train
from experiments.planar_tasks.semantic_rl.features import Features

BASE=ROOT/'outputs/semantic_rl/random_room_integration_20261117_01'
PARENT=BASE/'parent';DATA=BASE/'data';FIT=BASE/'fit';EVAL=BASE/'evaluation'
ARMS=('G','GA','GDA');CFG=EventConfig(mode='duration');CACHE={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def finish(c):
    with c.locked():c.state['phase']='finished';c.state['completion']='complete';c._save()
    return c.freeze()

def make(path):
    if (path/'campaign.json').exists():
        c=Campaign(path);state=read(path/'campaign.json')
        if state['status']=='frozen' or state['group']!='random-room-200-v1':raise ValueError('cannot mutate frozen/unrelated phase')
        c.verify_sources();return c
    return Campaign.create(path,fixture=read(SCENES/'scenes.json')['cases'][0]['fixture'],sequence=[dict(target_id=None,a0=44,a1=44,ticks=4)],group='random-room-200-v1',limits=dict(attempts=10000,resets=10000,execution_ticks=18000000,fits=4,rounds=0,author_turns=0))

def already_frozen(path):
    if (path/'manifest.json').exists():Campaign(path).verify_frozen();return True
    return False

def entries():return [dict(r,source=(PARENT/'inputs'/r['name']/'policy.py').read_text()) for r in read(PARENT/'library.json')]

def features():return LibraryFeatures(entries(),protocol(CFG),execution_backend='trusted_inprocess')

def prepare():
    if already_frozen(PARENT):return
    state=read(AUTHOR/'campaign.json');library=state['library']
    if not 1<=len(library)<=10 or state['reservations']:raise ValueError('finished nonempty library required')
    review=read(HOST/'SOURCE_REVIEW.json')
    if not review.get('accepted') or review['sources']!={r['name']:sha(AUTHOR/'library'/r['name']/'policy.py') for r in library}:raise ValueError('source review mismatch')
    Campaign(AUTHOR).verify_sources()
    work=Path('/home/zilitang/repo/coimagine_author_workspaces/random_room_20261117_01')
    if not already_frozen(AUTHOR):
        save(AUTHOR/'author_final.json',read(work/'FINAL.json'));shutil.copyfile(work/'FINAL.md',AUTHOR/'author_final.md');finish(Campaign(AUTHOR))
    Campaign(AUTHOR).verify_frozen()
    c=make(PARENT);frozen=[];submissions={s['id']:s for s in state['submissions']}
    for r in library:
        d=PARENT/'inputs'/r['name']
        if not d.exists():shutil.copytree(AUTHOR/'library'/r['name'],d)
        if digest((d/'policy.py').read_text())!=r['sha256']:raise ValueError('library source mismatch')
        groups=json.loads(json.dumps(submissions[r['sid']]['groups']))
        corrections=read(HOST/'GROUP_AUDIT_CORRECTIONS.json')['corrections']
        for key,group in corrections.items():
            if key in groups['state']:groups['state'][key]=group
        frozen.append(dict(name=r['name'],sid=r['sid'],groups=groups,sha256=r['sha256']))
    save(PARENT/'library.json',frozen);shutil.copyfile(SCENES/'scenes.json',PARENT/'scenes.json')
    shutil.copyfile(HOST/'SOURCE_REVIEW.json',PARENT/'inputs/source_review.json')
    shutil.copyfile(HOST/'GROUP_AUDIT_CORRECTIONS.json',PARENT/'inputs/group_audit_corrections.json')
    selected=[]
    for task in state['tasks']:
        options=[]
        for index,r in enumerate(library):
            if task not in r['success_tasks']:continue
            rid=r['runs'][task];summary=read(AUTHOR/'runs'/rid/'summary.json')
            if not summary['success']:raise ValueError('teacher success differs')
            options.append((summary['ticks'],index,rid,r['name']))
        if options:
            ticks,index,rid,name=min(options);selected.append(dict(task=task,rid=rid,teacher=name,ticks=ticks,result_path=str(AUTHOR/'runs'/rid/'result.json.gz'),sha256=sha(AUTHOR/'runs'/rid/'result.json.gz')))
    if not selected:raise ValueError('no successful teacher trajectories')
    save(PARENT/'demonstrations.json',selected)
    save(PARENT/'protocol.json',dict(arms=ARMS,seed=17,updates='min(30000,max(12000,100*ceil(N/32)))',architecture='shared conditional candidate scorer32x32 tanh, fixed-gain LayerNorm, complete legal action support',no_rl=True,selection='one lowest-time successful committed teacher per development initial state, tie by library order',author_seal=sha(AUTHOR/'manifest.json'),scenes_sha256=sha(PARENT/'scenes.json'),candidate_information='all programs consume same actual causal observation/action history; no task ID; legal public tree unchanged',test_selection='fixed final models; all200 scenes; all CAPs evaluated on40heldout; dev CAP coverage reused',fit_workers=3,extraction_workers=4,evaluation_workers=16,native_llm_token_accounting='unavailable from native subagent API; not estimated'))
    finish(c);save(HOST/'INTEGRATION_PREPARED.json',dict(teachers=len(library),demonstrations=len(selected),parent=str(PARENT)))


def bounded(jobs,fn,workers=16):
    pending={};cursor=0;rows=[]
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        while pending or cursor<len(jobs):
            while cursor<len(jobs) and len(pending)<workers:
                job=jobs[cursor];cursor+=1;pending[pool.submit(fn,job)]=job
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                job=pending.pop(future);r=future.result();rows.append(r)
                if len(rows)%8==0:print('PROGRESS',fn.__name__,len(rows),'/',len(jobs),flush=True)
            del done
    return rows


def extracted_rows(result,f,max_samples=None):
    h=f.new_history();verifier=StreamHistory() if result.get('history_protocol')==STREAM_VERSION else History()
    if result.get('action_protocol')!=protocol(CFG):raise ValueError('teacher protocol mismatch')
    rows=result['public_rows']
    for i,row in enumerate(rows):
        h.append(row['observation'],row['action']);verifier.append(row['observation'],row['action'])
        if verifier.sha256!=row['history_hash']:raise ValueError('teacher history corrupt')
        if i==len(rows)-1 or max_samples is not None and i>=max_samples:break
        teacher=rows[i+1]['action'];validate_action(teacher,current_event(h)['observation']);tree=ActionTree(current_event(h)['observation']);prefix={};panels=[];indices=[]
        with f.decision(h):
            for key in FIELDS:
                p=f.panel(h,tree,prefix);j=p['choices'].index(dict(prefix,**{key:teacher[key]}));indices.append(j);prefix=p['choices'][j];panels.append(p)
        yield dict(panels=panels,indices=indices,history_hash=h.sha256,boundary=i)


def admission():
    all_entries=entries();selected=read(PARENT/'demonstrations.json');checks=[]
    for e in all_entries:
        sample=next((r for r in selected if r['teacher']==e['name']),selected[0]);result=unpack(sample['result_path'])
        a=LibraryFeatures([e],protocol(CFG));b=LibraryFeatures([e],protocol(CFG),execution_backend='trusted_inprocess')
        try:
            aa=list(extracted_rows(result,a,max_samples=4));bb=list(extracted_rows(result,b,max_samples=4))
            for x,y in zip(aa,bb):
                for u,v in zip(x['panels'],y['panels']):
                    for key in ('attribution_context','attribution_candidates'):np.testing.assert_array_equal(u[key],v[key])
                    assert u['choices']==v['choices'] and u['proposals']==v['proposals']
            assert a.layout==b.layout;checks.append(dict(name=e['name'],boundaries=len(aa),identical=True))
        finally:a.close();b.close()
    full=LibraryFeatures(all_entries,protocol(CFG),execution_backend='trusted_inprocess')
    try:
        result=unpack(selected[0]['result_path']);aa=list(extracted_rows(result,full,max_samples=8))
        geom=GeometryOnlyLibraryFeatures(all_entries,protocol(CFG),full.layout)
        bb=list(extracted_rows(result,geom,max_samples=8))
        for x,y in zip(aa,bb):
            for u,v in zip(x['panels'],y['panels']):
                for key,kind in [('attribution_context','context'),('attribution_candidates','candidate')]:
                    mask=np.array([g in ('G','common') for g in full.layout[kind+'_groups']])
                    np.testing.assert_array_equal(u[key][...,mask],v[key][...,mask])
                assert u['choices']==v['choices']
        checks.append(dict(name='G_without_CAP_execution',boundaries=len(aa),identical_active_inputs=True))
    finally:full.close()
    save(HOST/'BACKEND_ADMISSION.json',dict(accepted=True,checks=checks))


def extract_worker(job):
    torch.set_num_threads(1)
    if 'features' not in CACHE:CACHE['features']=features()
    f=CACHE['features'];p=DATA/'shards'/job['task'];p.mkdir(parents=True,exist_ok=True)
    if (p/'complete.json').exists():return read(p/'complete.json')
    if sha(job['result_path'])!=job['sha256']:raise ValueError('teacher changed')
    started=time.monotonic();calls_before=f.calls
    result=unpack(job['result_path']);contexts=[];candidates=[];offsets=[];targets=[];meta=[];cursor=0
    for tr in extracted_rows(result,f):
        contexts.append(np.asarray(tr['panels'][0]['attribution_context'],dtype=np.float32));off=[cursor]
        for panel in tr['panels']:
            np.testing.assert_array_equal(panel['attribution_context'].astype(np.float32),contexts[-1]);x=panel['attribution_candidates'].astype(np.float32);candidates.append(x);cursor+=len(x);off.append(cursor)
        offsets.append(off);targets.append(tr['indices']);meta.append(dict(boundary=tr['boundary'],history_hash=tr['history_hash']))
    for name,value in dict(contexts=np.array(contexts),candidates=np.concatenate(candidates),offsets=np.array(offsets),targets=np.array(targets)).items():np.save(p/(name+'.npy'),value)
    save(p/'metadata.json',meta);save(p/'layout.json',f.layout)
    actions=[d['action'] for d in result['decisions']];goal_id=result['public_rows'][0]['observation']['goal']['target_id']
    action_audit=dict(non_target_requests=sum(a['target_id'] not in (None,goal_id) for a in actions),rotation_requests=sum(any(32<=a[k]<=37 for k in ('a0','a1')) for a in actions),paired_push_requests=sum(all(0<=a[k]<32 for k in ('a0','a1')) for a in actions),move_requests=sum(any(47<=a[k]<79 for k in ('a0','a1')) for a in actions),durations=sorted({a['ticks'] for a in actions}))
    row=dict(task=job['task'],path=str(p),boundaries=len(contexts),candidates=cursor,source_sha=f.sha256,action_protocol=f.action_protocol,context_width=len(contexts[0]),candidate_width=candidates[0].shape[1],action_audit=action_audit,feature_calls=f.calls-calls_before,wall_seconds=time.monotonic()-started);save(p/'complete.json',row);return row


def extract():
    if already_frozen(DATA):return
    Campaign(PARENT).verify_frozen();Campaign(PARENT).verify_sources();admission();c=make(DATA)
    rows=bounded(read(PARENT/'demonstrations.json'),extract_worker,workers=4);rows.sort(key=lambda r:r['task'])
    n=sum(r['boundaries'] for r in rows);m=sum(r['candidates'] for r in rows)
    arrays={key:np.lib.format.open_memmap(DATA/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,dtype,shape in [('contexts','float32',(n,rows[0]['context_width'])),('candidates','float32',(m,rows[0]['candidate_width'])),('offsets','int64',(n,5)),('targets','int64',(n,4))]}
    layout=read(Path(rows[0]['path'])/'layout.json');start=cursor=0;metadata=[]
    for r in rows:
        p=Path(r['path']);assert read(p/'layout.json')==layout;end=start+r['boundaries'];stop=cursor+r['candidates']
        for k in ('contexts','targets'):arrays[k][start:end]=np.load(p/(k+'.npy'),mmap_mode='r')
        arrays['candidates'][cursor:stop]=np.load(p/'candidates.npy',mmap_mode='r');arrays['offsets'][start:end]=np.load(p/'offsets.npy')+cursor
        metadata.extend(dict(v,task=r['task']) for v in read(p/'metadata.json'));start=end;cursor=stop
    for a in arrays.values():a.flush()
    save(DATA/'layout.json',layout);save(DATA/'metadata.json',metadata);save(DATA/'extraction.json',dict(demonstrations=len(rows),boundaries=n,source_sha=rows[0]['source_sha'],action_protocol=protocol(CFG),feature_calls=sum(r['feature_calls'] for r in rows),summed_worker_seconds=sum(r['wall_seconds'] for r in rows),rows=rows))
    finish(c);save(HOST/'DATA_FINISHED.json',dict(boundaries=n,demonstrations=len(rows)))


def dataset():return {k:np.load(DATA/(k+'.npy'),mmap_mode='r') for k in ('contexts','candidates','offsets','targets')}

def fit_arm(job):
    arm=job['arm'];updates=job['updates'];ds=dataset();ex=read(DATA/'extraction.json');layout=read(DATA/'layout.json');ids=list(range(len(ds['contexts'])))
    p=FIT/arm;p.mkdir(exist_ok=True)
    if (p/'complete.json').exists():return read(p/'complete.json')
    live=[]
    def progress(r):
        live.append(r);save(p/'curve_live.json',live);save(p/'progress.json',r)
        if r['update']%2000==0:print('FIT',arm,r,flush=True)
    model,curve=library_train(ds,ids,layout,arm,ex['source_sha'],ex['action_protocol'],updates=updates,wall_seconds=5400,progress=progress)
    acc=I.accuracy(model,ds,ids);save(p/'curve.json',curve);save(p/'accuracy.json',acc)
    model.save(p/'model.pt',dict(updates=updates,seed=17,demonstrations=ex['demonstrations'],boundaries=len(ids)))
    reloaded=I.Scorer.load(p/'model.pt');assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in model.state_dict().items())
    result=dict(arm=arm,updates=updates,accuracy=acc);save(p/'complete.json',result);print('FIT_DONE',arm,acc['exact_actions'],'/',len(ids),flush=True)
    return result


def fit():
    if already_frozen(FIT):return
    Campaign(DATA).verify_frozen();Campaign(DATA).verify_sources();c=make(FIT);ds=dataset();ex=read(DATA/'extraction.json');layout=read(DATA/'layout.json');ids=list(range(len(ds['contexts'])))
    canary=ids[:min(64,len(ids))]
    if (FIT/'canary.json').exists():accuracy=read(FIT/'canary.json')['accuracy']
    else:
        model,curve=library_train(ds,canary,layout,'GDA',ex['source_sha'],ex['action_protocol'],updates=2000,wall_seconds=1800)
        accuracy=I.accuracy(model,ds,canary);save(FIT/'canary.json',dict(accuracy=accuracy,curve=curve));del model
    if accuracy['exact_actions']!=len(canary):raise RuntimeError('training-only canary failed; inspect before full fits')
    updates=min(30000,max(12000,100*math.ceil(len(ids)/32)))
    bounded([dict(arm=arm,updates=updates) for arm in ARMS],fit_arm,workers=3)
    with c.locked():c.state['used']['fits']=len(ARMS)+1;c._save()
    finish(c);save(HOST/'FIT_FINISHED.json',dict(updates_per_arm=updates,arms=ARMS))


def evaluation_worker(job):
    torch.set_num_threads(1);p=Path(job['path']);p.mkdir(parents=True,exist_ok=True)
    if (p/'summary.json').exists():return read(p/'summary.json')
    key=job['condition']
    if CACHE.get('key')!=key:
        if 'f' in CACHE:CACHE['f'].close()
        CACHE.clear();all_entries=entries()
        if key in ARMS:
            model=I.Scorer.load(FIT/key/'model.pt')
            f=GeometryOnlyLibraryFeatures(all_entries,protocol(CFG),model.layout) if key=='G' else features()
        else:
            e=next(e for e in all_entries if e['name']==key);f=Features(e['source'],fast_embedding=True,action_protocol=protocol(CFG),execution_backend='trusted_inprocess',trusted_source_sha256=digest(e['source']));model=None
        CACHE.update(key=key,f=f,model=model)
    f=CACHE['f'];save(p/'started.json',dict(time=time.time()))
    result,_=rollout(job['fixture'],f.source,features=f,model=CACHE['model'],config=CFG,max_ticks=1800,max_decisions=300,wall_seconds=1800,deterministic=True,seed=17,fast_feedback=True)
    if key=='G':
        result['recommendation_audit']='not measured: geometry-only deployment skips CAP execution'
        for field in ('recommended','duration_only_deviation','motion_deviation'):result['summary']['choice_counts'][field]=None
    start=time.monotonic();pack(p/'result.json.gz',result,compresslevel=1);storage=time.monotonic()-start
    row=dict(condition=key,case_id=job['case_id'],split=job['split'],**result['summary'],storage_seconds=storage,result_path=str(p/'result.json.gz'))
    save(p/'summary.json',row);return row


def evaluate():
    if already_frozen(EVAL):
        result=read(EVAL/'result.json');result['seals']={str(p):Campaign(p).verify_frozen() for p in (AUTHOR,PARENT,DATA,FIT,EVAL)}
        save(HOST/'RESULT.json',result);save(HOST/'FINISHED.json',dict(status='complete',summary=result['summary']));return
    for p in (AUTHOR,PARENT,DATA,FIT):Campaign(p).verify_frozen()
    Campaign(PARENT).verify_sources()
    c=make(EVAL);scenes=read(PARENT/'scenes.json')['cases'];names=[e['name'] for e in entries()];jobs=[]
    for name in list(ARMS)+names:
        for r in scenes:
            if name not in ARMS and r['split']=='dev':continue
            jobs.append(dict(condition=name,case_id=r['id'],split=r['split'],fixture=r['fixture'],path=str(EVAL/'runs'/name/r['id'])))
    rows=bounded(jobs,evaluation_worker,workers=16)
    state=read(AUTHOR/'campaign.json');byid={r['id']:r for r in state['runs']}
    for member in state['library']:
        for task,rid in member['runs'].items():rows.append(dict(byid[rid],condition=member['name'],case_id=task,split='dev',imported=True))
    summary={}
    for split in ('dev','validation','test'):
        rr=[r for r in rows if r['split']==split];summary[split]={}
        for name in list(ARMS)+names:
            sub=[r for r in rr if r['condition']==name];summary[split][name]=dict(n=len(sub),success=sum(r['success'] for r in sub),invalid=sum(bool(r['error']) for r in sub))
        union={r['case_id'] for r in rr if r['condition'] in names and r['success']}
        summary[split]['teacher_union']=dict(success=len(union),n=sum(s['split']==split for s in scenes),offline_upper_reference=True)
        for arm in ARMS:
            wins={r['case_id'] for r in rr if r['condition']==arm and r['success']}
            summary[split][arm].update(teacher_union_retained=len(wins&union),teacher_union_lost=sorted(union-wins),outside_teacher_union=sorted(wins-union))
    best=max(names,key=lambda n:summary['dev'][n]['success'])
    result=dict(summary=summary,best_single_cap_selected_by_dev=best,rows=rows,new_eval_episodes=len(jobs),new_eval_ticks=sum(r['ticks'] for r in rows if not r.get('imported')),library=state['library'],native_author_used=state['used'],limitations=['single training seed','40 heldout scenes,20 final test; exploratory uncertainty','geometry-valid does not guarantee dynamic solvability','teacher success union is not executable oracle'])
    save(EVAL/'result.json',result)
    with c.locked():
        c.state['used'].update(attempts=len(jobs),resets=len(jobs),execution_ticks=result['new_eval_ticks']);c.state['runs']=[{k:r.get(k) for k in ('condition','case_id','split','success','ticks','stop','error','result_path')} for r in rows if not r.get('imported')];c._save()
    finish(c)
    result['seals']={str(p):Campaign(p).verify_frozen() for p in (AUTHOR,PARENT,DATA,FIT,EVAL)}
    save(HOST/'RESULT.json',result);save(HOST/'FINISHED.json',dict(status='complete',summary=summary));print('FINISHED',summary,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','extract','fit','evaluate']);a=p.parse_args()
    try:globals()[a.phase]()
    except BaseException:save(HOST/(a.phase+'_FAILURE.json'),dict(traceback=traceback.format_exc(),at=time.time()));raise
