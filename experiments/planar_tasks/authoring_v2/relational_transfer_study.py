"""Fixed-author -> teacher collection -> four input fits -> sealed transfer test.

No extra author calls or RL. Gates fail explicitly; no automatic task substitution.
"""
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,hashlib,json,multiprocessing as mp,shutil,subprocess,sys,time,traceback
import numpy as np
from . import large_coverage_study as L
from .relational_transfer_author import TransferCampaign
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I
from experiments.planar_tasks.semantic_rl.relational_transfer_features import TransferFeatures
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.learning import demonstration_traces
from experiments.planar_tasks.semantic_rl.runner import rollout
from experiments.planar_tasks.relational_search.transfer_tasks import cases as root_cases
S=L.S;T=L.T;BASE=S.BASE
HOST=Path('outputs/semantic_rl_authors/rel_transfer_20261117_01')
AUTHOR=BASE/'rel_transfer_author_20261117_01'
WORK=Path('/home/zilitang/repo/coimagine_author_workspaces/rel_transfer_20261117_01')
PARENT=BASE/'rel_transfer_parent_20261117_01';COLLECT=BASE/'rel_transfer_collection_20261117_01'
DATA=BASE/'rel_transfer_data_20261117_01';FIT=BASE/'rel_transfer_fit_20261117_01';EVAL=BASE/'rel_transfer_evaluation_20261117_01'
WORKERS=16;UPDATES=16800;CACHE={};RUNNER=None


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def final_manifest():
    final=S.read(WORK/'FINAL.json');state=S.read(AUTHOR/'campaign.json')
    if final.get('status')!='ready':raise ValueError('author returned partial; no automatic continuation')
    source_ids={r['id'] for r in state['submissions']};fid=final['feature_sid'];baselines=list(dict.fromkeys(final['cap_baseline_sids']))
    if fid not in source_ids or not 1<=len(baselines)<=4 or not set(baselines)<=source_ids:raise ValueError('declared source IDs')
    teachers=final['task_teachers']
    if set(teachers)!=set(state['tasks']):raise ValueError('four teacher assignments required')
    runs={r['id']:r for r in state['runs']}
    for task,t in teachers.items():
        r=runs[t['run_id']]
        if r['task']!=task or r['sid']!=t['sid'] or not r.get('success') or r.get('fit_id') is not None:raise ValueError('teacher lacks exact successful evidence')
        if t['sid'] not in baselines:raise ValueError('include every teacher in direct CAP baselines')
    # The final feature source must have real first-proposal tests on all four,
    # but it may differ from the successful local teachers.
    if {r['task'] for r in state['runs'] if r['sid']==fid and r.get('fit_id') is None}!=set(state['tasks']):raise ValueError('final feature source not checked on every root')
    return final,state


def prepare():
    final,state=final_manifest();TransferCampaign(AUTHOR).verify_sources()
    if any(r.get('status')=='running' for r in state['runs']) or state['reservations']:raise ValueError('author work/reservation still running')
    review=S.read(HOST/'SOURCE_REVIEW.json')
    if review.get('feature_sid')!=final['feature_sid'] or not review.get('accepted'):raise ValueError('host source/group review missing')
    for sid in set(final['cap_baseline_sids'])|{final['feature_sid']}:
        if review.get('source_hashes',{}).get(sid)!=sha(AUTHOR/'semantics'/f'{sid}.py'):raise ValueError('reviewed source changed')
    # Author exits before host seals; never silently reset quotas or alter source.
    S.save(AUTHOR/'author_final.json',final)
    author_seal=S.finish(S.Campaign(AUTHOR),'complete')
    c=T.make(PARENT,0);(PARENT/'inputs').mkdir();(PARENT/'panels').mkdir()
    manifest=S.read(HOST/'panels/manifest.json')
    for name,expected in manifest.items():
        panel=S.read(HOST/'panels'/name);assert S.digest(panel)==expected
        shutil.copyfile(HOST/'panels'/name,PARENT/'panels'/name)
    source_ids=set(final['cap_baseline_sids'])|{final['feature_sid']}
    for sid in source_ids:shutil.copyfile(AUTHOR/'semantics'/f'{sid}.py',PARENT/'inputs'/f'{sid}.py')
    item=next(r for r in state['submissions'] if r['id']==final['feature_sid'])
    S.save(PARENT/'inputs/groups.json',item['groups']);S.save(PARENT/'inputs/final.json',final)
    shutil.copyfile(Path(sys.modules[rollout.__module__].__file__),PARENT/'inputs/frozen_runner.py')
    protocol=dict(author_seal=author_seal,final=final,panel_hashes=manifest,source_hashes={s:sha(PARENT/'inputs'/f'{s}.py') for s in source_ids},groups_sha256=sha(PARENT/'inputs/groups.json'),conditions=list(I.ARMS),updates=UPDATES,seed=17,batch=32,lr=.003,clip=1,optimizer_wall_seconds=1800,workers=WORKERS,max_ticks=1800,max_decisions=300,episode_wall_seconds=180,no_rl=True,selection='all successful training teachers; fixed final models; no outcome-based panel changes',input_note='DA removes additional G channels, but all arms share public goal-relative action encoding; not geometry-free',source_review='host must write SOURCE_REVIEW.json before proceeding')
    S.save(PARENT/'protocol.json',protocol)
    shutil.copyfile(HOST/'SOURCE_REVIEW.json',PARENT/'inputs/source_review.json')
    S.save(HOST/'PARENT_FINISHED.json',dict(seal=S.finish(c,'complete')))


def source_features(sid,geometry=False):
    source=(PARENT/'inputs'/f'{sid}.py').read_text();kw=dict(fast_embedding=True,action_protocol=T.protocol(S.EventConfig(mode='duration')),execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source))
    return TransferFeatures(source,groups=S.read(PARENT/'inputs/groups.json'),**kw) if geometry else Features(source,**kw)


def worker(job):
    global RUNNER
    S.torch.set_num_threads(1)
    if RUNNER is None:RUNNER=T.load_frozen_runner(PARENT/'inputs/frozen_runner.py')
    key=job['condition'];sid=job['sid'];learned=key in I.ARMS
    if key not in CACHE:
        for m,f in CACHE.values():f.close()
        CACHE.clear();f=source_features(sid,geometry=learned);m=I.Scorer.load(FIT/key/'model.pt') if learned else None;CACHE[key]=(m,f)
    m,f=CACHE[key]
    result,_=RUNNER(job['fixture'],f.source,model=m,features=f,config=S.EventConfig(mode='duration'),max_ticks=1800,max_decisions=300,wall_seconds=180,deterministic=True,seed=17,fast_feedback=True,training_fastpath=False,record_details=True)
    return result


def collect(c,jobs):
    pending={};cursor=0;rows=[];io=S.SaveTiming()
    with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
        while pending or cursor<len(jobs):
            while len(pending)<WORKERS and cursor<len(jobs):
                job=jobs[cursor];rid=f'a{cursor:04d}';request=dict(id=rid,**{k:v for k,v in job.items() if k!='fixture'},status='running')
                with c.locked():
                    c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(request);c.state.update(phase='collecting',active_condition=job['condition']);c._save()
                io.call(S.save,c.path/'runs'/rid/'request.json',request);pending[pool.submit(worker,job)]=(request,cursor);cursor+=1
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                request,index=pending.pop(future);result=future.result();assert result['fixture_sha256']==request['fixture_sha256']
                result.update(split='train' if request['split']=='train' else 'test',task=request['case_id'],source_id=request['sid'])
                row=dict(request,**result['summary']);io.call(S.pack,c.path/'runs'/request['id']/'result.json.gz',result,compresslevel=1);io.call(S.save,c.path/'runs'/request['id']/'summary.json',row)
                with c.locked():
                    c.state['used']['execution_ticks']-=1800-row['ticks'];del c.state['reservations'][request['id']];c.state['runs'][index]=row;c._save()
                rows.append(row)
                if row['error'] or row['stop']=='wall_budget':raise RuntimeError('transfer rollout interface/wall error '+request['id'])
                if len(rows)%16==0:io.log('transfer',completed=len(rows),success=sum(r['success'] for r in rows),total=len(jobs))
    return sorted(rows,key=lambda r:r['id'])


def teacher_jobs():
    final=S.read(PARENT/'inputs/final.json');mapping={r['base_id']:f'dev_{i:02d}' for i,r in enumerate(S.read(PARENT/'panels/roots.json')['cases'][:4])}
    cases=S.read(PARENT/'panels/train.json')['cases']+[r for r in S.read(PARENT/'panels/roots.json')['cases'] if r['split']=='root_training']
    return [dict(case_id=r['id'],base_id=r['base_id'],layout=r['layout'],relation=r['relation'],split='train',fixture=r['fixture'],fixture_sha256=r['fixture_sha256'],sid=final['task_teachers'][mapping[r['base_id']]]['sid'],condition='teacher_'+final['task_teachers'][mapping[r['base_id']]]['sid']) for r in cases]


def admission_checks(rows):
    final=S.read(PARENT/'inputs/final.json');roots=S.read(PARENT/'panels/roots.json')['cases'][:4];checked=[]
    source=(PARENT/'inputs'/f"{final['feature_sid']}.py").read_text()
    cfg=T.protocol(S.EventConfig(mode='duration'))
    isolated=Features(source,fast_embedding=True,action_protocol=cfg)
    trusted=source_features(final['feature_sid'])
    try:
        for i,root in enumerate(roots):
            r=next(x for x in rows if x['case_id']==root['id']);teacher=final['task_teachers'][f'dev_{i:02d}']
            before=S.unpack(AUTHOR/'runs'/teacher['run_id']/'result.json.gz');after=S.unpack(COLLECT/'runs'/r['id']/'result.json.gz')
            assert before['source_sha256']==after['source_sha256'] and before['summary']['ticks']==after['summary']['ticks']
            assert [x['action'] for x in before['decisions']]==[x['action'] for x in after['decisions']]
            assert before['replay']['frames'][-1]==after['replay']['frames'][-1]
            assert after['summary']['success']
            aa=demonstration_traces(after,isolated,split='train',mode='duration',max_samples=2)
            bb=demonstration_traces(after,trusted,split='train',mode='duration',max_samples=2)
            for a,b in zip(aa,bb):
                for ap,bp in zip(a['panels'],b['panels']):
                    for key in ['semantic','raw']:np.testing.assert_array_equal(np.asarray(ap[key],dtype=float),np.asarray(bp[key],dtype=float))
                    assert ap['choices']==bp['choices'] and ap['proposals']==bp['proposals']
            checked.append(dict(case_id=root['id'],exact_final_frame=True,exact_action_sequence=True,feature_panels=len(aa)*4))
    finally:isolated.close();trusted.close()
    S.save(HOST/'BACKEND_ADMISSION.json',dict(accepted=True,checks=checked))


def build_data():
    c=T.make(DATA,0);final=S.read(PARENT/'inputs/final.json');f=source_features(final['feature_sid'],True)
    rows=S.read(COLLECT/'campaign.json')['runs'];selected=[r for r in rows if r['success']]
    if {r['base_id'] for r in selected}!=set(r['base_id'] for r in teacher_jobs()):raise ValueError('teacher successes do not cover four training categories')
    contexts=[];candidates=[];offsets=[];targets=[];meta=[];cursor=0
    try:
        for r in selected:
            result=S.unpack(COLLECT/'runs'/r['id']/'result.json.gz');traces=demonstration_traces(result,f,split='train',mode='duration')
            for j,tr in enumerate(traces):
                contexts.append(tr['panels'][0]['attribution_context']);off=[cursor]
                for p in tr['panels']:
                    np.testing.assert_array_equal(p['attribution_context'],contexts[-1]);x=p['attribution_candidates'];candidates.append(x);cursor+=len(x);off.append(cursor)
                offsets.append(off);targets.append(tr['indices']);meta.append(dict(run_id=r['id'],case_id=r['case_id'],base_id=r['base_id'],boundary=j,history_hash=tr['history_hash']))
        np.save(DATA/'contexts.npy',np.asarray(contexts));np.save(DATA/'candidates.npy',np.concatenate(candidates));np.save(DATA/'offsets.npy',np.array(offsets));np.save(DATA/'targets.npy',np.array(targets));S.save(DATA/'layout.json',f.layout);S.save(DATA/'metadata.json',meta)
        S.save(DATA/'extraction.json',dict(demonstrations=len(selected),attempts=len(rows),boundaries=len(meta),source_sha=f.sha256,action_protocol=f.action_protocol,selection='all successful training episodes, no truncation',selected_runs=[r['id'] for r in selected]))
    finally:f.close()
    S.save(HOST/'DATA_FINISHED.json',dict(seal=S.finish(c,'complete')))


def dataset():return {k:np.load(DATA/(k+'.npy'),mmap_mode='r') for k in ['contexts','candidates','offsets','targets']}


def fit_one(arm,updates=UPDATES):
    c=S.Campaign(FIT);out=FIT/arm;out.mkdir();ds=dataset();ex=S.read(DATA/'extraction.json');ids=list(range(len(ds['contexts'])))
    with c.locked():c._charge(fits=1);c.state.update(phase='fitting',active_fit=arm);c._save()
    def progress(row):
        S.save(out/'progress.json',row)
        if row['update']%2000==0:print('FIT',arm,row,flush=True)
    model,curve=I.train(ds,ids,S.read(DATA/'layout.json'),arm,ex['source_sha'],ex['action_protocol'],updates=updates,progress=progress)
    S.save(out/'curve.json',curve);S.save(out/'accuracy.json',I.accuracy(model,ds,ids));model.save(out/'model.pt',dict(arm=arm,updates=updates,seed=17,demonstrations=ex['demonstrations'],boundaries=len(ids)))
    restored=I.Scorer.load(out/'model.pt');assert all(S.torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items())
    print('FIT_DONE',arm,flush=True)


def summarize(rows):
    groups={}
    for split in ['train','root_training','root_seen_diagnostic','iid','transfer']:
        groups[split]={}
        rr=[r for r in rows if r['split']==split]
        for name in sorted({r['condition'] for r in rr}):
            sub=[r for r in rr if r['condition']==name];groups[split][name]=dict(n=len(sub),success=sum(r['success'] for r in sub),by_layout={l:dict(n=sum(r['layout']==l for r in sub),success=sum(r['success'] for r in sub if r['layout']==l)) for l in sorted({r['layout'] for r in sub})})
    paired={}
    for split in ['iid','transfer']:
        rr=[r for r in rows if r['split']==split];paired[split]={}
        for left,right in [('GDA','GA'),('GA','G'),('DA','G'),('GDA','G')]:
            a={r['case_id']:r for r in rr if r['condition']==left};b={r['case_id']:r for r in rr if r['condition']==right}
            if not a and not b:continue
            if a.keys()!=b.keys():raise ValueError('paired comparison cases differ')
            ids=sorted(a);gained=[k for k in ids if a[k]['success'] and not b[k]['success']];lost=[k for k in ids if b[k]['success'] and not a[k]['success']]
            paired[split][left+'-'+right]=dict(n=len(ids),net_success=len(gained)-len(lost),gained=gained,lost=lost,both_success=sum(a[k]['success'] and b[k]['success'] for k in ids),both_failure=sum(not a[k]['success'] and not b[k]['success'] for k in ids))
    return dict(groups=groups,paired=paired,scope='single training seed; newly frozen IID/transfer geometry plus separately reported seen development roots')


def run():
    S.torch.set_num_threads(1);started=time.monotonic();prepare();S.Campaign(PARENT).verify_frozen()
    c=T.make(COLLECT,52);rows=collect(c,teacher_jobs());S.save(COLLECT/'coverage.json',summarize(rows));admission_checks(rows);S.finish(c,'complete')
    subprocess.run([sys.executable,'-m',__spec__.name,'extract'],check=True)
    # Training-only gate must be explicitly admitted after source extraction.
    subprocess.run([sys.executable,'-m','experiments.planar_tasks.authoring_v2.relational_transfer_trainability','--data',str(DATA),'--output',str(HOST/'trainability')],check=True)
    if not S.read(HOST/'trainability/RESULT.json')['accepted']:raise ValueError('training-only canary failed; no silent hyperparameter/test tuning')
    T.make(FIT,0,fits=4)
    for arm in I.ARMS:subprocess.run([sys.executable,'-m',__spec__.name,'fit','--arm',arm],check=True)
    S.save(HOST/'FIT_FINISHED.json',dict(seal=S.finish(S.Campaign(FIT),'complete')));S.Campaign(FIT).verify_frozen()
    final=S.read(PARENT/'inputs/final.json');cases=sum([S.read(PARENT/'panels'/f'{k}.json')['cases'] for k in ['roots','iid','transfer']],[]);assert len(cases)==80
    conditions=[(arm,final['feature_sid']) for arm in I.ARMS]+[('CAP_'+sid,sid) for sid in dict.fromkeys(final['cap_baseline_sids']+[final['feature_sid']])]
    jobs=[dict(case_id=r['id'],base_id=r['base_id'],layout=r['layout'],relation=r['relation'],split=r['split'],fixture=r['fixture'],fixture_sha256=r['fixture_sha256'],condition=name,sid=sid) for name,sid in conditions for r in cases]
    e=T.make(EVAL,len(jobs));rows=collect(e,jobs);result=summarize(rows);result.update(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows),elapsed_seconds=time.monotonic()-started)
    S.save(EVAL/'comparison.json',result);S.finish(e,'complete');result['seals']={str(p):S.Campaign(p).verify_frozen() for p in [AUTHOR,PARENT,COLLECT,DATA,FIT,EVAL]};S.save(HOST/'RESULT.json',result);S.save(HOST/'FINISHED.json',dict(status='complete',episodes=len(rows)));print('FINISHED',result['groups'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','extract','fit']);p.add_argument('--arm',choices=I.ARMS);a=p.parse_args()
    try:
        if a.command=='run':run()
        elif a.command=='extract':build_data()
        else:fit_one(a.arm)
    except BaseException as exc:S.save(HOST/(a.command+'_FAILURE.json'),dict(error=repr(exc),traceback=traceback.format_exc()));raise
