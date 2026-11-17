"""Frozen CAP coverage expansion and real-prefix repair; no new CAP or RL.

All runtime policy inputs, controllers and horizons are reused unchanged. New
modules preserve the first random-room campaign's source and evidence seals.
"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse, collections, hashlib, json, multiprocessing as mp, shutil, time, traceback
import numpy as np
import torch
from . import random_room_study as O
from .random_room_tasks import generate
from experiments.planar_tasks.relational_search.pipeline import Campaign, ROOT, read, save, pack, unpack
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.semantic_rl.features import Features
from experiments.planar_tasks.semantic_rl.library_features import LibraryFeatures, GeometryOnlyLibraryFeatures
from experiments.planar_tasks.semantic_rl.library_learning import train
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I
from experiments.planar_tasks.semantic_rl.runner import rollout

BASE=ROOT/'outputs/semantic_rl/random_room_coverage_20261117_02'
HOST=ROOT/'outputs/semantic_rl_authors/random_room_coverage_20261117_02'
PARENT=BASE/'parent'
ARMS=('G','GA','GDA'); TIERS=('control','expanded','repaired'); UPDATES=28000
CACHE={}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def frozen(p):
    if (p/'manifest.json').exists(): Campaign(p).verify_frozen(); return True
    return False
def make(p):
    if (p/'campaign.json').exists():
        c=Campaign(p); c.verify_sources(); return c
    return Campaign.create(p,fixture=read(O.PARENT/'scenes.json')['cases'][0]['fixture'],
        sequence=[dict(target_id=None,a0=44,a1=44,ticks=4)],group='random-room-coverage-v2',
        limits=dict(attempts=8000,resets=8000,execution_ticks=14400000,fits=9,rounds=0,author_turns=0))
def finish(c): return O.finish(c)
def update(**kw): save(HOST/'progress.json',dict(at=time.time(),**kw))

def bounded(jobs,fn,workers=16,stage='work'):
    rows=[]; pending={}; cursor=0; start=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as pool:
        while pending or cursor<len(jobs):
            while len(pending)<workers and cursor<len(jobs):
                j=jobs[cursor];cursor+=1;pending[pool.submit(fn,j)]=j
            done,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:
                pending.pop(f);rows.append(f.result())
                update(stage=stage,completed=len(rows),total=len(jobs),elapsed_seconds=time.monotonic()-start)
                if len(rows)%16==0 or len(rows)==len(jobs):print('PROGRESS',stage,len(rows),'/',len(jobs),flush=True)
            del done
    return rows

def panel_rows(seed,split,count):
    # Same generator and geometric distribution; select equal counts per stratum
    # by index BEFORE outcomes. The old generator's split labels are ignored here.
    source=generate(seed);rows=[]
    for n in range(2,6):
        for j,r in enumerate([r for r in source['cases'] if r['movable_count']==n][:count//4]):
            rows.append(dict(r,id=f'{split}_{n}_{j:03d}',split=split,fixture_sha256=digest(r['fixture'])))
    return rows

def prepare():
    if frozen(PARENT):return
    seals={str(p):Campaign(p).verify_frozen() for p in (O.AUTHOR,O.PARENT,O.DATA,O.FIT,O.EVAL)}
    Campaign(O.PARENT).verify_sources()
    c=make(PARENT);development=panel_rows(2026111741,'newdev',200);test=panel_rows(2026111742,'fresh_test',100)
    old=read(O.PARENT/'scenes.json')['cases'];seen={digest(r['fixture']) for r in old}
    for r in development+test:
        if r['fixture_sha256'] in seen:raise ValueError('duplicate initial fixture')
        seen.add(r['fixture_sha256']);X.fixture(r['fixture'])
    save(PARENT/'development.json',dict(seed=2026111741,cases=development))
    save(PARENT/'fresh_test.json',dict(seed=2026111742,cases=test))
    save(PARENT/'old_diagnostic.json',dict(cases=old,note='previous validation/test are now known diagnostic only'))
    save(PARENT/'protocol.json',dict(input_seals=seals,development=200,fresh_test=100,
        frozen_programs=10,no_new_cap=True,no_rl=True,arms=ARMS,tiers=TIERS,updates_per_fit=UPDATES,
        seed=17,workers=16,fit_workers=3,extract_workers=4,max_ticks=1800,max_decisions=300,
        rollout_wall_seconds=1800,fit_wall_seconds=5400,max_episodes=8000,max_execution_ticks=14400000,
        diagnosis='all old GDA losses within teacher union, two real-prefix cuts, all ten receivers; old results not training labels',
        cuts='first action divergence from fastest successful teacher; divergence+8 boundaries; positive cuts with <=1200 prefix ticks; fallback earliest legal positive boundary',
        repair='expanded GDA on200newdev; at most6 missed teacher-solvable cases per movable-count stratum; both cuts, all10CAP; one fastest successful suffix per cut, exclude prefix labels',
        random_data='original124 plus one fastest complete success per newdev; ties library order; all failures retained in coverage denominator',
        control='old data trained from scratch for same28000 updates; original14000 checkpoint also evaluated on fresh test',
        final='all fixed versions frozen before fresh_test outcomes; no checkpoint or data selection using fresh test',
        failure_labels='failed least-bad rollouts are not positive imitation labels'))
    finish(c);save(HOST/'PREPARED.json',dict(status='complete',new_geometry_checks=300,seals=seals))

def entries(): return O.entries()
def model_path(condition):
    if condition.startswith('original_'):return O.FIT/condition.removeprefix('original_')/'model.pt'
    tier,arm=condition.split('_',1);return BASE/('fit_'+tier)/arm/'model.pt'
def get_runtime(condition):
    if CACHE.get('condition')==condition:return CACHE['f'],CACHE['model']
    if 'f' in CACHE:CACHE['f'].close()
    CACHE.clear();es=entries()
    if condition.startswith('CAP-'):
        e=next(e for e in es if e['name']==condition);source=e['source'];m=None
        f=Features(source,fast_embedding=True,action_protocol=O.protocol(O.CFG),execution_backend='trusted_inprocess',trusted_source_sha256=digest(source))
    else:
        m=I.Scorer.load(model_path(condition))
        f=GeometryOnlyLibraryFeatures(es,O.protocol(O.CFG),m.layout) if m.arm=='G' else LibraryFeatures(es,O.protocol(O.CFG),execution_backend='trusted_inprocess')
    CACHE.update(condition=condition,f=f,model=m);return f,m

def visible(rows): return [dict(action=r['action'],observation=r['observation']) for r in rows]
def prefix_audit(result,reference,cut):
    if result.get('prefix_replay',{}).get('executed')!=cut:raise ValueError('prefix not completely executed')
    if digest(visible(result['public_rows'][:cut+1]))!=digest(visible(reference['public_rows'][:cut+1])):raise ValueError('handoff public prefix changed')
    tick=result['public_rows'][cut]['observation']['tick']
    a=[f for f in result['replay']['frames'] if f['tick']<=tick];b=[f for f in reference['replay']['frames'] if f['tick']<=tick]
    if digest(a)!=digest(b):raise ValueError('handoff physics prefix changed')
    return dict(public_equal=True,physics_equal=True,cut=cut,ticks=tick)

def run_worker(job):
    torch.set_num_threads(1);p=Path(job['path']);p.mkdir(parents=True,exist_ok=True)
    if (p/'summary.json').exists():return read(p/'summary.json')
    f,m=get_runtime(job['condition']);prefix=[];ref=None
    if job.get('cut',0):
        if sha(job['reference'])!=job['reference_sha256']:raise ValueError('changed prefix source')
        ref=unpack(job['reference']);prefix=[r['action'] for r in ref['public_rows'][1:job['cut']+1]]
    save(p/'started.json',dict(at=time.time(),condition=job['condition'],case_id=job['case_id']))
    result,_=rollout(job['fixture'],f.source,features=f,model=m,config=O.CFG,max_ticks=1800,max_decisions=300,
        wall_seconds=1800,deterministic=True,seed=17,fast_feedback=True,prefix=prefix)
    audit=prefix_audit(result,ref,job['cut']) if ref is not None else None
    if m is not None and m.arm=='G':
        result['recommendation_audit']='not measured; G skips program execution'
        for k in ('recommended','duration_only_deviation','motion_deviation'):result['summary']['choice_counts'][k]=None
    if ref is not None:result['handoff']=dict(reference=job['reference'],reference_sha256=job['reference_sha256'],audit=audit,supervise_start_boundary=job['cut'])
    start=time.monotonic();pack(p/'result.json.gz',result,compresslevel=1);storage=time.monotonic()-start
    row=dict(condition=job['condition'],case_id=job['case_id'],split=job['split'],movable_count=job['movable_count'],
        **result['summary'],cut=job.get('cut',0),prefix_audit=audit,storage_seconds=storage,result_path=str(p/'result.json.gz'))
    if result['decisions'] and job.get('cut',0)<len(result['decisions']):row['first_suffix_action']=result['decisions'][job.get('cut',0)]['action']
    save(p/'summary.json',row);return row

def job_for(case,condition,out,**extra):
    key=case['id']+(f"__cut{extra['cut']:03d}" if extra.get('cut') else '')
    return dict(condition=condition,case_id=case['id'],split=case['split'],movable_count=case['movable_count'],
        fixture=case['fixture'],path=str(out/'runs'/condition/key),**extra)
def physical_phase(path,jobs,stage):
    if frozen(path):return read(path/'result.json')['rows']
    c=make(path);save(path/'jobs.json',jobs);rows=bounded(jobs,run_worker,stage=stage)
    save(path/'result.json',dict(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows)))
    with c.locked():
        c.state['used'].update(attempts=len(rows),resets=len(rows),execution_ticks=sum(r['ticks'] for r in rows));c.state['runs']=[{k:r.get(k) for k in ('condition','case_id','cut','success','ticks','stop','error','result_path')} for r in rows];c._save()
    # Policy-invalid recommendations remain failures. Runtime/OS errors need review.
    infra=[r for r in rows if r['stop']=='wall_budget' or r.get('error') and r['error'].get('message')!='CAP first recommendation missing/illegal']
    save(path/'execution_audit.json',dict(infrastructure_issues=infra,invalid_cap=sum(bool(r.get('error')) for r in rows),prefix_checks=sum(r['prefix_audit'] is not None for r in rows)))
    if infra:raise RuntimeError(f'{stage}: {len(infra)} infrastructure issues; artifacts retained')
    finish(c);return rows

def choose_cuts(learner,teacher):
    la=[r['action'] for r in learner['public_rows'][1:]];ta=[r['action'] for r in teacher['public_rows'][1:]]
    d=next((i for i,(a,b) in enumerate(zip(la,ta)) if a!=b),min(len(la),len(ta)))
    valid=[i for i in range(1,len(la)) if learner['public_rows'][i]['observation']['tick']<=1200]
    cuts=sorted({i for i in (d,d+8) if i in valid})
    if not cuts and valid:cuts=[valid[min(7,len(valid)-1)]]
    return dict(first_divergence=d,cuts=cuts)

def fastest(rows):
    wins=[r for r in rows if r['success']]
    return min(wins,key=lambda r:(r['ticks'],r['condition'])) if wins else None

def handoff_jobs(cases,learned,teachers,out,max_per_stratum=None):
    selected=[];counts=collections.Counter();jobs=[];names=[e['name'] for e in entries()]
    for case in sorted(cases,key=lambda r:r['id']):
        learner=next(r for r in learned if r['case_id']==case['id']);best=fastest([r for r in teachers if r['case_id']==case['id']])
        if learner['success'] or best is None:continue
        n=case['movable_count']
        if max_per_stratum is not None and counts[n]>=max_per_stratum:continue
        lr=unpack(learner['result_path']);tr=unpack(best['result_path']);choice=choose_cuts(lr,tr)
        if not choice['cuts']:continue
        counts[n]+=1;selected.append(dict(case_id=case['id'],teacher=best['condition'],learner_path=learner['result_path'],**choice))
        for cut in choice['cuts']:
            jobs.extend(job_for(case,name,out,cut=cut,reference=learner['result_path'],reference_sha256=sha(learner['result_path'])) for name in names)
    return jobs,selected

def diagnose():
    out=BASE/'diagnosis'
    if frozen(out):return
    r=read(O.HOST/'RESULT.json');cases=[c for c in read(PARENT/'old_diagnostic.json')['cases'] if c['split']!='dev']
    learned=[x for x in r['rows'] if x['condition']=='GDA'];teachers=[x for x in r['rows'] if x['condition'].startswith('CAP-')]
    jobs,selected=handoff_jobs(cases,learned,teachers,out);make(out);save(out/'selection.json',selected)
    physical_phase(out,jobs,'diagnostic_handoffs')

def collect():
    out=BASE/'collection';cases=read(PARENT/'development.json')['cases']
    physical_phase(out,[job_for(c,e['name'],out) for e in entries() for c in cases],'random_development_CAPs')

def extraction_worker(job):
    torch.set_num_threads(1)
    if 'extract_f' not in CACHE:CACHE['extract_f']=LibraryFeatures(entries(),O.protocol(O.CFG),execution_backend='trusted_inprocess')
    f=CACHE['extract_f'];p=Path(job['path']);p.mkdir(parents=True,exist_ok=True)
    if (p/'complete.json').exists():return read(p/'complete.json')
    if sha(job['result_path'])!=job['sha256']:raise ValueError('demonstration changed')
    result=unpack(job['result_path'])
    if not result['summary']['success']:raise ValueError('failed trajectory used as positive labels')
    contexts=[];xs=[];offsets=[];targets=[];meta=[];cursor=0;started=time.monotonic();before=f.calls
    for tr in O.extracted_rows(result,f):
        if tr['boundary']<job.get('start_boundary',0):continue
        panels=tr['panels'];ctx=np.asarray(panels[0]['attribution_context'],dtype=np.float32);contexts.append(ctx);off=[cursor]
        for panel in panels:
            np.testing.assert_array_equal(panel['attribution_context'].astype(np.float32),ctx)
            x=panel['attribution_candidates'].astype(np.float32);xs.append(x);cursor+=len(x);off.append(cursor)
        offsets.append(off);targets.append(tr['indices']);meta.append(dict(boundary=tr['boundary'],history_hash=tr['history_hash']))
    if not contexts:raise ValueError('empty suffix supervision')
    for k,a in dict(contexts=np.array(contexts),candidates=np.concatenate(xs),offsets=np.array(offsets),targets=np.array(targets)).items():np.save(p/(k+'.npy'),a)
    save(p/'metadata.json',meta);save(p/'layout.json',f.layout)
    row=dict(task=job['task'],path=str(p),teacher=job['teacher'],result_path=job['result_path'],result_sha256=job['sha256'],start_boundary=job.get('start_boundary',0),boundaries=len(contexts),candidates=cursor,source_sha=f.sha256,action_protocol=f.action_protocol,context_width=len(contexts[0]),candidate_width=xs[0].shape[1],feature_calls=f.calls-before,wall_seconds=time.monotonic()-started)
    save(p/'complete.json',row);return row

def combine(path,rows):
    n=sum(r['boundaries'] for r in rows);m=sum(r['candidates'] for r in rows)
    arrays={k:np.lib.format.open_memmap(path/(k+'.npy'),mode='w+',dtype=t,shape=s) for k,t,s in [('contexts','float32',(n,rows[0]['context_width'])),('candidates','float32',(m,rows[0]['candidate_width'])),('offsets','int64',(n,5)),('targets','int64',(n,4))]}
    layout=read(Path(rows[0]['path'])/'layout.json');start=cursor=0;metadata=[]
    for r in rows:
        p=Path(r['path']);assert read(p/'layout.json')==layout
        end=start+r['boundaries'];stop=cursor+r['candidates']
        for k in ('contexts','targets'):arrays[k][start:end]=np.load(p/(k+'.npy'),mmap_mode='r')
        arrays['candidates'][cursor:stop]=np.load(p/'candidates.npy',mmap_mode='r');arrays['offsets'][start:end]=np.load(p/'offsets.npy')+cursor
        metadata.extend(dict(v,task=r['task']) for v in read(p/'metadata.json'));start=end;cursor=stop
    for a in arrays.values():a.flush()
    save(path/'layout.json',layout);save(path/'metadata.json',metadata)
    save(path/'extraction.json',dict(demonstrations=len(rows),boundaries=n,source_sha=rows[0]['source_sha'],action_protocol=O.protocol(O.CFG),rows=rows))

def build_data(tier):
    path=BASE/('data_'+tier)
    if frozen(path):return
    c=make(path);jobs=[]
    if tier=='expanded':
        parent=O.DATA;raw=read(BASE/'collection/result.json')['rows'];groups={r['case_id'] for r in raw}
        for case in sorted(groups):
            best=fastest([r for r in raw if r['case_id']==case])
            if best:jobs.append(dict(task=case,teacher=best['condition'],result_path=best['result_path'],sha256=sha(best['result_path']),path=str(path/'shards'/case)))
    else:
        parent=BASE/'data_expanded';raw=read(BASE/'repair/result.json')['rows'];groups={(r['case_id'],r['cut']) for r in raw}
        for case,cut in sorted(groups):
            best=fastest([r for r in raw if r['case_id']==case and r['cut']==cut])
            if best:jobs.append(dict(task=f'{case}__repair{cut:03d}',teacher=best['condition'],result_path=best['result_path'],sha256=sha(best['result_path']),start_boundary=cut,path=str(path/'shards'/f'{case}_{cut:03d}')))
    save(path/'selected.json',jobs);Campaign(parent).verify_frozen()
    new=bounded(jobs,extraction_worker,workers=4,stage='extract_'+tier) if jobs else []
    new.sort(key=lambda r:r['task']);rows=read(parent/'extraction.json')['rows']+new;combine(path,rows)
    save(path/'data_lineage.json',dict(parent=str(parent),parent_seal=sha(parent/'manifest.json'),new_demonstrations=len(new),new_boundaries=sum(r['boundaries'] for r in new),no_failed_positive_labels=True))
    finish(c)

def data_path(tier):return O.DATA if tier=='control' else BASE/('data_'+tier)
def fit_worker(job):
    tier=job['tier'];arm=job['arm'];p=BASE/('fit_'+tier)/arm;p.mkdir(parents=True,exist_ok=True)
    if (p/'complete.json').exists():return read(p/'complete.json')
    dp=data_path(tier);ds={k:np.load(dp/(k+'.npy'),mmap_mode='r') for k in ('contexts','candidates','offsets','targets')};ex=read(dp/'extraction.json');layout=read(dp/'layout.json');ids=list(range(len(ds['contexts'])));live=[]
    def progress(r):
        live.append(r);save(p/'curve_live.json',live);save(p/'progress.json',r)
        if r['update']%4000==0:print('FIT',tier,arm,r,flush=True)
    m,curve=train(ds,ids,layout,arm,ex['source_sha'],ex['action_protocol'],updates=UPDATES,wall_seconds=5400,progress=progress)
    acc=I.accuracy(m,ds,ids);save(p/'curve.json',curve);save(p/'accuracy.json',acc)
    m.save(p/'model.pt',dict(tier=tier,updates=UPDATES,seed=17,boundaries=len(ids),demonstrations=ex['demonstrations']))
    reloaded=I.Scorer.load(p/'model.pt');assert all(torch.equal(v,reloaded.state_dict()[k]) for k,v in m.state_dict().items())
    result=dict(tier=tier,arm=arm,updates=UPDATES,accuracy=acc);save(p/'complete.json',result);print('FIT_DONE',tier,arm,acc['exact_actions'],'/',len(ids),flush=True);return result
def fit(tier):
    path=BASE/('fit_'+tier)
    if frozen(path):return
    Campaign(data_path(tier)).verify_frozen();c=make(path)
    bounded([dict(tier=tier,arm=a) for a in ARMS],fit_worker,workers=3,stage='fit_'+tier)
    with c.locked():c.state['used']['fits']=3;c._save()
    save(path/'data_reference.json',dict(path=str(data_path(tier)),seal=sha(data_path(tier)/'manifest.json')));finish(c)

def repair_collect():
    cases=read(PARENT/'development.json')['cases'];probe=BASE/'expanded_training_rollouts'
    learned=physical_phase(probe,[job_for(c,'expanded_GDA',probe) for c in cases],'expanded_GDA_newdev')
    out=BASE/'repair'
    if frozen(out):return
    teachers=read(BASE/'collection/result.json')['rows'];jobs,selection=handoff_jobs(cases,learned,teachers,out,max_per_stratum=6)
    make(out);save(out/'selection.json',selection);physical_phase(out,jobs,'repair_handoffs')

def evaluate():
    out=BASE/'evaluation'
    if frozen(out):return
    for tier in TIERS:Campaign(BASE/('fit_'+tier)).verify_frozen()
    cases=read(PARENT/'old_diagnostic.json')['cases'];fresh=read(PARENT/'fresh_test.json')['cases'];jobs=[]
    # New test is first executed only here, after all decisions about repair/data/fit.
    for tier in TIERS:
        for arm in ARMS:jobs.extend(job_for(c,tier+'_'+arm,out) for c in cases+fresh)
    for condition in [e['name'] for e in entries()]+['original_'+a for a in ARMS]:jobs.extend(job_for(c,condition,out) for c in fresh)
    physical_phase(out,jobs,'fixed_final_evaluation')

def summarize():
    ev=read(BASE/'evaluation/result.json')['rows'];old=read(O.HOST/'RESULT.json')['rows'];rows=list(ev)
    for r in old:
        name=r['condition'] if r['condition'].startswith('CAP-') else 'original_'+r['condition']
        rows.append(dict(r,condition=name,imported=True))
    stats={};names=sorted({r['condition'] for r in rows});paired={}
    for split in ('dev','validation','test','fresh_test'):
        sub=[r for r in rows if r['split']==split];union={r['case_id'] for r in sub if r['condition'].startswith('CAP-') and r['success']};stats[split]={'teacher_union':len(union)}
        for name in names:
            rr=[r for r in sub if r['condition']==name];wins={r['case_id'] for r in rr if r['success']}
            stats[split][name]=dict(n=len(rr),success=len(wins),retained=len(wins&union),lost=sorted(union-wins),outside=sorted(wins-union))
        for tier in ('expanded','repaired'):
            for arm in ARMS:
                left=tier+'_'+arm;right=('control' if tier=='expanded' else 'expanded')+'_'+arm
                a={r['case_id'] for r in sub if r['condition']==left and r['success']};b={r['case_id'] for r in sub if r['condition']==right and r['success']}
                paired[split+':'+left+' vs '+right]=dict(gained=sorted(a-b),lost=sorted(b-a))
    phases=[p for p in BASE.iterdir() if (p/'manifest.json').exists()];seals={str(p):Campaign(p).verify_frozen() for p in phases}
    executions=[r for phase in ('diagnosis','collection','expanded_training_rollouts','repair','evaluation') for r in read(BASE/phase/'result.json')['rows']]
    result=dict(status='complete',summary=stats,paired=paired,rows=rows,seals=seals,new_episodes=len(executions),new_ticks=sum(r['ticks'] for r in executions),fits=9,no_new_cap=True,no_rl=True,
        datasets={t:read(data_path(t)/'extraction.json')['demonstrations'] for t in TIERS},
        boundaries={t:read(data_path(t)/'extraction.json')['boundaries'] for t in TIERS},
        fit_accuracy={t+'_'+a:read(BASE/('fit_'+t)/a/'complete.json')['accuracy'] for t in TIERS for a in ARMS})
    if result['new_episodes']>8000 or result['new_ticks']>14400000:raise ValueError('declared total ceiling exceeded')
    save(HOST/'RESULT.json',result);report(result)
    save(HOST/'FINISHED.json',dict(status='complete',episodes=result['new_episodes'],ticks=result['new_ticks'],sealed_phases=len(seals)))

def report(r):
    lines=['# 冻结 CAP 库：随机覆盖扩充与接管后缀监督','',
        '10个CAP保持不变，200个新开发初态与100个新最终留出独立生成并预先冻结。旧40个留出仅为已知诊断面板。没有新CAP或PPO。',
        '',f"本轮实际{r['new_episodes']}个物理执行、{r['new_ticks']} ticks，9次拟合；每模型28000更新、seed17。前缀重放成本计入，失败不作正示范。",'',
        '| 数据版本 | 示范数 | 决策边界 |','|---|---:|---:|']
    for t in TIERS:lines.append(f"| {t} | {r['datasets'][t]} | {r['boundaries'][t]} |")
    lines+=['','control为原124示范的等28000更新对照；expanded加随机开发成功；repaired再加真实接管成功后缀，坏前缀不作为动作标签。original为上一轮14000更新权重。', '',
        '| 方法 | 原开发/160 | 旧验证/20（诊断） | 旧测试/20（诊断） | 新最终留出/100 |','|---|---:|---:|---:|---:|']
    for name in sorted(k for k in r['summary']['fresh_test'] if k!='teacher_union'):
        lines.append('| '+name+' | '+' | '.join(str(r['summary'][s][name]['success']) for s in ('dev','validation','test','fresh_test'))+' |')
    lines+=['| 教师成功并集（离线参考） | '+' | '.join(str(r['summary'][s]['teacher_union']) for s in ('dev','validation','test','fresh_test'))+' |','',
        '## 新留出的能力保留','', '| 方法 | 保留教师成功 | 丢失 | 教师并集外新增 |','|---|---:|---:|---:|']
    for t in TIERS:
        for a in ARMS:
            n=t+'_'+a;s=r['summary']['fresh_test'][n];lines.append(f"| {n} | {s['retained']} | {len(s['lost'])} | {len(s['outside'])} |")
    lines+=['','## 训练动作拟合','', '| 方法 | 完整动作匹配 |','|---|---:|']
    for n,s in r['fit_accuracy'].items():lines.append(f"| {n} | {s['exact_actions']}/{s['boundaries']} |")
    lines+=['','## 解释边界','',
        '- 原有CAP未更改，新留出在所有模型冻结后才执行；按固定最终权重比较，没有择优挑checkpoint。',
        '- 失败接管不作为正标签；失败保持全分母。可接管成功只证明后续策略可修复，不证明首个分歧动作单独造成成败。',
        '- 几何有效不保证动态可解；教师并集是事后覆盖参考，不是可部署最优策略。',
        '- 同更新预算下不同数据量对应不同epoch；必须结合动作拟合看，不能把欠拟合解释为表示必然不足。',
        '- 单seed；GA/GDA在线仍运行CAP程序，包含程序记忆/搜索知识，不能把所有增量归为描述语义或廉价网络蒸馏。',
        '',f"机器结果：`{HOST.relative_to(ROOT)}/RESULT.json`。每阶段result.json与完整gzip回放、前缀公开观测/物理一致性检查均保留。",'']
    p=ROOT/'docs/agent_exchange/reviews/Random_Room_Coverage_Repair_20261117.md';p.write_text('\n'.join(lines))
    current=ROOT/'docs/agent_exchange/CURRENT.md';s=current.read_text();i=s.index('\n')+1
    note='\n2026-11-17 `random_room_coverage_repair_complete`：[冻结库扩数据与接管整合完成](reviews/Random_Room_Coverage_Repair_20261117.md)。200新开发、100新最终留出；9模型、所有执行与封印完成。详细数字见报告和新宿主RESULT；无新CAP/PPO。\n'
    current.write_text(s[:i]+note+s[i:])

def run_all():
    HOST.mkdir(parents=True,exist_ok=True)
    phases=[('prepare',prepare),('diagnose',diagnose),('collect',collect),('data_expanded',lambda:build_data('expanded')),
        ('fit_control',lambda:fit('control')),('fit_expanded',lambda:fit('expanded')),('repair_collect',repair_collect),
        ('data_repaired',lambda:build_data('repaired')),('fit_repaired',lambda:fit('repaired')),('evaluate',evaluate),('summarize',summarize)]
    for name,fn in phases:
        update(stage=name,status='running');print('PHASE',name,flush=True);fn()
    update(stage='complete',status='complete')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['run','prepare']);a=p.parse_args();HOST.mkdir(parents=True,exist_ok=True)
    try:run_all() if a.command=='run' else prepare()
    except BaseException:
        save(HOST/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));update(stage='failed',status='failed');raise
