"""Frozen CAP coverage on the existing supervised/parameter-RL evaluation panel."""
from . import bounded_rl_study as S
from . import handoff_entry_study as H
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import argparse,csv,hashlib,importlib.util,multiprocessing as mp,shutil,subprocess,time,traceback
from collections import Counter

B=S.BASE
OUT=B/'cap_coverage_comparison_20261116_01'
HOST=Path('outputs/semantic_rl_authors/cap_coverage_comparison_20261116_01')
PARENT=B/'random_parameter_direction_long_parent_20261116_01'
REFERENCE=B/'random_parameter_direction_long_evaluation_20261116_01'
TEACHERS=B/'local_cap_loop_20261116_01'
IDS={'base':'CAP-001.v4','repair':'CAP-002.v2','supervised':'SUP-002.v1','long':'RL-004.v2'}
WORKERS=4
CACHE={}
RUNNER=None

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare():
    registry={r['id']:r for r in S.read('docs/research/Policy_Registry.json')['entries']}
    seals={str(p):S.Campaign(p).verify_frozen() for p in [PARENT,REFERENCE,TEACHERS]}
    snapshot=S.read(REFERENCE/'sources.json')
    changed=[p for p,h in snapshot.items() if not Path(p).exists() or sha(p)!=h]
    runner='experiments/planar_tasks/semantic_rl/runner.py'
    assert set(changed)<={runner},changed
    cases=S.read(PARENT/'panels/anchors.json')['cases']+S.read(PARENT/'panels/test.json')['cases']
    assert len(cases)==61 and len({c['id'] for c in cases})==61
    refs=[r for r in S.read(REFERENCE/'comparison.json') if r['condition'] in ('supervised','long')]
    assert len(refs)==122
    for c in cases:
        assert S.digest(c['fixture'])==c['fixture_sha256']
        for name in ('supervised','long'):
            rs=[r for r in refs if r['case_id']==c['id'] and r['condition']==name]
            assert len(rs)==1 and rs[0]['fixture_sha256']==c['fixture_sha256']
            assert rs[0]['error'] is None and rs[0]['stop']!='wall_budget'
    for rid in IDS.values():assert sha(registry[rid]['artifact'])==registry[rid]['file_sha256']
    c=S.Campaign.create(OUT,limits=dict(attempts=122,resets=122,execution_ticks=219600,rounds=0,fits=0,author_turns=0),model=dict(provider='none',name='frozen CAP coverage comparison'))
    (OUT/'inputs').mkdir()
    for name in ('base','repair'):shutil.copyfile(registry[IDS[name]]['artifact'],OUT/'inputs'/f'{name}.py')
    shutil.copyfile(REFERENCE/'sources'/runner,OUT/'inputs/frozen_runner.py')
    S.save(OUT/'panels.json',dict(cases=cases));S.save(OUT/'reference_rows.json',refs)
    protocol=dict(identities=IDS,registry_entries={k:registry[v] for k,v in IDS.items()},input_seals=seals,
        panel=str(PARENT),reference=str(REFERENCE),new_episodes=122,max_ticks=1800,max_decisions=300,episode_wall_seconds=180,
        workers=WORKERS,admission_seconds=1200,execution_mode='duration',failure_cost_seconds=180,
        runner_source=str(REFERENCE/'sources'/runner),runner_sha256=sha(OUT/'inputs/frozen_runner.py'),live_source_differences=changed,
        selection='fixed registered CAPs and historical final SUP/RL; no selection or training',
        panel_status='anchors are supervised demonstrations; test is previously viewed frozen panel, not a fresh blind test',
        details='all new episodes; save time reported',union='offline teacher success union; not a deployment selector or general optimal bound')
    S.save(OUT/'protocol.json',protocol)
    with c.locked():c.state.update(profile='frozen-cap-coverage-v1',phase='ready',runs=[],protocol=protocol);c._save()
    S.save(HOST/'PREPARED.json',dict(protocol=protocol,git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty_status=subprocess.check_output(['git','status','--short'],text=True).splitlines()))
    print('PREPARED 122 new CAP episodes; 122 frozen SUP/RL results reused',flush=True)

def worker(job):
    global RUNNER
    S.torch.set_num_threads(1)
    if RUNNER is None:
        spec=importlib.util.spec_from_file_location('experiments.planar_tasks.semantic_rl._coverage_frozen_runner',OUT/'inputs/frozen_runner.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);RUNNER=module.rollout
    name=job['condition'];source=(OUT/'inputs'/f'{name}.py').read_text()
    if name not in CACHE:CACHE[name]=S.Features(source,fast_embedding=True,action_protocol=H.protocol(S.EventConfig(mode='duration')),execution_backend='trusted_inprocess',trusted_source_sha256=S.digest(source))
    r,_=RUNNER(job['fixture'],source,features=CACHE[name],config=S.EventConfig(mode='duration'),max_ticks=1800,max_decisions=300,wall_seconds=180,seed=job['seed'],deterministic=True,fast_feedback=True,training_fastpath=True,record_details=True)
    return r

def aggregate(cases,new,reference):
    index={(r['case_id'],r['condition']):r for r in new+reference}
    assert len(index)==244
    paired=[]
    for case in cases:
        cid=case['id'];rr={k:index[cid,k] for k in IDS}
        assert all(r['fixture_sha256']==case['fixture_sha256'] for r in rr.values())
        successes={k:bool(r['success']) for k,r in rr.items()}
        paired.append(dict(case_id=cid,family=case['family'],split=case['split'],level=case['level'],fixture_sha256=case['fixture_sha256'],success=successes,
            teacher_union=successes['base'] or successes['repair'],teacher_intersection=successes['base'] and successes['repair'],
            costs={k:r['task_cost_seconds'] if r['success'] else 180 for k,r in rr.items()},
            runs={k:dict(campaign=str(OUT if k in ('base','repair') else REFERENCE),run=r['id'],identity=IDS[k],stop=r['stop'],ticks=r['ticks']) for k,r in rr.items()}))
    groups={}
    for group in ['anchor','test','all','open_transport','wall_release','clear_transport','small','medium']:
        rows=[r for r in paired if group=='all' or (group in ('anchor','test') and r['split']==group) or (group not in ('anchor','test','all') and r['split']=='test' and group in (r['family'],r['level']))]
        sets={k:{r['case_id'] for r in rows if r['success'][k]} for k in IDS};un=sets['base']|sets['repair'];inter=sets['base']&sets['repair'];allids={r['case_id'] for r in rows}
        coverage=dict(both=sorted(inter),base_only=sorted(sets['base']-sets['repair']),repair_only=sorted(sets['repair']-sets['base']),neither=sorted(allids-un))
        assert sum(len(v) for v in coverage.values())==len(rows)
        metrics={k:dict(n=len(rows),success=len(sets[k]),mean_cost_seconds=sum(r['costs'][k] for r in rows)/len(rows)) for k in IDS}
        learned={k:dict(retained_union=sorted(sets[k]&un),lost_union=sorted(un-sets[k]),gained_outside_union=sorted(sets[k]-un),
            versus={t:dict(gained=sorted(sets[k]-sets[t]),lost=sorted(sets[t]-sets[k])) for t in IDS if t!=k}) for k in ('supervised','long')}
        groups[group]=dict(n=len(rows),conditions=metrics,teacher_union=len(un),coverage=coverage,learned=learned,patterns=dict(Counter(''.join('1' if r['success'][k] else '0' for k in IDS) for r in rows)))
    return dict(identity_order=list(IDS),identities=IDS,groups=groups,paired=paired)

def historical_anchor_audit(rows):
    history=S.read(TEACHERS/'campaign.json')['runs'];audits=[]
    for row in rows:
        if row['split']!='anchor':continue
        task=row['case_id'].removeprefix('root_');sid={'base':'s000','repair':'s002'}[row['condition']]
        refs=[r for r in history if r['task']==task and r['sid']==sid]
        if not refs:continue
        assert len(refs)==1
        ref=refs[0];newdata=S.unpack(OUT/'runs'/row['id']/'result.json.gz');olddata=S.unpack(TEACHERS/'runs'/ref['id']/'result.json.gz')
        def visible(r):return [dict(action=x['action'],observation=x['observation']) for x in r['public_rows']]
        a=dict(case_id=row['case_id'],condition=row['condition'],reference_run=ref['id'],success_equal=row['success']==ref['success'],ticks_equal=row['ticks']==ref['ticks'],public_equal=H.serialized_equal(visible(newdata),visible(olddata)),physics_equal=H.serialized_equal(newdata['replay']['frames'],olddata['replay']['frames']))
        audits.append(a)
    return audits

def run():
    c=S.Campaign(OUT);c.verify_sources();p=S.read(OUT/'protocol.json')
    for path,expected in p['input_seals'].items():assert S.Campaign(path).verify_frozen()==expected
    cases=S.read(OUT/'panels.json')['cases'];jobs=[]
    for name in ('base','repair'):
        for i,case in enumerate(cases):jobs.append(dict(case,condition=name,seed=2026111607+i))
    io=S.SaveTiming();start=time.monotonic();done=[];cursor=0
    with c.locked():c.state['phase']='running';c._save()
    try:
        with ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn')) as pool:
            pending={}
            while pending or cursor<len(jobs):
                while len(pending)<WORKERS and cursor<len(jobs) and time.monotonic()-start<p['admission_seconds']:
                    job=jobs[cursor];rid=f'a{cursor:04d}';req=dict(id=rid,case_id=job['id'],family=job['family'],split=job['split'],level=job['level'],fixture_sha256=job['fixture_sha256'],condition=job['condition'],identity=IDS[job['condition']],seed=job['seed'],record_details=True,status='running')
                    with c.locked():c._charge(attempts=1,resets=1,execution_ticks=1800);c.state['reservations'][rid]=dict(ticks=1800);c.state['runs'].append(req);c._save()
                    io.call(S.save,OUT/'runs'/rid/'request.json',req);pending[pool.submit(worker,job)]=(req,cursor);cursor+=1
                if not pending:break
                finished,_=wait(pending,timeout=1,return_when=FIRST_COMPLETED)
                for future in finished:
                    req,i=pending.pop(future);rid=req['id']
                    try:
                        r=future.result();assert r['fixture_sha256']==req['fixture_sha256']
                        io.call(S.pack,OUT/'runs'/rid/'result.json.gz',r,compresslevel=1)
                        row=dict(r['summary'],**{k:v for k,v in req.items() if k!='status'})
                        actual_ticks=row['ticks']
                    except Exception as exc:
                        row=dict(req,status='tool_error',success=False,error=repr(exc),stop='worker_error',ticks=1800,task_cost_seconds=180,accounting='reserved max ticks retained; actual unknown')
                        actual_ticks=1800;io.call(S.save,OUT/'runs'/rid/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()))
                    io.call(S.save,OUT/'runs'/rid/'summary.json',row)
                    with c.locked():c.state['used']['execution_ticks']-=1800-actual_ticks;del c.state['reservations'][rid];c.state['runs'][i]=row;c._save()
                    done.append(row)
                    print('RESULT',len(done),'/122',rid,row['condition'],row['case_id'],row['success'],row['ticks'],row['stop'],flush=True)
                    if len(done)%16==0:io.log('progress',completed=len(done),elapsed_seconds=round(time.monotonic()-start,2))
        if len(done)!=122:
            S.save(HOST/'FINISHED.json',dict(status='incomplete_admission_budget',episodes=len(done),seal=S.finish(c,'incomplete_admission_budget')));return
        errors=[r['id'] for r in done if r.get('error') or r['stop']=='wall_budget']
        result=aggregate(cases,done,S.read(OUT/'reference_rows.json'))
        audits=historical_anchor_audit(done) if not errors else []
        mismatch=[a for a in audits if not all(a[k] for k in ('success_equal','ticks_equal','public_equal','physics_equal'))]
        S.save(OUT/'historical_anchor_audit.json',audits);S.save(OUT/'comparison.json',result)
        with (OUT/'comparison.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['case_id','split','family','level',*IDS.values(),'teacher_union'])
            for r in result['paired']:w.writerow([r['case_id'],r['split'],r['family'],r['level'],*[int(r['success'][k]) for k in IDS],int(r['teacher_union'])])
        input_seals={path:S.Campaign(path).verify_frozen() for path in p['input_seals']}
        assert input_seals==p['input_seals']
        report=dict(episodes=len(done),ticks=sum(r['ticks'] for r in done),errors=errors,historical_matches=len(audits)-len(mismatch),historical_mismatches=mismatch,elapsed_seconds=time.monotonic()-start,save_seconds=io.seconds,input_seals=input_seals,groups=result['groups'])
        S.save(OUT/'result.json',report);status='complete' if not errors and not mismatch else 'complete_with_audit_failures'
        S.save(HOST/'RESULT.json',report);S.save(HOST/'FINISHED.json',dict(status=status,seal=S.finish(c,status),episodes=len(done),errors=errors,historical_mismatches=mismatch));io.log('finished')
        print('FINAL',status,{k:{n:v['success'] for n,v in result['groups'][k]['conditions'].items()} for k in ('anchor','test')},flush=True)
    except BaseException as exc:
        with c.locked():c.state.update(phase='failed',completion='incomplete_error');c._save()
        S.save(HOST/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc()));raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','run']);args=parser.parse_args();HOST.mkdir(parents=True,exist_ok=True)
    prepare() if args.command=='prepare' else run()
