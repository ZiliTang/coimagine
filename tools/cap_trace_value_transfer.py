"""Read-only inventory/export/verification of complete CAP development runs."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from tools.priority_transfer import ROOT, read, save, sha, inside

CAPS=['CAP-RR-'+x for x in ('009','003','001','010','006','007','002')]
VERSION='cap-trace-value-dev-v1'

def digest(x):
    from experiments.planar_tasks.relational_search.contract import digest as f
    return f(x)

def content_digest(fixture):
    # Scene identity is separate from physical content. Keep entity identities and dynamics.
    return digest({k:v for k,v in fixture.items() if k != 'scene_id'})

def audit(job):
    row=dict(job);p=Path(row.pop('source'));issues=[]
    row['provenance']={'original_path':str(p)}
    row['missing_fields']=[]
    if not p.is_file():return dict(row,completion='missing',issues=['file absent'])
    row['sha256']=sha(p);row['bytes']=p.stat().st_size
    try:
        with gzip.open(p,'rt') as f:d=json.load(f)
        s=d['summary'];h=d['public_rows'];ev=d['events'];dec=d['decisions']
        required=['action_protocol','event_config','source_sha256','history_protocol','fixture_sha256','replay']
        row['missing_fields']=[k for k in required if k not in d]
        if row['missing_fields']:issues.append('required metadata missing')
        if d.get('fixture_sha256')!=row['fixture_sha256']:issues.append('fixture hash mismatch')
        for key in ('success','ticks','decisions','stop','termination'):
            if key in row['summary'] and s.get(key)!=row['summary'][key]:issues.append('summary mismatch: '+key)
        if s.get('error') or s.get('stop') not in ('terminal','decision_budget','tick_budget'):
            issues.append('infrastructure error or unrecognized termination')
        if s.get('stop')=='terminal' and not s.get('termination'):issues.append('terminal reason absent')
        if len(h)!=s['decisions']+1 or len(ev)!=s['decisions'] or len(dec)!=s['decisions']:
            issues.append('boundary/event/decision count mismatch')
        if not h or h[0]['observation']['tick']!=0 or h[0]['action'] is not None:issues.append('initial boundary missing')
        if h[-1]['observation']['tick']!=s['ticks']:issues.append('terminal boundary missing')
        from experiments.planar_tasks.semantic_rl.stream_history import StreamHistory,VERSION as SV
        from experiments.planar_tasks.relational_search.contract import History
        history=StreamHistory() if d.get('history_protocol')==SV else History()
        for i,r in enumerate(h):
            history.append(r['observation'],r['action'])
            if history.sha256!=r['history_hash']:issues.append('history hash mismatch at '+str(i));break
            if 'robots' not in r['observation']:issues.append('robot feedback absent at '+str(i))
            for robot in r['observation'].get('robots',[]):
                for key in ('own','scan','execution','tick','elapsed_seconds'):
                    if key not in robot:row['missing_fields'].append('robot.'+key)
            if i:
                e=ev[i-1]
                if e['start_tick']!=h[i-1]['observation']['tick'] or e['end_tick']!=r['observation']['tick'] or e['periods']!=e['end_tick']-e['start_tick']:
                    issues.append('event time discontinuity at '+str(i))
                if e['executed']!=r['action'] or dec[i-1]['event']!=e:issues.append('executed action/event mismatch at '+str(i))
                if e['end_tick']<=e['start_tick']:issues.append('nonpositive event duration')
        frames=d.get('replay',{}).get('frames',[])
        if not frames or frames[0]['tick']!=0 or frames[-1]['tick']!=s['ticks']:issues.append('replay tail missing')
        if s['stop']=='terminal' and (not ev or ev[-1].get('reason')!='terminal'):issues.append('terminal event missing')
        row.update(boundaries=len(h),decisions=s['decisions'],ticks=s['ticks'],success=s['success'],stop=s['stop'],termination=s.get('termination'),
                   action_protocol=d.get('action_protocol'),event_config=d.get('event_config'),history_protocol=d.get('history_protocol'),
                   semantic_source_sha256=d.get('source_sha256'),recording=d.get('recording'),
                   observation_fields=sorted(h[0]['observation']),robot_feedback_fields=sorted(h[0]['observation']['robots'][0]))
    except Exception as e:issues.append(type(e).__name__+': '+str(e))
    row['missing_fields']=sorted(set(row['missing_fields']))
    if row['missing_fields'] and 'required metadata missing' not in issues:issues.append('required feedback missing')
    row.update(completion='complete' if not issues else 'anomalous',issues=issues)
    return row

def export(out,workers):
    from experiments.planar_tasks.authoring_v2 import random_room_coverage_v2 as V
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);rows=[];fixtures={};source_manifests={}
    author=V.O.AUTHOR;scaling=ROOT/'outputs/semantic_rl/random_room_gda_scaling_20261117_03'
    for p in (author,V.O.PARENT,V.BASE/'collection',scaling/'collection'):
        source_manifests[str(p)]=V.Campaign(p).verify_frozen()
    state=read(author/'campaign.json')
    for c in state['tasks']:fixtures[c]=read(author/'tasks'/c/'fixture.json')
    for m in state['library']:
        for c in state['tasks']:
            p=author/'runs'/m['runs'][c]
            rows.append(dict(case_id=c,cap_id=m['name'],summary=read(p/'summary.json'),source=str(p/'result.json.gz')))
    for b in (V.BASE,scaling):
        for c in read(b/'parent/development.json')['cases']:
            assert c['id'] not in fixtures;fixtures[c['id']]=c['fixture']
        for r in read(b/'collection/result.json')['rows']:
            rows.append(dict(case_id=r['case_id'],cap_id=r['condition'],summary=r,source=r['result_path']))
    assert len(fixtures)==760 and len(rows)==7600
    assert len({(r['case_id'],r['cap_id']) for r in rows})==7600
    # All ten CAP summaries; original path fields are provenance, never lookup paths.
    summaries=[dict(case_id=r['case_id'],cap_id=r['cap_id'],success=r['summary']['success'],ticks=r['summary']['ticks'],
                    provenance={'source_path':r['source'],'original_summary':r['summary']}) for r in rows]
    save(out/'development_summaries.json',summaries)
    save(out/'scenes.json',[dict(case_id=k,fixture=v,fixture_sha256=digest(v)) for k,v in sorted(fixtures.items())])
    jobs=[dict(r,summary={k:v for k,v in r['summary'].items() if k in ('success','ticks','decisions','stop','termination','status','error')},fixture_sha256=digest(fixtures[r['case_id']])) for r in rows if r['cap_id'] in CAPS]
    assert len(jobs)==5320
    inventory=[]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(audit,jobs,chunksize=4)):
            inventory.append(r)
            if (i+1)%200==0:print('AUDIT',i+1,'/5320',flush=True)
    save(out/'inventory.json',inventory)
    counts={c:dict(Counter('anomalous' if r['completion']!='complete' else 'success' if r['success'] else 'failure' for r in inventory if r['cap_id']==c)) for c in CAPS}
    for c in CAPS:
        selected=[r for r in inventory if r['cap_id']==c]
        counts[c].update(anomalous=sum(r['completion']!='complete' for r in selected),decisions=sum(r.get('decisions',0) for r in selected),observation_boundaries=sum(r.get('boundaries',0) for r in selected),missing_fields=sorted({k for r in selected for k in r['missing_fields']}),stop_counts=dict(Counter(r.get('stop','unknown') for r in selected)))
    gaps=[r for r in inventory if r['completion']!='complete'];save(out/'gaps.json',gaps)
    save(out/'audit_summary.json',dict(complete=not gaps,per_cap=counts,records=len(inventory),
         decision_boundaries=sum(r.get('decisions',0) for r in inventory),observation_boundaries=sum(r.get('boundaries',0) for r in inventory)))
    # Never substitute another run. Incomplete exports are inventory deliveries only.
    if gaps:
        print(json.dumps({'complete':False,'gaps':len(gaps),'inventory':str(out)}));return
    (out/'trajectories').mkdir();index=[]
    for r in inventory:
        rel='trajectories/'+r['sha256']+'.json.gz';dest=out/rel
        if not dest.exists():shutil.copyfile(r['provenance']['original_path'],dest)
        if sha(dest)!=r['sha256']:raise ValueError('copy changed')
        index.append(dict(r,path=rel))
    save(out/'index.json',index)
    library=[]
    for e in V.entries():
        if e['name'] not in CAPS:continue
        rel='policies/'+e['name']+'.py';p=out/rel;p.parent.mkdir(exist_ok=True);p.write_text(e['source'])
        library.append(dict(cap_id=e['name'],path=rel,sha256=sha(p),groups=e['groups']))
    save(out/'library.json',library)
    sources={}
    for folder in ('experiments','src/coimagine'):
        for p in (ROOT/folder).rglob('*.py'):
            rel=str(p.relative_to(ROOT));dest=out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources[rel]=sha(p)
    for p in (ROOT/'src/coimagine/random_start/lidar_kernel.c',Path(__file__).resolve(),ROOT/'tools/priority_transfer.py',ROOT/'tools/program_trace_transfer.py'):
        rel=str(p.relative_to(ROOT));dest=out/'sources'/rel;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);sources[rel]=sha(p)
    save(out/'runtime_sources.json',sources)
    identities=[dict(case_id=k,fixture_sha256=digest(v),content_sha256=content_digest(v),role='development760') for k,v in fixtures.items()]
    panels=[V.O.PARENT/'scenes.json',V.PARENT/'development.json',V.PARENT/'fresh_test.json',scaling/'parent/development.json',scaling/'parent/fresh_test.json']
    for p in panels:
        for c in read(p)['cases']:identities.append(dict(case_id=c['id'],fixture_sha256=digest(c['fixture']),content_sha256=content_digest(c['fixture']),role='previously_used',provenance={'panel':str(p)}))
    p=ROOT/'outputs/semantic_rl/da_joint_20261117_01/inputs.json'
    for c in read(p)['fresh']:identities.append(dict(case_id=c['id'],fixture_sha256=digest(c['fixture']),content_sha256=content_digest(c['fixture']),role='viewed_DA80'))
    save(out/'used_scene_identities.json',identities)
    save(out/'protocol.json',dict(version=VERSION,caps=CAPS,development_cases=760,expected_runs=5320,expected_success=1759,expected_failure=3561,
          max_ticks=1800,max_decisions=300,provenance={'source_seals':source_manifests},
          history='initial public row then each executed option boundary; continuous sensor history between boundaries was not stored as public_rows',
          trace='locals/intermediate expressions not saved; replay original CAP on public_rows from initial state to recompute, validate output/memory equivalence before new instrumentation',
          returns='terminal success, executed tick durations and task_cost_seconds stored; reconstruct suffix return only after choosing discount/reward definition. decision_budget is completed truncation, not environmental terminal success',
          privileged='replay frames and scenes are offline evidence; do not feed simulator truth to deployed policy',
          paths='index.path and library.path relative to this bundle; raw gzip preserved byte-for-byte, embedded old paths are provenance only'))
    shutil.copyfile(ROOT/'docs/development/CAP_Trace_Value_Dev_Transfer_20261118.md',out/'README.md')
    save(out/'manifest.json',dict(version=VERSION,files={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file()}))
    print(json.dumps(read(out/'audit_summary.json')),flush=True)

def verify(out,workers):
    out=Path(out).resolve();m=read(out/'manifest.json');assert m['version']==VERSION
    actual={str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p!=out/'manifest.json'}
    assert actual==set(m['files'])
    for rel,h in m['files'].items():assert sha(inside(out,rel))==h,rel
    rows=read(out/'index.json');scenes={r['case_id']:r for r in read(out/'scenes.json')};summaries=read(out/'development_summaries.json')
    assert len(scenes)==760 and len(summaries)==7600
    catalog={(r['case_id'],r['cap_id']):r for r in summaries};assert len(catalog)==7600
    policies={r['cap_id']:digest(inside(out,r['path']).read_text()) for r in read(out/'library.json')}
    assert set(catalog)=={(c,cap) for c in scenes for cap in {r['cap_id'] for r in summaries}} and len({r['cap_id'] for r in summaries})==10
    keys={(r['case_id'],r['cap_id']) for r in rows};assert len(rows)==len(keys)==5320
    assert keys=={(c,cap) for c in scenes for cap in CAPS}
    jobs=[]
    for r in rows:
        ref=catalog[r['case_id'],r['cap_id']]
        assert r['success']==ref['success'] and r['ticks']==ref['ticks'] and r['completion']=='complete'
        assert r['fixture_sha256']==scenes[r['case_id']]['fixture_sha256']==digest(scenes[r['case_id']]['fixture'])
        assert r['semantic_source_sha256']==policies[r['cap_id']],(r['case_id'],r['cap_id'],'program identity')
        p=inside(out,r['path']);assert sha(p)==r['sha256']
        jobs.append(dict(case_id=r['case_id'],cap_id=r['cap_id'],source=str(p),summary=r['summary'],fixture_sha256=r['fixture_sha256']))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i,r in enumerate(pool.map(audit,jobs,chunksize=4)):
            assert r['completion']=='complete',r
            if (i+1)%500==0:print('VERIFY',i+1,flush=True)
    assert sum(r['success'] for r in rows)==1759
    print(json.dumps({'verified':True,'runs':5320,'success':1759,'failure':3561}),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('op',choices=['export','verify','pack']);p.add_argument('--bundle',required=True);p.add_argument('--workers',type=int,default=8);a=p.parse_args()
    if a.op=='export':export(a.bundle,a.workers)
    elif a.op=='verify':verify(a.bundle,a.workers)
    else:
        b=Path(a.bundle).resolve();read(b/'manifest.json');dest=b.with_name(b.name+'.tar.gz')
        if dest.exists():raise FileExistsError(dest)
        with tarfile.open(dest,'w:gz',compresslevel=1) as t:t.add(b,arcname=b.name)
        from tools.program_trace_transfer import audit_archive
        print('ARCHIVE_MEMBERS_VERIFIED',audit_archive(dest,b),flush=True)
        dest.with_name(dest.name+'.sha256').write_text(sha(dest)+'  '+dest.name+'\n');print(dest)
if __name__=='__main__':main()
