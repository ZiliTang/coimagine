"""Host-only collection, model freezes, paired tests and full-task evaluation."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from tools.cap_symbolic_general import OUT, AUTHOR, ROOT, read, save, immutable, sha, worker
from experiments.planar_tasks.authoring_v2.symbolic_general import simulation, frontend, scaffold
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel, outcome_valid
from tools.cap_symbolic_general_runtime import Controller


def state(stage, **fields):
    save(OUT/'progress.json', dict(stage=stage, updated=time.time(), **fields))


def bootstrap(workers=4):
    panels=read(OUT/'panels.json')['groups'];source=(AUTHOR/'starter.py').read_text()
    immutable(OUT/'bootstrap_source.json',dict(sha256=hashlib.sha256(source.encode()).hexdigest(),
              note='fixed host starter generates development prefixes; no confirmation outcomes'))
    jobs=[]
    for split in ('fitting','calibration'):
        for key,fixture in panels[split].items():
            path=OUT/'bootstrap'/split/(key+'.json.gz')
            if path.exists():continue
            jobs.append((key,fixture,source,path,dict(keep_observations=True,replay=False)))
    completed=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures={pool.submit(worker,job):job for job in jobs}
        for f in as_completed(futures):
            row=f.result();completed+=1
            state('author_round1_and_development_collection',completed=completed,total=len(jobs),last=row)
    rows=[]
    for split in ('fitting','calibration'):
        for path in sorted((OUT/'bootstrap'/split).glob('*.json.gz')):
            r=read(path);rows.append(dict(split=split,room=path.name[:-8],**r['summary']))
    save(OUT/'bootstrap_summary.json',rows)
    print(json.dumps(dict(rooms=len(rows),success=sum(r['success'] for r in rows))))


def on_policy(workers=4):
    frozen=read(OUT/'policy_frozen.json');source=Path(frozen['source']).read_text()
    panels=read(OUT/'panels.json')['groups'];jobs=[]
    immutable(OUT/'on_policy/manifest.json',dict(source_sha256=frozen['source_sha256'],
        splits=['fitting','calibration'],selection='all declared development rooms; no future outcome filtering'))
    for split in ('fitting','calibration'):
        for room,fixture in panels[split].items():
            path=OUT/'on_policy'/split/(room+'.json.gz')
            if not path.exists():jobs.append((room,fixture,source,path,dict(keep_observations=True,replay=False)))
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for i,result in enumerate(pool.map(worker,jobs)):
            state('on_policy_development_collection',completed=i+1,total=len(jobs),last=result)
    rows=[dict(split=split,room=path.name[:-8],**read(path)['summary'])for split in ('fitting','calibration')
          for path in sorted((OUT/'on_policy'/split).glob('*.json.gz'))]
    save(OUT/'on_policy/summary.json',rows)
    print(json.dumps(dict(rooms=len(rows),success=sum(r['success']for r in rows))))


def candidate_actions(policy, z, actual):
    """Frozen candidate rule independent of model version and future outcomes."""
    values=policy.values(z)
    from experiments.planar_tasks.authoring_v2.symbolic_general.core import evaluate
    ranked=sorted([(evaluate(r.get('score',['lit',0,'1']),values),-i,r['option'])
                   for i,r in enumerate(policy.rules) if evaluate(r['when'],values)],reverse=True)
    actions=[actual] if actual!='brake' else []
    options=frontend.option_table()
    if actions:
        a=options[actual]
        actions += [name for name,o in options.items() if name!=actual and o['pair']==a['pair'] and z['legal:'+name]]
    # A deliberate orbit alternative makes direction-visibility prediction testable.
    actions += [name for name in ('orbit_0_ccw','orbit_0_cw','orbit_1_ccw') if z['legal:'+name]]
    actions += [name for _,_,name in ranked if name!='brake']
    return list(dict.fromkeys(actions))[:4]


def fork_worker(job):
    fixture,prefix,actions,source,room,path=job
    rows=[]
    for action in actions:
        try:
            result=simulation.branch(fixture,prefix,action,source)
            rows.append(dict(room=room,**result))
        except Exception as exc:
            rows.append(dict(room=room,action=action,error=type(exc).__name__+': '+str(exc),
                             reservation_ticks=sum(a['ticks'] for a in prefix)+160))
    save(path,dict(prefix=prefix,actions=actions,rows=rows,
                   source_sha256=hashlib.sha256(source.encode()).hexdigest()))
    return dict(path=str(path),rows=len(rows),errors=sum('error' in r for r in rows))


def collect_forks(source_path, split, workers=4, roots_per_room=4):
    source=Path(source_path).read_text();policy=scaffold.verify_source(source)
    fixtures=read(OUT/'panels.json')['groups'][split]
    jobs=[];manifest=[]
    for room,fixture in fixtures.items():
        trace=OUT/'bootstrap'/split/(room+'.json.gz')
        if not trace.exists():raise ValueError('need frozen prefix-generating trajectory for '+room)
        rows=read(trace)['records']
        # Clock-spaced roots, chosen without reading future effects or success.
        indices=[]
        for threshold in (0,120,300,600)[:roots_per_room]:
            candidate=next((i for i,r in enumerate(rows) if r['tick']>=threshold and r['choice']['option']!='brake'),None)
            if candidate is not None and candidate not in indices:indices.append(candidate)
        for i in indices:
            row=rows[i];z=row['semantic'];actual=policy.select(z)['option']
            actions=candidate_actions(policy,z,actual or 'brake')
            prefix=[r['command'] for r in rows[:i]]
            path=OUT/'forks'/split/(room+f'_{i:04d}.json.gz')
            manifest.append(dict(room=room,boundary=i,tick=row['tick'],actions=actions,
                                 prefix_sha256=hashlib.sha256(json.dumps(prefix,sort_keys=True).encode()).hexdigest()))
            if not path.exists():jobs.append((fixture,prefix,actions,source,room,path))
    immutable(OUT/'forks'/split/'manifest.json',dict(source_sha256=sha(source_path),roots=manifest,
              room_count=len(fixtures),roots_per_room=roots_per_room,selection='clock-spaced first non-brake boundary, never outcome-selected'))
    complete=0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(fork_worker,jobs):
            complete+=1;state('development_forks_'+split,completed=complete,total=len(jobs),last=result)
    return dict(roots=len(manifest),room_count=len(fixtures))


def effect_rows(split):
    rows=[]
    paths=sorted((OUT/'bootstrap'/split).glob('*.json.gz'))+sorted((OUT/'on_policy'/split).glob('*.json.gz'))
    for path in paths:
        record=read(path);room=path.name[:-8]
        # Fixed stride limits near-identical repeated frames; no label selection.
        stride=1 if path.parent.parent.name=='on_policy'else 4
        for row in record['records'][::stride]:
            if row['choice']['option']=='brake':continue
            rows.append(dict(room=room,action=row['choice']['option'],semantic=row['semantic'],
                             next_semantic=row['next_semantic'],audit=row['audit'],origin=str(path),tick=row['tick']))
    for path in sorted((OUT/'forks'/split).glob('*.json.gz')):
        for row in read(path)['rows']:
            if 'error' not in row:
                rows.append({k:row[k] for k in ('room','action','semantic','next_semantic','audit')})
    # Author/bootstrap snapshots recorded pre-state contacts separately. Include
    # the initial endpoint in the path proposition, including zero-tick rejects.
    # This is a label correction from existing raw truth, never new physics.
    for row in rows:
        audit=deepcopy(row['audit'])
        for name in ('object_static','robot_static','robot_peer'):
            audit[name]=bool(audit.get(name,False) or audit.get('before',{}).get(name,False))
        row['audit']=audit
    return rows


def active_episode_worker(job):
    room,fixture,source,model_path,path=job
    controller=Controller(EffectModel(read(model_path)))
    result=simulation.episode(fixture,source,controller=controller,keep_observations=True,
                              replay=False,wall_seconds=240)
    save(path,result)
    return dict(room=room,**result['summary'])


def active_fork_worker(job):
    fixture,prefix,actions,source,room,path,model_path=job
    model=EffectModel(read(model_path));rows=[]
    for action in actions:
        controller=Controller(model)
        try:
            result=simulation.branch(fixture,prefix,action,source,controller=controller)
            rows.append(dict(room=room,**result,development_online=controller.ledger))
        except Exception as exc:
            rows.append(dict(room=room,action=action,error=type(exc).__name__+': '+str(exc),
                             development_online=controller.ledger,
                             reservation_ticks=sum(a['ticks']for a in prefix)+160))
    save(path,dict(prefix=prefix,actions=actions,rows=rows,model_path=str(model_path),
                   source_sha256=hashlib.sha256(source.encode()).hexdigest()))
    return dict(room=room,branches=len(rows),errors=sum('error'in r for r in rows))


def active_collect(version,split,keys,source,model_path,workers=6):
    """Run prior frozen deployment; repair ambiguous prefixes using real forks."""
    fixtures=read(OUT/'panels.json')['groups'][split];folder=OUT/'active'/f'v{version}'/split
    policy=scaffold.verify_source(source);jobs=[]
    immutable(folder/'protocol.json',dict(version=version,rooms=keys,source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        model_sha256=sha(model_path),selection='first uncertain boundary, first stop, first boundary at ticks120 and600; no future outcome selection'))
    for room in keys:
        path=folder/'episodes'/(room+'.json.gz')
        if not path.exists():jobs.append((room,fixtures[room],source,model_path,path))
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for i,result in enumerate(pool.map(active_episode_worker,jobs)):
            state(f'active_v{version}_{split}_episodes',completed=i+1,total=len(jobs),last=result)
    jobs=[];manifest=[]
    for room in keys:
        trace=read(folder/'episodes'/(room+'.json.gz'));records=trace['records'];indices=[]
        uncertain=next((i for i,r in enumerate(records)if any(c['source']!='model'for c in (r.get('evidence')or{}).get('checks',[]))),None)
        stopped=next((i for i,r in enumerate(records)if r['choice']['option']=='brake'),None)
        candidates=[uncertain,stopped]+[next((i for i,r in enumerate(records)if r['tick']>=t),None)for t in (120,600)]
        for i in candidates:
            if i is not None and i not in indices:indices.append(i)
        for i in indices:
            row=records[i];z={k:True if k.startswith('permit:')else v for k,v in row['semantic'].items()}
            actual=policy.select(z)['option'];actions=candidate_actions(policy,z,actual or 'brake')
            prefix=[r['command']for r in records[:i]];path=folder/'forks'/(room+f'_{i:04d}.json.gz')
            manifest.append(dict(room=room,boundary=i,tick=row['tick'],actions=actions))
            if not path.exists():jobs.append((fixtures[room],prefix,actions,source,room,path,model_path))
    immutable(folder/'roots.json',manifest)
    with ProcessPoolExecutor(max_workers=workers)as pool:
        for i,result in enumerate(pool.map(active_fork_worker,jobs)):
            state(f'active_v{version}_{split}_forks',completed=i+1,total=len(jobs),last=result)
    rows=[]
    for room in keys:
        for r in read(folder/'episodes'/(room+'.json.gz'))['records']:
            if r['choice']['option']=='brake':continue
            rows.append(dict(room=room,action=r['choice']['option'],semantic=r['semantic'],next_semantic=r['next_semantic'],
                             audit=r['audit'],origin=f'active_v{version}_{split}',tick=r['tick']))
    for path in sorted((folder/'forks').glob('*.json.gz')):
        for row in read(path)['rows']:
            if 'error'not in row:rows.append({k:row[k]for k in ('room','action','semantic','next_semantic','audit')})
    save(folder/'summary.json',dict(rooms=len(keys),roots=len(manifest),training_rows=len(rows)))
    return rows


def freeze_models():
    if not (OUT/'policy_frozen.json').exists():raise ValueError('freeze author policy before model comparisons')
    policy=scaffold.verify_source(Path(read(OUT/'policy_frozen.json')['source']).read_text())
    train=effect_rows('fitting');calibration=effect_rows('calibration')
    rooms=sorted(read(OUT/'panels.json')['groups']['fitting'])
    previous=None;models=[]
    source=Path(read(OUT/'policy_frozen.json')['source']).read_text()
    # Progressive acquisition from disjoint new rooms; query-driven retention plus
    # fixed audits prevents a version from learning only its admitted actions.
    cumulative=[];acquisitions=[]
    for version,limit in enumerate((8,24,48)):
        new_rooms=set(rooms[:limit])-set(rooms[:(0,8,24)[version]])
        selected=[]
        if version:
            model_path=OUT/'models'/f'v{version-1}.json'
            active_train=active_collect(version,'fitting',sorted(new_rooms),source,model_path)
            active_cal=active_collect(version,'calibration',sorted(read(OUT/'panels.json')['groups']['calibration']),source,model_path)
            selected+=active_train;calibration+=active_cal
            acquisitions += [dict(version=version,room=row['room'],action=row['action'],reason='previous_model_runtime_and_uncertain_prefix_fork')for row in active_train]
        for row in train:
            if row['room'] not in new_rooms:continue
            if previous is None:
                reason='initial'
            else:
                prediction=previous.predict(row['action'],row['semantic'],scaffold.host_schema())
                # Predetermined high-confidence audits; all novel room prefixes
                # with insufficient local support are selected for acquisition.
                audit=int(hashlib.sha256((row['room']+row['action']+str(row.get('tick',0))).encode()).hexdigest()[:8],16)%4==0
                verdicts=[outcome_valid(policy,row['action'],s['values'],path_safe=s['path_safe'])[0]
                          for s in prediction['samples']]
                disagreement=bool(verdicts and any(verdicts) and not all(verdicts))
                if prediction['supported'] and not disagreement and not audit:continue
                reason=('missing_local_support'if not prediction['supported']else
                        'admission_disagreement'if disagreement else 'high_confidence_audit')
            selected.append(row);acquisitions.append(dict(version=version,room=row['room'],action=row['action'],reason=reason))
        cumulative+=selected
        model=EffectModel.fit(cumulative,calibration,version=version,parent=previous.sha256 if previous else None)
        immutable(OUT/'models'/f'v{version}.json',model.packet)
        models.append(dict(version=version,sha256=model.sha256,rows=len(cumulative),rooms=len(model.packet['training_rooms']),calibrated_actions=sum(v['padding']is not None for v in model.packet['calibration'].values())))
        previous=model
    immutable(OUT/'models'/'acquisition.json',acquisitions);immutable(OUT/'models'/'manifest.json',models)
    print(json.dumps(models))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='op',required=True)
    b=sub.add_parser('bootstrap');b.add_argument('--workers',type=int,default=4)
    b=sub.add_parser('on-policy');b.add_argument('--workers',type=int,default=4)
    b=sub.add_parser('forks');b.add_argument('--policy',type=Path,required=True);b.add_argument('--split',choices=['fitting','calibration'],required=True);b.add_argument('--workers',type=int,default=4)
    sub.add_parser('freeze-models')
    a=p.parse_args()
    if a.op=='bootstrap':bootstrap(a.workers)
    elif a.op=='on-policy':on_policy(a.workers)
    elif a.op=='forks':print(json.dumps(collect_forks(a.policy,a.split,a.workers)))
    elif a.op=='freeze-models':freeze_models()


if __name__=='__main__':main()
