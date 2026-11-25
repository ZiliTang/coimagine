"""Versioned general symbolic CAP study and metered author development tools."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

from experiments.planar_tasks.authoring_v2.symbolic_general import scaffold, simulation
from experiments.planar_tasks.authoring_v2.symbolic_general.core import extract, digest
from experiments.planar_tasks.authoring_v2.symbolic_general.prediction import EffectModel
from experiments.planar_tasks.authoring_v2.random_room_author import RoomCampaign
from tools.cap_fault_curriculum_rooms import panel, SEEDS, content_hash


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/semantic_rl/cap_symbolic_general_v1'
AUTHOR = OUT/'author'
COUNTS = dict(author=16, fitting=48, calibration=16, confirmation=64)
PANEL_SEEDS = dict(author=851125101, fitting=851125103, calibration=851125107, confirmation=851125109)


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as f: return json.load(f)
    return json.loads(path.read_text())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp.'+str(os.getpid()))
    if path.suffix == '.gz':
        with gzip.open(tmp, 'wt') as f: json.dump(value, f, allow_nan=False, separators=(',', ':'))
    else:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value: raise ValueError('frozen artifact differs: '+str(path))
    else: save(path, value)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def locked():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT/'.lock').open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX); yield


def init():
    groups = {}; records = {}
    for split, count in COUNTS.items():
        SEEDS['symbolic_'+split] = PANEL_SEEDS[split]
        groups[split], audit = panel('symbolic_'+split, count)
        records.update({k: dict(v, split=split) for k, v in audit.items()})
    hashes = [content_hash(f) for group in groups.values() for f in group.values()]
    if len(set(hashes)) != len(hashes): raise ValueError('content overlap between study splits')
    immutable(OUT/'panels.json', dict(groups=groups, records=records, seeds=PANEL_SEEDS))
    immutable(OUT/'protocol.json', dict(version='symbolic-cap-general-v1', counts=COUNTS, seeds=PANEL_SEEDS,
        geometry='compact20', rays=72, push_face_degrees=45, orbit_target_gap_m=.05,
        max_ticks=1800, max_decisions=450, failure_cost='2 * environment time budget',
        author_seconds=3600, author_rounds_max=2, delta_C=.05, delta_F=.05,
        risk_threshold=.10, online=dict(branches=600,ticks=12000,seconds=60,queries_per_boundary=2),
        model_versions=3, follow_ticks=120,
        primary='fixed policy, candidate prefixes, risk threshold and online trigger; only effect model changes',
        coverage='admitted and safe successor completed / all tested candidates',
        failure='admitted but successor not completed / admitted; null when no admissions',
        online_metric='executed online branches per 100 decision boundaries; prefixes charged separately',
        model_quality='empirical conditional; no universal physical safety claim',
        completion='PUSH >=2cm goal advance; orbit obtains legal goal-aligned PUSH; MOVE >=5cm each robot; stops separate'))
    AUTHOR.mkdir(exist_ok=True)
    if not (AUTHOR/'starter.py').exists(): (AUTHOR/'starter.py').write_text(scaffold.source(scaffold.starter()))
    for key in list(groups['author'])[:3]:
        w, memory, _ = simulation.replay_boundary(groups['author'][key], [])
        immutable(AUTHOR/'boundaries'/(key+'.json'), dict(observation=simulation.observation(w)))
    save(OUT/'progress.json', dict(stage='framework_acceptance', updated=time.time(), author_started=False))
    return dict(status='initialized', counts=COUNTS, output=str(OUT))


def reserve(kind, *, ticks=0, resets=0):
    if type(ticks) is not int or type(resets) is not int or ticks < 0 or resets < 0:
        raise ValueError('nonnegative integer reservations required')
    with locked():
        state = read(AUTHOR/'clock.json')
        if time.time() >= state['deadline']-60: raise RuntimeError('author development ended; submit within reserved final minute')
        if state['ticks_reserved']+ticks > 250000 or state['resets']+resets > 2000:
            raise RuntimeError('author simulation budget exhausted')
        ident = len(state['calls'])
        row = dict(id=ident, kind=kind, at=time.time(), reserved_ticks=ticks, resets=resets, status='running')
        state['calls'].append(row);state['ticks_reserved'] += ticks;state['resets'] += resets
        save(AUTHOR/'clock.json', state)
    return ident


def settle(ident, actual_ticks, result):
    with locked():
        state = read(AUTHOR/'clock.json'); row = state['calls'][ident]
        if actual_ticks > row['reserved_ticks']: raise ValueError('work exceeded reservation')
        state['ticks_reserved'] -= row['reserved_ticks']-actual_ticks
        row.update(status='complete', actual_ticks=actual_ticks, finished=time.time(), result=result)
        save(AUTHOR/'clock.json', state)


def worker(job):
    key, fixture, source, path, kwargs = job
    try:
        result = simulation.episode(fixture, source, **kwargs)
        save(path, result)
        return dict(room=key, **result['summary'], file=str(path))
    except Exception as exc:
        row = dict(room=key, success=False, error=type(exc).__name__+': '+str(exc), ticks=kwargs.get('max_ticks',1800),
                   ticks_are_reservation=True)
        save(path, dict(summary=row, records=[], public=[]));return row


def test(policy, rooms, *, workers=4, max_ticks=1800):
    if type(max_ticks) is not int or not 1 <= max_ticks <= 1800 or not 1 <= workers <= 8:
        raise ValueError('bounded tick and worker counts required')
    source = Path(policy).read_text();scaffold.verify_source(source)
    fixtures = read(OUT/'panels.json')['groups']['author']
    keys = list(fixtures)[:rooms] if isinstance(rooms, int) else rooms
    if not keys or any(k not in fixtures for k in keys): raise ValueError('only author development rooms allowed')
    ident = reserve('test', ticks=len(keys)*max_ticks, resets=len(keys))
    directory = AUTHOR/'trials'/f'{ident:04d}'
    save(directory/'request.json', dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(), rooms=keys))
    (directory/'policy.py').write_text(source)
    jobs = [(k, fixtures[k], source, directory/(k+'.json.gz'),
             dict(max_ticks=max_ticks, keep_observations=True, replay=True)) for k in keys]
    with ProcessPoolExecutor(max_workers=workers) as pool: rows = list(pool.map(worker, jobs))
    summary = dict(id=ident, rooms=len(keys), success=sum(bool(r['success']) for r in rows),
                   object_static_commands=sum(r.get('object_static_commands', 0) for r in rows), rows=rows)
    save(directory/'summary.json', summary)
    settle(ident, sum(r['ticks'] for r in rows), str(directory/'summary.json'))
    return summary


def trace(trial, room, start, count):
    if room not in read(OUT/'panels.json')['groups']['author']: raise ValueError('not author room')
    row = read(AUTHOR/'trials'/f'{trial:04d}'/(room+'.json.gz'))
    result = []
    for record in row['records'][start:start+min(count, 20)]:
        result.append(dict(index=record['index'],tick=record['tick'], choice=record['choice'],
            semantic=record['semantic'], next_semantic=record['next_semantic'], command=record['command'],
            measured_effect={k: record['audit'][k] for k in ['goal_gain','displacement','object_static','robot_static','robot_peer']},
            event=record['audit']['event']))
    return dict(scope='development physical labels; never runtime inputs', summary=row['summary'], total=len(row['records']), rows=result)


def probe(policy, trial, room, boundary, actions):
    source = Path(policy).read_text();scaffold.verify_source(source)
    fixtures = read(OUT/'panels.json')['groups']['author']
    if room not in fixtures: raise ValueError('not author room')
    record = read(AUTHOR/'trials'/f'{trial:04d}'/(room+'.json.gz'))
    if not 0 <= boundary < len(record['records']): raise ValueError('unknown prefix boundary')
    prefix = [r['command'] for r in record['records'][:boundary]]
    prefix_max = sum(a['ticks'] for a in prefix)
    if not 1 <= len(actions) <= 8: raise ValueError('1..8 candidate options')
    ident = reserve('probe', ticks=len(actions)*(prefix_max+160), resets=len(actions))
    rows = []
    for action in actions:
        try:
            r = simulation.branch(fixtures[room], prefix, action, source)
            rows.append(dict(room=room, **r))
        except Exception as exc:
            # Failed replay may have consumed the full prefix reservation.
            rows.append(dict(room=room, action=action, error=repr(exc), charged_ticks=prefix_max+160))
    path = AUTHOR/'probes'/f'{ident:04d}.json.gz';save(path, rows)
    actual = sum(r.get('charged_ticks', r.get('prefix_ticks',0)+r.get('root_ticks',0)+r.get('follow_ticks',0)) for r in rows)
    settle(ident, actual, str(path))
    return dict(id=ident, output=str(path), rows=[{k:r[k] for k in ('action','error','safe_effective','successor','successor_completed','full_object_static') if k in r} for r in rows])


def fit_development():
    ident = reserve('fit')
    rows=[]
    for path in sorted((AUTHOR/'trials').glob('*/*.json.gz')):
        result=read(path)
        for row in result['records']:
            if row['choice']['option'] != 'brake':
                rows.append(dict(room=path.stem.removesuffix('.json'), action=row['choice']['option'],
                                 semantic=row['semantic'], next_semantic=row['next_semantic'],audit=row['audit']))
    for path in sorted((AUTHOR/'probes').glob('*.json.gz')):
        for row in read(path):
            if 'error' not in row:
                rows.append({k:row[k] for k in ('room','action','semantic','next_semantic','audit')})
    model=EffectModel.fit(rows)
    path=AUTHOR/'models'/f'{ident:04d}.json';save(path,model.packet);settle(ident,0,str(path))
    return dict(model=str(path),rows=len(rows),rooms=len(model.packet['training_rooms']),sha256=model.sha256)


def diagnose(policy, trial, room, boundary, model_path, *, repair=False):
    source=Path(policy).read_text();p=scaffold.verify_source(source)
    record=read(AUTHOR/'trials'/f'{trial:04d}'/(room+'.json.gz'))
    row=record['records'][boundary];z=row['semantic'];choice=p.select(z)
    action=choice['option']
    if action is None:return dict(status='no_action',choice=choice)
    model=EffectModel(read(model_path))
    prediction=model.predict(action,z,p.host)
    box={k:(v if type(v)is bool else [v,v]) for k,v in z.items()}
    report=p.check(box,action,model.interval_outcomes(prediction,p.host),quality='empirical',enumerate_rules=False)
    result=dict(choice=choice,model_sha256=model.sha256,support_rooms=prediction['local_rooms'],
                calibrated=prediction['calibrated'],report=report)
    if repair and (report['status']!='conditional' or not prediction['calibrated']):
        from experiments.planar_tasks.authoring_v2.symbolic_general import frontend
        option=frontend.option_table()[action]
        variants=[name for name,spec in frontend.option_table().items()
                  if z['legal:'+name] and (name==action or
                     (spec['family']==option['family'] and spec['pair']==option['pair']))]
        result['acquisition']=probe(policy,trial,room,boundary,variants[:4])
        result['refit']=fit_development()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='op',required=True)
    sub.add_parser('init'); sub.add_parser('status'); sub.add_parser('fit')
    b=sub.add_parser('build');b.add_argument('--rules',required=True,type=Path);b.add_argument('--out',required=True,type=Path)
    t=sub.add_parser('test');t.add_argument('--policy',required=True,type=Path);t.add_argument('--rooms',type=int,default=4);t.add_argument('--workers',type=int,default=4);t.add_argument('--max-ticks',type=int,default=1800)
    t=sub.add_parser('trace');t.add_argument('--trial',required=True,type=int);t.add_argument('--room',required=True);t.add_argument('--start',type=int,default=0);t.add_argument('--count',type=int,default=5)
    t=sub.add_parser('probe');t.add_argument('--policy',required=True,type=Path);t.add_argument('--trial',required=True,type=int);t.add_argument('--room',required=True);t.add_argument('--boundary',required=True,type=int);t.add_argument('--actions',nargs='+',required=True)
    t=sub.add_parser('diagnose');t.add_argument('--policy',required=True,type=Path);t.add_argument('--trial',required=True,type=int);t.add_argument('--room',required=True);t.add_argument('--boundary',required=True,type=int);t.add_argument('--model',required=True,type=Path);t.add_argument('--repair',action='store_true')
    args=parser.parse_args()
    if args.op=='init': result=init()
    elif args.op=='status': result=read(AUTHOR/'clock.json') if (AUTHOR/'clock.json').exists() else dict(status='not_started')
    elif args.op=='build':
        program=read(args.rules);text=scaffold.source(program);args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(text)
        result=dict(policy=str(args.out),sha256=sha(args.out),rules=len(program['rules']))
    elif args.op=='test': result=test(args.policy,args.rooms,workers=args.workers,max_ticks=args.max_ticks)
    elif args.op=='trace': result=trace(args.trial,args.room,args.start,args.count)
    elif args.op=='probe': result=probe(args.policy,args.trial,args.room,args.boundary,args.actions)
    elif args.op=='fit': result=fit_development()
    elif args.op=='diagnose': result=diagnose(args.policy,args.trial,args.room,args.boundary,args.model,repair=args.repair)
    print(json.dumps(result,ensure_ascii=False,allow_nan=False))


if __name__=='__main__':main()
