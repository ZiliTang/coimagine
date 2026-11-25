"""Frozen room-paired comparison of planar CAP authoring with and without checker feedback."""
import argparse
import csv
import gzip
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

from experiments.planar_tasks.authoring_v2.checked_submission import verify_checked
from experiments.planar_tasks.relational_search.motion import CODEBOOK_HASH
from tools import cap_safe_nominal_study as safe
from tools.cap_entry_residual_study import immutable
from tools.cap_fault_curriculum_rooms import SEEDS, STRATA, VERSION, content_hash, panel

ROOT = safe.ROOT
DEFAULT_OUT = ROOT / 'outputs/semantic_rl/cap_checked_pair_v1/evaluation'
PANEL = 'checked_pair_v1_confirm'
SEED = 822160317
N = 64
ROLES = ('detour', 'parent', 'control', 'checked')
HISTORICAL = {
    'detour': '660e2f3a1a13eff1abb8126c48435c277364e8b6576dac97cc6b750ed893755b',
    'parent': 'adf79ad691026c2eefb3a516fd717ba129f9b782cdf905f7e343d8569c8cb432',
}
read, save, sha = safe.read, safe.save, safe.sha


def freeze_panel(out):
    out.mkdir(parents=True, exist_ok=True)
    SEEDS[PANEL] = SEED
    fixtures, records = panel(PANEL, N)
    ids, contents, inventory = safe.known_rooms()
    previous = safe.OUT / 'fixtures.json'
    previous_rooms = read(previous)
    ids.update(previous_rooms)
    contents.update(content_hash(f) for f in previous_rooms.values())
    inventory[str(previous.relative_to(ROOT))] = dict(count=len(previous_rooms), sha256=sha(previous))
    if set(fixtures) & ids or {v['content_sha256'] for v in records.values()} & contents:
        raise ValueError('room ID or content overlaps an existing panel')
    if len({v['content_sha256'] for v in records.values()}) != N:
        raise ValueError('duplicate room content')
    strata = dict(Counter(v['stratum'] for v in records.values()))
    if strata != {s:N//4 for s in STRATA}:
        raise ValueError('unexpected holdout stratification')
    immutable(out/'fixtures.json', fixtures)
    immutable(out/'records.json', records)
    immutable(out/'panel.json', dict(panel=PANEL, n=N, seed=SEED, generator=VERSION, counts=strata,
              fixtures_sha256=sha(out/'fixtures.json'), records_sha256=sha(out/'records.json'),
              exclusions=inventory, excluded_ids=len(ids), excluded_contents=len(contents), codebook=CODEBOOK_HASH,
              settings='fixed72-local-v1 full start, split_precision, ordinary and PUSH progress boundaries; optional PUSH callbacks follow submitted source',
              max_ticks=safe.S.MAX_TICKS, max_decisions=safe.S.MAX_DECISIONS,
              primary='checked minus control success, room-paired',
              secondary=['control minus parent', 'checked minus parent', 'parent minus detour',
                         'control minus detour', 'checked minus detour'],
              uncertainty='paired percentile bootstrap within each of the four equally weighted strata; 20000 resamples, seed 814; descriptive room uncertainty conditional on authors',
              diagnostic='PUSH target displacement and target-to-goal gain; object/static contact sampled at physics phase endpoints; stationary holds and no-push rooms'))
    return read(out/'panel.json')


def freeze_audit_subset(out, records):
    chosen = {s:min(k for k,r in records.items() if r['stratum']==s) for s in STRATA}
    immutable(out/'full_trace_subset.json',dict(rule='first room ID lexicographically per stratum',
              rooms=chosen, path='full_traces/{role}/{room}.json.gz'))
    return chosen


def verify_panel(out):
    plan = read(out/'panel.json')
    for name in ('records','fixtures'):
        if sha(out/(name+'.json')) != plan[name+'_sha256']:
            raise ValueError('frozen '+name+' changed')
    return plan


def verify_sources(arms):
    if set(arms) != set(ROLES):
        raise ValueError(f'expected {ROLES}')
    if len({a['composition'] for a in arms.values()}) != len(arms):
        raise ValueError('composition names must distinguish the four submitted programs')
    sources, verified = {}, {}
    for role in ROLES:
        arm = arms[role]
        campaign = Path(arm['campaign']).resolve()
        v2 = safe.Campaign(campaign).verify_frozen()
        if v2.get('status') != 'verified':
            raise ValueError(f'{role}: campaign seal not verified')
        policy = campaign/'composed.py'
        checksum = sha(policy)
        if checksum != arm['sha256'] or HISTORICAL.get(role, checksum) != checksum:
            raise ValueError(f'{role}: composed policy differs from declared source')
        receipt = None
        if role not in HISTORICAL:
            folder = Path(arm['submission']).resolve()
            receipt = verify_checked(folder)
            if receipt['source_sha256'] != checksum:
                raise ValueError(f'{role}: checked submission differs from campaign source')
            policy = folder/'policy.py'
        source = policy.read_bytes().decode('utf8')
        sources[role] = source
        verified[role] = dict(composition=arm['composition'], source=str(policy), sha256=checksum,
                              campaign=str(campaign), v2_seal=v2,
                              push_takeover_callback=safe.host.enabled(source))
        if receipt:
            verified[role].update(submission=str(folder), submission_sha256=sha(folder/'manifest.json'),
                                  checked_observations=[x['observation_sha256'] for x in receipt['cases']])
    return sources, verified


def execution(job):
    key, fixture, hooks, source_hash, fixture_hash, full_trace = job
    base = dict(room=key, source_sha256=source_hash, content_sha256=fixture_hash)
    started = time.monotonic()
    try:
        _, full = safe.episode((key, fixture, hooks))
        compact = []
        metrics = Counter()
        stats = full['result']
        for index, step in enumerate(full['steps']):
            command = step['action']
            duration = step['end_tick'] - step['tick']
            if duration <= 0 or step['tick'] != (compact[-1]['end'] if compact else 0):
                raise ValueError('noncontiguous command timeline')
            phases = step['phase_truth']
            positions = [step['truth'], *[p['end_truth'] for p in phases], step['end_truth']]
            static = any(p['object_static'] for p in positions)
            metrics['object_static_contact_at_sample'] |= int(static)
            start, end = step['truth']['object'][:2], step['end_truth']['object'][:2]
            goal = step['goal']
            gain = math.dist(start,goal) - math.dist(end,goal)
            moved = math.dist(start,end)
            target_push = command['target_id'] == step['target_id'] and min(command['a0'], command['a1']) < 32
            if target_push:
                metrics['target_push_commands'] += 1
                metrics['target_push_moved_ge_2cm'] += int(moved >= .02)
                metrics['target_push_goal_gain_ge_2cm'] += int(gain >= .02)
                metrics['target_push_goal_loss_ge_2cm'] += int(gain <= -.02)
                metrics['target_push_little_motion_with_static_sample'] += int(static and moved < .02)
            if command['a0'] == command['a1'] == 44:
                metrics['double_hold_ticks'] += duration
            compact.append(dict(tick=step['tick'], end=step['end_tick'], command=command,
                                public_history=full['boundaries'][index]['public_history'],
                                source_boundary=step['origin'], object_start=step['truth']['object'],
                                object_end=step['end_truth']['object'], object_static_sampled=static,
                                pusher_contact_at_issue=step['contact'],
                                goal_gain_m=gain, target_push=target_push))
        if not compact or compact[-1]['end'] != stats['ticks']:
            raise ValueError('incomplete terminal measurement')
        metrics['commands'] = len(compact)
        metrics['rooms_without_target_push'] = int(not metrics['target_push_commands'])
        metrics['final_goal_distance_m'] = math.dist(end,goal)
        if full_trace:
            # Keep complete, losslessly compressed host evidence only for the fixed subset.
            full_trace.parent.mkdir(parents=True,exist_ok=True)
            temporary = full_trace.with_suffix('.tmp')
            temporary.write_bytes(gzip.compress(json.dumps(full,ensure_ascii=False,allow_nan=False,
                separators=(',',':')).encode(),compresslevel=3,mtime=0))
            temporary.replace(full_trace)
            base['full_trace_sha256'] = sha(full_trace)
        return key, dict(base,status='complete',result=stats,metrics=dict(metrics),steps=compact,
                         elapsed_seconds=time.monotonic()-started)
    except Exception as exc:
        return key, dict(base,status='error',error=type(exc).__name__+': '+str(exc),
                         elapsed_seconds=time.monotonic()-started)


def compare(records, results, a, b):
    diffs = {s:[] for s in STRATA}
    outcomes = Counter()
    for room, record in records.items():
        x = results[a][room]['result']['success']
        y = results[b][room]['result']['success']
        outcomes[(x,y)] += 1
        diffs[record['stratum']].append(x-y)
    rng = np.random.default_rng(814)
    n = len(records)
    samples = np.zeros(20000)
    for group in diffs.values():
        samples += rng.choice(group,(20000,len(group))).sum(axis=1)
    better, worse = outcomes[1,0], outcomes[0,1]
    return dict(rooms=n,both_success=outcomes[1,1],both_failed=outcomes[0,0],
                rescued=better,lost=worse,delta_pp=100*(better-worse)/n,
                paired_ci95_pp=np.quantile(samples*100/n,[.025,.975]).tolist(),
                strata={s:dict(n=len(d),rescued=d.count(1),lost=d.count(-1)) for s,d in diffs.items()})


def summarize(out, records, verified, results):
    by_role = {}
    flat = []
    for role in ROLES:
        counts = Counter()
        for room, row in results[role].items():
            values = row['metrics']
            counts.update(values)
            r = row['result']
            flat.append(dict(role=role, composition=verified[role]['composition'], room=room,
                             content_sha256=records[room]['content_sha256'],stratum=records[room]['stratum'],
                             sha256=verified[role]['sha256'],success=r['success'],ticks=r['ticks'],
                             decisions=r['decisions'],termination=r['termination'],history=r['history'],**values))
        by_role[role] = dict(composition=verified[role]['composition'],rooms=len(records),
                             success=sum(row['result']['success'] for row in results[role].values()),
                             successes_by_stratum={s:sum(results[role][k]['result']['success'] for k,v in
                                   records.items() if v['stratum']==s) for s in STRATA},
                             totals=dict(counts))
    with (out/'per_room.csv').open('w',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in flat for k in row)))
        writer.writeheader()
        for row in flat:
            writer.writerow({k:row.get(k,0) for k in writer.fieldnames})
    comps = [('checked','control'),('control','parent'),('checked','parent'),('parent','detour'),
             ('control','detour'),('checked','detour')]
    return dict(panel=PANEL,room_count=len(records),execution_count=len(flat),arms=by_role,
                paired={a+'_minus_'+b:compare(records,results,a,b) for a,b in comps})


def evaluate(out, arms_file, workers):
    if workers < 1:
        raise ValueError('positive number of workers required')
    plan = verify_panel(out)
    records, fixtures = read(out/'records.json'), read(out/'fixtures.json')
    traces = freeze_audit_subset(out,records)
    arms = read(arms_file)
    sources, verified = verify_sources(arms)
    sources_hashed = safe.runtime_sources()
    sources_hashed[str(Path(__file__).resolve().relative_to(ROOT))] = sha(Path(__file__))
    lock_record = dict(panel=PANEL, panel_sha256=sha(out/'panel.json'),arms=verified,runtime=sources_hashed)
    immutable(out/'compositions.json',lock_record)
    status = dict(status='running',rooms=plan['n'], workers=workers,expected=len(ROLES)*len(records),
                  started=time.time(), completed={},errors={})
    save(out/'progress.json',status)
    results = {}
    try:
        for role in ROLES:
            status['active'] = role
            save(out/'progress.json',status)
            arm = verified[role]
            jobs = [(k,f,arm['push_takeover_callback'],arm['sha256'],records[k]['content_sha256'],
                     out/'full_traces'/role/(k+'.json.gz') if k in traces.values() else None)
                    for k,f in fixtures.items()]
            dest = out/'runs'/role
            safe.host.old.collect(sources[role],jobs,execution,min(workers,len(jobs)),dest)
            rows = {k:read(dest/(k+'.json')) for k in fixtures}
            for room, row in rows.items():
                if row['source_sha256']!=arm['sha256'] or row['content_sha256']!=records[room]['content_sha256'] or row['room']!=room:
                    raise ValueError('saved result identity mismatch')
                if room in traces.values() and row['status']=='complete':
                    trace = out/'full_traces'/role/(room+'.json.gz')
                    if row['full_trace_sha256']!=sha(trace):
                        raise ValueError('host full trace mismatch')
            results[role] = rows
            status['errors'][role] = {k:r['error'] for k,r in rows.items() if r['status']=='error'}
            status['completed'][role] = sum(r['status']=='complete' for r in rows.values())
            save(out/'progress.json',status)
        if any(status['errors'].values()):
            raise ValueError('host exceptions occurred; see progress.json before interpreting the experiment')
        verify_panel(out)
        if verify_sources(arms)[1] != verified or any(sha(ROOT/path)!=digest for path,digest in sources_hashed.items()):
            raise ValueError('closed-loop inputs changed during execution')
        summary = summarize(out,records,verified,results)
        immutable(out/'result.json',summary)
        status.update(status='complete',active=None,finished=time.time(),result_sha256=sha(out/'result.json'))
    except Exception as exc:
        status.update(status='incomplete',finished=time.time(),exception=type(exc).__name__+': '+str(exc))
        raise
    finally:
        save(out/'progress.json',status)
    return summary


def main():
    import fcntl
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest='phase',required=True)
    p = sub.add_parser('panel')
    p.add_argument('--out',type=Path,default=DEFAULT_OUT)
    e = sub.add_parser('evaluate')
    e.add_argument('--out',type=Path,default=DEFAULT_OUT)
    e.add_argument('--arms',type=Path,required=True)
    e.add_argument('--workers',type=int,default=8)
    args = cli.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/'.evaluation.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        result = freeze_panel(args.out) if args.phase == 'panel' else evaluate(args.out,args.arms,args.workers)
    print(json.dumps(result,ensure_ascii=False))


if __name__ == '__main__':
    main()
