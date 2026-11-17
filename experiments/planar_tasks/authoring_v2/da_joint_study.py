"""Real-prefix action disputes; separate immutable inputs from prior studies."""
import argparse
import collections
import json
import time
from pathlib import Path

import torch
from . import random_room_coverage_v2 as V
from . import random_room_gda_scaling as S
from experiments.planar_tasks.semantic_rl.actions import ActionTree
from experiments.planar_tasks.semantic_rl.handoff_features import HandoffLibraryFeatures

BASE = V.ROOT/'outputs/semantic_rl/da_joint_20261117_01'
HOST = V.ROOT/'outputs/semantic_rl_authors/da_joint_20261117_01'
DONOR = S.BASE/'fit/control_s17/model.pt'
CACHE = {}


def prepare():
    if (BASE/'inputs.json').exists():
        verify_inputs()
        return
    seals = {str(p): V.Campaign(p).verify_frozen() for p in
             (S.PARENT, S.BASE/'fit', V.BASE/'data_expanded', V.O.PARENT)}
    cases = V.read(S.PARENT/'development.json')['cases']
    cases = [r for n in range(2, 6) for r in sorted(
        [c for c in cases if c['movable_count'] == n], key=lambda c:c['id'])[:12]]
    fresh = V.panel_rows(2026111781, 'joint_fresh', 80)
    old = [V.read(p)['cases'] for p in (V.O.PARENT/'scenes.json',
        V.PARENT/'development.json', V.PARENT/'fresh_test.json',
        S.PARENT/'development.json', S.PARENT/'fresh_test.json')]
    S.assert_disjoint(old+[fresh])
    counts = collections.Counter(r['condition'] for r in
        V.read(S.BASE/'collection/result.json')['rows'] if r['success'])
    order = sorted([e['name'] for e in V.entries()], key=lambda n:(-counts[n], n))
    hashes = {str(DONOR): V.sha(DONOR)}
    for e in V.entries():
        p = V.O.PARENT/'inputs'/e['name']/'policy.py'
        hashes[str(p)] = V.sha(p)
    V.save(BASE/'inputs.json', dict(seals=seals, hashes=hashes, cases=cases,
        fresh=fresh, cap_order=order, cap_rank_counts=dict(counts),
        rank_scope='fixed prior400 development, not fresh test',
        protocol=V.sha(V.ROOT/'docs/agent_exchange/tasks/DA_Joint_Representation_20261117.md')))
    V.save(HOST/'progress.json', dict(stage='prepared', donors=48, max_states=24))


def verify_inputs():
    for p, sha in V.read(BASE/'inputs.json')['hashes'].items():
        if V.sha(p) != sha:
            raise ValueError('changed frozen input: '+p)


def runtime(name):
    if CACHE.get('name') == name:
        return CACHE['f'], CACHE['model']
    if 'f' in CACHE:
        CACHE['f'].close()
    if name == 'donor':
        model = V.I.Scorer.load(DONOR)
        f = HandoffLibraryFeatures(V.entries(), V.O.protocol(V.O.CFG), execution_backend='trusted_inprocess')
    else:
        e = next(e for e in V.entries() if e['name']==name)
        model = None
        f = V.Features(e['source'], fast_embedding=True, action_protocol=V.O.protocol(V.O.CFG),
            execution_backend='trusted_inprocess', trusted_source_sha256=V.digest(e['source']))
    CACHE.update(name=name, f=f, model=model)
    return f, model


def audit_prefix(result, reference, cut):
    if result.get('prefix_replay', {}).get('executed', 0) < cut:
        raise ValueError('incomplete replay prefix')
    if V.digest(V.visible(result['public_rows'][:cut+1])) != V.digest(V.visible(reference['public_rows'][:cut+1])):
        raise ValueError('public prefix mismatch')
    tick = reference['public_rows'][cut]['observation']['tick']
    def frames(r): return [f for f in r['replay']['frames'] if f['tick'] <= tick]
    if V.digest(frames(result)) != V.digest(frames(reference)):
        raise ValueError('physics prefix mismatch')
    return dict(public_equal=True, physics_equal=True, cut=cut, ticks=tick)


def worker(job):
    torch.set_num_threads(1)
    p = Path(job['path'])
    p.mkdir(parents=True, exist_ok=True)
    if (p/'summary.json').exists():
        return V.read(p/'summary.json')
    f, model = runtime(job['condition'])
    prefix = []; ref = None
    if job.get('reference'):
        if V.sha(job['reference']) != job['reference_sha256']:
            raise ValueError('changed reference')
        ref = V.unpack(job['reference'])
        prefix = [r['action'] for r in ref['public_rows'][1:job['cut']+1]]
        if job.get('forced_action') is not None:
            prefix.append(job['forced_action'])
    result, _ = V.rollout(job['fixture'], f.source, model=model, features=f,
        config=V.O.CFG, max_ticks=1800, max_decisions=300, wall_seconds=1800,
        deterministic=True, seed=17, fast_feedback=True, prefix=prefix)
    start = time.monotonic()
    V.pack(p/'result.json.gz', result, compresslevel=1)
    storage = time.monotonic()-start
    audit = audit_prefix(result, ref, job['cut']) if ref else None
    if job.get('identity'):
        if V.digest(V.visible(result['public_rows'])) != V.digest(V.visible(ref['public_rows'])):
            raise ValueError('identity full public mismatch')
        if V.digest(result['replay']['frames']) != V.digest(ref['replay']['frames']):
            raise ValueError('identity full physics mismatch')
        audit['full_identity'] = True
    row = dict(job={k:v for k,v in job.items() if k!='fixture'}, **result['summary'],
        prefix_audit=audit, storage_seconds=storage,
        result_path=str(p/'result.json.gz'))
    V.save(p/'summary.json', row)
    return row


def phase(jobs, name):
    verify_inputs()
    if (BASE/name/'result.json').exists():
        return V.read(BASE/name/'result.json')['rows']
    startfile = HOST/'started.json'
    if not startfile.exists(): V.save(startfile, dict(at=time.time()))
    if time.time()-V.read(startfile)['at'] > 28800:
        raise TimeoutError('study wall budget8h')
    V.HOST = HOST
    source_files = [Path(__file__), Path(V.__file__), Path(V.O.__file__)]
    source_files += list((V.ROOT/'experiments/planar_tasks/semantic_rl').glob('*.py'))
    snapshot={}
    for source in source_files:
        rel=source.relative_to(V.ROOT); snapshot[str(rel)]=V.sha(source)
        dest=BASE/name/'sources'/rel; dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists() and V.sha(dest)!=V.sha(source):
            raise ValueError('running stage source changed: '+str(rel))
        dest.write_bytes(source.read_bytes())
    V.save(BASE/name/'sources.json',snapshot)
    V.save(BASE/name/'jobs.json', jobs)
    rows = V.bounded(jobs, worker, workers=16, stage=name)
    V.save(BASE/name/'result.json', dict(rows=rows))
    issues = [r for r in rows if r['stop']=='wall_budget' or
        r.get('error') and r['error']['message']!='CAP first recommendation missing/illegal']
    if issues: raise RuntimeError(f'{name}: infrastructure failures, see results')
    return rows


def collect():
    cases = V.read(BASE/'inputs.json')['cases']
    jobs = [dict(case_id=c['id'], movable_count=c['movable_count'], fixture=c['fixture'],
        condition='donor', path=str(BASE/'donors/runs'/c['id'])) for c in cases]
    return phase(jobs, 'donors')


def select_worker(job):
    """Reconstruct program memories from actual observations; no physics/query labels."""
    f = V.LibraryFeatures(V.entries(), V.O.protocol(V.O.CFG), execution_backend='trusted_inprocess')
    ref = V.unpack(job['result_path']); h = f.new_history()
    try:
        for i, row in enumerate(ref['public_rows'][:-1]):
            if i > 24: break
            h.append(row['observation'], row['action'])
            if h.sha256 != row['history_hash']: raise ValueError('history identity')
            with f.decision(h):
                if i not in (8,16,24) or row['observation']['tick'] > 1200: continue
                donor = ref['public_rows'][i+1]['action']; proposals = []
                by_name = {e['name']:m for e,m in zip(V.entries(),f.members)}
                for name in job['cap_order']:
                    q = by_name[name].worker.recommendation()
                    if q['first_legal'] and q['raw_first'] != donor and not any(q['raw_first']==p['action'] for p in proposals):
                        proposals.append(dict(cap=name, action=q['raw_first']))
                    if len(proposals)==2: break
                if proposals:
                    return dict(case_id=job['case_id'], movable_count=job['movable_count'],
                        cut=i, tick=row['observation']['tick'], history_hash=h.sha256,
                        reference=job['result_path'], reference_sha256=V.sha(job['result_path']),
                        donor_action=donor, alternatives=proposals)
    finally: f.close()
    return dict(case_id=job['case_id'], movable_count=job['movable_count'], excluded='no_dispute_at_predeclared_cuts')


def select():
    if (BASE/'selection.json').exists(): return V.read(BASE/'selection.json')['states']
    data = V.read(BASE/'inputs.json')
    rows = V.read(BASE/'donors/result.json')['rows']
    jobs = [dict(case_id=r['job']['case_id'], movable_count=r['job']['movable_count'],
        result_path=r['result_path'], cap_order=data['cap_order']) for r in rows]
    V.HOST = HOST
    found = V.bounded(jobs, select_worker, workers=4, stage='select_disputes')
    states=[]
    for n in range(2,6):
        rr=sorted([r for r in found if r['movable_count']==n and 'cut' in r], key=lambda r:r['case_id'])[:6]
        for j,r in enumerate(rr):
            r['split']='repair_train' if j<4 else 'handoff_holdout'
            states.append(r)
    V.save(BASE/'selection.json',dict(states=states,screening=found))
    return states


def branches():
    states=select(); cases={c['id']:c for c in V.read(BASE/'inputs.json')['cases']};jobs=[]
    for state in states:
        common=dict(case_id=state['case_id'],split=state['split'],cut=state['cut'],
            fixture=cases[state['case_id']]['fixture'],reference=state['reference'],
            reference_sha256=state['reference_sha256'])
        variants=[dict(kind='identity',condition='donor',identity=True)]
        for j,p in enumerate(state['alternatives']):
            variants += [dict(kind=f'action{j}',condition='donor',forced_action=p['action'],proposer=p['cap']),
                         dict(kind=f'cap{j}',condition=p['cap'],proposer=p['cap'])]
        for v in variants:
            jobs.append(dict(common,**v,path=str(BASE/'branches/runs'/state['case_id']/v['kind'])))
    return phase(jobs,'branches')


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','collect','select','branches','diagnose'])
    a=p.parse_args();prepare()
    if a.stage=='diagnose':
        collect();select();branches()
    elif a.stage!='prepare':globals()[a.stage]()


if __name__=='__main__':main()
