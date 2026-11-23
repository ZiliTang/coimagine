"""One irreversible CAP handoff after the first distinct proposals, on the paced executor."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import hashlib
import numpy as np

from tools import cap_option_semantic_factorial as study

OUT = study.OUT.parent / 'cap_route_horizon_20261123_01'
DOC = study.DOC.parent / '2026-11-23_cap_route_horizon'
EXECUTOR = 'split_precision'
HORIZONS = (0, 3, 10)
FOLDS = ('development', 'held')


def label(fold, expert, room):
    return study.read(study.OUT/'singles'/fold/EXECUTOR/expert/(room+'.json'))


def freeze():
    config = study.read(study.OUT/'protocol.json')
    experts = {m['sha']:bank+'/'+m['id'] for bank, ms in config['libraries'].items() for m in ms}
    dev = study.read(study.OUT/'development.json')
    wins = {e:{r for r in dev if label('development', e, r)['success']} for e in experts}
    a = max(sorted(wins), key=lambda e:len(wins[e]))
    b = max(sorted(set(wins)-{a}), key=lambda e:len(wins[e]-wins[a]))
    rooms = {}
    for fold in FOLDS:
        fixtures = study.read(study.OUT/(fold+'.json'))
        groups = {}
        for r,f in fixtures.items():
            k = (r[:6] if fold=='held' else '',len(f['movable_objects']))
            groups.setdefault(k,[]).append(r)
        rooms[fold] = sorted(r for group in groups.values() for r in sorted(group, key=lambda x:hashlib.sha256(
            ('horizon1/'+x).encode()).digest())[:len(group)//2])
        assert len(rooms[fold])==200
    hashes = []
    for fold in FOLDS:
        fixtures = study.read(study.OUT/(fold+'.json'))
        hashes.append({study.phase.input_hash(fixtures[r]) for r in rooms[fold]})
    assert not hashes[0] & hashes[1] and all(len(h)==200 for h in hashes)
    p = dict(pair=[a,b],names=[experts[a],experts[b]],development_full=[len(wins[a]),len(wins[b]),len(wins[b]-wins[a])],
             rooms=rooms, horizons=HORIZONS, controller=EXECUTOR,
             design='Follow A until first distinct legal proposals, then 0/3/10 more A proposals. Commit to B or stay. If B unavailable at cutoff keep A. Ended or never diverged retains A.',
             readout='ridge on B terminal win minus A terminal win, same public prefix, 5-fold development selects delay and switch threshold by all-room success. Compare to A on held.',
             fingerprints=config['code'])
    path=OUT/'protocol.json'
    if path.exists():
        assert study.phase.cap.digest(study.read(path))==study.phase.cap.digest(p)
    else:
        study.save(path,p)
    return p


def try_action(policy,obs):
    try:
        return policy.propose(obs)
    except ValueError:
        return None


def features(obs, first_geom, previous_geom, actions, counts, start_tick, h):
    geom = study.old.base.geom(obs)
    c = study.corridor_semantics(obs)
    context = [float(c[k]) for k in ('target_estimated','static_hit','movable_hit','swept_visibility',
                                      'observed_clear','unknown','visible_other_movables','goal_wall_distance_m')]
    feedback = [int(r['contacting_object']) for r in obs['robots']]
    angle = study.reference_angle(obs)
    moves = []
    for action in actions:
        # Compact explicit command, including the pushing angle relative to the target direction.
        vec = study.old.dq.action_encoding(action, obs['goal']['target_id'], angle)
        moves.extend(vec[:-40])
    return np.r_[geom,geom-first_geom,geom-previous_geom,context,feedback,moves,
                 obs['tick']/1800,(obs['tick']-start_tick)/1800,
                 counts['contact_ready']/(1+h),counts['contact_contact_timeout']/(1+h),counts['retimed']/(1+h)].tolist()


def rollout(fixture, pair, horizon, decide=None):
    policies = [study.phase.FrozenCAP(study.MODULES[e]) for e in pair]
    counts = Counter()
    checkpoints = {}
    first = cut = None
    first_geom = previous_geom = None
    route = 0
    invalid = 0
    n = 0
    with study.phase.world(fixture) as (w,_):
        def update(history):
            for p in policies:
                p.update(history)
        update(w.history)
        while not study.ended(w):
            obs = study.observe(w)
            a = policies[0].propose(obs) if not route else None
            b = try_action(policies[1], obs) if route or first is None or (
                horizon is not None and n==first['n']+horizon) else None
            if first is None and b is not None and a!=b:
                first = dict(n=n,tick=obs['tick'],history=w.history.sha256,actions=[a,b])
                first_geom = study.old.base.geom(obs)
                previous_geom = first_geom
                counts.clear()
            if horizon is None and first is not None and n-first['n'] in HORIZONS:
                checkpoints[str(n-first['n'])] = w.history.sha256
            if first is not None and horizon is not None and n==first['n']+horizon:
                cut = dict(history=w.history.sha256,tick=obs['tick'],n=n,eligible=b is not None)
                if b is not None:
                    cut['x'] = features(obs,first_geom,previous_geom,(a,b),counts,first['tick'],horizon)
                    route = int(decide is None or decide(cut['x']))
                    cut['switched'] = bool(route)
            if route and b is None:
                a = try_action(policies[0],obs)
                invalid += 1
            step = (b if route else a) or a or dict(target_id=None,a0=44,a1=44,ticks=1)
            if first is not None and not route:
                previous_geom = study.old.base.geom(obs)
            study.tick_action(w,step,EXECUTOR,update,counts)
            n+=1
        result = study.output(w,counts)
    return dict(checkpoints=checkpoints,first=first,cut=cut,invalid=invalid,success=result['success'],ticks=result['ticks'],
                termination=result['termination'],history=result['history'],decisions=result['decisions'])


def run(job, pair):
    fold,room,fixture=job
    path = OUT/'runs'/fold/(room+'.json')
    if path.exists():
        return
    fixed = [label(fold,e,room) for e in pair]
    # Audited A run saves the history of each later fork for cross-branch verification.
    results={}
    baseline = rollout(fixture,pair,None)
    for k in ('success','history','decisions','ticks'):
        assert baseline[k]==fixed[0][k],(room,k)
    first = baseline['first']
    for h in HORIZONS:
        res = rollout(fixture,pair,h) if first is not None else baseline
        assert res['first']==first
        assert (res['cut']['history'] if res['cut'] else None)==baseline['checkpoints'].get(str(h))
        if h==0:
            for k in ('success','history','decisions','ticks'):
                assert res[k]==fixed[1][k],(room,k)
        if res['cut'] is None or not res['cut']['eligible']:
            assert res['history']==baseline['history']
        results[str(h)] = res
    study.save(path,dict(room=room,fold=fold,baseline=baseline,horizons=results))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=14)
    args=p.parse_args()
    plan=freeze()
    study.initialize()
    jobs=[]
    for fold in FOLDS:
        fixtures=study.read(study.OUT/(fold+'.json'))
        jobs.extend((fold,r,fixtures[r]) for r in plan['rooms'][fold] if not (OUT/'runs'/fold/(r+'.json')).exists())
    with ProcessPoolExecutor(args.workers) as pool:
        for i,_ in enumerate(pool.map(partial(run,pair=plan['pair']),jobs,chunksize=1),1):
            if i%25==0 or i==len(jobs):
                print(i,len(jobs),flush=True)


if __name__=='__main__':
    main()
