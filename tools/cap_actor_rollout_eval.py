"""Single-intervention held-room comparison with one shared frozen continuation value."""
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp
import time

import numpy as np
import torch

from tools import cap_actor_rollout_value as run
from tools.cap_actor_rollout_critic import load_model

VALUE=None
DOCS=Path(__file__).resolve().parents[1]/'docs/research/key_findings/2026-11-21_actor_value_rollout'
HORIZONS=(1,4)


def now():
    return time.perf_counter_ns()


def baseline(fixture,landmark):
    prefix=[]
    fork=None
    with run.runtime.world(fixture) as (env,library):
        while not run.runtime.ended(env):
            start=now()
            panel=library.panel(env.history)
            choice_start=now()
            action=run.ref.choose(run.ACTOR,panel)
            elapsed=now()-choice_start
            if fork is None:
                if env.decision_count>=landmark and len(panel[4])>1:
                    fork=dict(step=env.decision_count,tick=env.env.total_periods,hash=env.history.sha256,
                              pool=panel[4],preferred=panel[4].index(action),
                              root_panel_ms=(choice_start-start)/1e6,actor_score_ms=elapsed/1e6)
                else:
                    prefix.append(action)
            run.runtime.advance(env,action)
        return prefix,fork,dict(success=run.terminal(env),hash=env.history.sha256,ticks=env.env.total_periods)


def simulate(fixture,prefix,fork,first):
    # Reconstruct the simulator state and CAP memories. Charge this separately from short-lookahead work.
    started=now()
    with run.runtime.world(fixture) as (env,library):
        for action in prefix:
            panel=library.panel(env.history)
            assert action in panel[4]
            run.runtime.advance(env,action)
        panel=library.panel(env.history)
        assert env.history.sha256==fork['hash'] and panel[4]==fork['pool']
        prep=now()-started
        scores={}
        elapsed=0
        depth=0
        flagged_steps=0
        while not run.runtime.ended(env):
            start=now()
            action=first if depth==0 else run.ref.choose(run.ACTOR,panel)
            reason=run.runtime.advance(env,action)
            depth+=1
            done=run.runtime.ended(env)
            if not done:
                panel=library.panel(env.history)
            elapsed+=now()-start
            if depth<=max(HORIZONS):
                observation=run.ref.dq.current_event(env.history)['observation']
                reports=[robot['execution'] for robot in observation['robots']]
                if reason=='blocked' or any(e.get('direction_unavailable') or
                        e.get('stage') in ('blocked','move_blocked','yield_blocked') for e in reports):
                    flagged_steps+=1
            if depth in HORIZONS or done and depth<max(HORIZONS):
                leaf_start=now()
                score=float(run.terminal(env)) if done else float(VALUE(torch.from_numpy(panel[0][None]))[0])
                cost_ms=(elapsed+now()-leaf_start)/1e6
                for h in HORIZONS:
                    if h==depth or done and h>depth:
                        scores[str(h)]=dict(score=score,terminal=done,steps=depth,flagged_steps=flagged_steps,
                                            ticks=env.env.total_periods-fork['tick'],ms=cost_ms)
        assert set(scores)=={'1','4'}
        return dict(action=first,leaf=scores,success=run.terminal(env),ticks=env.env.total_periods,
                    history=env.history.sha256,replay_ms=prep/1e6)


@torch.inference_mode()
def case(job):
    k,f=job
    landmark=(0,5,12,25)[int(k.split('_')[-1])%4]
    prefix,fork,actual=baseline(f,landmark)
    row=dict(room=k,objects=1+len(f['movable_objects']),landmark=landmark,fork=fork,actor=actual['success'])
    if fork is None:
        return dict(row,branches=[],h1=actual['success'],h4=actual['success'])
    branches=[simulate(f,prefix,fork,a) for a in fork['pool']]
    pref=fork['preferred']
    assert (branches[pref]['history'],branches[pref]['success'])==(actual['hash'],actual['success'])
    for h in HORIZONS:
        key=str(h)
        win=max(range(len(branches)),key=lambda i:(branches[i]['leaf'][key]['score'],i==pref,-i))
        row['choice'+key]=win
        row['h'+key]=branches[win]['success']
    return dict(row,branches=branches)


def summarize(rows):
    active=[r for r in rows if r['branches']]
    mixed=[r for r in active if len(set(b['success'] for b in r['branches']))>1]
    def paired(a,b):
        return run.ref.scale.paired(*(np.array([r[k] for r in rows]) for k in (a,b,'objects')))
    stats=dict(n=len(rows),eligible=len(active),unchanged=len(rows)-len(active),mixed=len(mixed),
               realizable=sum(max(b['success'] for b in r['branches']) for r in active),
               candidates=sum(len(r['branches']) for r in active),
               actor_h1=paired('actor','h1'),actor_h4=paired('actor','h4'),h1_h4=paired('h1','h4'),
               mixed_success={k:sum(r[k] for r in mixed) for k in ('actor','h1','h4')},
               actor_score_ms=float(np.mean([r['fork']['actor_score_ms'] for r in active])),
               root_panel_ms=float(np.mean([r['fork']['root_panel_ms'] for r in active])))
    for h in HORIZONS:
        key=str(h)
        leaves=[(b['leaf'][key],b['success']) for r in active for b in r['branches']]
        cost={}
        for field in ('steps','ticks','ms'):
            vals=[sum(b['leaf'][key][field] for b in r['branches']) for r in active]
            cost[field]=dict(total=sum(vals),mean=float(np.mean(vals)),median=float(np.median(vals)))
        stats['look'+key]=dict(cost=cost,terminal=sum(l['terminal'] for l,y in leaves),
                               branch_brier=float(np.mean([(l['score']-y)**2 for l,y in leaves])),
                               overrides=sum(r['choice'+key]!=r['fork']['preferred'] for r in active))
        fewer=tie=more=actor_flagged=planned_flagged=0
        for r in active:
            branches=r['branches']
            old=branches[r['fork']['preferred']]['leaf'][key]['flagged_steps']
            new=branches[r['choice'+key]]['leaf'][key]['flagged_steps']
            fewer+=new<old
            more+=new>old
            tie+=new==old
            actor_flagged+=old>0
            planned_flagged+=new>0
        stats['look'+key]['execution_flags']=dict(fewer=fewer,tie=tie,more=more,
            actor_flagged=actor_flagged,planned_flagged=planned_flagged)
    stats['mean_replay_ms_per_fork']=float(np.mean([sum(b['replay_ms'] for b in r['branches']) for r in active]))
    common=[r for r in mixed if all(not b['leaf']['4']['terminal'] for b in r['branches'])]
    stats['mixed_all_continuing']=dict(n=len(common),**{k:sum(r[k] for r in common) for k in ('actor','h1','h4')})
    stats['both_win']=sum(all(b['success'] for b in r['branches']) for r in active)
    stats['all_fail']=sum(not any(b['success'] for b in r['branches']) for r in active)
    return stats


def main():
    global VALUE
    run.initialize()
    protocol=run.ref.read(run.OUT/'protocol.json')
    assert run.campaign.sha(run.ref.fit.OUT/'best_train.pt')==protocol['actor']
    assert run.campaign.sha(run.OUT/'value.pt')==run.ref.read(run.OUT/'fit.json')['sha']
    VALUE=load_model()
    test=run.ref.read(run.OUT/'panels.json')['test']
    assert run.campaign.base.digest(test)==protocol['hashes']['test']
    dest=run.OUT/'evaluation'
    dest.mkdir(exist_ok=True)
    jobs=[(k,f) for k,f in test.items() if not (dest/(k+'.json')).exists()]
    run.status('配对分叉闭环',len(test)-len(jobs),len(test))
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        for n,future in enumerate(as_completed([pool.submit(case,j) for j in jobs]),1):
            row=future.result()
            run.ref.save(dest/(row['room']+'.json'),row)
            run.status('配对分叉闭环',len(test)-len(jobs)+n,len(test))
    rows=[run.ref.read(dest/(k+'.json')) for k in sorted(test)]
    stats=summarize(rows)
    DOCS.mkdir(parents=True,exist_ok=True)
    with (DOCS/'per_room.csv').open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['room','objects','landmark','fork_step','prefix','num_candidates','actor_choice','h1_choice','h4_choice',
                    'actor_success','h1_success','h4_success','successful_candidates'])
        for r in rows:
            fork=r['fork'] or {}
            w.writerow([r['room'],r['objects'],r['landmark'],fork.get('step'),fork.get('hash'),len(r['branches']),
                        fork.get('preferred'),r.get('choice1'),r.get('choice4'),r['actor'],r['h1'],r['h4'],
                        sum(b['success'] for b in r['branches'])])
    run.ref.save(run.OUT/'FINISHED.json',stats)
    run.status('完成',len(rows),len(test))


if __name__=='__main__':
    main()
