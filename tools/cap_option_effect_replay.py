"""Measure one-option effects for the archived action-ranking forks."""
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.candidate_semantics import fitted
from tools.cap_semantic_fork_prefixes import DOC as PREVIOUS_DOC, OUT as PREFIX, ref, run

OUT = PREFIX.parent/'option_effects_20261122'
DOC = PREVIOUS_DOC.parent/'2026-11-22_option_effects'


def public_centre(obs):
    box = fitted(obs,obs['goal']['target_id'])
    return None if box is None else [box['x'],box['y']]


@torch.inference_mode()
def measure(job):
    key, fixture, row, ticks = job
    if 'unchanged' in row:
        return []
    with run.runtime.world(fixture) as (env, library):
        prefix = []
        for _ in range(row['step']):
            action = ref.choose(run.ACTOR,library.panel(env.history))
            prefix.append(action)
            run.runtime.advance(env,action)
        assert env.history.sha256==row['prefix']
    result = []
    for i,cmd in enumerate(row['actions']):
        with run.runtime.world(fixture) as (env,_):
            for a in prefix:
                run.runtime.advance(env,a)
            assert env.history.sha256 == row['prefix']
            before = ref.dq.current_event(env.history)['observation']
            assert before['tick']==row['current']['tick']
            target0,robots0 = ref.position(env)
            estimate0 = public_centre(before)
            run.runtime.advance(env,cmd)
            after = ref.dq.current_event(env.history)['observation']
            actual = after['tick']-before['tick']
            assert actual==ticks[i] and 0<actual<=cmd['ticks']
            target1,robots1 = ref.position(env)
            # Simulator positions are used as response variables and audit values only.
            result.append(dict(room=key,split=row['split'],i=i,prefix=row['prefix'],step=row['step'],
                               action=cmd,dt=actual,target0=target0,target1=target1,
                               robots_delta=np.subtract(robots1,robots0).tolist(),
                               observed0=estimate0,observed1=public_centre(after),
                               after_hash=env.history.sha256))
    return result


def main():
    _,dev,_ = run.initialize()
    p = ref.read(run.OUT/'protocol.json')
    assert ref.campaign.sha(ref.fit.OUT/'best_train.pt') == p['actor'] == ref.read(ref.OUT/'protocol.json')['layernorm']
    test = ref.read(run.OUT/'panels.json')['test']
    assert ref.campaign.base.digest(test)==p['hashes']['test'] and ref.campaign.base.digest(dev)==ref.read(ref.OUT/'protocol.json')['panel']
    with gzip.open(PREFIX/'prefixes.jsonl.gz','rt') as f:
        rows=[json.loads(line) for line in f]
    jobs=[]
    for r in rows:
        k=r['room']
        if r['split']=='development':
            archive=ref.read(ref.OUT/'rooms'/(k+'.json'))
            if 'actions' in r:
                assert r['actions']==archive['actions'] and r['prefix']==archive['prefix']
            jobs.append((k,dev[k],r,archive.get('ticks',[])))
        else:
            archive=ref.read(run.OUT/'evaluation'/(k+'.json'))
            branches=archive['branches']
            assert r.get('actions',[])==[b['action'] for b in branches]
            jobs.append((k,test[k],r,[b['leaf']['1']['ticks'] for b in branches]))
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(8,mp_context=mp.get_context('fork')) as pool:
        with gzip.open(OUT/'steps.tmp.gz','wt') as f:
            for n,batch in enumerate(pool.map(measure,jobs,chunksize=1),1):
                for entry in batch:
                    f.write(json.dumps(entry,separators=(',',':'))+'\n')
                if n%50==0:
                    print(n,len(jobs),flush=True)
    (OUT/'steps.tmp.gz').replace(OUT/'steps.jsonl.gz')


if __name__=='__main__':
    main()
