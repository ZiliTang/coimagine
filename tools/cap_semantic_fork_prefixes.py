"""Reconstruct only archived pre-intervention public inputs with the frozen actor."""
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch

from tools import cap_actor_rollout_value as run

ref = run.ref
OUT = run.campaign.OUT/'action_semantics_20261122'
DOC = Path(__file__).resolve().parents[1]/'docs/research/key_findings/2026-11-22_action_semantics'


def public(obs):
    keep = ('own','arena','scan','execution','contacting_static','contacting_object','contacting_teammate')
    return dict(tick=obs['tick'], goal=obs['goal'], entities=obs['entities'],
                robots=[{k:r[k] for k in keep} for r in obs['robots']])


@torch.inference_mode()
def collect(job):
    split, room, fixture, row = job
    if split == 'development':
        step = row['cut']
        if step is None:
            return dict(room=room, split=split, unchanged=row['actual'][1])
        prefix, expected, pref = row['prefix'], row['actions'], 1
        outcomes = [row['cross'][1],row['actual'][1]]
    else:
        fork = row['fork']
        if fork is None:
            return dict(room=room, split=split, unchanged=row['actor'])
        step, prefix, expected, pref = fork['step'],fork['hash'],fork['pool'],fork['preferred']
        assert [b['action'] for b in row['branches']] == expected
        outcomes = [b['success'] for b in row['branches']]
    previous = prior_action = None
    with run.runtime.world(fixture) as (env, library):
        for _ in range(step):
            panel = library.panel(env.history)
            prior_action = ref.choose(run.ACTOR, panel)
            previous = public(ref.dq.current_event(env.history)['observation'])
            run.runtime.advance(env, prior_action)
        assert env.history.sha256 == prefix
        current = public(ref.dq.current_event(env.history)['observation'])
        panel = library.panel(env.history)
        ids = [panel[4].index(a) for a in expected]
        assert ref.choose(run.ACTOR, panel) == expected[pref]
        logits = run.ACTOR.values(*(torch.from_numpy(a[None]) for a in panel[:3]))[0].numpy()
        return dict(room=room, split=split, step=step, prefix=prefix, current=current, previous=previous,
                    prior_action=prior_action, actions=expected, pref=pref, outcomes=outcomes,
                    logits=logits[ids].tolist(), sources=[np.asarray(panel[2][i]).tolist() for i in ids])


def main():
    _, development, _ = run.initialize()
    protocol = ref.read(run.OUT/'protocol.json')
    held = ref.read(run.OUT/'panels.json')['test']
    assert protocol['actor'] == ref.campaign.sha(ref.fit.OUT/'best_train.pt') == ref.read(ref.OUT/'protocol.json')['layernorm']
    assert ref.campaign.base.digest(held) == protocol['hashes']['test']
    assert ref.campaign.base.digest(development) == ref.read(ref.OUT/'protocol.json')['panel']
    hashes = [run.campaign.input_hash(f) for db in [development,held] for f in db.values()]
    assert len(set(hashes)) == len(hashes) == 400
    jobs = [(s,k,f,ref.read(path/(k+'.json'))) for s,db,path in
            [('development',development,ref.OUT/'rooms'),('held',held,run.OUT/'evaluation')] for k,f in sorted(db.items())]
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(8,mp_context=mp.get_context('fork')) as pool:
        with gzip.open(OUT/'prefixes.tmp.gz','wt',compresslevel=1) as file:
            for i, row in enumerate(pool.map(collect,jobs,chunksize=2),1):
                file.write(json.dumps(row,separators=(',',':'),allow_nan=False)+'\n')
                if i%25 == 0: print(i,len(jobs),flush=True)
    (OUT/'prefixes.tmp.gz').replace(OUT/'prefixes.jsonl.gz')


if __name__ == '__main__':
    main()
