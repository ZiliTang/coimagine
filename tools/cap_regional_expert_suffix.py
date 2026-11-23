"""Paired action-specific expert vs actor continuation on archived verified forks."""
import hashlib
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import torch
from tools import cap_regional_actor_iteration as old

study=old.study
OUT=old.OUT.parent/'cap_regional_replay_20261120_01'/'expert_suffix'
MODEL=None


@torch.inference_mode()
def expert_end(fixture, fork, first):
    env=old.ppo.environment(fixture)
    caps=old.cpp.Proposals()
    expert=None
    try:
        for step in range(300):
            x,_,sources,_,pool,raw=caps.panel(env.history)
            if expert is None:
                logits=MODEL(torch.from_numpy(x[None]))[0].numpy()
                i=int(logits.argmax())
                action=raw[i] if np.isfinite(logits[i]) else pool[0]
                if step==fork['step']:
                    assert env.history.sha256==fork['prefix'] and action==fork['base']
                    action=fork[first]
                    ids=np.flatnonzero(sources[pool.index(action)])
                    assert len(ids) and np.isfinite(logits[ids]).all()
                    expert=int(ids[np.argmax(logits[ids])]); supporters=ids.tolist()
                    assert action==raw[expert]
            else:
                action=raw[expert]
                if action not in pool:
                    return dict(complete=False,expert=expert,supporters=supporters,invalid_at=step)
            env.step(action,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:break
        assert expert is not None
        return dict(complete=True,expert=expert,supporters=supporters,success=int(env.env.success),
                    ticks=env.env.total_periods,steps=step+1,history=env.history.sha256)
    finally:
        env.close_reads();study.dq.fast_runtime.ACTIVE=None


def job(args):
    room,arch,ordinal=args
    path=OUT/'rooms'/(room['id']+'.json')
    if path.exists():return study.read(path)
    branches=arch['branches']
    row=dict(id=room['id'],objects=room['objects'],available=len(branches))
    if branches:
        branch=branches[ordinal%len(branches)]
        row.update(prefix=branch['prefix'],step=branch['step'],base=branch['base'],alt=branch['alternative'],
                   actor_base=arch['baseline'],actor_alt=branch['end'],
                   expert_base=expert_end(room['fixture'],branch,'base'),
                   expert_alt=expert_end(room['fixture'],branch,'alternative'))
    study.save(path,row)
    return row


def summarize(rows):
    counts=[[0]*3 for _ in range(3)]
    actor_only=expert_only=both=neither=pure=multi=0
    for r in rows:
        if not r['available'] or not all(r['expert_'+a]['complete'] for a in ('base','alt')):continue
        pure+=1
        for a in ('base','alt'):
            av=r['actor_'+a]['success']; ev=r['expert_'+a]['success']
            actor_only+=int(av>ev);expert_only+=int(ev>av);both+=av*ev;neither+=int(not av and not ev)
            multi+=int(len(r['expert_'+a]['supporters'])>1)
        actor=r['actor_alt']['success']-r['actor_base']['success']
        expert=r['expert_alt']['success']-r['expert_base']['success']
        counts[actor+1][expert+1]+=1
    return dict(done=len(rows),total=96,available=sum(bool(r['available']) for r in rows),pure=pure,
                actor_only=actor_only,expert_only=expert_only,both=both,neither=neither,multi=multi,
                deltas=counts,actor_only_rescue=sum(counts[2][:2]),expert_only_rescue=counts[0][2]+counts[1][2],
                both_rescue=counts[2][2])


def run():
    global MODEL
    torch.set_num_threads(1)
    protocol=study.read(old.OUT/'protocol.json')
    assert old.cpp.load()[1]['binary']==protocol['backend']
    (OUT/'rooms').mkdir(parents=True,exist_ok=True)
    jobs_count=0;all_rows=[]
    for rid in (1,2):
        model=old.INITIAL if rid==1 else old.OUT/'r01/checkpoint/actor.pt'
        assert study.sha(model)==study.read(old.OUT/f'r{rid:02d}/checkpoint/fit.json')['parent_sha']
        MODEL=old.actor_load(model)
        rooms=old.panel(old.SEED+rid,48)
        jobs=[(room,study.read(old.OUT/f'r{rid:02d}/iterative'/(room['id']+'.json')),i) for i,room in enumerate(rooms)]
        jobs_count+=len(jobs)
        with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as workers:
            for f in as_completed([workers.submit(job,j) for j in jobs]):
                all_rows.append(f.result());study.save(OUT/'progress.json',summarize(all_rows))
    assert len(all_rows)==jobs_count==96
    study.save(OUT/'result.json',summarize(all_rows))


if __name__=='__main__':run()
