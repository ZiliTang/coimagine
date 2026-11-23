"""Execute one original CAP proposal by the argmax of the ten identity logits."""
import argparse
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import torch

from tools import cap_regional_id_actor as run

study = run.study
MODEL = None


@torch.inference_mode()
def episode(job):
    panel,cid,fixture,key=job
    directory=run.OUT/key/'evaluation'/panel
    result_file=directory/(cid+'.json')
    if result_file.exists():
        return study.read(result_file)
    dq=study.dq
    env=dq.fast_runtime.ScopedEnvironment(fixture,dq.study.CFG,fast_feedback=True,record_details=False,
                                         history_factory=dq.fast_runtime.AuditedCachedHistory)
    library=dq.Proposals()
    steps=[]
    prev=None
    try:
        for _ in range(300):
            # Take the ten original proposals in frozen identity order. The common panel
            # also provides the established no-proposal fallback used by the other policies.
            x,_,_,_,fallback_pool,proposals=library.panel(env.history)
            logits=MODEL(torch.from_numpy(x[None]))[0]
            legal=MODEL.legal(torch.from_numpy(x[None]))[0]
            if legal.any():
                assert torch.isfinite(logits[legal]).all()
                index=int(logits.argmax())
                action=proposals[index]
                assert action is not None and legal[index]
            else:
                index=None
                assert len(fallback_pool)==1
                action=fallback_pool[0]
            switch=prev is not None and index is not None and prev!=index
            steps.append(dict(cap=index,action=action,switch=switch,
                              switch_distinct=bool(switch and (not legal[prev] or proposals[prev]!=action)),
                              supporters=[i for i,p in enumerate(proposals) if legal[i] and p==action]))
            prev=index
            env.step(action,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        row=dict(case_id=cid,checkpoint=key,boundaries=len(steps),success=bool(env.env.success),
                 ticks=env.env.total_periods,history=env.history.sha256,termination=env.env.termination,
                 initial_cap=steps[0]['cap'],switches=sum(s['switch'] for s in steps),
                 distinct_switches=sum(s['switch_distinct'] for s in steps),
                 fallback=sum(s['cap'] is None for s in steps))
        tmp=directory/(cid+'.part.gz')
        with gzip.open(tmp,'wt',compresslevel=1) as f:
            json.dump(steps,f,separators=(',',':'))
        tmp.replace(directory/(cid+'.json.gz'))
        study.save(result_file,row)
        return row
    finally:
        env.close_reads()
        dq.fast_runtime.ACTIVE=None


def evaluate(key):
    global MODEL
    torch.set_num_threads(1)
    frozen=study.read(study.OUT/'frozen.json')
    dev=study.read(study.OUT/study.ext.DEV)
    held=study.read(study.OUT/'held.json')
    assert study.ext.old.digest(dev)==frozen['development'] and study.ext.old.digest(held)==frozen['held']
    fixtures=dev|held
    study.init_models(frozen['sids'])
    MODEL=run.ExpertActor()
    MODEL.load_state_dict(torch.load(run.OUT/(key+'.pt'),map_location='cpu',weights_only=True))
    MODEL.eval()
    for panel,n in run.PANELS.items():
        path=run.OUT/key/'evaluation'/panel
        path.mkdir(parents=True,exist_ok=True)
        jobs=[(panel,cid,fixture,key) for cid,fixture in sorted(fixtures.items()) if cid.startswith(panel+'_')]
        assert len(jobs)==n
        with ProcessPoolExecutor(max_workers=16,mp_context=mp.get_context('fork')) as pool:
            rows=list(pool.map(episode,jobs))
        assert {r['case_id'] for r in rows}=={j[1] for j in jobs}
        print('EVAL',key,panel,n,sum(r['success'] for r in rows),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('key',choices=('fixed','continued'))
    evaluate(parser.parse_args().key)
