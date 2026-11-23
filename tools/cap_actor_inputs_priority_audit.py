"""Check what fixed CAP priority actually executes on its and the full actor's own prefixes."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import torch

from tools import cap_actor_inputs as exp

MODEL=None


@torch.inference_mode()
def audit(job):
    key,fixture,arm=job
    multi=agree=unsupported=fallback=0
    ranks=np.zeros(5,dtype=int)
    with exp.run.runtime.world(fixture) as (env,lib):
        while not exp.run.runtime.ended(env):
            panel=lib.panel(env.history)
            pool=panel[4]
            src=panel[2][:len(pool)].astype(bool)
            cap=np.flatnonzero(src.any(0))
            preferred=pool[0] if not len(cap) else pool[np.flatnonzero(src[:,cap[0]]).item()]
            if len(cap): ranks[cap[0]]+=1
            else: fallback+=1
            selected=exp.run.ref.choose(MODEL,panel)
            if len(pool)>1:
                multi+=1
                agree+=selected==preferred
                unsupported+=not bool(len(cap) and cap[0]==0)
            exp.run.runtime.advance(env,selected if arm=='plain800' else preferred)
        path=(exp.prior.OUT/'norm_comparison/plain800' if arm=='plain800' else exp.OUT/'evaluation/priority')/(key+'.json')
        assert env.history.sha256==exp.read(path)['history'] and exp.run.terminal(env)==exp.read(path)['success']
    return dict(room=key,arm=arm,multi=multi,agree=agree,no_top_at_multi=unsupported,rank_counts=ranks.tolist(),fallback=fallback)


if __name__=='__main__':
    exp.init()
    MODEL=exp.run.ref.sets.checkpoint(exp.run.ref.scale.OUT)
    panel=exp.read(exp.prior.OUT/'panel.json')
    result=[]
    with ProcessPoolExecutor(6,mp_context=mp.get_context('fork')) as pool:
        jobs=[(k,f,arm) for k,f in panel.items() for arm in ('plain800','priority')]
        for future in as_completed([pool.submit(audit,j) for j in jobs]): result.append(future.result())
    exp.save(exp.OUT/'priority_audit.json',sorted(result,key=lambda r:(r['room'],r['arm'])))
