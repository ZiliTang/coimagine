"""Frozen run12 weights: eight selected diagnostic roots, four single-change views.

This is deliberate offline weight reuse for diagnosis, not checkpoint resume or
a new trained-policy claim. All variants share the same original weights/stats.
"""
from __future__ import annotations
import argparse
import hashlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path
import time
import numpy as np
import torch
from experiments.option_rl_v12.train import load_checkpoint,run_episode
from experiments.option_rl_v12.mode_env import ContinuousModeEnv as Baseline
from experiments.option_rl_v13.mode_env import ContinuousModeEnv as Fixed
from experiments.option_rl_v13.interface import Policy
from experiments.option_rl_v13.render_episodes import instance_from_dict
from experiments.option_rl_v13.parallel import worker_init
from experiments.option_rl_v13.diagnose_recovery import SOURCE

ROOTS=(0,1,8,18,19,23,34,62)
VARIANTS=('baseline','lifecycle_only','prior_only','both')
_CHECKPOINT=None

def job(task):
    global _CHECKPOINT
    checkpoint,idx,variant=task
    if _CHECKPOINT is None:_CHECKPOINT=load_checkpoint(Path(checkpoint))
    old,stats,data=_CHECKPOINT
    if variant in ('prior_only','both'):
        policy=Policy(data['obs_dim'],data['hidden']);policy.load_state_dict(old.state_dict());policy.eval()
    else:policy=old
    factory=Fixed if variant in ('lifecycle_only','both') else Baseline
    _,s=run_episode(policy,instance_from_dict(data['manifest']['validation_instances'][idx]),stats,
                    deterministic=True,env_factory=factory)
    phases=Counter(p for d in s['actions'] for p in d['phases']);denom=sum(phases.values())
    s.update(root=idx,variant=variant,phase_shares={k:v/denom for k,v in phases.items()},
             failed_decision_share=sum(e!='continuing' for d in s['actions'] for e in d['exits'])/denom,
             failure_or_wait_share=sum(e!='continuing' or p in ('peer_wait','perception_wait') for d in s['actions'] for e,p in zip(d['exits'],d['phases']))/denom)
    return s

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'roots').mkdir();started=time.monotonic()
    checkpoint=SOURCE/'policy.pt'
    tasks=[(str(checkpoint),i,v) for v in VARIANTS for i in ROOTS]
    # Bound wall time and preserve any diagnostic output already written.
    import os
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
    with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=worker_init) as executor:
        results=[]
        for r in executor.map(job,tasks,timeout=90):
            results.append(r);(args.out/'roots'/f"root_{r['root']:02d}__{r['variant']}.json").write_text(json.dumps(r))
    summary={}
    for variant in VARIANTS:
        rows=[x for x in results if x['variant']==variant]
        summary[variant]=dict(roots=len(rows),successes=sum(x['success'] for x in rows),
            mean_progress=float(np.mean([x['progress'] for x in rows])),
            failed_decision_share=float(np.mean([x['failed_decision_share'] for x in rows])),
            failure_or_wait_share=float(np.mean([x['failure_or_wait_share'] for x in rows])),
            anomalies=sum(not x['eligible_for_ppo'] for x in rows))
    result=dict(checkpoint=str(checkpoint),weight_migration='offline diagnostic only',roots=ROOTS,
        selection='selected development failures plus one previously successful root; not full validation or test',
        seconds=time.monotonic()-started,summary=summary,
        source_sha256={str(p.relative_to(Path(__file__).resolve().parents[2])):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('mode_env.py'),Path(__file__).with_name('push_controller.py'),Path(__file__).with_name('interface.py')]})
    (args.out/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
