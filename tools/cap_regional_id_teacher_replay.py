"""Evaluate both actors on identical held histories from verified fastest-success paths."""
import csv
import gzip
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from tools import cap_regional_id_actor as run

study=run.study
MODELS={}


@torch.inference_mode()
def replay(job):
    panel,cid,fixture,sid,sids=job
    teacher=sids.index(sid)
    directory=study.OUT/'evaluation'/sid
    original=study.read(directory/(cid+'.json'))
    with gzip.open(directory/(cid+'.json.gz'),'rt') as f:
        trace=json.load(f)
    dq=study.dq
    env=dq.fast_runtime.ScopedEnvironment(fixture,dq.study.CFG,fast_feedback=True,record_details=False,
                                         history_factory=dq.fast_runtime.AuditedCachedHistory)
    proposals=dq.Proposals()
    row=dict(panel=panel,case_id=cid,teacher=sid,boundaries=len(trace),distinct=0)
    for key in MODELS:
        for metric in ('id_hit','id_action','hard_action','distinct_id_action','distinct_hard_action'):
            row[key+'_'+metric]=0
    try:
        for expected in trace:
            x,c,s,valid,pool,raw=proposals.panel(env.history)
            teacher_action=raw[teacher]
            assert teacher_action==expected['a'] and teacher_action in pool
            group=pool.index(teacher_action)
            active=len(pool)>1
            row['distinct']+=active
            state=torch.from_numpy(x[None])
            for key,(identity,hard) in MODELS.items():
                cap=int(identity(state)[0].argmax())
                action_match=raw[cap]==teacher_action
                score=hard.values(state,torch.from_numpy(c[None]),torch.from_numpy(s[None]))[0].numpy()
                picked=int(np.argmax(np.where(valid,score,-np.inf)))
                row[key+'_id_hit']+=cap==teacher
                row[key+'_id_action']+=action_match
                row[key+'_hard_action']+=picked==group
                row[key+'_distinct_id_action']+=active and action_match
                row[key+'_distinct_hard_action']+=active and picked==group
            ev=env.step(teacher_action,remaining_ticks=1800-env.env.total_periods)
            assert ev['end_tick']==expected['tick']
        assert env.env.success and env.history.sha256==original['history'] and len(trace)==original['decisions']
    finally:
        env.close_reads()
        dq.fast_runtime.ACTIVE=None
    return row


def run_replays():
    torch.set_num_threads(1)
    frozen=study.read(study.OUT/'frozen.json')
    fixtures=study.read(study.OUT/'held.json')
    assert study.ext.old.digest(fixtures)==frozen['held']
    study.init_models(frozen['sids'])
    for key,old in [('fixed','actor'),('continued','polish')]:
        identity=run.ExpertActor()
        identity.load_state_dict(torch.load(run.OUT/(key+'.pt'),map_location='cpu',weights_only=True))
        hard=run.hard.actor()
        hard.load_state_dict(torch.load(run.hard.OUT/(old+'.pt'),map_location='cpu',weights_only=True))
        MODELS[key]=identity.eval(),hard.eval()
    with (run.hard.DOCS/'held_per_scene.csv').open() as f:
        jobs=[(r['panel'],r['case_id'],fixtures[r['case_id']],r['fastest'],frozen['sids'])
              for r in csv.DictReader(f) if r['fastest']]
    with ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('fork')) as pool:
        rows=list(pool.map(replay,jobs))
    assert len(rows)==215
    study.save(run.OUT/'held_teacher_replay.json',rows)


if __name__=='__main__':
    run_replays()
