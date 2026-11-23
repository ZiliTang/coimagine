"""Greedy closed loop on a freshly frozen content-disjoint 200-room panel."""
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from .cap_ga_effect_data import CAPS, OUT, W, embedding, environment, fast, old, publish, read, save, step, tensors, terminal
from .cap_ga_effect_fit import Heads

HEADS={}


def initialize():
    old.initialize()
    base=old.NETWORKS['q']
    frozen=read(OUT/'models_frozen.json')
    for arm in 'ABC':
        p=OUT/'models'/(arm+'.pt')
        assert old.sha(p)==frozen['sha'][arm]
        model=Heads(base,arm)
        model.load_state_dict(torch.load(p,map_location='cpu',weights_only=True))
        HEADS[arm]=model.eval()


def run_single(case,arm):
    cfg=W.EventConfig(mode='duration')
    if arm not in W.CAP_FEATURES:
        source=W.CAP_LIBRARY[arm]
        W.CAP_FEATURES[arm]=W.Features(source,fast_embedding=True,action_protocol=W.protocol(cfg),
                                       execution_backend='trusted_inprocess',trusted_source_sha256=W.digest(source))
    result,_=W.rollout(case['fixture'],W.CAP_LIBRARY[arm],features=W.CAP_FEATURES[arm],config=cfg,
                       max_ticks=1800,max_decisions=300,wall_seconds=1800,deterministic=True,fast_feedback=True,
                       record_details=False)
    s=result['summary']
    assert s['error'] is None and s['stop'] in ('terminal','decision_budget','execution_budget'),s
    return dict(success=int(s['success']),ticks=s['ticks'],decisions=s['decisions'],stop=s['stop'])


@torch.no_grad()
def run_model(case,arm):
    env=environment(case['fixture'])
    started=time.monotonic()
    W.LIBRARY.reset()
    try:
        while not terminal(env):
            if time.monotonic()-started>1800:raise TimeoutError(case['case_id'])
            packet=W.LIBRARY.advance(env.history)
            u,q=embedding(old.NETWORKS['q'],*tensors(packet))
            score=q if arm=='original' else HEADS[arm](u)[0]
            assert score.isfinite().all()
            chosen=int(score[0,:len(packet['pool'])].argmax())
            step(env,packet['pool'][chosen])
        return dict(success=int(env.env.success),ticks=env.env.total_periods,decisions=env.decision_count,
                    stop=env.env.termination or 'budget',hash=env.history.sha256)
    finally:
        env.close_reads()
        fast.ACTIVE=None


def job(task):
    case,arm=task
    path=OUT/'evaluations'/arm/(case['case_id']+'.json')
    if path.exists():return read(path)
    result=run_single(case,arm) if arm in CAPS else run_model(case,arm)
    result.update(case_id=case['case_id'],arm=arm,movable_count=case['movable_count'])
    save(path,result)
    return result


def main():
    while not (OUT/'models_frozen.json').exists():
        if '异常' in read(OUT/'live/state.json')['stage']:raise RuntimeError('upstream job failed')
        time.sleep(3)
    panel=read(OUT/'panel.json')
    assert old.sha(OUT/'panel.json')==read(OUT/'protocol.json')['panel']
    jobs=[(case,arm) for case in panel for arm in ['original','A','B','C',*CAPS]]
    publish('新面板贪心闭环评估','每个房间执行四个评分器及七个单 CAP',3,0)
    completed=[]
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for fut in as_completed([pool.submit(job,j) for j in jobs]):
            completed.append(fut.result())
            if len(completed)%10==0:
                publish('新面板贪心闭环评估','每个房间执行四个评分器及七个单 CAP',3,len(completed))
    assert len({(r['case_id'],r['arm']) for r in completed})==len(jobs)==2200
    save(OUT/'evaluation.json',sorted(completed,key=lambda r:(r['case_id'],r['arm'])))
    from .cap_ga_effect_report import report
    report()


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        publish('评估异常',str(exc));raise
