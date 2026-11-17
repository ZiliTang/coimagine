"""One matched GD fit/evaluation and16-method coverage report on known100 cases."""
from pathlib import Path
import time
import traceback
import numpy as np
import torch

from . import random_room_coverage_v2 as V
from experiments.planar_tasks.semantic_rl.gd_attribution import GDScorer,train,verify_A_mask

BASE=V.ROOT/'outputs/semantic_rl/gd_coverage_comparison_20261117_01'
HOST=V.ROOT/'outputs/semantic_rl_authors/gd_coverage_comparison_20261117_01'
DATA=V.BASE/'data_expanded'
RETRIEVAL=V.ROOT/'outputs/semantic_rl/program_retrieval_comparison_20261117_01'
OLD_HOST=V.HOST
CACHE={}


def make(path):
    if (path/'campaign.json').exists():
        c=V.Campaign(path);c.verify_sources();return c
    return V.Campaign.create(path,fixture=V.read(V.PARENT/'fresh_test.json')['cases'][0]['fixture'],
        sequence=[dict(target_id=None,a0=44,a1=44,ticks=4)],group='gd-coverage-comparison-v1',
        limits=dict(attempts=100,resets=100,execution_ticks=180000,fits=1,rounds=0,author_turns=0))


def prepare():
    p=BASE/'parent'
    if V.frozen(p):return
    refs=[DATA,V.BASE/'fit_expanded',V.BASE/'evaluation',RETRIEVAL/'evaluation']
    seals={str(r):V.Campaign(r).verify_frozen() for r in refs}
    c=make(p)
    V.save(p/'cases.json',V.read(V.PARENT/'fresh_test.json'))
    V.save(p/'protocol.json',dict(data=str(DATA),seals=seals,arm='GD',demonstrations=270,boundaries=9292,
        seed=17,updates=28000,optimizer='same library_learning Adam/loss/minibatches',
        active_groups=['G','D','common'],candidate_support='unchanged complete legal action tree',
        no_A_values_or_missingness=True,retrain_from_scratch=True,
        outcome_panel='previously inspected100; fixed diagnostic not new blind test',
        reference_learners='expanded_G/GA/GDA only, all270 demos/28000 updates/seed17',
        retrieval='same270 trajectories, snapshot k1; input difference must be disclosed',
        fit_wall_seconds=7200,workers=4,max_ticks=1800,max_decisions=300,rollout_wall_seconds=1800,
        max_episodes=100,max_total_ticks=180000,total_wall_seconds=14400,
        no_rl=True,no_llm=True,no_new_CAP=True))
    V.finish(c)


def fit():
    p=BASE/'fit'
    if V.frozen(p):return
    c=make(p);ex=V.read(DATA/'extraction.json');layout=V.read(DATA/'layout.json')
    ds={k:np.load(DATA/(k+'.npy'),mmap_mode='r') for k in ('contexts','candidates','offsets','targets')}
    ids=list(range(len(ds['contexts'])));live=[]
    def progress(row):
        live.append(row);V.save(p/'curve_live.json',live);V.save(p/'progress.json',row)
        V.update(stage='fit_GD',completed=row['update'],total=28000)
        if row['update']%2000==0:print('FIT_GD',row,flush=True)
    model,curve=train(ds,ids,layout,'GD',ex['source_sha'],ex['action_protocol'],updates=28000,seed=17,wall_seconds=7200,progress=progress)
    n=8;mask=verify_A_mask(model,ds['contexts'][:n],ds['candidates'][:n])
    accuracy=V.I.accuracy(model,ds,ids)
    model.save(p/'model.pt',dict(arm='GD',seed=17,updates=28000,demonstrations=270))
    loaded=GDScorer.load(p/'model.pt')
    assert all(torch.equal(v,loaded.state_dict()[k]) for k,v in model.state_dict().items())
    V.save(p/'curve.json',curve);V.save(p/'accuracy.json',accuracy)
    V.save(p/'MASK_AUDIT.json',mask)
    V.save(p/'complete.json',dict(accuracy=accuracy,mask=mask,seed=17,updates=28000))
    with c.locked():c.state['used']['fits']=1;c._save()
    V.finish(c)


def runtime(condition):
    assert condition=='GD'
    if not CACHE:
        CACHE['model']=GDScorer.load(BASE/'fit/model.pt')
        CACHE['features']=V.LibraryFeatures(V.entries(),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess')
    return CACHE['features'],CACHE['model']


def worker(job):
    V.get_runtime=runtime
    return V.run_worker(job)


def evaluate():
    p=BASE/'evaluation'
    if V.frozen(p):return
    c=make(p);cases=V.read(BASE/'parent/cases.json')['cases']
    jobs=[V.job_for(case,'GD',p) for case in cases]
    V.save(p/'jobs.json',jobs)
    rows=V.bounded(jobs,worker,workers=4,stage='GD_fixed_diagnostic')
    V.save(p/'result.json',dict(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows)))
    errors=[r for r in rows if r.get('error') or r['stop']=='wall_budget']
    V.save(p/'execution_audit.json',dict(errors=errors))
    if errors:raise RuntimeError('GD execution errors; inspect before conclusions')
    assert len(rows)==100 and sum(r['ticks'] for r in rows)<=180000
    with c.locked():
        c.state['used'].update(attempts=len(rows),resets=len(rows),execution_ticks=sum(r['ticks'] for r in rows));c._save()
    V.finish(c)


def main():
    V.HOST=HOST
    prepare();fit();evaluate()
    from .coverage_report import render
    report=render(include_gd=True)
    seals={str(p):V.Campaign(p).verify_frozen() for p in [BASE/'parent',BASE/'fit',BASE/'evaluation']}
    V.save(HOST/'RESULT.json',dict(status='complete',summary=report['summary'],seals=seals,report=report['report_dir'],fit=V.read(BASE/'fit/complete.json')))
    V.save(HOST/'FINISHED.json',dict(status='complete',episodes=100,ticks=V.read(BASE/'evaluation/result.json')['ticks']))
    V.update(stage='complete',completed=100,total=100)


if __name__=='__main__':
    try:main()
    except BaseException:
        V.save(HOST/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));V.update(stage='failed',status='failed');raise
