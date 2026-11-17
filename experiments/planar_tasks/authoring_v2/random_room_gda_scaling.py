"""GDA-only random coverage expansion; fixed library, paired training seeds.

Separate output/sources from the completed coverage study. No CAP authoring or RL.
"""
import argparse
import collections
import json
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from . import random_room_coverage_v2 as V
from .random_room_tasks import generate

ROOT = V.ROOT
BASE = ROOT / 'outputs/semantic_rl/random_room_gda_scaling_20261117_03'
HOST = ROOT / 'outputs/semantic_rl_authors/random_room_gda_scaling_20261117_03'
PARENT = BASE / 'parent'
OLD = V.BASE
OLD_DATA = OLD / 'data_expanded'
SEEDS = (17, 29, 43)
UPDATES = 42000
TIERS = ('control', 'expanded')
GROUP = 'random-room-gda-scaling-v1'


def make(path):
    if (path / 'campaign.json').exists():
        c = V.Campaign(path)
        assert c.state['group'] == GROUP
        c.verify_sources()
        return c
    return V.Campaign.create(path, fixture=V.read(V.O.PARENT/'scenes.json')['cases'][0]['fixture'],
        sequence=[dict(target_id=None,a0=44,a1=44,ticks=4)], group=GROUP,
        limits=dict(attempts=7800,resets=7800,execution_ticks=14040000,fits=6,rounds=0,author_turns=0))


def conditions():
    return [f'{t}_s{s}' for t in TIERS for s in SEEDS]


def panel(seed, split, block=0):
    return [dict(r, id=f'{split}_b{block}_{r["movable_count"]}_{j:03d}', split=split,
                 fixture_sha256=V.digest(r['fixture'])) for j,r in enumerate(generate(seed)['cases'])]


def assert_disjoint(groups):
    seen = set()
    for group in groups:
        for r in group:
            key = V.digest(r['fixture'])
            if key in seen:
                raise ValueError('duplicate fixture across frozen cohorts')
            seen.add(key)
    return len(seen)


def select_successes(rows):
    grouped = collections.defaultdict(list)
    for r in rows:
        grouped[r['case_id']].append(r)
    return [winner for key in sorted(grouped) if (winner := V.fastest(grouped[key])) is not None]


def prepare():
    if V.frozen(PARENT):
        return
    for p in (V.O.PARENT, OLD_DATA, OLD/'parent'):
        V.Campaign(p).verify_frozen()
    previous = [V.read(p)['cases'] for p in (V.O.PARENT/'scenes.json', OLD/'parent/development.json',
                  OLD/'parent/fresh_test.json', OLD/'repair_development_panel/cases.json')]
    development = panel(2026111751,'development',0) + panel(2026111752,'development',1)
    test = panel(2026111753,'fresh_test',0)
    count = assert_disjoint(previous + [development, test])
    for r in development + test:
        V.X.fixture(r['fixture'])
    c = make(PARENT)
    V.save(PARENT/'development.json',dict(cases=development,seeds=[2026111751,2026111752]))
    V.save(PARENT/'fresh_test.json',dict(cases=test,seed=2026111753))
    V.save(PARENT/'diagnostic.json',dict(cases=previous[2],note='previously inspected100; not independent test'))
    V.save(PARENT/'protocol.json',dict(version=GROUP,old_data=str(OLD_DATA),old_data_seal=V.sha(OLD_DATA/'manifest.json'),
        control_demonstrations=270,development=400,test=200,diagnostic=100,training_seeds=SEEDS,
        updates=UPDATES,fits=6,workers=16,fit_workers=3,feature_workers=4,
        selection='one fastest successful full trajectory per new root, library order tie-break, retain original270',
        success_selection='no incomplete/failed positives; no extra trajectories from previously inspected test',
        changes='data size only between paired arms; all six from scratch with42000 updates',
        no_new_cap=True,no_rl=True,no_repair_suffixes=True,max_episodes=7800,max_ticks=14040000,
        total_wall_seconds=21600,fit_wall_seconds=7200,eval='fixed final checkpoints, no test model selection',
        interpretation='three optimizer seeds share one dataset/test; not three independent data-generation replications',
        preflight='read-only source and one-boundary policy recomputation; no extra physics'))
    V.finish(c)
    V.save(HOST/'PREPARED.json',dict(geometry_checks=600,unique_roots_including_prior=count,status='complete'))


def model_path(condition):
    return BASE/'fit'/condition/'model.pt'


def runtime(condition):
    if condition.startswith('CAP-'):
        return ORIGINAL_RUNTIME(condition)
    if V.CACHE.get('condition') == condition:
        return V.CACHE['f'], V.CACHE['model']
    if 'f' in V.CACHE:
        V.CACHE['f'].close()
    V.CACHE.clear()
    m = V.I.Scorer.load(model_path(condition))
    f = V.LibraryFeatures(V.entries(),V.O.protocol(V.O.CFG),execution_backend='trusted_inprocess')
    V.CACHE.update(condition=condition,f=f,model=m)
    return f,m


ORIGINAL_RUNTIME = V.get_runtime


def worker(job):
    V.get_runtime = runtime
    return V.run_worker(job)


def phase(path,jobs,stage):
    if V.frozen(path):
        return V.read(path/'result.json')['rows']
    c = make(path)
    V.save(path/'jobs.json',jobs)
    rows = V.bounded(jobs,worker,workers=16,stage=stage)
    V.save(path/'result.json',dict(rows=rows,episodes=len(rows),ticks=sum(r['ticks'] for r in rows)))
    issues = [r for r in rows if r['stop']=='wall_budget' or r.get('error') and r['error'].get('message')!='CAP first recommendation missing/illegal']
    V.save(path/'execution_audit.json',dict(infrastructure_issues=issues))
    if issues:
        raise RuntimeError(f'{stage}: infrastructure issues; not silently counted as task failures')
    with c.locked():
        c.state['used'].update(attempts=len(rows),resets=len(rows),execution_ticks=sum(r['ticks'] for r in rows))
        c._save()
    V.finish(c)
    return rows


def build_data(rows):
    p = BASE/'data_expanded'
    if V.frozen(p):
        return
    c = make(p)
    selected = select_successes(rows)
    jobs = [dict(task=r['case_id'],teacher=r['condition'],result_path=r['result_path'],sha256=V.sha(r['result_path']),path=str(p/'shards'/r['case_id'])) for r in selected]
    V.save(p/'selected.json',jobs)
    new = V.bounded(jobs,V.extraction_worker,workers=4,stage='extract_GDA_data')
    new.sort(key=lambda r:r['task'])
    V.Campaign(OLD_DATA).verify_frozen()
    V.combine(p,V.read(OLD_DATA/'extraction.json')['rows']+new)
    V.save(p/'data_lineage.json',dict(parent=str(OLD_DATA),parent_seal=V.sha(OLD_DATA/'manifest.json'),
        new_demonstrations=len(new),selection='same fastest-per-root rule',positive_failures=0))
    V.finish(c)


def fit_worker(job):
    tier,seed = job['tier'],job['seed']
    p = BASE/'fit'/f'{tier}_s{seed}'
    if (p/'complete.json').exists():
        return V.read(p/'complete.json')
    p.mkdir(parents=True,exist_ok=True)
    dp = OLD_DATA if tier=='control' else BASE/'data_expanded'
    ds = {k:np.load(dp/(k+'.npy'),mmap_mode='r') for k in ('contexts','candidates','offsets','targets')}
    ex,layout = V.read(dp/'extraction.json'),V.read(dp/'layout.json')
    ids = list(range(len(ds['contexts'])))
    curve_live=[]
    def progress(r):
        curve_live.append(r)
        V.save(p/'curve_live.json',curve_live)
        V.save(p/'progress.json',r)
        if r['update']%6000==0:
            print('FIT',tier,seed,r,flush=True)
    model,curve = V.train(ds,ids,layout,'GDA',ex['source_sha'],ex['action_protocol'],
        updates=UPDATES,seed=seed,wall_seconds=7200,progress=progress)
    acc = V.I.accuracy(model,ds,ids)
    V.save(p/'accuracy.json',acc)
    V.save(p/'curve.json',curve)
    model.save(p/'model.pt',dict(tier=tier,seed=seed,updates=UPDATES,demonstrations=ex['demonstrations']))
    reloaded = V.I.Scorer.load(p/'model.pt')
    assert all(torch.equal(x,reloaded.state_dict()[k]) for k,x in model.state_dict().items())
    result = dict(condition=f'{tier}_s{seed}',seed=seed,demonstrations=ex['demonstrations'],accuracy=acc,updates=UPDATES)
    V.save(p/'complete.json',result)
    return result


def fit():
    p = BASE/'fit'
    if V.frozen(p):
        return
    c = make(p)
    jobs = [dict(tier=t,seed=s) for t in TIERS for s in SEEDS]
    rows = V.bounded(jobs,fit_worker,workers=3,stage='GDA_fits')
    V.save(p/'result.json',dict(rows=rows))
    with c.locked():
        c.state['used']['fits']=6
        c._save()
    V.finish(c)


def summarize():
    rows = V.read(BASE/'evaluation/result.json')['rows']
    summary={}
    for split in ('fresh_test','fresh_test_diagnostic'):
        rr=[r for r in rows if r['split']==split]
        names=sorted({r['condition'] for r in rr})
        wins={n:{r['case_id'] for r in rr if r['condition']==n and r['success']} for n in names}
        union=set().union(*(v for n,v in wins.items() if n.startswith('CAP-')))
        if split=='fresh_test_diagnostic':
            old=V.read(ROOT/'outputs/semantic_rl_authors/random_room_coverage_20261117_02/RESULT.json')['rows']
            union={r['case_id'] for r in old if r['split']=='fresh_test' and r['condition'].startswith('CAP-') and r['success']}
        summary[split]=dict(teacher_union=len(union),methods={n:dict(success=len(v),n=sum(r['condition']==n for r in rr),retained=len(v&union),outside=len(v-union)) for n,v in wins.items()},
            paired={str(s):dict(gained=sorted(wins[f'expanded_s{s}']-wins[f'control_s{s}']),lost=sorted(wins[f'control_s{s}']-wins[f'expanded_s{s}'])) for s in SEEDS})
    execution = V.read(BASE/'collection/result.json')['rows']+rows
    assert len(execution)==7800
    ticks=sum(r['ticks'] for r in execution)
    assert ticks<=14040000
    seals={str(p):V.Campaign(p).verify_frozen() for p in (PARENT,BASE/'collection',BASE/'data_expanded',BASE/'fit',BASE/'evaluation')}
    result=dict(status='complete',summary=summary,episodes=len(execution),ticks=ticks,seals=seals,
        datasets=dict(control=270,expanded=V.read(BASE/'data_expanded/extraction.json')['demonstrations']),
        fits=V.read(BASE/'fit/result.json')['rows'],no_rl=True,no_new_cap=True)
    V.save(HOST/'RESULT.json',result)
    lines=['# GDA随机覆盖扩充：三训练种子配对结果','',
        '固定10 CAP，仅增加随机初态覆盖；每初态保留最快成功完整轨迹。原270示范与扩充数据均从头42000更新，无新CAP/RL。','',
        '| 训练seed | 原270条：新测试/200 | 扩数据：新测试/200 | 新增成功 | 丢失成功 |','|---|---:|---:|---:|---:|']
    for s in SEEDS:
        a=summary['fresh_test'];b=a['paired'][str(s)]
        lines.append(f"| {s} | {a['methods'][f'control_s{s}']['success']} | {a['methods'][f'expanded_s{s}']['success']} | {len(b['gained'])} | {len(b['lost'])} |")
    lines += ['',f"扩充后示范{result['datasets']['expanded']}条；教师并集{summary['fresh_test']['teacher_union']}/200。",
        '三优化种子共享同一训练/测试场景，不是三次独立场景采样。固定更新数不等于固定epoch；检查各组拟合率后解释。已有100例只作诊断，不混入新测试。',
        f'原始结果：`{HOST}/RESULT.json`。']
    (ROOT/'docs/agent_exchange/reviews/GDA_Coverage_Scaling_20261117.md').write_text('\n'.join(lines)+'\n')
    V.save(HOST/'FINISHED.json',dict(status='complete',episodes=len(execution),ticks=ticks))
    V.update(stage='complete',completed=7800,total=7800)


def main():
    V.HOST=HOST
    prepare()
    cases=V.read(PARENT/'development.json')['cases']
    p=BASE/'collection'
    rows=phase(p,[V.job_for(c,e['name'],p) for e in V.entries() for c in cases],'random_CAP_collection')
    build_data(rows)
    fit()
    fresh=V.read(PARENT/'fresh_test.json')['cases']
    known=[dict(c,split='fresh_test_diagnostic') for c in V.read(PARENT/'diagnostic.json')['cases']]
    p=BASE/'evaluation'
    jobs=[V.job_for(c,n,p) for n in conditions() for c in fresh+known]
    jobs += [V.job_for(c,e['name'],p) for e in V.entries() for c in fresh]
    phase(p,jobs,'fixed_GDA_evaluation')
    summarize()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    try:
        V.HOST=HOST
        prepare() if args.prepare_only else main()
    except BaseException:
        V.save(HOST/'FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()))
        V.update(stage='failed',status='failed')
        raise
