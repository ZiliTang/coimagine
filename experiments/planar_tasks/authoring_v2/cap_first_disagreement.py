"""Frozen-pair replay for independent and near-neighbor semantic gate diagnostics."""
from coimagine.random_start.spec import ROBOT_DIAMETER
import copy
import csv
import hashlib
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from . import cap_inverse_scale as scale
from . import cap_temporal_attention as attention
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import load_checkpoint

OUT=scale.OUT.parent/'cap_semantic_disagreement_20261119_01'
GA=str(scale.OUT/'fit_1001/models/GA_s17/offline.pt')
ATT=str(attention.OUT/'Q.pt')
OLD={'ga':scale.OUT/'evaluation/q1001','attention':attention.OUT/'evaluation'}
KEY=Path('docs/research/key_findings/2026-11-19_global_attention')
SEED=2026111933

def legal_shift(f):
    rooms=scale.family.rooms
    robot=f['robots'][0]
    shape=rooms.polygon(robot['x'],robot['y'],ROBOT_DIAMETER,ROBOT_DIAMETER,robot['angle'])
    bodies=[f['object'],*f['movable_objects'],f['robots'][1]]
    sizes=[(.25,1.)]+[(b['width'],b['height']) for b in f['movable_objects']]+[(ROBOT_DIAMETER,ROBOT_DIAMETER)]
    obstacles=[rooms.polygon(b['cx'],b['cy'],2*b['half_w'],2*b['half_h'],b['angle']) for b in f['room']['obstacles']]
    others=[rooms.polygon(b['x'],b['y'],*size,b['angle']) for b,size in zip(bodies,sizes,strict=True)]
    return (rooms.inside(shape) and all(rooms.separated(shape,q) for q in obstacles+others)
            and .65<=math.hypot(robot['x']-f['object']['x'],robot['y']-f['object']['y'])<=1.4)

def freeze_panels():
    stored=read(attention.OUT/'frozen.json')
    assert sha(GA)==stored['ga'] and sha(ATT)==stored['attention']
    known=set(read(scale.OUT/'forbidden.json'))
    for p in scale.OUT.glob('batches/*_plan.json'):
        known.update(c['content_sha256'] for c in read(p))
    cases={c['case_id']:c for c in read(scale.OUT/'evaluation_panel.json')}
    fresh=[dict(case_id='sem_fresh_'+r['id'],fixture=r['fixture'],content_sha256=scale.family.content(r['fixture']),
                movable_count=r['movable_count']) for r in scale.family.rooms.generate(SEED)['cases']]
    neighbors=[]
    with (KEY/'first_disagreement.csv').open() as f:
        rows=[r for r in csv.DictReader(f) if r['ga']!=r['attention'] and not r['case_id'].startswith('inverse_test_2_')]
    rows.sort(key=lambda r:hashlib.sha256(('near33/'+r['case_id']).encode()).hexdigest())
    counts={'0':0,'1':0}
    for row in rows:
        arm=row['ga']
        if counts[arm]>=12:continue
        parent=cases[row['case_id']]
        variants=[]
        for axis in ('x','y'):
            for sign in (-1,1):
                f=copy.deepcopy(parent['fixture'])
                f['robots'][0][axis]+=sign*.02
                variants.append(dict(case_id=f"sem_near_{parent['case_id']}_{axis}{sign}",fixture=f,content_sha256=scale.family.content(f),
                        parent=parent['case_id'],movable_count=parent['movable_count'],axis=axis,shift_m=sign*.02,original_ga=int(arm)))
        if all(legal_shift(c['fixture']) for c in variants):
            counts[arm]+=1
            neighbors.extend(variants)
    assert counts=={'0':12,'1':12} and len(fresh)==200
    assert len({c['content_sha256'] for c in fresh+neighbors}-known)==296
    assert len({c['case_id'] for c in fresh+neighbors}-cases.keys())==296
    for name,panel in [('fresh',fresh),('neighbors',neighbors)]:
        path=OUT/(name+'_panel.json')
        if path.exists(): assert read(path)==panel
        else: save(path,panel)
    return neighbors+fresh

def initialize():
    attention.initialize_worker(ATT)
    fast_runtime.W.MODELS[GA]=load_checkpoint(GA)[0]

def execute(job):
    case,arm=job
    checkpoint=GA if arm=='ga' else ATT
    entry=dict(case=case,condition='semantic_'+arm,checkpoint=checkpoint,epsilon=0.,
               directory=str(OUT/'evaluation'/arm/case['case_id']))
    s=(attention.evaluate_case(entry) if arm=='attention' else scale.family.worker(entry))
    assert s['complete'] and s['error'] is None
    return s['case_id'],arm

def main():
    OUT.mkdir(exist_ok=True)
    cases=freeze_panels()
    jobs=[(c,a) for c in cases for a in ('ga','attention')]
    done=set()
    with ProcessPoolExecutor(9,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        for fut in as_completed([pool.submit(execute,j) for j in jobs]):
            done.add(fut.result())
            if len(done)%10==0:save(OUT/'progress.json',dict(completed=len(done),total=len(jobs)))
    assert len(done)==len(jobs)
    save(OUT/'progress.json',dict(completed=len(done),total=len(jobs)))

if __name__=='__main__':main()
