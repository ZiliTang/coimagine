"""Outcome-independent initial-state draws; seeds and geometry rejections auditable."""
from copy import deepcopy
import random
from .semantic_rl_perturbation import BOUNDS
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search.contract import digest


def draw(base,level,rng,*,identifier,split,seen):
    if level=='anchor':return dict(id=identifier,split=split,family=base['family'],level=level,base_id=base['id'],fixture=deepcopy(base['fixture']),fixture_sha256=digest(base['fixture']),rejected=[])
    b=BOUNDS[level];rejected=[]
    for attempt in range(1000):
        f=deepcopy(base['fixture']);delta={}
        for key,body,pos,angle in [('object',f['object'],b['object_position'],b['object_angle']),*[(f'robot{i}',r,b['robot_position'],b['robot_angle']) for i,r in enumerate(f['robots'])],*[(r['label'],r,b['obstacle_position'],b['obstacle_angle']) for r in f['movable_objects']]]:
            for axis,scale in [('x',pos),('y',pos),('angle',angle)]:
                d=rng.uniform(-scale,scale);body[axis]+=d;delta[key+'.'+axis]=d
        try:X.fixture(f)
        except ValueError as exc:rejected.append(dict(delta=delta,reason=str(exc)));continue
        sha=digest(f)
        if sha in seen:rejected.append(dict(delta=delta,reason='duplicate'));continue
        seen.add(sha)
        return dict(id=identifier,split=split,family=base['family'],level=level,base_id=base['id'],fixture=f,fixture_sha256=sha,delta=delta,rejected=rejected)
    raise ValueError('geometry rejection cap exhausted; no outcome-based replacement')


def build(base_cases,*,rounds=32,seeds=(2026111604,2026111605,2026111606),exclude=()):
    seen={digest(c['fixture']) for c in base_cases}|set(exclude);families=sorted({c['family'] for c in base_cases});groups={f:[c for c in base_cases if c['family']==f] for f in families};panels={}
    for split,seed in zip(('train','validation','test'),seeds):
        rng=random.Random(seed);cases=[]
        specs=[]
        if split=='train':
            for _ in range(rounds):
                levels=['anchor']*4+['small']*8+['medium']*4;rng.shuffle(levels)
                specs.extend((rng.choice(families),level) for level in levels)
        else:
            count=4 if split=='validation' else 8
            specs=[(family,level) for family in families for level in ('small','medium') for _ in range(count)]
        for family,level in specs:
            base=rng.choice(groups[family]);cases.append(draw(base,level,rng,identifier=f'{split}_{len(cases):04d}',split=split,seen=seen))
        panels[split]=dict(seed=seed,bounds=BOUNDS,cases=cases,selection='geometry-valid only; no policy or outcome queried')
    return panels
