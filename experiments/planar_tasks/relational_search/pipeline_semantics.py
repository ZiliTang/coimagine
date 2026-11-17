"""Hermetic semantic feature evaluation and reproducible S/E/M ranking fits."""
from pathlib import Path
import json
import subprocess
import tempfile
import numpy as np

from experiments.planar_tasks.sandbox import policy_command
from .contract import canonical, digest, public_observation, validate_action
from .fit import fit, Ranker


# This wrapper is trusted and appended after the submitted definitions. User
# code runs with no repository, credentials, network or simulator mount.
WRAPPER = '''
import json, sys, math
from copy import deepcopy
request = json.loads(sys.stdin.read())
answers = []
if not isinstance(NAMES, (list, tuple)) or not 1 <= len(NAMES) <= 16 or len(set(NAMES)) != len(NAMES):
    raise ValueError('one to sixteen unique names required')
for query in request:
    memory = initial_memory()
    previous = None
    for event in query['history']:
        observation = deepcopy(event['observation'])
        memory = update_memory(memory, dict(observation=observation,
            action=None if previous is None else deepcopy(event['action']),
            executed_ticks=None if previous is None else observation['tick']-previous))
        if not isinstance(memory, dict) or len(json.dumps(memory, allow_nan=False).encode()) > 8192:
            raise ValueError('memory limit')
        previous = observation['tick']
    values = semantic_features(deepcopy(memory), deepcopy(query['history'][-1]['observation']), query['action'])
    if set(values) != set(NAMES): raise ValueError('semantic schema')
    row = [values[k] for k in NAMES]
    if any(v is not None and (type(v) not in (int,float,bool) or not math.isfinite(v)) for v in row):
        raise ValueError('finite scalar or unknown features required')
    answers.append(row)
print(json.dumps(dict(names=list(NAMES), values=answers), allow_nan=False))
'''


def evaluate(source, queries):
    if not isinstance(source, str) or len(source.encode()) > 32768:
        raise ValueError('semantic source at most 32 KiB')
    if not 1 <= len(queries) <= 256:
        raise ValueError('semantic batch 1..256')
    safe = []
    for h, a in queries:
        if not h.events: raise ValueError('history required')
        validate_action(a, h.events[-1]['observation'])
        safe.append(dict(history=[dict(observation=public_observation(e['observation']), action=e['action'])
                                  for e in h.events], action=a))
    with tempfile.TemporaryDirectory(prefix='coimagine_semantic_') as d:
        path = Path(d)/'policy.py'; path.write_text(source+'\n'+WRAPPER)
        result=None;cache={}
        for query in safe:
            # Each process sees ONE causal history and candidate, never the
            # positive/negative batch order, labels, or another query's future.
            key=digest(query)
            if key not in cache:
                with (Path(d)/'stdout').open('w+b') as out, (Path(d)/'stderr').open('w+b') as err:
                    proc = subprocess.run(policy_command(path), input=canonical([query]).encode(),
                                          stdout=out, stderr=err, timeout=30)
                    out.seek(0); err.seek(0)
                    output, errors = out.read(1048576), err.read(2000)
                if proc.returncode:
                    raise RuntimeError('isolated semantic worker failed: '+errors.decode(errors='replace'))
                cache[key]=json.loads(output)
            single=cache[key]
            if result is None:result=dict(names=single['names'],values=[])
            if single['names']!=result['names'] or len(single['values'])!=1:
                raise ValueError('feature schema changed across queries')
            result['values'].append(single['values'][0])
    names, values = result['names'], result['values']
    if (not isinstance(names, list) or not 1 <= len(names) <= 16 or len(set(names)) != len(names)
            or not all(isinstance(n, str) and 0 < len(n) <= 100 for n in names)
            or len(values) != len(queries)):
        raise ValueError('worker feature schema')
    for row in values:
        if len(row) != len(names) or any(v is not None and
                (type(v) not in (int,float,bool) or not np.isfinite(v)) for v in row):
            raise ValueError('worker numeric output')
    return result


EMBEDDING_NAMES = ['duration_s', 'goal_target', 'push_x', 'push_y', 'move_x', 'move_y',
                   'visible_points', 'target_dx', 'target_dy', 'goal_dx', 'goal_dy',
                   'target_seen_age_s', 'target_motion_x', 'target_motion_y',
                   'robot0_vx', 'robot0_vy', 'robot1_vx', 'robot1_vy']


def embedding(history, action):
    """Fixed geometry/history summary, NOT a learned/full-history embedding."""
    import math
    o = history.events[-1]['observation']; label = action['target_id']
    samples=[]
    for e in history.events:
        for obj in e['observation']['entities']:
            if obj['id']==label and obj['points']:
                p=np.mean(obj['points'],axis=0)
                samples.append((e['observation']['tick'],p,len(obj['points'])))
    origin=np.mean([[r['own']['x'],r['own']['y']] for r in o['robots']],axis=0)
    p=samples[-1][1] if samples else None
    vectors=[]
    for kind in ('push','move'):
        v=np.zeros(2)
        for a in (action['a0'],action['a1']):
            start=0 if kind=='push' else 47
            if start<=a<start+32:
                direction,speed=divmod(a-start,4); angle=direction*math.pi/4
                v+=.1*(speed+1)*np.array([math.cos(angle),math.sin(angle)])
        vectors.extend(v.tolist())
    delta=[None,None] if p is None else (p-origin).tolist()
    goal=[None,None] if p is None else (np.asarray(o['goal']['xy'])-p).tolist()
    movement=[None,None] if len(samples)<2 else (samples[-1][1]-samples[0][1]).tolist()
    return [action['ticks']*.05, float(label==o['goal']['target_id']), *vectors,
            0 if not samples else samples[-1][2], *delta, *goal,
            None if not samples else (o['tick']-samples[-1][0])*.05,*movement,
            *[float(r['own'][k]) for r in o['robots'] for k in ('vx','vy')]]


def fit_models(rows, source):
    """Same verified physical pairs for S/E/M; never fit validation/test rows."""
    from .dataset import verify
    selected=[];groups={}
    for row in rows:
        h=verify(row)
        if groups.setdefault(row['group'],row['split'])!=row['split']:
            raise ValueError('mother group crosses data splits')
        if row['comparison']['status']=='preference':
            c=row['comparison']
            selected.append((row,h,c['preferred']['action'],c['other']['action']))
    if not selected or not any(r['split']=='train' for r,_,_,_ in selected):
        raise ValueError('no comparable physical training preferences; do not invent negatives')
    if len(selected)>128:raise ValueError('fit batch capped at 128 pairs')
    queries=[(h,a) for _,h,a,b in selected]+[(h,b) for _,h,a,b in selected]
    semantic=evaluate(source,queries); n=len(selected)
    s=semantic['values']; e=[embedding(h,a) for h,a in queries]
    train=[i for i,(r,*_) in enumerate(selected) if r['split']=='train']
    result=dict(version='relational-fit-v1',dataset_sha256=digest(rows),semantic_sha256=digest(source),
                semantic_names=semantic['names'],embedding_names=EMBEDDING_NAMES,
                embedding_note='fixed history/geometry summary; not a learned complete-history encoder',
                pair_count=len(selected),training_pairs=len(train),models={})
    result['mother_groups_by_split']={split:sorted(g for g,s in groups.items() if s==split)
                                     for split in ('train','validation','test')}
    result['origins_by_split']={split:len({h.sha256 for r,h,_,_ in selected if r['split']==split})
                               for split in ('train','validation','test')}
    sm,sl=fit([s[i] for i in train],[s[n+i] for i in train])
    em,el=fit([e[i] for i in train],[e[n+i] for i in train])
    offsets=sm.score([s[i] for i in train])-sm.score([s[n+i] for i in train])
    rm,rl=fit([e[i] for i in train],[e[n+i] for i in train],offsets=offsets,residual_penalty=.05)
    for name,model,loss,features in [('S',sm,sl,s),('E',em,el,e),('M',rm,rl,e)]:
        pred=model.score(features); residual=pred[:n]-pred[n:]
        base=sm.score(s) if name=='M' else np.zeros(2*n)
        scores=pred+base; margins=scores[:n]-scores[n:]
        result['models'][name]=dict(checkpoint=model.record(),loss=loss,
            semantic_checkpoint=sm.record() if name=='M' else None,
            residual_penalty=.05 if name=='M' else 0.,
            pairs=[dict(group=r['group'],split=r['split'],origin=h.sha256,
                        preferred=a,other=b,preferred_score=float(scores[i]),other_score=float(scores[n+i]),
                        margin=float(margins[i]),residual_margin=float(residual[i]) if name=='M' else None)
                   for i,(r,h,a,b) in enumerate(selected)],
            metrics={split:dict(count=sum(r['split']==split for r,*_ in selected),
                       correct=int(sum(r['split']==split and margins[i]>1e-8 for i,(r,*_) in enumerate(selected))),
                       ties=int(sum(r['split']==split and abs(margins[i])<=1e-8 for i,(r,*_) in enumerate(selected))))
                     for split in ('train','validation','test')})
    return result


def score_candidates(source, fitted, history, actions, condition='M'):
    """One shared scalar scorer for a variable-size entity-bound candidate batch."""
    if digest(source)!=fitted['semantic_sha256']:raise ValueError('semantic version differs from fit')
    if condition not in ('S','E','M'):raise ValueError('condition')
    for a in actions:validate_action(a,history.events[-1]['observation'])
    model=fitted['models'][condition]
    x=(evaluate(source,[(history,a) for a in actions])['values'] if condition=='S'
       else [embedding(history,a) for a in actions])
    scores=Ranker.restore(model['checkpoint']).score(x)
    if condition=='M':
        s=evaluate(source,[(history,a) for a in actions])['values']
        scores+=Ranker.restore(model['semantic_checkpoint']).score(s)
    return scores.tolist()
