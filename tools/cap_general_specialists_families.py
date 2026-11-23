"""Select cross-room failure families using only recorded public controller feedback."""
import collections
import gzip
import json
import math
import random

import numpy as np

from tools import cap_general_specialists as run

LABELS={
    'no_target':'Goal object has no current public point cloud',
    'paired_push':'Both controllers report push, with little observed progress toward goal',
    'navigation':'Move, yield, or preparation is blocked or aborted',
    'acquisition':'Seeking a face, preparing contact, or waiting for the other pusher',
    'idle':'At least one robot brakes or yields without visible progress',
    'other':'Other publicly observed stalled behavior',
}


def describe(directory,cid):
    with gzip.open(directory/(cid+'.json.gz'),'rt') as f:
        rows=json.load(f)
    votes=collections.Counter()
    example={}
    last=0
    for i,r in enumerate(rows):
        tick=r['tick'];duration=tick-last;last=tick
        if tick<100:continue
        # Compare against a preceding boundary at least 100 executed ticks earlier.
        j=i-1
        while j>0 and tick-rows[j]['tick']<100:j-=1
        if j<0 or tick-rows[j]['tick']<100:continue
        xy,old=r['xy'],rows[j]['xy']
        if xy is not None and old is not None:
            if math.dist(old,r['goal'])-math.dist(xy,r['goal'])>.06:continue
        stages=[str(bot[2] or '').lower() for bot in r['robots']]
        reasons=[str(bot[3] or '').lower() for bot in r['robots']]
        flags=' '.join(stages+reasons)
        if xy is None:key='no_target'
        elif all(k=='push' for k in stages):key='paired_push'
        elif any(w in flags for w in ('blocked','stalled','unavailable')):key='navigation'
        elif any(w in flags for w in ('prepare','standoff','final','align','find','wait_peer','approach')):key='acquisition'
        elif any(w in flags for w in ('yield','brake','idle')):key='idle'
        else:key='other'
        votes[key]+=duration
        if key not in example:example[key]=dict(boundary=i,tick=tick,stages=stages,reason=reasons)
    key=max(votes,key=votes.get) if votes else 'other'
    return dict(label=key,ticks=dict(votes),example=example.get(key,{}))


def construct(s):
    base=s['rounds'][0]['sid']
    dev=run.read(run.OUT/'development.json')
    directory=run.OUT/'runs'/base
    parents={}
    for k in sorted(dev):
        r=run.read(directory/(k+'.json'))
        parents[k]=dict(success=bool(r['success']),**describe(directory,k))
    extra={}
    mapping={}
    forbidden={run.input_hash(f) for f in [*dev.values(),*run.read(run.OUT/'held.json').values()]}
    for i,(k,r) in enumerate(parents.items()):
        if r['success']:continue
        _,n,j=k.split('_')
        for v in range(2):
            key=f'aug_{int(n)+(v+1)*10}_{j}'
            f=run.base.variant(dict(case_id=k,fixture=dev[k]),run.SEED*10+2*i+v)
            h=run.input_hash(f)
            assert h not in forbidden
            forbidden.add(h)
            extra[key]=f;mapping[key]=k
    run.save(run.OUT/'extra.json',extra)
    run.save(run.OUT/'extra_parents.json',mapping)
    records=run.measure(base,extra) if extra else []
    if any(not r['complete'] for r in records):raise RuntimeError('incomplete generalist expansion')
    variants={r['id']:dict(success=bool(r['success']),parent=mapping[r['id']],**describe(directory,r['id'])) for r in records}
    run.save(run.OUT/'families.json',dict(parents=parents,variants=variants,labels=LABELS))


def choose(s):
    groups=run.read(run.OUT/'families.json')
    parents=groups['parents'];variants=groups['variants']
    union={k for r in s['rounds'] for k in r['wins']}
    seen={k for r in s['rounds'] for k in r['roots']}
    by=collections.defaultdict(list)
    for k,r in parents.items():
        if not r['success'] and k not in union|seen:by[r['label']].append(k)
    by={k:v for k,v in by.items() if len(v)>=2}
    if not by:return None
    topic=max(sorted(by),key=lambda c:len(by[c]))
    used_topics={r['topic'] for r in s['rounds']}
    untouched={k:v for k,v in by.items() if k not in used_topics}
    if untouched:topic=max(sorted(untouched),key=lambda c:len(untouched[c]))
    rng=random.Random(run.SEED+len(s['rounds']))
    roots=rng.sample(sorted(by[topic]),min(8,len(by[topic])))
    samples=list(roots)
    for parent in roots:
        choices=[k for k,v in variants.items() if v['parent']==parent and not v['success'] and v['label']==topic]
        if choices:samples.append(rng.choice(sorted(choices)))
    dev=run.read(run.OUT/'development.json')
    geom=run.read(run.base.OUT/'regions.json')
    mat=np.array(geom['x']); mat=(mat-mat.mean(0))/np.maximum(.1,mat.std(0));ix={k:i for i,k in enumerate(geom['ids'])}
    candidates=[k for k,r in parents.items() if r['success']]
    def control_key(k):
        return (parents[k]['label']!=topic,min(np.linalg.norm(mat[ix[k]]-mat[ix[j]]) for j in roots),k)
    controls=sorted(candidates,key=control_key)[:4]
    samples+=controls
    evidence={k:(parents.get(k) or variants[k]) for k in samples}
    run.save(run.WORK/'family_brief.json',dict(topic=topic,meaning=LABELS[topic],parent_failures=roots,
        similar_original_successes=controls,initials=samples,baseline=evidence,
        interpretation='stall is public point-cloud centroid progress proxy. Stage tags are development group labels; inspect the actual boundaries. All task scores are real terminal outcomes.'))
    run.save(run.WORK/'family_initial_public.json',{k:run.base.public(f) for k,f in run.read(run.OUT/'extra.json').items() if k in samples})
    return samples,topic,roots
