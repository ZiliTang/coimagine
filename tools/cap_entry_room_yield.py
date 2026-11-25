"""Measure training-room selection yield using only initial obstacle centres and other rooms' gate hits."""
from pathlib import Path

import numpy as np

from experiments.planar_tasks.relational_search.pipeline import read, save


def positions(fixtures, keys):
    coordinates=[]
    for key in keys:
        fixture=fixtures[key]
        static=fixture['room']['obstacles'][0]
        movable=fixture['movable_objects'][0]
        coordinates.append([static['cx'],static['cy'],movable['x'],movable['y']])
    return np.asarray(coordinates)


def candidates(x, labels, query):
    # Only initial layouts and past gate incidence; no test labels or trajectory outcomes.
    sd=x.std(axis=0)
    scores=[]
    for chunk in np.array_split(query,max(1,(len(query)+255)//256)):
        d=np.linalg.norm((chunk[:,None,:]-x)/sd,axis=-1)
        indices=np.argpartition(d,63,axis=1)[:,:64]
        weights=1/(np.take_along_axis(d,indices,axis=1)+.1)
        scores.extend((labels[indices]*weights).sum(axis=1)/weights.sum(axis=1))
    return np.argsort(-np.asarray(scores),kind='stable')[:len(query)//5]


def labels(out,split,keys):
    y=[]
    for key in keys:
        entries=read(out/'runs'/split/'zero'/(key+'.json'))['entries']
        if entries:
            rows=read(out/'labels'/split/(key+'.json'))['rows']
            if not rows: raise ValueError('missing paired opportunities')
            y.append([1,int(any(r['take']!=r['defer'] for r in rows))])
        else:
            y.append([0,0])
    return np.asarray(y)


def freeze(out):
    fixtures=read(out/'fixtures/train.json')
    keys=sorted(fixtures)
    x=positions(fixtures,keys)
    y=labels(out,'train',keys)
    held=read(out/'fixtures/held.json')
    ordered=sorted(held)
    picked=[ordered[j] for j in candidates(x,y[:,0],positions(held,ordered))]
    selected=[]
    for val in np.array_split(np.random.default_rng(17).permutation(len(x)),5):
        train=np.setdiff1d(np.arange(len(x)),val)
        selected.extend(val[candidates(x[train],y[train,0],x[val])])
    record=dict(method='fixed 64 nearest rooms on train-standardized 4 obstacle coordinates, inverse (distance + 0.1) weighting, top fifth by predicted natural gate incidence',
                held_ids=picked,train_cv=dict(all=len(x),triggered=int(y[:,0].sum()),discordant=int(y[:,1].sum()),
                selected=len(selected),selected_triggered=int(y[selected,0].sum()),selected_discordant=int(y[selected,1].sum())))
    path=out/'room_selector.json'
    if path.exists():
        if read(path)!=record:raise ValueError('changed initial-layout selector')
    else:save(path,record)
    return record


def evaluate(out):
    frozen=read(out/'room_selector.json')
    keys=sorted(read(out/'fixtures/held.json'))
    y=labels(out,'held',keys)
    ids=set(frozen['held_ids'])
    mask=np.asarray([k in ids for k in keys])
    return dict(training=frozen['train_cv'],held=dict(all=len(y),triggered=int(y[:,0].sum()),discordant=int(y[:,1].sum()),
                selected=int(mask.sum()),selected_triggered=int(y[mask,0].sum()),selected_discordant=int(y[mask,1].sum())))


if __name__=='__main__':
    from tools.cap_entry_residual_study import OUT
    print(freeze(OUT)['train_cv'])
