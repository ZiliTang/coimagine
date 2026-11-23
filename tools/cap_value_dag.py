"""Finite-horizon empirical CAP backups with room-paired terminal supervision."""
import argparse
from collections import defaultdict
import hashlib
import pickle
from pathlib import Path

import numpy as np

from tools import cap_topology_graph_report as graph
from tools import cap_value_dag_data as data

ROUTES=data.g.ROUTES
MODES=('budget','physical','physical_history','public','public_history')
OUT,DOC=data.OUT,data.DOC
read,save=data.read,data.save
ALPHA=8


def variance(m):
    return np.maximum(0,m[...,1]-m[...,0]**2)


def terminal(b):
    d=b['gap']
    return np.array([d,d*d,b['success'],d>1.,d<=.2,0.,0.],dtype=float)


def dataset(fold):
    paths=sorted((OUT/'data'/fold).glob('*.json'))
    assert len(paths)==200
    rooms=[read(p) for p in paths]
    centres,predict=graph.frozen_model()
    points=graph.extract(rooms)
    pts=list(points.values())
    _,public,_=predict(pts)
    physical=graph.nearest(graph.matrix(pts,'y'),centres)
    meta={}
    for (k,s),a,b in zip(points.items(),physical,public):
        t=(1800-s['tick']-1)//300
        n=(300-s['decisions']-1)//50
        assert t>=0 and n>=0
        x=s['x']
        progress='unseen' if not x[23] else ('toward' if x[24]>=.02 else 'away' if x[24]<=-.02 else 'small')
        meta[k]=(int(a),int(b),int(t),int(n),progress)
    cases=[]
    for r in rooms:
        origin='AB'[r['origin']]
        for rec in r['records']:
            paths={}
            for route,br in rec['branches'].items():
                seq=br['states']
                states=[meta[r['room'],h]+((origin+route[:i])[-2:],) for i,h in enumerate(seq)]
                vals=[rec['points'][h] for h in seq]
                paths[route]=dict(states=states,y=terminal(br),hashes=seq,
                                  positions=[s['gap'] for s in vals],ticks=[s['tick'] for s in vals])
            cases.append(dict(room=r['room'],depth=rec['depth'],branches=paths))
    return cases


def keys(s,action,mode):
    phys,pub,t,n,progress,via=s
    levels=[(action,)]
    if mode=='constant':return levels
    p=levels[-1]+(t,n)
    levels.append(p)
    if mode=='budget':return levels
    p=p+(phys if mode.startswith('physical') else pub,)
    levels.append(p)
    if mode.endswith('_history'):
        levels.extend((p+(via,),p+(via,progress)))
    return levels


def add(table,s,action,mode,y):
    for k in keys(s,action,mode):
        if k not in table:table[k]=[0,np.zeros_like(y)]
        table[k][0]+=1
        table[k][1]+=y


def lookup(table,s,action,mode):
    levels=keys(s,action,mode)
    n,v=table[levels[0]]
    m=v/n
    for k in levels[1:]:
        if k in table:
            n,v=table[k]
            m=(v+ALPHA*m)/(n+ALPHA)
    return m.copy()


def backups(cases,mode):
    # Keep branches balanced; each of the four predeclared continuations has one vote per reachable prefix.
    tables={key:{} for key in ('leaf','mid','root','direct_mid','one','direct')}
    paths=[(route,br['states'],br['y']) for c in cases for route,br in c['branches'].items() if br['states']]
    for route,s,y in paths:
        add(tables['direct'],s[0],route,mode,y)
        if len(s)>=2:add(tables['direct_mid'],s[1],route[1],mode,y)
        if len(s)==3:add(tables['leaf'],s[2],route[1],mode,y)
    for route,s,y in paths:
        if len(s)<2:continue
        q=y.copy()
        if len(s)==3:
            q=lookup(tables['leaf'],s[2],route[1],mode)
            q[5]=variance(q)
        add(tables['mid'],s[1],route[1],mode,q)
    for route,s,y in paths:
        if len(s)>=2:
            q=lookup(tables['mid'],s[1],route[1],mode)
            q[6]=variance(q)
            single=lookup(tables['direct_mid'],s[1],route[1],mode)
        else:q=single=y
        add(tables['root'],s[0],route,mode,q)
        add(tables['one'],s[0],route,mode,single)
    return tables


def freeze():
    cases=dataset('dev')
    book={m:backups(cases,m) for m in ('constant',*MODES)}
    y=np.array([[c['branches'][r]['y'] for r in ROUTES] for c in cases])
    ref=dict(success=ROUTES[y[:,:,2].mean(0).argmax()],distance=ROUTES[y[:,:,0].mean(0).argmin()])
    raw=pickle.dumps(book,protocol=5)
    (OUT/'tables.pkl').write_bytes(raw)
    save(OUT/'frozen.json',dict(alpha=ALPHA,modes=MODES,reference=ref,rooms=200,cases=len(cases),
                                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                model_sha256=hashlib.sha256(raw).hexdigest(),
                                graph_model=read(data.g.OUT/'dev_freeze.json')['model_sha256']))


def forecasts(book,mode,route,b):
    s,y=b['states'],b['y']
    if not s:return {}
    t=book[mode]
    result={k:lookup(t[k],s[0],route,mode) for k in ('root','one','direct')}
    result['observed_mid']=lookup(t['mid'],s[1],route[1],mode) if len(s)>=2 else y.copy()
    result['observed_leaf']=lookup(t['leaf'],s[2],route[1],mode) if len(s)==3 else y.copy()
    return result


def evaluate(fold):
    raw=(OUT/'tables.pkl').read_bytes()
    frozen=read(OUT/'frozen.json')
    assert hashlib.sha256(raw).hexdigest()==frozen['model_sha256']
    book=pickle.loads(raw)
    cases=dataset(fold)
    names=['constant/direct','budget/root']
    for mode in MODES[1:]:
        names+= [mode+'/'+stage for stage in ('root','one','direct','observed_mid','observed_leaf')]
    values={key:[] for key in names}
    for case in cases:
        local=defaultdict(list)
        for route in ROUTES:
            b=case['branches'][route]
            cache={}
            for key in names:
                mode,stage=key.split('/')
                if mode not in cache:cache[mode]=forecasts(book,mode,route,b)
                m=cache[mode].get(stage,np.full(7,np.nan))
                if b['states']:
                    assert m[1]+1e-8>=m[0]**2 and (m[2:5]>=-1e-9).all() and (m[2:5]<=1+1e-9).all()
                    if stage=='root':
                        assert m[5]<=m[6]+1e-8 and m[6]<=variance(m)+1e-8
                local[key].append(m)
        for key,v in local.items():values[key].append(v)
    return cases,{k:np.array(v) for k,v in values.items()}


if __name__=='__main__':
    cli=argparse.ArgumentParser()
    cli.add_argument('stage',choices=['freeze'])
    cli.parse_args()
    freeze()
