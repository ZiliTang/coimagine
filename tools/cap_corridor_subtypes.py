"""Match public transport-corridor evidence and audit geometry to frozen CAP outcomes."""
import csv
import gzip
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np

from coimagine.random_start.geometry import box_polygon, rotate
from experiments.planar_tasks.semantic_rl.corridor_semantics import (
    FEATURE_NAMES, COVERAGE_REQUIRED, blocker_type, corridor_semantics, sweep_bounds)
from tools import cap_s025_initial as source

run=source.run
read,save=source.read,source.save
ROOT=source.ROOT
OUT=ROOT/'corridor_subtypes_20261122'
DOC=source.DOC.parent/'2026-11-22_corridor_subtypes'
KINDS=('clear','static','movable','both')
MIN_DEV=12


def audit_type(f):
    pose=f['object']
    centre=np.array([pose['x'],pose['y']])
    target=np.array([rotate(p,pose['angle']) for p in ((-.125,-.5),(.125,-.5),(.125,.5),(-.125,.5))])+centre
    goal=np.array([f['goal']['x'],f['goal']['y']])
    vertices,axes=sweep_bounds(target,goal-centre)
    def intersects(poly):
        edges=np.array([poly[1]-poly[0],poly[2]-poly[1]])
        normals=np.vstack([axes,edges[:,[1,0]]*[-1.,1.]])
        a,b=vertices@normals.T,poly@normals.T
        return bool(((a.max(0)>=b.min(0)-1e-10)&(b.max(0)>=a.min(0)-1e-10)).all())
    static=any(intersects(np.array(box_polygon(b['cx'],b['cy'],b['angle'],b['half_w'],b['half_h'])))
               for b in f['room']['obstacles'])
    movable=any(intersects(np.array(box_polygon(b['x'],b['y'],b['angle'],b['width']/2,b['height']/2)))
                for b in f['movable_objects'])
    return blocker_type(static,movable)


def measure(fixture):
    with run.runtime.world(fixture) as (env,_):
        obs=run.ref.dq.current_event(env.history)['observation']
        assert obs['tick']==0
        return audit_type(fixture),corridor_semantics(obs),source.geo.geom(obs).tolist()


def selector(rows,key,caps,levels):
    result={}
    for kind in levels:
        subset=[r for r in rows if r[key]==kind]
        result[kind]=(max(caps,key=lambda c:sum(r[c] for r in subset))
                      if kind!='unknown' and len(subset)>=MIN_DEV else 's025')
    return result


def stats(rows,key,caps,levels):
    result={}
    for kind in levels:
        group=[r for r in rows if r[key]==kind]
        result[kind]=dict(n=len(group),wins={c:sum(r[c] for r in group) for c in caps},
                          union=sum(any(r[c] for c in caps) for r in group))
    return result


def score(rows,key,mapping,reference):
    cap=[mapping[r[key]] for r in rows]
    obj=np.array([r['objects'] for r in rows])
    y=np.array([r[c] for r,c in zip(rows,cap)])
    return run.ref.scale.paired(np.array([r[reference] for r in rows]),y,obj)


def replay(key):
    f=read(ROOT/'held.json')[key]
    with gzip.open(ROOT/'runs/s025'/(key+'.json.gz'),'rt') as file: path=json.load(file)
    seq=[]
    with run.runtime.world(f) as (env,_):
        for step in path:
            obs=run.ref.dq.current_event(env.history)['observation']
            seq.append(dict(room=key,tick=obs['tick'],**corridor_semantics(obs)))
            run.runtime.advance(env,step['a'])
            assert env.env.total_periods==step['tick']
        assert env.history.sha256==read(ROOT/'runs/s025'/(key+'.json'))['history']
    return seq


def write_csv(path,rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def main():
    records,fixtures=source.inventory()
    source.folds(records)
    scale=run.ref.scale
    caps=read(scale.OUT/'protocol.json')['sids']
    assert caps[0]=='s025'
    archive=run.ref.sets.unzip(scale.OUT/'new_outcomes.json.gz')
    OUT.mkdir(exist_ok=True); DOC.mkdir(exist_ok=True,parents=True)
    design=dict(dev='388 natural parents',sensitivity='800 including variants',min_dev=MIN_DEV,
                primary='fixed-orientation translation swept polygon, internal static boxes and other movable boxes',
                coverage=COVERAGE_REQUIRED,tags='goal wall <=1m; >=3 other movable boxes',
                held='200 original with five outcomes, 200 current with s025 only',
                fallback='s025 for unknown or insufficient support',tie_order=caps,
                panels=read(source.OUT/'protocol.json')['manifests'])
    if (OUT/'protocol.json').exists(): assert read(OUT/'protocol.json')==design
    save(OUT/'protocol.json',design)
    with ProcessPoolExecutor(10,mp_context=mp.get_context('fork')) as workers:
        measured=list(workers.map(measure,fixtures,chunksize=6))
    aux=[]; rows=[]
    for r,(truth,features,compact) in zip(records,measured):
        k=r['room']
        item=dict(room=k,panel=r['panel'],parent=r['parent'],variant=r['variant'],objects=r['objects'],
                  truth=truth,observed=features['kind'],**{n:features[n] for n in FEATURE_NAMES})
        item['wall']='near' if item['goal_wall_distance_m']<=1. else 'interior'
        item['clutter']='many' if item['objects']>=4 else 'few'
        for cap in caps:
            if r['panel']=='current':
                item[cap]=r['y'] if cap=='s025' else None
            else:
                result=archive[k][cap] if k in archive else read(ROOT/'runs'/cap/(k+'.json'))
                assert result['complete'] and not result['error'] and result['id']==k
                item[cap]=int(result['success'])
        assert item['s025']==r['y']
        aux.append(compact); rows.append(item)
    dev=[r for r in rows if r['panel']=='development' and not r['variant']]
    devall=[r for r in rows if r['panel']=='development']
    held=[r for r in rows if r['panel']=='original']
    both=[r for r in rows if r['panel']!='development']
    best=max(caps,key=lambda c:sum(r[c] for r in dev))
    result=dict(best=best,types={},routes={},recognition={})
    for key,levels in [('truth',KINDS),('observed',(*KINDS,'unknown')),('wall',('near','interior')),('clutter',('few','many'))]:
        rule=selector(dev,key,caps,levels)
        alternate=selector(devall,key,caps,levels)
        result['types'][key]={split:stats(group,key,caps,levels) for split,group in [('dev',dev),('all_dev',devall),('held',held)]}
        result['routes'][key]=dict(rule=rule,dev=score(dev,key,rule,'s025'),held=score(held,key,rule,'s025'),
                                    held_vs_devbest=score(held,key,rule,best),all_dev_rule=alternate,
                                    all_dev_held=score(held,key,alternate,'s025'))
        if key=='truth':
            mapped=dict(rule,unknown='s025')
            result['routes']['truth_map_on_public']=score(held,'observed',mapped,'s025')
        for r in rows:
            r[key+'_choice']=rule[r[key]]
    for name,group in [('original',held),('current',[r for r in both if r['panel']=='current']),('combined',both)]:
        result['recognition'][name]={t:dict(Counter(r['observed'] for r in group if r['truth']==t)) for t in KINDS}
    write_csv(DOC/'rooms.csv',rows)
    np.savez(OUT/'compact_initial.npz',x=np.array(aux,dtype=np.float32))
    keys=[f'new_dev_{n}_{j:02d}' for n in range(2,6) for j in (0,1)]
    with ProcessPoolExecutor(8,mp_context=mp.get_context('fork')) as workers:
        traces=list(workers.map(replay,keys))
    write_csv(DOC/'boundaries.csv',[r for seq in traces for r in seq])
    result['replay']=dict(rooms=len(keys),boundaries=sum(map(len,traces)),
                          changed=sum(len({r['kind'] for r in seq})>1 for seq in traces))
    save(OUT/'FINISHED.json',result)


if __name__=='__main__':main()
