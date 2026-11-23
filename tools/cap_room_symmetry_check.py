"""Development-only check of real CAP outcomes under rectangle symmetries."""
import copy
from math import atan2,cos,sin
from pathlib import Path
from experiments.planar_tasks.relational_search.motion import ACTIONS,canonical_index
from tools import cap_union_trend as t

e=t.e
OUT=t.OUT/'symmetry'
TRANSFORMS={'original':(1,1),'halfturn':(-1,-1),'mirror_x':(-1,1),'mirror_y':(1,-1)}


def transform(f,name):
    v=copy.deepcopy(f)
    sx,sy=TRANSFORMS[name]
    width,height=v['room']['arena']
    for item in [v['object'],*v['robots'],*v['movable_objects'],v['goal'],*v['room']['obstacles']]:
        x,y=('cx','cy') if 'cx' in item else ('x','y')
        if sx<0:item[x]=width-item[x]
        if sy<0:item[y]=height-item[y]
        if 'angle' in item and name!='original':
            a=item['angle']
            item['angle']=atan2(sy*sin(a),sx*cos(a))
    return v


def option(i,name):
    sx,sy=TRANSFORMS[name]
    if i<32 or i>=47:
        start=0 if i<32 else 47
        direction,speed=divmod(i-start,4)
        if sx<0:direction=(4-direction)%8
        if sy<0:direction=(-direction)%8
        return start+4*direction+speed
    if sx*sy<0 and 32<=i<44:
        return canonical_index(i+3 if i in (32,33,34,38,39,40) else i-3)
    return canonical_index(i)


def command(cmd,name):
    return dict(cmd,a0=option(cmd['a0'],name),a1=option(cmd['a1'],name))


def check(job):
    k,f=job
    path=OUT/'rooms'/(k+'.json')
    if path.exists():return
    row={}
    for name in TRANSFORMS:
        v=transform(f,name)
        with e.s.phase.world(v) as (w,_):
            obs=e.s.observe(w)
            heads={}
            for arm,mod in [('A',e.AM),('B',e.BM)]:
                cap=e.s.phase.FrozenCAP(mod)
                cap.update(w.history)
                heads[arm]=cap.propose(obs)
        if name=='original':
            outcomes=e.read(t.PARENT/'screen'/(k+'.json'))
        else:
            outcomes={arm:e.simulate(v,arm,'stall')['result']['success'] for arm in heads}
        row[name]=dict(outcomes=outcomes,first=heads)
        row[name]['aligned']={arm:command(row['original']['first'][arm],name)==command(heads[arm],'original') for arm in heads}
    e.save(path,row)


def main():
    t.d.init(profile=e.perception.LEGACY)
    for name in TRANSFORMS:
        assert all(option(option(i,name),name)==canonical_index(i) for i in range(len(ACTIONS)))
    dev=t.p.dev()
    keys=sorted(dev,key=lambda k:e.key('symmetry-check/'+k))[:32]
    e.pool(check,[(k,dev[k]) for k in keys],4)
    data=[e.read(OUT/'rooms'/(k+'.json')) for k in keys]
    out={}
    for name in TRANSFORMS:
        orig=[r['original']['outcomes'] for r in data]
        rot=[r[name]['outcomes'] for r in data]
        def summary(a,b):
            return dict(original=sum(a),transformed=sum(b),flip=sum(x!=y for x,y in zip(a,b)),
                        gained=sum(y>x for x,y in zip(a,b)),lost=sum(y<x for x,y in zip(a,b)))
        stats={arm:summary([r[arm] for r in orig],[r[arm] for r in rot]) for arm in ('A','B')}
        stats['union']=summary([max(r.values()) for r in orig],[max(r.values()) for r in rot])
        stats['first_agreement']={arm:sum(r[name]['aligned'][arm] for r in data) for arm in ('A','B')}
        out[name]=stats
    e.save(OUT/'summary.json',dict(parents=keys,transforms=out))


if __name__=='__main__':
    main()
