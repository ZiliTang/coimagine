"""Intervene only in the estimated target centre at the first divergent CAP decision."""
import copy
from collections import Counter
from pathlib import Path
from tools import cap_symmetry_mechanism as m

s,e=m.s,m.e


def boundary(f,mod,index):
    cap=e.s.phase.FrozenCAP(mod)
    with e.s.phase.world(f) as (w,_):
        cap.update(w.history)
        for i in range(index+1):
            o=e.s.observe(w)
            a=cap.propose(o)
            if i==index:return cap.memory,o,a
            e.s.tick_action(w,a,e.g.prep.prior.EXECUTOR,cap.update,Counter())


def case(row):
    k,arm,name=row['room'],row['arm'],row['name']
    path=m.OUT/'ablations'/f'{k}_{arm}_{name}.json'
    if path.exists():return
    f=ROOMS[k]
    mod=e.AM if arm=='A' else e.BM
    index=row['first']['i']
    mem,obs,act=boundary(f,mod,index)
    c,_,_=mod.geometry(obs,mem['c'])
    tm,to,ta=boundary(s.transform(f,name),mod,index)
    assert s.command(act,name)==row['first']['mapped'] and ta==row['first']['actual']
    # Execute an offline copy of the author source in Python with deferred Cython annotations.
    code=SOURCE[arm].replace('import cython','',1)
    ns={}
    exec(compile('from __future__ import annotations\n'+code,'frozen-cap-diagnostic','exec'),ns)
    def proposal():
        return ns['prepare_context'](copy.deepcopy(tm),copy.deepcopy(to))['proposals'][0]
    unmodified=proposal()
    assert unmodified==ta
    geometry=ns['geometry']
    tc,_,_=geometry(to,tm['c'])
    output=dict(room=k,arm=arm,name=name,boundary=index,actual=ta,expected=s.command(act,name))
    if c and tc:
        sx,sy=s.TRANSFORMS[name]
        xy=[c[0] if sx>0 else 6-c[0],c[1] if sy>0 else 4.8-c[1]]
        def centered(*args):
            val,fs,pts=geometry(*args)
            return ([*xy,*val[2:]] if val else val),fs,pts
        ns['geometry']=centered
        output.update(error=m.math.dist(xy,tc[:2]),corrected=proposal())
    ns['geometry']=geometry
    sx,sy=s.TRANSFORMS[name]
    base_mem=copy.deepcopy(tm)
    waypoint=mem.get('route')
    tm['route']=None if waypoint is None else (waypoint[0] if sx>0 else 6-waypoint[0], waypoint[1] if sy>0 else 4.8-waypoint[1])
    output['waypoint']=proposal()
    tm=base_mem
    def alternative(code_variant,target_obs):
        alt={}
        exec(compile('from __future__ import annotations\n'+code_variant,'frozen-cap-factor','exec'),alt)
        return alt['prepare_context'](copy.deepcopy(tm),copy.deepcopy(target_obs))['proposals'][0]
    newobs=copy.deepcopy(to)
    for entity,pairs in newobs['entity_legal_pairs'].items():
        refpairs=obs['entity_legal_pairs'].get(entity,[])
        ranks={tuple(s.option(v,name) for v in pair):j for j,pair in enumerate(refpairs)}
        pairs.sort(key=lambda pair:ranks.get(tuple(pair),len(ranks)))
    order=[s.option(4*i,name)//4 for i in range(8)]
    reordered=code.replace('for i,d in enumerate(DIR):',f'for i in {order}:\n        d=DIR[i]')
    output['enumeration']=alternative(reordered,newobs)
    output['pair_order']=alternative(code,newobs)
    output['move_order']=alternative(reordered,to)
    if sx*sy<0:
        output['joint_side']=alternative(code.replace('return lower,upper','return upper,lower'),to)
        output['stall_turn']=alternative(code.replace('turn=(.8,-.8,1.9,-1.9)[sector]','turn=-(.8,-.8,1.9,-1.9)[sector]'),to)
    e.save(path,output)


if __name__=='__main__':
    s.t.d.init(profile=e.perception.LEGACY)
    ROOMS=s.t.p.dev()
    SOURCE={'A':(e.WORK/'basis.py').read_text(),'B':Path(e.read(e.OUT/'member.json')['path']).read_text()}
    rows=[r for p in sorted((m.OUT/'rooms').glob('*.json')) for r in e.read(p) if not r['first']['drift_before_or_now']]
    assert len(rows)==37
    e.pool(case,rows,5)
