"""Read-only instrumentation of the frozen 42 identical failures and paired successful controls."""
import csv
from collections import Counter
from math import dist
from unittest.mock import patch
from experiments.planar_tasks.relational_search.environment import MultiBackend
from tools import cap_free_repair_round as r
from tools.cap_fault_curriculum_rooms import panel, SEEDS

DEST=r.OUT/'shared_failure_audit'
def selection():
    SEEDS['freerepair_confirm']=842105733
    fixture,strata=panel('freerepair_confirm',200)
    assert fixture==r.read(r.OUT/'fixtures.json')
    cases=[];controls=[]
    with (r.DOC/'rooms.csv').open() as f:
        for line in csv.DictReader(f):
            k=line.pop('room')
            if all(int(v)==1 for v in line.values()):controls.append(k)
            if any(int(v) for v in line.values()):continue
            traces=[r.read(r.OUT/'evaluation'/arm/(k+'.json')) for arm in line]
            if len({x['result']['history'] for x in traces})==1:
                assert all(x['boundaries']==traces[0]['boundaries'] for x in traces)
                cases.append(k)
    def layout(k):
        f=fixture[k]; a=f['room']['obstacles'][0]; b=f['movable_objects'][0]
        return [a['cx'],a['cy'],b['x'],b['y']]
    pairs={}
    for k in sorted(cases,key=lambda k:(sum(strata[u]['stratum']==strata[k]['stratum'] for u in controls),k)):
        opts=[u for u in controls if strata[u]['stratum']==strata[k]['stratum']]
        if opts:
            other=min(opts,key=lambda u:(dist(layout(k),layout(u)),u))
            controls.remove(other)
            pairs[k]=other
        else:pairs[k]=None
    return fixture,strata,pairs

def run(job):
    k,f,src=job
    host=r.host
    engine=r.prev.PREFIX.s
    tick_action=engine.tick_action
    step=MultiBackend.apply_and_step
    rows=[]; memory={}
    class Read:
        def boundary(self,o,m,a,pending):
            memory.clear()
            for key in ['compact','route','stale_ticks','tslow','avoid']:
                memory[key]=m.get(key)
            memory['slot_plan']=bool(m.get('slot',{}).get('plan'))
            memory['outer_plan']=bool(m.get('recovery',{}).get('command'))
            memory['seen']=any(e['id']==o['goal']['target_id'] and e['points'] and e['last_seen_tick']==o['tick'] for e in o['entities'])
        def checkpoint(self,*args):pass
    def traced(w,action,*args,**kw):
        backend=w.env.env.backend
        obj=backend.bodies['entity-a']
        def pose():return [*obj.position,obj.angle]
        def flags():
            cs=backend.contacts()
            return {c['pair'] for c in cs if c['pair'] not in ('object_wall','robot_object') or c['entity_id']=='entity-a'}
        start=w.env.total_periods
        row=dict(t=start,object=pose(),robot=[list(x.position) for x in backend.robot_bodies],**memory)
        counter=Counter()
        def spy(b,*aa,**kk):
            out=step(b,*aa,**kk)
            counter.update(flags());counter['samples']+=1
            return out
        with patch.object(MultiBackend,'apply_and_step',spy):
            res=tick_action(w,action,*args,**kw)
        row.update(dt=w.env.total_periods-start,end=pose(),contacts=dict(counter),robot_end=[list(x.position) for x in backend.robot_bodies])
        rows.append(row)
        return res
    with patch.object(engine,'tick_action',traced):
        _,out=host.episode((k,f,True,None,'boundary',False,host.enabled(src),src),Read())
    frozen=r.read(r.OUT/'evaluation/parent'/(k+'.json'))
    assert out['result']==frozen['result'] and out['boundaries']==frozen['boundaries']
    assert len(rows)==len(frozen['boundaries']) and all(z['t']==b['tick'] and z['dt']>0 for z,b in zip(rows,frozen['boundaries']))
    return k,dict(rows=rows,result=out['result'])
if __name__=='__main__':
    r.configure()
    src=r.host.seal('P')
    fixture,strata,pairs=selection()
    assert len(pairs)==42
    r.save(DEST/'groups.json',dict(pairs=pairs,strata={k:v['stratum'] for k,v in strata.items()}))
    keys=list(pairs)+[v for v in pairs.values() if v]
    r.host.old.collect(src,[(k,fixture[k],src) for k in keys],run,12,DEST/'replays')
