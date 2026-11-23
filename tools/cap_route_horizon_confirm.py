"""Fit only on development and deploy the locked selector on new rooms."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
import numpy as np

from tools import cap_route_horizon as exp
from tools import cap_route_horizon_report as report

study=exp.study
OUT=exp.OUT/'confirmation'


def freeze():
    path=OUT/'panel.json'
    if path.exists():return study.read(path)
    known=set()
    for f in [study.OUT/'development.json',study.OUT/'held.json',study.OUT/'selective_semantics/held.json',
              study.OUT/'initial_confirmation/held.json',study.OUT/'pacing/rooms.json',study.phase.OUT/'held.json']:
        db=study.read(f)
        known.update(study.phase.input_hash(v) for v in db.values() if isinstance(v,dict) and 'scene_id' in v)
    known.update(x['content_sha256'] for x in study.read(study.phase.cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json'))
    db={}
    for r in study.phase.cap.rooms.generate(662315841)['cases']:
        identity=study.phase.input_hash(r['fixture'])
        assert identity not in known
        known.add(identity)
        db['delay_confirm_'+r['id']]=r['fixture']
    study.save(path,db)
    return db


def choose(plan):
    dev=[study.read(exp.OUT/'runs/development'/(r+'.json')) for r in plan['rooms']['development']]
    opts=[]
    for h in exp.HORIZONS:
        a,b,x,mask=report.data(dev,h)
        cv,_=report.estimate(dev,dev,h)
        for t in report.THRESHOLDS:
            use=cv>t
            opts.append((int(np.where(use,b,a).sum()),-int(use.sum()),-h,t))
    wins,n,nh,t=max(opts)
    h=-nh
    a,b,x,mask=report.data(dev,h)
    x=x[mask]
    y=(b-a)[mask]
    mean=x.mean(0)
    sd=np.maximum(.1,x.std(0))
    centered=(x-mean)/sd
    w=np.linalg.solve(centered.T@centered+len(x)*.15*np.eye(x.shape[1]),centered.T@(y-y.mean()))
    assert np.allclose((x-mean)/sd@w+y.mean(),report.ridge(x,y,x))
    return dict(h=h,t=t,cv_wins=wins,cv_switches=-n,w=w.tolist(),mean=mean.tolist(),sd=sd.tolist(),offset=float(y.mean()))


def job(item,pair,model):
    room,fixture=item
    path=OUT/'runs'/(room+'.json')
    if path.exists():return
    def decision(x):
        s=(np.array(x)-model['mean'])/model['sd']@model['w']+model['offset']
        return s>model['t']
    a=exp.rollout(fixture,pair,None)
    b=exp.rollout(fixture,pair,0)
    mixed=exp.rollout(fixture,pair,model['h'],decide=decision)
    cut=mixed['cut']
    assert (cut['history'] if cut else None)==a['checkpoints'].get(str(model['h']))
    switched=bool(cut and cut.get('switched'))
    if not switched:assert a['history']==mixed['history']
    if model['h']==0 and switched:assert b['history']==mixed['history']
    study.save(path,dict(room=room,A=a['success'],B=b['success'],mixed=mixed['success'],switched=switched,
                         tick=cut['tick'] if cut else None))


def main():
    db=freeze()
    if '--freeze' in sys.argv:return
    plan=study.read(exp.OUT/'protocol.json')
    model=choose(plan)
    mp=OUT/'locked_model.json'
    if mp.exists():assert study.read(mp)==model
    else:study.save(mp,model)
    study.initialize()
    with ProcessPoolExecutor(12) as pool:
        for _ in pool.map(partial(job,pair=plan['pair'],model=model),db.items()):pass
    rows=[study.read(OUT/'runs'/(r+'.json')) for r in db]
    a,b,m=[np.array([r[k] for r in rows]) for k in ('A','B','mixed')]
    rng=np.random.default_rng(662315841)
    size=np.array([len(f['movable_objects']) for f in db.values()])
    ix=np.concatenate([rng.choice(group,(10000,len(group))) for s in np.unique(size) if len(group:=np.flatnonzero(size==s))],axis=1)
    summary=dict(A=int(a.sum()),B=int(b.sum()),union=int(np.maximum(a,b).sum()),mixed=int(m.sum()),
                 switches=sum(r['switched'] for r in rows),vs_A=report.paired(m,a,ix),vs_B=report.paired(m,b,ix),
                 horizon=model['h'],threshold=model['t'],n=len(rows))
    study.save(exp.DOC/'confirmation.json',summary)
    print(summary)


if __name__=='__main__':
    main()
