"""Evaluate two independent CAP values on the same actual mid-episode state."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import argparse
import math
import numpy as np
import torch

from tools import cap_terminal_value_fit as fit
from tools import cap_route_horizon_report as scoring

work=fit.study
study=work.study
PAIR=('usual_s003','specialized_s002')
WHEN=(600,1200)
OUT=work.OUT/'handoff'


def rollout(job):
    target,room,fixture,shas=job
    dest=OUT/str(target)/(room+'.npz')
    if dest.exists():return
    policies=[study.phase.FrozenCAP(study.MODULES[s]) for s in shas]
    counts=Counter(); memory={}
    tried=False; chosen=False; available=False
    x=[];tick=-1;n=0;cut=-1;fallback=0
    with study.phase.world(fixture) as (w,_):
        def update(h):
            for policy in policies:policy.update(h)
        update(w.history)
        while not study.ended(w):
            obs=study.observe(w)
            if chosen:
                cmd=work.info.parent.try_action(policies[1],obs)
                if cmd is None:
                    cmd=work.info.parent.try_action(policies[0],obs) or dict(target_id=None,a0=44,a1=44,ticks=1)
                    fallback+=1
            else:
                cmd=policies[0].propose(obs)
                if not tried and obs['tick']>=target:
                    tried=True;tick=obs['tick'];cut=n
                    alternative=work.info.parent.try_action(policies[1],obs)
                    if alternative is not None:
                        # Both feature vectors use the historical observations accrued under A.
                        copy={**memory,'past':list(memory.get('past',[]))}
                        x=[work.view(obs,cmd,copy,counts),work.view(obs,alternative,memory,counts)]
                        cmd=alternative;chosen=available=True
                if not tried:work.view(obs,cmd,memory,counts)
            study.tick_action(w,cmd,work.EX,update,counts)
            n+=1
        res=study.output(w,counts)
        obs=study.observe(w)
        d=math.dist(w.env.env.backend.bodies[obs['goal']['target_id']].position,obs['goal']['xy'])
    dest.parent.mkdir(parents=True,exist_ok=True)
    tmp=dest.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,x=x,cut=cut,tick=tick,available=available,d=d,y=res['success'],history=res['history'],fallback=fallback)
    tmp.replace(dest)


def get_choice(y,a,base_d,alt_d,m,ix):
    routed=np.where(m,a,y)
    delta_d=np.where(m,alt_d,base_d)-base_d
    return dict(distance_delta_m=float(delta_d.mean()),distance_delta_ci95=np.quantile(delta_d[ix].mean(1),[.025,.975]).tolist(),
                success=int(routed.sum()),overrides=int(m.sum()),mean_terminal_m=float(np.where(m,alt_d,base_d).mean()),
                vs_A=scoring.paired(routed,y,ix))


def evaluate():
    pred={}
    for e in PAIR:pred[e]=fit.load_models(e)
    result={}
    for t in WHEN:
        rows=[]
        for p in sorted((OUT/str(t)).glob('*.npz')):
            with np.load(p) as z, np.load(work.path_for('fresh',PAIR[0],p.stem)) as ref:
                r=dict(room=p.stem,A=int(ref['y']),B=int(z['y']),dA=float(ref['d']),dB=float(z['d']),
                       possible=bool(z['available']),fallback=int(z['fallback']),sA=0.,sB=0.,pA=0.,pB=0.)
                if r['possible']:
                    assert np.array_equal(z['x'][0],ref['x'][int(z['cut'])])
                    for i,e in enumerate(PAIR):
                        q=fit.predict(pred[e],[dict(x=z['x'][i:i+1])])[0]
                        r[['sA','sB'][i]]=float(q['pred_d'][0]);r[['pA','pB'][i]]=float(q['pred_p'][0])
                else:assert z['history']==ref['final_hash'] and r['A']==r['B']
                rows.append(r)
        assert len(rows)==200
        y,b,yd,bd,sa,sb,pa,pb=[np.array([r[k] for r in rows]) for k in ('A','B','dA','dB','sA','sB','pA','pB')]
        ix=np.random.default_rng(126).integers(len(rows),size=(5000,len(rows)))
        result[t]=dict(rooms=len(rows),available=sum(r['possible'] for r in rows),fallback=sum(r['fallback'] for r in rows),
               fixed_A=int(y.sum()),force_B=int(b.sum()),A_mean_m=float(yd.mean()),B_mean_m=float(bd.mean()),
               hindsight_success=int(np.maximum(y,b).sum()),distance_choice=get_choice(y,b,yd,bd,sb<sa,ix),
               probability_choice=get_choice(y,b,yd,bd,pb>pa,ix))
        study.save(OUT/f'choices_{t}.json',rows)
    study.save(work.DOC/'handoff.json',result)
    print(result,flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--report',action='store_true');p.add_argument('--workers',type=int,default=5)
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.report:
        evaluate()
        return
    experts,_,fresh,_=work.panels()
    study.initialize()
    jobs=[(t,r,f,[experts[e] for e in PAIR]) for t in WHEN for r,f in fresh.items()]
    with ProcessPoolExecutor(args.workers) as pool:
        for i,_ in enumerate(pool.map(rollout,jobs),1):
            if i%100==0:print(i,flush=True)


if __name__=='__main__':main()
