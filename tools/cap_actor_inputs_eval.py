"""Paired closed loops and factorial contrasts on the frozen 200-room panel."""
import csv
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

from tools import cap_actor_inputs as exp
from tools import cap_actor_inputs_fit as fit

MODELS={}
PANELS={}


@torch.inference_mode()
def episode(job):
    arm,key=job
    fixture=PANELS[key]
    disagreements=multi=ties=0
    with exp.run.runtime.world(fixture) as (env,lib):
        while not exp.run.runtime.ended(env):
            packet=lib.panel(env.history)
            row,actions=exp.features(lib,packet)
            b={k:torch.from_numpy(row[k][None]) for k in set(fit.fields(arm))|{'s','valid'}}
            logits=fit.scores(arm,MODELS[arm],b)[0].numpy()
            assert np.isfinite(logits[:len(actions)]).all()
            pick=int(np.argmax(logits))
            if len(actions)>1:
                multi+=1
                ties+=int((logits[:len(actions)]==logits[pick]).sum()>1)
                disagreements+=int(pick!=fit.scores('priority',None,b)[0].argmax().item())
            exp.run.runtime.advance(env,actions[pick])
        return dict(room=key,arm=arm,success=exp.run.terminal(env),history=env.history.sha256,
                    ticks=env.env.total_periods,decisions=env.decision_count,multi=multi,ties=ties,
                    disagreed_priority=disagreements)


def report():
    rooms=sorted(PANELS)
    old=exp.prior.OUT
    names=list(exp.ARMS)
    rows=[]
    for k in rooms:
        r=dict(room=k,objects=int(k.split('_')[1]))
        for arm in ('plain800','full_ln',*names):
            if arm=='plain800': path=old/'norm_comparison'/arm/(k+'.json')
            elif arm=='full_ln': path=old/'episodes'/'actor'/(k+'.json')
            else: path=exp.OUT/'evaluation'/arm/(k+'.json')
            item=exp.read(path)
            assert item['room']==k
            r[arm]=item['success']
        rows.append(r)
    def values(k): return np.array([r[k] for r in rows])
    def pair(a,b): return exp.run.ref.scale.paired(values(a),values(b),values('objects'))
    contrasts={f'{a}_to_{b}':pair(a,b) for a,b in [('action','geometry_action'),('action_ln','geometry_action_ln'),
                ('action','action_ln'),('geometry_action','geometry_action_ln'),('priority','router')]
               +[('plain800',arm) for arm in names]+[('priority',arm) for arm in names if arm not in ('priority','router')]}
    delta=values('geometry_action_ln')-values('action_ln')-values('geometry_action')+values('action')
    rng=np.random.default_rng(17)
    draws=np.zeros(10000)
    for n in range(2,6):
        v=delta[values('objects')==n]
        draws+=rng.choice(v,size=(len(draws),len(v))).sum(1)
    interaction=dict(pp=100*delta.mean(),ci=(np.quantile(draws,[.025,.975])/2).tolist())
    exp.DOC.mkdir(parents=True,exist_ok=True)
    with (exp.DOC/'per_room.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    result=dict(n=len(rows),contrasts=contrasts,interaction=interaction,arms={})
    for arm in names:
        records=[exp.read(exp.OUT/'evaluation'/arm/(k+'.json')) for k in rooms]
        result['arms'][arm]=dict(success=int(values(arm).sum()),fit=exp.read(exp.OUT/(arm+'_fit.json')),
            **{field:sum(r[field] for r in records) for field in ('multi','ties','disagreed_priority')})
    exp.save(exp.OUT/'FINISHED.json',result)


def main():
    exp.init()
    PANELS.update(exp.read(exp.prior.OUT/'panel.json'))
    assert len(PANELS)==200 and exp.run.campaign.base.digest(PANELS)==exp.read(exp.OUT/'protocol.json')['current_panel']
    done={}
    for arm in exp.ARMS:
        dir=exp.OUT/'evaluation'/arm
        dir.mkdir(parents=True,exist_ok=True)
        done[arm]=sum((dir/(k+'.json')).exists() for k in PANELS)
    while any(n<len(PANELS) for n in done.values()):
        ready=[arm for arm in exp.ARMS if done[arm]<len(PANELS) and (arm=='priority' or (exp.OUT/(arm+'_fit.json')).exists())]
        if not ready:
            time.sleep(3)
            continue
        jobs=[]
        for arm in ready:
            MODELS[arm]=None if arm=='priority' else fit.Scorer(arm)
            if arm!='priority':
                MODELS[arm].load_state_dict(torch.load(exp.OUT/(arm+'.pt'),weights_only=True))
                MODELS[arm].eval()
        for k in sorted(PANELS,key=lambda k:(k[-3:],k)):
            for arm in ready:
                if not (exp.OUT/'evaluation'/arm/(k+'.json')).exists():
                    jobs.append((arm,k))
        with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
            for f in as_completed([pool.submit(episode,j) for j in jobs]):
                row=f.result()
                arm=row['arm']
                exp.save(exp.OUT/'evaluation'/arm/(row['room']+'.json'),row)
                done[arm]+=1
                exp.save(exp.OUT/'eval_progress.json',done)
        MODELS.clear()
    report()


if __name__=='__main__': main()
