"""One physical-action override, followed by the original greedy ten-CAP actor."""
import csv
import hashlib
import math
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch import nn

from tools import cap_regional_cpp as cpp
from tools import cap_regional_id_ppo as ppo

study = ppo.study
OUT = ppo.OUT.parent / 'cap_regional_once_20261120_01'
DOCS = ppo.DOCS.parent / '2026-11-20_regional_once'
WINDOWS = ((0, 5), (6, 23), (24, 79))
THRESHOLDS = (0., .03, .07, .12)
ACTOR = HIDDEN = MODEL = None
STATUS = {}


def freeze():
    old = study.ext.old
    historical = study.read(old.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json')
    used = {r['content_sha256'] for r in historical}
    names = {r['case_id'] for r in historical}
    used.update(study.read(ppo.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    for file in [study.OUT/study.ext.DEV, study.OUT/'held.json', old.OUT/'held.json']:
        prev = study.read(file)
        names.update(prev)
        used.update(map(study.ext.input_hash,prev.values()))
    for pat in ('*/panel.json','*/evaluation_panel.json','*/batches/*_plan.json','cap_semantic_disagreement*/*_panel.json','*/panels/*.json'):
        for path in ppo.OUT.parent.glob(pat):
            data=study.read(path)
            for r in data if isinstance(data,list) else data.get('cases',[]):
                if isinstance(r,dict):
                    used.add(r.get('content_sha256')); names.add(r.get('case_id'))
    cases = {}
    def add(split, seed, row):
        cid = f"once_{seed}_{row['id']}"
        fp=study.ext.input_hash(row['fixture'])
        assert fp not in used and cid not in names
        used.add(fp); names.add(cid)
        cases[cid]=dict(split=split,fixture=row['fixture'],hash=fp,objects=row['movable_count'])
    for row in old.rooms.generate(6312017)['cases']:
        add('fit' if row['split']=='dev' else 'tune',6312017,row)
    # Independently draw the object count per acceptance room for the paired binomial bound.
    bank={n:[] for n in range(2,6)}
    for seed in (6312018,6312019):
        for row in old.rooms.generate(seed)['cases']:bank[row['movable_count']].append((seed,row))
    for n in np.random.default_rng(6312020).integers(2,6,200):
        seed,row=bank[n].pop()
        add('accept',seed,row)
    assert [sum(v['split']==s for v in cases.values()) for s in ('fit','tune','accept')]==[160,40,200]
    actor_sha=study.sha(ppo.base.OUT/'fixed.pt')
    assert actor_sha==study.read(ppo.OUT/'protocol.json')['parent']
    backend=cpp.load()[1]
    assert backend['binary']==study.read(ppo.OUT/'cpp_backend.json')['binary']
    protocol=dict(actor=actor_sha,backend=backend['binary'],cases=old.digest(cases),windows=WINDOWS,
                  candidate='first eligible boundary per window, baseline and next two highest distinct legal actions',
                  features='frozen actor hidden vector, source masks, baseline and alternate action parameters and geometry, score gap and public decision count',
                  fit='seed17, 32-tanh shared head, ternary cross entropy, 120 full-batch AdamW steps, equal room weight',
                  thresholds=THRESHOLDS,selection='largest paired tune net, then fewer overrides, then higher threshold',
                  accept='one frozen candidate, 200 new rooms drawn from uniform 2..5 object mixture, one-sided 95% lower bound on net difference strictly positive',
                  bound='Bonferroni CP: 97.5% two-sided interval for discordance and 97.5% lower bound for rescue conditional on discordance',
                  actor_continuation='greedy supervised 24000-step seed17 actor after the one override',
                  limits=[1800,300],fit_rooms=160,tune_rooms=40,accept_rooms=200,source=study.sha(__file__))
    for name,data in [('cases',cases),('protocol',protocol)]:
        path=OUT/(name+'.json')
        if path.exists():assert study.ext.old.digest(study.read(path))==study.ext.old.digest(data)
        else:study.save(path,data)
    return cases


def state(stage, **changes):
    STATUS.update(stage=stage, **changes)
    study.save(OUT/'progress.json',STATUS)


class Override(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.register_buffer('mean',torch.zeros(width))
        self.register_buffer('scale',torch.ones(width))
        self.net=nn.Sequential(nn.Linear(width,32),nn.Tanh(),nn.Linear(32,3))

    def forward(self,x):
        return self.net(((x-self.mean)/self.scale).clamp(-5,5))

    def score(self,features):
        p=self(torch.tensor(np.stack(features))).softmax(-1)
        return (p[:,2]-p[:,0]).tolist()


def step_choices(library,env,step,need):
    x,candidates,sources,_,pool,raw=library.panel(env.history)
    t=torch.from_numpy(x[None])
    hidden=HIDDEN(ACTOR.encode(t))
    logits=ACTOR.net[-1](hidden)[0].masked_fill(~ACTOR.legal(t)[0],-torch.inf)
    primary=int(logits.argmax())
    if not torch.isfinite(logits[primary]):return pool[0],[]
    action=raw[primary]; alternatives=[]
    if not need or len(pool)<2:return action,alternatives
    scores=logits.numpy()
    which=pool.index(action)
    for i in np.argsort(-scores,kind='stable'):
        if not np.isfinite(scores[i]):break
        a=raw[i]
        if a==action or any(item[0]==a for item in alternatives):continue
        j=pool.index(a)
        values=np.concatenate((hidden[0].numpy(),sources[which],sources[j], candidates[which],
                               candidates[j],[scores[primary]-scores[i],step/300])).astype(np.float32)
        values=np.r_[np.nan_to_num(values,nan=0.),(~np.isfinite(values)).astype(np.float32)]
        alternatives.append((a,values))
        if len(alternatives)==2:break
    return action,alternatives


@torch.inference_mode()
def episode(fixture, mode, *, forced=None, threshold=None):
    env=ppo.environment(fixture); library=cpp.Proposals()
    opportunities=[]; visited=set(); chosen=None
    try:
        for step in range(300):
            window=next((i for i,(lo,hi) in enumerate(WINDOWS) if lo<=step<=hi),None)
            eligible=window is not None and window not in visited
            inspect=(mode in ('collect','candidate') and eligible and chosen is None) or (mode=='force' and step==forced['step'])
            action,alts=step_choices(library,env,step,inspect)
            if inspect and alts:
                visited.add(window)
                prefix=env.history.sha256
                if mode=='collect':
                    for a,x in alts:
                        opportunities.append(dict(step=step,prefix=prefix,base_action=action,action=a,x=x))
                elif mode=='force':
                    assert prefix==forced['prefix'] and action==forced['base_action']
                    assert forced['action'] in [a for a,_ in alts]
                    chosen=step; action=forced['action']
                else:
                    scores=MODEL.score([x for _,x in alts])
                    j=int(np.argmax(scores))
                    if scores[j]>threshold:
                        chosen=step;action=alts[j][0]
            env.step(action,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:break
        if mode=='force':assert chosen==forced['step']
        return dict(success=int(env.env.success),history=env.history.sha256,steps=step+1,
                    ticks=env.env.total_periods,override=chosen),opportunities
    finally:
        env.close_reads();study.dq.fast_runtime.ACTIVE=None


def room_job(job):
    cid,case,threshold=job
    path=OUT/'rooms'/(cid+'.json')
    if path.exists():return study.read(path)
    f=case['fixture']
    if case['split']=='accept':
        baseline,_=episode(f,'baseline')
        candidate,_=episode(f,'candidate',threshold=threshold)
        if candidate['override'] is None:assert candidate==baseline
        r=dict(cid=cid,split='accept',baseline=baseline,candidate=candidate)
    else:
        baseline, opportunities=episode(f,'collect')
        rows=[];features=[]
        for o in opportunities:
            features.append(o.pop('x'))
            b,_=episode(f,'force',forced=o)
            rows.append(dict(**o,outcome=b,delta=b['success']-baseline['success']))
        if rows:
            np.save(OUT/'rooms'/(cid+'.npy'),np.stack(features))
        r=dict(cid=cid,split=case['split'],baseline=baseline,branches=rows)
    study.save(path,r)
    return r


def collect(cases, phase, threshold=None):
    jobs=[(k,v,threshold) for k,v in cases.items() if (v['split']=='accept')==(phase=='accept')]
    rows=[]; curve=[]
    counts=dict(done=0,total=len(jobs),wins=0,losses=0,neutral=0,branches=0,rescuable=0,vulnerable=0,
                baseline=0,candidate=0,overrides=0)
    state(phase,**{phase:counts})
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(room_job,j) for j in jobs]
        for future in as_completed(futures):
            r=future.result();rows.append(r);counts['done']+=1
            counts['baseline']+=r['baseline']['success']
            if phase=='accept':
                c=r['candidate'];d=c['success']-r['baseline']['success']
                counts['wins']+=int(d>0);counts['losses']+=int(d<0)
                counts['candidate']+=c['success'];counts['overrides']+=int(c['override'] is not None)
            else:
                ds=[b['delta'] for b in r['branches']]
                counts['wins']+=ds.count(1);counts['losses']+=ds.count(-1);counts['neutral']+=ds.count(0)
                counts['branches']+=len(ds);counts['rescuable']+=int(1 in ds);counts['vulnerable']+=int(-1 in ds)
            if counts['done']%5==0 or counts['done']==counts['total']:
                curve.append([counts[k] for k in ('done','wins','losses')])
            state(phase,**{phase:counts,phase+'_curve':curve})
    return sorted(rows,key=lambda r:r['cid'])


def examples(rows):
    parts=[];labels=[];weights=[]
    for r in rows:
        if not r['branches']:continue
        a=np.load(OUT/'rooms'/(r['cid']+'.npy'));n=len(a)
        assert n==len(r['branches'])
        parts.append(a);labels.extend(b['delta']+1 for b in r['branches']);weights.extend([1/n]*n)
    return torch.from_numpy(np.concatenate(parts)),torch.tensor(labels),torch.tensor(weights)


def override_from_branches(row,scores,threshold):
    branches=row['branches']
    for step in dict.fromkeys(b['step'] for b in branches):
        ix=[i for i,b in enumerate(branches) if b['step']==step]
        best=max(ix,key=lambda i:scores[i])
        if scores[best]>threshold:return branches[best]['delta'],step
    return 0,None


def train(rows):
    global MODEL
    fit=[r for r in rows if r['split']=='fit'];tune=[r for r in rows if r['split']=='tune']
    x,y,w=examples(fit);tx,ty,tw=examples(tune)
    torch.manual_seed(17)
    MODEL=Override(x.shape[1])
    with torch.no_grad():
        MODEL.mean.copy_(x.mean(0));MODEL.scale.copy_(x.std(0,unbiased=False).clamp_min(.001))
    optimizer=torch.optim.AdamW(MODEL.parameters(),lr=.001,weight_decay=.01)
    curve=[]
    for epoch in range(1,121):
        logits=MODEL(x)
        loss=(nn.functional.cross_entropy(logits,y,reduction='none')*w).sum()/w.sum()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch==1 or epoch%5==0:
            with torch.no_grad():
                def ce(xx,yy,ww):
                    return float((nn.functional.cross_entropy(MODEL(xx),yy,reduction='none')*ww).sum()/ww.sum())
                curve.append([epoch,ce(x,y,w),ce(tx,ty,tw)])
            state('fit',fit_curve=curve)
    MODEL.eval()
    with torch.no_grad():
        summaries=[]
        for t in THRESHOLDS:
            changes=[]
            for r in tune:
                scores=MODEL.score(np.load(OUT/'rooms'/(r['cid']+'.npy'))) if r['branches'] else []
                changes.append(override_from_branches(r,scores,t))
            ds=[d for d,_ in changes]
            summaries.append(dict(threshold=t,net=sum(ds),rescued=ds.count(1),lost=ds.count(-1),
                                  overrides=sum(s is not None for _,s in changes)))
    choice=max(summaries,key=lambda s:(s['net'],-s['overrides'],s['threshold']))
    torch.save(MODEL.state_dict(),OUT/'override.pt')
    frozen=dict(threshold=choice['threshold'],model_sha=study.sha(OUT/'override.pt'),tuning=summaries)
    study.save(OUT/'frozen.json',frozen)
    state('frozen',frozen=frozen)
    return choice['threshold']


def binom_lower(k,n,alpha):
    if k==0:return 0.
    lo,hi=0.,1.
    coeff=[math.comb(n,i) for i in range(k,n+1)]
    for _ in range(60):
        p=(lo+hi)/2
        tail=sum(c*p**i*(1-p)**(n-i) for i,c in enumerate(coeff,k))
        if tail<alpha:lo=p
        else:hi=p
    return (lo+hi)/2


def lower_bound(w,l,n):
    d=w+l
    rate_lo=binom_lower(d,n,.0125);rate_hi=1-binom_lower(n-d,n,.0125)
    signed=2*binom_lower(w,d,.025)-1
    return signed*(rate_lo if signed>=0 else rate_hi)


def run():
    global ACTOR,HIDDEN,MODEL
    OUT.mkdir(exist_ok=True);(OUT/'rooms').mkdir(exist_ok=True)
    torch.set_num_threads(1)
    cases=freeze()
    ACTOR=ppo.base.ExpertActor()
    ACTOR.load_state_dict(torch.load(ppo.base.OUT/'fixed.pt',map_location='cpu',weights_only=True))
    ACTOR.eval();HIDDEN=ACTOR.net[:-1]
    rows=collect(cases,'collect')
    if (OUT/'frozen.json').exists():
        frozen=study.read(OUT/'frozen.json')
        assert frozen['model_sha']==study.sha(OUT/'override.pt')
        params=torch.load(OUT/'override.pt',weights_only=True)
        MODEL=Override(len(params['mean']));MODEL.load_state_dict(params);MODEL.eval()
        t=frozen['threshold'];state('frozen',frozen=frozen)
    else:t=train(rows)
    acceptance=collect(cases,'accept',t)
    from tools.cap_regional_once_report import report
    outcome=report(rows,acceptance,STATUS)
    state('finished',result=outcome)


if __name__=='__main__':
    try:run()
    except Exception as e:
        state('error',error=repr(e))
        raise
