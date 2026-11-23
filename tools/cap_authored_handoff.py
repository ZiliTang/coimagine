"""Recovery authoring from real frozen-A prefixes and held-room graph routing."""
import argparse
import ast
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
from experiments.planar_tasks.semantic_rl import cap_pair_perception as perception
from tools import cap_option_handoff as old
from tools import cap_regional_five as base

g,s=old.g,old.s
OUT=g.OUT.parent/'cap_authored_handoff_20261123_01'
DOC=g.DOC.parent/'2026-11-23_authored_handoff'
WORK=base.WS.parent/'cap_handoff_b_1123'
CAM=OUT/'campaign'
A='8125941d92702e82da6e57532286d675f8beffe7ddf00abe76e3c45d0105f2ef'
FAMILIES=('stall','no_touch','touch','barrier','near','lost')
read,save=s.read,s.save
SEED=662316119


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()


def init(source=None, *, profile=perception.DEFAULT):
    global AM,BM
    text=(WORK/'basis.py').read_text()
    assert base.digest(text)==A
    a = perception.source_for_profile(text, profile)
    b = perception.source_for_profile(source, profile) if source is not None else None
    AM = s.old.dq.load_native(a)[0]
    BM = s.old.dq.load_native(b)[0] if b is not None else None
    return dict(profile=profile, rays=72, A=perception.fingerprint(a),
                B=perception.fingerprint(b) if b is not None else None)


def sample(obs,hist):
    frame=g.live_frame(obs)
    hist.append(frame)
    seen=bool(frame['seen'])
    gap=float(frame['gap']) if seen else -1.
    contact=int(any(frame['touch']))
    change=hist[0]['gap']-gap if seen and hist[0]['seen'] else None
    stall=bool(len(hist)==5 and change is not None and change<.035)
    barrier=any(r['contacting_static'] for r in obs['robots'])
    fam=[int(flag) for flag in (stall,stall and not contact,stall and contact,stall and barrier,
                                 stall and gap<=1,not seen and len(hist)==5 and not hist[0]['seen'])]
    # Public-only bins retain distinctions important to near-goal contact and preparation.
    band=0 if not seen else int(np.searchsorted([.25,.50,1.,2.],gap))+1
    prog=int(change is not None and change>=.035)
    z=((band*2+contact)*2+prog)*2+int(barrier)
    return dict(z=z,flags=fam,tick=int(obs['tick']),gap=round(gap,3),touch=contact,
                progress=None if change is None else round(float(change),3),
                xy=frame['xy'].round(2).tolist())


def simulate(fixture,mode,family=None,match=None,record=False,seed=0,chooser=None):
    pol=[s.phase.FrozenCAP(AM)]
    if mode!='scan':pol.append(s.phase.FrozenCAP(BM))
    hist=deque(maxlen=5)
    counts=Counter()
    rng=np.random.default_rng(seed)
    trace=[]; edges=[]; origin_hashes=[]; evidence=[]; hit=None; signature=None; n=0; left=0; who=0
    hit_decisions=None; prev=None; elapsed=0; used_decisions=0; selected=Counter(); switches=0; last=0; first_take=None; changed=0; post_counts=Counter()
    with s.phase.world(fixture) as (w,_):
        def update(h):
            for p in pol:p.update(h)
        update(w.history)
        while not s.ended(w):
            obs=s.observe(w)
            v=sample(obs,hist);tick=v['tick']
            if record and prev is not None:edges.append([*prev,v['z'],tick-elapsed,w.decision_count-used_decisions])
            if hit is None and family is not None and eligible(v,n,family):
                hit=dict(n=n,tick=tick,h=w.history.sha256,z=v['z'])
                hit_decisions=w.decision_count
                if match is not None:assert hit==match
            if mode=='B' or (mode in ('lock','probe') and hit is not None):who=1
            elif mode=='mixed' and hit is not None:
                if left==0:who=int(rng.integers(2));left=int(rng.choice([1,3,6,12]))
                left-=1
            elif mode=='graph' and hit is not None:
                who=int(chooser(v['z'],s.remaining(w),s.MAX_DECISIONS-w.decision_count))
            else:who=0
            switches+=int(last!=who)
            last=who
            selected[who]+=1
            if who==1 and first_take is None:first_take=dict(n=n,tick=tick,h=w.history.sha256)
            action=pol[who].propose(obs)
            if hit is not None:post_counts[who]+=1
            if mode in ('graph','lock') and who:changed+=int(action!=pol[0].propose(obs))
            if mode=='scan' or (mode=='probe' and hit and n<hit['n']+10):
                item=dict(**v,n=n,action=action,h=w.history.sha256)
                if mode=='scan':trace.append(item)
                else:
                    item['execution']=[r['execution'] for r in obs['robots']]
                    evidence.append(item)
            if hit is not None and n==hit['n']:signature=action
            if record:origin_hashes.append(w.history.sha256)
            prev=(v['z'],who)
            elapsed=tick;used_decisions=w.decision_count
            s.tick_action(w,action,g.prep.prior.EXECUTOR,update,counts)
            n+=1
        res=s.output(w,counts);res.pop('counts')
        if record:edges.append([*prev,48+res['success'],res['ticks']-elapsed,w.decision_count-used_decisions])
        if match is not None:assert hit==match
    return dict(result=res,hit=hit,first=signature,trace=trace,edges=edges,evidence=evidence,counts=dict(selected),switches=switches,first_take=first_take,hit_decisions=hit_decisions,origin_hashes=origin_hashes,changed=changed,post_counts=dict(post_counts))


def eligible(v,n,family):
    return n>=5 and 180<=v['tick']<=1150 and v['flags'][FAMILIES.index(family)]


def pool(fn,jobs,workers=16):
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork')) as ex:
        for task in as_completed([ex.submit(fn,j) for j in jobs]):task.result()


def scan_one(job):
    k,f=job
    path=OUT/'baseline'/(k+'.json')
    if not path.exists():
        row=simulate(f,'scan',record=True)
        ref=read(s.OUT/'singles/development'/g.prep.prior.EXECUTOR/A/(k+'.json'))
        assert row['result']['history']==ref['history']
        save(path,row)


def dev():
    return old.panel('dev')[0]


def prepare():
    fixtures=dev()
    init(profile=perception.LEGACY)
    pool(scan_one,fixtures.items())
    candidates=[]
    for family in FAMILIES:
        hits=[]
        for k in sorted(fixtures):
            row=read(OUT/'baseline'/(k+'.json'));trace=row['trace']
            point=next((v for v in trace if eligible(v,v['n'],family)),None)
            if point:
                failed=not row['result']['success']
                late=sum(v['flags'][FAMILIES.index(family)] for v in trace[-8:])>=2
                hits.append((k,point,failed,late))
        bad=sum(f for _,_,f,_ in hits);good=len(hits)-bad
        if bad>=16 and good>=8:
            terminal=sum(f and end for _,_,f,end in hits)
            candidates.append((terminal/len(hits),bad/len(hits),family,hits))
    _,_,family,hits=max(candidates,key=lambda x:x[:3])
    chosen=[]
    for fail,count in [(True,16),(False,8)]:
        options=sorted((r for r in hits if r[2]==fail),key=lambda r:(not r[3] if fail else False,key('Bauthor/'+r[0])))
        chosen+=options[:count]
    refs={};packet=[]
    for k,v,fail,_ in chosen:
        row=read(OUT/'baseline'/(k+'.json'));i=v['n']
        refs[k]=dict(n=i,tick=v['tick'],h=v['h'],z=v['z'])
        packet.append(dict(room=k,baseline_success=not fail,remaining_ticks=1800-v['tick'],
                           lead_and_follow=row['trace'][max(0,i-5):i+13],takeover_at=i,
                           final_public_samples=row['trace'][-4:]))
    lock=dict(family=family,refs=refs,baseline_candidates=[dict(family=q[2],n=len(q[3]),persistent_fail_fraction=q[0],fail_fraction=q[1]) for q in candidates])
    save(OUT/'family.json',lock)
    if not (CAM/'campaign.json').exists():
        campaign=base.host.RoomCampaign.create(CAM,{k:fixtures[k] for k in refs})
        with campaign.locked():
            campaign.state['model'].update(name='gpt-6.2-sol',reasoning_effort='high')
            campaign.state['author_protocol'].update(probe_design='24 actual A prefixes per submitted revision, at most three revisions',executor=g.prep.prior.EXECUTOR)
            campaign._save()
    save(WORK/'author_cases.json',dict(family=family,fields='gap is scan target-centroid distance in metres, -1 unseen; progress is reduction over four preceding commands; xy is robot odometry. Flags ordered '+str(FAMILIES),cases=packet))
    print(family,flush=True)


def probe_one(job):
    k,f,idx,match,family=job
    b=simulate(f,'probe',family,match)
    save(OUT/'probes'/f'v{idx}'/(k+'.json'),b)


def probe(idx):
    if idx not in (1,2,3):raise ValueError('use slots 1..3')
    family=read(OUT/'family.json');fixtures=dev()
    src=(WORK/f'v{idx}.py').read_text()
    log=OUT/f'probe{idx}.json'
    if log.exists():
        r=read(log);assert r['sha']==base.digest(src);print(r);return
    sess=base.host.RoomSession(CAM)
    prior=next((v for v in read(CAM/'campaign.json')['submissions'] if v['summary']==f'B revision {idx}'),None)
    if prior:
        assert prior['sha256']==base.digest(src);sid=prior['id']
    else:
        cfg=base.stream_config(src)
        names=next(ast.literal_eval(n.value) for n in ast.parse(src).body if isinstance(n,ast.Assign) and getattr(n.targets[0],'id','')=='NAMES')
        sid=sess.tool('submit',dict(source=src,summary=f'B revision {idx}',groups=dict(candidate=dict.fromkeys(names,'A'),state=dict.fromkeys(cfg['state_names'],'A'))))['id']
    init(src, profile=perception.LEGACY);refs=family['refs']
    jobs=[(k,fixtures[k],idx,v,family['family']) for k,v in refs.items() if not (OUT/'probes'/f'v{idx}'/(k+'.json')).exists()]
    with sess.c.locked():
        sess.c._charge(attempts=len(jobs),resets=len(jobs),execution_ticks=1800*len(jobs),feature_batches=301*len(jobs));sess.c._save()
    pool(probe_one,jobs,8)
    rows=[]
    for k in refs:
        r=read(OUT/'probes'/f'v{idx}'/(k+'.json'))
        a=read(OUT/'baseline'/(k+'.json'))['result']
        rows.append(dict(room=k,a=a['success'],b=r['result']['success'],ticks=r['result']['ticks'],decisions=r['result']['decisions'],follow=r['evidence']))
    just={j[0] for j in jobs}
    with sess.c.locked():
        sess.c.state['used']['execution_ticks']-=sum(1800-r['ticks'] for r in rows if r['room'] in just)
        sess.c.state['used']['feature_batches']-=sum(301-r['decisions'] for r in rows if r['room'] in just);sess.c._save()
    r=dict(revision=idx,sid=sid,sha=base.digest(src),rescued=sum(t['b']>t['a'] for t in rows),lost=sum(t['a']>t['b'] for t in rows))
    save(WORK/f'results{idx}.json',rows);save(log,r);print(r,flush=True)


if __name__=='__main__':
    import sys
    if sys.argv[1]=='prepare':prepare()
    elif sys.argv[1]=='probe':probe(int(sys.argv[2]))
