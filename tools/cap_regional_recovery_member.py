"""An eleventh ordinary proposal, authored from historical failures of the frozen ID actor."""
import argparse
import ast
import collections
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch import nn

from tools import cap_regional_cycle_mask as cycles

ref = cycles.ref
study = ref.study
host = study.ext.old.host
OUT = cycles.OUT.parent / 'cap_recovery_member_20261120_01'
AUTHOR = OUT / 'author_workspace/recovery.py'
CAM = OUT / 'campaign'
PANEL_SEED = 6820553
WIDTH = 64 + 86 + 10 + 4
NATIVE = HEAD = None


def freeze_panels():
    jobs = study.read(cycles.OUT / 'panels.json')
    used = {study.ext.input_hash(row[3]) for row in jobs}
    used.update(study.read(OUT.parent / 'cap_inverse_scale_20261119_01/forbidden.json'))
    for p in [study.OUT/study.ext.DEV, study.OUT/'held.json',study.ext.old.OUT/'held.json']:
        used.update(study.ext.input_hash(f) for f in study.read(p).values())
    used.update(r['hash'] for r in study.read(OUT.parent/'cap_regional_once_20261120_01/cases.json').values())
    for p in (OUT.parent/'cap_regional_actor_iteration_20261120_01/panels').glob('*.json'):
        used.update(study.ext.input_hash(r['fixture']) for r in study.read(p))
    for p in study.read(ref.OUT/'panel.json'):
        used.update(p['hashes'])
    groups = collections.defaultdict(list)
    for group,cid,n,fixture in jobs:
        result = study.read(cycles.OUT/'rooms'/group/(cid+'.json'))['original']
        groups[group,n,bool(result['windows'])].append(dict(cid=cid, objects=n, fixture=fixture, original=result))
    train = []
    rng = np.random.default_rng(17)
    for (_,_,triggered), rows in sorted(groups.items()):
        keep = 8 if triggered else 2
        train.extend(rows[j] for j in rng.permutation(len(rows))[:keep])
    counts = collections.Counter()
    test = []
    for row in study.ext.old.rooms.generate(PANEL_SEED)['cases']:
        n = row['movable_count']
        if counts[n] == 20:
            continue
        counts[n] += 1
        h = study.ext.input_hash(row['fixture'])
        assert h not in used
        used.add(h)
        test.append(dict(cid=row['id'], objects=n, fixture=row['fixture']))
    assert len(test)==80 and len(train)==80
    for name,value in [('train',train),('test',test)]:
        p=OUT/(name+'.json')
        if p.exists():
            assert study.read(p)==value
        else:
            study.save(p,value)
    print('frozen panels',len(train),len(test),flush=True)


def freeze_author():
    train = study.read(OUT/'train.json')
    if not CAM.exists():
        host.RoomCampaign.create(CAM, {r['cid']:r['fixture'] for r in train[:4]})
    c = host.RoomCampaign(CAM)
    source = AUTHOR.read_text()
    if study.read(CAM/'campaign.json')['status']!='frozen':
        names = next(ast.literal_eval(t.value) for t in ast.parse(source).body if isinstance(t,ast.Assign) and getattr(t.targets[0],'id','')=='NAMES')
        cfg=study.ext.old.stream_config(source)
        state=study.read(CAM/'campaign.json')
        if not state['submissions']:
            host.RoomSession(CAM).tool('submit',dict(source=source,summary='Public-history recovery member',groups=dict(candidate={k:'A' for k in names},state={k:'A' for k in cfg['state_names']})))
        c.freeze()
    seal=c.verify_frozen()
    manifest=study.read(CAM/'campaign.json')
    sid,=manifest['submissions']
    assert (CAM/'semantics'/(sid['id']+'.py')).read_text()==source
    assert study.sha(ref.base.OUT/'fixed.pt')==study.read(ref.OUT/'protocol.json')['id_sha']
    protocol=dict(candidate_sha=study.sha(AUTHOR),seal=seal,actor_sha=study.sha(ref.base.OUT/'fixed.pt'),
        train_sha=study.sha(OUT/'train.json'),test_sha=study.sha(OUT/'test.json'),test_seed=PANEL_SEED,
        train='eight previously observed first-cycle rooms and two no-cycle rooms per original panel and object stratum, seed 17; no outcome selection',
        fork='first historical cycle, else initial boundary; lock new member to terminal; baseline original closed-loop from identical prefix',
        router='old ten logits fixed. 11th logit = best legal legacy logit plus learned margin. legal candidate competes at every boundary; ties prefer old ID. Features from public frozen backbone, eleventh proposal, old support and current public feedback.',
        fit='positive differing actions on new successful suffixes where old failed; negative differing actions on original success trajectories. Unit weight per trajectory, equal class weight. 500 Adam steps, lr .001, seed 17, 16-unit ReLU. No positive or no negative training examples defaults to old ranking.',
        test='80 new natural rooms, 20 per 2..5 objects; original ten vs augmented actor. Also paired prefix assay from first observed cycle in original test trajectory.',
        code=study.sha(__file__))
    dest=OUT/'protocol.json'
    if dest.exists():
        assert study.read(dest)==protocol
    else:
        study.save(dest,protocol)


def init():
    global NATIVE
    ref.init()
    NATIVE=study.dq.load_native(AUTHOR.read_text())[0]


def margin_model():
    return nn.Sequential(nn.Linear(WIDTH,16),nn.ReLU(),nn.Linear(16,1))


@torch.inference_mode()
def run(row, mode, origin=None):
    env=ref.ppo.environment(row['fixture'])
    legacy=ref.cpp.Proposals()
    memory=NATIVE.initial_memory()
    features,actions=[],[]
    history=collections.deque(maxlen=9)
    hit=None
    override=unique=chosen_new=invalid=0
    fork_matched=False
    try:
        for step in range(300):
            x,_,_,_,pool,raw=legacy.panel(env.history)
            event=study.dq.current_event(env.history)
            obs=event['observation']
            ticks=None if env.history.previous_tick is None else obs['tick']-env.history.previous_tick
            memory=NATIVE.update_memory(memory,dict(event,executed_ticks=ticks))
            proposals=NATIVE.prepare_context(memory,obs)['proposals']
            new=proposals[0] if proposals else None
            if new is not None:
                try:
                    study.dq.validate_action(new,obs)
                except (ValueError,KeyError,TypeError):
                    new=None
            invalid+=int(new is None)
            t=torch.from_numpy(x[None])
            hidden=ref.ID.net[:-1](ref.ID.encode(t))
            scores=ref.ID.net[-1](hidden).masked_fill(~ref.ID.legal(t),-torch.inf)[0]
            old=raw[int(scores.argmax())] if scores.isfinite().any() else pool[0]
            old_key=cycles.identity(old)
            xy=tuple((r['own']['x'],r['own']['y']) for r in obs['robots'])
            if mode=='old' and hit is None:
                found=cycles.cycle(history,xy,old_key)
                if found:
                    hit=dict(step=step,prefix=env.history.sha256,tick=obs['tick'])
            if origin is not None and step==origin['step']:
                assert env.history.sha256==origin['prefix']
                fork_matched=True
            take=mode=='suffix' and step>=origin['step']
            action=old
            if new is not None:
                unique+=int(new not in pool)
                if new!=old and (mode=='mix' or mode=='fit_old' or take):
                    geo=legacy.geometry
                    adv=np.r_[study.dq.action_encoding(new,geo.goal_id,geo.theta),geo.candidate_geometry(new),1.]
                    support=[a==new for a in raw]
                    tail=[obs['tick']/1800,step/300,*[float(r['contacting_static']) for r in obs['robots']]]
                    vec=np.r_[hidden[0].numpy(),adv,support,tail].astype('f')
                    vec=np.nan_to_num(vec,nan=0.,posinf=10.,neginf=-10.).clip(-10.,10.)
                    assert len(vec)==WIDTH
                    if mode=='mix':
                        score=float(HEAD(torch.from_numpy(vec)))
                        take=score>0
                    else:
                        features.append(vec)
                if take:
                    action=new
                    chosen_new+=1
                    override+=int(new!=old)
            elif take:
                raise ValueError(f"missing locked proposal at {row['cid']} step {step}")
            actions.append(action)
            history.append((cycles.identity(action),xy))
            env.step(action,remaining_ticks=1800-env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods>=1800:
                break
        if origin is not None:
            assert fork_matched
        return dict(success=int(env.env.success),ticks=env.env.total_periods,steps=step+1,
            end=env.env.termination or 'decision_budget',hash=env.history.sha256,hit=hit,invalid=invalid,
            distinct_from_pool=unique,changed=override,selected=chosen_new),features,actions
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE=None


def worker(args):
    phase,row=args
    cid=row['cid']
    dest=OUT/phase/(cid+'.json')
    if dest.exists():
        return study.read(dest)
    if phase=='development':
        prior=row['original']
        if prior['windows']:
            origin=prior['windows'][0]
            triggered=True
        else:
            # Initial boundary is a fully reproducible ordinary non-cycle takeover.
            env=ref.ppo.environment(row['fixture'])
            try:
                origin=dict(step=0,prefix=env.history.sha256,tick=0)
            finally:
                env.close_reads()
                study.dq.fast_runtime.ACTIVE=None
            triggered=False
        old,neg,_=run(row,'fit_old',origin)
        assert (old['hash'],old['success'],old['ticks'])==(prior['history'],prior['success'],prior['ticks'])
        new,pos,acts=run(row,'suffix',origin)
        if not old['success']:
            neg=[]
        if old['success'] or not new['success']:
            pos=[]
        dest.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(dest.with_suffix('.npz'),positive=np.asarray(pos,dtype='f').reshape(-1,WIDTH),
                            negative=np.asarray(neg,dtype='f').reshape(-1,WIDTH))
        entry=dict(cid=cid,objects=row['objects'],triggered=triggered,fork_step=origin['step'],
            fork_prefix=origin['prefix'],old=old,new=new,actions=acts,positive=len(pos),negative=len(neg))
    else:
        old,_,_=run(row,'old')
        new,_,_=run(row,'mix')
        if not new['changed']:
            assert (old['hash'],old['success'])==(new['hash'],new['success'])
        branch=None
        if old['hit']:
            branch,_,_=run(row,'suffix',old['hit'])
        entry=dict(cid=cid,objects=row['objects'],old=old,new=new,branch=branch)
    study.save(dest,entry)
    return entry


def fit(rows):
    global HEAD
    torch.manual_seed(17)
    HEAD=margin_model()
    grouped=[[],[]]
    for row in rows:
        with np.load(OUT/'development'/(row['cid']+'.npz')) as data:
            for i,key in enumerate(('negative','positive')):
                if len(data[key]):
                    grouped[i].append(torch.from_numpy(data[key]))
    if all(grouped):
        opt=torch.optim.Adam(HEAD.parameters(),lr=.001)
        rng=np.random.default_rng(17)
        for _ in range(500):
            blocks=[]
            for group in grouped:
                xs=[]
                for j in rng.integers(len(group),size=32):
                    xs.append(group[j][rng.integers(len(group[j]))])
                blocks.append(torch.stack(xs))
            logits=HEAD(torch.cat(blocks))[:,0]
            loss=nn.functional.binary_cross_entropy_with_logits(logits,(torch.arange(64)>=32).float())
            opt.zero_grad()
            loss.backward()
            opt.step()
    else:
        with torch.no_grad():
            for param in HEAD.parameters():
                param.zero_()
            HEAD[-1].bias.fill_(-1)
    HEAD.eval()
    torch.save(HEAD.state_dict(),OUT/'head.pt')
    study.save(OUT/'fit.json',dict(rooms=list(map(len,grouped)),boundaries=[sum(map(len,g)) for g in grouped],updates=500 if all(grouped) else 0))


def summarize(rows, comparator='new'):
    values=[dict(parent=r['cid'],objects=r['objects'],a=r['old']['success'],b=r[comparator]['success']) for r in rows]
    return study.compare(values,'a','b') if values else dict(n=0)


def batch(phase,rows,workers):
    with ProcessPoolExecutor(workers,mp_context=mp.get_context('fork')) as pool:
        futures=[pool.submit(worker,(phase,r)) for r in rows]
        finished=[]
        for future in as_completed(futures):
            finished.append(future.result())
            study.save(OUT/'progress.json',dict(phase=phase,done=len(finished),total=len(rows),time=time.time()))
    return sorted(finished,key=lambda r:r['cid'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['panels','run'])
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args()
    if args.stage=='panels':
        freeze_panels()
        return
    freeze_author()
    init()
    train=batch('development',study.read(OUT/'train.json'),args.workers)
    fit(train)
    test=batch('evaluation',study.read(OUT/'test.json'),args.workers)
    metrics=dict(development_cycles=summarize([r for r in train if r['triggered']]),
        development_other=summarize([r for r in train if not r['triggered']]),
        fresh=summarize(test),fresh_cycle_takeover=summarize([r for r in test if r['branch']],'branch'),
        fresh_changed_rooms=sum(bool(r['new']['changed']) for r in test),
        fresh_changed_steps=sum(r['new']['changed'] for r in test),
        fresh_steps=sum(r['new']['steps'] for r in test),fit=study.read(OUT/'fit.json'))
    study.save(OUT/'FINISHED.json',metrics)
    print(metrics,flush=True)


if __name__=='__main__':
    main()
