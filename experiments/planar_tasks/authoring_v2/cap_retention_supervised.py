"""Supervised expert continuation scores, matched sampling and commitment ablation."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from collections import defaultdict
import gzip
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn
from . import cap_trace_ppo_comparison as prior
from .cap_trace_value_long import sha,phase
from .random_room_tasks import generate
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import fast_runtime as fast,full_worker as worker
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read,save,flatten
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.resident_replay import ReplayCache
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.runtime import CAPS

BASE=prior.BASE
PARENT=prior.PARENT
TRAIN=PARENT/'arrays/train'
ARMS=('boundary','root','paired')
MODES=('once','free','guard')
SEEDS=(17,29)
STEPS=6000
WIDTH=11304+602+2
CACHE={}


class Router(nn.Module):
    def __init__(self):
        super().__init__()
        norms=np.load(PARENT/'normalizers.npz')
        self.register_buffer('mean',torch.tensor(norms['mean'][:WIDTH]))
        self.register_buffer('scale',torch.tensor(norms['scale'][:WIDTH]))
        self.net=nn.Sequential(nn.Linear(WIDTH*2,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,7))

    def forward(self,x):
        x=x[:,:WIDTH]
        available=x.isfinite()
        z=torch.where(available,(x-self.mean)/self.scale,0).clamp(-10,10)
        return self.net(torch.cat([z,available.float()],1))


class Sampler:
    def __init__(self,store):
        a=store.arrays
        roots,caps,steps,actions,labels=(a[k] for k in ('root','cap','boundary','chosen','y'))
        self.count=len(roots)
        starts=np.flatnonzero(steps==0)
        self.roots=np.unique(roots)
        assert len(starts)==len(self.roots)*7
        order=np.lexsort((caps[starts],roots[starts]))
        self.starts=starts[order].reshape(-1,7)
        self.lengths=np.diff(np.r_[starts,len(roots)])[order].reshape(-1,7)
        assert np.all(caps[self.starts]==np.arange(7)) and np.all(roots[self.starts]==self.roots[:,None])
        # Under deterministic simulation a common root and requested action prefix
        # determine the next boundary. At each shared node the pool order is fixed.
        edges={};groups=defaultdict(list)
        for start,length in zip(starts,np.diff(np.r_[starts,len(roots)])):
            node=-1-int(roots[start])
            for i in range(start,start+length):
                groups[node].append(i)
                node=edges.setdefault((node,int(actions[i])),len(edges))
        self.pairs=defaultdict(list)
        count=noninitial=0
        for ids in groups.values():
            if len(ids)<2:continue
            pairs=[(p,n) for p in ids if labels[p] for n in ids if not labels[n] and actions[p]!=actions[n]]
            if not pairs:continue
            x=store.decode(ids)[:,:WIDTH]
            assert np.all((x==x[0])|(np.isnan(x)&np.isnan(x[0])))
            for key in ('valid','candidates','sources'):
                vals=a[key][ids]
                assert all(np.array_equal(v,vals[0],equal_nan=True) for v in vals)
            self.pairs[int(roots[ids[0]])].append(pairs)
            count+=1;noninitial+=int(steps[ids[0]]>0)
        self.eligible=sorted(self.pairs)
        self.inventory=dict(roots=len(self.roots),trajectories=len(starts),boundaries=len(roots),
            mixed_prefixes=count,noninitial_prefixes=noninitial,eligible_roots=len(self.eligible),
            cap_train_successes=labels[self.starts].sum(0).astype(int).tolist())

    def root_rows(self,rng,n):
        r=rng.integers(len(self.roots),size=n); c=rng.integers(7,size=n)
        return self.starts[r,c]+rng.integers(self.lengths[r,c])

    def batch(self,arm,rng):
        if arm=='boundary':return rng.integers(self.count,size=64)
        if arm=='root':return self.root_rows(rng,64)
        ids=list(self.root_rows(rng,32))
        for root in rng.choice(self.eligible,16):
            prefixes=self.pairs[root]
            pairs=prefixes[rng.integers(len(prefixes))]
            ids.extend(pairs[rng.integers(len(pairs))])
        return np.asarray(ids)


def fit_all(out):
    torch.set_num_threads(4)
    store=ReplayCache(BASE/'cap_trace_value_perf_20261118_01/packed').get(TRAIN)
    sampler=Sampler(store)
    save(out/'training_inventory.json',sampler.inventory)
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            rng=np.random.default_rng(seed)
            model=Router()
            optimizer=torch.optim.Adam(model.parameters(),lr=3e-4)
            root_draws=np.zeros(760,int)
            positives=0
            for i in range(STEPS):
                ids=sampler.batch(arm,rng)
                cap=torch.from_numpy(store.arrays['cap'][ids])
                y=torch.from_numpy(store.arrays['y'][ids])
                logits=model(torch.from_numpy(store.decode(ids))).gather(1,cap[:,None]).squeeze(1)
                loss=nn.functional.binary_cross_entropy_with_logits(logits,y)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                positives+=int(y.sum())
                np.add.at(root_draws,store.arrays['root'][ids],1)
                if (i+1)%1000==0:phase(out,'fit',arm=arm,seed=seed,updates=i+1)
            torch.save(model.state_dict(),out/f'{arm}_s{seed}.pt')
            save(out/f'{arm}_s{seed}_samples.json',dict(draws=STEPS*64,positive=positives,
                roots_seen=int((root_draws>0).sum()),root_draws=root_draws.tolist()))


def validation(out):
    from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
    store=Store(PARENT/'arrays/validation')
    store.decode=lambda ids:np.asarray(store.arrays['x'][ids])
    sampler=Sampler(store)
    save(out/'validation_inventory.json',sampler.inventory)
    initial=sampler.starts
    prefix_groups=[]
    for root,nodes in sampler.pairs.items():
        for pairs in nodes:
            pos,neg=np.array(pairs).T
            prefix_groups.append((root,pos[0],store.arrays['cap'][pos],store.arrays['cap'][neg]))
    records=[]
    for arm in ARMS:
        for seed in SEEDS:
            model=Router()
            model.load_state_dict(torch.load(out/f'{arm}_s{seed}.pt',weights_only=True))
            def scores(ids):
                with torch.inference_mode():
                    return np.concatenate([model(torch.from_numpy(store.decode(ix))).numpy()
                        for ix in np.array_split(ids,max(1,(len(ids)+63)//64))])
            start_scores=scores(initial[:,0])
            outcomes=store.arrays['y'][initial]
            winner=start_scores.argmax(1)
            predictions=scores([g[1] for g in prefix_groups])
            ranks=defaultdict(list)
            for row,(root,_,pos,neg) in zip(predictions,prefix_groups):
                ranks[root].append(float(np.mean((row[pos]>row[neg])+0.5*(row[pos]==row[neg]))))
            records.append(dict(arm=arm,seed=seed,initial_success=int(outcomes[np.arange(len(initial)),winner].sum()),
                initial_total=len(initial),mixed_prefix_root_equal_ranking=float(np.mean([np.mean(r) for r in ranks.values()])),
                eligible_roots=len(ranks),eligible_prefixes=len(predictions)))
    save(out/'validation_metrics.json',records)


def freeze_panel(out,ref):
    known=read(Path(ref['bundle'])/'used_scene_identities.json')
    for file in [PARENT/'panels/final_test.json',PARENT/'panels/online_training.json',prior.SEALED,
                 BASE/'cap_trace_ppo_continue_20261118_01/confirmation.json']:
        known+=read(file)['cases']
    ids={r['case_id'] for r in known};hashes={r['content_sha256'] for r in known}
    cases=[]
    for r in generate(2026111901)['cases']:
        f=r['fixture'];cid='retention_'+r['id'];h=digest({k:v for k,v in f.items() if k!='scene_id'})
        assert cid not in ids and h not in hashes
        ids.add(cid);hashes.add(h)
        cases.append(dict(case_id=cid,fixture=f,movable_count=r['movable_count'],content_sha256=h,fixture_sha256=digest(f)))
    assert len(cases)==200
    save(out/'panel.json',dict(cases=cases,generation_seed=2026111901))
    return cases


def choose(mode,scores,packet,incumbent,pending):
    legal=np.asarray(packet['legal'])
    if not legal.any():
        return incumbent,None,packet['pool'][0],'fallback'
    best=int(np.argmax(np.where(legal,scores,-np.inf)))
    if incumbent is None or not legal[incumbent]:
        return best,None,packet['proposals'][best],'initial' if incumbent is None else 'forced'
    current=packet['proposals'][incumbent]
    if mode=='once' or packet['proposals'][best]==current:
        return incumbent,None,current,'hold'
    if mode=='free' or (scores[best]-scores[incumbent]>=.15 and pending==best):
        return best,None,packet['proposals'][best],'switch'
    return incumbent,best if scores[best]-scores[incumbent]>=.15 else None,current,'hold'


def execute(job):
    case,mode,cp,dest,name=job
    if mode=='cap':
        return worker.run(dict(case=case,directory=str(dest),cap=name,condition=name))
    if mode=='ppo':
        return prior.ppo.rollout((case,dest,prior.origin('GA',int(name.split('s')[-1])),cp,17,False))
    if cp not in CACHE:
        model=Router();model.load_state_dict(torch.load(cp,weights_only=True));CACHE[cp]=model.eval()
    dest.mkdir(parents=True)
    model=CACHE[cp]
    env=fast.ScopedEnvironment(case['fixture'],worker.EventConfig(mode='duration'),fast_feedback=True,
        history_factory=fast.AuditedCachedHistory,record_details=True)
    h=env.history;public=[dict(worker.current_event(h),history_hash=h.sha256)]
    traj=[];incumbent=pending=initial=first_switch=None
    forced=switches=0;start=time.monotonic()
    try:
        worker.LIBRARY.reset()
        for i in range(300):
            if time.monotonic()-start>1800:raise TimeoutError(case['case_id'])
            packet=worker.LIBRARY.advance(h)
            with torch.inference_mode():
                scores=model(torch.from_numpy(flatten(packet)[None])).sigmoid()[0].numpy()
            incumbent,pending,action,reason=choose(mode,scores,packet,incumbent,pending)
            if initial is None:initial=incumbent
            if reason in ('switch','forced'):
                switches+=1
                if first_switch is None:first_switch=i
            forced+=reason in ('forced','fallback')
            event=env.step(action,remaining_ticks=1800-env.env.total_periods)
            traj.append(dict(index=i,cap=incumbent,reason=reason,scores=scores.tolist(),
                proposals=packet['proposals'],requested=action,event=event))
            public.append(dict(worker.current_event(h),history_hash=h.sha256))
            if env.env.termination is not None or env.env.total_periods>=1800:break
        row=dict(case_id=case['case_id'],condition=name,success=bool(env.env.success),complete=True,
            decisions=len(traj),ticks=env.env.total_periods,stop=env.env.termination or ('tick_budget' if env.env.total_periods>=1800 else 'decision_budget'),
            first_cap=CAPS[initial] if initial is not None else None,switches=switches,first_switch=first_switch,forced=forced)
        with gzip.open(dest/'result.json.gz','wt',compresslevel=1) as fp:
            fp.write(json.dumps(dict(summary=row,decisions=traj,public_rows=public,replay=env.replay()),allow_nan=False))
        save(dest/'summary.json',row)
        return row
    finally:
        env.close_reads();fast.ACTIVE=None


def evaluate(out,cases,ref):
    conditions=[(c,'cap',None) for c in (f'CAP-RR-{n:03d}' for n in range(1,11))]
    for seed in SEEDS:
        for arm in ARMS:
            for mode in MODES:
                conditions.append((f'{arm}_s{seed}_{mode}',mode,out/f'{arm}_s{seed}.pt'))
        cp=BASE/'cap_trace_ppo_confirm_20261118_01'/f'GA_s{seed}'/'actor_critic_r40.pt'
        conditions.append((f'PPO_s{seed}','ppo',cp))
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=prior.init_worker,
            initargs=(ref['bundle'],ref['priority_bundle'])) as pool:
        for name,mode,cp in conditions:
            phase(out,'evaluation',condition=name)
            dest=out/'evaluation'/name
            futures=[pool.submit(execute,(c,mode,cp,dest/c['case_id'],name)) for c in cases]
            rows=[]
            for f in as_completed(futures):
                rows.append(f.result())
                if len(rows)%20==0:save(dest/'progress.json',dict(completed=len(rows),total=200))
            assert len(rows)==200 and all(r['complete'] for r in rows)
            save(dest/'results.json',rows)


def run(out):
    from tools.cap_retention_report import report
    out.mkdir(parents=True,exist_ok=False)
    ref=read(prior.SHUFFLE/'inputs.json')
    cases=freeze_panel(out,ref)
    for f in (Path(__file__),Path('tools/cap_retention_report.py')):shutil.copy(f,out/f.name)
    save(out/'protocol.json',dict(seeds=SEEDS,arms=ARMS,modes=MODES,steps=STEPS,batch=64,
        primary='paired minus boundary under free reselect: proportion retained among new roots solved by any of the seven fixed CAPs',
        secondary='root vs boundary; paired vs root; once/guard vs free; rescue outside union; retain CAP009 and descriptive best of ten',
        strata='200 new scenes, 50 each object count, 10000 paired stratified scene bootstrap resamples; two fixed training seeds',
        architecture='shared GA geometry+advice+budget with train normalizers and validity bits; 64/64 ReLU, seven continuation logits, BCE on recorded CAP only',
        sampling='boundary uniform; root then CAP then boundary uniform; paired half root and half 16 success/failure pairs sampled root then shared prefix then different first action',
        router='same scoring network for all three execution rules; retain incumbent when same action; guard switches if same challenger leads by >=.15 at two consecutive different-action boundaries',
        emergency='invalid incumbent forced to legal best, no legal advice uses library fallback; report all exceptions',
        training=TRAIN.as_posix(),panel_sha=sha(out/'panel.json'),normalizers=sha(PARENT/'normalizers.npz'),
        baseline='frozen GA PPO 2000 episode policies, external reference with different supervision and budget',
        total_evaluations=6000,training_scoring='6 fresh supervised fits; fixed budgets; no panel-based selection'))
    phase(out,'fit')
    fit_all(out)
    validation(out)
    ppo_hashes=read(BASE/'cap_trace_ppo_confirm_20261118_01/FROZEN.json')
    frozen={str(p):sha(p) for p in out.glob('*.pt')}
    for seed in SEEDS:
        f=BASE/'cap_trace_ppo_confirm_20261118_01'/f'GA_s{seed}/actor_critic_r40.pt'
        assert sha(f)==ppo_hashes[str(f)]
        frozen[str(f)]=ppo_hashes[str(f)]
    assert len(frozen)==8 and sha(out/'panel.json')==read(out/'protocol.json')['panel_sha']
    save(out/'FROZEN.json',frozen)
    evaluate(out,cases,ref)
    report(out)
    phase(out,'complete')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out.resolve())
