"""On-policy Monte Carlo value and single-decision truncated rollout on fresh rooms."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from tools import cap_general_actor_availability as runtime

ref = runtime.ref
campaign = ref.campaign
OUT = campaign.OUT / 'actor_value_rollout_20261121_s17'
SEEDS = {'train': [66221010,66221011,66221012], 'val': [66221013], 'test': [66221014]}
SIZES = {'train':512, 'val':96, 'test':200}
CHECKS = (0,250,500,1000,2000,4000,8000)
ACTOR = None


def initialize():
    global ACTOR
    torch.set_num_threads(1)
    old,held,sids=ref.scale.inputs()
    ACTOR=ref.fit.load_best()
    assert ref.read(ref.fit.OUT/'FINISHED.json')['best_train_arm']=='layernorm'
    ref.dq.MODULES=[ref.dq.load_native((campaign.CAM/'semantics'/(s+'.py')).read_text())[0] for s in sids]
    return old,held,sids


def freeze():
    old,held,sids=initialize()
    historical=ref.read(campaign.base.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json')
    used={r['content_sha256'] for r in historical}
    used.update(ref.read(campaign.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    for panel in (old,held,ref.read(ref.scale.OUT/'natural.json'),ref.read(ref.scale.OUT/'variants.json'),
                  ref.read(campaign.base.OUT/'held.json'),ref.read(campaign.OUT.parent/'cap_regional_extension_20261119_01/held.json')):
        used.update(campaign.input_hash(f) for f in panel.values())
    panels={}
    for fold,seeds in SEEDS.items():
        panel={}
        counts=Counter()
        for seed in seeds:
            for room in campaign.base.rooms.generate(seed)['cases']:
                n=room['movable_count']
                if counts[n]>=SIZES[fold]//4:
                    continue
                key=f"{fold}_{n}_{counts[n]:03d}"
                fixture=room['fixture']
                h=campaign.input_hash(fixture)
                assert h not in used
                used.add(h)
                panel[key]=fixture
                counts[n]+=1
        assert len(panel)==SIZES[fold]
        panels[fold]=panel
    OUT.mkdir(exist_ok=True)
    protocol=dict(actor=campaign.sha(ref.fit.OUT/'best_train.pt'),sids=sids,seed=17,seeds=SEEDS,
                  sizes=SIZES,hashes={s:campaign.base.digest(p) for s,p in panels.items()},
                  horizons=[1,4],landmarks=[0,5,12,25],fit_checks=CHECKS,batch=64,lr=0.0001,
                  selection='minimum room-balanced validation Brier; earliest tie',
                  weighting='sample rooms uniformly then boundaries uniformly',
                  target='terminal success; gamma=1; task budget exhausted is failure',
                  features='same causal GA public history, CAP proposals and remaining budgets as actor',
                  evaluation='once per room, first multi-action boundary at or after preassigned offset; ties favor actor')
    for name,value in [('protocol',protocol),('panels',panels)]:
        path=OUT/(name+'.json')
        if path.exists():
            assert campaign.base.digest(ref.read(path))==campaign.base.digest(value)
        else:
            ref.save(path,value)
    return panels


def status(stage,completed,total,**other):
    ref.save(OUT/'progress.json',dict(stage=stage,completed=completed,total=total,**other))


def terminal(env):
    assert runtime.ended(env) and (env.env.termination in ('success','task_budget_exhausted') or
           env.env.termination is None and (env.env.total_periods==1800 or env.decision_count==300))
    return int(env.env.success)


@torch.inference_mode()
def collect_room(job):
    key,fixture,cache=job
    with runtime.world(fixture) as (env,library):
        states=[]
        while not runtime.ended(env):
            panel=library.panel(env.history)
            states.append(panel[0])
            runtime.advance(env,ref.choose(ACTOR,panel))
        success=terminal(env)
        path=cache/(key+'.npy')
        np.save(path,np.stack(states))
        return key,dict(success=success,ticks=env.env.total_periods,decisions=len(states),history=env.history.sha256)


def collect(cache):
    panels=freeze()
    cache.mkdir(parents=True,exist_ok=True)
    results={}
    jobs=[(k,f,cache) for s in ('train','val') for k,f in panels[s].items()]
    status('采集冻结 actor 轨迹',0,len(jobs))
    with ProcessPoolExecutor(12,mp_context=mp.get_context('fork')) as pool:
        for n,future in enumerate(as_completed([pool.submit(collect_room,j) for j in jobs]),1):
            k,row=future.result()
            results[k]=row
            if n%5==0 or n==len(jobs):
                status('采集冻结 actor 轨迹',n,len(jobs))
    assert set(results)==set(panels['train'])|set(panels['val'])
    ref.save(OUT/'on_policy.json',results)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('cache',type=Path)
    collect(parser.parse_args().cache)
