"""Paired frozen CAPs with baseline, automatic, and explicitly chosen PUSH phases."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
from pathlib import Path
import random

import numpy as np

from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl import push_phases
from experiments.planar_tasks.semantic_rl.phase_aware_cap import ContactAwareCAP, signature, translating
from experiments.planar_tasks.semantic_rl.trusted_native import owned_copy
from tools import cap_regional_five as cap
from tools import push_contract_bench as bench
from tools.cap_general_actor_availability import world
from tools.cap_regional_evaluate import paired
from tools.cap_regional_extension import input_hash
from tools.cap_semantic_library_pilot import OUT as SEMANTIC
from tools.cap_semantic_library_eval import dq

OUT=bench.OUT.parent/'cap_push_phase_comparison_20261123_01'
DOC=cap.ROOT/'docs/research/key_findings/2026-11-23_cap_push_phases'
SOURCE=bench.SOURCE_DIR/'author'
CAPS=('s025','s027')
ARMS=('coupled','automatic','feedback','feedback_precision')
SEED=662313811
MAX_TICKS=1800
MAX_DECISIONS=300
MODULES={}


def freeze():
    forbidden={r['content_sha256'] for r in cap.read(cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json')}
    forbidden.update(cap.read(OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    panels=[bench.SOURCE_DIR/'development.json',bench.SOURCE_DIR/'held.json',bench.OUT/'fresh.json',SEMANTIC/'test.json']
    for d in (OUT.parent/'cap_regional_five_20261119_01',OUT.parent/'cap_regional_extension_20261119_01'):
        panels.append(d/'held.json')
    for p in panels:
        forbidden.update(input_hash(f) for f in cap.read(p).values())
    choices=list(bench.DEV)
    random.Random(17).shuffle(choices)
    dev={k:bench.DEV[k] for n in range(2,6) for k in [k for k in choices if len(bench.DEV[k]['movable_objects'])+1==n][:16]}
    held={f'phase_{r["id"]}':r['fixture'] for r in cap.rooms.generate(SEED)['cases']}
    assert len(dev)==64 and len(held)==200 and not {input_hash(f) for f in held.values()} & forbidden
    assert len({input_hash(f) for f in held.values()})==len(held)
    cap.host.RoomCampaign(SOURCE).verify_frozen()
    protocol=dict(seed=SEED,split={'development':cap.digest(dev),'held':cap.digest(held)},
                  caps={sid:hashlib.sha256((SOURCE/'semantics'/f'{sid}.py').read_bytes()).hexdigest() for sid in CAPS},
                  code={name:hashlib.sha256((cap.ROOT/name).read_bytes()).hexdigest() for name in (
                      'tools/cap_push_phase_comparison.py','experiments/planar_tasks/semantic_rl/push_phases.py',
                      'experiments/planar_tasks/semantic_rl/phase_aware_cap.py')},
                  budgets=dict(ticks=MAX_TICKS,boundaries=MAX_DECISIONS),
                  physics='frozen duration adapter, no anchor modifications, phases for single translating pusher only',
                  major='feedback vs automatic, paired per CAP and equally weighted across both CAPs with room bootstrap',
                  validation='feedback_precision is a separately reported fixed secondary arm, not held-set model selection')
    for p,v in [('protocol',protocol),('development',dev),('held',held)]:
        path=OUT/(p+'.json')
        if path.exists():assert cap.read(path)==v
        else:cap.save(path,v)


def init():
    import torch
    torch.set_num_threads(1)
    for sid in CAPS:
        text=(SOURCE/'semantics'/f'{sid}.py').read_text()
        MODULES[sid]=dq.load_native(text)[0]


class FrozenCAP:
    def __init__(self,module):
        self.module=module
        self.memory=module.initial_memory()
        self.index=-1

    def update(self,history):
        assert history.index==self.index+1
        event=cap.current_event(history)
        dt=None if history.previous_tick is None else event['observation']['tick']-history.previous_tick
        self.memory=self.module.update_memory(self.memory,owned_copy(dict(event,executed_ticks=dt)))
        self.index=history.index

    def propose(self,obs):
        context=self.module.prepare_context(owned_copy(self.memory),owned_copy(obs))
        return validate_action(context['proposals'][0],obs)


def play(fixture,sid,arm):
    policy=FrozenCAP(MODULES[sid])
    aware=ContactAwareCAP(precision=arm=='feedback_precision') if arm.startswith('feedback') else None
    counts=Counter()
    changes=[]
    with world(fixture) as (w,_):
        def observation():
            return cap.current_event(w.history)['observation']
        def remaining():
            return MAX_TICKS-w.env.total_periods
        def running():
            return w.env.termination is None and remaining()>0 and w.decision_count<MAX_DECISIONS
        policy.update(w.history)
        while running():
            before=observation()
            proposal=policy.propose(before)
            counts['proposals']+=1
            if arm=='coupled' or len(translating(proposal))!=1:
                w.step(proposal,remaining_ticks=remaining())
                policy.update(w.history)
                continue
            contact=aware.choose_contact(before,proposal) if aware else proposal
            validate_action(contact,before)
            if contact!=proposal:
                old_sig,new_sig=signature(proposal),signature(contact)
                counts['retarget_contact']+=1
                counts['pusher_changed']+=old_sig[1]!=new_sig[1]
                counts['normal_changed']+=old_sig[2]!=new_sig[2]
            result=push_phases.contact(w,contact,min(remaining(),contact['ticks']))
            counts['contact_'+result.status]+=1
            counts['contact_ticks']+=result.ticks
            if result.ticks:
                policy.update(w.history)
            after=observation()
            if aware:
                aware.finish_contact(contact,before,after,result.status,result.ticks)
            if contact!=proposal:
                changes.append(dict(tick=before['tick'],decision='contact',proposal=proposal,action=contact,
                                    outcome=result.status,periods=result.ticks))
            left=contact['ticks']-result.ticks
            if result.status!='ready' or not running() or left<=0:
                # A confirmed contact may have consumed the full envelope.
                assert result.ticks or not running(), (sid,arm,result)
                continue
            push=aware.choose_push(after,contact,left) if aware else dict(contact,ticks=left)
            if push!=dict(contact,ticks=left):
                counts['retime_push']+=1
                changes.append(dict(tick=after['tick'],decision='push',proposal=dict(contact,ticks=left),action=push))
            drive=push_phases.push_from_contact(w,push,min(remaining(),push['ticks']))
            counts['push_'+drive.status]+=1
            counts['push_ticks']+=drive.ticks
            if drive.ticks:
                policy.update(w.history)
            else:
                # A stale joint command can lose its menu entry even though contact
                # is certified. Spend a real brake step before CAP reconsideration.
                counts['drive_refused']+=1
                if not result.ticks:
                    brake=dict(target_id=None,a0=44,a1=44,ticks=1)
                    w.step(brake,remaining_ticks=remaining())
                    policy.update(w.history)
        y=w.env.success
        return dict(success=bool(y),ticks=w.env.total_periods,decisions=w.decision_count,
                    termination=w.env.termination,history=w.history.sha256,counts=dict(counts),changes=changes)


def task(job):
    panel,sid,arm,k,f=job
    path=OUT/'runs'/panel/sid/arm/(k+'.json')
    if path.exists():
        return
    row=play(f,sid,arm)
    if panel=='development' and arm=='coupled':
        reference=cap.read(bench.SOURCE_DIR/'runs'/sid/(k+'.json'))
        assert reference['complete'] and not reference['error']
        assert (row['success'],row['ticks'],row['history'])==(reference['success'],reference['ticks'],reference['history']),(sid,k,row,reference)
    cap.save(path,row)


def run(panel,workers):
    fixtures=cap.read(OUT/(panel+'.json'));protocol=cap.read(OUT/'protocol.json')
    assert cap.digest(fixtures)==protocol['split'][panel]
    for sid,h in protocol['caps'].items():
        assert hashlib.sha256((SOURCE/'semantics'/f'{sid}.py').read_bytes()).hexdigest()==h
    jobs=[(panel,sid,arm,k,f) for k,f in fixtures.items() for sid in CAPS for arm in ARMS
          if not (OUT/'runs'/panel/sid/arm/(k+'.json')).exists()]
    if not jobs:
        return
    with ProcessPoolExecutor(workers,initializer=init) as pool:
        futures=[pool.submit(task,job) for job in jobs]
        for n,f in enumerate(as_completed(futures),1):
            f.result()
            if n%25==0 or n==len(jobs):
                print(panel,n,len(jobs),flush=True)
                cap.save(OUT/f'progress_{panel}.json',dict(done=n,total=len(jobs)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--freeze',action='store_true')
    p.add_argument('--panel',choices=('development','held'))
    p.add_argument('--workers',type=int,default=15)
    a=p.parse_args()
    if a.freeze:freeze()
    if a.panel:run(a.panel,a.workers)
