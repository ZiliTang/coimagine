"""Paired full-episode safeguard of a frozen experimental PUSH controller."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np

from tools import cap_regional_five as cap
from tools import push_contract_bench as bench
from tools.cap_semantic_library_pilot import OUT as SEMANTIC, previous
from experiments.planar_tasks.semantic_rl.push_contact_recovery import ContactRecovery
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.memory_flow import current_event


GUARDS={'repeat_once':1,'repeat_cooldown':100}


def setup():
    source=(previous.CAM/'semantics/s025.py').read_text()
    cap.start_worker(source)


def episode(job):
    room,fixture,arm,panel=job
    out=bench.OUT/'closedloop'/panel/arm/(room+'.json')
    if out.exists():return
    info=[]
    def construct(*args,**kwargs):
        kwargs['history_factory']=cap.fast.AuditedCachedHistory
        world=cap.fast.ScopedEnvironment(*args,**kwargs)
        advance=world.step
        update=world.detector.update
        previous=None
        def step(request,remaining_ticks=None):
            nonlocal previous
            movers=bench.translation(request)
            begin=world.env.total_periods
            if len(movers)!=1:
                previous=None
                return advance(request,remaining_ticks)
            robot=movers[0]
            code=bench.ACTIONS[request[f'a{robot}']]
            signature=(request['target_id'],robot,tuple(code.effect_direction))
            def observation():
                r=current_event(world.history)['observation']['robots'][robot]
                return (np.array([r['own']['x'],r['own']['y']]),r['contacting_object'],r['execution']['stage'])
            before=observation()
            old=previous
            guarded=arm in GUARDS
            permitted=not guarded
            if guarded and old is not None and signature==old['signature']:
                timed=bool(old['no_touch'] and old['ticks']>=12 and old['motion']<.045)
                room=not info or arm=='repeat_cooldown' and begin-info[-1]['tick']>=GUARDS[arm]
                permitted=bool(timed and room and before[2]=='standoff' and not before[1])
            recovery=ContactRecovery(*bench.CONFIGS['near5' if guarded else arm]) if permitted else None
            budget=min(request['ticks'],remaining_ticks or request['ticks'])
            ever_touched=bool(before[1])
            def observe(packet):
                nonlocal ever_touched
                ever_touched|=bool(packet['robots'][robot]['contacting_object'] and world.env.active_target==request['target_id'])
                reason=update(packet)
                if recovery is not None and reason not in ('blocked','contact_lost','controller_end') and not world.env.termination and world.env.total_periods<begin+budget:
                    recovery.tick(world.env,packet,request)
                return reason
            world.detector.update=observe
            try:
                event=advance(request,remaining_ticks)
            finally:
                if recovery is not None:recovery.close()
                world.detector.update=update
            after=observation()
            previous=dict(signature=signature,no_touch=not ever_touched,ticks=event['periods'],motion=np.linalg.norm(after[0]-before[0]))
            if recovery is not None:info.extend(recovery.applied)
            return event
        if arm!='base':world.step=step
        return world
    try:
        result,_=cap.rollout(fixture,cap.SOURCE,features=cap.FEATURES,config=cap.CFG,
                              deterministic=True,fast_feedback=True,record_details=False,environment_factory=construct,
                              max_ticks=1800,max_decisions=300,wall_seconds=480)
        row=result['summary']
        if row['error'] or row['stop'] not in ('terminal','decision_budget','execution_budget'):
            raise RuntimeError((room,row))
        cap.save(out,dict(success=bool(row['success']),ticks=row['ticks'],decisions=row['decisions'],
                          termination=row['termination'],history=cap.fast.ACTIVE.history.sha256,interventions=info))
    finally:
        if cap.fast.ACTIVE:
            cap.fast.ACTIVE.close_reads()
            cap.fast.ACTIVE=None


def freeze_fresh():
    library=cap.read(SEMANTIC/'test.json')
    used={previous.input_hash(f) for f in [*bench.DEV.values(),*library.values()]}
    used.update(cap.read(bench.OUT.parent/'cap_inverse_scale_20261119_01/forbidden.json'))
    used.update(r['content_sha256'] for r in cap.read(cap.ROOT/'outputs/transfers/cap_trace_value_dev_v1/used_scene_identities.json'))
    for folder in ('cap_general_specialists_62_20261120_01','cap_regional_extension_20261119_01','cap_regional_five_20261119_01'):
        used.update(previous.input_hash(f) for f in cap.read(bench.OUT.parent/folder/'held.json').values())
    scenes={f'push_{r["id"]}':r['fixture'] for r in cap.rooms.generate(662311507)['cases']}
    identities={previous.input_hash(f) for f in scenes.values()}
    assert len(identities)==200 and not identities & used
    path=bench.OUT/'fresh.json'
    if path.exists():assert cap.read(path)==scenes
    else:cap.save(path,scenes)


def run(panel,arm,workers):
    if panel=='development':cases=bench.DEV
    elif panel=='diagnostic':
        cases=cap.read(SEMANTIC/'test.json')
        assert digest(cases)==cap.read(SEMANTIC/'protocol.json')['held']
    else:
        cases=cap.read(bench.OUT/'fresh.json')
    source=(previous.CAM/'semantics/s025.py').read_text()
    if panel=='development':base=previous.OUT/'runs/s025'
    elif panel=='diagnostic':base=SEMANTIC/'singles'/digest(source)
    else:base=bench.OUT/'closedloop'/panel/'base'
    jobs=[(k,f,arm,panel) for k,f in sorted(cases.items())]
    with ProcessPoolExecutor(workers,initializer=setup) as pool:
        for n,_ in enumerate(pool.map(episode,jobs,chunksize=1),1):
            if n%20==0:print(panel,n,len(jobs),flush=True)
    ratios={k:cap.read(bench.OUT/'closedloop'/panel/arm/(k+'.json')) for k in cases}
    if panel=='fresh' and arm=='base':return
    pairs=[]
    for k,r in ratios.items():
        b=cap.read(base/(k+'.json'))
        if panel=='development':assert b['complete'] and not b['error']
        if not r['interventions']:assert r['history']==b['history'] and r['success']==b['success'],k
        pairs.append(dict(room=k,objects=1+len(cases[k]['movable_objects']),base=int(b['success']),
                          candidate=int(r['success']),interventions=len(r['interventions']),base_ticks=b['ticks'],ticks=r['ticks']))
    from tools.cap_regional_evaluate import paired
    import numpy as np
    score=paired(*(np.array([v[k] for v in pairs]) for k in ('base','candidate','objects')))
    result=dict(arm=arm,paired=score,rooms=pairs)
    cap.save(bench.OUT/f'closedloop_{panel}_{arm}.json',result)
    if arm not in GUARDS and panel!='fresh':cap.save(bench.OUT/f'closedloop_{panel}.json',result)
    print(score,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--panel',choices=('development','diagnostic','fresh'),required=True)
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--arm',choices=(*bench.CONFIGS,*GUARDS))
    a=p.parse_args()
    if a.panel=='fresh':freeze_fresh()
    run(a.panel,a.arm or cap.read(bench.OUT/'selected.json')['candidate'],a.workers)
