"""Freeze physical effect envelopes, verify authored continuations, evaluate live on new layouts."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
import math
from pathlib import Path
import random

from experiments.planar_tasks.relational_search.joint_orbit import face_owner
from coimagine.random_start.spec import GEOMETRY_PROFILE
from experiments.planar_tasks.relational_search.motion import LEGACY_VERSION
from experiments.planar_tasks.semantic_rl import guarded_transport as transport
from experiments.planar_tasks.authoring_v2 import guarded_options
from tools import cap_guarded_footprint as previous

base,forks,loop=previous.base,previous.forks,previous.loop
ROOT=forks.OUT.parent/'cap_symbolic_transport_20261125'


def run_face(world,action,route):
    if route=='ccw':
        return transport.face.change_face_ccw(world,action)
    if route!='cw':
        raise ValueError('unknown route')
    robot,_=transport.push_phases.pusher(action)
    code=action[f'a{robot}']
    follow=transport.face.changed_action(action,robot,((code//4-2)%8)*4+1,24)
    orbit=transport.face.changed_action(action,robot,43,12)
    obs=transport.current_event(world.history)['observation']
    _,normal=transport.push_phases.pusher(follow)
    if transport.push_phases.contact_state(world,follow,robot,normal)!='ready':
        owner=face_owner(orbit,robot,obs)
        if owner!=robot:
            follow=dict(follow,a0=follow['a1'],a1=follow['a0'])
            orbit=dict(orbit,a0=orbit['a1'],a1=orbit['a0'])
            robot=owner
    start=world.env.total_periods
    orbit_next=True
    while True:
        step=transport.face.next_step(world,robot,follow,orbit,orbit_next,300)
        left=transport.face.FOLLOW_BUDGET-(world.env.total_periods-start)
        if step not in ('orbit','approach') or left<=0:
            break
        if step=='orbit':
            ticks=world.step(orbit,remaining_ticks=min(left,orbit['ticks']))['periods']
            orbit_next=False
        else:
            phase=transport.push_phases.contact_on_new_face(world,follow,min(left,follow['ticks']))
            ticks=phase.ticks
            orbit_next=phase.status!='contact_timeout'
        if not ticks:
            raise ValueError('non-progressing clockwise change of face')
    return dict(followup_completed=step=='ready',followup_ticks=world.env.total_periods-start,
                followup_reason='budget' if step in ('orbit','approach') else step)


def execute_face_change_benchmark(world, perception, planner, obs, commands):
    """Historical benchmark explicitly chains face acquisition after a selected PUSH."""
    live=transport.execute_selected(world,perception,planner,obs,commands)
    selected=live['selected']
    if selected is None:
        return live
    if not live['push_ticks'] or world.env.termination:
        return dict(live, followup_completed=False)
    if not live['continuation_width_passed']:
        return dict(live, followup_completed=False, followup_reason='entry_held')
    cmd,route=next((c,r) for c in commands for d in transport.DOSES for r in transport.ROUTES
                   if transport.option_key(c,d,r)==selected)
    return dict(live, **run_face(world,cmd,route))


@contextmanager
def prepared(fixture, *, historical=True):
    if historical:
        if GEOMETRY_PROFILE != "legacy36":
            raise ValueError("historical replay requires COIMAGINE_PLANAR_GEOMETRY=legacy36")
        fixture = dict(fixture, overrides={**fixture['overrides'], 'push_face_contract': LEGACY_VERSION})
    record,anchor,*_=base.scene_geometry()
    original=anchor['original_action']
    memory=transport.TransportPerception(original['target_id'])
    with forks.archive.S.phase.world(fixture) as (world,_):
        for row in record['steps'][:anchor['step_index']]:
            for seg in row['segments']:
                obs=forks.archive.S.observe(world)
                memory.observe(obs)
                if world.env.termination or not forks.legal(seg['requested'],obs):
                    yield world,memory,obs,[]
                    return
                world.step(seg['requested'],remaining_ticks=seg['end_tick']-seg['start_tick'])
        obs=forks.archive.S.observe(world)
        memory.observe(obs)
        robot,_=transport.push_phases.pusher(original)
        code=original[f'a{robot}']
        commands=[]
        for offset in (0,-1,1,-2,2,-3,3,4):
            request=dict(original,**{f'a{robot}':((code//4+offset)%8)*4+(code%4)})
            _,axis=transport.push_phases.pusher(request)
            if forks.legal(request,obs) and transport.push_phases.contact_state(world,request,robot,axis)=='ready':
                commands.append(request)
        yield world,memory,obs,commands


@contextmanager
def measure_ticks(world,plane,angle):
    # Offline auditor: the return to the control path is always the original
    # detector result. Plane position, body vertices and contacts never enter planning.
    normal=(math.cos(angle),math.sin(angle))
    def sample():
        truth=forks.archive.truth(world)
        return (plane-base.bench.extent(world,normal),truth['object'][:2],truth['object_static'])
    history={world.env.total_periods:sample()}
    previous_update=world.detector.update
    def update(packet):
        history[world.env.total_periods]=sample()
        return previous_update(packet)
    world.detector.update=update
    try:
        yield history
    finally:
        world.detector.update=previous_update


def physical(samples,end_push,axis,completed):
    start=min(samples)
    before,xy,_=samples[start]
    endpoint=samples[end_push]
    forward=sum((v-u)*n for u,v,n in zip(xy,endpoint[1],axis))
    push_contact=any(s[2] for t,s in samples.items() if start<t<=end_push)
    follow_contact=any(s[2] for t,s in samples.items() if t>end_push)
    return dict(forward=forward,start_gap=before,endpoint_loss=max(0.,before-endpoint[0]),
                push_sweep=max(0.,before-min(s[0] for t,s in samples.items() if t<=end_push)),
                full_sweep=max(0.,before-min(s[0] for s in samples.values())),push_contact=push_contact,
                follow_contact=follow_contact,completed=bool(completed),ticks=max(samples)-start,
                valid=bool(forward>=.02 and not push_contact and not follow_contact and completed))


def force_branch(fixture,plane,angle,cmd,dose,route,prefix):
    with prepared(fixture) as (world,_,_,commands):
        if cmd not in commands or (prefix is not None and world.history.sha256!=prefix):
            return None
        if forks.archive.truth(world)['object_static']:
            return None
        with measure_ticks(world,plane,angle) as samples:
            phase=transport.run_push(world,cmd,dose)
            if not phase.ticks:
                raise ValueError('ready PUSH failed to execute')
            end=world.env.total_periods
            follow=run_face(world,cmd,route)
        _,axis=transport.push_phases.pusher(cmd)
        return physical(samples,end,axis,follow['followup_completed'])


def collect(job):
    spec,split,model=job
    path=ROOT/split/(spec['id']+'.json')
    if path.exists():
        return loop.read(path)
    fixture,plane,angle=base.room(spec)
    p=dict(id=spec['id'],fixture_hash=forks.archive.content_hash(fixture),branches={})
    with prepared(fixture) as (world,memory,obs,commands):
        p['public_history']=world.history.sha256
        if not commands or forks.archive.truth(world)['object_static']:
            p['status']='no_pairable_start'
            loop.seal(path,p)
            return p
        p['status']='ready'
        p['commands']=commands
        p['geometry']={str(c['a0'])+','+str(c['a1']):memory.clearance(obs,c) for c in commands}
        if model:
            planner=transport.TransportPlanner(model)
            with measure_ticks(world,plane,angle) as samples:
                live=execute_face_change_benchmark(world=world,perception=memory,planner=planner,obs=obs,commands=commands)
            if live['selected']:
                cmd=next(c for c in commands if any(transport.option_key(c,d,r)==live['selected']
                          for d in transport.DOSES for r in transport.ROUTES))
                _,axis=transport.push_phases.pusher(cmd)
                live['measured']=physical(samples,live['push_end_tick'],axis,live['followup_completed'])
            p['live']=live
    twin=deepcopy(fixture)
    twin['room']['obstacles'].pop()  # Remove only the experiment's added wall, for development dynamics.
    for cmd in commands:
        _,axis=transport.push_phases.pusher(cmd)
        projected=math.remainder(angle-math.atan2(axis[1],axis[0]),2*math.pi)
        for dose in transport.DOSES:
            for route in transport.ROUTES:
                key=transport.option_key(cmd,dose,route)
                actual=force_branch(fixture,plane,angle,cmd,dose,route,p['public_history'])
                if actual is None:
                    raise ValueError('frozen physical start changed')
                row=dict(cmd=cmd,dose=dose,route=route,angle=projected,actual=actual)
                if split=='development':
                    row['free']=force_branch(twin,plane,angle,cmd,dose,route,None)
                p['branches'][key]=row
    loop.seal(path,p)
    return p


def fit(dev):
    calibration=[]
    margin_errors=[]
    for p in dev:
        if p['status']!='ready':
            continue
        errors=[]
        for row in p['branches'].values():
            c=row['cmd']
            geom=p['geometry'][str(c['a0'])+','+str(c['a1'])]
            if not geom:
                continue
            actual=row['actual']
            errors.append(geom['bounds'][0]-actual['start_gap'])
            if row['free'] is None:
                continue
            # Use both wall-on and wall-removed physical motion for each room.
            # This avoids inferring low unrestricted motion from a wall-stopped box.
            # Outcomes "valid", "completed" and contact flags are never fit.
            effects={k:(min if k=='forward' else max)(actual[k],row['free'][k]) for k in transport.EFFECTS}
            calibration.append(dict(room=p['id'],pair=[c['a0'],c['a1']],dose=row['dose'],
                                    route=row['route'],angle=geom['angle'],**effects))
        if errors:
            margin_errors.append(max(errors))
    margin=transport.upper_order(margin_errors)
    if margin is None:
        raise ValueError('insufficient independent development layouts for gap calibration')
    model = dict(effects=calibration,gap_margin=max(0.,margin),margin_rooms=len(margin_errors))
    if GEOMETRY_PROFILE != 'legacy36':
        model['geometry_profile'] = GEOMETRY_PROFILE
    return model


def main():
    if GEOMETRY_PROFILE != "legacy36":
        raise ValueError("frozen transport study requires COIMAGINE_PLANAR_GEOMETRY=legacy36")
    old=loop.read(previous.ROOT/'plan.json')
    seed=819743
    rng=random.Random(seed)
    fresh=[dict(id=f'contract-{i:03}',gap=rng.uniform(.065,.575),normal_offset=rng.uniform(20,70),
              lateral=rng.uniform(-.15,.15),width=rng.uniform(.65,.86),
              dx=rng.uniform(-.02,.02) if i%2 else 0.,dy=rng.uniform(-.02,.02) if i%2 else 0.) for i in range(120)]
    plan=dict(development=old['development'],fresh=fresh,seed=seed,coverage_order=.90,
              heading_window_deg=25,clearance_margin='empirical, development-only',routes=list(transport.ROUTES),
              doses=list(transport.DOSES),prior=loop.sha(previous.ROOT/'plan.json'),
              sources={Path(m.__file__).name:loop.sha(Path(m.__file__)) for m in
                (previous,base,base.bench,transport,transport.contract_planner,guarded_options,transport.face)},
              collector=forks.frozen_hash(),runner=loop.sha(Path(__file__)),contract=loop.read(transport.CONTRACT_FILE))
    loop.seal(ROOT/'plan.json',plan)
    old_hashes={loop.read(f)['fixture_hash'] for folder in (base.ROOT/'validation',previous.ROOT/'development',
                previous.ROOT/'review',previous.ROOT/'fresh') for f in folder.glob('*.json')}
    hashes=[forks.archive.content_hash(base.room(s)[0]) for s in fresh]
    if len(set(hashes))!=len(hashes) or set(hashes)&old_hashes:
        raise ValueError('duplicate layout')
    model=None
    for split in ('development','fresh'):
        with ProcessPoolExecutor(10) as pool:
            packets=list(pool.map(collect,[(spec,split,model) for spec in plan[split]]))
        print(split,Counter(p['status'] for p in packets),flush=True)
        if split=='development':
            model=fit(packets)
            loop.seal(ROOT/'physical_model.json',model)


if __name__=='__main__':
    main()
