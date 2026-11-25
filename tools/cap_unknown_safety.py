"""Short physical testbed for unknown -> safe observation -> action -> brake."""
import argparse
from pathlib import Path
from dataclasses import asdict
from copy import deepcopy
import math
import time

from shapely.geometry import Polygon, Point, mapping
from tools.cap_symbolic_general import ROOT,save,read,sha
from tools.cap_unknown_safety_audit import OUT
from experiments.planar_tasks.semantic_rl.environment import Environment,EventConfig
from experiments.planar_tasks.authoring_v2.symbolic_general import simulation
from experiments.planar_tasks.authoring_v2.symbolic_unknown.runtime import Controller
from experiments.planar_tasks.authoring_v2.symbolic_unknown.geometry import MotionContract,disc
from experiments.planar_tasks.authoring_v2.random_room_tasks import polygon
from experiments.planar_tasks.relational_search.contract import digest


def fixture(blocked=False, *, x=2.5,y=2.4,robot_gap=.675,obstacle_gap=.125):
    obstacles=[]
    if blocked:obstacles=[dict(cx=x+.125+obstacle_gap+.10,cy=y,half_w=.10,half_h=.20,angle=0.)]
    return dict(scene_id='random_room_v1',room=dict(arena=[6.,4.8],obstacles=obstacles),
        object=dict(x=x,y=y,angle=0.),robots=[dict(x=x-.125-robot_gap,y=y-.30,angle=0.),dict(x=x-.125-robot_gap,y=y+.30,angle=0.)],
        goal=dict(x=4.5,y=y,radius=.2),movable_objects=[],overrides={})


def audit_initial(f,contract):
    target=Polygon(polygon(f['object']['x'],f['object']['y'],.25,1.,f['object']['angle']))
    bodies=[target,*[Point(r['x'],r['y']).buffer(contract.robot_radius)for r in f['robots']]]
    obstacles=[Polygon(polygon(b['cx'],b['cy'],2*b['half_w'],2*b['half_h'],b['angle']))for b in f['room']['obstacles']]
    arena=Polygon([(0,0),(6,0),(6,4.8),(0,4.8)])
    separation=min([b.distance(arena.boundary)for b in bodies]+[b.distance(o)for b in bodies for o in obstacles])
    return dict(initial_separation=separation,initial_condition_valid=separation>=contract.initial_static_separation-1e-8,
        minimum_obstacle_area=min([p.area for p in obstacles],default=None),area_prior_valid=all(p.area>=contract.minimum_static_component_area for p in obstacles))


def run(f, *, observe=True,max_ticks=1600,max_decisions=400,record=True):
    contract=MotionContract();initial=audit_initial(f,contract)
    if not(initial['initial_condition_valid']and initial['area_prior_valid']):raise ValueError('fixture outside declared host assumptions')
    world=Environment(f,EventConfig(mode='duration'),fast_feedback=True,record_details=True)
    controller=Controller(contract,observe=observe);records=[];started=time.monotonic();start=simulation.truth(world)
    observations=[];terminal=None;target=f['object'];first_hash=digest(simulation.observation(world))
    for step in range(max_decisions):
        o=simulation.observation(world)
        if world.env.total_periods>=max_ticks:terminal='tick_budget';break
        if time.monotonic()-started>180:terminal='wall_budget';break
        decision=controller.decide(o);name=decision['action'];before_frames=len(world.env.frames)
        measured=simulation.measured_step(world,decision['command'])
        new_frames=world.env.frames[before_frames:];violations=[]
        for frame in new_frames:
            for i,robot in enumerate(frame['robots']):
                if not decision['region']['robots'][i].buffer(1e-7).covers(disc(robot[0],robot[1],contract.robot_radius)):
                    violations.append(dict(tick=frame['tick'],body='robot'+str(i)))
            if decision['region']['target']is not None:
                p=frame['entities'][world.env.primary_id];actual=Polygon(polygon(p[0],p[1],.25,1.,p[2]))
                if not decision['region']['target'].buffer(1e-7).covers(actual):violations.append(dict(tick=frame['tick'],body='target'))
        # Independent map check is evaluation-only and never feeds the policy.
        obstacles=[Polygon(polygon(b['cx'],b['cy'],2*b['half_w'],2*b['half_h'],b['angle']))for b in f['room']['obstacles']]
        false_free_area=sum(controller.map.free.intersection(p).area for p in obstacles)
        row={k:v for k,v in decision.items()if k not in ('region','pose')}
        row.update(tick=o['tick'],public_sha256=digest(o),audit=measured,envelope_violations=violations,
                   false_free_area_m2=false_free_area,free_area=controller.map.free.area)
        records.append(row)
        if record:observations.append(o)
        progress=simulation.truth(world)['xy'][0]-start['xy'][0]
        if measured['object_static']or measured['robot_static']or measured['robot_peer']:
            terminal='physical_contact';break
        if violations or false_free_area>1e-8:
            terminal='host_assumption_refuted';break
        if progress>=.02:
            terminal='progress_pending_stop';break
        if name=='brake'and (decision['blocked_goal']or not observe or not decision['certified']):
            terminal='obstacle_rejected'if decision['blocked_goal']else 'unknown_withheld';break
        if len(records)>=6 and all(r['action']=='brake'for r in records[-6:]):terminal='no_certified_observation_route';break
    # Every ordinary terminal decision is followed by a fixed two-second
    # physical braking window, not a zero-tick label or a single short command.
    # Truth is used only to score the final rest condition, never to choose when
    # the policy sends another action.
    stopping=None
    if records and terminal not in ('physical_contact','host_assumption_refuted'):
        count=len(world.env.frames);audits=[]
        for _ in range(10):
            audits.append(simulation.measured_step(world,dict(target_id=None,a0=44,a1=44,ticks=contract.brake_ticks)))
        extra=[]
        for frame in world.env.frames[count:]:
            p=frame['entities'][world.env.primary_id]
            if decision['region']['target']is not None and not decision['region']['target'].buffer(1e-7).covers(Polygon(polygon(p[0],p[1],.25,1.,p[2]))):extra.append(dict(tick=frame['tick'],body='target'))
            for i,r in enumerate(frame['robots']):
                if not decision['region']['robots'][i].buffer(1e-7).covers(disc(r[0],r[1],contract.robot_radius)):extra.append(dict(tick=frame['tick'],body='robot'+str(i)))
        body=world.env.env.backend.bodies[world.env.primary_id]
        residual=max([math.hypot(r['own']['vx'],r['own']['vy'])for r in simulation.observation(world)['robots']]+[math.hypot(body.velocity.x,body.velocity.y),abs(body.angular_velocity)*math.hypot(.125,.5)])
        contact=any(a[k]for a in audits for k in ('object_static','robot_static','robot_peer'))
        stopping=dict(ticks=40,audits=audits,envelope_violations=extra,contact=contact,residual_speed_m_s=residual,
                      passed=not extra and not contact and residual<.005)
        if not stopping['passed']:terminal='stopping_failure'
        elif terminal=='progress_pending_stop':terminal='safe_progress_and_stop'
    result=dict(summary=dict(termination=terminal or 'decision_budget',ticks=world.env.total_periods,
        decisions=len(records),progress_m=simulation.truth(world)['xy'][0]-start['xy'][0],
        views=controller.ledger['views'],pushes=controller.ledger['pushes'],
        certified_stops=controller.ledger['certified_stops'],uncertified_emergency_stops=controller.ledger['uncertified_emergency_stops'],
        contact_commands=sum(any(r['audit'][k]for k in ('object_static','robot_static','robot_peer'))for r in records)+sum(any(a[k]for k in ('object_static','robot_static','robot_peer'))for a in (stopping or {}).get('audits',[])),
        envelope_violations=sum(len(r['envelope_violations'])for r in records)+len((stopping or {}).get('envelope_violations',[])),
        stopping_passed=(stopping or {}).get('passed',False),stopping_residual_speed_m_s=(stopping or {}).get('residual_speed_m_s'),
        false_free_area_max=max([r['false_free_area_m2']for r in records],default=0),
        initial_public_sha256=first_hash,wall_seconds=time.monotonic()-started),
        initial_audit=initial,motion_contract=asdict(contract),records=records,public=observations,
        map=controller.map.snapshot(),stopping=stopping,replay=world.replay())
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--blocked',action='store_true');p.add_argument('--no-observe',action='store_true');p.add_argument('--name',required=True)
    a=p.parse_args();path=OUT/'development'/(a.name+'.json.gz')
    if path.exists():raise FileExistsError(path)
    source_files=[*sorted((ROOT/'experiments/planar_tasks/authoring_v2/symbolic_unknown').glob('*.py')),Path(__file__)]
    manifest={str(p.relative_to(ROOT)):sha(p)for p in source_files}
    for p in source_files:
        dest=OUT/'development_sources'/a.name/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(p.read_bytes())
    save(OUT/'development_sources'/a.name/'manifest.json',manifest)
    result=run(fixture(a.blocked),observe=not a.no_observe);save(path,result);print(result['summary'])


if __name__=='__main__':main()
