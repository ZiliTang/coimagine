"""Development wall-gap bench. Fork two progress doses at identical ready contacts."""
from copy import deepcopy
import argparse
import math
from pathlib import Path

from experiments.planar_tasks.authoring_v2.option_effect_model import EffectModel
from experiments.planar_tasks.semantic_rl import push_phases
from experiments.planar_tasks.semantic_rl.push_takeover import push_checkpoints
from tools import cap_guarded_calibration_forks as forks
from tools import cap_guarded_calibration_loop as loop

ROOM = 'diag_dev_nat_001'
OUT = forks.OUT.parent / 'cap_guarded_wall_pair_v1'


def reference():
    packet = loop.read(forks.OUT / 'forks' / 'dev' / (ROOM + '.json'))
    return next(a for a in packet['anchors'] if a['anchor_id'] == 'early')


def extent(world, axis):
    backend = world.env.env.backend
    return max(shape.body.local_to_world(v).dot(axis)
               for shape in backend.shapes[world.env.primary_id] for v in shape.get_vertices())


def layout(gap, record, anchor, normal_offset_deg):
    fixture = deepcopy(forks.panels()['dev'][ROOM])
    # Translate the contact setup relative to a planar wall by changing only its offset.
    _, axis = push_phases.pusher(anchor['original_action'])
    angle = math.atan2(axis[1], axis[0]) + math.radians(normal_offset_deg)
    axis = (math.cos(angle), math.sin(angle))
    pose = record['steps'][anchor['step_index']]['truth']['object']
    with forks.archive.S.phase.world(fixture) as (world, _):
        for row in record['steps'][:anchor['step_index']]:
            forks.old.restore_step(world, row)
        far = extent(world,axis)
    offset = far - (axis[0] * pose[0] + axis[1] * pose[1]) + gap + .04
    wall = dict(angle=angle, cx=pose[0]+offset*axis[0], cy=pose[1]+offset*axis[1],
                half_w=.04, half_h=.80)
    fixture['room']['obstacles'].append(wall)
    return fixture, far+gap, angle


def branch(fixture, plane, angle, record, anchor, dose, predictor, particles):
    def gap_now(world):
        return plane-extent(world,(math.cos(angle),math.sin(angle)))
    with forks.archive.S.phase.world(fixture) as (world, _):
        # Reissue the development preparation commands in the new geometry.
        for row in record['steps'][:anchor['step_index']]:
            for seg in row['segments']:
                world.step(seg['requested'], remaining_ticks=seg['end_tick']-seg['start_tick'])
        action = anchor['original_action']
        robot, axis = push_phases.pusher(action)
        obs = forks.archive.S.observe(world)
        initial = forks.archive.truth(world)
        result = dict(public_history=world.history.sha256, tick=obs['tick'], action=action,
                      start_wall_gap_m=gap_now(world), start_pose=initial['object'],
                      initial_contact=push_phases.contact_state(world, action, robot, axis),
                      legal=forks.legal(action,obs), dose_m=dose)
        if not result['legal'] or result['initial_contact']!='ready' or initial['object_static']:
            result['status']='unavailable_at_start'
            return result
        public = dict(features=forks.features(obs,action), original_action=action, public_history=world.history.sha256)
        if dose in forks.PROGRESS:
            decision = loop.dispatch(predictor,dose,public,loop.read(loop.RULE),particles,forks.simulate_development_particle)
            result['decision'] = {k:decision[k] for k in ('model_decision','source','admitted','support','calls','predicted_contract_risk','predicted_forward_m')}
        def stop(public_obs, running, left, reason):
            if reason == 'progress':
                return dict(target_id=None,a0=44,a1=44,ticks=1)
        previous = world.detector.update
        touches = []
        def audit(packet):
            if forks.archive.truth(world)['object_static']:
                touches.append(world.env.total_periods)
            return previous(packet)
        world.detector.update = audit
        try:
            request = dict(action,ticks=forks.PUSH_BUDGET)
            review = push_checkpoints(world,request,stop,progress_m=dose,max_decisions=forks.archive.S.MAX_DECISIONS)
            pushed = push_phases.push_from_contact(world,request,forks.PUSH_BUDGET,review=review)
        finally:
            world.detector.update = previous
        if not pushed.ticks:
            raise ValueError('accepted PUSH executed no ticks')
        end = forks.archive.truth(world)
        forward = sum((y-x)*u for x,y,u in zip(initial['object'],end['object'],axis))
        result.update(status='measured', end_wall_gap_m=gap_now(world), forward_m=forward,
                      wall_at_endpoint=end['object_static'], first_static_tick=min(touches,default=None),
                      push_ticks=pushed.ticks, stop=pushed.status)
        touches.clear()
        world.detector.update = audit
        try:
            result.update(forks.followup(world,action,execute=True))
        finally:
            world.detector.update = previous
        result['first_followup_static_tick'] = min(touches,default=None)
        result['collision_free_through_contact_check'] = result['first_static_tick'] is None and not touches
        result['useful'] = forward >= .02 and not end['object_static'] and result['followup_completed']
        return result


def run(gaps, normal_offset_deg, doses):
    record, anchor = forks.record_for('dev',ROOM), reference()
    path = forks.OUT/'models'/'v2.json'
    predictor = EffectModel.from_dict(loop.read(path)['model'])
    particles = loop.donor_index(loop.packets('dev'))
    rows = []
    for gap in gaps:
        fixture, plane, angle = layout(gap, record, anchor, normal_offset_deg)
        pair = [branch(fixture,plane,angle,record,anchor,dose,predictor,particles) for dose in doses]
        if any(pair[0][k]!=pair[1][k] for k in ('public_history','start_pose','initial_contact','start_wall_gap_m','legal')):
            raise ValueError('forks do not share a starting state')
        rows.append(dict(design_gap_m=gap, fixture=fixture, branches=pair))
        print(gap,[(r['status'],round(r['start_wall_gap_m'],3),r.get('first_static_tick'),
              r.get('wall_at_endpoint'),r.get('followup_completed'),r.get('useful')) for r in pair],flush=True)
    return dict(source_room=ROOM,wall_normal_offset_deg=normal_offset_deg, collector_sha256=forks.frozen_hash(),model_sha256=loop.sha(path),
                rule_sha256=loop.sha(loop.RULE),bench_sha256=loop.sha(Path(__file__)),rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gaps',type=float,nargs='+',default=[.06,.09,.12,.15,.18,.21,.24])
    parser.add_argument('--doses', type=float, nargs=2, default=forks.PROGRESS)
    parser.add_argument('--normal-offset-deg',type=float,default=45.)
    parser.add_argument('--out',type=Path,default=OUT/'horizontal_wall_final.json')
    args=parser.parse_args()
    loop.seal(args.out,run(args.gaps,args.normal_offset_deg,args.doses))
