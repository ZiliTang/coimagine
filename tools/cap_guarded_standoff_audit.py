"""Test one contact pre-alignment distance on the 60 frozen development PUSH suffixes.

Both arms execute the same CCW suffix. Only a newly constructed approach skill
in the short arm gets the existing 5-cm pre-alignment hook in place of 15 cm.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from unittest.mock import patch

from experiments.option_rl_v19 import unknown_shape
from experiments.planar_tasks.semantic_rl import push_contact_recovery
from tools.guarded_calibration_v1 import cap_guarded_calibration_forks as forks


OUTPUT = forks.OUT / 'change_face_standoff_dev.json'


def run_suffix(room, anchor, dose, shorten):
    counts = Counter()
    first, last = {}, {}
    controller = unknown_shape.UnknownShapePush
    step = controller._step_impl
    unmodified_suffix = forks.followup
    instrumented = set()

    def stepped(skill, observation, dt):
        if skill not in instrumented:
            instrumented.add(skill)
            if shorten:
                recovery = push_contact_recovery.ContactRecovery(wait=0, strategy='standoff', reach=.05)
                recovery.reduce_standoff(skill)
            navigate = skill._allocated_navigate

            def measured(obs, *args):
                result = navigate(obs, *args)
                counts['navigation_ticks'] += 1
                counts[skill.stage + '_ticks'] += 1
                counts['repulsion_ticks'] += int(skill.avoidance_active)
                evidence = dict(time=obs.time, stage=skill.stage, tangential_error_m=skill.candidate_err_t,
                                face_gap_m=skill.gap)
                if not first:
                    first.update(evidence)
                last.update(evidence)
                return result

            skill._allocated_navigate = measured
        return step(skill, observation, dt)

    def suffix(world, action, *, execute):
        robot, _ = forks.push_phases.pusher(action)
        previous = world.detector.update

        def measure_contacts(feedback):
            contacts = world.env.env.backend.contacts()
            pairs = {c['pair'] for c in contacts if c['robot_id'] == robot}
            counts['observed_suffix_ticks'] += 1
            counts['peer_contact_ticks'] += int('robot_robot' in pairs)
            counts['static_contact_ticks'] += int('robot_wall' in pairs)
            counts['other_object_contact_ticks'] += int(any(
                c['robot_id'] == robot and c['pair'] == 'robot_object' and c['entity_id'] != action['target_id']
                for c in contacts))
            return previous(feedback)

        with patch.object(controller, '_step_impl', stepped), patch.object(world.detector, 'update', measure_contacts):
            return unmodified_suffix(world, action, execute=execute)

    with patch.object(forks, 'followup', suffix):
        result = forks.evaluate(forks.development_rooms()[room], forks.record_for('dev', room),
                                anchor, dose, execute_followup=True)
    if counts['observed_suffix_ticks'] != result['followup_ticks']:
        raise ValueError('incomplete physical suffix monitoring')
    return result, dict(counts=counts, first=first or None, last=last or None)


def paired_room(room):
    packet = forks.archive.read(forks.OUT/'forks/dev'/(room + '.json'))
    if packet['collector_sha256'] != forks.frozen_hash():
        raise ValueError('development archive collector differs')
    anchors = {a['anchor_id']: a for a in packet['anchors']}
    rows = []
    for row in packet['rows']:
        name, dose = row['anchor_id'], row['progress_m']
        baseline, before = run_suffix(room, anchors[name], dose, False)
        if baseline != row['outcome']:
            raise ValueError(f'baseline replay changed at {room}/{name}/{dose}')
        alternate, after = run_suffix(room, anchors[name], dose, True)
        if any(alternate[k] != value for k, value in baseline.items()
               if k not in ('orbit_ticks', 'approach_ticks', 'followup_reason', 'followup_ticks', 'followup_completed')):
            raise ValueError(f'prefix or PUSH endpoint changed at {room}/{name}/{dose}')
        rows.append(dict(room=room, anchor_id=name, progress_m=dose, history=row['public_history'],
                         baseline_matches_archive=True, outcomes_identical=baseline==alternate,
                         baseline=dict(nav=before, outcome=baseline), shorter=dict(nav=after, outcome=alternate)))
    return rows


def summarize(rows):
    result = {}
    for key in ('baseline', 'shorter'):
        def ok(r): return r[key]['outcome']['followup_completed']
        result[key] = dict(candidates=len(rows), successful=sum(map(ok, rows)),
                           success_rooms=sorted({r['room'] for r in rows if ok(r)}),
                           success_anchors=len({(r['room'],r['anchor_id']) for r in rows if ok(r)}),
                           called=sum(bool(r[key]['nav']['counts'].get('navigation_ticks')) for r in rows),
                           pusher_contact_candidates={k:sum(bool(r[key]['nav']['counts'].get(k)) for r in rows)
                                for k in ('peer_contact_ticks', 'static_contact_ticks', 'other_object_contact_ticks')},
                           success_early=sum(ok(r) for r in rows if r['anchor_id']=='early'),
                           success_late=sum(ok(r) for r in rows if r['anchor_id']=='late'))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3,choices=range(1,9))
    args=parser.parse_args()
    rooms=sorted(forks.development_rooms())
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows=[r for batch in pool.map(paired_room,rooms) for r in batch]
    result=summarize(rows)
    output=dict(protocol='guarded-development-ccw-change-contact-prealignment-v1',
                reach_m=dict(baseline=.15,shorter=.05),
                collector_sha256=forks.frozen_hash(), source_sha256=forks.archive.sha(Path(__file__)),
                underlying_sources={module.__name__:forks.archive.sha(Path(module.__file__))
                                    for module in (unknown_shape,push_contact_recovery)},
                rooms=rooms, summary=result, rows=rows)
    forks.write_json(OUTPUT,output)
    print(result)


if __name__ == '__main__':
    main()
