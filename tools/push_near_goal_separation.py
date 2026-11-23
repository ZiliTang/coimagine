"""Outcome-blind near-goal prefix selection for paired option separation."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from tools import push_boundary_separation as comparison
from tools import push_contract_bench as bench
from tools import cap_option_semantic_factorial as study
from experiments.planar_tasks.semantic_rl.candidate_semantics import fitted


def cases():
    result = []
    for split, fixtures in [('development',bench.DEV),('validation',bench.HELD)]:
        for room in sorted(fixtures):
            records = bench.archive('s025', room)
            for index, (before, step) in enumerate(zip(records, records[1:]), 1):
                cmd = step['a']
                if cmd['target_id'] != 'entity-a' or len(bench.translation(cmd)) != 1 or before['xy'] is None:
                    continue
                _, axis = study.push_phases.pusher(cmd)
                delta = np.subtract(before['goal'],before['xy'])
                gap = np.linalg.norm(delta)
                if gap <= .48 and delta @ axis >= .82 * gap:
                    result.append((split, room, index))
                    break
    return result


def measure(job):
    split, room, index = job
    fixture = (bench.DEV if split=='development' else bench.HELD)[room]
    records = bench.archive('s025',room)
    cmd = records[index]['a']
    robot, axis = study.push_phases.pusher(cmd)
    axis = np.asarray(axis)
    tangent = np.array([-axis[1],axis[0]])
    goal = np.array([fixture['goal'][key] for key in ('x','y')])
    radius = fixture['goal']['radius']
    rows = []
    for off,(n,t) in bench.OFFSETS.items():
        row = dict(room=room, prefix=room, split=split, offset=off, retimed=0)
        prefix = None
        for arm in study.CONTROLLERS:
            with bench.world(fixture) as (w,_):
                for step in records[:index]:
                    w.step(step['a'],remaining_ticks=study.remaining(w))
                    assert w.env.total_periods==step['tick']
                if off!='zero' and bench.perturb(w, robot, n*axis+t*tangent) is not None:
                    break
                obs = study.observe(w)
                try:
                    bench.validate_action(cmd,obs)
                except ValueError:
                    break
                if prefix is None:
                    prefix = w.history.sha256
                assert w.history.sha256 == prefix
                start = np.array(bench.position(w.env,cmd['target_id'])[:2])
                if arm=='coupled':
                    geom = bench.geometry(obs,robot,cmd['target_id'],axis)
                    box = fitted(obs,cmd['target_id'])
                    delta = np.array([0.,0.]) if box is None else goal - [box['x'],box['y']]
                    row['features'] = [cmd['ticks']*.05,bench.ACTIONS[cmd[f'a{robot}']].speed,
                        *[float(geom[k]) for k in ('visible','points','contact','goal_gap','tangent','station_clear','peer')],
                        float(box is not None),float(np.linalg.norm(delta)),float(delta@axis),float(delta@tangent)]
                hits = []
                detector = w.detector.update

                def contact(packet):
                    status = detector(packet)
                    hits.append(w.env.active_target==cmd['target_id'] and packet['robots'][robot]['contacting_object'])
                    return status

                w.detector.update = contact
                counts = Counter()
                study.tick_action(w,cmd,arm,lambda history:None,counts)
                final = np.array(bench.position(w.env,cmd['target_id'])[:2])
                along = float((final-start)@axis)
                lateral = abs(float((final-start)@tangent))
                stable = any(all(hits[i:i+3]) for i in range(len(hits)-2))
                row[arm] = int(stable and along>=bench.CONTACT_M and lateral<=max(bench.SIDE_M,.75*along))
                row['inside_'+arm] = int(np.linalg.norm(final-goal)<=radius)
                if arm=='split_precision':
                    row['retimed'] = counts['retimed']
            if arm=='split_precision':
                rows.append(row)
    return rows


def main():
    torch.set_num_threads(1)
    path = comparison.OUT/'near_goal.json'
    if not path.exists():
        jobs = cases()
        with ProcessPoolExecutor(12) as pool:
            rows = [r for batch in pool.map(measure,jobs) for r in batch]
        bench.save(path,rows)
    rows = bench.read(path)
    result = {}
    for label in ('contract','ends_inside_goal'):
        part = [dict(r) for r in rows]
        if label=='ends_inside_goal':
            for r in part:
                for arm in study.CONTROLLERS:
                    r[arm]=r['inside_'+arm]
        result[label] = comparison.analysis(part, splits=('validation',))['validation']
    bench.save(comparison.DEST/'near_goal.json',result)
    print(result)


if __name__=='__main__':
    main()
