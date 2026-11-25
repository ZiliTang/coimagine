"""Compare issued push permits with the measured physical endpoint of the command."""
from collections import Counter
import csv
import json
from math import atan2, cos, pi, sin
from statistics import median
from tools.cap_safe_nominal_outcomes import DOC, OUT, read


def realized_followup(data, index, follow):
    steps = data['steps']
    end = steps[index]['end_tick']
    deadline = end + 120
    robot = follow['robot']
    direction = follow['direction_index']
    dx, dy = cos(direction*pi/4), sin(direction*pi/4)
    attempted = False
    for row in steps[index+1:]:
        if row['end_tick'] > deadline:
            break
        action = row['action']
        if action['target_id'] != 'entity-a' or action['a'+str(robot)]//4 != direction:
            continue
        attempted = True
        start = row['truth']['object']
        finish = row['end_truth']['object']
        if (finish[0]-start[0])*dx+(finish[1]-start[1])*dy >= .02:
            return 'matching_push_and_2cm_advance'
    if data['result']['success'] and data['terminal']['tick']<=deadline:
        return 'goal_reached_in_window'
    if data['terminal']['tick']<deadline:
        return 'window_truncated'
    return 'matching_push_without_2cm' if attempted else 'did_not_execute_matching_push'


def audit(path):
    data = read(path)
    stats, reasons = Counter(), Counter()
    entries = []
    for index,step in enumerate(data['steps']):
        a = step['action']
        permit = step.get('permit') or {}
        matches = permit.get('command') == a
        stats['commands'] += 1
        stats['matching_permit'] += matches
        reasons[(permit.get('decision'),permit.get('reason'))] += 1
        if a['target_id'] != 'entity-a' or min(a['a0'],a['a1'])>=32:
            continue
        stats['pushes'] += 1
        follow = permit.get('followup')
        if not matches or not isinstance(follow, dict) or not follow.get('envelope'):
            continue
        start = step['truth']['object']
        end = step['end_truth']['object']
        dx, dy = end[0]-start[0], end[1]-start[1]
        ux,uy = permit['object_axis']
        forward = dx*ux+dy*uy
        lateral = -dx*uy+dy*ux
        angle = atan2(sin(end[2]-start[2]),cos(end[2]-start[2]))
        limits = follow['envelope']
        limit = limits['max_forward_m']
        lat = limits['lateral_m']
        yaw = limits['yaw_rad']
        crossed = forward>limit+1e-6
        inside = -1e-6<=forward<=limit+1e-6 and abs(lateral)<=lat+1e-6 and abs(angle)<=yaw+1e-6
        stats['witnesses'] += 1
        stats['inside_endpoint'] += inside
        stats['inside_with_2cm_and_2deg_tolerance'] += (
            -.02<=forward<=limit+.02 and abs(lateral)<=lat+.02 and abs(angle)<=yaw+.034907)
        stats['forward_over'] += crossed
        stats['backward_2cm'] += forward<-.02
        stats['lateral_over'] += abs(lateral)>lat+1e-6
        stats['yaw_over'] += abs(angle)>yaw+1e-6
        stats['meaningful_forward_2cm'] += forward>=.02
        entries.append(dict(room=path.stem,tick=step['tick'],end_tick=step['end_tick'],
            witness=follow['kind'],units=permit.get('robot_units'),progress_m=round(forward,5),
            lateral_m=round(lateral,5),yaw_rad=round(angle,5),limit_m=limit,
            lateral_limit_m=lat,yaw_limit_rad=yaw,inside_endpoint=bool(inside),
            followup_120=realized_followup(data,index,follow)))
    return stats,reasons,entries


def main():
    name=read(OUT/'locked.json')['name']
    files=sorted((OUT/'runs/confirm'/name/'full').glob('*.json'))
    if len(files)!=64:
        raise ValueError('incomplete confirmation')
    stats, reasons, entries = Counter(), Counter(), []
    per_room=[]
    for p in files:
        s,r,rows=audit(p)
        stats.update(s);reasons.update(r);entries.extend(rows)
        if s['witnesses']:per_room.append(s['inside_endpoint']/s['witnesses'])
    report=dict(totals=stats,reasons={str(k):v for k,v in reasons.most_common()},
                rooms_with_push_witness=len(per_room),rooms_without_push_witness=len(files)-len(per_room),
                median_within_room_endpoint_coverage=median(per_room) if per_room else None,
                witness_kinds=Counter(x['witness'] for x in entries),
                followup_120=Counter(x['followup_120'] for x in entries))
    (DOC/'permits.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    if entries:
        with (DOC/'push_endpoints.csv').open('w') as f:
            w=csv.DictWriter(f,fieldnames=entries[0]);w.writeheader();w.writerows(entries)
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
