"""Development-only review of PUSH predictions against later public estimates."""
from collections import Counter
import json
import math
from pathlib import Path

from experiments.planar_tasks.authoring_v2.policy_effect_contract import audit_contract


def public_reviews(data):
    steps = data['steps']
    rows = []
    for i, step in enumerate(steps):
        a = step['action']
        if step['pending'] or a['target_id'] != step['target_id'] or min(a['a0'], a['a1']) >= 32:
            continue
        permit = step.get('permit') or {}
        follow = permit.get('followup')
        if (permit.get('command') != a or permit.get('tick') != step['tick']
                or not isinstance(follow, dict) or 'envelope' not in follow):
            rows.append(dict(tick=step['tick'], status='no_comparable_issued_claim'))
            continue
        bounds = follow['envelope']
        axis = permit.get('object_axis')
        origin = permit.get('object_origin')
        needed = (bounds.get('max_forward_m'), bounds.get('lateral_m'))
        if (not axis or not origin or len(axis) != 2 or len(origin) != 2
                or not all(isinstance(v, (float, int)) and math.isfinite(v) and v >= 0 for v in needed)
                or not math.isclose(math.hypot(*axis), 1., abs_tol=1.e-4)
                or follow.get('robot') not in (0, 1) or follow.get('direction_index') not in range(8)):
            rows.append(dict(tick=step['tick'], status='no_comparable_issued_claim'))
            continue
        claim = dict(precondition={'visible': [1, 1]},
                     outcome={'along_m': [0., needed[0]], 'cross_m': [-needed[1], needed[1]]},
                     continuation={'named_push_was_issued': {'issued': [1, 1]},
                                   'goal': {'completed': [1, 1]}})
        ends = steps[i + 1] if i + 1 < len(steps) else None
        along = across = [None, None]
        if (ends and ends['tick'] == step['end_tick'] and ends['seen'] and not ends['pending']
                and ends.get('compact') and ends['active_target'] == a['target_id']):
            dx, dy = ends['compact'][0] - origin[0], ends['compact'][1] - origin[1]
            along = [dx * axis[0] + dy * axis[1]] * 2
            across = [dy * axis[0] - dx * axis[1]] * 2
        limit = step['end_tick'] + 120
        issued = any(s['tick'] <= limit and s['action']['target_id'] == a['target_id']
                     and s['action']['a' + str(follow['robot'])] // 4 == follow['direction_index']
                     for s in steps[i+1:])
        finished = data['result']['success'] and data['result']['ticks'] <= limit
        evidence = dict(precondition={'visible': [int(step['seen'])] * 2},
                        outcome={'along_m': along, 'cross_m': across},
                        continuation={'issued': [int(issued)] * 2, 'completed': [int(finished)] * 2})
        audit = audit_contract(claim, evidence)
        rows.append(dict(tick=step['tick'], status='reviewed', along_m=along[0], cross_m=across[0],
                         maximum_forward_m=needed[0], lateral_bound_m=needed[1],
                         estimated_endpoint=audit['outcome']['verdict'],
                         named_push_or_goal=audit['continuation']['verdict']))
    return rows


def write_public_checks(folder, name):
    folder = Path(folder)
    rooms = {}
    for path in sorted((folder/'feedback'/name).glob('*.json')):
        rows = public_reviews(json.loads(path.read_bytes()))
        rooms[path.stem] = dict(
            ordinary_target_pushes=len(rows),
            endpoint=dict(Counter(r.get('estimated_endpoint', r['status']) for r in rows)),
            continuation=dict(Counter(r.get('named_push_or_goal', r['status']) for r in rows)),
            first_examples=rows[:4],
            first_outside_estimate=next((r for r in rows if r.get('estimated_endpoint') == 'contradicted'), None))
    report = dict(source='public development records only; quantities are estimates, not object truth',
                  scope='ordinary target pushes with matching permits. Endpoint compares the program’s visible next centre estimate to its own forward and lateral bounds. No yaw comparison. Missing or pending next estimate is unknown.',
                  continuation='Did the program actually issue its named robot/direction by 120 ticks after the command, or reach goal in that window? A missed witness does not imply the next push was impossible.',
                  rooms=rooms)
    dest = folder/'checks'/('effects_'+name+'.json')
    dest.parent.mkdir(exist_ok=True)
    temp = dest.with_suffix('.tmp')
    temp.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)+'\n')
    temp.replace(dest)
    return report
