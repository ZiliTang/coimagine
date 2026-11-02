"""Option-level audit over the D009 v2 acceptance cases (testbed m3b)."""
from __future__ import annotations
import json, math, time, urllib.request, urllib.error
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ACC = ROOT / 'docs/agent_exchange/tasks/D009_v2_acceptance.json'
OUT = ROOT / 'outputs/agent_exchange/D011_20261030/option_audit_full.json'
URL = 'http://127.0.0.1:8765/api/run'


def grouped(commands):
    seg = []
    for a0, a1 in commands:
        if seg and seg[-1]['a0'] == a0 and seg[-1]['a1'] == a1:
            seg[-1]['decisions'] += 1
        else:
            seg.append(dict(a0=a0, a1=a1, decisions=1))
    return seg


def post(case):
    inst, seg = case['instance'], grouped(case['commands'])
    body = dict(scene=case['family'], name='audit ' + case['id'], object=inst['object'],
                robots=inst['robots'], sequence=seg,
                overrides={}, goal=dict(x=inst['goal_centre'][0], y=inst['goal_centre'][1],
                                        radius=inst['goal_radius']), controller_version='m3b')
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json'}, method='POST')
    return json.load(urllib.request.urlopen(req, timeout=180))


def main():
    acc = json.loads(ACC.read_text())
    rows = []
    for case in [c for c in acc['cases'] if c['id'].startswith('T_')]:
        t0 = time.time()
        try:
            r = post(case)
            d = json.loads((ROOT / 'outputs/testbed/runs' / f"{r['id']}.json").read_text())
        except Exception as e:
            rows.append(dict(id=case['id'], family=case['family'], error=repr(e)))
            print(case['id'], 'ERROR', e); continue
        fr = d['frames']
        contacts = []
        for i, f in enumerate(fr):
            mb = f.get('mb') or []
            if any(bool(m.get('contact')) for m in mb):
                contacts.append(i)
        goal = case['instance']['goal_centre']
        obj0, obj1 = fr[0]['object'], fr[-1]['object']
        start_dist = math.hypot(obj0[0]-goal[0], obj0[1]-goal[1])
        final_dist = math.hypot(obj1[0]-goal[0], obj1[1]-goal[1])
        progress = start_dist - final_dist
        max_angle = max(abs(math.atan2(math.sin(f['object'][2]-obj0[2]),
                                       math.cos(f['object'][2]-obj0[2]))) for f in fr)
        contact_s = len(contacts) * .05
        true_fail = bool((not contacts) or contact_s < .5 or progress < .05 or max_angle > .35)
        row = dict(id=case['id'], family=case['family'], run=r['id'], success=bool(r['success']),
                   periods=len(fr), contact_ticks=len(contacts), contact_s=round(contact_s, 3),
                   first_contact=contacts[0] if contacts else None,
                   last_contact=contacts[-1] if contacts else None,
                   progress=round(progress, 4), final_dist=round(final_dist, 4),
                   max_angle=round(max_angle, 4), task_success_distance=round(final_dist, 4),
                   option_true_fail=true_fail, wall_s=round(time.time()-t0, 2))
        rows.append(row)
        print(row)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dict(rows=rows), indent=1))
    bad = [r for r in rows if r.get('option_true_fail') or r.get('error')]
    print('\nTOTAL', len(rows), 'TRUE_FAIL', len(bad), 'TASK_SUCCESS', sum(bool(r.get('success')) for r in rows))
    print('failures')
    for r in bad: print(r)


if __name__ == '__main__':
    main()
