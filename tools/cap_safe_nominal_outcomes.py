"""Physical effects of a frozen safety CAP. Every held-out room counts once."""
import collections
import csv
import json
import math
from pathlib import Path
from statistics import median

import numpy as np


OUT = Path('outputs/semantic_rl/cap_safe_nominal_v1')
DOC = Path('docs/research/key_findings/2026-11-24_safe_nominal')


def read(path):
    return json.loads(path.read_text())


def measure(path):
    data = read(path)
    rows = data['steps']
    if not rows:
        raise ValueError(f'no command records: {path}')
    pushes = []
    for z in rows:
        a = z['action']
        if a['target_id'] == 'entity-a' and min(a['a0'], a['a1']) < 32:
            pushes.append(z)
    def moved(z):
        return math.dist(z['truth']['object'][:2], z['end_truth']['object'][:2])
    def contact(z):
        return z['truth']['object_static'] or z['end_truth']['object_static']
    ticks = sum(z['end_tick']-z['tick'] for z in rows)
    assert all(a['end_tick']==b['tick'] for a,b in zip(rows,rows[1:]))
    assert ticks == rows[-1]['end_tick'] and all(z['end_tick']>z['tick'] for z in rows)
    last = rows[-1]['end_tick']
    tail = [z for z in rows if z['tick'] >= last-400]
    return dict(
        success=int(data['result']['success']),ticks=ticks,commands=len(rows),pushes=len(pushes),
        pushes_moving_2cm=sum(moved(z)>=.02 for z in pushes),
        pushes_near_static_little_motion=sum(contact(z) and moved(z)<.02 for z in pushes),
        object_static_at_any_boundary=int(any(contact(z) for z in rows)),
        tail_path_m=round(sum(moved(z) for z in tail),4),
        stalled_failure=int(not data['result']['success'] and sum(moved(z) for z in tail)<=.10),
        all_hold_ticks=sum(z['end_tick']-z['tick'] for z in rows if z['action']['a0']==z['action']['a1']==44),
    )


def summarize(rows):
    count_keys=('success','ticks','commands','pushes','pushes_moving_2cm',
        'pushes_near_static_little_motion','object_static_at_any_boundary','stalled_failure','all_hold_ticks')
    return {**{k:sum(r[k] for r in rows) for k in count_keys},
            'median_ticks':median(r['ticks'] for r in rows)}


def main():
    final = read(OUT/'author/FINAL.json')
    arms = {'base':'P','safe':final['name']}
    pairs = {}
    for arm, name in arms.items():
        path=OUT/'runs/confirm'/name/'full'
        pairs[arm]={p.stem:measure(p) for p in sorted(path.glob('*.json'))}
    if len(pairs['base'])!=64 or set(pairs['base'])!=set(pairs['safe']):
        raise ValueError('incomplete confirmation')
    by_room=[]
    strata = read(OUT / 'room_records.json')
    for k,r in pairs['base'].items():
        row=dict(room=k, stratum=strata[k]['stratum'])
        for arm in arms:
            row.update({f'{arm}_{field}':v for field,v in pairs[arm][k].items()})
        by_room.append(row)
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/'rooms.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=by_room[0]);writer.writeheader();writer.writerows(by_room)
    delta=np.array([r['safe_success']-r['base_success'] for r in by_room])
    interval=np.quantile(np.random.default_rng(814).choice(delta,(20000,len(delta))).mean(1)*100,[.025,.975])
    output={a:summarize(list(pairs[a].values())) for a in arms}
    output.update(n=len(delta),rescued=int((delta>0).sum()),lost=int((delta<0).sum()),
                  ci_percentage_points=interval.tolist(),delta_percentage_points=float(100*delta.mean()))
    output['strata'] = {}
    for stratum in sorted({r['stratum'] for r in by_room}):
        subset = [r for r in by_room if r['stratum'] == stratum]
        output['strata'][stratum] = {
            'rooms': len(subset),
            'base_success': sum(r['base_success'] for r in subset),
            'safe_success': sum(r['safe_success'] for r in subset),
        }
    (DOC/'metrics.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':
    main()
