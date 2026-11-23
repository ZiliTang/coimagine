"""Summarize the paired development lidar probe. Distances in output tables use cm."""
import csv
import json
from pathlib import Path

import numpy as np

from tools.cap_lidar_density_probe import DOC, MODES, OUT, e, s


def write_table(name, rows):
    with (DOC / name).open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def percentiles(values):
    return [round(float(v), 4) for v in np.percentile(values, [50, 90, 95, 100])]


def cm(a, b):
    return 100 * np.linalg.norm(np.subtract(a, b))


def summarize():
    ids = e.read(s.OUT / 'summary.json')['parents']
    rooms = {key: e.read(OUT / 'rooms' / (key + '.json')) for key in ids}
    DOC.mkdir(parents=True, exist_ok=True)
    samples = [row for room in rooms.values() for row in room['original']['baseline']['samples']]
    metrics = []
    for group in ('all', 'moving', 'still'):
        frames = [row for row in samples if group == 'all' or row['moving'] == (group == 'moving')]
        common = [row for row in frames if all(row[m] is not None for m in MODES)]
        for mode in MODES:
            valid = [row[mode] for row in frames if row[mode] is not None]
            same = [row[mode] for row in common]
            metrics.append(dict(group=group, mode=mode, frames=len(frames), valid=len(valid),
                                valid_within_5deg=sum(angle <= 5 for angle, _ in valid),
                                valid_within_5deg_and_1cm=sum(angle <= 5 and dist <= .01 for angle, dist in valid),
                                common=len(common), angle_deg_p50_p90_p95_max=percentiles([a for a,d in same]),
                                support_error_cm_p50_p90_p95_max=percentiles([100*d for a,d in same])))
    fixed = []
    results = []
    for key, room in rooms.items():
        for resolution in ('72', '144'):
            for origin in ('world', 'local'):
                c, m, j = [room[case]['fixed'][resolution][origin] for case in ('original', 'mirror_y', 'jitter')]
                if c is None or m is None or j is None:
                    continue
                fixed.append(dict(room=key, rays=resolution, origin=origin, truth_error_cm=cm(c,room['original']['truth']),
                                  mirror_truth_error_cm=cm(m,room['mirror_y']['truth']),
                                  mirror_disagreement_cm=cm([c[0],4.8-c[1]],m), heading_jitter_cm=cm(c,j)))
        for mode in MODES:
            for case in ('original','mirror_y'):
                run=room[case][mode]
                results.append(dict(room=key,case=case,mode=mode,success=run['success'],ticks=run['ticks'],
                                    cpu_seconds=run['seconds'],baseline=room[case]['baseline']['success']))
    write_table('face_metrics.csv', metrics)
    write_table('initial_centers.csv', fixed)
    write_table('episodes.csv', results)
    centers=[]
    for resolution in ('72','144'):
        for origin in ('world','local'):
            rows=[r for r in fixed if r['rays']==resolution and r['origin']==origin]
            centers.append(dict(rays=resolution,origin=origin,rooms=len(rows),
                                physical_error_cm=percentiles([r[k] for r in rows for k in ('truth_error_cm','mirror_truth_error_cm')]),
                                mirror_cm=percentiles([r['mirror_disagreement_cm'] for r in rows]),
                                jitter_cm=percentiles([r['heading_jitter_cm'] for r in rows]),
                                example=next(r for r in rows if r['room']=='1_dev_3_13')))
    episodes=[]
    for mode in MODES:
        rows=[r for r in results if r['mode']==mode]
        original=[r for r in rows if r['case']=='original']
        mirror=[r for r in rows if r['case']=='mirror_y']
        episodes.append(dict(mode=mode,original=sum(r['success'] for r in original),mirror=sum(r['success'] for r in mirror),
                             flips=sum(a['success']!=b['success'] for a,b in zip(original,mirror,strict=True)),
                             recovered=sum(r['success']>r['baseline'] for r in rows),lost=sum(r['success']<r['baseline'] for r in rows),
                             mirror_cpu_ms_per_tick=round(1000*sum(r['cpu_seconds'] for r in mirror)/sum(r['ticks'] for r in mirror),3)))
    summary=dict(episodes=episodes,centers=centers,faces=metrics)
    (OUT / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


if __name__=='__main__':
    print(json.dumps(summarize(),indent=2))
