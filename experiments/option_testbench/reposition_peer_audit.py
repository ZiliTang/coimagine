"""Independent own-travel, peer separation, static clearance and retained failures."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from coimagine.random_start import geometry as G, spec
if __package__:
    from .reposition_arc_audit import audit as boundary_audit
else:
    from reposition_arc_audit import audit as boundary_audit


def audit(record):
    result=boundary_audit(record)
    result['peer_detour']=result.pop('rounded')
    obstacles=record['case']['instance']['obstacles']
    for physical in result['physical']:
        i=physical['robot'];rows=[r for r in record['rows'] if r[f'r{i}']['kind']=='reposition']
        physical['minimum_static_body_gap_m']=min((G.circle_polygon_gap(tuple(r['true']['robots'][i][:2]),
            spec.ROBOT_RADIUS,G.box_polygon(b['cx'],b['cy'],b['angle'],b['half_w'],b['half_h']))
            for r in rows for b in obstacles),default=None)
        physical['peer_route_frames']=sum(bool((r[f'r{i}'].get('peer_detour') or {}).get('active')) for r in rows)
        physical['observed_plane_guard_frames']=sum((r[f'r{i}'].get('peer_detour') or {}).get('object_guard')=='observed_plane' for r in rows)
        stops=[r for r in rows if r[f'r{i}']['exit_reason']=='blocked_static_route']
        physical['static_stop_time_s']=stops[0]['time'] if stops else None
    return result


def main(out):
    paths=sorted((out/'logs').glob('*.json'));records=[json.loads(p.read_text()) for p in paths]
    results=[audit(r) for r in records if r['metric']['status']=='completed']
    counts={str(v):dict(Counter(o['exit'] for r in results if r['peer_detour']==v
                              for o in r['options'] if o['kind']=='reposition')) for v in (False,True)}
    historical=out.parents[1]/'reposition_scales_v2/round_002/logs'
    reproduced=[]
    for r in records:
        if r['metric']['variant']:continue
        old=historical/(r['metric']['case'].replace('detour_','scale2_')+'__current.json')
        if old.exists():
            frozen=json.loads(old.read_text())
            equal=([x['true'] for x in r['rows']]==[x['true'] for x in frozen['rows']]
                   and [x['forces'] for x in r['rows']]==[x['forces'] for x in frozen['rows']])
            assert equal,r['metric']['case'];reproduced.append(r['metric']['case'])
    summary={'version':'peer-outer-v2-offline-audit','results':results,'reposition_exits':counts,
             'old_physical_branches_exactly_reproduced':reproduced,
             'max_reconstruction_error_m':max(r['max_reconstruction_error_m'] for r in results),
             'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),
             'logs_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
             'audit_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             'truth_use':'offline independent travel, body clearances and final geometry only'}
    (out/'peer_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# REPOSITION peer detour audit','','Distance completion is signed OWN route travel, including the detour. It is not opposite-face arrival.',
           'Static rejection is a local own-scan swept-disc check, not global path infeasibility. Missing observation is not free space.',
           '', '| Case | Detour | RP exits | Min peer body gap / m | Seconds |', '|---|---|---|---:|---:|']
    for r in results:
        lines.append(f"| {r['case']} | {r['peer_detour']} | {' / '.join(o['exit'] for o in r['options'] if o['kind']=='reposition')} | {r['minimum_robot_centre_distance_m']-2*spec.ROBOT_RADIUS:.4f} | {r['seconds']:.2f} |")
    lines+=['',f'RP exits: `{json.dumps(counts)}`',f"Max signed-travel reconstruction error: {summary['max_reconstruction_error_m']:.3g} m.",
            f'Exactly reproduced old physical branches: {len(reproduced)}.',
            'The two stationary-peer branches retain patch_lost. Short/long route completion does not promise a full face change.',
            'In the peer+static obstruction cases RP stops first; the other PUSH continues under the existing independent execution rule.']
    (out/'PEER_AUDIT.md').write_text('\n'.join(lines)+'\n');plot(records,out)
    print(json.dumps({'audited':len(results),'rp_exits':counts,'reproduced':len(reproduced),
                      'max_error_m':summary['max_reconstruction_error_m']}))


def plot(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    names=['detour_rp_CCW_push_same_1.80','detour_rp_CW_push_opposite_1.80',
           'detour_rp_CCW_static_gap_0.26','detour_rp_CCW_wait_same_1.80']
    fig,axes=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
    for col,name in enumerate(names):
        for enabled,color in ((False,'#b67b39'),(True,'#218b78')):
            r=next(r for r in records if r['metric']['case']==name and r['metric']['variant']==enabled)
            rows=[x for x in r['rows'] if x['r0']['kind']=='reposition'];label='outer routing' if enabled else 'local repulsion'
            p=[x['true']['robots'][0] for x in rows]
            axes[0,col].plot([x[0] for x in p],[x[1] for x in p],color=color,label=label)
            axes[0,col].scatter(p[-1][0],p[-1][1],color=color,marker='x')
            axes[1,col].plot([x['time'] for x in rows],[x['r0']['signed_progress'] for x in rows],color=color,label=label)
            axes[1,col].axhline(r['metric']['target'],color='gray',linestyle=':',alpha=.6)
            if enabled:
                instance=r['case']['instance'];obj=instance['object'];mate=instance['robots'][1]
                axes[0,col].add_patch(Rectangle((obj['x']-.5,obj['y']-.125),1,.25,facecolor='#acb8c7',alpha=.4))
                axes[0,col].scatter(mate['x'],mate['y'],color='#df9b45',s=55,label='peer start')
                for b in instance['obstacles']:
                    axes[0,col].add_patch(Rectangle((b['cx']-b['half_w'],b['cy']-b['half_h']),2*b['half_w'],2*b['half_h'],facecolor='#a77478',alpha=.6))
        axes[0,col].set(title=name.replace('detour_rp_',''),aspect='equal',xlabel='World x / m',ylabel='World y / m')
        axes[1,col].set(xlabel='Time / s',ylabel='Signed own route / m')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.suptitle('Outer peer routing: matched budgets, static rejection, retained visibility failure')
    fig.savefig(out/'reposition_peer_detour.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
