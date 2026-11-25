"""Raw-pose clearance and smoothness labels for a matched local-boundary comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

if __package__:
    from .reposition_combinations_audit import audit as path_audit
else:
    from reposition_combinations_audit import audit as path_audit


def local_position(robot,obj):
    x,y=robot[0]-obj[0],robot[1]-obj[1];c,s=math.cos(obj[2]),math.sin(obj[2])
    return (c*x+s*y,-s*x+c*y)


def geometry(record):
    # Archived logs predate geometry metadata and used 36 cm discs.
    return record.get("geometry", {"robot_radius_m": .18, "orbit_gap_m": .10})


def true_gap(robot,obj,radius):
    # Privileged object geometry is used ONLY for this offline physical label.
    x,y=local_position(robot,obj);dx,dy=max(abs(x)-.5,0),max(abs(y)-.125,0)
    distance=math.hypot(dx,dy) if dx or dy else -min(.5-abs(x),.125-abs(y))
    return distance-radius


def audit(record):
    report=path_audit(record);sides=[]
    for i,o in enumerate(report['options']):
        if o['kind']!='reposition':continue
        rows=[r for r in record['rows'] if r[f'r{i}']['kind']=='reposition']
        start=record['case']['instance']['robots'][i];previous=(start['x'],start['y'])
        total=0.;gaps=[];turns=[];last_tangent=None;speeds=[];arc_frames=0
        for row in rows:
            info=row[f'r{i}'];robot=row['true']['robots'][i];obj=row['true']['object']
            total+=math.dist(previous,robot[:2]);previous=robot[:2]
            gaps.append(true_gap(robot,obj,geometry(record)['robot_radius_m']))
            t=info['tangent_world']
            if t and last_tangent:
                turns.append(abs(math.atan2(last_tangent[0]*t[1]-last_tangent[1]*t[0],last_tangent[0]*t[0]+last_tangent[1]*t[1])))
            if t:last_tangent=t
            if info['stage']=='move' and row['time']>.5:
                speeds.append(math.hypot(robot[3],robot[4]))
            arc_frames+=(info.get('boundary_follow') or {}).get('mode')=='corner_arc'
        final=rows[-1]['true'];local=local_position(final['robots'][i],final['object'])
        initial_obj=record['case']['instance']['object'];initial_local=local_position((start['x'],start['y']),
            (initial_obj['x'],initial_obj['y'],initial_obj['angle']))
        opposite=(abs(local[0])<=.5 and local[1]*(-1 if initial_local[1]<0 else 1)<-.125)
        sides.append({'robot':i,'actual_centre_path_m':total,'min_true_body_gap_m':min(gaps),
                      'max_true_body_gap_m':max(gaps),'command_tangent_total_variation_deg':math.degrees(sum(turns)),
                      'max_command_tangent_step_deg':math.degrees(max(turns,default=0.)),
                      'moving_speed_mean_m_s':statistics.mean(speeds) if speeds else None,
                      'moving_speed_cv':statistics.pstdev(speeds)/statistics.mean(speeds) if speeds and statistics.mean(speeds)>1e-8 else None,
                      'corner_arc_frames':arc_frames,'final_object_local_xy_m':local,
                      'opposite_long_face_reached':opposite})
    report.update(rounded=record['metric']['variant'],physical=sides)
    return report


def main(out):
    records=[json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))]
    results=[audit(r) for r in records if r['metric']['status']=='completed']
    dimensions = {(geometry(r)['robot_radius_m'],geometry(r)['orbit_gap_m']) for r in records}
    if len(dimensions) != 1:
        raise ValueError('a matched arc comparison requires one robot geometry')
    radius, orbit_gap = dimensions.pop()
    sensitivity=[]
    for base in ('arc_rp_CW_corner_1.00','arc_rp_CCW_corner_1.00'):
        for v in (False,True):
            group=[r for r in results if r['rounded']==v and r['case'] in [base,base+'_dx-0.01',base+'_dx+0.01']]
            xy=[r['physical'][0]['final_object_local_xy_m'] for r in group]
            sensitivity.append({'base':base,'rounded':v,'samples':len(group),
                'max_endpoint_separation_m':max(math.dist(a,b) for a in xy for b in xy)})
    report={'version':'reposition-arc-audit-v1','results':results,'local_sensitivity':sensitivity,
            'ideal_geometry':{'object_size_m':[1.,.25],'robot_radius_m':radius,'clearance_m':orbit_gap,
                              'front_midpoint_to_opposite_face_m':.5+.25+math.pi*(radius+orbit_gap),
                              'near_corner_to_opposite_face_m':.05+.25+math.pi*(radius+orbit_gap)},
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'truth_use':'offline labels only; runtime accepts own scan/pose/contact and public clock'}
    (out/'arc_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# REPOSITION 圆弧换面：匹配请求的独立审计','','实际轨迹长度、有符号计程、真实实体间隙及最终位置分别报告。1 m 不等于绕至反面。','',
           '| 场景 | 圆弧 | RP 退出 | 实际本体路径 m | 真实间隙 min/max m | 最大切向指令跳变 ° | 耗时 s |',
           '|---|---|---|---|---|---|---:|']
    for r in results:
        physics=r['physical'];paths=' / '.join(f"{p['actual_centre_path_m']:.4f}" for p in physics)
        gap=' / '.join(f"{p['min_true_body_gap_m']:.4f}–{p['max_true_body_gap_m']:.4f}" for p in physics)
        jump=' / '.join(f"{p['max_command_tangent_step_deg']:.2f}" for p in physics)
        exits=' / '.join(o['exit'] for o in r['options'] if o['kind']=='reposition')
        lines.append(f"| {r['case']} | {r['rounded']} | {exits} | {paths} | {gap} | {jump} | {r['seconds']:.2f} |")
    (out/'ARC_AUDIT.md').write_text('\n'.join(lines)+'\n')
    plot(records,out)
    print(json.dumps({'audited':len(results),'periods':sum(r['samples'] for r in results),
                      'max_path_reconstruction_error_m':max(r['max_reconstruction_error_m'] for r in results),
                      'sensitivity':sensitivity}))


def plot(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    name='arc_rp_CCW_corner_1.00'
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for v,color in ((False,'#c08a38'),(True,'#218d70')):
        r=next(r for r in records if r['metric']['case']==name and r['metric']['variant']==v)
        rows=[x for x in r['rows'] if x['r0']['kind']=='reposition'];t=[x['time'] for x in rows]
        label='finite-boundary arc' if v else 'legacy face tangent'
        poses=[x['true']['robots'][0] for x in rows]
        axes[0,0].plot([p[0] for p in poses],[p[1] for p in poses],color=color,label=label)
        axes[0,0].scatter(poses[-1][0],poses[-1][1],color=color,marker='x')
        axes[0,1].plot(t,[true_gap(p,x['true']['object'],geometry(r)['robot_radius_m']) for p,x in zip(poses,rows)],color=color,label=label)
        moving=[x for x in rows if x['r0']['stage']=='move' and x['time']>.5]
        axes[0,2].plot([x['time'] for x in moving],[math.hypot(*x['true']['robots'][0][3:5]) for x in moving],color=color,label=label)
        vectors=[x for x in rows if x['r0']['tangent_world']];variation=0.;vs=[0.]
        for a,b in zip(vectors,vectors[1:]):
            a,b=a['r0']['tangent_world'],b['r0']['tangent_world'];variation+=abs(math.atan2(a[0]*b[1]-a[1]*b[0],a[0]*b[0]+a[1]*b[1]));vs.append(math.degrees(variation))
        axes[1,0].plot([x['time'] for x in vectors],vs,color=color,label=label)
        path=0.;previous=(r['case']['instance']['robots'][0]['x'],r['case']['instance']['robots'][0]['y']);ds=[]
        for p in poses:path+=math.dist(previous,p[:2]);previous=p[:2];ds.append(path)
        axes[1,1].plot(t,ds,color=color,label=label+' actual path')
        axes[1,1].plot(t,[x['r0']['signed_progress'] for x in rows],color=color,linestyle=':',label=label+' signed credit')
        for dx,marker in (('', 'o'),('_dx-0.01','v'),('_dx+0.01','^')):
            q=next(r for r in records if r['metric']['case']==name+dx and r['metric']['variant']==v)
            x=q['rows'][-1]['true']['robots'][0];axes[1,2].scatter(x[0],x[1],color=color,marker=marker,label=label+(dx or ' base'))
    axes[0,0].add_patch(Rectangle((2.,1.875),1.,.25,color='#aeb9c9',alpha=.4))
    dimensions = geometry(records[0])
    offset = dimensions["robot_radius_m"] + dimensions["orbit_gap_m"]
    for cx,cy,start in ((3.,1.875,-math.pi/2),(3.,2.125,0.)):
        angles=[start+i*math.pi/2/100 for i in range(101)]
        axes[0,0].plot([cx+offset*math.cos(a) for a in angles],[cy+offset*math.sin(a) for a in angles],color='gray',linestyle='--')
    axes[0,0].set(xlim=(2.75,3.45),ylim=(1.50,2.48),aspect='equal',title='Matched 1 m request: centre paths',xlabel='World x / m',ylabel='World y / m')
    axes[0,1].axhline(dimensions["orbit_gap_m"],color='gray',linestyle='--');axes[0,1].set(title='True body clearance (offline)',xlabel='Time / s',ylabel='Gap / m')
    axes[0,2].set(title='Speed during MOVE, after 0.5 s',xlabel='Time / s',ylabel='Body speed / m/s')
    axes[1,0].set(title='Command direction variation',xlabel='Time / s',ylabel='Accumulated absolute turn / degrees')
    axes[1,1].set(title='Distance budget kept unchanged',xlabel='Time / s',ylabel='Distance / m')
    axes[1,2].set(title='+/-1 cm start perturbations',xlabel='Final world x / m',ylabel='Final world y / m')
    for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=6)
    fig.suptitle('Rounded own-scan boundary following: no hidden polygon and no extra transfer budget')
    fig.savefig(out/'reposition_arc.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
