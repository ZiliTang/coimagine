"""Independent boundary-path accounting for mixed-option replay coverage."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def audit(record):
    outcomes = []
    pose = [[p["x"],p["y"]] for p in record["case"]["instance"]["robots"]]
    previous_pose, previous = None, [None,None]
    totals, positive, backward, switches = [0.,0.],[0.,0.],[0.,0.],[0,0]
    errors = []
    for row in record["rows"]:
        for i in (0,1):
            info, old = row[f"r{i}"], previous[i]
            if info["kind"] == "reposition":
                if (old and old["kind"] == "reposition" and old["stage"] == "move" and not old["done"]
                        and old["patch_valid"] and not old["contacting_object"]
                        and info["patch_valid"] and not info["contacting_object"] and old["tangent_world"]):
                    t = old["tangent_world"];length = math.hypot(*t)
                    inc = sum((pose[i][j]-previous_pose[i][j])*t[j]/length for j in (0,1))
                    totals[i] += inc;positive[i] += max(0.,inc);backward[i] += max(0.,-inc)
                switches[i] += bool(info["corner_event"])
                errors.extend([abs(totals[i]-info["signed_progress"]),
                               abs(positive[i]-info["positive_travel"]),abs(backward[i]-info["backward_travel"])])
                assert switches[i] == info["face_switches"]
            previous[i] = info
        previous_pose = pose
        pose = [p[:2] for p in row["true"]["robots"]]
    assert max(errors,default=0.) < 1e-8, max(errors)
    for i, logged in enumerate(record["metric"]["option_outcomes"]):
        item = dict(logged,robot=i)
        if logged["kind"] == "reposition":
            item.update(independent_signed_travel_m=totals[i], positive_m=positive[i],backward_m=backward[i],
                        observed_switch_events=switches[i],target_reached=totals[i]>=logged["target"]-1e-9)
            if logged["exit"] == "reposition_complete":
                assert item["target_reached"]
        elif logged["kind"] == "follow":
            item["maintenance_overrun_s"] = logged["recorded_progress"]-logged["target"]
        outcomes.append(item)
    rows = record["rows"]
    return {"case":record["metric"]["case"],"options":outcomes,"seconds":record["metric"]["seconds"],
            "max_reconstruction_error_m":max(errors,default=0.),"samples":len(rows),
            "object_displacement_xy_m":record["metric"]["actual_displacement_xy"],
            "object_rotation_deg":record["metric"]["rotation_deg"],
            "minimum_robot_centre_distance_m":min(math.dist(r["true"]["robots"][0][:2],r["true"]["robots"][1][:2]) for r in rows),
            "max_penetration_m":max(r["penetration"] for r in rows)}


def main(out):
    records = [json.loads(p.read_text()) for p in sorted((out/'logs').glob('*.json'))]
    results = [audit(r) for r in records if r['metric']['status']=='completed']
    report = {'version':'mixed-reposition-audit-v1','results':results,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'truth_use':'offline own-position accounting and physical outcomes only'}
    (out/'reposition_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# REPOSITION 混合组合审计','','有符号行程由本体位置差沿前一周期指令切向独立复算；换面次数是观测事件，不是真实面 ID。',
           '', '| 场景 | 机器人 0 | 机器人 1 | 请求 / 记录进度 | 耗时 s |', '|---|---|---|---|---:|']
    for r in results:
        a,b=r['options'];progress=' / '.join(f"{o['target']:.2f} → {o['recorded_progress']:.3f} {o['unit']}" if o['recorded_progress'] is not None else '未记录' for o in (a,b))
        lines.append(f"| {r['case']} | {a['action']}：{a['exit']} | {b['action']}：{b['exit']} | {progress} | {r['seconds']:.2f} |")
    lines+=['','FOLLOW 的记录维持时间与请求时长分开报告；controller exit 为完成，不等于符合请求时长。',
            '混合组合使用已有共享执行时钟，提前退出侧进入 Hold；v8 联合恢复仅对双 PUSH 生效。']
    (out/'COMBINATIONS_AUDIT.md').write_text('\n'.join(lines)+'\n')
    plot(records,out)
    print(json.dumps({'audited':len(results),'samples':sum(r['samples'] for r in results),
                      'max_error_m':max(r['max_reconstruction_error_m'] for r in results)}))


def plot(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    names=['rp_CCW_corner_1.00','rp_CW_push_opposite_0.50_0.30','rp_CCW_push_same_0.50_0.30','rp_pair_CW_CCW_0.50']
    names=[n for n in names if any(r['metric']['case']==n for r in records)]
    fig,axes=plt.subplots(2,len(names),figsize=(4*len(names),7),squeeze=False,constrained_layout=True)
    for col,name in enumerate(names):
        r=next(r for r in records if r['metric']['case']==name)
        obj=r['case']['instance']['object'];ax=axes[0,col]
        ax.add_patch(Rectangle((obj['x']-.5,obj['y']-.125),1,.25,facecolor='#acb8c7',alpha=.4))
        for i,color in ((0,'#3d8ac7'),(1,'#d49336')):
            option=r['metric']['option_outcomes'][i]
            if option['kind']=='wait':continue
            points=[v['true']['robots'][i] for v in r['rows']]
            ax.plot([p[0] for p in points],[p[1] for p in points],color=color,label=f"R{i} {option['kind']}")
            ax.scatter(points[0][0],points[0][1],color=color,marker='o');ax.scatter(points[-1][0],points[-1][1],color=color,marker='x')
            if option['kind']=='reposition':
                active=[v for v in r['rows'] if v[f'r{i}']['kind']=='reposition']
                axes[1,col].plot([v['time'] for v in active],[v[f'r{i}']['signed_progress'] for v in active],color=color,label=f'R{i} signed path')
                axes[1,col].axhline(option['target'],color=color,linestyle=':')
        ax.set(title=name,aspect='equal',xlabel='World x / m',ylabel='World y / m');ax.legend(fontsize=7);ax.grid(alpha=.2)
        axes[1,col].set(xlabel='Time / s',ylabel='Signed boundary travel / m');axes[1,col].legend(fontsize=7);axes[1,col].grid(alpha=.2)
    fig.suptitle('REPOSITION and mixed options: observed boundary travel, physical paths, and retained failures')
    fig.savefig(out/'reposition_combinations.png',dpi=150);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out.resolve())
