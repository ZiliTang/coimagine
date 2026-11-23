"""Post-test public report illustrations; replay truth is audit-only."""
import argparse
import math
from pathlib import Path

import numpy as np

from tools.cap_trace_value_report import CAPS, read, save, unpack


def analyze(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon

    out=Path(out).resolve()
    selections=read(out/'report/replay_cases.json')
    records=[]
    fig,axes=plt.subplots(2,2,figsize=(11,9),constrained_layout=True)
    for number,selection in enumerate(selections):
        case=selection['case_id'];pair=[]
        ax=axes.flat[number] if number<4 else None
        for condition,color in [('CAP-RR-009','#3274a1'),('GLOBAL_s17_online','#e1812c')]:
            result=unpack(out/'evaluation'/condition/case/'result.json.gz')
            scene=result['replay']['scene'];frames=result['replay']['frames']
            xy=np.array([f['entities'][scene['primary_id']][:2] for f in frames])
            distance=np.linalg.norm(xy-np.array([scene['goal']['x'],scene['goal']['y']]),axis=1)
            index=selection['common_action_prefix']
            decision=result['decisions'][index] if index<len(result['decisions']) else None
            source=None if decision is None else ([CAPS[i] for i in decision['sources'][decision['chosen']]]
                if 'sources' in decision else [condition])
            summary=result['summary']
            pair.append(dict(condition=condition,success=summary['success'],ticks=summary['ticks'],
                decisions=summary['decisions'],initial_q=summary.get('initial_value'),
                initial_distance=float(distance[0]),final_distance=float(distance[-1]),
                minimum_distance=float(distance.min()),
                first_divergent_action=None if decision is None else decision['action'],
                supporting_caps=source))
            if ax is not None:
                ax.plot(xy[:,0],xy[:,1],color=color,lw=1.5,
                        label=f"{condition}: {'success' if summary['success'] else 'failure'}, {summary['ticks']} ticks")
                ax.scatter(*xy[-1],color=color,marker='o' if summary['success'] else 'x',s=45,zorder=4)
        records.append(dict(**selection,pair=pair))
        if ax is not None:
            for obstacle in scene['obstacles']:
                points=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*[obstacle['half_w'],obstacle['half_h']]
                c,s=math.cos(obstacle['angle']),math.sin(obstacle['angle'])
                points=points@np.array([[c,s],[-s,c]])+np.array([obstacle['cx'],obstacle['cy']])
                ax.add_patch(Polygon(points,color='#bbbbbb',alpha=.65))
            ax.add_patch(Circle((scene['goal']['x'],scene['goal']['y']),scene['goal']['radius'],color='#8cc98e',alpha=.7))
            ax.scatter(*xy[0],color='black',marker='s',s=20,label='same start')
            ax.set(title=f"{selection['kind']}: {case}",xlim=(0,scene['arena'][0]),
                   ylim=(0,scene['arena'][1]),aspect='equal',xlabel='x (m)',ylabel='y (m)')
            ax.legend(fontsize=6,loc='upper right')
    fig.suptitle('Primary-object center paths: replay truth for human audit only',fontsize=12)
    fig.savefig(out/'report/paired_target_paths.png',dpi=180)
    plt.close(fig)
    save(out/'report/case_metrics.json',records)
    (out/'report/case_analysis.py').write_bytes(Path(__file__).read_bytes())
    return records


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True);args=parser.parse_args()
    for record in analyze(args.out):print(record)
