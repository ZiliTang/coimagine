"""Build a printable audit of the 42 frozen episodes and the planner's shadow decisions."""
from collections import Counter
from math import dist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle
import numpy as np

from tools.cap_shared_failure_report import DOC, DEST, GOAL, export, intervals, r, summarize


def annotate(key, data):
    summary = summarize(key, data)
    shadow = r.read(DEST / 'shadow' / (key + '.json'))
    original = r.read(r.OUT / 'evaluation/parent' / (key + '.json'))
    gates = Counter()
    unlocked = 0
    matches = 0
    for step, probe, actual in zip(data, shadow, original['boundaries'], strict=True):
        assert step['t'] == probe['t'] == actual['tick']
        if step['t'] < summary['tail_start']:
            continue
        gates[probe['why']] += step['dt']
        matches += probe.get('nominal') == actual['action']
        if probe.get('without_avoid') == 'accepted':
            unlocked += step['dt']
    if summary['tail_object_path'] > .10:
        kind = '末段仍在输运'
    elif summary['target_touch'] == 0:
        kind = '末段全程未接触目标'
    elif summary['target_touch'] >= .5 and summary['push'] >= .5:
        kind = '持续下达推动，目标顶住固定边界'
        assert summary['object_wall'] == 1
    else:
        kind = '少量接触，物体几乎未动'
    return dict(room=key[-3:],observed=kind,tail_start=summary['tail_start'],tail_ticks=sum(gates.values()),
                no_candidate_ticks=gates['no_candidate'],alignment_rejected_ticks=gates['alignment'],
                score_rejected_ticks=gates['score'],push_gate_pass_ticks=gates['accepted'],
                other_ticks=sum(gates.values())-sum(gates[x] for x in ('no_candidate','alignment','score','accepted')),
                gate_pass_if_avoid_removed_ticks=unlocked,nominal_matches=matches,boundaries=summary['boundaries'])


def draw_room(ax, key, fixture, data, tag):
    summary = summarize(key, data)
    tail = np.array([v['xy'] for v in data if v['t'] >= summary['tail_start']] + [data[-1]['end']])
    centers = np.array([z['xy'] for z in data]+[data[-1]['end']])
    replay = r.read(DEST/'replays'/(key+'.json'))['rows']
    robots = np.array([z['robot'] for z in replay]+[replay[-1]['robot_end']])
    for obstacle in fixture['room']['obstacles']:
        x,y=obstacle['cx'],obstacle['cy']
        w,h=obstacle['half_w'],obstacle['half_h']
        assert obstacle['angle']==0
        ax.add_patch(Rectangle((x-w,y-h),2*w,2*h,color='#555a63'))
    for obj in fixture['movable_objects']:
        assert obj['angle']==0
        w,h=obj['width'],obj['height']
        ax.add_patch(Rectangle((obj['x']-w/2,obj['y']-h/2),w,h,fill=False,
                               edgecolor='#6c7077',linestyle='--',linewidth=1))
    ax.add_patch(Circle(GOAL,fixture['goal']['radius'],color='#7bbb78',alpha=.65))
    for i,color in enumerate(('#278c9a','#c6922d')):
        ax.plot(*robots[:,i,:].T,color=color,lw=.8,alpha=.75)
    ax.plot(*centers.T,color='#354b8b',lw=2)
    ax.plot(*tail.T,color='#bb3a66',lw=2.5)
    ax.plot(*centers[0],marker='o',ms=4,color='#354b8b')
    ax.plot(*centers[-1],marker='x',ms=7,color='#bb3a66')
    ax.set(xlim=(0,fixture['room']['arena'][0]),ylim=(0,fixture['room']['arena'][1]),aspect='equal')
    ax.set_title(f"Room {key[-3:]}  |  {tag}",loc='left',fontsize=10,weight='bold')
    ax.set_xlabel(f"Push {summary['push']:.0%}   Touch {summary['target_touch']:.0%}   "
                  f"End gap {dist(centers[-1],GOAL):.2f} m",fontsize=9)
    ax.tick_params(labelsize=7)


def main():
    groups=r.read(DEST/'groups.json')['pairs']
    fixtures=r.read(r.OUT/'fixtures.json')
    keys=sorted(groups)
    data={k:intervals(k) for k in keys}
    detail=[annotate(k,data[k]) for k in keys]
    export('planner.csv',detail)
    tags={'末段全程未接触目标':'No target contact','持续下达推动，目标顶住固定边界':'Push against fixed geometry',
          '少量接触，物体几乎未动':'Brief contact, little motion','末段仍在输运':'Still transporting'}
    with PdfPages(DOC/'trajectories.pdf') as pdf:
        for first in range(0,len(keys),6):
            fig,axes=plt.subplots(3,2,figsize=(8.3,11.7))
            fig.subplots_adjust(top=.90,bottom=.065,hspace=.33,wspace=.14)
            fig.suptitle('Shared CAP failures | Full episode paths',x=.08,ha='left',weight='bold',fontsize=16)
            fig.text(.08,.928,'Box center: blue (full), pink (final 20 s), cross (end). Robots: teal and ochre. Goal: green.\n'
                     'Gray: fixed obstacle. Dashed: initial movable obstacle. Lines join decision boundaries.',fontsize=9)
            for j,ax in enumerate(axes.flat):
                k=keys[first+j];draw_room(ax,k,fixtures[k],data[k],tags[detail[first+j]['observed']])
            fig.text(.92,.025,f'{first//6+1} / 7',ha='right',fontsize=9)
            pdf.savefig(fig);plt.close(fig)

if __name__=='__main__':
    main()
