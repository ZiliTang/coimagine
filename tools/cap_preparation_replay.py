"""Plot the physical trajectories of four verified handoffs in a previously known rescue."""
from contextlib import contextmanager
import csv
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from tools import cap_preparation_pairs as exp
from tools.cap_preparation_report import DOC

ROOM = 'test_a_dev_2_05'


def replay(fixture, ids, arm, cut, reference):
    original_world = exp.study.phase.world
    frames = []
    @contextmanager
    def with_recorder(f):
        with original_world(f) as (w, other):
            obs = exp.study.observe(w)
            target = obs['goal']['target_id']
            goal = np.asarray(obs['goal']['xy'])
            backend = w.env.env.backend
            original = w.detector.update
            def record():
                pos = np.asarray(backend.bodies[target].position)
                xy = [tuple(r.position) for r in backend.robot_bodies]
                frames.append([w.env.total_periods, *pos, *xy[0], *xy[1], np.linalg.norm(pos-goal)])
            def wrap(packet):
                result = original(packet)
                record()
                return result
            record()
            w.detector.update = wrap
            yield w, other
    with patch.object(exp.study.phase,'world',with_recorder):
        row = exp.simulate(fixture,ids,arm,cut)
    assert row['history']==reference['history'] and row['success']==reference['success']
    arr=np.asarray(frames)
    assert np.array_equal(arr[:,0],np.arange(reference['ticks']+1))
    return arr


def main():
    p=exp.protocol()
    exp.study.initialize()
    fixture=exp.study.read(exp.study.OUT/'held.json')[ROOM]
    rows=exp.study.read(exp.OUT/'runs'/'diagnostic'/(ROOM+'.json'))
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    goal=fixture['goal']
    stats={}
    dump=[]
    for arm,label,ink in [('B','B immediately','#7f5362'),('A10','A prepares','#00766e'),
                          ('C10','C prepares','#465aa3'),('wait10','Both brake','#997018')]:
        hand=rows[arm]['hand']['tick']
        fork=rows['A']['first']['tick']
        cut=dict(rows['A']['times']['0' if arm=='B' else '10'],h=0 if arm=='B' else 10)
        arr=replay(fixture,p['pair']+[p['control']],arm,cut,rows[arm])
        for t in sorted(set([0,fork,hand,int(arr[-1,0])]+list(range(0,len(arr),10)))):
            dump.append([arm,*arr[t]])
        post=arr[hand:]
        start=post[0,-1]
        when=np.flatnonzero((start-post[:,-1])>=.02)
        stats[arm]=dict(handoff=hand,first_2cm_after=int(when[0]) if len(when) else None,
                        final_gap_m=float(arr[-1,-1]),final_tick=int(arr[-1,0]))
        ax=axs[0,0]
        ax.plot(arr[:,0],arr[:,-1],color=ink,label=label,lw=1.8)
        ax.scatter(hand,arr[hand,-1],color=ink,edgecolor='white',s=55,zorder=4)
        axs[0,1].plot(arr[:,1],arr[:,2],color=ink,lw=1.6,label=label)
        if arm in ('A10','C10'):
            ax=axs[1,0 if arm=='A10' else 1]
            seg=arr[fork:hand+1]
            for q,color,who in [(3,'#31548b','Robot 0'),(5,'#b96935','Robot 1'),(1,'#353535','Target')]:
                ax.plot(seg[:,q],seg[:,q+1],c=color,label=who)
                ax.scatter(seg[0,q],seg[0,q+1],marker='o',facecolor='white',edgecolor=color)
                ax.scatter(seg[-1,q],seg[-1,q+1],marker='x',color=color,s=52)
            ax.set_title(label+' | fork to handoff')
            ax.set_aspect('equal',adjustable='box')
            ax.legend(fontsize=8)
            ax.set(xlabel='x (m)',ylabel='y (m)')
    axs[0,0].set(xlabel='Simulation tick',ylabel='Object-to-goal distance (m)',title='Whole episode  |  dot marks handoff')
    axs[0,0].legend(fontsize=8)
    axs[0,1].set(xlabel='x (m)',ylabel='y (m)',title='Object trajectories')
    goal_xy=(goal['x'],goal['y'])
    axs[0,1].scatter(*goal_xy,marker='*',s=110,color='k',label='Goal')
    axs[0,1].add_patch(plt.Circle(goal_xy,goal['radius'],fill=False,color='k',alpha=.4))
    for axis in ('x','y'):
        lo,hi=zip(*(getattr(ax,'get_'+axis+'lim')() for ax in axs[1]))
        for ax in axs[1]:
            getattr(ax,'set_'+axis+'lim')(min(lo),max(hi))
    axs[0,1].set_aspect('equal',adjustable='box')
    for ax in axs.flat:
        ax.grid(alpha=.17)
    fig.suptitle(ROOM+'   |   Same fork, same receiving expert')
    DOC.mkdir(parents=True,exist_ok=True)
    fig.savefig(DOC/'paired_replay.png',dpi=150)
    with (DOC/'replay.csv').open('w') as f:
        writer=csv.writer(f)
        writer.writerow(['arm','tick','object_x','object_y','robot0_x','robot0_y','robot1_x','robot1_y','goal_gap_m'])
        writer.writerows(dump)
    exp.study.save(DOC/'replay.json',stats)


if __name__=='__main__':
    main()
