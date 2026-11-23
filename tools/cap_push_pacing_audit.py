"""Replay policy-outcome changes to audit the first explicit pacing intervention."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np

from tools import cap_push_phase_comparison as study

WORLD=study.world


def replay(job):
    sid,room,fixture,arm,reference=job
    frames=[]
    @contextmanager
    def watched(f):
        with WORLD(f) as (world,lib):
            env=world.env
            original_step=env.joint_step
            goal=np.array(env.instance.goal_centre)
            backend=env.env.backend
            def sample():
                body=backend.bodies[env.primary_id]
                pos=tuple(body.position)
                frames.append([env.total_periods,*pos,float(np.linalg.norm(np.subtract(pos,goal)))])
            sample()
            def advance(*a,**k):
                value=original_step(*a,**k)
                sample()
                return value
            env.joint_step=advance
            yield world,lib
    with patch.object(study,'world',watched):
        result=study.play(fixture,sid,arm)
    assert result==reference
    return room,sid,arm,frames


def main():
    held=study.cap.read(study.OUT/'held.json')
    root=study.OUT/'runs'/'held'
    arms=('automatic','feedback_precision')
    candidates=[];jobs=[]
    for sid in study.CAPS:
        for room,fixture in held.items():
            a,b=(study.cap.read(root/sid/arm/(room+'.json')) for arm in arms)
            if a['success']==b['success'] or not b['changes'] or b['changes'][0]['decision']!='push':
                continue
            candidates.append((sid,room,a['success'],b['success'],b['changes'][0]))
            jobs.extend((sid,room,fixture,arm,r) for arm,r in zip(arms,(a,b)))
    with ProcessPoolExecutor(5,initializer=study.init) as pool:
        results={(sid,room,arm):np.array(arr) for room,sid,arm,arr in pool.map(replay,jobs,chunksize=1)}
    rows=[]
    for sid,room,a,b,change in candidates:
        before,after=(results[sid,room,arm] for arm in arms)
        tick=change['tick']
        assert np.array_equal(before[:tick+1],after[:tick+1])
        x=before[tick:tick+41]
        y=after[tick:tick+41]
        rows.append(dict(cap=sid,room=room,tick=tick,old_success=a,new_success=b,change=change,
                         distance_at_start=float(x[0,3]),min_distance_first_2s=[float(x[:,3].min()),float(y[:,3].min())]))
    study.cap.save(study.DOC/'paired_pacing_examples.json',rows)
    import matplotlib.pyplot as plt

    win=[r for r in rows if r['new_success']]
    loss=[r for r in rows if not r['new_success']]
    show=[g[0] for g in (win,loss) if g]
    if not show:
        print('No qualifying change')
        return
    fig,axes=plt.subplots(len(show),1,figsize=(8.8,2.9*len(show)),squeeze=False,layout='constrained')
    for r,(ax,) in zip(show,axes):
        tick=r['tick']
        for mode,label,color in zip(arms,['Automatic split','Post-contact pacing'],['#4b6a8a','#268178']):
            trace=results[r['cap'],r['room'],mode]
            subset=trace[max(0,tick-20):min(len(trace),tick+181)]
            ax.plot((subset[:,0]-tick)*.05,100*subset[:,3],label=label,color=color)
        ax.axvline(0,color='#353535',ls=':',lw=1)
        ax.axhline(20,color='#999',ls='--',lw=.7)
        title=f'{r["cap"]} · {r["room"]} · '+('rescued' if r['new_success'] else 'lost')
        ax.set(title=title,ylabel='Target to goal (cm)',xlabel='Time relative to first changed PUSH (s)')
        ax.legend(frameon=False)
    fig.savefig(study.DOC/'paired_pacing.png',dpi=150)
    plt.close(fig)
    print('audited',len(rows),dict(Counter('rescued' if r['new_success'] else 'lost' for r in rows)),flush=True)


if __name__=='__main__':
    main()
