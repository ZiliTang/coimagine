"""Match the first planning boundary for the once-versus-every frequency comparison."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv

import numpy as np

from tools import cap_actor_replan as study


def first_only(job):
    key, fixture, arm = job
    # The original episode driver maps suffix 000 to landmark zero; the ID is external
    # bookkeeping and never enters the fixture, actor, simulator, or critic.
    result = study.episode(('landmark_000',fixture,arm))
    return dict(result,room=key)


def main():
    panel=study.freeze()
    study.VALUE=study.load_model()
    out=study.OUT/'matched_start'
    out.mkdir(exist_ok=True)
    arms=('value_once','flags_once')
    jobs=[]
    for arm in arms:
        (out/arm).mkdir(exist_ok=True)
        for key,fixture in panel.items():
            path=out/arm/(key+'.json')
            if path.exists():
                continue
            if int(key[-3:])%4==0:
                row=study.run.ref.read(study.OUT/'episodes'/arm/(key+'.json'))
                study.run.ref.save(path,row)
            else:
                jobs.append((key,fixture,arm))
    def progress(left):
        study.run.ref.save(out/'progress.json',dict(total=400-left,goal=400))
    progress(len(jobs))
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as workers:
        futures=[workers.submit(first_only,j) for j in jobs]
        for i,future in enumerate(as_completed(futures),1):
            r=future.result()
            study.run.ref.save(out/r['arm']/(r['room']+'.json'),r)
            progress(len(jobs)-i)
    result={}
    table=[]
    for arm in arms:
        rule=arm.split('_')[0]
        for k in sorted(panel):
            row=study.run.ref.read(out/arm/(k+'.json'))
            every=study.run.ref.read(study.OUT/'episodes'/(rule+'_every')/(k+'.json'))
            original=study.run.ref.read(study.OUT/'episodes'/'actor'/(k+'.json'))
            assert row['room']==k and row['plans']==int(every['plans']>0)
            if row['overrides']==0:
                assert row['history']==original['history']
            table.append(dict(room=k,objects=int(k.split('_')[1]),rule=rule,
                              actor=original['success'],first_only=row['success'],every=every['success']))
        group=[r for r in table if r['rule']==rule]
        def paired(a,b):
            return study.run.ref.scale.paired(*[np.array([r[f] for r in group]) for f in (a,b,'objects')])
        result[rule]=dict(vs_actor=paired('actor','first_only'),frequency=paired('first_only','every'))
    with (study.DOC/'matched_start.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    study.run.ref.save(out/'FINISHED.json',result)


if __name__=='__main__':
    main()
