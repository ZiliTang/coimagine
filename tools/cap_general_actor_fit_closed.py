"""Held-room closed-loop comparison for the hard actor selected by training fit."""
import csv
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from tools import cap_general_actor_fit_diagnostic as fit
from tools.cap_general_actor_800 import campaign, dq, original, paired, read, save
from tools import cap_general_actor_800 as scale
from tools import cap_general_actor_sets as sets


OUT = fit.OUT/'closed_loop'


def run():
    torch.set_num_threads(1)
    _,rooms,sids=scale.inputs()
    selection=read(fit.OUT/'FINISHED.json')
    protocol=read(scale.OUT/'protocol.json')
    assert selection['best_train_arm']=='layernorm' and protocol['held']==campaign.base.digest(rooms)
    assert sids==protocol['sids'] and campaign.sha(original.BASE/'normalizers.npz')==protocol['normalizers']
    baseline=sets.OUT/'per_scene.csv'
    with baseline.open() as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==len(rooms)==200 and {r['room'] for r in rows}==rooms.keys()
    OUT.mkdir(exist_ok=True)
    settings=dict(model=selection['best_sha'],panel=protocol['held'],sids=sids,baseline=campaign.sha(baseline),
                  max_ticks=1800,max_decisions=300,workers=12)
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==settings
    save(OUT/'protocol.json',settings)
    if (OUT/'FINISHED.json').exists():
        return
    dq.MODEL=fit.load_best()
    dq.MODULES=[dq.load_native((campaign.CAM/'semantics'/(sid+'.py')).read_text())[0] for sid in sids]
    def progress(stage,**info):
        save(OUT/'progress.json',dict(stage=stage,**info))
    with TemporaryDirectory(dir='/dev/shm',prefix='coimagine-ln-eval-') as tmp:
        dq.OUT=Path(tmp)
        dq.progress=progress
        progress('闭环评估',completed=0,total=len(rooms))
        episodes=dq.batch([(k,f,'layernorm',None,None,'fresh') for k,f in sorted(rooms.items())],12,'闭环评估')
        by_room={r['case_id']:r for r in episodes}
        assert len(episodes)==len(by_room)==len(rooms) and by_room.keys()==rooms.keys()
        for row in rows:
            e=by_room[row['room']]
            assert 0<e['ticks']<=1800 and 0<e['boundaries']<=300
            if e['success']:
                assert e['termination']=='success'
            else:
                assert e['termination']=='task_budget_exhausted' or (e['termination'] is None and (e['ticks']==1800 or e['boundaries']==300))
            row.update(layernorm=int(e['success']),ln_ticks=e['ticks'],ln_decisions=e['boundaries'],
                       ln_termination=e['termination'] or 'budget',ln_history=e['history'])
    with (OUT/'per_scene.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def values(name):
        return np.array([int(r[name]) for r in rows])
    union=values('union').astype(bool)
    best=read(campaign.OUT/'FINISHED.json')['best_dev_sid']
    result=dict(best_dev_sid=best,union=int(union.sum()),
                paired={key:paired(values(key),values('layernorm'),values('objects')) for key in ('actor800','sets',best,'mix')},
                coverage={key:dict(inside=int(values(key)[union].sum()),outside=int(values(key)[~union].sum()))
                          for key in ('actor800','layernorm','sets')})
    save(OUT/'FINISHED.json',result)
    progress('完成',completed=len(rows),total=len(rows))
    print(result,flush=True)


if __name__=='__main__':
    run()
