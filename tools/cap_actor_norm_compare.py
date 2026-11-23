"""Unassisted greedy evaluation of two archived actors on the frozen replanning panel."""
import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tools import cap_actor_replan as study
from tools import cap_regional_five as geometry

run = study.run
OUT = study.OUT/'norm_comparison'
DOC = study.DOC
MODELS = {}


def episode(job):
    key, fixture, arm = job
    run.ACTOR = MODELS[arm]
    row = study.episode((key,fixture,'actor'))
    return dict(row,arm=arm)


def main():
    old, _, sids = run.initialize()
    read, save = run.ref.read, run.ref.save
    panel = read(study.OUT/'panel.json')
    protocol = read(study.OUT/'protocol.json')
    assert run.campaign.base.digest(panel)==protocol['geometry'] and sids==protocol['members']
    assert run.campaign.sha(run.ref.fit.OUT/'best_train.pt')==protocol['actor']
    scale = run.ref.scale
    OUT.mkdir(exist_ok=True)
    hashes = {}
    for arm, directory in [('plain330',scale.original.OUT),('plain800',scale.OUT)]:
        sha = run.campaign.sha(directory/'actor.pt')
        assert sha==read(directory/'FINISHED.json')['checkpoint']
        orig = read(directory/'protocol.json')
        assert orig['sids']==sids and orig['normalizers']==run.campaign.sha(scale.original.BASE/'normalizers.npz')
        MODELS[arm] = run.ref.sets.checkpoint(directory)
        hashes[arm] = sha
        (OUT/arm).mkdir(exist_ok=True)
    design=dict(plain=hashes,layernorm=protocol['actor'],panel=protocol['geometry'],members=sids,
                geometry='9 public initial geometric features, nearest among 491 teachers, training std floor 0.1, two equal-distance-rank halves')
    if (OUT/'protocol.json').exists():
        assert read(OUT/'protocol.json')==design
    save(OUT/'protocol.json',design)
    jobs=[(k,panel[k],arm) for k in sorted(panel,key=lambda k:(k[-3:],k)) for arm in MODELS
          if not (OUT/arm/(k+'.json')).exists()]
    save(OUT/'progress.json',dict(done=400-len(jobs),total=400))
    with ProcessPoolExecutor(16,mp_context=mp.get_context('fork')) as pool:
        for i,future in enumerate(as_completed([pool.submit(episode,j) for j in jobs]),1):
            row=future.result()
            save(OUT/row['arm']/(row['room']+'.json'),row)
            save(OUT/'progress.json',dict(done=400-len(jobs)+i,total=400))
    teachers=read(scale.OUT/'teachers.json')
    fixtures=old|read(scale.OUT/'natural.json')|read(scale.OUT/'variants.json')
    assert len(teachers)==491
    coords=lambda f:geometry.geom(geometry.public(f))
    train=np.stack([coords(fixtures[k]) for k in sorted(teachers)])
    spread=np.maximum(.1,train.std(0))
    rows=[]
    for k,f in sorted(panel.items()):
        ln=read(study.OUT/'episodes'/'actor'/(k+'.json'))
        repl=read(study.OUT/'episodes'/'value_every'/(k+'.json'))
        dist=float(np.linalg.norm((train-coords(f))/spread,axis=1).min())
        r=dict(room=k,objects=int(k.split('_')[1]),initial_distance=dist,layernorm=ln['success'],planned=repl['success'])
        for arm in MODELS:
            a=read(OUT/arm/(k+'.json'))
            assert a['room']==k and a['plans']==0 and a['arm']==arm
            r[arm]=a['success']
        rows.append(r)
    by_dist=sorted(rows,key=lambda r:(r['initial_distance'],r['room']))
    for i,r in enumerate(by_dist):
        r['group']='near' if i<100 else 'far'
    with (DOC/'norm_per_room.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def paired(group,a,b):
        return scale.paired(*(np.array([r[key] for r in group]) for key in (a,b,'objects')))
    result={arm:paired(rows,arm,'layernorm') for arm in MODELS}
    result['size']=paired(rows,'plain330','plain800')
    result['planned_vs_plain330']=paired(rows,'plain330','planned')
    result['planned_vs_plain800']=paired(rows,'plain800','planned')
    result['geometry']={g:paired([r for r in rows if r['group']==g],'plain800','layernorm') for g in ('near','far')}
    save(OUT/'FINISHED.json',result)


if __name__=='__main__':
    main()
