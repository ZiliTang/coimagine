"""Execute the public-history selector once at first disagreement in each fresh room."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import numpy as np
import torch

from tools import cap_route_information_fit as fit
from tools import cap_route_horizon_report as report

data=fit.data
study=data.study
MODELS=[]
PAIR=[]


def public_score(sequence,actions):
    row=dict(prefix=torch.from_numpy(np.stack(sequence)),actions=torch.from_numpy(actions))
    with torch.inference_mode():
        return np.asarray([m([row],'public',0).item() for m in MODELS])


def run(item):
    room,fixture=item
    dest=data.OUT/'deployment'/(room+'.json')
    if dest.exists():return
    policies=[study.phase.FrozenCAP(study.MODULES[s]) for s in PAIR]
    selected=None
    scores=None
    history=[]
    origin=''
    with study.phase.world(fixture) as (w,_):
        def update(h):
            for p in policies:p.update(h)
            if selected is None:history.append(data.public_frame(study.phase.cap.current_event(h)))
        update(w.history)
        while not study.ended(w):
            obs=study.observe(w)
            if selected is None:
                a=policies[0].propose(obs)
                b=data.parent.try_action(policies[1],obs)
                cmd=a
                if b is not None and b!=a:
                    actions=np.concatenate([data.action_vector(v,obs) for v in (a,b)]).astype(np.float32)
                    scores=public_score(history,actions)
                    selected=int(scores.mean()>0)
                    origin=w.history.sha256
                    cmd=(a,b)[selected]
            else:
                cmd=policies[selected].propose(obs)
            study.tick_action(w,cmd,data.parent.EXECUTOR,update,Counter())
        result=study.output(w,Counter())
    with np.load(data.OUT/'samples/fresh'/(room+'.npz')) as z:
        assert origin==str(z['origin'])
        y=z['labels']
        chosen=selected or 0
        assert result['history']==z['histories'][chosen] and result['success']==y[chosen]
        if origin:
            assert np.array_equal(z['prefix'],np.stack(history)) and np.array_equal(z['actions'],actions)
    study.save(dest,dict(room=room,A=int(y[0]),B=int(y[1]),route=chosen,success=result['success'],
                         scores=scores.tolist() if scores is not None else None,origin=origin))


def main():
    global MODELS,PAIR
    torch.set_num_threads(1)
    db,PAIR=data.plan()
    hashes=study.read(data.OUT/'locked_models.json')
    assert all(hashlib.sha256((fit.MODELS/k).read_bytes()).hexdigest()==hashes[k] for k in hashes)
    MODELS=fit.ensemble(0,'public')
    study.initialize()
    with ProcessPoolExecutor(12) as pool:
        for i,_ in enumerate(pool.map(run,db['fresh'].items()),1):
            if i%50==0:print('deploy',i,flush=True)
    rows=[study.read(data.OUT/'deployment'/(r+'.json')) for r in db['fresh']]
    a,b,m=[np.array([r[k] for r in rows]) for k in ('A','B','success')]
    size=np.array([len(f['movable_objects']) for f in db['fresh'].values()])
    rng=np.random.default_rng(61829)
    ix=np.concatenate([rng.choice(ids,(10000,len(ids))) for s in np.unique(size) if len(ids:=np.flatnonzero(size==s))],axis=1)
    valid=np.array([r['scores'] is not None for r in rows])
    scores=np.array([r['scores'] for r in rows if r['scores'] is not None]).T
    summary=dict(n=len(rows),divergences=int(sum(valid)),A=int(a.sum()),B=int(b.sum()),union=int(np.maximum(a,b).sum()),
                 mixed=int(m.sum()),switches=sum(r['route'] for r in rows),vs_A=report.paired(m,a,ix),
                 auc=report.auc((b-a)[valid],scores.mean(0)),seeds=list(data.SEEDS),
                 seed_success=[int(a.sum()+sum((b-a)[valid][p>0])) for p in scores])
    study.save(data.DOC/'deployment.json',summary)
    print(summary,flush=True)


if __name__=='__main__':
    main()
