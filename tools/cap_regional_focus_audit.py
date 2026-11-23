"""Post hoc audit of late coverage concentration and regional author sampling."""
import numpy as np
from tools import cap_regional_five as e


def main():
    data=e.read(e.OUT/'regions.json');ids=data['ids'];x=np.array(data['x'])
    z=(x-x.mean(0))/np.maximum(.1,x.std(0));dist=np.linalg.norm(z[:,None]-z[None,:],axis=2)
    nearest=np.argsort(dist,axis=1,kind='stable')[:,:24]
    radii=dist[np.arange(len(ids)),nearest[:,-1]]
    candidates=nearest[radii<=np.median(radii)]
    root=e.OUT/'regional';sw=e.read(root/'study.json')['sweeps'];past=[];cov=set()
    rows=[];previous_key=''
    for i,rev in enumerate(sw):
        center,focus=e.region(dict(sweeps=past));assert focus==rev['focus'] and center==rev['center']
        opened=set(ids)-cov; probes=[]
        for path in sorted((root/'logs').glob('*_test_*')):
            if previous_key<path.name<rev['key']:
                probes.extend(e.read(p)['id'] for p in path.glob('*.json'))
        added=set(rev['wins'])-cov
        rows.append(dict(round=i+1,center=center,open=len(opened),open_in_focus=len(opened&set(focus)),
                         probes=len(probes),in_focus=sum(k in focus for k in probes),
                         unresolved_probes=sum(k in opened for k in probes),distinct_probed=len(set(probes)),
                         distinct_unresolved_probed=len(set(probes)&opened),added=len(added),added_in_focus=len(added&set(focus))))
        previous_key=rev['key'];cov.update(rev['wins']);past.append(rev)
    y=~np.isin(ids,list(set().union(*(r['wins'] for r in sw[:4]))))
    obj=np.array([int(s.split('_')[1]) for s in ids])
    observ=int(y[candidates].sum(1).max())
    permutations={}
    for name in ['all','by_movable_count']:
        rng=np.random.default_rng(1977);null=[]
        for _ in range(5000):
            if name=='all':v=rng.permutation(y)
            else:
                v=y.copy()
                for k in [2,3,4,5]:v[obj==k]=rng.permutation(y[obj==k])
            null.append(int(v[candidates].sum(1).max()))
        permutations[name]=dict(mean=float(np.mean(null)),q05_q95=np.quantile(null,[.05,.95]).tolist(),fraction_ge=float(np.mean(np.array(null)>=observ)))
    result=dict(rounds=rows,pre_round5_unresolved=int(y.sum()),max_open=observ,permutations=permutations)
    e.save(e.OUT/'focus_audit.json',result)
    print(result)

if __name__=='__main__':main()
