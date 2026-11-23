"""Seal each author campaign and evaluate fixed libraries on common unseen rooms."""
import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from tools import cap_regional_five as e


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def frozen():
    policies={}
    seals={}
    for arm in e.ARMS:
        root=e.OUT/arm
        s=e.read(root/'study.json'); c=e.host.RoomCampaign(root)
        assert s['final'] and not s['active']
        assert len(s['sweeps'])==5 if arm!='single' else 0<len(s['sweeps'])<=5
        cp=e.read(root/'campaign.json')
        assert cp['limits']['attempts']==960 and cp['used']['attempts']<=960 and cp['used']['execution_ticks']<=1728000
        for rev in s['sweeps']:
            sid=rev['sid'];folder=root/'logs'/rev['key']
            records=[e.read(p) for p in sorted(folder.glob('*.json'))]
            assert len(records)==len({r['id'] for r in records})==160 and all(r['complete'] for r in records)
            assert sorted(r['id'] for r in records if r['success'])==rev['wins']
        if cp['status']!='frozen':c.freeze()
        seals[arm]=c.verify_frozen()
        for sid in s['final']:
            src=root/'semantics'/(sid+'.py')
            assert e.digest(src.read_text())==next(r['sha256'] for r in cp['submissions'] if r['id']==sid)
            policies[arm+'_'+sid]=sha(src)
    ledger=dict(policies=policies,seals=seals,protocol=sha(e.OUT/'protocol.json'),panel=sha(e.OUT/'held.json'),host=sha(e.__file__),
                evaluator=sha(__file__))
    if (e.OUT/'frozen.json').exists():assert ledger==e.read(e.OUT/'frozen.json')
    else:e.save(e.OUT/'frozen.json',ledger)
    return policies


def paired(a,b,counts,seed=17):
    delta=b.astype(int)-a.astype(int)
    rng=np.random.default_rng(seed);draws=np.zeros(10000)
    for k in np.unique(counts):
        d=delta[counts==k];draws+=d[rng.integers(len(d),size=(len(draws),len(d)))].sum(1)
    return dict(n=len(a),first=int(a.sum()),second=int(b.sum()),rescued=int((delta==1).sum()),lost=int((delta==-1).sum()),
                pp=100*float(delta.mean()),ci=(100*np.quantile(draws,[.025,.975])/len(a)).tolist())


def report(policies,held):
    out=e.OUT;ordered=sorted(held)
    ref=e.read(out/'regions.json');z=np.array(ref['x']);std=np.maximum(.1,z.std(0))
    features=np.stack([e.geom(e.public(held[k])) for k in ordered])
    distances=np.linalg.norm((features[:,None]-z)/std,axis=2)
    nearest=np.argsort(distances,axis=1,kind='stable')[:,:15]
    count=np.array([1+len(held[k]['movable_objects']) for k in ordered])
    masks={p:np.array([k.startswith(p+'_') for k in ordered]) for p in ['fresh','near']}
    metrics={p:{} for p in masks}; table={}; assignments={}; local=[]
    selections={}
    for arm in e.ARMS:
        s=e.read(out/arm/'study.json')
        matrix=np.column_stack([[int(e.read(out/'evaluation'/(arm+'_'+sid)/(k+'.json'))['success']) for k in ordered] for sid in s['final']])
        dev=np.column_stack([np.isin(ref['ids'],next(v for v in s['sweeps'] if v['sid']==sid)['wins']) for sid in s['final']])
        fixed=int(dev.sum(0).argmax())
        choice=dev[nearest].mean(axis=1).argmax(1)
        selections[arm]=dict(development_best=s['final'][fixed],frozen=s['final'])
        assignments[arm]=[s['final'][i] for i in choice]
        series=dict(best=matrix[:,fixed],commit=matrix[np.arange(len(ordered)),choice],union=matrix.max(1))
        for k,x in series.items():table[arm+'_'+k]=x
        for j,sid in enumerate(s['final']):table[arm+'_'+sid]=matrix[:,j]
        for p,m in masks.items():
            metrics[p][arm]={k:int(v[m].sum()) for k,v in series.items()}
        if arm=='regional':
            for i,rev in enumerate(s['sweeps']):
                ids=set(rev['focus'])
                near=np.array([k.startswith('near_') and k[5:] in ids for k in ordered])
                center=ref['ids'].index(rev['center'])
                close=np.flatnonzero(masks['fresh'])[np.argsort(distances[masks['fresh'],center],kind='stable')[:30]]
                for name,m in [('near_region',np.flatnonzero(near)),('fresh_closest_30',close)]:
                    local.append(dict(round=i+1,center=rev['center'],subset=name,n=len(m),
                        regional_member=int(matrix[m,i].sum()),regular_same_round=int(table['regular_'+selections['regular']['frozen'][i]][m].sum()),
                        single=int(table['single_best'][m].sum()) if 'single_best' in table else None,
                        ids=[ordered[j] for j in m]))
    for row in local:
        row['single']=sum(int(table['single_best'][ordered.index(k)]) for k in row['ids'])
    contrasts={}
    for p,m in masks.items():
        pair=lambda a,b,mask:paired(table[a][mask],table[b][mask],count[mask]) if mask.any() else None
        shared=m & table['regular_union'].astype(bool) & table['regional_union'].astype(bool)
        contrasts[p]=dict(union=pair('regular_union','regional_union',m),simple_router=pair('regular_commit','regional_commit',m),
                          common_solvable_router=pair('regular_commit','regional_commit',shared),
                          single_vs_regular=pair('single_best','regular_commit',m),single_vs_regional=pair('single_best','regional_commit',m))
    docs=e.ROOT/'docs/research/key_findings/2026-11-19_regional_cap_authoring';docs.mkdir(parents=True,exist_ok=True)
    with (docs/'per_scene.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['id','movable',*table,*['selected_'+a for a in e.ARMS]])
        for i,k in enumerate(ordered):w.writerow([k,count[i],*[int(v[i]) for v in table.values()],*[assignments[a][i] for a in e.ARMS]])
    budgets={}
    for arm in e.ARMS:
        root=out/arm;s=e.read(root/'study.json');c=e.read(root/'campaign.json')
        budgets[arm]=dict(probes=s['probes'],sweeps=s['sweeps'],used=c['used'],time_to_final_seconds=(root/'study.json').stat().st_mtime-c['created'])
    result=dict(metrics=metrics,paired=contrasts,local=local,selection=selections,budgets=budgets)
    e.save(docs/'summary.json',result)
    for fn in ['protocol.json','frozen.json']:e.save(docs/fn,e.read(out/fn))
    e.save(out/'FINISHED.json',dict(complete=True,episodes=len(ordered)*len(policies),summary=sha(docs/'summary.json')))
    print(metrics,flush=True)


def main():
    while not all(e.read(e.OUT/arm/'study.json')['final'] for arm in e.ARMS):
        time.sleep(8)
    policies=frozen()
    held=e.read(e.OUT/'held.json')
    assert e.digest(held)==e.read(e.OUT/'protocol.json')['held_sha']
    total=len(held)*len(policies)
    for ix,key in enumerate(policies):
        arm,sid=key.split('_')
        src=e.OUT/arm/'semantics'/(sid+'.py')
        assert sha(src)==policies[key]
        directory=e.OUT/'evaluation'/key
        e.save(e.OUT/'eval_progress.json',dict(total=total,models_completed=ix,active=key))
        records=e.run_batch(src.read_text(),held,directory,14)
        errors=[r for r in records if not r['complete']]
        if errors:
            e.save(e.OUT/'evaluation_errors.json',dict(model=key,errors=errors))
            raise RuntimeError(f'{key}: {len(errors)} incomplete rollouts')
    report(policies,held)
    e.save(e.OUT/'eval_progress.json',dict(total=total,models_completed=len(policies),active=None))


if __name__=='__main__':main()
