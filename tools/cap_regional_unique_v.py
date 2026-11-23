"""Score frozen expert V heads on matched initial histories with full episode labels."""
import csv
from collections import Counter

import numpy as np
import torch

from tools import cap_regional_extension_run as run
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import all_v, load_checkpoint

DOCS=run.DOCS.parent/'2026-11-20_regional_unique_v'


def summarize(rows,sids):
    unique=[r for r in rows if r['n_success']==1]
    def tally(items,key):
        return dict(Counter(r[key] for r in items))
    def mean(items,key):
        return float(np.mean([r[key] for r in items])) if items else None
    scores=np.array([[r['v_'+sid] for sid in sids] for r in unique])
    labels=np.array([[r['success_'+sid] for sid in sids] for r in unique])
    equal_brier=float(((scores.mean(1,keepdims=True)-labels)**2).mean()) if unique else None
    return dict(unique_equal_score_brier=equal_brier,n=len(rows),unique=len(unique),top=tally(rows,'top'),unique_top=tally(unique,'top'),
                unique_winners=tally(unique,'winner'),unique_correct=sum(r['v_success'] for r in unique),
                unique_top3=sum(r['rank']<=3 for r in unique),unique_q=sum(r['directq'] for r in unique),
                median_rank=float(np.median([r['rank'] for r in unique])) if unique else None,
                winner=mean(unique,'winner_value'),losers=mean(unique,'loser_mean'),best_loser=mean(unique,'best_loser'),
                mean_margin=mean(unique,'margin'),unique_brier=mean(unique,'brier'),top_ties=sum(r['top_ties']>1 for r in rows),
                all_v_success=sum(r['v_success'] for r in rows),best_success=sum(r['best'] for r in rows),
                brier=mean(rows,'brier'),by_winner={sid:dict(n=len(g),correct=sum(r['v_success'] for r in g),
                median_rank=float(np.median([r['rank'] for r in g]))) for sid in sids if (g:=[r for r in unique if r['winner']==sid])})


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    ext=run.ext;dq=run.dq;out=ext.OUT;directory=out/'directq/models/GA_s17'
    manifest=ext.read(out/'frozen.json');sids=manifest['sids'];run.init_models(sids)
    v,checkpoint=load_checkpoint(directory/'V.pt')
    assert checkpoint['kind']=='V' and checkpoint['updates']==16000 and v.cap_count==len(sids)==10
    fixtures=ext.read(out/ext.DEV);held=ext.read(out/'held.json')
    assert ext.old.digest(fixtures)==manifest['development'] and ext.old.digest(held)==manifest['held']
    fixtures.update(held)
    store=Store(out/'directq/arrays/train');a=store.arrays
    initials=np.flatnonzero(a['boundary']==0);g=v.layout['geometry_width']
    with (run.DOCS/'per_scene.csv').open() as f: outcomes=list(csv.DictReader(f))
    assert len(fixtures)==len(outcomes)==len(set(r['case_id'] for r in outcomes))==505
    records=[]
    for r in outcomes:
        cid=r['case_id'];labels=np.array([int(r[s]) for s in sids])
        env=dq.fast_runtime.ScopedEnvironment(fixtures[cid],dq.study.CFG,fast_feedback=True,record_details=False,
                                             history_factory=dq.fast_runtime.AuditedCachedHistory)
        try:x,c,s,valid,pool,_=dq.Proposals().panel(env.history)
        finally:env.close_reads();dq.fast_runtime.ACTIVE=None
        legal=s[valid].any(0)
        assert legal.all()
        if r['panel'] in ('dev','aug'):
            key=int(cid.split('_')[1])*100+int(cid.split('_')[2])
            ix=initials[a['root'][initials]==key]
            ix=ix[np.argsort(a['cap'][ix])]
            np.testing.assert_equal(a['cap'][ix],np.arange(10))
            np.testing.assert_equal(a['x'][ix],np.broadcast_to(x,(10,len(x))))
            np.testing.assert_equal(a['y'][ix],labels)
        t=torch.from_numpy(x[None])
        scores=all_v(v,t)[0].numpy()
        # Compare the individual expert forward used for V fitting against Q's factored labeler.
        single=v(t.expand(10,-1),torch.arange(10)).numpy()
        np.testing.assert_allclose(scores,single,atol=5e-6,rtol=2e-5)
        ix=np.argsort(-scores,kind='stable');top=ix[0]
        row=dict(panel=r['panel'],case_id=cid,parent=r['parent'],n_success=int(labels.sum()),top=sids[top],
                 top_ties=int((scores==scores[top]).sum()),v_success=int(labels[top]),directq=int(r['directq']),
                 best=int(r['best']),brier=float(np.mean((scores-labels)**2)),winner='',rank=None,winner_value=None,
                 loser_mean=None,best_loser=None,margin=None)
        if labels.sum()==1:
            w=labels.argmax();losers=scores[labels==0]
            row.update(winner=sids[w],rank=int(np.flatnonzero(ix==w)[0]+1),winner_value=float(scores[w]),
                       loser_mean=float(losers.mean()),best_loser=float(losers.max()),margin=float(scores[w]-losers.max()))
        row.update({'v_'+sid:float(val) for sid,val in zip(sids,scores)})
        row.update({'success_'+sid:int(val) for sid,val in zip(sids,labels)})
        records.append(row)
    summary={p:summarize([r for r in records if r['panel']==p],sids) for p in ['dev','aug','fresh','near']}
    summary['train']=summarize([r for r in records if r['panel'] in ('dev','aug')],sids)
    counts=np.zeros(10,dtype=int);stored=np.load(directory/'Q_init_targets.npy',mmap_mode='r')
    nonterminal=np.flatnonzero(~a['done']);fallback=0;ties=0
    for start in range(0,len(nonterminal),128):
        idx=nonterminal[start:start+128];x=torch.from_numpy(a['x'][a['next'][idx]])
        mask=x[:,g:g+10*86].reshape(-1,10,86)[:,:,-1]==1
        values=all_v(v,x).masked_fill(~mask,-torch.inf)
        high,expert=values.max(1);ok=mask.any(1)
        ties+=int(((values.eq(high[:,None]).sum(1)>1)&ok).sum())
        np.testing.assert_allclose(stored[idx],torch.where(ok,high,0.).numpy(),atol=2e-6,rtol=1e-5)
        counts+=np.bincount(expert[ok].numpy(),minlength=10);fallback+=int((~ok).sum())
    summary['successor']=dict(n=len(nonterminal),fallback=fallback,top_ties=ties,counts=dict(zip(sids,counts.tolist())))
    saved=dq.read(directory/'V_metrics.json')['train']
    np.testing.assert_allclose(saved['initial_brier'],summary['train']['brier'],atol=1e-6)
    summary['fit']=dict(total_boundaries=len(store),singleton_initial_rows=summary['train']['unique']*len(sids),
                        factual_boundary_brier=saved['brier'],initial_brier=saved['initial_brier'])
    summary['v_sha256']=run.sha(directory/'V.pt')
    DOCS.mkdir(parents=True,exist_ok=True)
    with (DOCS/'per_scene.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
    ext.save(DOCS/'summary.json',summary)
    for key,m in summary.items():
        if isinstance(m,dict):print(key,{k:v for k,v in m.items() if k!='by_winner'})

if __name__=='__main__':main()
