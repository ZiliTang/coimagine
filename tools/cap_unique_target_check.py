"""Compare frozen max-V targets and maximizers on identical logged transitions."""
import numpy as np
import torch

from experiments.planar_tasks.authoring_v2 import cap_unique_initial_weight as study


def initial_scores(rows, models):
    a=rows.arrays
    idx=np.flatnonzero(a['boundary']==0)
    groups=idx[np.lexsort((a['cap'][idx],a['root'][idx]))].reshape(-1,7)
    groups=groups[a['y'][groups].sum(1)==1]
    public_width=models[0].layout['width']-models[0].layout['trace_width']
    for ids in groups:
        x=rows.batch(ids,True)['x'][:,:public_width]
        np.testing.assert_equal(x,np.broadcast_to(x[0],x.shape))
    study.save(study.OUT/'root_input_check.json',dict(identical=len(groups),total=len(groups)))
    x=rows.batch(groups[:,0],True)['x']
    y=a['y'][groups].astype(bool)
    winner=y.argmax(1)
    rare=winner!=0
    stats={}
    with torch.inference_mode():
        for arm,model in zip(['baseline','weighted'],models,strict=True):
            pred=np.vstack([study.all_v(model,torch.from_numpy(block)).numpy() for block in np.array_split(x,8)])
            correct=pred[rare,winner[rare]]
            stats[arm]=dict(train_unique=len(groups),winner_probability=float(pred[y].mean()),
                            loser_probability=float(pred[~y].mean()),brier=float(((pred-y)**2).mean()),
                            mean_each_cap=pred.mean(0).tolist(),cap009_on_rare=float(pred[rare,0].mean()),
                            winner_on_rare=float(correct.mean()),
                            median_winner_rank_on_rare=float(np.median(1+(pred[rare]>correct[:,None]).sum(1))))
    study.save(study.OUT/'train_singleton_scores.json',stats)


def main():
    torch.set_num_threads(1)
    rows=study.Rows(study.OUT/'staging')
    live=np.flatnonzero(~rows.data['done'])
    base=np.load(study.prior.BASE/'Q_init_targets.npy')
    weighted=np.load(study.MODELS/'Q_init_targets.npy')
    assert np.array_equal(base[rows.data['done']],weighted[rows.data['done']])
    ids=np.sort(np.random.default_rng(171719).choice(live,16384,replace=False))
    next_ids=rows.data['next'][ids]
    model=[study.load_checkpoint(p/'V.pt')[0] for p in [study.prior.BASE,study.MODELS]]
    initial_scores(rows,model)
    offset=model[0].layout['geometry_width']
    count=np.zeros((7,7),int)
    fallback=0
    with torch.inference_mode():
        for idx,nxt in zip(np.array_split(ids,64),np.array_split(next_ids,64),strict=True):
            x=torch.tensor(rows.batch(nxt,True)['x'])
            legal=x[:,offset:offset+602].reshape(-1,7,86)[:,:,-1]==1
            any_legal=legal.any(1).numpy()
            fallback+=int((~any_legal).sum())
            picked=[]
            for m,ref in zip(model,[base,weighted],strict=True):
                values=study.all_v(m,x).masked_fill(~legal,-torch.inf)
                high,cap=values.max(1)
                np.testing.assert_allclose(ref[idx],np.where(any_legal,high.numpy(),0.),atol=2e-6,rtol=1e-6)
                picked.append(cap.numpy()[any_legal])
            np.add.at(count,tuple(picked),1)
    study.save(study.OUT/'target_shift.json',dict(nonterminal=len(live),sample=len(ids),fallback=fallback,
               cap_order=study.CAPS,base_to_weighted=count.tolist(),base_target_mean=float(base[live].mean()),
               weighted_target_mean=float(weighted[live].mean()),
               mean_absolute_change=float(abs(weighted[live]-base[live]).mean())))


if __name__=='__main__':
    main()
