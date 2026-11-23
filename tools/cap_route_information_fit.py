"""Fit parameter-matched temporal scorers on room-paired expert outcomes."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_sequence

from tools import cap_route_information as data
from tools import cap_route_horizon_report as old_report

ARMS=('last','public','physical','future')
TRAIN=[]
MODELS=data.OUT/'models'


def load(fold):
    items=[]
    for path in sorted((data.OUT/'samples'/fold).glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            x={k:torch.from_numpy(z[k].copy()).float() for k in ('prefix','actions','truth','A','B') if k in z}
            x.update(room=path.stem,labels=z['labels'].tolist(),valid=bool(z['valid']),
                     ends=z['ends'].tolist() if 'ends' in z else None)
            items.append(x)
    assert len(items)==200,fold
    return items


def eligible(row,h):
    if not row['valid']:return False
    if h==0:return True
    i=data.HORIZONS.index(h)
    return min(e[i] for e in row['ends'])>0


class Scorer(nn.Module):
    def __init__(self,width,nstate,nactions):
        super().__init__()
        self.frame=nn.Sequential(nn.Linear(width,32),nn.LayerNorm(32),nn.Tanh())
        self.gru=nn.GRU(32,32,batch_first=True)
        self.physical=nn.Linear(nstate,32,bias=False)
        self.readout=nn.Sequential(nn.Linear(32*4+nactions,32),nn.Tanh(),nn.Linear(32,1))

    def encode(self,sequences,start=None):
        lens=[len(s) for s in sequences]
        embedded=self.frame(torch.cat(sequences,dim=0)).split(lens)
        packed=pack_sequence(list(embedded),enforce_sorted=False)
        _,hidden=self.gru(packed,None if start is None else start.unsqueeze(0))
        return hidden[0]

    def forward(self,rows,arm,h):
        past=self.encode([r['prefix'][-1:] if arm=='last' else r['prefix'] for r in rows])
        state=left=right=torch.zeros_like(past)
        if arm=='physical':
            state=self.physical(torch.stack([r['truth'] for r in rows])).tanh()
        if arm=='future':
            idx=data.HORIZONS.index(h)
            left,right=[self.encode([r[key][:r['ends'][i][idx]] for r in rows],past) for i,key in enumerate(('A','B'))]
        x=torch.cat([past,state,left,right-left,torch.stack([r['actions'] for r in rows])],dim=1)
        return self.readout(x).flatten()


def build(row):
    return Scorer(row['prefix'].shape[1],row['truth'].numel(),row['actions'].numel())


@torch.no_grad()
def score(model,rows,arm,h):
    model.eval()
    return torch.cat([model(rows[i:i+24],arm,h) for i in range(0,len(rows),24)]).numpy()


def fit(job, cohort=None, directory=MODELS):
    h,arm,seed=job
    dest=directory/f'{h}_{arm}_{seed}.pt'
    if dest.exists():return
    rows=[r for r in TRAIN if eligible(r,h if cohort is None else cohort)]
    y=np.array([r['labels'][1]-r['labels'][0] for r in rows],dtype='float32')
    torch.manual_seed(seed)
    model=build(rows[0])
    opt=torch.optim.AdamW(model.parameters(),lr=.001)
    batch_rng=np.random.default_rng(seed)
    losses=[]
    for epoch in range(30):
        order=batch_rng.permutation(len(rows))
        model.train()
        sq=0.
        for j in range(0,len(order),24):
            ids=order[j:j+24]
            pred=model([rows[k] for k in ids],arm,h)
            loss=(pred-torch.from_numpy(y[ids])).square().mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
            opt.step()
            sq+=float(loss.detach())*len(ids)
        losses.append(sq/len(rows))
    pred=score(model,rows,arm,h)
    tmp=dest.with_suffix('.tmp')
    torch.save(dict(weights=model.state_dict(),dims=(rows[0]['prefix'].shape[1],rows[0]['truth'].numel(),rows[0]['actions'].numel()),
                    losses=losses,train_auc=old_report.auc(y,pred),rooms=len(y)),tmp)
    tmp.replace(dest)
    print('fit',h,arm,seed,flush=True)


def ensemble(h,arm,directory=MODELS):
    models=[]
    for seed in data.SEEDS:
        ck=torch.load(directory/f'{h}_{arm}_{seed}.pt',map_location='cpu',weights_only=True)
        model=Scorer(*ck['dims'])
        model.load_state_dict(ck['weights'])
        models.append(model.eval())
    return models


def statistics(rows,pred):
    a,b=np.asarray([r['labels'] for r in rows]).T
    d=b-a
    scores=pred.mean(0)
    win=np.where(scores>0,b,a)
    return dict(n=len(d),only_A=int(sum(d<0)),only_B=int(sum(d>0)),fixed_A=int(a.sum()),fixed_B=int(b.sum()),
                mse=float(np.mean((scores-d)**2)),auc=old_report.auc(d,scores),correct=int(sum(((scores>0)==(d>0))&(d!=0))),
                wins=int(win.sum()),overrides=int(sum(scores>0)),rescued=int(sum(win>a)),lost=int(sum(win<a)),
                seed_auc=[old_report.auc(d,p) for p in pred],seed_wins=[int(sum(np.where(p>0,b,a))) for p in pred])


def evaluate(fold):
    rooms=load(fold)
    result={}
    flat=[]
    for h in data.HORIZONS:
        rows=[r for r in rooms if eligible(r,h)]
        d=np.array([r['labels'][1]-r['labels'][0] for r in rows])
        stats,preds={},{}
        for arm in ARMS:
            models=ensemble(h,arm)
            p=np.stack([score(model,rows,arm,h) for model in models])
            stats[arm]=statistics(rows,p)
            preds[arm]=p.mean(0)
            flat.extend(dict(room=r['room'],window=h,arm=arm,A=r['labels'][0],B=r['labels'][1],score=float(s))
                        for r,s in zip(rows,preds[arm]))
        # Paired room bootstrap, keeping the same observed horizons on every arm.
        rng=np.random.default_rng(319+int(h))
        samples=[]
        for _ in range(4000):
            ix=rng.integers(len(rows),size=len(rows))
            aucs=[old_report.auc(d[ix],preds[k][ix]) for k in ARMS]
            if None not in aucs:samples.append(aucs)
        boot=np.array(samples)
        for i,arm in enumerate(ARMS):
            stats[arm]['auc_ci95']=np.quantile(boot[:,i],[.025,.975]).tolist()
            stats[arm]['auc_minus_public_ci95']=np.quantile(boot[:,i]-boot[:,1],[.025,.975]).tolist()
        result[str(h)]=stats
    data.study.save(data.DOC/(fold+'.json'),result)
    data.study.save(data.OUT/(fold+'_scores.json'),flat)


def main():
    global TRAIN
    torch.set_num_threads(1)
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('fit','held','fresh'))
    args=parser.parse_args()
    if args.mode!='fit':
        evaluate(args.mode)
        return
    MODELS.mkdir(parents=True,exist_ok=True)
    TRAIN=load('development')
    jobs=[(h,arm,s) for h in data.HORIZONS for arm in ARMS for s in data.SEEDS]
    jobs.extend((0,'public',s) for s in data.SEEDS)
    with ProcessPoolExecutor(6) as pool:
        list(pool.map(fit,jobs,chunksize=1))
    digest={p.name:__import__('hashlib').sha256(p.read_bytes()).hexdigest() for p in sorted(MODELS.glob('*.pt'))}
    assert len(digest)==len(jobs)
    data.study.save(data.OUT/'locked_models.json',digest)


if __name__=='__main__':
    main()
