"""Fit a scalar V of the frozen greedy actor using only on-policy terminal outcomes."""
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn

from tools import cap_actor_rollout_value as run


class Critic(nn.Module):
    def __init__(self,mean,scale,mask_mean):
        super().__init__()
        self.register_buffer('mean',mean.detach().clone())
        self.register_buffer('scale',scale.detach().clone())
        self.register_buffer('mask_mean',mask_mean.detach().clone())
        self.net=nn.Sequential(nn.Linear(2*len(mean),64),nn.LayerNorm(64),nn.ReLU(),
                               nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())

    def forward(self,x):
        valid=torch.isfinite(x)
        values=torch.where(valid,(x-self.mean)/self.scale,0.).clamp(-10,10)
        return self.net(torch.cat((values,valid.float()-self.mask_mean),dim=1)).flatten()


def load_model():
    cp=torch.load(run.OUT/'value.pt',map_location='cpu',weights_only=True)
    model=Critic(cp['mean'],cp['scale'],cp['mask_mean'])
    model.load_state_dict(cp)
    return model.eval()


@torch.inference_mode()
def metrics(model,states,targets,prior):
    briers=[]
    constant=[]
    initial=[]
    early=[]
    for x,z in zip(states,targets):
        y=model(torch.from_numpy(np.asarray(x))).numpy()
        err=(y-z)**2
        briers.append(float(err.mean()))
        early.append(float(err[:5].mean()))
        initial.append(float(err[0]))
        constant.append(float((prior-z)**2))
    return dict(room_brier=float(np.mean(briers)),base_brier=float(np.mean(constant)),
                initial_brier=float(np.mean(initial)),first_five=float(np.mean(early)))


def fit(cache):
    torch.set_num_threads(1)
    torch.manual_seed(17)
    rng=np.random.default_rng(17)
    record=run.ref.read(run.OUT/'on_policy.json')
    data={}
    for fold in ('train','val'):
        keys=sorted(k for k in record if k.startswith(fold+'_'))
        states=[np.load(cache/(k+'.npy'),mmap_mode='c') for k in keys]
        assert len(keys)==run.SIZES[fold] and all(len(x)==record[k]['decisions'] for k,x in zip(keys,states))
        data[fold]=states,np.array([record[k]['success'] for k in keys],np.float32)
    train,labels=data['train']
    length=np.array([len(x) for x in train])
    frozen=run.ref.fit.load_best()
    # Center the validity bits using training data only and retain the frozen public numeric scaling.
    mask=sum(np.isfinite(x).sum(0,dtype=np.int64) for x in train)/length.sum()
    model=Critic(frozen.mean,frozen.scale,torch.from_numpy(mask.astype(np.float32)))
    optim=torch.optim.Adam(model.parameters(),lr=1e-4)
    series=[]
    best=float('inf')
    for step in range(run.CHECKS[-1]+1):
        if step in run.CHECKS:
            val=metrics(model,*data['val'],float(labels.mean()))
            series.append(dict(step=step,**val))
            run.ref.save(run.OUT/'validation.json',series)
            if val['room_brier']<best:
                best=val['room_brier']
                torch.save(model.state_dict(),run.OUT/'value.pt')
            if step==run.CHECKS[-1]:
                break
        if step%100==0:
            run.status('离线拟合 V',step,run.CHECKS[-1],best_val_brier=best)
        ids=rng.integers(len(train),size=64)
        js=(rng.random(64)*length[ids]).astype(int)
        x=torch.from_numpy(np.stack([train[i][j] for i,j in zip(ids,js)]))
        pred=model(x)
        loss=(pred-torch.from_numpy(labels[ids])).square().mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optim.step()
    model=load_model()
    result=dict(selected=min(series,key=lambda v:v['room_brier']),sha=run.campaign.sha(run.OUT/'value.pt'),
                train=metrics(model,train,labels,float(labels.mean())),train_rooms=len(labels),
                train_wins=int(labels.sum()),train_boundaries=int(length.sum()),val_rooms=len(data['val'][0]),
                val_wins=int(data['val'][1].sum()),val_boundaries=sum(map(len,data['val'][0])))
    run.ref.save(run.OUT/'fit.json',result)


if __name__=='__main__':
    fit(Path(sys.argv[1]))
