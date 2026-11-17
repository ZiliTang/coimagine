"""Two-hidden-layer nonlinear rankers, imitation sets and physical preferences."""
import numpy as np
import torch
from torch import nn


def matrix(rows):
    return np.asarray([[np.nan if v is None else v for v in row] for row in rows],dtype=float)


class MLP:
    def __init__(self,mean,scale,state=None):
        torch.set_num_threads(1)
        self.mean=np.asarray(mean);self.scale=np.asarray(scale)
        self.net=nn.Sequential(nn.Linear(2*len(mean),32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
        if state is not None:self.net.load_state_dict({k:torch.tensor(v,dtype=torch.float32) for k,v in state.items()})

    def transform(self,rows):
        x=matrix(rows);missing=np.isnan(x)
        if not np.isfinite(x[~missing]).all() or x.shape[1]!=len(self.mean):raise ValueError('feature matrix')
        x=np.where(missing,self.mean,x)
        return torch.tensor(np.concatenate([(x-self.mean)/self.scale,missing],axis=1),dtype=torch.float32)

    def score(self,rows):
        with torch.no_grad():return self.net(self.transform(rows)).flatten().numpy()

    def record(self):
        return dict(architecture='input-32-tanh-32-tanh-1',mean=self.mean.tolist(),scale=self.scale.tolist(),state={k:v.detach().tolist() for k,v in self.net.state_dict().items()})

    @classmethod
    def restore(cls,value):return cls(value['mean'],value['scale'],value['state'])


def train(items,feature_key,*,offsets=None,epochs=400,seed=17,residual_penalty=0.):
    torch.manual_seed(seed);torch.set_num_threads(1)
    flat=[x for it in items for x in it[feature_key]];a=matrix(flat);missing=np.isnan(a);count=(~missing).sum(0)
    mean=np.nansum(a,0)/np.maximum(count,1);scale=np.sqrt(np.where(missing,0.,(a-mean)**2).sum(0)/np.maximum(count,1));scale=np.where(scale<1e-5,1.,scale)
    model=MLP(mean,scale);x=model.transform(flat);groups=[];start=0
    for i,it in enumerate(items):
        groups.append((start,start+len(it[feature_key]),it));start+=len(it[feature_key])
    off=torch.zeros(start) if offsets is None else torch.tensor(np.concatenate(offsets),dtype=torch.float32)
    optimizer=torch.optim.Adam(model.net.parameters(),lr=.005,weight_decay=.0001)
    curves=[]
    for epoch in range(epochs):
        raw=model.net(x).flatten();pred=raw+off;demo=[];physical=[];pen=[]
        for lo,hi,it in groups:
            q=pred[lo:hi]
            if it['kind']=='demonstration':demo.append(torch.logsumexp(q,0)-torch.logsumexp(q[it['positive']],0))
            else:physical.append(torch.nn.functional.softplus(-(q[it['preferred']]-q[it['other']])))
            rr=raw[lo:hi];pen.append(((rr-rr.mean())**2).mean())
        dl=torch.stack(demo).mean() if demo else raw.sum()*0
        pl=torch.stack(physical).mean() if physical else raw.sum()*0
        loss=dl+pl+residual_penalty*torch.stack(pen).mean()
        optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch%20==0 or epoch==epochs-1:curves.append(dict(epoch=epoch,loss=float(loss.detach()),imitation=float(dl.detach()),physical=float(pl.detach())))
    return model,curves


def fit(items):
    if not items or not any(i['kind']=='demonstration' for i in items):raise ValueError('successful demonstration data required')
    sm,sl=train(items,'semantic');em,el=train(items,'raw')
    offsets=[sm.score(i['semantic']) for i in items];rm,rl=train(items,'raw',offsets=offsets,residual_penalty=.05)
    result={}
    for name,model,curve,key in [('S',sm,sl,'semantic'),('E',em,el,'raw'),('M',rm,rl,'raw')]:
        metrics=dict(demonstrations=0,demonstration_correct=0,physical_pairs=0,physical_correct=0);details=[]
        for idx,it in enumerate(items):
            scores=model.score(it[key])+(offsets[idx] if name=='M' else 0);best=int(np.argmax(scores))
            if it['kind']=='demonstration':
                ok=best in it['positive'];metrics['demonstrations']+=1;metrics['demonstration_correct']+=int(ok)
            else:
                ok=bool(scores[it['preferred']]>scores[it['other']]+1e-8);metrics['physical_pairs']+=1;metrics['physical_correct']+=int(ok)
            details.append(dict(origin=it['origin'],kind=it['kind'],correct=ok,selected=best,scores=scores.tolist()))
        metrics.update(count=metrics['demonstrations']+metrics['physical_pairs'],correct=metrics['demonstration_correct']+metrics['physical_correct'],ties=0)
        result[name]=dict(checkpoint=model.record(),semantic_checkpoint=sm.record() if name=='M' else None,loss=[r['loss'] for r in curve],training_curve=curve,metrics={'train':metrics},examples=details)
    return result
