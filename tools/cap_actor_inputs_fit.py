"""Room-balanced hard-label fits on the frozen action/geometry inputs."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools import cap_actor_inputs as exp


class Scorer(nn.Module):
    def __init__(self,arm):
        super().__init__()
        self.arm=arm
        self.geometry=arm.startswith('geometry') or arm=='router'
        self.router=arm=='router'
        self.norm=arm.endswith('_ln')
        # Identical candidate weights and later layers across the 2x2 comparison.
        # Draw a common full-width affine layer before selecting the retained columns.
        torch.manual_seed(17)
        width=11736
        original=nn.Linear(width*2+170+5,64)
        middle=nn.Linear(64,64)
        last=nn.Linear(64,1)
        self.bias=nn.Parameter(original.bias.detach().clone())
        self.middle=middle
        if not self.router:
            self.action=nn.Parameter(original.weight[:,2*width:2*width+69].detach().clone())
            self.last=last
        if self.geometry:
            keep=np.r_[exp.gkeep,width+exp.gkeep]
            self.shared=nn.Parameter(original.weight[:,keep].detach().contiguous())
            if self.router:
                self.head=nn.Linear(64,5)
            else:
                j=2*width
                at=np.r_[j+np.arange(69,85),j+85+np.arange(69,85)]
                self.target=nn.Parameter(original.weight[:,at].detach().contiguous())

    def forward(self,b):
        if self.router:
            h=F.relu(F.linear(b['g'],self.shared,self.bias))
            return self.head(F.relu(self.middle(h))).masked_fill(~b['s'].any(1),-torch.inf)
        h=F.linear(b['a'],self.action,self.bias)
        if self.geometry:
            h=h+F.linear(b['g'],self.shared)[:,None]+F.linear(b['t'],self.target)
        if self.norm: h=F.layer_norm(h,(64,))
        return self.last(F.relu(self.middle(F.relu(h)))).squeeze(-1).masked_fill(~b['valid'],-torch.inf)


def scores(arm,model,b):
    if arm=='priority':
        # The existing train-selected order is identical to the frozen library order.
        logits=torch.arange(5,0,-1,dtype=torch.float32).expand(len(b['s']),5)
    else:
        logits=model(b)
        if arm!='router':
            return logits
    result=logits[:,None,:].masked_fill(~b['s'],-torch.inf).amax(2)
    # A legal fallback wait is used if all five CAPs have no legal proposal.
    fallback=~b['s'].any(dim=(1,2))
    return torch.where(fallback[:,None] & b['valid'],0.,result)


def fields(arm):
    if arm=='priority': return ('s','valid')
    if arm=='router': return ('g','s','valid')
    if arm.startswith('geometry'): return ('g','a','t','valid')
    return ('a','valid')


def tensor_batch(data,keys,ix):
    return {k:torch.from_numpy(data[k][ix]) for k in keys}


def dataset(fold):
    return {p.stem:np.load(p,mmap_mode='c') for p in (exp.SCRATCH/fold).glob('*.npy')}


@torch.inference_mode()
def metrics(arm,model,data):
    keys=fields(arm)
    preds=[]
    for i in range(0,len(data['y']),128):
        b=tensor_batch(data,keys,slice(i,i+128))
        preds.extend(scores(arm,model,b).argmax(1).tolist())
    hit=np.asarray(preds)==data['y']
    multi=data['valid'].sum(1)>1
    return dict(correct=int(hit[multi].sum()),multi=int(multi.sum()),boundaries=len(hit),
                perfect_rooms=int(sum(hit[data['room']==k].all() for k in np.unique(data['room']))))


def train(arm):
    torch.set_num_threads(1)
    train_data=dataset('train')
    held=dataset('held')
    model=None if arm=='priority' else Scorer(arm)
    if model is not None:
        opt=torch.optim.Adam(model.parameters(),lr=.0003)
        rng=np.random.default_rng(17)
        names=sorted(exp.read(exp.run.ref.scale.OUT/'teachers.json'))
        order=sorted(range(len(names)),key=lambda i:int(names[i].split('_')[-2])*100+int(names[i].split('_')[-1]))
        buckets=[np.flatnonzero(train_data['room']==i) for i in order]
        lengths=np.array(list(map(len,buckets)))
        ids=np.concatenate(buckets)
        offsets=np.cumsum(np.r_[0,lengths[:-1]])
        keys=fields(arm)
        label='teacher' if arm=='router' else 'y'
        curve=[]
        for step in range(1,24001):
            r=rng.integers(len(buckets),size=64)
            ix=ids[offsets[r]+(rng.random(64)*lengths[r]).astype(int)]
            batch=tensor_batch(train_data,keys,ix)
            loss=F.cross_entropy(model(batch),torch.from_numpy(train_data[label][ix]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            if step%500==0:
                curve.append(dict(step=step,batch_ce=float(loss.detach())))
                exp.save(exp.OUT/(arm+'_curve.json'),curve)
        path=exp.OUT/(arm+'.pt')
        tmp=path.with_suffix('.tmp.pt')
        torch.save(model.state_dict(),tmp)
        os.replace(tmp,path)
    result={fold:metrics(arm,model,data) for fold,data in [('train',train_data),('held',held)]}
    assert result['train']['multi']==11254 and result['held']['multi']==3162
    if arm=='priority':
        reference=exp.read(exp.run.ref.fit.OUT/'priority_reference.json')
        for fold in ('train','held'): assert result[fold]['correct']==reference[fold]['correct']
    exp.save(exp.OUT/(arm+'_fit.json'),result)


def main():
    assert exp.read(exp.OUT/'protocol.json')['members']==exp.read(exp.OUT/'protocol.json')['priority']
    assert (exp.OUT/'data_ready.json').exists()
    pending=[a for a in exp.ARMS if not (exp.OUT/(a+'_fit.json')).exists()]
    with ProcessPoolExecutor(5,mp_context=mp.get_context('fork')) as pool:
        for f in as_completed([pool.submit(train,a) for a in pending]): f.result()


if __name__=='__main__': main()
