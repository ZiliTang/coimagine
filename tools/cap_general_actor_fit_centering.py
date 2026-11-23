"""Short bias-compensated input-centering controls for the early gate collapse."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

from tools import cap_general_actor_fit_diagnostic as study


def measure_scale(model,data,w_initial,b_initial):
    with torch.no_grad():
        x,c,s,v,_=study.imitation.batch(data,slice(None))
        layer=model.net[0]
        width=x.shape[-1]
        z=model.encode(x)
        common=F.linear(z,layer.weight[:,:2*width],layer.bias)
        finite=torch.isfinite(c)
        encoded=torch.cat((torch.where(finite,(c-model.candidate_mean)/model.candidate_scale,0.).clamp(-10.,10.),finite.float(),s),-1)
        local=F.linear(encoded,layer.weight[:,2*width:])
        span=local.masked_fill(~v[:,:,None],-torch.inf).max(1).values-local.masked_fill(~v[:,:,None],torch.inf).min(1).values
        dw=layer.weight-w_initial
        val_shift=F.linear(z[:,:width],dw[:,:width])
        mask_shift=F.linear(z[:,width:],dw[:,width:2*width])
        rms=lambda t:float(t.square().mean().sqrt())
        return dict(common_rms=rms(common),candidate_span_rms=rms(span),
                    value_update_rms=rms(val_shift),mask_update_rms=rms(mask_shift),
                    bias_update_rms=rms(layer.bias-b_initial))


def run(job):
    arm,work=job
    torch.set_num_threads(1)
    data=study.load(work,'train')
    sample=study.probe_rows(data)
    m=study.configure('replicate')
    original_encode=m.encode
    width=m.layout['width']
    with torch.no_grad():
        mean=torch.zeros(2*width,dtype=torch.float64)
        if arm!='baseline':
            for start in range(0,len(data['x']),128):
                z=original_encode(torch.from_numpy(data['x'][start:start+128]))
                mean+=z.sum(0,dtype=torch.float64)
            mean/=len(data['x'])
            if arm=='mask':mean[:width]=0
            elif arm=='values':mean[width:]=0
            else:raise ValueError(arm)
            mean=mean.float()
            m.net[0].bias.add_(F.linear(mean,m.net[0].weight[:,:2*width]))
            m.encode=lambda x:original_encode(x)-mean
        w0=m.net[0].weight.detach().clone()
        b0=m.net[0].bias.detach().clone()
    rng=np.random.default_rng(17)
    buckets=[np.flatnonzero(data['root']==r) for r in np.unique(data['root'])]
    lengths=np.array([len(b) for b in buckets])
    offsets=np.cumsum(np.r_[0,lengths[:-1]])
    all_ids=np.concatenate(buckets)
    opt=torch.optim.Adam(m.parameters(),lr=.0003)
    curve=[]
    steps=200 if arm=='baseline' else 3000
    for step in range(steps+1):
        if step in (0,1,10,50,200,1000,3000):
            row=dict(step=step,**measure_scale(m,sample,w0,b0),margin=study.margin_gradient(m,sample))
            fit,_=study.imitation.assess(m,data)
            row['correct']=fit['informative_matched']
            row['ce']=fit['ce']
            curve.append(row)
            study.save(study.OUT/('center_'+arm+'.json'),curve)
            if step==steps:break
        ri=rng.integers(len(buckets),size=64)
        ix=all_ids[offsets[ri]+(rng.random(64)*lengths[ri]).astype(int)]
        x,c,s,v,y=study.imitation.batch(data,ix)
        loss=F.cross_entropy(m.values(x,c,s).masked_fill(~v,-torch.inf),y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(),1.,error_if_nonfinite=True)
        opt.step()


if __name__=='__main__':
    work=Path(sys.argv[1])
    with ProcessPoolExecutor(3) as pool:
        list(pool.map(run,[(arm,work) for arm in ('baseline','mask','values')]))
