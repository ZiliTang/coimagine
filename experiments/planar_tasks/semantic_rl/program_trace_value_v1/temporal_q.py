"""Candidate-conditioned, cross-CAP attention over the actual recent decision history."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

H=8
D=32
CAP_COUNT=7
ACT=85
SOURCES=7


class Layout:
    def __init__(self,layout):
        self.g=layout['geometry_width']
        self.frame=self.g//(H+1)
        assert self.frame*(H+1)==self.g and layout['advice_width']==CAP_COUNT*(ACT+1)
        self.ga=self.g+layout['advice_width']+2
        self.width=layout['width']
        self.traces=layout['trace_widths']
        self.tail=self.frame+self.width-self.g
        self.row=self.tail+SOURCES+1

    def compact(self,x,previous_sources,periods):
        return np.concatenate((x[...,self.g-self.frame:self.g],x[...,self.g:],previous_sources,periods),axis=-1)


def encoded(x,mean,scale):
    valid=torch.isfinite(x)
    return torch.cat((torch.where(valid,(x-mean)/scale,0.).clamp(-10.,10.),valid.to(x.dtype)),dim=-1)


class TemporalQ(nn.Module):
    def __init__(self,layout,norms):
        super().__init__()
        self.layout=layout
        self.f=Layout(layout)
        f=self.f
        for k,v in norms.items():
            self.register_buffer(k,torch.as_tensor(v,dtype=torch.float32).clone())
        self.global_state=nn.Linear(2*f.ga,64)
        self.identity_term=nn.Linear(SOURCES,64,bias=False)
        self.action_term=nn.Linear(2*ACT,64,bias=False)
        self.geom=nn.Linear(2*f.frame+2+SOURCES+1,D)
        self.experts=nn.ModuleList([nn.Linear(2*(ACT+1+width),D) for width in f.traces])
        self.slot=nn.Parameter(torch.randn(CAP_COUNT+1,D)*.02)
        self.lag=nn.Parameter(torch.randn(H,D)*.02)
        self.query=nn.Linear(2*ACT+SOURCES,D)
        self.query_context=nn.Linear(64,D,bias=False)
        self.cross=nn.MultiheadAttention(D,4,batch_first=True,dropout=0)
        self.residual=nn.Linear(D,64,bias=False)
        self.head=nn.Sequential(nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1),nn.Sigmoid())

    def forward(self,hist,initial,exists,candidates,sources,expert=False):
        f=self.f
        frames=hist[:,:,:f.frame]
        advstart=f.frame
        advend=advstart+CAP_COUNT*(ACT+1)
        adv=hist[:,:,advstart:advend].reshape(-1,H,CAP_COUNT,ACT+1)
        budget=hist[:,:,advend:advend+2]
        static=torch.cat((initial,frames.flatten(1),hist[:,-1,advstart:advend+2]),dim=1)
        base=self.global_state(encoded(static,self.mean[:f.ga],self.scale[:f.ga]))
        raw_numeric=encoded(frames,self.mean[f.g-f.frame:f.g],self.scale[f.g-f.frame:f.g])
        geo=self.geom(torch.cat((raw_numeric,torch.nan_to_num(budget),torch.nan_to_num(hist[:,:,f.tail:])),dim=-1))
        representations=[geo]
        start=f.ga
        for i,width in enumerate(f.traces):
            offset=f.frame+start-f.g
            a=f.g+i*(ACT+1)
            advice=encoded(adv[:,:,i],self.mean[a:a+ACT+1],self.scale[a:a+ACT+1])
            trace=encoded(hist[:,:,offset:offset+width],self.mean[start:start+width],self.scale[start:start+width])
            representations.append(geo+self.experts[i](torch.cat((advice,trace),dim=-1)))
            start+=width
        tokens=F.layer_norm(torch.stack(representations,dim=2)+self.slot[None,None]+self.lag[None,:,None],(D,))
        actions=encoded(candidates,self.candidate_mean,self.candidate_scale)
        query=self.query(torch.cat((actions,sources),dim=-1))+self.query_context(base)[:,None]
        keys=tokens.flatten(1,2)
        mask=(~exists).repeat_interleave(CAP_COUNT+1,dim=1)
        context=self.cross(query,keys,keys,key_padding_mask=mask,need_weights=False)[0]
        first=base[:,None]+self.identity_term(sources)+self.residual(context)
        if not expert:
            first=first+self.action_term(actions)
        return self.head(first).squeeze(-1)

    def expert_actions(self,hist):
        f=self.f
        proposals=hist[:,-1,f.frame:f.frame+CAP_COUNT*(ACT+1)].reshape(-1,CAP_COUNT,ACT+1)
        identity=torch.eye(CAP_COUNT,device=hist.device)[None].expand(len(hist),-1,-1)
        return proposals[:,:,:ACT],identity,proposals[:,:,-1]==1


class LiveScorer:
    """The caller starts a new episode and reports the previous executed action's elapsed ticks.
    For greedy deployment the previous action is exactly the greedy pool choice."""
    def __init__(self,net):
        self.net=net.eval()
        self.reset()

    def reset(self):
        self.history=[]
        self.initial=None
        self.previous_sources=np.zeros(SOURCES,np.float32)
        self.previous_tick=None

    def values(self,x,candidates,sources):
        f=self.net.f
        raw=x.detach().cpu().numpy()[0]
        # Geometry records seconds in its first per-frame field, at dt=.05.
        tick=round(float(raw[f.g-f.frame])*20)
        periods=0 if self.previous_tick is None else tick-self.previous_tick
        row=f.compact(raw,self.previous_sources,np.array([periods/40],np.float32))
        if self.initial is None:
            self.initial=raw[:f.frame].copy()
            assert tick==0
        else:
            assert 0<periods<=40
        self.history=(self.history+[row])[-H:]
        hist=np.full((1,H,f.row),np.nan,dtype=np.float32)
        hist[0,-len(self.history):]=self.history
        mask=np.arange(H)[None,:]>=H-len(self.history)
        with torch.no_grad():
            scores=self.net(torch.from_numpy(hist),torch.from_numpy(self.initial[None]),torch.from_numpy(mask),candidates,sources)
        self.previous_tick=tick
        return scores

    def executed(self,support):
        self.previous_sources=np.array(support,dtype=np.float32,copy=True)
