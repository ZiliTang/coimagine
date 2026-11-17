"""Same supervised objective with re-iterable streaming normalization inputs.

Masked candidate slices are owned arrays. Never collect all such slices into a
list: that duplicates gigabytes per fit worker. Each pass releases one slice.
"""
import time
import numpy as np
import torch
from .input_attribution_v2 import Scorer,statistics


class CandidateRanges:
    def __init__(self,candidates,offsets,ids):
        self.candidates=candidates;self.offsets=offsets;self.ids=ids
    def __iter__(self):
        for i in self.ids:
            off=self.offsets[i]
            yield self.candidates[off[0]:off[-1]]


def train(dataset,ids,layout,arm,source_sha,action_protocol,*,updates=28000,seed=17,wall_seconds=7200,progress=None):
    torch.set_num_threads(1);torch.manual_seed(seed)
    contexts=dataset['contexts'];candidates=dataset['candidates'];offsets=dataset['offsets'];targets=dataset['targets'];ids=np.asarray(ids,dtype=int)
    cstats=statistics([contexts[i:i+1] for i in ids],contexts.shape[1])
    xstats=statistics(CandidateRanges(candidates,offsets,ids),candidates.shape[1])
    model=Scorer(layout,arm,cstats,xstats,source_sha,action_protocol)
    opt=torch.optim.Adam(model.parameters(),lr=.003);curve=[];step=epoch=0;start=time.monotonic()
    while step<updates:
        for batch in torch.randperm(len(ids)).split(32):
            if step>=updates:break
            if time.monotonic()-start>wall_seconds:raise TimeoutError('joint fit wall budget')
            bi=ids[batch.numpy()];xs=[];owners=[];sizes=[];labels=[]
            for j,i in enumerate(bi):
                off=offsets[i];xs.append(candidates[off[0]:off[-1]])
                owners.extend([j]*int(off[-1]-off[0]));sizes.extend(np.diff(off).tolist());labels.extend(targets[i].tolist())
            q=model.logits(contexts[bi],np.concatenate(xs),torch.tensor(owners))
            loss=torch.stack([p.logsumexp(0)-p[j] for p,j in zip(q.split(sizes),labels)]).sum()/len(bi)
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();step+=1
            if step%100==0 or step==updates:
                row=dict(update=step,epoch=epoch,imitation=float(loss.detach()),elapsed_seconds=time.monotonic()-start);curve.append(row)
                if progress:progress(row)
        epoch+=1
    return model,curve
