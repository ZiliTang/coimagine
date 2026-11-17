"""Memory-bounded preprocessing; same scorer/loss and double-precision statistics."""
import numpy as np
import torch
from .policy import Policy


def compact(traces):
    """Keep training inputs/labels; discard duplicated candidate dictionaries."""
    result=[]
    for t in traces:
        item={k:t[k] for k in ('action','indices','action_protocol') if k in t}
        item['panels']=[]
        for p in t['panels']:
            q={k:p[k] for k in ('semantic_sha256','schema','semantic_protocol','action_protocol') if k in p}
            # Preserve original double values until normalizer statistics are fit.
            for k in ('semantic','raw'):q[k]=np.asarray(p[k],dtype=np.float64)
            item['panels'].append(q)
        result.append(item)
    return result


def streaming_policy(panels,source_sha,condition):
    """Two-pass sufficient statistics without constructing nested Python floats."""
    for k in ('action_protocol','schema','semantic_protocol'):
        if any(p.get(k)!=panels[0].get(k) for p in panels):raise ValueError('mixed '+k)
    model=Policy([panels[0]],source_sha,condition)
    for key,scale in [('semantic',model.s),('raw',model.e)]:
        width=len(panels[0][key][0]);count=np.zeros(width,dtype=np.int64);total=np.zeros(width)
        for p in panels:
            a=np.asarray(p[key],dtype=np.float64);count+=np.isfinite(a).sum(0);total+=np.nansum(a,axis=0)
        mean=total/np.maximum(count,1);squares=np.zeros(width)
        for p in panels:squares+=np.nansum((np.asarray(p[key],dtype=np.float64)-mean)**2,axis=0)
        std=np.sqrt(squares/np.maximum(count,1))
        scale.mean.copy_(torch.tensor(mean,dtype=torch.float32))
        scale.std.copy_(torch.tensor(np.where(std<1e-5,1.,std),dtype=torch.float32))
    return model
