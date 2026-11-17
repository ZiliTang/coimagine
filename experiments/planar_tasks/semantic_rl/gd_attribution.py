"""Add GD without changing the frozen G/GA/DA/GDA runtime or parameterization."""
import numpy as np
import torch
from .input_attribution_v2 import Scorer as FrozenScorer


class GDScorer(FrozenScorer):
    def __init__(self,layout,arm,context_stats,candidate_stats,source_sha,action_protocol,*,projection_norm=True):
        if arm!='GD':
            raise ValueError('this isolated extension is GD only')
        super().__init__(layout,'GDA',context_stats,candidate_stats,source_sha,action_protocol,projection_norm=projection_norm)
        self.arm='GD'
        with torch.no_grad():
            for key in ('context','candidate'):
                mask=torch.tensor([g in ('G','D','common') for g in layout[key+'_groups']],dtype=torch.float32)
                getattr(self,key+'_mask').copy_(mask)


def train(*args,**kwargs):
    # Dedicated process only: reuse identical optimizer, sampling and loss while
    # injecting the subclass at construction. No source file or other process changes.
    from . import library_learning as L
    old=L.Scorer
    try:
        L.Scorer=GDScorer
        return L.train(*args,**kwargs)
    finally:
        L.Scorer=old


def verify_A_mask(model,contexts,candidates):
    """Both values and missingness of every A channel must be invisible."""
    c=np.array(contexts,dtype=np.float32,copy=True)
    x=np.array(candidates,dtype=np.float32,copy=True)
    c2=c.copy();x2=x.copy()
    for key,arr in [('context',c2),('candidate',x2)]:
        mask=np.array(model.layout[key+'_groups'])=='A'
        arr[...,mask]=np.nan
        assert not torch.any(getattr(model,key+'_mask')[mask])
    with torch.no_grad():
        assert torch.equal(model.normalize(c,'context'),model.normalize(c2,'context'))
        assert torch.equal(model.normalize(x,'candidate'),model.normalize(x2,'candidate'))
        assert torch.equal(model.logits(c,x),model.logits(c2,x2))
    return dict(A_value_and_missingness_invariant=True,groups=['G','D','common'])
