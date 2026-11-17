import numpy as np
import torch
from experiments.planar_tasks.semantic_rl import input_attribution_v2 as I
from experiments.planar_tasks.semantic_rl.library_learning import train

def test_streamed_minibatch_has_same_updates_as_precomputed_normalization():
    rng=np.random.default_rng(17);n=35;width=9
    ds=dict(contexts=rng.normal(size=(n,width)).astype('float32'),candidates=rng.normal(size=(n*12,width)).astype('float32'),offsets=np.arange(n)[:,None]*12+np.arange(5)[None,:]*3,targets=rng.integers(0,3,size=(n,4)))
    ds['contexts'][::3,1]=np.nan;ds['candidates'][::7,2]=np.nan
    layout=dict(context_groups=['G','D','A']*3,candidate_groups=['G','A','common']*3)
    a,_=I.train(ds,list(range(n)),layout,'GDA','test',None,updates=4)
    b,_=train(ds,list(range(n)),layout,'GDA','test',None,updates=4)
    for key,tensor in a.state_dict().items():torch.testing.assert_close(tensor,b.state_dict()[key],rtol=1e-5,atol=1e-6)
