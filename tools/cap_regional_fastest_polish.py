"""Bounded train-only interpolation test after the fixed fastest-actor comparison."""
import numpy as np
import torch
from torch.nn import functional as F

from tools import cap_regional_fastest_actor as ex


def run():
    torch.set_num_threads(1)
    fixtures,teachers,data = ex.load_data()
    model = ex.actor()
    model.load_state_dict(torch.load(ex.OUT/'actor.pt',map_location='cpu',weights_only=True))
    baseline,_ = ex.assess(model,data)
    if baseline['perfect']==len(teachers):
        ex.study.save(ex.OUT/'polish.json',dict(skipped=True,reason='fixed checkpoint matches every teacher path'))
        return
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-5)
    rng = np.random.default_rng(ex.SEED)
    buckets = [np.flatnonzero(data['root']==r) for r in np.unique(data['root'])]
    lengths = np.array([len(b) for b in buckets])
    offsets = np.cumsum(np.r_[0,lengths[:-1]])
    ids = np.concatenate(buckets)
    series = []
    best = None
    for extra in range(12001):
        if extra%500==0:
            metrics,_ = ex.assess(model,data)
            rank = (metrics['perfect'],metrics['matched'],-metrics['ce'])
            if best is None or rank>best:
                best = rank
                best_step = extra
                torch.save(model.state_dict(),ex.OUT/'polish.pt')
            series.append(dict(extra=extra,**metrics))
            ex.study.save(ex.OUT/'polish.json',dict(selected_extra=best_step,series=series))
            print(series[-1],flush=True)
            if metrics['perfect']==len(teachers) or extra==12000:
                break
        roots = rng.integers(len(buckets),size=64)
        ix = ids[offsets[roots]+(rng.random(64)*lengths[roots]).astype(int)]
        x,c,s,valid,y = ex.batch(data,ix)
        loss = F.cross_entropy(model.values(x,c,s).masked_fill(~valid,-torch.inf),y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    model.load_state_dict(torch.load(ex.OUT/'polish.pt',map_location='cpu',weights_only=True))
    ex.evaluate(model.eval(),fixtures,'polish')


if __name__=='__main__':
    run()
