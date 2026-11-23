"""Warm-start three trace-blind GA-Q readouts on identical option forks."""
import copy
import time
import numpy as np
import torch
from torch import nn

from .cap_ga_effect_data import OUT, embedding, old, publish, read, save


class Heads(nn.Module):
    def __init__(self,q,arm):
        super().__init__()
        self.arm=arm
        self.shared=copy.deepcopy(q.net[2])
        self.value=copy.deepcopy(q.net[4])
        self.effects=nn.Sequential(nn.Linear(64,32),nn.Tanh(),nn.Linear(32,5))
        self.effect_weights=nn.Linear(64,5)
        nn.init.zeros_(self.effect_weights.weight)
        nn.init.zeros_(self.effect_weights.bias)

    def forward(self,u,swap=False):
        shared=torch.relu(self.shared(u))
        logits=self.value(shared).squeeze(-1)
        pred=self.effects(shared) if self.arm!='A' else None
        if self.arm=='C':
            # Swap only within a held-out two-action state.
            consequence=pred.flip(1) if swap else pred
            logits=logits+(self.effect_weights(shared)*consequence).sum(-1)
        return torch.sigmoid(logits),pred


def samples(kind,keys):
    rows=[]
    for p in sorted((OUT/kind).glob('*.npz')):
        with np.load(p) as d:
            rows.append([d[k].copy() for k in keys])
    combine=np.stack if kind=='held' else np.concatenate
    return [torch.tensor(combine(v),dtype=torch.float32) for v in zip(*rows)]


def aux_loss(pred,truth):
    mask=truth.isfinite()
    errors=(pred-torch.nan_to_num(truth)).square()*mask
    counts=mask.sum(0)
    return (errors.sum(0)/counts.clamp(min=1))[counts>0].mean()


def held_stats(head,u,e,z,q,y,mean,scale):
    with torch.no_grad():
        scores,pred=head(u)
        choices=scores.argmax(1)
        ties=(scores[:,0]==scores[:,1]).sum().item()
        info=z[:,0]!=z[:,1]
        out=dict(pairs=len(u),informative=int(info.sum()),correct=int(z[info].gather(1,choices[info,None]).sum()),
                 wins=int(z.gather(1,choices[:,None]).sum()),ties=ties,q_mse=float((scores-y).square().mean()),
                 choices=choices.tolist(),scores=scores.tolist())
        if pred is not None:
            delta=(pred*scale+mean-e).abs()
            out['mae']=torch.nanmean(delta.flatten(0,1),dim=0).tolist()
            out['mean_baseline_mae']=torch.nanmean((mean-e).abs().flatten(0,1),dim=0).tolist()
        if head.arm=='C':
            swapped,_=head(u,swap=True)
            pick=swapped.argmax(1)
            out['swap_changes']=int((pick!=choices).sum())
            out['swap_correct']=int(z[info].gather(1,pick[info,None]).sum())
        return out


def fit():
    torch.set_num_threads(1)
    base,_=old.load_checkpoint(old.Q)
    assert base.arm=='GA'
    u,y,e,oq=samples('samples',('u','y','e','q'))
    hu,he,hz,hq,hy=samples('held',('u','e','z','q','y'))
    mean=torch.nanmean(e,dim=0)
    scale=torch.nanmean((e-mean).square(),dim=0).sqrt().maximum(torch.tensor([.02,.05,.1,.1,1.]))
    assert torch.isfinite(mean).all() and torch.isfinite(scale).all()
    target=(e-mean)/scale
    history=[];res={}
    out=OUT/'models'
    out.mkdir(exist_ok=True)
    for ai,arm in enumerate('ABC'):
        torch.manual_seed(17)
        net=Heads(base,arm)
        optim=torch.optim.Adam(net.parameters(),lr=3e-4)
        rng=np.random.default_rng(100017)
        with torch.no_grad():
            torch.testing.assert_close(net(u)[0],oq,atol=1e-6,rtol=1e-5)
        block=[]
        for k in range(1,1201):
            idx=rng.integers(len(u),size=64)
            score,pred=net(u[idx])
            qloss=(score-y[idx]).square().mean()
            if arm!='A':
                eloss=aux_loss(pred,target[idx])
                loss=qloss+.05*eloss
            else:
                loss=qloss
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optim.step()
            assert torch.isfinite(loss)
            block.append([float(qloss.detach()),float(eloss.detach()) if arm!='A' else 0.])
            if k%50==0:
                ql,el=np.mean(block,axis=0)
                block=[]
                history.append(dict(step=k,arm=arm,q=ql,effect=el if arm!='A' else None))
                publish('拟合 '+arm,'相同 64 条候选样本/批，固定末轮',2,ai*1200+k,history=history)
        net.eval()
        res[arm]=held_stats(net,hu,he,hz,hq,hy,mean,scale)
        torch.save(net.state_dict(),out/(arm+'.pt'))
    info=hz[:,0]!=hz[:,1]
    res['original']=dict(correct=int(hz[info,0].sum()),informative=int(info.sum()),pairs=len(hz),wins=int(hz[:,0].sum()),
                         q_mse=float((hq-hy).square().mean()))
    save(OUT/'held_results.json',res)
    save(OUT/'training_curve.json',history)
    save(OUT/'models_frozen.json',dict(sha={arm:old.sha(out/(arm+'.pt')) for arm in 'ABC'},
                                       training_transitions=len(u),effect_mean=mean.tolist(),effect_scale=scale.tolist()))


if __name__=='__main__':
    while not (OUT/'data_complete.json').exists():
        if '异常' in read(OUT/'live/state.json')['stage']:raise RuntimeError('collection failed')
        time.sleep(3)
    try:fit()
    except BaseException as exc:
        publish('训练异常',str(exc));raise
