"""Synchronous on-policy PPO: independent episode GAE, minibatched updates."""
import math
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from .learning import advantages
from .scale import check_trace


def prepare(model,steps):
    rows=[]
    for step in steps:
        check_trace(model,step)
        panels=[]
        for p,i in zip(step['panels'],step['indices']):
            if p.get('semantic_protocol')!=model.semantic_protocol:raise ValueError('semantic history protocol mismatch')
            if p.get('action_protocol') != model.action_protocol:raise ValueError('action-scale protocol mismatch: panel')
            if p['semantic_sha256']!=model.semantic_sha256 or model.schema is not None and p.get('schema')!=model.schema:
                raise ValueError('mixed semantic versions in PPO batch')
            panels.append((model.s(p['semantic']),model.e(p['raw']),i))
        t=step.get('temperature',1.)
        if len(panels)!=4 or not math.isfinite(t) or t<=0:raise ValueError('invalid conditional trace')
        rows.append((panels,t))
    return rows


def evaluate_prepared(model,rows):
    panels=[p for row,_ in rows for p in row];lengths=[len(p[0]) for p in panels]
    sem=model.semantic(torch.cat([p[0] for p in panels])).flatten()
    raw=model.residual(torch.cat([p[1] for p in panels])).flatten()
    logits=sem if model.condition=='S' else raw if model.condition=='E' else sem+raw
    padded=pad_sequence(logits.split(lengths),batch_first=True,padding_value=-torch.inf)
    temperatures=torch.tensor([t for _,t in rows for _ in range(4)])
    dist=torch.distributions.Categorical(logits=padded/temperatures[:,None])
    selected=torch.tensor([p[2] for p in panels])
    lp=dist.log_prob(selected).reshape(-1,4).sum(1)
    entropy=dist.entropy().reshape(-1,4).sum(1)  # sampled-prefix conditional entropy sum
    padded_raw=pad_sequence(raw.split(lengths),batch_first=True)
    mask=torch.arange(padded_raw.shape[1])[None,:]<torch.tensor(lengths)[:,None]
    mean=padded_raw.sum(1)/torch.tensor(lengths)
    penalty=(((padded_raw-mean[:,None]).square()*mask).sum(1)/torch.tensor(lengths)).reshape(-1,4).mean(1)
    if model.condition!='M':penalty=penalty*0
    roots=torch.stack([torch.cat((p[0][0].mean(0),p[0][1].mean(0))) for p,_ in rows])
    value=model.critic(roots).flatten()
    return lp,value,penalty,entropy


def ppo_batch_update(model,optimizer,episodes,*,epochs=4,minibatch_size=64,seed=17,
                     residual_coef=.05,clip=.2,target_kl=.03,on_update=None):
    if epochs<1 or minibatch_size<1:raise ValueError('positive PPO limits required')
    steps=[];targets=[];advs=[]
    for episode in episodes:
        batch=episode['steps']
        if not batch:continue
        y,a=advantages(batch,bootstrap=episode['bootstrap'])
        steps.extend(batch);targets.extend(y.tolist());advs.extend(a.tolist())
    if not steps:raise ValueError('no eligible rollout steps')
    started=time.monotonic();rows=prepare(model,steps)
    old=torch.tensor([s['logp'] for s in steps]);targets=torch.tensor(targets);adv=torch.tensor(advs)
    if len(adv)>1 and adv.std(unbiased=False)>1e-8:adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    with torch.no_grad():
        errors=[float((evaluate_prepared(model,rows[i:i+minibatch_size])[0]-old[i:i+minibatch_size]).abs().max()) for i in range(0,len(rows),minibatch_size)]
    error=max(errors)
    if error>2e-5:raise ValueError('rollout/PPO log-prob mismatch: '+str(error))
    generator=torch.Generator().manual_seed(seed);curve=[];stopped=False
    for epoch in range(epochs):
        for idx in torch.randperm(len(rows),generator=generator).split(minibatch_size):
            lp,value,penalty,entropy=evaluate_prepared(model,[rows[i] for i in idx.tolist()])
            delta=lp-old[idx];ratio=delta.exp();kl=((ratio-1)-delta).mean()
            if target_kl and float(kl.detach())>1.5*target_kl:stopped=True;break
            actor=-torch.minimum(ratio*adv[idx],ratio.clamp(1-clip,1+clip)*adv[idx]).mean()
            critic=(value-targets[idx]).square().mean();residual=penalty.mean()
            loss=actor+.5*critic+residual_coef*residual
            optimizer.zero_grad();loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            if not torch.isfinite(loss) or not torch.isfinite(norm):raise ValueError('non-finite PPO update')
            optimizer.step()
            variance=targets[idx].var(unbiased=False)
            row=dict(epoch=epoch,update=len(curve)+1,samples=len(idx),loss=float(loss.detach()),actor=float(actor.detach()),
                     critic=float(critic.detach()),residual=float(residual.detach()),kl=float(kl.detach()),
                     entropy=float(entropy.mean().detach()),clip_fraction=float(((ratio-1).abs()>clip).float().mean()),
                     grad_norm=float(norm),explained_variance=float((1-(targets[idx]-value.detach()).var(unbiased=False)/variance)) if variance>1e-8 else None)
            curve.append(row)
            if on_update:on_update(row)
        if stopped:break
    return dict(updates=len(curve),samples=len(steps),logp_recompute_error=error,early_stop_kl=stopped,
                wall_seconds=time.monotonic()-started,curve=curve)
