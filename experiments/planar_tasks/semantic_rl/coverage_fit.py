"""Supervised control with an exact optimizer-update budget; no PPO."""
import time
import torch
from .policy import Policy


def fit_updates(traces, features, *, updates, seed=17, residual_coef=.05, wall_seconds=600, on_progress=None, policy_factory=Policy):
    if not traces or updates<1:raise ValueError('nonempty data and positive update budget required')
    expected=traces[0].get('action_protocol')
    if any(t.get('action_protocol')!=expected or any(p.get('action_protocol')!=expected for p in t['panels']) for t in traces):raise ValueError('mixed action protocol')
    torch.manual_seed(seed);torch.set_num_threads(1)
    model=policy_factory([p for t in traces for p in t['panels']],features.sha256,'M')
    optimizer=torch.optim.Adam(model.parameters(),lr=.003)
    prepared=[[(model.s(p['semantic']),model.e(p['raw']),i) for p,i in zip(t['panels'],t['indices'])] for t in traces]
    curve=[];start=time.monotonic();step=0;epoch=0
    while step<updates:
        for indices in torch.randperm(len(traces)).split(32):
            if step>=updates:break
            if time.monotonic()-start>wall_seconds:raise TimeoutError('supervised fit wall budget')
            panels=[p for j in indices.tolist() for p in prepared[j]]
            sem=model.semantic(torch.cat([p[0] for p in panels])).flatten()
            raw=model.residual(torch.cat([p[1] for p in panels])).flatten();logits=sem+raw
            losses=[];penalties=[];offset=0
            for xs,xe,selected in panels:
                end=offset+len(xs);q=logits[offset:end];r=raw[offset:end]
                losses.append(torch.logsumexp(q,0)-q[selected]);penalties.append((r-r.mean()).square().mean());offset=end
            imitation=torch.stack(losses).sum()/len(indices);residual=torch.stack(penalties).mean();loss=imitation+residual_coef*residual
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();step+=1
            if step%50==0 or step==updates:
                row=dict(update=step,epoch=epoch,loss=float(loss.detach()),imitation=float(imitation.detach()),residual=float(residual.detach()),elapsed_seconds=time.monotonic()-start)
                curve.append(row)
                if on_progress:on_progress(row)
        epoch+=1
    return model,curve
