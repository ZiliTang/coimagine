"""Supervised warm start and gamma=1 physical-time GAE/PPO, as in v16/v21."""
import math
import numpy as np
import torch
from experiments.planar_tasks.relational_search.contract import History, validate_action
from .actions import ActionTree, FIELDS
from .policy import Policy
from .scale import protocol


def demonstration_traces(result, features, *, split, mode, max_samples=None):
    if split != 'train' or result.get('split', split) != 'train' or result['summary']['status'] != 'success':
        raise ValueError('successful training demonstration required')
    if max_samples is not None and (type(max_samples) is not int or max_samples < 1):
        raise ValueError('positive demonstration sample cap required')
    if result.get('execution_mode', 'time') != mode:
        raise ValueError('event/time demonstration mismatch; recollect under target protocol')
    if mode in ('progress', 'duration'):
        from dataclasses import asdict
        from .environment import EventConfig
        if result.get('event_config') != asdict(EventConfig(mode=mode)):
            raise ValueError('event configuration mismatch; recollect under target protocol')
    action_protocol = None
    if mode == 'duration':
        action_protocol = protocol(EventConfig(mode=mode))
        if result.get('action_protocol') != action_protocol:
            raise ValueError('demonstration action-scale protocol mismatch')
        if features.action_protocol != action_protocol:
            raise ValueError('feature action-scale protocol mismatch')
    rows = result['public_rows']
    h = features.new_history(); traces = []
    from .stream_history import StreamHistory, VERSION as STREAM_VERSION
    # Check the recorded history under its original protocol while re-extracting
    # new semantics causally from the full raw trajectory, never a truncated tail.
    verifier=StreamHistory() if result.get("history_protocol")==STREAM_VERSION else History()
    for i, row in enumerate(rows):
        h.append(row['observation'], row['action'])
        verifier.append(row['observation'],row['action'])
        if verifier.sha256 != row['history_hash']:
            raise ValueError('demonstration history checksum')
        if i == len(rows)-1 or max_samples is not None and len(traces) >= max_samples:
            continue
        teacher = rows[i+1]['action']
        validate_action(teacher, h.events[-1]['observation'])
        # The selected prefix is contiguous; every prior decision was consumed once.
        tree = ActionTree(h.events[-1]['observation']); prefix = {}; panels = []; indices = []
        for key in FIELDS:
            panel = features.panel(h, tree, prefix)
            selected = panel['choices'].index(dict(prefix, **{key: teacher[key]}))
            prefix = panel['choices'][selected]; panels.append(panel); indices.append(selected)
        traces.append(dict(action=teacher, panels=panels, indices=indices, history_hash=h.sha256))
        if action_protocol is not None:traces[-1]['action_protocol']=action_protocol
    if not traces:
        raise ValueError('empty demonstrations')
    return traces


def warm_start(traces, features, *, epochs=80, condition='M', seed=17, residual_coef=.05, on_epoch=None):
    if epochs < 1:
        raise ValueError('positive supervised epochs required')
    expected = traces[0].get('action_protocol')
    if any(t.get('action_protocol') != expected or any(p.get('action_protocol') != expected for p in t['panels']) for t in traces):
        raise ValueError('mixed action-scale protocol in demonstrations')
    torch.manual_seed(seed); torch.set_num_threads(1)
    model = Policy([p for t in traces for p in t['panels']], features.sha256, condition)
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    # Frozen normalizers: transform once, then batch variable-sized panels.
    prepared = []
    for t in traces:
        prepared.append([(model.s(p['semantic']),model.e(p['raw']),i)
                         for p,i in zip(t['panels'],t['indices'])])
    curve = []
    for epoch in range(epochs):
        total_loss = total_imitation = 0.
        for indices in torch.randperm(len(traces)).split(32):
            panels = [p for j in indices.tolist() for p in prepared[j]]
            sem = model.semantic(torch.cat([p[0] for p in panels])).flatten()
            raw = model.residual(torch.cat([p[1] for p in panels])).flatten()
            logits = sem if condition=='S' else raw if condition=='E' else sem+raw
            losses=[]; penalties=[]; offset=0
            for xs,xe,selected in panels:
                end=offset+len(xs);q=logits[offset:end];r=raw[offset:end]
                losses.append(torch.logsumexp(q,0)-q[selected])
                penalties.append((r-r.mean()).square().mean());offset=end
            imitation = torch.stack(losses).sum()/len(indices)
            residual = torch.stack(penalties).mean() if condition=='M' else raw.sum()*0
            loss = imitation+residual_coef*residual
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            total_loss += float(loss.detach())*len(indices)
            total_imitation += float(imitation.detach())*len(indices)
        curve.append(dict(epoch=epoch,loss=total_loss/len(traces),imitation=total_imitation/len(traces)))
        if on_epoch:on_epoch(curve[-1])
    return model, curve


def advantages(steps, *, bootstrap=0., trace_seconds=10.):
    if trace_seconds <= 0:
        raise ValueError('positive physical trace time required')
    adv = np.zeros(len(steps), dtype=np.float32); carry = 0.
    for i in range(len(steps)-1, -1, -1):
        s = steps[i]; live = not s['terminal']
        nxt = steps[i+1]['value'] if i+1 < len(steps) else bootstrap
        delta = s['reward']+(nxt if live else 0.)-s['value']
        carry = delta+(math.exp(-s['periods']*.05/trace_seconds)*carry if live else 0.)
        adv[i] = carry
    return adv+np.array([s['value'] for s in steps], dtype=np.float32), adv


def ppo_update(model, optimizer, steps, *, bootstrap=0., epochs=4, residual_coef=.05, clip=.2, target_kl=.03):
    if not steps:
        raise ValueError('empty PPO batch')
    old = torch.tensor([s['logp'] for s in steps])
    with torch.no_grad():
        error = float((torch.stack([model.evaluate(s)[0] for s in steps])-old).abs().max())
    if error > 2e-5:
        raise ValueError('rollout/PPO log-prob mismatch: '+str(error))
    targets, adv = advantages(steps, bootstrap=bootstrap)
    adv = torch.tensor(adv)
    if len(adv) > 1 and adv.std(unbiased=False) > 1e-8:
        adv = (adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
    targets = torch.tensor(targets); curve = []
    for epoch in range(epochs):
        outputs = [model.evaluate(s) for s in steps]
        lp = torch.stack([v[0] for v in outputs]); value = torch.stack([v[1] for v in outputs])
        penalty = torch.stack([v[2] for v in outputs]).mean()
        ratio = (lp-old).exp(); kl = float(((ratio-1)-(lp-old)).mean().detach())
        if target_kl and kl > 1.5*target_kl:
            break
        actor = -torch.minimum(ratio*adv, ratio.clamp(1-clip, 1+clip)*adv).mean()
        critic = (value-targets).square().mean()
        loss = actor+.5*critic+residual_coef*penalty
        optimizer.zero_grad(); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(loss) or not torch.isfinite(norm):
            raise ValueError('non-finite PPO update')
        optimizer.step()
        curve.append(dict(epoch=epoch, loss=float(loss.detach()), actor=float(actor.detach()), critic=float(critic.detach()), kl=kl))
    return dict(logp_recompute_error=error, updates=len(curve), curve=curve)
