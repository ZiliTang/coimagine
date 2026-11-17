"""Validated continuation of frozen PPO runs; never resets Adam silently."""
from pathlib import Path
import hashlib
import random
import torch
from experiments.planar_tasks.relational_search.pipeline import Campaign,read
from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.policy import Policy


def schedule(cases,seed,count):
    rng=random.Random(seed);result=[]
    if not cases:raise ValueError('empty training panel')
    while len(result)<count:
        ids=[c['id'] for c in cases];rng.shuffle(ids);result.extend(ids)
    return result[:count]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_resume(path,source,panel,cfg,reference):
    p=Path(path).resolve();seal=Campaign(p).verify_frozen();state=read(p/'campaign.json')
    if state.get('completion')!='complete' or state.get('completed_rounds',0)<1:
        raise ValueError('resume requires a complete frozen PPO run')
    for name in ('workers','epochs','minibatch_size','max_ticks','max_decisions','episode_seconds',
                 'temperature','seed','semantic_capacity','feature_runtime','execution_mode','execution_backend'):
        if state['config'][name]!=getattr(cfg,name):raise ValueError('resume config mismatch: '+name)
    if (p/'inputs/source.py').read_text()!=source or read(p/'panels/train.json')!=panel:
        raise ValueError('resume source or training panel mismatch')
    count=state['completed_rounds'];previous=state.get('continuation') or {}
    base_round=previous.get('round_offset',0)
    offset=base_round+count;episode_offset=offset*cfg.workers
    expected=schedule(panel['cases'],cfg.seed,episode_offset)
    if state['training_schedule']!=expected[base_round*cfg.workers:]:
        raise ValueError('resume training schedule mismatch')
    checkpoint=p/'models/final.pt';last=p/'models'/f'round_{count-1:04d}.pt'
    optimizer=p/'models'/f'optimizer_{count-1:04d}.pt'
    if sha(checkpoint)!=sha(last) or sha(checkpoint)!=state['final_checkpoint_sha256']:
        raise ValueError('resume final checkpoint mismatch')
    model,_=Policy.load(checkpoint,digest(source))
    for name in ('schema','action_protocol','semantic_protocol','condition'):
        if getattr(model,name)!=getattr(reference,name):raise ValueError('resume model protocol mismatch: '+name)
    payload=torch.load(optimizer,map_location='cpu',weights_only=True)
    if set(payload)!= {'optimizer','rng'}:raise ValueError('resume optimizer/RNG payload')
    probe=torch.optim.Adam(model.parameters(),lr=3e-4);probe.load_state_dict(payload['optimizer'])
    if not probe.state or any(g['lr']!=3e-4 for g in probe.param_groups):raise ValueError('resume missing Adam state or learning-rate mismatch')
    for param,values in probe.state.items():
        for key in ('exp_avg','exp_avg_sq'):
            if values[key].shape!=param.shape or not torch.isfinite(values[key]).all():raise ValueError('resume invalid Adam moments')
    metrics=read(p/'metrics.json');updates=previous.get('optimizer_step_offset',0)+len(metrics['updates'])
    record=dict(parent=str(p),seal=seal,round_offset=offset,episode_offset=episode_offset,
        optimizer_step_offset=updates,tick_offset=previous.get('tick_offset',0)+sum(r['ticks'] for r in metrics['rounds']),
        decision_offset=previous.get('decision_offset',0)+sum(r['decisions'] for r in metrics['rounds']),
        checkpoint_sha256=sha(checkpoint),optimizer_sha256=sha(optimizer),
        schedule='same seeded stream after original episode offset',optimizer_restored=True,rng_restored=True)
    return model,record,checkpoint,optimizer
