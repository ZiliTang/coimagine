"""Global program trace versus the matched 64x64 supervised GA actor."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from . import cap_actor_capacity as base
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel

COMPARISON = base.distill.previous.BASE/'cap_actor_capacity_20261119_01'
PARENT = base.distill.PARENT
GA_WIDTH = base.distill.WIDTH
TABLES = {}


def teacher_data(part):
    data = base.TeacherData(part)
    with np.load(base.PRIOR/f'{part}_prefixes.npz') as file:
        ids = file['row'][file['priority_action'] >= 0]
    raw = np.load(PARENT/'arrays'/part/'x.npy', mmap_mode='r')
    features = raw[ids]
    for begin in range(0,len(ids),256):
        sl = slice(begin,begin+256)
        assert np.array_equal(features[sl,:GA_WIDTH],data.features[sl],equal_nan=True)
    data.features = features
    return data


def trace_actor():
    layout = read(PARENT/'dataset.json')['layout']
    with np.load(PARENT/'normalizers.npz') as n:
        model = QModel(layout,n['mean'],n['scale'],'GLOBAL',n['candidate_mean'],n['candidate_scale'])
    model.net[-1] = nn.Identity()
    return model


@torch.no_grad()
def matched_start(seed):
    torch.manual_seed(seed)
    ga = base.make_actor(64)
    model = trace_actor()
    full = model.layout['width']
    assert full-model.layout['trace_width']==GA_WIDTH
    src, dst = ga.net[0], model.net[0]
    dst.weight.zero_()
    dst.weight[:,:GA_WIDTH].copy_(src.weight[:,:GA_WIDTH])
    dst.weight[:,full:full+GA_WIDTH].copy_(src.weight[:,GA_WIDTH:2*GA_WIDTH])
    dst.weight[:,2*full:].copy_(src.weight[:,2*GA_WIDTH:])
    dst.bias.copy_(src.bias)
    for layer in (2,4):
        model.net[layer].load_state_dict(ga.net[layer].state_dict())
    return ga, model


def train(job):
    out,seed=job
    torch.set_num_threads(4)
    rng=np.random.default_rng(seed)
    ga,model=matched_start(seed)
    with torch.inference_mode():
        max_error=0.
        for data in TABLES.values():
            for start in range(0,len(data.roots),256):
                x,c,s,_,_=data.tensors(slice(start,start+256))
                reference=ga.values(x,c,s)
                result=model.values(x,c,s)
                torch.testing.assert_close(result,reference,rtol=1e-5,atol=1e-6)
                max_error=max(max_error,float((result-reference).abs().max()))
    opt=torch.optim.Adam(model.parameters(),lr=3e-4)
    series=[]
    started=time.monotonic()
    for step in range(base.END+1):
        if step in base.CHECKS:
            stats={part:base.evaluate(model,data) for part,data in TABLES.items()}
            series.append(dict(step=step,elapsed_seconds=time.monotonic()-started,**stats))
            save(out/f'global_s{seed}.json',series)
            if step in (6000,base.END):
                torch.save(model.state_dict(),out/f'global_s{seed}_u{step}.pt')
            print('GLOBAL',seed,step,stats,flush=True)
        if step%250==0:
            save(out/f'progress_s{seed}.json',dict(updates=step,total=base.END))
        if step==base.END:
            return str(seed),dict(initial_max_abs_logit_difference=max_error,
                                  parameters=sum(p.numel() for p in model.parameters()),
                                  trace_weight_norm=float(model.net[0].weight[:,GA_WIDTH:model.layout['width']].detach().norm()))
        data=TABLES['train']
        x,c,s,valid,label=data.tensors(data.sample(rng))
        target=nn.functional.one_hot(label,7).bool()
        loss=base.distill.set_loss(model.values(x,c,s),valid,target).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'source.py')
    previous=read(COMPARISON/'protocol.json')
    assert read(COMPARISON/'FINISHED.json')['complete']
    frozen=read(COMPARISON/'FROZEN.json')['files']
    for seed in base.SEEDS:
        for step in (6000,base.END):
            name=f'w64_s{seed}_u{step}.pt'
            assert sha(COMPARISON/name)==frozen[name]
        shutil.copyfile(COMPARISON/f'w64_s{seed}.json',out/f'ga_s{seed}.json')
    for part,h in previous['prefix_hashes'].items():
        assert sha(base.PRIOR/f'{part}_prefixes.npz')==h
        TABLES[part]=teacher_data(part)
    assert sha(PARENT/'normalizers.npz')==read(base.PRIOR/'protocol.json')['normalizers_sha256']
    save(out/'protocol.json',dict(primary='mean paired difference in held-out teacher top-1 at 24000 updates, equal weight per root then fixed seed',
         secondary=['training top-1','teacher probability','negative log likelihood','6000 step diagnostic'],
         previous=str(COMPARISON),base_model_hashes={k:v for k,v in frozen.items() if k.startswith('w64')},
         matched_initialization='seed-matched fresh GA common weights; new trace value and availability columns initialized to zero',
         layout=read(PARENT/'dataset.json')['layout'],source_sha=sha(Path(__file__)),base_training=previous,
         runs='2 offline GLOBAL fits; reuse frozen 64x64 GA for seed17/29; no closed-loop execution'))
    with ProcessPoolExecutor(2,mp_context=mp.get_context('fork')) as pool:
        metadata=dict(pool.map(train,[(out,seed) for seed in base.SEEDS]))
    save(out/'FROZEN.json',dict(runs=metadata,files={p.name:sha(p) for p in sorted(out.glob('*.pt'))}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    run(parser.parse_args().out.resolve())
