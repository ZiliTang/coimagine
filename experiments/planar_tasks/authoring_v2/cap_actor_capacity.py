"""Factorial width and update-budget diagnostic on the frozen single-teacher GA data."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import shutil
import time

import numpy as np
import torch
from torch import nn

from . import cap_actor_distillation as distill
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store

PRIOR = distill.previous.BASE/'cap_actor_distillation_20261119_01'
WIDTHS = (64, 128)
SEEDS = (17, 29)
END = 24000
CHECKS = (0, 1500, 3000, 6000, 9000, 12000, 18000, END)
TABLES = {}


class TeacherData:
    def __init__(self, part):
        with np.load(PRIOR/f'{part}_prefixes.npz') as data:
            mask = data['priority_action'] >= 0
            ids = data['row'][mask]
            self.roots = data['root'][mask]
            self.labels = data['priority_action'][mask]
        store = Store(distill.PARENT/'arrays'/part)
        a = store.arrays
        self.features = a['x'][ids,:distill.WIDTH]
        self.candidates = a['candidates'][ids]
        self.sources = a['sources'][ids]
        self.valid = a['valid'][ids]
        assert self.valid[np.arange(len(ids)),self.labels].all()
        self.root_ids, ranks, counts = np.unique(self.roots, return_inverse=True, return_counts=True)
        self.weights = 1./(len(counts)*counts[ranks])
        buckets = [np.flatnonzero(self.roots==r) for r in self.root_ids]
        self.per_root = np.concatenate(buckets)
        self.offsets = np.cumsum([0]+[len(b) for b in buckets])

    def tensors(self, idx):
        return [torch.from_numpy(x[idx]) for x in (self.features,self.candidates,self.sources,self.valid,self.labels)]

    def sample(self, rng):
        roots = rng.integers(len(self.root_ids),size=64)
        offsets = self.offsets[roots]+(rng.random(64)*np.diff(self.offsets)[roots]).astype(int)
        return self.per_root[offsets]


def make_actor(width):
    model = distill.Actor()
    if width!=64:
        inputs = model.net[0].in_features
        model.net = nn.Sequential(nn.Linear(inputs,width),nn.ReLU(),nn.Linear(width,width),
                                  nn.ReLU(),nn.Linear(width,1),nn.Identity())
    return model


@torch.inference_mode()
def evaluate(model, data):
    predictions=[]
    for begin in range(0,len(data.roots),256):
        x,c,s,valid,labels = data.tensors(slice(begin,begin+256))
        scores = model.values(x,c,s).masked_fill(~valid,-torch.inf)
        predictions.append(torch.stack([scores.argmax(1)==labels,
            -scores.log_softmax(1).gather(1,labels[:,None]).flatten()],1).double().numpy())
    hit, nll = np.concatenate(predictions).T
    return dict(top1=float(data.weights@hit),nll=float(data.weights@nll),
                teacher_p=float(data.weights@np.exp(-nll)),boundary_top1=float(hit.mean()))


def train(job):
    out, width, seed = job
    torch.set_num_threads(4)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = make_actor(width)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    key = f'w{width}_s{seed}'
    prior = read(PRIOR/'fit_metrics.json')[f'single_s{seed}']
    series = []
    train_data = TABLES['train']
    started = time.monotonic()
    for step in range(END+1):
        if step in CHECKS:
            stats = {part:evaluate(model,data) for part,data in TABLES.items()}
            if width==64 and step==6000:
                old = torch.load(PRIOR/f'single_s{seed}.pt',weights_only=True)
                exact = all(torch.equal(t, old[name]) for name,t in model.state_dict().items())
                stats['exact_previous_weights'] = exact
                for part in TABLES:
                    assert abs(stats[part]['top1']-prior[part]['priority']['all']['top1'])<1e-10 or not exact
            series.append(dict(step=step,elapsed_seconds=time.monotonic()-started,**stats))
            save(out/f'{key}.json',series)
            if step in (6000,END):
                torch.save(model.state_dict(),out/f'{key}_u{step}.pt')
            print(key, step, 'train',round(stats['train']['top1'],4),'held',round(stats['validation']['top1'],4), flush=True)
        if step==END:
            return key, sum(p.numel() for p in model.parameters())
        ids = train_data.sample(rng)
        x,c,s,valid,labels = train_data.tensors(ids)
        logits = model.values(x,c,s)
        loss = distill.set_loss(logits,valid,nn.functional.one_hot(labels,7).bool()).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(__file__,out/'source.py')
    old = read(PRIOR/'protocol.json')
    assert read(PRIOR/'FINISHED.json')['complete']
    assert sha(distill.PARENT/'normalizers.npz')==old['normalizers_sha256']
    assert sha(distill.PARENT/'split.json')==old['split_sha256']
    assert sha(Path(distill.__file__))==old['source_sha256']
    previous_models=read(PRIOR/'FROZEN.json')
    for seed in SEEDS:
        name=f'single_s{seed}.pt'
        assert sha(PRIOR/name)==previous_models[name]
    for part in ('train','validation'):
        TABLES[part] = TeacherData(part)
    assert set(TABLES['train'].roots).isdisjoint(TABLES['validation'].roots)
    save(out/'protocol.json',dict(widths=WIDTHS,seeds=SEEDS,updates=[6000,END],checkpoints=CHECKS,
        subject='single priority teacher; offline imitation only',previous=str(PRIOR),
        feature_width=distill.WIDTH,root_weighted=True,root_sampler='same root and within-root draws for each width given seed',
        optimizer='Adam, 0.0003, batch 64, gradient norm clip 1',source_sha=sha(Path(__file__)),
        prefix_hashes={k:sha(PRIOR/f'{k}_prefixes.npz') for k in TABLES},
        sizes={k:dict(roots=len(v.root_ids),boundaries=len(v.roots)) for k,v in TABLES.items()},
        primary='root-weighted train and validation teacher top-1 at both fixed budgets; also NLL',
        scope='same previously inspected development validation; no model selection or closed-loop episodes'))
    jobs=[(out,w,s) for s in SEEDS for w in WIDTHS]
    with ProcessPoolExecutor(4,mp_context=mp.get_context('fork')) as pool:
        params=dict(f.result() for f in as_completed([pool.submit(train,j) for j in jobs]))
    save(out/'FROZEN.json',dict(parameters=params,files={p.name:sha(p) for p in sorted(out.glob('*.pt'))}))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    run(parser.parse_args().out.resolve())
