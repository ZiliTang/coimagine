"""Fastest successful teacher, ten regional CAPs, training-panel closure diagnostic."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from tools import cap_regional_extension_run as study
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.full_data import Store
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel

BASE = study.OUT/'directq'
OUT = study.OUT.parent/'cap_regional_fastest_actor_20261120_01'
DOCS = study.DOCS.parent/'2026-11-20_regional_fastest_actor'
SEED = 17
STEPS = 24000
CHECKS = {0, 1000, 3000, 6000, 12000, 18000, STEPS}


def load_data():
    fixtures = study.read(study.OUT/study.ext.DEV)
    frozen = study.read(study.OUT/'frozen.json')
    assert study.ext.old.digest(fixtures) == frozen['development']
    teachers, data = teacher_data(fixtures, frozen['sids'], BASE)
    assert len(fixtures)==215 and len(teachers)==161
    return fixtures, teachers, data


def teacher_data(fixtures, sids, base):
    a = Store(base/'arrays/train').arrays
    teachers = {}
    selected = []
    for cid in sorted(fixtures):
        panel, group, index = cid.split('_')
        root = int(group)*100 + int(index)
        candidates = [study.read(base/'shards'/sid/(cid+'.json')) for sid in sids]
        successes = [r for r in candidates if r['success']]
        if not successes:
            continue
        teacher = min(successes, key=lambda r:(r['ticks'],r['cap_id']))
        ids = np.flatnonzero((a['root']==root) & (a['cap']==sids.index(teacher['cap_id'])))
        ids = ids[np.argsort(a['boundary'][ids])]
        np.testing.assert_array_equal(a['boundary'][ids], np.arange(len(ids)))
        assert (a['y'][ids]==1).all() and int(a['duration'][ids].sum()) == teacher['ticks']
        teachers[cid] = dict(root=root, sid=teacher['cap_id'], ticks=teacher['ticks'], ncap=len(successes),
                             boundaries=len(ids), history=teacher['history'])
        selected.extend(ids.tolist())
    ids = np.array(selected)
    data = {k: a[k][ids] for k in ('x','candidates','sources','valid','chosen','root','boundary')}
    return teachers, data


def actor(base=None):
    base = BASE if base is None else base
    layout = study.read(base/'dataset.json')['layout']
    with np.load(base/'normalizers.npz') as norms:
        model = QModel(layout,norms['mean'],norms['scale'],'GA',norms['candidate_mean'],norms['candidate_scale'])
    model.net[-1] = torch.nn.Identity()
    return model


def batch(data, indices):
    return [torch.from_numpy(data[k][indices]) for k in ('x','candidates','sources','valid','chosen')]


@torch.inference_mode()
def assess(model, data):
    predictions = []
    losses = []
    for start in range(0,len(data['root']),128):
        x,c,s,valid,labels = batch(data,slice(start,start+128))
        logits = model.values(x,c,s).masked_fill(~valid,-torch.inf)
        predictions.extend(logits.argmax(1).tolist())
        losses.extend(F.cross_entropy(logits,labels,reduction='none').tolist())
    predictions = np.asarray(predictions)
    hit = predictions == data['chosen']
    active = data['valid'].sum(1)>1
    perfect = sum(hit[data['root']==r].all() for r in np.unique(data['root']))
    return dict(matched=int(hit.sum()), boundaries=len(hit), informative=int(active.sum()),
                informative_matched=int(hit[active].sum()), perfect=int(perfect), ce=float(np.mean(losses))), predictions


def fit(data, *, base=None, out=None, target_sets=None):
    if target_sets is not None:
        from experiments.planar_tasks.authoring_v2.cap_actor_distillation import set_loss
    out = OUT if out is None else out
    torch.manual_seed(SEED)
    model = actor(base)
    rng = np.random.default_rng(SEED)
    optimizer = torch.optim.Adam(model.parameters(),lr=3e-4)
    buckets = [np.flatnonzero(data['root']==r) for r in np.unique(data['root'])]
    lengths = np.array([len(b) for b in buckets])
    offsets = np.cumsum(np.r_[0,lengths[:-1]])
    indices = np.concatenate(buckets)
    series = []
    for step in range(STEPS+1):
        if step in CHECKS:
            metrics, preds = assess(model,data)
            series.append(dict(step=step, **metrics))
            study.save(out/'fit.json',series)
            print(series[-1],flush=True)
            if step==STEPS:
                np.save(out/'predictions.npy',preds)
                torch.save(model.state_dict(),out/'actor.pt')
                break
        roots = rng.integers(len(buckets),size=64)
        ids = indices[offsets[roots] + (rng.random(64)*lengths[roots]).astype(int)]
        x,c,s,valid,labels = batch(data,ids)
        logits = model.values(x,c,s).masked_fill(~valid,-torch.inf)
        if target_sets is None:
            loss = F.cross_entropy(logits,labels)
        else:
            targets = torch.from_numpy(target_sets[ids])
            mixed = targets.sum(1)>1
            if mixed.any():
                losses = F.cross_entropy(logits,labels,reduction='none')
                losses[mixed] = set_loss(logits[mixed],valid[mixed],targets[mixed])
                loss = losses.mean()
            else:
                loss = F.cross_entropy(logits,labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
    return model.eval()


class Lookup:
    """Memorization control keyed only by the normalized observation and ordered proposals."""
    def __init__(self,data):
        self.norms = dict(np.load(BASE/'normalizers.npz'))
        self.table = {}
        self.duplicates = self.conflicts = 0
        for x,c,s,y in zip(data['x'],data['candidates'],data['sources'],data['chosen']):
            key = self.key(x,c,s)
            if key in self.table:
                self.duplicates += 1
                self.conflicts += int(self.table[key] != y)
            else:
                self.table[key] = int(y)

    def key(self,x,c,s):
        digest = hashlib.sha256()
        for a, prefix in ((x,''),(c,'candidate_')):
            valid = np.isfinite(a)
            values = np.where(valid,(a-self.norms[prefix+'mean'])/self.norms[prefix+'scale'],0).clip(-10,10)
            values[values==0] = 0
            digest.update(values.tobytes())
            digest.update(valid.tobytes())
        digest.update(s.tobytes())
        return digest.digest()

    def values(self,x,c,s):
        index = self.table[self.key(x[0].numpy(),c[0].numpy(),s[0].numpy())]
        scores = torch.zeros((1,c.shape[1]))
        scores[0,index] = 1
        return scores


def evaluate(model, fixtures, arm):
    dq = study.dq
    dq.OUT = OUT/arm
    dq.OUT.mkdir(parents=True,exist_ok=True)
    study.init_models(study.read(study.OUT/'frozen.json')['sids'])
    dq.MODEL = model
    for panel in ('dev','aug'):
        jobs = [(cid,f,arm,None,None,panel) for cid,f in sorted(fixtures.items()) if cid.startswith(panel+'_')]
        dq.batch(jobs,10,'evaluate_'+panel)


def run(mode):
    fixtures, teachers, data = load_data()
    OUT.mkdir(parents=True,exist_ok=True)
    if mode=='lookup':
        lookup = Lookup(data)
        study.save(OUT/'lookup.json',dict(keys=len(lookup.table), duplicates=lookup.duplicates, conflicts=lookup.conflicts))
        if lookup.conflicts==0:
            evaluate(lookup,{cid:fixtures[cid] for cid in teachers},mode)
    elif mode=='actor':
        study.save(OUT/'teachers.json',teachers)
        model = fit(data)
        evaluate(model,fixtures,mode)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('lookup','actor'))
    torch.set_num_threads(1)
    run(parser.parse_args().mode)
