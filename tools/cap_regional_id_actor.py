"""A ten-logit policy over frozen regional experts, supervised by fastest teacher ID."""
from collections import Counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tools import cap_regional_fastest_actor as hard

study = hard.study
OUT = hard.OUT.parent/'cap_regional_id_actor_20261120_01'
DOCS = hard.DOCS.parent/'2026-11-20_regional_id_actor'
EXTRA = 4500
PANELS = {'dev':160, 'aug':55, 'fresh':200, 'near':90}


class ExpertActor(nn.Module):
    def __init__(self):
        super().__init__()
        layout = study.read(hard.BASE/'dataset.json')['layout']
        self.geometry = layout['geometry_width']
        self.count = layout['cap_count']
        assert self.count==10 and layout['trace_width']==0
        with np.load(hard.BASE/'normalizers.npz') as norms:
            for key in ('mean','scale'):
                self.register_buffer(key,torch.tensor(norms[key]))
        self.net = nn.Sequential(nn.Linear(2*layout['width'],64),nn.ReLU(),nn.Linear(64,64),
                                 nn.ReLU(),nn.Linear(64,self.count))

    def encode(self,x):
        valid = torch.isfinite(x)
        z = torch.where(valid,(x-self.mean)/self.scale,0.).clamp(-10.,10.)
        return torch.cat((z,valid.float()),dim=-1)

    def legal(self,x):
        advice = x[:,self.geometry:self.geometry+86*self.count].reshape(-1,self.count,86)
        return advice[:,:,-1]==1

    def forward(self,x):
        return self.net(self.encode(x)).masked_fill(~self.legal(x),-torch.inf)


def assess(model,z,legal,target,teacher_action,roots,distinct):
    with torch.inference_mode():
        logits = torch.cat([model.net(chunk) for chunk in z.split(128)]).masked_fill(~legal,-torch.inf)
        pred = logits.argmax(1)
        identity = pred==target
        action = teacher_action[torch.arange(len(z)),pred]
        exact_id = identity.numpy()
        exact_action = action.numpy()
        return dict(boundaries=len(z),cap_correct=int(identity.sum()),action_correct=int(action.sum()),
                    informative=int(distinct.sum()),informative_correct=int(action[distinct].sum()),
                    all_cap=sum(int(exact_id[roots==r].all()) for r in np.unique(roots)),
                    all_action=sum(int(exact_action[roots==r].all()) for r in np.unique(roots)),
                    ce=float(F.cross_entropy(logits,target)))


def train():
    _,teachers,data = hard.load_data()
    frozen = study.read(study.OUT/'frozen.json')
    sids = frozen['sids']
    assert teachers == study.read(hard.OUT/'teachers.json')
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(hard.SEED)
    model = ExpertActor()
    roots = data['root']
    by_root = {t['root']:sids.index(t['sid']) for t in teachers.values()}
    target = torch.tensor([by_root[int(r)] for r in roots])
    x = torch.from_numpy(data['x'])
    legal = model.legal(x)
    np.testing.assert_array_equal(legal.numpy(),data['sources'].any(1))
    teacher_action = torch.from_numpy(data['sources'][np.arange(len(x)),data['chosen']].astype(bool))
    assert teacher_action[torch.arange(len(x)),target].all()
    distinct = torch.from_numpy(data['valid'].sum(1)>1)
    protocol = dict(sids=sids,seed=hard.SEED,head='single 64x64 ReLU MLP with ten fixed CAP logits',
                    target='fastest successful complete trajectory: one CAP ID per episode; ties by ID',
                    primary='24000 vs fixed hard action actor, natural held success',secondary='4500 extra updates vs same-budget hard action actor',
                    panels=PANELS,frozen=frozen['development'],held=frozen['held'],
                    norms=study.sha(hard.BASE/'normalizers.npz'),teachers_sha=study.sha(hard.OUT/'teachers.json'),
                    hard_model=study.sha(hard.OUT/'actor.pt'),hard_continued=study.sha(hard.OUT/'polish.pt'),
                    teacher_rooms=dict(Counter(t['sid'] for t in teachers.values())),boundaries=len(x),
                    shared_action=int((teacher_action.sum(1)>1).sum()),parameters=sum(p.numel() for p in model.parameters()),
                    hard_parameters=sum(p.numel() for p in hard.actor().parameters()))
    study.save(OUT/'protocol.json',protocol)
    z = torch.cat([model.encode(chunk) for chunk in x.split(128)])
    # Encoding is parameter-free; each training batch sees the same rows as the hard actor.
    buckets=[np.flatnonzero(roots==r) for r in np.unique(roots)]
    sizes=np.array([len(b) for b in buckets])
    offsets=np.cumsum(np.r_[0,sizes[:-1]])
    indices=np.concatenate(buckets)
    curve=[]
    for key,end,lr in [('fixed',hard.STEPS,3e-4),('continued',EXTRA,3e-5)]:
        rng=np.random.default_rng(hard.SEED)
        opt=torch.optim.Adam(model.parameters(),lr=lr)
        for step in range(end+1):
            if step==end or step in hard.CHECKS:
                line=dict(checkpoint=key,step=step,**assess(model,z,legal,target,teacher_action,roots,distinct))
                curve.append(line)
                study.save(OUT/'fit.json',curve)
                print('FIT',line,flush=True)
            if step==end:
                break
            groups=rng.integers(len(buckets),size=64)
            batch=indices[offsets[groups]+(rng.random(64)*sizes[groups]).astype(int)]
            logits=model.net(z[batch]).masked_fill(~legal[batch],-torch.inf)
            loss=F.cross_entropy(logits,target[batch])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            opt.step()
        path=OUT/(key+'.pt')
        tmp=path.with_suffix('.tmp')
        torch.save(model.state_dict(),tmp)
        tmp.replace(path)


if __name__=='__main__':
    train()
