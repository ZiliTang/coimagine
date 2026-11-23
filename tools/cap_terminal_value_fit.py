"""Independent supervised on-policy values. The distance head outputs meters."""
from concurrent.futures import ProcessPoolExecutor
import argparse
import numpy as np
import torch
from torch import nn

from tools import cap_terminal_value_study as study

SEEDS=(17,29,43)
MODELS=study.OUT/'models'


def read(fold,expert):
    rows=[]
    for path in sorted((study.OUT/'runs'/fold/expert).glob('*.npz')):
        with np.load(path) as z:
            rows.append(dict(room=path.stem,**{name:z[name] for name in z.files}))
    return rows


def net(width):
    return nn.Sequential(nn.Linear(width,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))


def train(job):
    expert,seed=job
    dest=MODELS/f'{expert}_{seed}.pt'
    if dest.exists():return
    rows=read('development',expert)
    assert len(rows)==400
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed)
    lengths=np.array([len(r['x']) for r in rows])
    starts=np.cumsum(lengths)-lengths
    x=np.concatenate([r['x'] for r in rows])
    weights=np.repeat(1/(len(rows)*lengths),lengths)
    mu=np.average(x,axis=0,weights=weights).astype(np.float32)
    sd=np.sqrt(np.average((x-mu)**2,axis=0,weights=weights)).clip(.05).astype(np.float32)
    x=torch.from_numpy(((x-mu)/sd).clip(-8,8))
    labels=torch.tensor([[r['d'].item(),r['y'].item()] for r in rows],dtype=torch.float32)
    models=[net(x.shape[1]) for _ in range(2)]
    opt=torch.optim.AdamW([p for m in models for p in m.parameters()],lr=.0005)
    for _ in range(2000):
        episodes=rng.integers(len(rows),size=128)
        ix=starts[episodes]+(rng.random(len(episodes))*lengths[episodes]).astype(int)
        values=x[ix]; targets=labels[episodes]
        distance=models[0](values).flatten()
        logits=models[1](values).flatten()
        loss=(distance-targets[:,0]).square().mean()+nn.functional.binary_cross_entropy_with_logits(logits,targets[:,1])
        opt.zero_grad()
        loss.backward()
        opt.step()
    tmp=dest.with_suffix('.tmp')
    torch.save(dict(mu=torch.from_numpy(mu),sd=torch.from_numpy(sd),distance=models[0].state_dict(),success=models[1].state_dict()),tmp)
    tmp.replace(dest)
    print('fit',expert,seed,flush=True)


def load_models(expert):
    models=[]
    for seed in SEEDS:
        ck=torch.load(MODELS/f'{expert}_{seed}.pt',weights_only=True)
        ds=net(len(ck['mu']));cl=net(len(ck['mu']))
        ds.load_state_dict(ck['distance']);cl.load_state_dict(ck['success'])
        models.append((ck['mu'],ck['sd'],ds.eval(),cl.eval()))
    return models


@torch.inference_mode()
def predict(models,rows):
    x=torch.from_numpy(np.concatenate([r['x'] for r in rows]))
    ds=[];pr=[]
    for mu,sd,m1,m2 in models:
        v=((x-mu)/sd).clamp(-8,8)
        ds.append(m1(v).flatten().clamp_min(0).numpy())
        pr.append(m2(v).flatten().sigmoid().numpy())
    indices=np.cumsum([len(r['x']) for r in rows])[:-1]
    for key,arr in [('pred_d',ds),('pred_p',pr)]:
        for r,v in zip(rows,np.split(np.mean(arr,axis=0),indices)):
            r[key]=v
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('experts',nargs='*')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    torch.set_num_threads(1)
    MODELS.mkdir(parents=True,exist_ok=True)
    experts=args.experts or list(study.study.read(study.OUT/'protocol.json')['experts'])
    with ProcessPoolExecutor(args.workers) as pool:
        list(pool.map(train,[(e,s) for e in experts for s in SEEDS]))


if __name__=='__main__':main()
