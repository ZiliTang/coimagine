"""Paired retention minibatches on independently collected ungated trajectories."""
import hashlib

import numpy as np
import torch
from torch import nn

from tools import cap_ungated_replay as run


def signatures(x):
    # Compare only actor-visible coordinates, canonicalizing signed zero.
    return [hashlib.sha256(np.where(row==0, np.float32(0),row).tobytes()).digest() for row in x]


def load_rows(seed,mode,plan):
    root = run.OUT
    parent = f"s{seed}_parent"
    xs,bs,keys=[],[],[]
    strict_ids,targets=[],[]
    routes={n:[] for n in ("own","A","B")}
    offset=0
    for source in ([parent] if mode=="actor" else [parent,"A","B"]):
        for room in plan["train"]:
            path=root/"train"/source/(room+".json")
            meta=run.e.read(path)
            if source!=parent and not meta["success"]:
                continue
            with np.load(path.with_suffix(".npz")) as z:
                x=run.d.project(z["x"],"history")
                b=z["b"]
            xs.append(x)
            bs.append(b)
            keys.extend(signatures(x))
            if meta["success"] and len(x):
                routes["own" if source==parent else source].append(np.arange(offset,offset+len(x)))
            if source==parent:
                for file in sorted((root/"forks"/parent).glob(room+"_*.json")):
                    idx=int(file.stem.split("_")[-1])
                    row=run.e.read(file)
                    assert [row["n"],row["prefix"]]==meta["points"][idx]
                    outcomes=row["outcomes"]
                    assert outcomes[b[idx]]==meta["success"]
                    if outcomes[0]!=outcomes[1]:
                        strict_ids.append(offset+idx)
                        targets.append(outcomes[1])
            offset+=len(x)
        if source==parent:
            anchors=offset
    assert len(strict_ids)>0
    x=np.concatenate(xs)
    b=np.concatenate(bs)
    strict={keys[i] for i in strict_ids}
    votes={}
    for rooms in routes.values():
        for indices in rooms:
            for i in indices:
                key=keys[i]
                votes[key]=votes.get(key,0) | (1<<int(b[i]))
    bad=strict | {k for k,v in votes.items() if v==3}
    for name,rooms in routes.items():
        filtered=[np.asarray([i for i in a if keys[i] not in bad],int) for a in rooms]
        routes[name]=[a for a in filtered if len(a)]
    return x,b,anchors,np.array(strict_ids),np.asarray(targets,np.float32),routes,sum(v==3 for v in votes.values())


def draw(rooms, u, v):
    if not rooms:
        raise ValueError("Empty eligible successful demonstration pool")
    picked=[rooms[int(q*len(rooms))] for q in u]
    return [indices[int(q*len(indices))] for indices,q in zip(picked,v,strict=True)]


def fit(seed,mode="no_replay",plan=None,*,output=None):
    """Fit an ungated actor; successful-trajectory replay is opt-in."""
    if plan is None:
        plan=run.e.read(run.OUT/"protocol.json")
    output = run.OUT if output is None else output
    if mode not in ("actor","mixed","no_replay"):
        raise ValueError(mode)
    x,b,anchors,ids,y,routes,ambiguous=load_rows(seed,"actor" if mode=="no_replay" else mode,plan)
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    parent=run.study.cold.weights(run.prior.checkpoint(seed,4))
    with torch.no_grad():
        for param,w in zip(model.parameters(),parent,strict=True):
            param.copy_(torch.from_numpy(w.T.copy()))
    run.d.MODELS["history"]=parent
    assert np.array_equal(run.d.logits(x[:anchors],"history")>0,b[:anchors])
    xx=torch.from_numpy(x)
    hard=xx[ids]
    target=torch.from_numpy(y)
    signs=torch.tensor(b*2-1,dtype=torch.float32)
    with torch.no_grad():
        original=model(xx[:anchors]).flatten().sigmoid()
    rng=np.random.default_rng(50000+seed)
    torch.manual_seed(seed+5)
    optimizer=torch.optim.Adam(model.parameters(),lr=plan["lr"])
    for _ in range(plan["updates"]):
        i=torch.randint(len(ids),(64,))
        j=torch.randint(anchors,(128,))
        pref=nn.functional.binary_cross_entropy_with_logits(model(hard[i]).flatten(),target[i])
        soft=nn.functional.binary_cross_entropy_with_logits(model(xx[j]).flatten(),original[j])
        loss=pref+soft
        if mode!="no_replay":
            u,v=rng.random((2,64))
            if mode=="actor":
                idx=draw(routes["own"],u,v)
            else:
                idx=(draw(routes["own"],u[:32],v[:32])+draw(routes["A"],u[32:48],v[32:48])
                     +draw(routes["B"],u[48:],v[48:]))
            margin=torch.relu(plan["margin"]-signs[idx]*model(xx[idx]).flatten()).mean()
            loss=loss+plan["margin_weight"]*margin
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    weights=[p.detach().numpy().T.copy() for p in model.parameters()]
    name=f"s{seed}_{mode}"
    model_path=output/"models"/(name+".npz")
    run.exp.save_npz(model_path,**{f"arr_{i}":w for i,w in enumerate(weights)})
    run.e.save(output/"fit"/(name+".json"),dict(seed=seed,arm=mode,sha=run.d.source.graph.file_sha(model_path),
        strict=len(ids),strict_B=int(y.sum()),anchors=anchors,ambiguous_signatures=ambiguous,
        rooms={k:len(v) for k,v in routes.items()},boundaries={k:sum(map(len,v)) for k,v in routes.items()}))
    return weights
