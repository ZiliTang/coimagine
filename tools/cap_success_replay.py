"""Single paired update with trajectory-uniform margin retention on saved round-5 forks."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch import nn
from tools import cap_self_budget as prior

study, d, e, exp = prior.study, prior.d, prior.e, prior.exp
OUT = prior.OUT.parent / "cap_success_replay_20261123_01"
DOC = prior.DOC.parent / "2026-11-23_success_replay"
SEEDS = prior.SEEDS
MODES = ("parent","vanilla","all","success")
NEW_ROOM_SEED = 662321067
MARGIN = 0.2
WEIGHT = 0.25
MODELS = {}


def freeze():
    saved=e.read(prior.OUT/"protocol.json")
    train=saved["train"][:50]
    fresh=prior.fixtures(NEW_ROOM_SEED)
    known=e.s.collect_identities()
    for name,fields in [(prior.OUT,("train","confirm","trend")),(prior.first.OUT,("train","fresh","diagnostic")),
                        (prior.first.PREVIOUS,("train","held"))]:
        for field in fields:
            known.update(e.read(name/"protocol.json")[field])
    for group in e.read(exp.OUT/"protocol.json")["ids"].values():
        known.update(group)
    for group in e.read(d.OUT/"protocol.json")["rooms"]:
        known.update(group.values())
    for path in (exp.trend.OUT/"confirmation_ids.json",exp.p.OUT/"fresh_ids.json",
                 d.source.OUT/"confirmation_ids.json",e.OUT/"confirmation_ids.json",e.old.OUT/"fresh_panel.json"):
        known.update(e.read(path).values())
    assert not set(fresh)&known
    plan=dict(train=train,fresh=sorted(fresh),fresh_room_seed=NEW_ROOM_SEED,seeds=list(SEEDS),
              modes=list(MODES),parent=saved["parents"],vanilla={str(s):e.read(prior.OUT/f"s{s}"/"self"/"fit5.json")["sha"] for s in SEEDS},
              margin_logit=MARGIN,margin_weight=WEIGHT,updates=300,learning_rate=.0003,
              minibatches=[64,128,64],hard_soft_rng="identical seed + 5; optional margin draws use isolated common RNG",
              extra="uniform over trajectories with eligible non-informative or untested boundaries, then uniform over those boundaries; old executed greedy label",
              success_only="same draw rule but restrict to successful parent trajectories; all unique-winner measured pairs excluded from extra in both arms",
              primary="on new 200 same-room success arm vs all arm; retention of round4 successes, rescue of round4 failures and net difference",
              secondary="vanilla vs both; rollout on the 50 training rooms and teacher-forced choice changes on saved successful prefixes")
    if (OUT/"protocol.json").exists():
        assert e.read(OUT/"protocol.json")==plan
    else:
        e.save(OUT/"protocol.json",plan)
    return plan


def training_data(seed,plan):
    root=prior.OUT/f"s{seed}"/"self"
    parent=study.cold.weights(prior.checkpoint(seed,4))
    d.MODELS["history"]=parent
    anchors,labels,valid_rooms,success_mask=[],[],[],[]
    hard_ids,hard_labels=[],[]
    offset=0
    digest=hashlib.sha256()
    for k in plan["train"]:
        path=root/"parents"/(k+".npz")
        row=e.read(path.with_suffix(".json"))
        with np.load(path) as z:
            x=d.project(z["x"],"history")
            b=z["b"]
        assert np.array_equal(d.logits(x,"history")>0,b)
        digest.update(d.source.graph.file_sha(path).encode())
        digest.update(d.source.graph.file_sha(path.with_suffix(".json")).encode())
        anchors.append(x)
        labels.append(b)
        success_mask.append(np.full(len(x),bool(row["success"])))
        keep=np.ones(len(x),bool)
        files=sorted((root/"forks").glob(k+"_*.json"),key=lambda f:int(f.stem.split("_")[-1]))
        # Original sampler emits at most one index from each ascending time stratum.
        for file in files:
            i=int(file.stem.split("_")[-1])
            fork=e.read(file)
            assert fork["prefix"]==row["points"][i][1] and fork["outcomes"][int(b[i])]==row["success"]
            digest.update(d.source.graph.file_sha(file).encode())
            a,alt=fork["outcomes"]
            if a!=alt:
                hard_ids.append(offset+i)
                hard_labels.append(alt)
                keep[i]=False
        if keep.any():
            valid_rooms.append((bool(row["success"]),np.flatnonzero(keep)+offset))
        offset+=len(x)
    x=np.concatenate(anchors)
    assert len(hard_ids)==e.read(root/"fit5.json")["informative"]
    return x,np.concatenate(labels),np.concatenate(success_mask),np.array(hard_ids),np.array(hard_labels,np.float32),valid_rooms,digest.hexdigest()


def fit(seed,mode,plan,data,*,meter=None,soft_enabled=True,margin_batch=64):
    x,b,success,ids,y,rooms,data_hash=data
    parent=study.cold.weights(prior.checkpoint(seed,4))
    model=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    with torch.no_grad():
        for parameter,value in zip(model.parameters(),parent,strict=True):
            parameter.copy_(torch.from_numpy(value.T.copy()))
    torch.manual_seed(seed+5)
    optim=torch.optim.Adam(model.parameters(),lr=.0003)
    anchors=torch.from_numpy(x)
    hard=anchors[ids]
    targets=torch.from_numpy(y)
    sign=torch.tensor(2*b-1,dtype=torch.float32)
    if soft_enabled:
        with torch.no_grad():
            original=model(anchors).flatten().sigmoid()
    pool=[indices for ok,indices in rooms if mode=="all" or ok or mode=="vanilla"]
    rng=np.random.default_rng(50000+seed)
    extra_rng=np.random.default_rng(60000+seed)
    for _ in range(300):
        i=torch.randint(len(ids),(64,))
        j=torch.randint(len(x),(128,))  # Keep the historical hard-minibatch RNG stream in both arms.
        pref=nn.functional.binary_cross_entropy_with_logits(model(hard[i]).flatten(),targets[i])
        soft=None
        loss=pref
        if soft_enabled:
            soft=nn.functional.binary_cross_entropy_with_logits(model(anchors[j]).flatten(),original[j])
            loss=loss+soft
        margin=None
        margin_active=None
        if mode!="vanilla":
            u,v=rng.random((2,64))
            if margin_batch>64:
                extra_u,extra_v=extra_rng.random((2,margin_batch-64))
                u,v=np.concatenate((u,extra_u)),np.concatenate((v,extra_v))
            selected=[pool[int(a*len(pool))] for a in u]
            idx=np.array([r[int(q*len(r))] for r,q in zip(selected,v,strict=True)])
            logit=model(anchors[idx]).flatten()
            violation=MARGIN-sign[idx]*logit
            margin=WEIGHT*torch.relu(violation).mean()
            if meter is not None:
                margin_active=float((violation.detach()>0).float().mean())
            loss=loss+margin
        if meter is not None:
            meter.before((pref,soft,margin),model,optim,margin_active)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if meter is not None:
            meter.after(model,optim)
    weights=[p.detach().numpy().T.copy() for p in model.parameters()]
    if mode=="vanilla":
        expected=study.cold.weights(prior.checkpoint(seed,5))
        assert all(np.array_equal(a,b) for a,b in zip(weights,expected,strict=True))
    name=f"s{seed}_{mode}"
    path=OUT/"models"/(name+".npz")
    if meter is not None:
        saved=study.cold.weights(path)
        assert all(np.array_equal(v,w) for v,w in zip(weights,saved,strict=True))
        return
    exp.save_npz(path,**{f"arr_{k}":v for k,v in enumerate(weights)})
    MODELS[name]=weights
    d.MODELS["history"]=weights
    prediction=d.logits(x,"history")>0
    prev=x[:,-16].astype(bool)
    e.save(OUT/"fit"/(name+".json"),dict(seed=seed,mode=mode,data_hash=data_hash,sha=d.source.graph.file_sha(path),
           pool_rooms=len(pool),pool_boundaries=sum(map(len,pool)),successful_anchors=int(success.sum()),
           changed_on_success=int(sum(prediction[success]!=b[success])),hard_correct=int(sum(prediction[ids]==y)),hard=len(ids),
           chose_different_expert=int(sum(prediction!=prev)),parent_switches=int(sum(b!=prev)),anchors=len(x)))


def score(job):
    panel,key,name=job
    d.MODELS["history"]=MODELS[name]
    r=d.episode(study.ROOMS[key],"history",label=False)[0]
    e.save(OUT/panel/name/(key+".json"),dict(success=int(r["success"]),ticks=r["ticks"]))


def main():
    torch.set_num_threads(1)
    plan=freeze()
    profile=prior.initialize()
    for seed in SEEDS:
        root=prior.OUT/f"s{seed}"/"self"
        assert e.read(root/"start.json")==dict(seed=seed,round4_sha=plan["parent"][str(seed)],profile=profile)
        assert d.source.graph.file_sha(prior.checkpoint(seed,4))==plan["parent"][str(seed)]
        assert d.source.graph.file_sha(prior.checkpoint(seed,5))==plan["vanilla"][str(seed)]
        MODELS[f"s{seed}_parent"]=study.cold.weights(prior.checkpoint(seed,4))
        data=training_data(seed,plan)
        for mode in MODES[1:]:
            fit(seed,mode,plan,data)
    e.save(OUT/"locked.json",{n:e.read(OUT/"fit"/(n+".json"))["sha"] if not n.endswith("_parent")
            else plan["parent"][n.split("_")[0][1:]] for n in MODELS})
    exp.OUT=OUT
    exp.WORKERS=prior.first.WORKERS
    for panel,room_seed in [("train",prior.TRAIN_SEED),("fresh",NEW_ROOM_SEED)]:
        rooms=prior.fixtures(room_seed)
        assert set(plan[panel])<=set(rooms)
        study.ROOMS.update(rooms)
        for seed in SEEDS:
            if panel=="train":
                for key in plan[panel]:
                    r=e.read(prior.OUT/f"s{seed}"/"self"/"parents"/(key+".json"))
                    e.save(OUT/panel/f"s{seed}_parent"/(key+".json"),dict(success=int(r["success"]),ticks=r["ticks"]))
        jobs=[(panel,k,n) for k in plan[panel] for n in MODELS if not (OUT/panel/n/(k+".json")).exists()]
        exp.timed_pool(panel,score,jobs)


if __name__=="__main__":
    main()
