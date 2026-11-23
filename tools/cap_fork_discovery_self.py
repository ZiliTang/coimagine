"""A cold actor collects its own prefixes and both counterfactual continuations."""
import numpy as np
import torch
from torch import nn
from tools import cap_fork_discovery as study

d,e,p,exp=study.d,study.e,study.p,study.exp
out=study.OUT/"self"
mode="history"


def alt(job):
    key,i=job
    parent=e.read(out/"parents"/(key+".json"))
    n, prefix=parent["points"][i]
    with np.load(out/"parents"/(key+".npz")) as data:
        x=data["x"][i]
        chosen=int(data["b"][i])
    other=("B","A")[chosen]
    tail,_=study.rollout(key,"self",(n,prefix,x,other),continuation="self")
    result=[None,None]
    result[chosen]=int(parent["success"])
    result[1-chosen]=int(tail["success"])
    e.save(out/"forks"/f"{key}_{i}.json",dict(outcomes=result,prefix=prefix,ticks=[parent["ticks"],tail["ticks"]]))


def evaluate(job):
    model,key=job
    path=out/f"actor{model}.npz"
    d.MODELS[mode]=study.cold.weights(path)
    r=d.episode(study.ROOMS[key], mode,label=False)[0]
    e.save(study.OUT/"test"/f"self{model}"/(key+".json"),dict(success=r["success"],ticks=r["ticks"]))


def train_round(step,block,seed,weights):
    d.MODELS["history"]=weights
    exp.timed_pool(f"parents{step}",study.collect_parent,[(k,"self") for k in block if not (out/"parents"/(k+".json")).exists()])
    schedule={}
    for k in block:
        nr=len(e.read(out/"parents"/(k+".json"))["points"])
        schedule[k]=[exp.shuffled(list(map(int,group)),f"{k}/{j}")[0]
                     for j,group in enumerate(np.array_split(np.arange(nr),min(nr,3)))] if nr else []
    exp.timed_pool(f"pairs{step}",alt,[(k,i) for k in block for i in schedule[k] if not (out/"forks"/f"{k}_{i}.json").exists()])
    anchors,xs,ys=[],[],[]
    for k in block:
        with np.load(out/"parents"/(k+".npz")) as data:
            x=d.project(data["x"],mode)
            assert np.array_equal((d.logits(x,mode)>0).astype(int),data["b"])
        anchors.append(x)
        for i in schedule[k]:
            a,b=e.read(out/"forks"/f"{k}_{i}.json")["outcomes"]
            if a!=b:
                xs.append(x[i])
                ys.append(b)
    d.MODELS["semantic"]=weights  # p.fit uses this key for the masked regression arrays.
    p.fit_preferences(step,np.concatenate(anchors),np.asarray(xs),np.asarray(ys,np.float32),
                      dict(rooms=len(block),forks=sum(map(len,schedule.values()))),seed=seed)
    return d.MODELS["semantic"]


def main(*, seed=17, ids=None, workers=8, train_only=False):
    if ids is None:
        ids=e.read(study.OUT/"protocol.json")
    keys, held=ids["train"],ids["held"]
    # Only the two frozen CAPs and the public transition environment are initialized.
    member=e.read(e.OUT/"member.json")
    from pathlib import Path
    source=Path(member["path"]).read_text()
    assert e.base.digest(source)==member["sha"]
    profile=e.init(source,profile=e.perception.DEFAULT)
    for room_seed,expected in [(study.TRAIN_SEED,keys),(study.TEST_SEED,held)]:
        fixtures={e.s.phase.input_hash(f):f for f in d.rooms(room_seed).values()}
        assert set(fixtures)==set(expected)
        study.ROOMS.update(fixtures)
    exp.OUT=out
    exp.WORKERS=workers
    p.OUT=out
    out.mkdir(exist_ok=True,parents=True)
    e.save(out/"protocol.json",dict(rooms=keys,test=held,profile=profile,seed=seed,inputs="history, last 9 explicitly masked",
          initialization=f"PyTorch seed{seed}, no pretrained parameters",forks="up to 3 per room, temporally stratified",
          continuation="frozen pre-round self",training="50 fresh rooms per wave, 4 waves, 300 updates each",
          comparison="final vs A, B, and cold0; all on same natural 200 rooms"))
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    net=nn.Sequential(nn.Linear(349,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,1))
    weights=[q.detach().numpy().T.copy() for q in net.parameters()]
    np.savez(out/"actor0.npz",*weights)
    for step,block in enumerate(np.array_split(keys,4),1):
        weights=train_round(step,block,seed,weights)
    e.save(out/"locked.json",{i:d.source.graph.file_sha(out/f"actor{i}.npz") for i in (0,4)})
    if train_only:
        return
    exp.timed_pool("test",evaluate,[(i,k) for k in held for i in (0,4)
                   if not (study.OUT/"test"/f"self{i}"/(k+".json")).exists()])
    e.save(out/"done.json",dict(rooms=len(held)))


if __name__=="__main__":
    main()
