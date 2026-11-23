"""Continue each locked fourth-round self-router for four more matched waves."""
import argparse
from pathlib import Path

import numpy as np
import torch
from tools import cap_self_replication as first

learner, study, d, e, exp = first.learner, first.study, first.d, first.e, first.exp
OUT = first.OUT.parent/"cap_self_budget_20261123_01"
DOC = first.DOC.parent/"2026-11-23_self_budget"
TRAIN_SEED, CONFIRM_SEED = 662320511, 662320577
SEEDS = first.SEEDS


def fixtures(room_seed):
    rooms = {e.s.phase.input_hash(f): f for f in d.rooms(room_seed).values()}
    assert len(rooms) == 200
    return rooms


def freeze():
    prior = e.read(first.OUT/"protocol.json")
    known = e.s.collect_identities()
    for name in ("train","fresh","diagnostic"):
        known.update(prior[name])
    for group in e.read(exp.OUT/"protocol.json")["ids"].values():
        known.update(group)
    for group in e.read(d.OUT/"protocol.json")["rooms"]:
        known.update(group.values())
    for path in (exp.trend.OUT/"confirmation_ids.json", exp.p.OUT/"fresh_ids.json",
                 d.source.OUT/"confirmation_ids.json", e.OUT/"confirmation_ids.json",e.old.OUT/"fresh_panel.json"):
        known.update(e.read(path).values())
    train,confirm=fixtures(TRAIN_SEED),fixtures(CONFIRM_SEED)
    assert not (set(train)&known or set(confirm)&(known|set(train)))
    parents=e.read(first.OUT/"locked.json")
    plan=dict(seeds=list(SEEDS),train=exp.shuffled(train,str(TRAIN_SEED)),confirm=sorted(confirm),
              trend=prior["fresh"],parents={str(s):parents[f"s{s}_r4"] for s in SEEDS},
              train_room_seed=TRAIN_SEED,confirm_room_seed=CONFIRM_SEED,trend_room_seed=first.FRESH_SEED,
              rounds=[5,6,7,8],diagnostic_rounds=[4,5,6,7,8],confirm_models=["r4","r8","B"],
              training="50 new rooms per round, same order and prefix sampling rule across seeds, unchanged self collection and continuation, masked semantics, 300 Adam updates each round with seed + absolute round",
              primary="mean paired round8 minus round4 success across three locked models on fresh 200, compare each and the mean to B",
              analysis="room-cluster bootstrap 10000 replicates; report paired rescues, losses, retention; no checkpoint selection")
    if (OUT/"protocol.json").exists():
        assert e.read(OUT/"protocol.json")==plan
    else:
        e.save(OUT/"protocol.json",plan)


def initialize():
    torch.set_num_threads(1)
    member=e.read(e.OUT/"member.json")
    code=Path(member["path"]).read_text()
    assert e.base.digest(code)==member["sha"]
    return e.init(code,profile=e.perception.DEFAULT)


def checkpoint(seed,stage):
    root=first.model_dir(seed) if stage==4 else OUT/f"s{seed}"/"self"
    return root/f"actor{stage}.npz"


def train(seed):
    plan=e.read(OUT/"protocol.json")
    profile=initialize()
    prior=e.read(first.model_dir(seed)/"protocol.json")
    assert prior["seed"]==seed and prior["profile"]==profile
    parent=checkpoint(seed,4)
    assert d.source.graph.file_sha(parent)==plan["parents"][str(seed)]
    rooms=fixtures(TRAIN_SEED)
    assert set(rooms)==set(plan["train"])
    study.ROOMS.update(rooms)
    study.OUT=OUT/f"s{seed}"
    directory=study.OUT/"self"
    learner.out=learner.p.OUT=exp.OUT=directory
    exp.WORKERS=first.WORKERS
    e.save(directory/"start.json",dict(seed=seed,round4_sha=plan["parents"][str(seed)],profile=profile))
    weights=study.cold.weights(parent)
    for stage,keys in zip(plan["rounds"],np.array_split(plan["train"],4),strict=True):
        fit=directory/f"fit{stage}.json"
        if fit.exists():
            assert e.read(fit)["sha"]==d.source.graph.file_sha(checkpoint(seed,stage))
            weights=study.cold.weights(checkpoint(seed,stage))
        else:
            weights=learner.train_round(stage,keys,seed,weights)
    e.save(directory/"locked.json",{i:d.source.graph.file_sha(checkpoint(seed,i)) for i in range(4,9)})


def evaluate():
    plan=e.read(OUT/"protocol.json")
    profile=initialize()
    exp.OUT=OUT
    exp.WORKERS=first.WORKERS
    hashes={}
    for seed in SEEDS:
        root=OUT/f"s{seed}"/"self"
        assert e.read(root/"start.json")==dict(seed=seed,round4_sha=plan["parents"][str(seed)],profile=profile)
        for stage,sha in e.read(root/"locked.json").items():
            model=f"s{seed}_r{stage}"
            file=checkpoint(seed,int(stage))
            assert d.source.graph.file_sha(file)==sha
            first.WEIGHTS[model]=study.cold.weights(file)
            hashes[model]=sha
    assert len(hashes)==15
    e.save(OUT/"locked.json",hashes)
    first.OUT=OUT  # place reused evaluation worker's files in this run
    for panel,room_seed in (("confirm",CONFIRM_SEED),("trend",first.FRESH_SEED)):
        rooms=fixtures(room_seed)
        assert set(rooms)==set(plan[panel])
        study.ROOMS.update(rooms)
        batches=([("confirm",[f"s{s}_r{r}" for s in SEEDS for r in (4,8)]+["B"])] if panel=="confirm"
                 else [(f"trend_r{r}",[f"s{s}_r{r}" for s in SEEDS]) for r in (5,6,7,8)])
        for name, models in batches:
            jobs=[(panel,key,m) for key in plan[panel] for m in models
                  if not (OUT/panel/"test"/m/(key+".json")).exists()]
            exp.timed_pool(name,first.evaluate_one,jobs)
            if panel=="trend":
                from tools.cap_self_budget_report import plot_trend
                prior=e.read(first.DOC/"results.json")["panels"]["fresh"]
                totals={"B":prior["B"],**{f"s{s}_r4":prior["seed"][str(s)]["final"] for s in SEEDS}}
                for r in range(5,int(name.removeprefix("trend_r"))+1):
                    for s in SEEDS:
                        m=f"s{s}_r{r}"
                        totals[m]=sum(e.read(OUT/panel/"test"/m/(key+".json"))["success"] for key in plan[panel])
                DOC.mkdir(exist_ok=True,parents=True)
                plot_trend(totals)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("step",choices=["freeze","train","evaluate"])
    parser.add_argument("--seed",type=int,choices=SEEDS)
    args=parser.parse_args()
    if args.step=="freeze":
        freeze()
    elif args.step=="train":
        assert args.seed is not None
        train(args.seed)
    else:
        evaluate()
