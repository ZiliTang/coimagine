"""One paired, ungated update with actor-only or mixed successful replay."""
from collections import Counter
import sys

import numpy as np
from tools import cap_success_replay as old

prior, study, d, e, exp = old.prior, old.study, old.d, old.e, old.exp
OUT = prior.OUT.parent / "cap_ungated_expert_replay_20261123_01"
DOC = prior.DOC.parent / "2026-11-23_ungated_expert_replay"
FRESH_SEED = 662321541
SEEDS = prior.SEEDS
WEIGHTS = {}
ARMS = ("actor", "mixed")


def freeze():
    base = e.read(prior.OUT / "protocol.json")
    new = prior.fixtures(FRESH_SEED)
    train = base["train"][:50]
    known = e.s.collect_identities()
    for root, fields in [(prior.OUT, ("train", "confirm", "trend")), (old.OUT, ("train","fresh")),
                         (prior.first.OUT, ("train","fresh","diagnostic")), (prior.first.PREVIOUS, ("train","held"))]:
        for field in fields:
            known.update(e.read(root/"protocol.json")[field])
    for group in e.read(exp.OUT/"protocol.json")["ids"].values():
        known.update(group)
    for group in e.read(d.OUT/"protocol.json")["rooms"]:
        known.update(group.values())
    for path in (exp.trend.OUT/"confirmation_ids.json",exp.p.OUT/"fresh_ids.json",
                 d.source.OUT/"confirmation_ids.json",e.OUT/"confirmation_ids.json",e.old.OUT/"fresh_panel.json"):
        known.update(e.read(path).values())
    assert not set(new) & known
    study.ROOMS.update(prior.fixtures(prior.TRAIN_SEED))
    study.ROOMS.update(new)
    plan = dict(train=train, fresh=sorted(new), fresh_seed=FRESH_SEED, seeds=list(SEEDS),
                parents=base["parents"], profile=prior.initialize(), arms=list(ARMS), updates=300,
                lr=.0003, batch=dict(strict=64, soft=128, margin=64), margin=.2, margin_weight=.25,
                retention="64 own successful vs 32 own + 16 pure A + 16 pure B, trajectory-uniform",
                forks="up to 3 time strata per ungated parent; one forced action, same parent continuation",
                gate=False, features="history, teacher semantics masked; stall history retained",
                primary="fresh mixed minus actor-only overall and expert-union recoveries, outside-union parent retention")
    if (OUT/"protocol.json").exists():
        assert plan == e.read(OUT/"protocol.json")
    else:
        e.save(OUT/"protocol.json",plan)
    for seed in SEEDS:
        path = prior.checkpoint(seed,4)
        assert d.source.graph.file_sha(path)==plan["parents"][str(seed)]
        WEIGHTS[f"s{seed}_parent"] = study.cold.weights(path)
    exp.OUT, exp.WORKERS = OUT, prior.first.WORKERS
    return plan


def native(key, who):
    cap = e.s.phase.FrozenCAP(e.AM if who=="A" else e.BM)
    counts=Counter()
    with e.s.phase.world(study.ROOMS[key]) as (w,_):
        cap.update(w.history)
        while not e.s.ended(w):
            e.s.tick_action(w,cap.propose(e.s.observe(w)),e.g.prep.prior.EXECUTOR,cap.update,counts)
        return e.s.output(w,counts)


def play(job, *, output=None):
    panel,key,name = job
    root = (OUT if output is None else output)/panel/name
    expert = name in ("A","B")
    if not expert:
        d.MODELS["history"] = WEIGHTS[name]
    r, points = study.rollout(key,"expert_"+name if expert else "self",stall_gate=False)
    if expert:
        ref = native(key,name)
        assert all(r[k]==v for k,v in ref.items()), (key,name,r,ref)
    row = {k:r[k] for k in ("ticks","success","history","first_stall")}
    row["different"] = len(points)
    row["pre_stall_B"] = sum(p[3] for p in points if r["first_stall"] is None or p[0]<r["first_stall"])
    if panel=="train" and (expert or name.endswith("_parent")):
        row["points"] = [[p[0],p[1]] for p in points]
        exp.save_npz(root/(key+".npz"),x=np.asarray([p[2] for p in points],np.float32).reshape(-1,349),
                     b=np.asarray([p[3] for p in points],int))
    e.save(root/(key+".json"),row)


def branch(job):
    key,seed,idx = job
    name=f"s{seed}_parent"
    path=OUT/"train"/name/(key+".json")
    row=e.read(path)
    n,prefix=row["points"][idx]
    with np.load(path.with_suffix(".npz")) as z:
        b=int(z["b"][idx])
        x=z["x"][idx]
    d.MODELS["history"]=WEIGHTS[name]
    r,_=study.rollout(key,"self",(n,prefix,x,"A" if b else "B"),continuation="self",stall_gate=False)
    outcomes=[None,None]
    outcomes[b],outcomes[1-b]=row["success"],r["success"]
    e.save(OUT/"forks"/name/f"{key}_{idx}.json",dict(n=n,prefix=prefix,outcomes=outcomes))


def run_panel(plan,panel,names,stage):
    jobs=[(panel,k,n) for k in plan[panel] for n in names if not (OUT/panel/n/(k+".json")).exists()]
    exp.timed_pool(stage,play,jobs)


def main():
    plan=freeze()
    if sys.argv[1]=="collect":
        run_panel(plan,"train",["A","B",*WEIGHTS],"collection")
        jobs=[]
        for key in plan["train"]:
            for seed in SEEDS:
                name=f"s{seed}_parent"
                row=e.read(OUT/"train"/name/(key+".json"))
                n=len(row["points"])
                if n:
                    for j,group in enumerate(np.array_split(np.arange(n),min(n,3))):
                        idx=exp.shuffled(list(map(int,group)),f"{key}/{j}")[0]
                        if not (OUT/"forks"/name/f"{key}_{idx}.json").exists():
                            jobs.append((key,seed,idx))
        exp.timed_pool("forks",branch,jobs)
        e.save(OUT/"collection_done.json",dict(total=sum(1 for _ in (OUT/"forks").glob("*/*.json"))))
    elif sys.argv[1]=="fit-eval":
        from tools.cap_ungated_fit import fit
        assert (OUT/"collection_done.json").exists()
        for seed in SEEDS:
            for mode in ARMS:
                name=f"s{seed}_{mode}"
                WEIGHTS[name]=fit(seed,mode,plan)
        e.save(OUT/"locked.json",{n:d.source.graph.file_sha(OUT/"models"/(n+".npz")) if not n.endswith("_parent")
                                 else plan["parents"][n.split("_")[0][1:]] for n in WEIGHTS})
        run_panel(plan,"fresh",["A","B",*WEIGHTS],"fresh")
        run_panel(plan,"train",[n for n in WEIGHTS if not n.endswith("_parent")],"train_updated")
    else:
        raise ValueError(sys.argv[1])


if __name__=="__main__":
    main()
