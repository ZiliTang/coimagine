"""Three training seeds with a shared development stream and a new locked panel."""
import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tools import cap_fork_discovery_self as learner

study, d, e, exp = learner.study, learner.d, learner.e, learner.exp
PREVIOUS = study.OUT
OUT = PREVIOUS.parent / "cap_self_replication_20261123_01"
DOC = study.DOC.parent / "2026-11-23_self_replication"
SEEDS = (17, 29, 43)
FRESH_SEED = 662320271
WORKERS = min(len(os.sched_getaffinity(0)),24)
WEIGHTS = {}


def model_dir(seed):
    return PREVIOUS / "self" if seed == 17 else OUT / f"s{seed}" / "self"


def freeze():
    old = e.read(PREVIOUS / "protocol.json")
    fixtures = {e.s.phase.input_hash(f): f for f in d.rooms(FRESH_SEED).values()}
    known = set(old["train"] + old["held"]) | e.s.collect_identities()
    for group in e.read(exp.OUT / "protocol.json")["ids"].values():
        known.update(group)
    for group in e.read(d.OUT / "protocol.json")["rooms"]:
        known.update(group.values())
    for path in (exp.trend.OUT/"confirmation_ids.json", exp.p.OUT/"fresh_ids.json",
                 d.source.OUT/"confirmation_ids.json", e.OUT/"confirmation_ids.json",e.old.OUT/"fresh_panel.json"):
        known.update(e.read(path).values())
    assert len(fixtures)==200 and not set(fixtures).intersection(known)
    lock=e.read(model_dir(17)/"locked.json")
    for k,sha in lock.items():
        assert d.source.graph.file_sha(model_dir(17)/f"actor{k}.npz")==sha
    protocol=dict(seeds=list(SEEDS),train=old["train"],diagnostic=old["held"],fresh=sorted(fixtures),
        fresh_seed=FRESH_SEED,seed17_checkpoints=lock,
        training="fixed original room order and temporal fork selection rule, vary initialization and minibatches (seed + wave); own collection and own suffix; four rounds, 300 updates each",
        primary="on 200 new rooms mean across three seed finals relative to B (stronger in prior panel); also compare both single experts and each seed's own initial",
        intervals="paired bootstrap of rooms keeping all seeds together, conditional on these trained policies; list individual seed outcomes",
        endpoints=[0,4],previous_panel="diagnostic only; no checkpoint or seed selection")
    path=OUT/"protocol.json"
    if path.exists():
        assert e.read(path)==protocol
    else:
        e.save(path,protocol)


def train(seed):
    protocol=e.read(OUT/"protocol.json")
    study.OUT=OUT/f"s{seed}"
    learner.out=model_dir(seed)
    learner.main(seed=seed,ids=dict(train=protocol["train"],held=protocol["diagnostic"]),
                 workers=WORKERS,train_only=True)


def evaluate_one(job):
    panel,key,name=job
    if name in ("A","B"):
        study.OUT=OUT/panel
        study.evaluate((key,name))
    else:
        d.MODELS["history"]=WEIGHTS[name]
        result=d.episode(study.ROOMS[key],"history",label=False)[0]
        e.save(OUT/panel/"test"/name/(key+".json"),dict(success=int(result["success"]),ticks=result["ticks"]))


def verify_initial(seed, file):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        layers=[torch.nn.Linear(a,b) for a,b in [(349,64),(64,64),(64,1)]]
    parameters=[p.detach().numpy().T for layer in layers for p in layer.parameters()]
    saved=study.cold.weights(file)
    assert all(np.array_equal(a,b) for a,b in zip(parameters,saved,strict=True)), seed


def evaluate():
    torch.set_num_threads(1)
    protocol=e.read(OUT/"protocol.json")
    member=e.read(e.OUT/"member.json")
    code=Path(member["path"]).read_text()
    assert e.base.digest(code)==member["sha"]
    profile=e.init(code,profile=e.perception.DEFAULT)
    models=["A","B"]
    hashes={}
    for seed in SEEDS:
        path=model_dir(seed)
        config=e.read(path/"protocol.json")
        assert config["seed"]==seed and config["rooms"]==protocol["train"] and config["profile"]==profile
        verify_initial(seed,path/"actor0.npz")
        for stage,sha in e.read(path/"locked.json").items():
            model=f"s{seed}_r{stage}"
            file=path/f"actor{stage}.npz"
            assert d.source.graph.file_sha(file)==sha
            WEIGHTS[model]=study.cold.weights(file)
            models.append(model)
            hashes[model]=sha
    e.save(OUT/"locked.json",hashes)
    exp.OUT=OUT
    exp.WORKERS=WORKERS
    for panel,room_seed in [("fresh",FRESH_SEED),("diagnostic",study.TEST_SEED)]:
        fixtures={e.s.phase.input_hash(f):f for f in d.rooms(room_seed).values()}
        assert set(fixtures)==set(protocol[panel])
        study.ROOMS.update(fixtures)
        names=models if panel=="fresh" else [m for m in models if m.startswith(("s29_","s43_"))]
        jobs=[(panel,k,m) for k in protocol[panel] for m in names
              if not (OUT/panel/"test"/m/(k+".json")).exists()]
        exp.timed_pool(panel,evaluate_one,jobs)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("step",choices=["freeze","train","evaluate"])
    parser.add_argument("--seed",type=int,choices=(29,43))
    args=parser.parse_args()
    if args.step=="freeze":
        freeze()
    elif args.step=="train":
        assert args.seed is not None
        train(args.seed)
    else:
        evaluate()
