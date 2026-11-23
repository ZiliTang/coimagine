"""Matched prefix collectors, fixed continuation, and crossed router feature ablation."""
from collections import Counter
import csv
import hashlib
import json

import numpy as np
from tools import cap_cold_start as cold
from tools import cap_expanding_pool as exp

d, e, p = exp.d, exp.e, exp.p
OUT = exp.OUT.parent / "cap_fork_discovery_20261123_01"
DOC = exp.DOC.parent / "2026-11-23_fork_discovery"
COLLECTORS = ("expert_A", "distilled")
FEATURES = ("history", "semantic")
SIMPLE = "expert_A_own_suffix"
TRAIN_SEED, TEST_SEED = 662319821, 662319893
ROOMS = {}
WARM = None


def rollout(key, collector, intervention=None, *, continuation="distilled", stall_gate=True):
    hist = d.PublicHistory()
    digest = hashlib.sha256()
    points = []
    matched = False
    teacher_memory = {}
    if WARM is not None:
        d.MODELS["semantic"] = WARM

    def choose(unused, obs, a, b, ctx):
        nonlocal matched
        x = hist.vector(obs, a, b, ctx)
        n = ctx["n"]
        eligible = (ctx["triggered"] or not stall_gate) and a != b
        choice = "A"
        controller = continuation if matched else collector
        if controller == "rule_teacher":
            choice = d.TEACHER(teacher_memory,obs,a,b,ctx)
        elif eligible and controller == "expert_B":
            choice = "B"
        elif eligible and controller == "self" and d.logits(x, "history") > 0:
            choice = "B"
        elif eligible and controller == "distilled" and d.logits(x, "semantic") > 0:
            choice = "B"
        if intervention is None or not matched:
            digest.update(x.tobytes())
            digest.update(json.dumps([a,b], sort_keys=True).encode())
            if eligible and (intervention is None or n == intervention[0]):
                prefix = digest.copy()
                prefix.update(json.dumps(obs, sort_keys=True).encode())
                if intervention is None:
                    points.append((n, prefix.hexdigest(), x, int(choice == "B")))
                else:
                    assert prefix.hexdigest() == intervention[1]
                    assert np.array_equal(x, intervention[2])
                    choice = intervention[3]
                    matched = True
            digest.update(choice.encode())
        hist.executed(choice, a, b, ctx)
        return choice

    d.source.ROUTE = choose
    row = d.source.play(ROOMS[key]) if stall_gate else d.source.play(ROOMS[key], stall_gate=False)
    result = row["result"]
    if not stall_gate:
        result = dict(result, first_stall=row["hit"]["n"] if row["hit"] else None)
    assert intervention is None or matched
    return result, points


def collect_parent(job):
    key, collector = job
    root = OUT / collector / "parents"
    result, points = rollout(key, collector)
    exp.save_npz(root / (key + ".npz"),
                 x=np.asarray([q[2] for q in points], np.float32).reshape(-1,349),
                 b=np.asarray([q[3] for q in points], int))
    e.save(root / (key + ".json"), dict(success=result["success"], ticks=result["ticks"],
                                        points=[[q[0],q[1]] for q in points]))


def collect_rule_pair(job):
    key, index = job
    root = OUT / "rule_teacher"
    n, prefix = e.read(root / "parents" / (key + ".json"))["points"][index]
    with np.load(root / "parents" / (key + ".npz")) as arr:
        x = arr["x"][index]
    # A deterministic room, identical full command/choice prefix and current
    # observation lead to the same A-continuation pair, whichever collector found it.
    a_parent = e.read(OUT/"expert_A"/"parents"/(key+".json"))
    for ai, point in enumerate(a_parent["points"]):
        stored = OUT/SIMPLE/"forks"/f"{key}_{ai}.json"
        if point == [n,prefix] and stored.is_file():
            r = e.read(stored)
            assert r["n"] == n and r["prefix"] == prefix
            e.save(root/"forks"/f"{key}_{index}.json",dict(**r,reused=True))
            return
    observed = [rollout(key,"rule_teacher",(n,prefix,x,c),continuation="expert_A")[0] for c in ("A","B")]
    e.save(root/"forks"/f"{key}_{index}.json",dict(outcomes=[int(r["success"]) for r in observed],
                                                   ticks=[r["ticks"] for r in observed],n=n,prefix=prefix))


def collect_pair(job):
    key, collector, index = job
    root = OUT / collector
    parent = e.read(root / "parents" / (key + ".json"))
    n, prefix = parent["points"][index]
    with np.load(root / "parents" / (key + ".npz")) as arr:
        x = arr["x"][index]
        original = int(arr["b"][index])
    outcomes, ticks = [], []
    for choice in ("A","B"):
        r, _ = rollout(key, collector, (n, prefix, x, choice))
        outcomes.append(int(r["success"]))
        ticks.append(r["ticks"])
    if collector == "distilled":
        assert outcomes[original] == parent["success"] and ticks[original] == parent["ticks"]
    e.save(root / "forks" / f"{key}_{index}.json",
           dict(outcomes=outcomes, ticks=ticks, n=n, prefix=prefix))


def simple_pair(job):
    key, index = job
    root = OUT / "expert_A" / "parents"
    parent = e.read(root / (key + ".json"))
    n, prefix = parent["points"][index]
    with np.load(root / (key + ".npz")) as arr:
        x = arr["x"][index]
        assert arr["b"][index] == 0
    # A followed by A exactly reproduces the saved A-only parent episode.
    alt, _ = rollout(key, "expert_A", (n,prefix,x,"B"), continuation="expert_A")
    e.save(OUT / SIMPLE / "forks" / f"{key}_{index}.json",
           dict(outcomes=[int(parent["success"]),int(alt["success"])],
                ticks=[parent["ticks"],alt["ticks"]], n=n, prefix=prefix))


def train(keys, selections, sources=(*COLLECTORS,SIMPLE)):
    for source in sources:
        collector = "expert_A" if source in (SIMPLE,"matched_A") else source
        fork_dir = SIMPLE if source == "matched_A" else source
        for mode in (FEATURES if source in COLLECTORS else ("history",)):
            p.OUT = OUT / source / mode
            p.OUT.mkdir(exist_ok=True, parents=True)
            d.MODELS["semantic"] = cold.weights(cold.OUT / "cold0.npz")
            for stage, block in enumerate(np.array_split(keys,4),1):
                anchors, xs, ys = [], [], []
                n_pairs = 0
                for key in block:
                    path = OUT / collector / "parents" / (key + ".npz")
                    with np.load(path) as z:
                        x = d.project(z["x"], mode)
                    anchors.append(x)
                    for i in selections[key][collector]:
                        a, b = e.read(OUT / fork_dir / "forks" / f"{key}_{i}.json")["outcomes"]
                        n_pairs += 1
                        if a != b:
                            xs.append(x[i])
                            ys.append(b)
                totals = dict(rooms=len(block), forks=n_pairs, ties=n_pairs-len(ys))
                p.fit_preferences(stage,np.concatenate(anchors),np.asarray(xs),
                                  np.asarray(ys,np.float32),totals)


def evaluate(job):
    key, model = job
    if model == "rule0":
        d.source.ROUTE = d.TEACHER
        r = d.source.play(ROOMS[key])["result"]
        success, ticks = r["success"],r["ticks"]
    elif model in ("A","B"):
        policy = e.s.phase.FrozenCAP({"A":e.AM, "B":e.BM}[model])
        with e.s.phase.world(ROOMS[key]) as (w,_):
            policy.update(w.history)
            counts = Counter()
            while not e.s.ended(w):
                cmd = policy.propose(e.s.observe(w))
                e.s.tick_action(w,cmd,e.g.prep.prior.EXECUTOR,policy.update,counts)
            success, ticks = int(w.env.success), w.env.total_periods
    else:
        if model == "distilled0":
            arm, weights = "semantic", WARM
        else:
            collector, arm = model.split("/")
            weights = cold.weights(OUT / collector / arm / "actor4.npz")
        d.MODELS[arm] = weights
        r = d.episode(ROOMS[key],arm,label=False)[0]
        success, ticks = r["success"], r["ticks"]
    e.save(OUT / "test" / model / (key + ".json"), dict(success=success, ticks=ticks))


def run():
    global WARM
    exp.setup()
    d.load("semantic")
    WARM = d.MODELS["semantic"]
    known = set(exp.ROOMS) | e.s.collect_identities()
    for group in e.read(d.OUT/"protocol.json")["rooms"]:
        known.update(group.values())
    for path in (exp.trend.OUT/"confirmation_ids.json", exp.p.OUT/"fresh_ids.json",
                 d.source.OUT/"confirmation_ids.json", e.OUT/"confirmation_ids.json",e.old.OUT/"fresh_panel.json"):
        known.update(e.read(path).values())
    panels = []
    for seed in (TRAIN_SEED,TEST_SEED):
        fixtures = {e.s.phase.input_hash(f):f for f in d.rooms(seed).values()}
        assert len(fixtures)==200 and not (set(fixtures) & (known | set(ROOMS)))
        ROOMS.update(fixtures)
        panels.append(exp.shuffled(fixtures,str(seed)))
    train_keys, held = panels
    e.save(OUT/"protocol.json",dict(train=train_keys, held=held, collectors=COLLECTORS,
           modes=FEATURES, suffix="fixed original distilled actor for both first commands; repeat A prefixes with A continuation",
           features="history zeros the last nine teacher-inspired features during fit and inference",
           parent=d.source.graph.file_sha(d.OUT/"fit/semantic.npz"),
           initial=d.source.graph.file_sha(cold.OUT/"cold0.npz"), seed=17,
           budget="same 200 unfiltered rooms, at most three paired forks each, min available across collectors per room, temporally stratified; 4 x 300 updates"))
    exp.OUT=OUT   # route live progress to this experiment
    jobs=[(k,c) for k in train_keys for c in COLLECTORS if not (OUT/c/"parents"/(k+".json")).exists()]
    exp.timed_pool("parents",collect_parent,jobs)
    selections={}
    jobs=[]
    for k in train_keys:
        parents={c:e.read(OUT/c/"parents"/(k+".json"))["points"] for c in COLLECTORS}
        size=min(3,*(len(q) for q in parents.values()))
        selections[k]={}
        for c, points in parents.items():
            picks=[exp.shuffled(list(map(int,g)), f"{k}/{j}")[0]
                   for j,g in enumerate(np.array_split(np.arange(len(points)),size))] if size else []
            selections[k][c]=picks
            jobs += [(k,c,i) for i in picks if not (OUT/c/"forks"/f"{k}_{i}.json").exists()]
    e.save(OUT/"selections.json",selections)
    exp.timed_pool("pairs",collect_pair,jobs)
    simple_jobs = [(k,i) for k in train_keys for i in selections[k]["expert_A"]
                   if not (OUT/SIMPLE/"forks"/f"{k}_{i}.json").exists()]
    exp.timed_pool("simple_pairs",simple_pair,simple_jobs)
    train(train_keys,selections)
    models=["A","B","distilled0",*(c+"/"+m for c in COLLECTORS for m in FEATURES),SIMPLE+"/history"]
    e.save(OUT/"locked.json",{name:d.source.graph.file_sha(OUT/name/"actor4.npz") for name in models[3:]})
    exp.timed_pool("evaluation",evaluate,[(k,m) for k in held for m in models if not (OUT/"test"/m/(k+".json")).exists()])
    report(held,models,selections)


def report(held,models,selections):
    DOC.mkdir(parents=True,exist_ok=True)
    with (DOC/"rooms.csv").open("w") as f:
        writer=csv.writer(f)
        writer.writerow(["room",*models])
        rows=[]
        for k in held:
            row=[e.read(OUT/"test"/m/(k+".json"))["success"] for m in models]
            rows.append(row)
            writer.writerow([k,*row])
    values=np.array(rows)
    counts=dict(zip(models,values.sum(0).astype(int).tolist()))
    pairs={}
    for i in range(len(models)):
        for j in range(i):
            delta=values[:,i]-values[:,j]
            interval=np.percentile(np.random.default_rng(17).choice(delta,(10000,len(held))).mean(1)*100,[2.5,97.5])
            pairs[f"{models[i]} vs {models[j]}"]=dict(wins=int(sum(delta>0)),losses=int(sum(delta<0)),ci_pp=interval.tolist())
    sources={}
    for c in (*COLLECTORS,SIMPLE):
        collector = "expert_A" if c == SIMPLE else c
        pairs_by_room=[[e.read(OUT/c/"forks"/f"{k}_{i}.json")["outcomes"] for i in s[collector]] for k,s in selections.items()]
        observed=Counter(tuple(v) for rr in pairs_by_room for v in rr)
        sources[c]=dict(forks=sum(observed.values()),A_wins=observed[1,0],B_wins=observed[0,1],both=observed[1,1],neither=observed[0,0],
                        informative_rooms=sum(any(a!=b for a,b in rr) for rr in pairs_by_room))
    e.save(DOC/"results.json",dict(success=counts,pairs=pairs,sources=sources))
    e.save(OUT/"FINISHED.json",dict(success=counts,sources=sources))


if __name__=="__main__":
    run()
