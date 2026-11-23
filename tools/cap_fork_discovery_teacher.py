"""Direct authored-router prefix discovery with expert A as the common continuation."""
import sys
import numpy as np
from tools import cap_fork_discovery as study

exp, d, e, out = study.exp, study.d, study.e, study.OUT


def main(stage):
    protocol=e.read(out/"protocol.json")
    keys,held=protocol["train"],protocol["held"]
    d.init()
    d.load("semantic")
    study.WARM=d.MODELS["semantic"]
    for seed,expected in [(study.TRAIN_SEED,keys),(study.TEST_SEED,held)]:
        data={e.s.phase.input_hash(f):f for f in d.rooms(seed).values()}
        assert set(data)==set(expected)
        study.ROOMS.update(data)
    exp.OUT=out/"rule_progress"
    if stage=="parents":
        jobs=[(k,"rule_teacher") for k in keys if not (out/"rule_teacher"/"parents"/(k+".json")).exists()]
        exp.WORKERS=4
        exp.timed_pool("parents",study.collect_parent,jobs)
        return
    base=e.read(out/"selections.json")
    selection={}
    for k in keys:
        points=e.read(out/"rule_teacher"/"parents"/(k+".json"))["points"]
        available=base[k]["expert_A"]
        n=min(len(available),len(points))
        # Keep all already selected A prefixes when quotas match. Thin identically
        # across temporal strata if the teacher visits fewer eligible boundaries.
        store= lambda count: [exp.shuffled(list(map(int,g)), f"{k}/{j}")[0]
                              for j,g in enumerate(np.array_split(np.arange(count),n))] if n else []
        chosen=available if len(available)==n else [available[i] for i in store(len(available))]
        selection[k]=dict(expert_A=chosen,rule_teacher=store(len(points)))
    e.save(out/"rule_selection.json",selection)
    jobs=[(k,i) for k in keys for i in selection[k]["rule_teacher"]
          if not (out/"rule_teacher"/"forks"/f"{k}_{i}.json").exists()]
    exp.timed_pool("rule_pairs",study.collect_rule_pair,jobs)
    sources=["rule_teacher"]
    if any(base[k]["expert_A"]!=selection[k]["expert_A"] for k in keys):
        sources.append("matched_A")
    study.train(keys,selection,sources)
    models=[s+"/history" for s in sources]+["rule0"]
    e.save(out/"rule_locked.json",{m:d.source.graph.file_sha(out/m/"actor4.npz") for m in models[:-1]})
    exp.timed_pool("rule_eval",study.evaluate,[(k,m) for k in held for m in models if not (out/"test"/m/(k+".json")).exists()])
    e.save(out/"rule_done.json",dict(models=models,paired_forks=sum(len(row["expert_A"]) for row in selection.values())))


if __name__=="__main__":
    main(sys.argv[1])
