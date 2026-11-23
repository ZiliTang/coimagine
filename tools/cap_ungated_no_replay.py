"""Matched one-round ablation of successful actor replay under ungated routing."""
from functools import partial
import numpy as np

from tools import cap_ungated_replay as run
from tools.cap_ungated_fit import fit, load_rows

OUT=run.OUT.parent/"cap_ungated_no_replay_20261123_01"
DOC=run.DOC.parent/"2026-11-23_ungated_no_replay"


def main():
    plan=run.freeze()
    prior_lock=run.e.read(run.OUT/"locked.json")
    protocol=dict(source_protocol=run.d.source.graph.file_sha(run.OUT/"protocol.json"),
                  source_models=prior_lock,seeds=list(run.SEEDS),train_rooms=len(plan["train"]),
                  diagnostic_rooms=len(plan["fresh"]),primary="no replay minus replay on shared previously viewed 200 rooms",
                  controls="same parent, forks, ungated rollout, 64 strict + 128 soft per step, 300 Adam steps; remove successful-route hinge only")
    if (OUT/"protocol.json").exists():
        assert run.e.read(OUT/"protocol.json")==protocol
    run.e.save(OUT/"protocol.json",protocol)
    locks={}
    for seed in run.SEEDS:
        reference=f"s{seed}_actor"
        path=run.OUT/"models"/(reference+".npz")
        assert run.d.source.graph.file_sha(path)==prior_lock[reference]
        reproduced=fit(seed,"actor",plan,output=OUT/"control_refit")
        assert all(np.array_equal(a,b) for a,b in zip(reproduced,run.study.cold.weights(path),strict=True))
        name=f"s{seed}_no_replay"
        weights=fit(seed,plan=plan,output=OUT)
        run.WEIGHTS[name]=weights
        locks[name]=run.d.source.graph.file_sha(OUT/"models"/(name+".npz"))
        x,labels,_,ids,y,routes,_=load_rows(seed,"actor",plan)
        success=np.concatenate(routes["own"])
        audit={}
        for group,params in [("replay",reproduced),("no_replay",weights)]:
            run.d.MODELS["history"]=params
            b=run.d.logits(x,"history")>0
            audit[group]=dict(changed=int((b[success]!=labels[success]).sum()),n=len(success),
                              hard_correct=int((b[ids]==y).sum()),hard_n=len(ids))
        run.e.save(OUT/"audit"/f"s{seed}.json",audit)
    run.e.save(OUT/"locked.json",locks)
    run.exp.OUT=OUT
    for panel in ("fresh","train"):
        jobs=[(panel,k,n) for k in plan[panel] for n in locks if not (OUT/panel/n/(k+".json")).exists()]
        run.exp.timed_pool(panel,partial(run.play,output=OUT),jobs)


if __name__=="__main__":
    main()
