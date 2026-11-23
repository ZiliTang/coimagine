"""Frozen A/B at first disagreement, public successor nodes and a five-command handoff."""
import hashlib
import json
from pathlib import Path

import numpy as np
from tools import cap_ungated_no_replay as current

r=current.run
OUT=r.OUT.parent/"cap_first_choice_transfer_20261123_01"
DOC=r.DOC.parent/"2026-11-23_first_choice_transfer"
WINDOWS=(0,1,5,10,30)
ACTORS={}


def episode(job, *, output=None):
    panel,key,route=job
    path=(OUT if output is None else output)/panel/route/(key+".json")
    hist=r.d.PublicHistory()
    first=None
    frames={}
    proposals={}
    prefix=hashlib.sha256()
    start=None
    def choose(_,obs,a,b,c):
        nonlocal first,start
        x=hist.vector(obs,a,b,c)
        if first is None:
            prefix.update(json.dumps([obs,a,b],sort_keys=True).encode())
            if a!=b:
                first=c["n"]
                start=dict(n=first,tick=c["tick"],prefix=prefix.hexdigest(),z=c["z"])
                if route=="A":
                    proposals["x"]=r.d.project(x,"history").tolist()
                    for seed,w in ACTORS.items():
                        r.d.MODELS["history"]=w
                        proposals[str(seed)]=float(r.d.logits(x,"history"))
        who="A"
        if first is not None:
            k=c["n"]-first
            if k in WINDOWS:
                frames[str(k)]=dict(z=c["z"],tick=c["tick"],gap=c["gap"],touch=c["touch"])
            who=route[0] if k<5 else route[-1]
        if a==b:
            who="A"
        hist.executed(who,a,b,c)
        return who
    r.d.source.ROUTE=choose
    row=r.d.source.play(r.study.ROOMS[key],stall_gate=False)["result"]
    if panel=="held" and len(route)==1:
        ref=r.e.read(r.OUT/"fresh"/route/(key+".json"))
        assert all(row[k]==ref[k] for k in ("success","history","ticks"))
    r.e.save(path,dict(start=start,frames=frames,proposals=proposals,
                       success=row["success"],ticks=row["ticks"],history=row["history"]))


def run():
    base=r.e.read(r.OUT/"protocol.json")
    assert r.prior.initialize()==base["profile"]
    for seed in r.SEEDS:
        path=current.OUT/"models"/f"s{seed}_no_replay.npz"
        assert r.d.source.graph.file_sha(path)==r.e.read(current.OUT/"locked.json")[f"s{seed}_no_replay"]
        ACTORS[seed]=r.study.cold.weights(path)
    dev=r.prior.fixtures(r.prior.TRAIN_SEED)
    test=r.prior.fixtures(r.FRESH_SEED)
    assert not set(dev)&set(test) and sorted(test)==base["fresh"]
    r.study.ROOMS.update(dev)
    r.study.ROOMS.update(test)
    plan=dict(dev=sorted(dev),held=sorted(test),windows=WINDOWS,handoff=5,
              primary="locked no-replay actors' first argmax then commit; compare with always B in 52 known exclusive wins",
              linear="separate first-decision ridge on 200 development rooms with a fork; target B minus A terminal success; standardized visible inputs, penalty n, B if prediction positive",
              later="diagnostic paired public gap: smaller goal distance favors branch; both alive and gap visible at compared time; compare initial actor AUC on matched cohort",
              edge="5-command public 48-node successors plus terminal codes; fit source-conditional pooled and arm-specific histograms with 8 pseudo-observations from pooled; develop top non-self nonterminal directional edge with >=12 source rooms; test held",
              routes="paired A/B, A for 5 then B and vice versa from first disagreement; evaluated terminally even when no switch possible",
              selection="existing viewed 200 room panel. No held-dependent tuning")
    r.e.save(OUT/"protocol.json",plan)
    r.exp.OUT=OUT
    r.exp.WORKERS=r.prior.first.WORKERS
    for panel,rooms,routes in (("dev",dev,("A","B")),("held",test,("A","B","AB","BA"))):
        jobs=[(panel,k,v) for k in sorted(rooms) for v in routes if not (OUT/panel/v/(k+".json")).exists()]
        r.exp.timed_pool(panel,episode,jobs)


if __name__=="__main__":
    run()
