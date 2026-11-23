"""Remove the soft term and pair 64 vs nested 192 draws from successful routes."""
import csv
import torch
from tools import cap_success_replay as old
from tools.cap_success_replay_gradients import GradientMeter,aggregate

BASE=old.OUT
OUT=BASE.parent/"cap_margin_batch_20261123_01"
DOC=old.DOC.parent/"2026-11-23_margin_batch"
SEEDS=old.SEEDS
ARMS={"only64":64,"only192":192}
e=old.e


def main():
    torch.set_num_threads(1)
    source=e.read(BASE/"protocol.json")
    locks=e.read(BASE/"locked.json")
    plan=dict(seeds=list(SEEDS),train=source["train"],diagnostic=source["fresh"],models=ARMS,
              baseline="locked soft128 plus success64 on previously inspected panel",
              primary="only192 vs success; fixed 64 isolate removal; only192 vs only64 isolates batch size",
              weight=old.WEIGHT,margin=old.MARGIN,updates=300,fork_batch=64,
              sampling="with replacement; nested first 64 same in both arms and same as baseline; same paired preference minibatches")
    if (OUT/"protocol.json").exists():
        assert e.read(OUT/"protocol.json")==plan
    else:
        e.save(OUT/"protocol.json",plan)
    profile=old.prior.initialize()
    old.OUT=OUT
    DOC.mkdir(exist_ok=True,parents=True)
    gradients={}
    gradrows=[]
    for seed in SEEDS:
        assert e.read(old.prior.OUT/f"s{seed}"/"self"/"start.json")["profile"]==profile
        assert old.d.source.graph.file_sha(old.prior.checkpoint(seed,4))==source["parent"][str(seed)]
        data=old.training_data(seed,source)
        reference=BASE/"models"/f"s{seed}_success.npz"
        assert old.d.source.graph.file_sha(reference)==locks[f"s{seed}_success"]
        assert data[-1]==e.read(BASE/"fit"/f"s{seed}_success.json")["data_hash"]
        for arm,size in ARMS.items():
            old.fit(seed,arm,source,data,soft_enabled=False,margin_batch=size)
            meter=GradientMeter()
            old.fit(seed,arm,source,data,meter=meter,soft_enabled=False,margin_batch=size)
            name=f"s{seed}_{arm}"
            assert all(r["soft_norm"]==0 and r["soft_adam_share"]==0 for r in meter.rows)
            gradients[name]=aggregate(meter.rows)
            gradrows.extend(dict(seed=seed,arm=arm,**r) for r in meter.rows)
    for arm in ARMS:
        gradients[arm]={p:aggregate([r for r in gradrows if r["arm"]==arm and lo<=r["step"]<=hi])
                        for p,lo,hi in [("early",1,50),("all",1,300),("late",251,300)]}
    with (DOC/"gradients.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(gradrows[0]))
        w.writeheader()
        w.writerows(gradrows)
    e.save(DOC/"gradient_summary.json",gradients)
    e.save(OUT/"locked.json",{n:e.read(OUT/"fit"/(n+".json"))["sha"] for n in old.MODELS})
    old.exp.OUT=OUT
    old.exp.WORKERS=old.prior.first.WORKERS
    for panel,room_seed in [("train",old.prior.TRAIN_SEED),("fresh",old.NEW_ROOM_SEED)]:
        rooms=old.prior.fixtures(room_seed)
        assert set(source[panel])<=set(rooms)
        old.study.ROOMS.update(rooms)
        jobs=[(panel,k,m) for k in source[panel] for m in old.MODELS if not (OUT/panel/m/(k+".json")).exists()]
        old.exp.timed_pool(panel,old.score,jobs)


if __name__=="__main__":
    main()
