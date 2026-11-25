"""40k progress-event PPO from the event-supervised seed-17 actor."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import fcntl
import multiprocessing as mp
from pathlib import Path
import shutil
import time
import traceback

import numpy as np
import torch

from tools import cap_event_supervision as bc
from tools.cap_research_journal import event_curves

event, base, old = bc.event, bc.base, bc.RUN
OUT = old.parent / "cap_event_supervised_ppo_20261124_01"
MODE = "progress"
SEED = 17


def publish():
    data = event_curves(dict(run=OUT.name))
    base.save(OUT/"live-data.json", data)


def paired(rows, other, keys):
    d = np.array([rows[k]["success"] - other[k]["success"] for k in keys])
    samples = np.random.default_rng(17).choice(d, (4000, len(d))).mean(axis=1)
    return dict(gain=int((d>0).sum()), loss=int((d<0).sum()), pp=float(100*d.mean()),
                ci95=(100*np.quantile(samples,[.025,.975])).tolist())


def run():
    setup = base.setup()
    plan = base.read(old/"protocol.json")
    assert all(event.long.ppo.sha(bc.ROOT/path)==sha for path,sha in plan["sources"].items())
    assert setup["expert_sha"]==plan["experts"] and setup["profile"]==plan["profile"]
    rooms = base.get_rooms(bc.PANEL_SEED)
    keys = plan["confirmation"]
    source = bc.OUT/f"actor_{SEED}.npz"
    before = base.read(bc.OUT/f"panel_{SEED}.json")
    assert set(keys)==set(rooms)==set(before["rooms"]) and len(keys)==200
    assert before["policy_sha"]==event.long.ppo.sha(source)
    base.ROOMS.update(rooms)
    diagnostic = plan["diagnostic"]
    assert len(diagnostic)==64 and set(diagnostic)<=base.ROOMS.keys() and not set(diagnostic)&set(keys)
    excluded = {bc.s.phase.input_hash(f) for f in base.ROOMS.values()}
    manifest = dict(seed=SEED, workers=24, batch_size=50, rounds=800, arms=[MODE],
                    actor_sha=before["policy_sha"], critic="fresh, seed 17", optimizer="fresh Adam",
                    diagnostic=diagnostic, confirmation=keys, room_seeds=plan["room_seeds"], profile=setup["profile"],
                    executor=setup["executor"], experts=plan["experts"], parent_protocol=event.long.ppo.sha(old/"protocol.json"),
                    reward=plan["reward"], progress_m=plan["progress_m"], contact_ticks=plan["contact_ticks"],
                    source=event.long.ppo.sha(Path(__file__)))
    assert len(manifest["room_seeds"])==800 and event.long.SEED==SEED
    if (OUT/"protocol.json").exists():
        assert base.read(OUT/"protocol.json")==manifest
    else:
        base.save(OUT/"protocol.json",manifest)
    torch.manual_seed(SEED)
    model=event.Policy()
    with torch.no_grad():
        for param,values in zip(model.actor.parameters(),event.load_weights(source),strict=True):
            param.copy_(torch.from_numpy(values.T))
            assert np.array_equal(param.numpy().T,values)
    opt=torch.optim.Adam([dict(params=model.actor.parameters(),lr=3e-5),
                          dict(params=model.critic.parameters(),lr=3e-4)])
    start=event.actor_path(OUT,MODE,0)
    start.parent.mkdir(exist_ok=True)
    if not start.is_file():
        shutil.copyfile(source,start)
    assert event.long.ppo.sha(start)==before["policy_sha"]
    done=event.restore_latest(OUT,MODE,model,opt,800)
    with ProcessPoolExecutor(24,mp_context=mp.get_context("fork")) as pool:
        for i in range(801):
            actor=event.actor_path(OUT,MODE,i)
            if i:
                matched=base.read(event.metrics_path(old,MODE,i))["rooms"]
                assert len(set(matched))==len(matched)==50 and excluded.isdisjoint(matched)
                excluded.update(matched)
                if i>done:
                    started=time.monotonic()
                    fixtures=base.get_rooms(plan["room_seeds"][i-1])
                    stage=event.long.END+i
                    prev=event.actor_path(OUT,MODE,i-1)
                    jobs=[(key, fixtures[key], MODE, stage, j, 12, prev) for j,key in enumerate(matched)]
                    rows=list(pool.map(event.collect_row,jobs))
                    stats=event.long.dense.fit(model,opt,rows,stage)
                    assert np.isfinite(stats["old_logp_error"]) and stats["old_logp_error"]<.001
                    counts=Counter()
                    for row in rows:
                        counts.update(row[10])
                    event.save_training_round(OUT,MODE,i,model,opt,dict(stats,rooms=matched,reviews=dict(counts),
                                        wall_seconds=time.monotonic()-started))
                    print(i,stats["success"],stats["entropy"],flush=True)
                else:
                    assert base.read(event.metrics_path(OUT,MODE,i))["rooms"]==matched
            if i%20==0:
                event.evaluate(pool,OUT/"diagnostic"/MODE/f"r{i:04d}.json",MODE,actor,diagnostic,12)
            if i>=done:
                publish()
        event.evaluate(pool,OUT/"evaluation/progress.json",MODE,actor,keys,12)
    after=base.read(OUT/"evaluation/progress.json")
    prev=base.read(old/"evaluation/progress.json")
    singles=base.read(OUT.parent/"cap_single_progress_eval_20261124_01/evaluation.json")["auto"]
    strata={}
    for n in range(4):
        subset=[k for k in keys if sum(singles[e][k]["success"] for e in base.EXPERTS)==n]
        strata[n]=dict(rooms=len(subset),before=sum(before["rooms"][k]["success"] for k in subset),
                       after=sum(after["rooms"][k]["success"] for k in subset))
    base.save(OUT/"finished.json",dict(episodes=40000,actor_sha=after["policy_sha"],parent=before["success"],
                    final=after["success"],vs_parent=paired(after["rooms"],before["rooms"],keys),
                    vs_previous_ppo=paired(after["rooms"],prev["rooms"],keys),by_successful_experts=strata))
    publish()


if __name__=="__main__":
    torch.set_num_threads(1)
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/".driver.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            run()
        except Exception:
            base.save(OUT/"ERROR.json",dict(error=traceback.format_exc()))
            raise
