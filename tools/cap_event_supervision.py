"""Imitate successful progress-triggered experts, with original-only keep demonstrations."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from tools import cap_abc_event_ppo as event
from tools.cap_abc_event_routes import RUN, PANEL_SEED, ROOT

base, s = event.base, event.s
OUT = RUN.parent / "cap_event_supervision_20261124_01"


def trace(key, expert, keep):
    members = [s.phase.FrozenCAP(base.third.MEMBERS[a]) for a in base.EXPERTS]
    slot = base.EXPERTS.index(expert)
    history = deque(maxlen=4)
    counts = Counter()
    rows, labels, ticks = [], [], []

    def update(events):
        for member in members:
            member.update(events)

    def teach(obs, proposals, index, ongoing=None):
        panel = event.candidate_rows(base.public_frame(obs), history, obs, proposals, ongoing)
        assert panel[index, -1] and panel[:, -1].sum() > 1
        rows.append(panel)
        labels.append(index)
        ticks.append(obs["tick"])

    def review(world, obs, action, left, reason):
        elapsed = obs["tick"] - s.observe(world)["tick"]
        proposals = [event.preview(m, obs, action, elapsed) for m in members]
        ongoing = dict(action, ticks=left)
        if not any(p is not None and not event.same_control(p, ongoing) for p in proposals):
            return None
        own = proposals[slot]
        if keep or own is None or event.same_control(own, ongoing):
            teach(obs, proposals, event.CONTINUE, ongoing)
            return None
        teach(obs, proposals, proposals.index(own), ongoing)
        return own

    with s.phase.world(base.ROOMS[key]) as (world, _):
        update(world.history)
        trigger = event.PushReview("progress", review, progress_m=.20, contact_ticks=3)
        pending = None
        while not s.ended(world):
            obs = s.observe(world)
            if pending is None:
                proposals = [m.propose(obs) for m in members]
                action = proposals[slot]
                if any(p != action for p in proposals):
                    teach(obs, proposals, proposals.index(action))
            else:
                action = pending
            history.append(dict(base.public_frame(obs), action=action))
            pending = s.tick_action(world, action, base.third.EXECUTOR, update, counts, review=trigger)
        result = s.output(world, counts)
    if result['success'] and rows:
        file = OUT / "demonstrations" / expert / (key+".npz")
        file.parent.mkdir(exist_ok=True, parents=True)
        np.savez_compressed(file, x=np.stack(rows), y=labels, tick=ticks)
    result.update(rows=len(rows), keep_labels=labels.count(event.CONTINUE))
    return result


def collect(job):
    key, expert = job
    old = base.read(base.OUT / "demonstration" / expert / (key+".json"))
    auto = trace(key, expert, False)
    keep = None
    if old["success"] and not auto["success"]:
        keep = trace(key, expert, True)
        assert all(keep[f] == old[f] for f in ("success", "ticks", "history"))
    return dict(key=key, expert=expert, original=old["success"], auto=auto, keep=keep)


def train(records, seed):
    xs, ys, weights = [], [], []
    by_room = Counter(r["key"] for r in records if r["selected"] and r["samples"])
    for row in records:
        if not row["selected"] or not row["samples"]:
            continue
        with np.load(OUT/"demonstrations"/row["expert"]/(row["key"]+".npz")) as data:
            xs.append(data["x"])
            ys.append(data["y"])
            weights.extend([1/(by_room[row["key"]]*len(data["y"]))]*len(data["y"]))
    x = torch.from_numpy(np.concatenate(xs))
    y = torch.from_numpy(np.concatenate(ys))
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = event.Policy()
    opt = torch.optim.Adam(model.actor.parameters(), lr=.001)
    for _ in range(base.WARM_UPDATES):
        index = np.searchsorted(cdf, rng.random(128))
        dist, _ = model.distribution(x[index])
        loss = -dist.log_prob(y[index]).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = torch.cat([base_batch.probs.argmax(1) for base_batch, _ in
                          (model.distribution(part) for part in x.split(1024))])
    correct = pred == y
    masks = dict(all=torch.ones_like(correct), boundary=x[:,3,-1]==0, review=x[:,3,-1]>0,
                 continue_label=y==event.CONTINUE)
    event.export(OUT/f"actor_{seed}.npz", model)
    return {k:dict(correct=int(correct[m].sum()), total=int(m.sum())) for k,m in masks.items()}


def main():
    torch.set_num_threads(1)
    prev = base.read(RUN/"protocol.json")
    assert all(event.long.ppo.sha(ROOT/src)==sha for src,sha in prev["sources"].items())
    setup = base.setup()
    assert setup["expert_sha"] == prev["experts"]
    dev = setup["demonstration"]
    panel = base.get_rooms(PANEL_SEED)
    keys = prev["confirmation"]
    assert len(dev)==len(panel)==len(keys)==200 and set(panel)==set(keys)
    assert not set(panel).intersection(s.phase.input_hash(base.ROOMS[k]) for k in dev)
    base.ROOMS.update(panel)
    protocol = dict(development=dev, evaluation=keys, panel_seed=PANEL_SEED, seeds=base.SEEDS,
                    updates=base.WARM_UPDATES,batch=128,lr=.001,expert_sha=setup["expert_sha"],
                    source=event.long.ppo.sha(Path(__file__)),event_source=prev["sources"],
                    trigger_m=.20,contact_ticks=3,rule="auto success else original success keep")
    OUT.mkdir(exist_ok=True,parents=True)
    base.save(OUT/"protocol.json",protocol)
    with ProcessPoolExecutor(24,mp_context=mp.get_context("fork")) as pool:
        records=[]
        for r in pool.map(collect,[(k,e) for k in dev for e in base.EXPERTS]):
            demo=r["keep"] or r["auto"]
            r.update(selected=("keep" if r["keep"] else "auto") if demo["success"] else None,
                     samples=demo["rows"] if demo["success"] else 0)
            records.append(r)
            if len(records)%60==0:
                print("development",len(records),flush=True)
        base.save(OUT/"development.json",records)
        for seed in base.SEEDS:
            base.save(OUT/f"fit_{seed}.json",train(records,seed))
            for split, rooms in (("dev",dev),("panel",keys)):
                event.evaluate(pool,OUT/f"{split}_{seed}.json","progress",OUT/f"actor_{seed}.npz",rooms,12)
                print(seed,split,base.read(OUT/f"{split}_{seed}.json")["success"],flush=True)


if __name__ == "__main__":
    main()
