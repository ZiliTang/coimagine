"""Matched online policy iteration and pooled frozen-policy rollout classification."""

import argparse
import fcntl
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F

from tools import cap_regional_cpp as cpp
from tools import cap_regional_id_ppo as ppo

study = ppo.study
OUT = ppo.OUT.parent / "cap_regional_actor_iteration_20261120_01"
DOCS = ppo.DOCS.parent / "2026-11-20_regional_actor_iteration"
INITIAL = ppo.base.OUT / "fixed.pt"
ROUNDS, ROOMS, MONITOR, UPDATES = 24, 48, 48, 24
SEED = 7222000
ACTOR = None
STATUS = {}


def publish(**fields):
    STATUS.update(fields)
    study.save(OUT / "progress.json", STATUS)


class Paused(Exception):
    pass


def check_pause():
    if (OUT / "PAUSE").exists():
        raise Paused()


def actor_load(path):
    actor = ppo.base.ExpertActor()
    actor.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return actor.eval()


def panel(seed, count):
    file = OUT / "panels" / f"{seed}.json"
    if not file.exists():
        generated = study.ext.old.rooms.generate(seed)["cases"]
        rows = []
        for n in range(2, 6):
            group = [r for r in generated if r["movable_count"] == n]
            rows.extend(group[: count // 4])
        assert len(rows) == count
        study.save(
            file,
            [
                dict(id=f"{seed}_{r['id']}", objects=r["movable_count"], fixture=r["fixture"])
                for r in rows
            ],
        )
    return study.read(file)


@torch.inference_mode()
def rollout(fixture, *, sample=False, force=None):
    env = ppo.environment(fixture)
    lib = cpp.Proposals()
    eligible = []
    changed = False
    try:
        for step in range(300):
            x, _, sources, _, pool, raw = lib.panel(env.history)
            logits = ACTOR(torch.from_numpy(x[None]))[0].numpy()
            i = int(logits.argmax())
            action = raw[i] if np.isfinite(logits[i]) else pool[0]
            if sample and len(pool) > 1:
                a = pool.index(action)
                order = sorted(
                    (j for j in range(len(pool)) if j != a),
                    key=lambda j: -max(logits[sources[j].astype(bool)]),
                )
                eligible.append(
                    dict(
                        step=step,
                        prefix=env.history.sha256,
                        x=x,
                        base=action,
                        base_mask=sources[a].astype(bool),
                        alternatives=[(pool[j], sources[j].astype(bool)) for j in order],
                    )
                )
            if force is not None and step == force["step"]:
                assert action == force["base"] and env.history.sha256 == force["prefix"]
                assert force["alternative"] in pool and force["alternative"] != action
                action = force["alternative"]
                changed = True
            env.step(action, remaining_ticks=1800 - env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods >= 1800:
                break
        assert force is None or changed
        return dict(
            success=int(env.env.success),
            history=env.history.sha256,
            ticks=env.env.total_periods,
            steps=step + 1,
        ), eligible
    finally:
        env.close_reads()
        study.dq.fast_runtime.ACTIVE = None


def slots(points, seed):
    if not points:
        return [None, None]
    rng = np.random.default_rng(seed)
    mid = (len(points) + 1) // 2
    early = int(rng.integers(mid))
    late = int(rng.integers(mid, len(points))) if mid < len(points) else early
    best, explore = (early, late) if rng.integers(2) else (late, early)
    selected = []
    for index, other in [(best, False), (explore, True)]:
        p = points[index]
        n = len(p["alternatives"])
        if other and best == explore and n == 1:
            selected.append(None)
            continue
        j = int(rng.integers(1, n)) if other and n > 1 else 0
        action, mask = p["alternatives"][j]
        selected.append(
            dict(
                step=p["step"],
                prefix=p["prefix"],
                base=p["base"],
                alternative=action,
                x=p["x"],
                base_mask=p["base_mask"],
                alt_mask=mask,
                rank=j + 1,
                explored=other,
                alternatives=n,
            )
        )
    return selected


def job(args):
    room, directory, training, seed = args
    cid = room["id"]
    path = directory / (cid + ".json")
    if path.exists():
        return study.read(path)
    if (OUT / "PAUSE").exists():
        return None
    base, points = rollout(room["fixture"], sample=training)
    record = dict(id=cid, objects=room["objects"], baseline=base)
    if training:
        rows = []
        xs = []
        bm = []
        am = []
        for fork in slots(points, seed):
            if fork is None:
                control, _ = rollout(room["fixture"])
                assert control == base
                continue
            xs.append(fork.pop("x"))
            bm.append(fork.pop("base_mask"))
            am.append(fork.pop("alt_mask"))
            end, _ = rollout(room["fixture"], force=fork)
            rows.append(dict(**fork, delta=end["success"] - base["success"], end=end))
        if rows:
            shard = path.with_suffix(".npz")
            tmp = shard.with_suffix(".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, x=np.stack(xs), a=np.stack(bm), b=np.stack(am))
            tmp.replace(shard)
        record.update(branches=rows, controls=2 - len(rows), eligible=len(points))
    study.save(path, record)
    return record


def batch(rooms, directory, checkpoint, training, phase, round_id):
    global ACTOR
    check_pause()
    ACTOR = actor_load(checkpoint)
    directory.mkdir(parents=True, exist_ok=True)
    args = [(r, directory, training, SEED + round_id * 4096 + i) for i, r in enumerate(rooms)]
    rows = []
    live = dict(done=0, total=len(args), rescues=0, losses=0, equal=0, controls=0, success=0)
    publish(stage=phase, round=round_id, live=live)
    with ProcessPoolExecutor(16, mp_context=mp.get_context("fork")) as executor:
        for future in as_completed([executor.submit(job, a) for a in args]):
            row = future.result()
            if row is None:
                continue
            rows.append(row)
            labels = [r["delta"] for r in row.get("branches", [])]
            live["rescues"] += labels.count(1)
            live["losses"] += labels.count(-1)
            live["equal"] += labels.count(0)
            live["controls"] += row.get("controls", 0)
            live["success"] += row["baseline"]["success"]
            live["done"] += 1
            publish(live=live)
    check_pause()
    assert len(rows) == len(args)
    return sorted(rows, key=lambda r: r["id"])


def datasets(dirs):
    parts = {k: [] for k in ("x", "a", "b", "d")}
    for directory in dirs:
        for path in sorted(directory.glob("*.json")):
            branches = study.read(path)["branches"]
            if not branches:
                continue
            with np.load(path.with_suffix(".npz")) as f:
                assert len(branches) == len(f["x"])
                for k in ("x", "a", "b"):
                    parts[k].append(f[k])
            parts["d"].append(np.array([b["delta"] for b in branches], np.float32))
    if not parts["x"]:
        return None
    return {k: torch.from_numpy(np.concatenate(v)) for k, v in parts.items()}


def group_margin(scores, a, b):
    return scores.masked_fill(~b, -torch.inf).amax(1) - scores.masked_fill(~a, -torch.inf).amax(1)


def fit(dirs, parent, directory, updates, seed):
    directory.mkdir(parents=True, exist_ok=True)
    mark = directory / "fit.json"
    if mark.exists():
        return study.read(mark)
    check_pause()
    data = datasets(dirs)
    model = actor_load(parent)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
    if parent != INITIAL and (parent.parent / "adam.pt").exists():
        optimizer.load_state_dict(
            torch.load(parent.parent / "adam.pt", map_location="cpu", weights_only=True)
        )
    rng = np.random.default_rng(seed)
    log = []
    initial = actor_load(INITIAL)
    signed = np.array([], int) if data is None else np.flatnonzero(data["d"].numpy())

    def accuracy():
        if not len(signed):
            return None
        with torch.no_grad():
            hits = 0
            for ix in np.array_split(signed, max(1, (len(signed) + 127) // 128)):
                score = model(data["x"][ix])
                hits += int(
                    (group_margin(score, data["a"][ix], data["b"][ix]) * data["d"][ix] > 0).sum()
                )
            return hits / len(signed)

    before = accuracy()
    if len(signed):
        # Reference logits on all sampled decision boundaries (including tied outcomes).
        with torch.no_grad():
            anchor = torch.cat([initial(x) for x in data["x"].split(128)])
        for i in range(updates):
            check_pause()
            ix = rng.choice(signed, 32)
            keep = rng.integers(len(data["d"]), size=32)
            score = model(data["x"][ix])
            diff = group_margin(score, data["a"][ix], data["b"][ix])
            rank = F.softplus(-data["d"][ix] * diff / 4.0).mean()
            # Original actor remains the same retention reference for both arms.
            new = model(data["x"][keep]).masked_fill(~torch.isfinite(anchor[keep]), -1e8) / 4.0
            old = anchor[keep].masked_fill(~torch.isfinite(anchor[keep]), -1e8) / 4.0
            kl = F.kl_div(new.log_softmax(1), old.softmax(1), reduction="batchmean")
            loss = rank + 0.25 * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            log.append([float(rank.detach()), float(kl.detach())])
    # Write completion marker last. An interrupted fit is repeated from its parent.
    for name, value in [("actor", model.state_dict()), ("adam", optimizer.state_dict())]:
        tmp = directory / (name + ".tmp")
        torch.save(value, tmp)
        tmp.replace(directory / (name + ".pt"))
    stats = dict(
        steps=len(log),
        pairs=len(signed),
        before=before,
        after=accuracy(),
        log=log,
        actor_sha=study.sha(directory / "actor.pt"),
        parent_sha=study.sha(parent),
    )
    study.save(mark, stats)
    return stats


def paired(first, second):
    b = {r["id"]: r for r in first}
    a = {r["id"]: r for r in second}
    assert b.keys() == a.keys()
    return study.compare(
        [
            dict(
                parent=k,
                objects=b[k]["objects"],
                initial=b[k]["baseline"]["success"],
                final=a[k]["baseline"]["success"],
            )
            for k in sorted(b)
        ],
        "initial",
        "final",
    )


def count_labels(rows):
    ds = [b["delta"] for row in rows for b in row["branches"]]
    return dict(
        wins=ds.count(1), losses=ds.count(-1), equal=ds.count(0), controls=2 * len(rows) - len(ds)
    )


def run():
    torch.set_num_threads(1)
    backend = cpp.load()[1]["binary"]
    assert backend == study.read(ppo.OUT / "cpp_backend.json")["binary"]
    assert study.sha(INITIAL) == study.read(ppo.OUT / "protocol.json")["parent"]
    proto = dict(
        rounds=ROUNDS,
        rooms=ROOMS,
        monitor=MONITOR,
        monitor_seed=SEED + 100,
        final_seed=SEED + 101,
        training_seeds=[SEED + i for i in range(1, ROUNDS + 1)],
        initial=study.sha(INITIAL),
        backend=backend,
        design="same rooms and 2 suffix slots per arm per room; identical first round shared; empty slots verify unmodified replay",
        sample="uniform random point in each half of all baseline boundaries with multiple distinct options, randomized order of best alternative versus random lower alternative",
        fit="binary-success discordant rollout pairs; max logit over CAPs proposing same physical action; batch 32 Adam 3e-5, 24 steps per informative round, rank temperature 4, initial-policy KL weight .25",
        control="frozen initial collector/continuation, one pooled fit at end with matched optimizer steps; iterative arm fits each round on current data",
        analysis="each round 48 fixed diagnostic rooms; fixed last checkpoint on 200 new rooms, primary iterative vs pooled, paired stratified bootstrap; one seed",
        training_dedup=False,
        source=study.sha(__file__),
    )
    f = OUT / "protocol.json"
    if f.exists():
        assert study.read(f) == proto
    else:
        study.save(f, proto)
    hist = []
    monitor = panel(SEED + 100, MONITOR)
    publish(history=[], rounds=ROUNDS, rooms=ROOMS)
    baseline = batch(monitor, OUT / "monitor" / "r00", INITIAL, False, "monitor_initial", 0)
    prev = baseline
    parent = INITIAL
    static_dirs = []
    for r in range(1, ROUNDS + 1):
        dest = OUT / f"r{r:02d}"
        rooms = panel(SEED + r, ROOMS)
        current = dest / "iterative"
        static = current if r == 1 else dest / "frozen"
        static_dirs.append(static)
        ck = dest / "checkpoint"
        marker = dest / "complete.json"
        if marker.exists():
            item = study.read(marker)
            assert study.sha(ck / "actor.pt") == item["actor_sha"]
            prev = [study.read(p) for p in sorted((OUT / "monitor" / f"r{r:02d}").glob("*.json"))]
            assert len(prev) == MONITOR
        else:
            rows = batch(rooms, current, parent, True, "collect_iterative", r)
            reference = rows if r == 1 else batch(rooms, static, INITIAL, True, "collect_frozen", r)
            publish(stage="fit")
            stats = fit([current], parent, ck, UPDATES, SEED + 771 + r)
            evals = batch(monitor, OUT / "monitor" / f"r{r:02d}", ck / "actor.pt", False, "monitor", r)
            item = dict(
                round=r,
                iterative=count_labels(rows),
                frozen=count_labels(reference),
                updates=stats["steps"],
                before=stats["before"],
                after=stats["after"],
                actor_sha=stats["actor_sha"],
                monitor=paired(baseline, evals),
                previous=paired(prev, evals),
            )
            study.save(marker, item)
            prev = evals
        parent = ck / "actor.pt"
        hist.append(item)
        publish(history=hist)
    publish(stage="fit_pooled")
    pooled = OUT / "pooled"
    stats = fit(static_dirs, INITIAL, pooled, sum(r["updates"] for r in hist), SEED + 771 + 1)
    final = panel(SEED + 101, 200)
    results = {}
    for name, path in [("initial", INITIAL), ("iterative", parent), ("pooled", pooled / "actor.pt")]:
        results[name] = batch(final, OUT / "final" / name, path, False, "final_" + name, ROUNDS)
    from tools.cap_regional_actor_iteration_report import report

    summary = report(hist, results, stats)
    publish(stage="finished", result=summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run", "pause", "resume"], default="run", nargs="?")
    cmd = parser.parse_args().command
    OUT.mkdir(parents=True, exist_ok=True)
    if cmd == "pause":
        (OUT / "PAUSE").touch()
        return
    with (OUT / "run.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if cmd == "resume":
            (OUT / "PAUSE").unlink(missing_ok=True)
        if (OUT / "FINISHED.json").exists():
            return
        try:
            run()
        except Paused:
            publish(stage="paused")
        except Exception as e:
            publish(stage="error", error=repr(e))
            raise


if __name__ == "__main__":
    main()
