"""Current actor rollouts with room-balanced historical preference replay."""

import argparse
import fcntl
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tools import cap_regional_actor_iteration as engine

study = engine.study
OLD = engine.OUT
OUT = OLD.parent / "cap_regional_replay_20261120_01"
DOCS = engine.DOCS.parent / "2026-11-20_regional_actor_replay"


def records(directory):
    return [study.read(path) for path in sorted(directory.glob("*.json"))]


def sample_history(groups, rng, n=16):
    room_indices = rng.integers(len(groups), size=n)
    return np.array([rng.choice(groups[i]) for i in room_indices])


def load_pairs(directories):
    xs = []
    a = []
    b = []
    ds = []
    groups = []
    offset = 0
    for directory in directories:
        for path in sorted(directory.glob("*.json")):
            fork = study.read(path)["branches"]
            signed = [i for i, row in enumerate(fork) if row["delta"]]
            if not signed:
                continue
            with np.load(path.with_suffix(".npz")) as f:
                xs.append(f["x"][signed])
                a.append(f["a"][signed])
                b.append(f["b"][signed])
            ds.extend(fork[i]["delta"] for i in signed)
            groups.append(np.arange(offset, offset + len(signed)))
            offset += len(signed)
    if not groups:
        return None, []
    data = {k: torch.from_numpy(np.concatenate(v)) for k, v in [("x", xs), ("a", a), ("b", b)]}
    data["d"] = torch.tensor(ds, dtype=torch.float)
    return data, groups


def acc(model, data):
    if data is None or not len(data["d"]):
        return None
    with torch.no_grad():
        hits = 0
        for idx in torch.arange(len(data["d"])).split(128):
            diff = engine.group_margin(model(data["x"][idx]), data["a"][idx], data["b"][idx])
            hits += int((diff * data["d"][idx] > 0).sum())
        return hits / len(data["d"])


def fit(parent, directory, current, history, rid):
    marker = directory / "fit.json"
    if marker.exists():
        result = study.read(marker)
        assert study.sha(directory / "actor.pt") == result["actor_sha"] and result[
            "parent_sha"
        ] == study.sha(parent)
        return result
    directory.mkdir(exist_ok=True, parents=True)
    model = engine.actor_load(parent)
    opt = torch.optim.Adam(model.parameters(), lr=3e-5)
    opt.load_state_dict(torch.load(parent.parent / "adam.pt", weights_only=True, map_location="cpu"))
    cur = engine.datasets([current])
    signed = np.flatnonzero(cur["d"].numpy()) if cur is not None else np.array([], int)
    cur_rank = {k: v[signed] for k, v in cur.items()} if cur is not None else None
    past, groups = load_pairs(history)
    before = acc(model, cur_rank)
    old_before = acc(model, past)
    rng = np.random.default_rng(engine.SEED + 771 + rid)
    log = []
    hist_draws = 0
    new_draws = 0
    if len(signed):
        initial = engine.actor_load(engine.INITIAL)
        with torch.no_grad():
            anchor = initial(cur["x"])
        for _ in range(engine.UPDATES):
            engine.check_pause()
            half = 16 if groups else 0
            ix = rng.choice(signed, 32 - half)
            batch = {k: v[ix] for k, v in cur.items()}
            if half:
                j = sample_history(groups, rng, half)
                batch = {k: torch.cat([v, past[k][j]]) for k, v in batch.items()}
            hist_draws += half
            new_draws += len(ix)
            diff = engine.group_margin(model(batch["x"]), batch["a"], batch["b"])
            rank = F.softplus(-batch["d"] * diff / 4.0).mean()
            keep = rng.integers(len(cur["d"]), size=32)
            mask = ~torch.isfinite(anchor[keep])
            old = anchor[keep].masked_fill(mask, -1e8) / 4.0
            new = model(cur["x"][keep]).masked_fill(mask, -1e8) / 4.0
            kl = F.kl_div(new.log_softmax(1), old.softmax(1), reduction="batchmean")
            opt.zero_grad(set_to_none=True)
            (rank + 0.25 * kl).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            opt.step()
            log.append([float(rank.detach()), float(kl.detach())])
    for name, value in [("actor", model.state_dict()), ("adam", opt.state_dict())]:
        temp = directory / (name + ".tmp")
        torch.save(value, temp)
        temp.replace(directory / (name + ".pt"))
    result = dict(
        parent_sha=study.sha(parent),
        actor_sha=study.sha(directory / "actor.pt"),
        pairs=len(signed),
        before=before,
        after=acc(model, cur_rank),
        history_before=old_before,
        history_after=acc(model, past),
        history_rooms=len(groups),
        history_pairs=0 if past is None else len(past["d"]),
        new_draws=new_draws,
        history_draws=hist_draws,
        steps=len(log),
        log=log,
    )
    study.save(marker, result)
    return result


def run():
    torch.set_num_threads(1)
    # The existing runner's write target also controls its lock-free pause checks in fork workers.
    engine.OUT = OUT
    orig = study.read(OLD / "protocol.json")
    ref = study.read(OLD / "FINISHED.json")
    assert (
        orig["source"] == study.sha(engine.__file__)
        and orig["backend"] == engine.cpp.load()[1]["binary"]
    )
    assert orig["initial"] == study.sha(engine.INITIAL) and ref["updates"] == 576
    proto = dict(
        parent_protocol=study.sha(OLD / "protocol.json"),
        source=study.sha(__file__),
        primary="matched paired final replay vs archived current-only policy on previously viewed panel",
        shared="round 1 collection, update, monitor; round 2 collection from identical parent",
        samples="16 current uniformly sampled signed pairs + 16 past, uniformly draw informative room then pair, with replacement",
        labels="signed terminal difference under frozen collection actor at that round; archive parent sha per round",
        kl="32 current boundaries as in current-only control",
        diagnostic="96 development rooms from rounds 1 and 2, alternate best/exploration slot by room ordinal; same archived actor tails; actual own expert tails; ties highest parent logit; invalid suffix separately",
        rounds=24,
        seed=17,
        reference_driver=orig["source"],
        expert_driver=study.sha(Path(__file__).with_name("cap_regional_expert_suffix.py")),
    )
    file = OUT / "protocol.json"
    if file.exists():
        assert study.read(file) == proto
    else:
        study.save(file, proto)
    monitor = study.read(OLD / "panels" / f"{engine.SEED + 100}.json")
    baseline = records(OLD / "monitor/r00")
    prev = baseline
    parent = engine.INITIAL
    history = []
    dirs = []
    engine.publish(history=[], rounds=24, rooms=48)
    for r in range(1, 25):
        engine.check_pause()
        dest = OUT / f"r{r:02d}"
        dest.mkdir(exist_ok=True)
        source = OLD / f"r{r:02d}"
        cur = (source if r <= 2 else dest) / "iterative"
        ck = (source if r == 1 else dest) / "checkpoint"
        monitor_dir = (OLD if r == 1 else OUT) / "monitor" / f"r{r:02d}"
        marker = dest / "complete.json"
        if marker.exists():
            item = study.read(marker)
            prev = records(monitor_dir)
            assert (
                item["actor_sha"] == study.sha(ck / "actor.pt")
                and item["collector_sha"] == study.sha(parent)
                and len(prev) == 48
            )
        else:
            if r <= 2:
                assert study.read(source / "checkpoint/fit.json")["parent_sha"] == study.sha(parent)
                rows = records(cur)
            else:
                rooms = study.read(OLD / "panels" / f"{engine.SEED + r}.json")
                rows = engine.batch(rooms, cur, parent, True, "collect_replay", r)
            assert len(rows) == 48
            engine.publish(stage="fit_replay", round=r)
            if r == 1:
                stats = study.read(ck / "fit.json")
                evals = records(monitor_dir)
            else:
                stats = fit(parent, ck, cur, dirs, r)
                evals = engine.batch(monitor, monitor_dir, ck / "actor.pt", False, "monitor", r)
            cold = records(OLD / "monitor" / f"r{r:02d}")
            item = dict(
                round=r,
                collector_sha=study.sha(parent),
                actor_sha=stats["actor_sha"],
                pairs=stats["pairs"],
                before=stats["before"],
                after=stats["after"],
                steps=stats["steps"],
                old_pairs=stats.get("history_pairs", 0),
                history_before=stats.get("history_before"),
                history_after=stats.get("history_after"),
                history_draws=stats.get("history_draws", 0),
                current_draws=stats.get("new_draws", stats["steps"] * 32),
                monitor=engine.paired(baseline, evals),
                vs_current=engine.paired(cold, evals),
                previous=engine.paired(prev, evals),
                train_vs_current=engine.paired(records(source / "iterative"), rows),
            )
            study.save(marker, item)
            prev = evals
        history.append(item)
        dirs.append(cur)
        parent = ck / "actor.pt"
        engine.publish(history=history)
    final = study.read(OLD / "panels" / f"{engine.SEED + 101}.json")
    evals = engine.batch(final, OUT / "final", parent, False, "final_replay", 24)
    from tools.cap_regional_actor_replay_report import report

    result = report(history, evals)
    engine.publish(stage="finished", result=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", default="run", choices=["run", "pause", "resume"])
    cmd = parser.parse_args().command
    OUT.mkdir(exist_ok=True, parents=True)
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
        except engine.Paused:
            engine.publish(stage="paused")
        except Exception as e:
            engine.publish(stage="error", error=repr(e))
            raise


if __name__ == "__main__":
    main()
