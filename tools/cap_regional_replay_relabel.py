"""Revisit the unchanged first-round forks under the final actor continuation."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch

from tools import cap_regional_actor_iteration as bank

PREFIX = TAIL = None


@torch.inference_mode()
def outcome(fixture, row, side):
    env = bank.ppo.environment(fixture)
    library = bank.cpp.Proposals()
    reached = False
    try:
        for t in range(300):
            x, _, _, _, pool, raw = library.panel(env.history)
            model = TAIL if reached else PREFIX
            logits = model(torch.from_numpy(x[None]))[0]
            idx = int(logits.argmax())
            action = raw[idx] if logits[idx].isfinite() else pool[0]
            if t == row["step"]:
                assert env.history.sha256 == row["prefix"] and action == row["base"]
                action = row[side]
                assert action in pool
                reached = True
            env.step(action, remaining_ticks=1800 - env.env.total_periods)
            if env.env.termination is not None or env.env.total_periods >= 1800:
                break
        assert reached
        return dict(
            success=int(env.env.success),
            steps=t + 1,
            ticks=env.env.total_periods,
            history=env.history.sha256,
        )
    finally:
        env.close_reads()
        bank.study.dq.fast_runtime.ACTIVE = None


def job(args):
    room, row, path = args
    if path.exists():
        return bank.study.read(path)
    rec = dict(id=row["id"], prefix=row["prefix"], step=row["step"])
    for side in ("base", "alt"):
        rec["original_" + side] = row["actor_" + side]["success"]
        rec["latest_" + side] = outcome(room["fixture"], row, side)
    bank.study.save(path, rec)
    return rec


def run(reference, output):
    global PREFIX, TAIL
    rooms = bank.study.read(reference / "panels" / f"{bank.SEED + 1}.json")
    PREFIX = bank.actor_load(bank.INITIAL)
    TAIL = bank.actor_load(output / "r24/checkpoint/actor.pt")
    tasks = []
    for room in rooms:
        row = bank.study.read(output / "expert_suffix/rooms" / (room["id"] + ".json"))
        if row["available"]:
            tasks.append((room, row, output / "relabel" / (room["id"] + ".json")))
    counts = [[0] * 3 for _ in range(3)]
    with ProcessPoolExecutor(16, mp_context=mp.get_context("fork")) as executor:
        for f in as_completed([executor.submit(job, t) for t in tasks]):
            r = f.result()
            d = r["original_alt"] - r["original_base"]
            new = r["latest_alt"]["success"] - r["latest_base"]["success"]
            counts[d + 1][new + 1] += 1
    return dict(
        n=len(tasks),
        labels=counts,
        strict_flips=counts[0][2] + counts[2][0],
        preference_to_tie=counts[0][1] + counts[2][1],
        tie_to_preference=counts[1][0] + counts[1][2],
        original_informative=sum(counts[0]) + sum(counts[2]),
    )
