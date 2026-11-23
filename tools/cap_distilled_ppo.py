"""Short paired PPO with a distilled or randomly initialized semantic actor."""
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import torch

from tools import cap_ungated_ppo as p

r = p.r
d = r.d
OUT = p.OUT.parent / "cap_distilled_ppo_20261123_01"
DOC = p.DOC.parent / "2026-11-23_distilled_ppo"
ARMS = ("distilled", "cold")
SEEDS = (17, 29, 43)
TRAIN_ROOM_SEED = 662323977
WORKERS = 24
MODELS = {}


def checkpoint(arm, seed, stage):
    return OUT / "models" / f"{arm}_s{seed}_r{stage}.npz"


def freeze():
    prior = r.e.read(p.OUT / "protocol.json")
    profile = d.init()
    assert prior["profile"] == profile
    d.load("semantic")
    train = r.prior.fixtures(TRAIN_ROOM_SEED)
    panel = r.prior.fixtures(p.CONFIRM)
    assert sorted(panel) == prior["confirm"]
    known = set(prior["confirm"] + prior["diagnostic"])
    for group in r.e.read(d.OUT / "protocol.json")["rooms"]:
        known.update(group.values())
        assert set(panel).isdisjoint(group.values())
    assert set(train).isdisjoint(known)
    keys = np.random.default_rng(TRAIN_ROOM_SEED).permutation(sorted(train)).tolist()
    plan = dict(profile=profile, seeds=list(SEEDS), train_seed=TRAIN_ROOM_SEED,
                stages=[keys[i:i+50] for i in range(0, 200, 50)], panel=sorted(panel),
                checkpoint=p.sha(d.OUT / "fit/semantic.npz"), eval_stages=[0, 4],
                gate="A before first public stall; A for equal proposals",
                inputs="all 349 including nine semantic comparisons, both arms",
                optimizer=prior["optimizer"], rewards=prior["sample"],
                comparison="R4 distilled minus cold, 200 viewed rooms; also R4 minus R0 and teacher",
                ppo_source=p.sha(Path(p.__file__)), driver=p.sha(Path(__file__)))
    if (OUT / "protocol.json").exists():
        assert plan == r.e.read(OUT / "protocol.json")
    r.e.save(OUT / "protocol.json", plan)
    r.study.ROOMS.update(train | panel)
    return plan


def rollout(job, *, greedy):
    arm, seed, stage, key, idx = job
    tag = (arm, seed, stage)
    if tag not in MODELS:
        MODELS[tag] = r.study.cold.weights(checkpoint(*tag))
    d.MODELS["semantic"] = MODELS[tag]
    history = d.PublicHistory()
    rng = np.random.default_rng([seed, stage+1, idx, 921])
    xs, actions, logps = [], [], []

    def choose(_, obs, a, b, context):
        x = history.vector(obs, a, b, context)
        who = "A"
        if context["triggered"] and a != b:
            score = float(d.logits(x, "semantic"))
            if greedy:
                selected = score > 0
            else:
                selected = rng.random() < np.exp(-np.logaddexp(0., -score))
                xs.append(x)
                actions.append(int(selected))
                logps.append(-np.logaddexp(0., -score if selected else score))
            who = "B" if selected else "A"
        history.executed(who, a, b, context)
        return who

    d.source.ROUTE = choose
    result = d.source.play(r.study.ROOMS[key], stall_gate=True)["result"]
    if greedy:
        r.e.save(OUT / "evaluation" / f"{arm}_s{seed}_r{stage}" / (key+".json"),
                 {k: result[k] for k in ("success", "ticks", "history")})
        return
    return (seed, key, result["success"], result["ticks"],
            np.asarray(xs, np.float32).reshape(-1, 349), np.array(actions, int),
            np.asarray(logps, np.float32))


def train(plan):
    models, optimizers = {}, {}
    for seed in SEEDS:
        for arm in ARMS:
            torch.manual_seed(seed)
            model = p.Policy(None)
            if arm == "distilled":
                with torch.no_grad():
                    for param, weight in zip(model.actor.parameters(), d.MODELS["semantic"], strict=True):
                        param.copy_(torch.from_numpy(weight.T.copy()))
            tag = arm, seed
            models[tag] = model
            optimizers[tag] = torch.optim.Adam([dict(params=model.actor.parameters(), lr=3e-5),
                                                dict(params=model.critic.parameters(), lr=3e-4)])
            save_actor(*tag, 0, model)
        for x, y in zip(models["cold", seed].critic.parameters(), models["distilled", seed].critic.parameters()):
            assert torch.equal(x, y)
    with ProcessPoolExecutor(WORKERS, mp_context=mp.get_context("fork")) as pool:
        for stage, keys in enumerate(plan["stages"], 1):
            jobs = [(arm, seed, stage-1, key, idx) for idx, key in enumerate(keys) for arm, seed in models]
            rows = {tag: [] for tag in models}
            for job, row in zip(jobs, pool.map(partial(rollout, greedy=False), jobs), strict=True):
                rows[job[:2]].append(row)
            for tag, episodes in rows.items():
                arm, seed = tag
                assert [row[1] for row in episodes] == keys
                stats = p.fit(models[tag], optimizers[tag], episodes, seed, stage)
                save_actor(arm, seed, stage, models[tag])
                r.e.save(OUT / "metrics" / f"{arm}_s{seed}_r{stage}.json", stats)
            print("completed round", stage, flush=True)


def save_actor(arm, seed, stage, model):
    fields = {f"arr_{i}": w.detach().numpy().T.copy() for i, w in enumerate(model.actor.parameters())}
    r.exp.save_npz(checkpoint(arm, seed, stage), **fields)


def main():
    torch.set_num_threads(1)
    plan = freeze()
    if not (OUT / "lock.json").exists():
        train(plan)
        r.e.save(OUT / "lock.json", {f.name: p.sha(f) for f in (OUT / "models").glob("*.npz")})
    for name, digest in r.e.read(OUT / "lock.json").items():
        assert p.sha(OUT / "models" / name) == digest
    jobs = []
    for stage in plan["eval_stages"]:
        for arm in ARMS:
            for seed in SEEDS:
                if stage == 0 and arm == "distilled" and seed != SEEDS[0]:
                    continue
                for idx, key in enumerate(plan["panel"]):
                    if not (OUT / "evaluation" / f"{arm}_s{seed}_r{stage}" / (key+".json")).exists():
                        jobs.append((arm, seed, stage, key, idx))
    r.exp.OUT = OUT
    r.exp.WORKERS = WORKERS
    r.exp.timed_pool("evaluation", partial(rollout, greedy=True), jobs)


if __name__ == "__main__":
    main()
