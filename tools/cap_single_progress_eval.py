"""Compare automatic and frozen PPO-timed updates for each isolated expert."""
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from pathlib import Path

import torch

from tools import cap_abc_event_ppo as event
from tools.cap_abc_event_routes import PANEL_SEED, ROOT, RUN

OUT = RUN.parent / "cap_single_progress_eval_20261124_01"
s = event.s
base = event.base


def evaluate(job):
    key, expert, rule = job
    member = s.phase.FrozenCAP(base.third.MEMBERS[expert])
    slot = base.EXPERTS.index(expert)
    counts, reviews = Counter(), Counter()
    history = deque(maxlen=4)

    def decide(world, obs, action, left, reason):
        reviews[reason] += 1
        elapsed = obs["tick"] - s.observe(world)["tick"]
        assert elapsed > 0 and left >= 1
        new = event.preview(member, obs, action, elapsed)
        if new is None or event.same_control(new, action):
            return None
        reviews["eligible"] += 1
        if rule == "continue":
            return None
        if rule == "scored":
            proposals = [None] * len(base.EXPERTS)
            proposals[slot] = new
            panel = event.candidate_rows(base.public_frame(obs), history, obs, proposals,
                                         dict(action, ticks=left))
            assert panel[:, -1].nonzero()[0].tolist() == [slot, event.CONTINUE]
            if base.scores(panel, WEIGHTS).argmax() == event.CONTINUE:
                return None
        reviews["switched"] += 1
        return new

    with s.phase.world(base.ROOMS[key]) as (world, _):
        member.update(world.history)
        pending = None
        review = event.PushReview("progress", decide, progress_m=PLAN["progress_m"],
                                  contact_ticks=PLAN["contact_ticks"])
        while not s.ended(world):
            obs = s.observe(world)
            action = member.propose(obs) if pending is None else pending
            if rule == "scored":
                history.append(dict(base.public_frame(obs), action=action))
            pending = s.tick_action(world, action, base.third.EXECUTOR, member.update,
                                    counts, review=review)
        result = s.output(world, counts)
    if rule == "continue" or not reviews["switched"]:
        baseline = ORIGINAL[expert]["rooms"][key]
        assert all(result[field] == baseline[field] for field in result), (expert, key)
    return expert, rule, key, dict(result, reviews=dict(reviews))


def main():
    global PLAN, WEIGHTS, ORIGINAL
    torch.set_num_threads(1)
    PLAN = base.read(RUN / "protocol.json")
    assert all(event.long.ppo.sha(ROOT / src) == digest for src, digest in PLAN["sources"].items())
    setup = base.setup()
    assert setup["expert_sha"] == PLAN["experts"]
    assert all(setup[k] == PLAN[k] for k in ("profile", "executor"))
    rooms = base.get_rooms(PANEL_SEED)
    keys = PLAN["confirmation"]
    assert len(keys) == len(rooms) == 200 and set(rooms) == set(keys)
    base.ROOMS.update(rooms)
    ORIGINAL = {e: base.read(RUN / "evaluation" / f"{e}.json") for e in base.EXPERTS}
    assert all(set(v["rooms"]) == set(keys) and v["policy_sha"] == PLAN["experts"][e]
               for e, v in ORIGINAL.items())
    final = base.read(RUN / "finished.json")
    actor = event.actor_path(RUN, "progress", final["rounds"])
    assert event.long.ppo.sha(actor) == final["actors"]["progress"]
    WEIGHTS = event.load_weights(actor)
    jobs = [(key, expert, rule) for key in keys for expert in base.EXPERTS
            for rule in ("auto", "scored")]
    parity = [(key, expert, "continue") for key in keys[:4] for expert in base.EXPERTS]
    base.save(OUT / "protocol.json", dict(panel=keys, panel_seed=PANEL_SEED, profile=PLAN["profile"],
              experts=PLAN["experts"], actor_sha=final["actors"]["progress"],
              sources_sha=event.long.ppo.sha(Path(__file__)), progress_m=PLAN["progress_m"],
              contact_ticks=PLAN["contact_ticks"], rules=["auto", "scored"], parity=parity))
    with ProcessPoolExecutor(24, mp_context=mp.get_context("fork")) as pool:
        checks = list(pool.map(evaluate, parity))
        base.save(OUT / "parity.json", checks)
        results = {}
        for i, (expert, rule, key, row) in enumerate(pool.map(evaluate, jobs), 1):
            results.setdefault(rule, {}).setdefault(expert, {})[key] = row
            if i % 120 == 0:
                print(f"{i}/{len(jobs)}", flush=True)
        assert all(set(rows) == set(keys) for by_expert in results.values() for rows in by_expert.values())
        base.save(OUT / "evaluation.json", results)


if __name__ == "__main__":
    main()
