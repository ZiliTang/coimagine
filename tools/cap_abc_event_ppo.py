"""Option-internal CAP decisions and a matched closed-loop preflight."""
import argparse
from collections import Counter, deque
from concurrent.futures import ProcessPoolExecutor
import fcntl
from functools import lru_cache
from math import dist
import multiprocessing as mp
from pathlib import Path
import time

import numpy as np
import torch

from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl.push_phases import pusher
from experiments.planar_tasks.semantic_rl.trusted_native import owned_copy
from tools import cap_abc_ppo_extended as long
from tools import cap_abc_semantic_ppo as base

s = base.s
save = base.save
OUT = long.OUT.parent/"cap_abc_ppo_event_preflight_20261124_01"
DOC = long.DOC.parent/"2026-11-24_abc_ppo_event_preflight"
MODES = ("original", "time", "progress")
CONTINUE = 3
CHOICES = 4
CONTINUE_FEATURE = base.BASE_SIZE
INTERIM_ROUND = 500


class PushReview:
    """Track contact and signed robot travel during the certified single push."""

    def __init__(self, mode, decide, *, time_ticks=4, progress_m=.20, contact_ticks=3):
        if mode not in MODES[1:] or time_ticks < 1 or progress_m <= 0 or contact_ticks < 1:
            raise ValueError("invalid review configuration")
        self.mode = mode
        self.decide = decide
        self.time_ticks = time_ticks
        self.progress_m = progress_m
        self.contact_ticks = contact_ticks

    def __call__(self, world, action):
        robot, axis = pusher(action)
        initial = s.observe(world)["robots"][robot]
        previous = initial["own"]
        stable = 0
        contacted = False
        was_valid = False
        travel = 0.
        start = world.env.total_periods

        def review(feedback, left):
            nonlocal previous, stable, contacted, was_valid, travel
            # An actual replacement needs at least one further command slot.
            if world.decision_count+1 >= s.MAX_DECISIONS:
                return None
            sensor = feedback["robots"][robot]
            own = sensor["own"]
            execution = sensor["execution"]
            valid = (sensor["contacting_object"] and execution.get("direction_valid", False)
                     and execution.get("stage") not in ("brake", "switch_brake")
                     and world.env.active_target == action["target_id"])
            stable = stable+1 if valid else 0
            reason = None
            if self.mode == "time":
                if (world.env.total_periods-start) % self.time_ticks == 0:
                    reason = "time"
            elif not contacted and stable >= self.contact_ticks:
                contacted = True
                reason = "contact"
            elif contacted:
                if was_valid and valid:
                    travel += (own["x"]-previous["x"])*axis[0] + (own["y"]-previous["y"])*axis[1]
                if travel+1e-9 >= self.progress_m:
                    # Keep odometry overshoot for the next threshold.
                    travel -= self.progress_m
                    reason = "progress"
            previous, was_valid = own, valid
            if reason is None:
                return None
            obs = world.env.public()
            return self.decide(world, obs, action, left, reason)

        return review


class Policy(base.Policy):
    """Keep the shared score model and admit an explicit continuation row."""

    def forward(self, x):
        distribution, legal = self.distribution(x)
        original = x[:, :CONTINUE, :base.BASE_SIZE]
        with_continue = (x[..., :base.BASE_SIZE] * legal[..., None]).sum(1) / legal.sum(1, keepdim=True)
        value_inputs = torch.where(legal[:, CONTINUE, None], with_continue, original.mean(1))
        return distribution, self.critic(value_inputs).flatten(), legal


def preview(member, obs, action, elapsed):
    event = dict(observation=obs, action=action, executed_ticks=elapsed)
    memory = member.module.update_memory(owned_copy(member.memory), owned_copy(event))
    context = member.module.prepare_context(owned_copy(memory), owned_copy(obs))
    try:
        return validate_action(context["proposals"][0], obs)
    except ValueError:
        # An authored proposal can fall outside the menu at a novel boundary.
        return None


def same_control(first, second):
    return all(first[field] == second[field] for field in ("target_id", "a0", "a1"))


def candidate_rows(frame, history, obs, proposals, continuation=None):
    if len(proposals) != CONTINUE:
        raise ValueError("expected the three expert proposals")
    # Fill unavailable rows for the encoder. Only real proposals enter the mask.
    filler = continuation or proposals[0]
    encoded = [a if a is not None else filler for a in proposals]
    x = np.zeros((CHOICES,base.SIZE),np.float32)
    x[:CONTINUE] = base.project(base.candidate_inputs(frame,history,obs,encoded),"basic")
    if continuation is None:
        return x
    for i, action in enumerate(proposals):
        x[i,-1] = action is not None and not same_control(action, continuation) and action not in proposals[:i]
        x[i,base.BASE_SIZE-3:base.BASE_SIZE] = [
            float(action is not None and a==action) for a in proposals
        ]
    context = base.candidate_inputs(frame,history,obs,[continuation]*CONTINUE)[0]
    x[CONTINUE] = base.project(context,"basic")
    x[CONTINUE,base.BASE_SIZE-3:base.BASE_SIZE] = [
        float(a is not None and same_control(a, continuation)) for a in proposals
    ]
    x[CONTINUE,CONTINUE_FEATURE] = 1
    x[CONTINUE,-1] = 1
    return x


def episode(key, weights, mode, *, review_policy="actor", rng=None, time_ticks=4, progress_m=.20,
            collect=False):
    """Run a normal rollout or collect PPO decisions with option review."""
    if collect and rng is None:
        raise ValueError("PPO collection requires a sampling generator")
    members = [s.phase.FrozenCAP(base.third.MEMBERS[expert]) for expert in base.EXPERTS]
    counts, reviews, history = Counter(), Counter(), deque(maxlen=4)
    x_rows, chosen, logp, distances, choice_ticks = [], [], [], [], []
    pending = None
    with s.phase.world(base.ROOMS[key]) as (world, _):
        def update(events):
            for member in members:
                member.update(events)

        if collect:
            goal = s.observe(world)["goal"]
            body = world.env.env.backend.bodies[goal["target_id"]]
            start_distance = dist(body.position, goal["xy"])

        def choose(x):
            scores = base.scores(x,weights)
            index = int(scores.argmax())
            if rng is not None:
                p = np.exp((scores-scores.max()).astype(np.float64))
                p /= p.sum()
                index = int(rng.choice(CHOICES,p=p))
                if collect:
                    x_rows.append(x)
                    chosen.append(index)
                    logp.append(np.log(p[index]))
                    distances.append(dist(body.position,goal["xy"]))
                    choice_ticks.append(world.env.total_periods)
            return index

        def on_review(world, obs, action, left, reason):
            reviews[reason] += 1
            elapsed = obs["tick"]-s.observe(world)["tick"]
            assert 0<elapsed<=action["ticks"] and left>=1
            proposals = [preview(member,obs,action,elapsed) for member in members]
            ongoing = dict(action,ticks=left)
            alternatives = [i for i,p in enumerate(proposals) if p is not None and not same_control(p, ongoing)]
            if not alternatives:
                reviews["no_alternative"] += 1
                return None
            reviews["eligible"] += 1
            if review_policy == "continue":
                return None
            if review_policy == "switch":
                index = alternatives[0]
            else:
                x = candidate_rows(base.public_frame(obs),history,obs,proposals,ongoing)
                index = choose(x)
                if index == CONTINUE:
                    return None
            reviews["switched"] += 1
            return proposals[index]

        review = None if mode=="original" else PushReview(mode,on_review,time_ticks=time_ticks,progress_m=progress_m)
        update(world.history)
        while not s.ended(world):
            obs = s.observe(world)
            frame = base.public_frame(obs)
            if pending is None:
                proposals = [member.propose(obs) for member in members]
                if any(action != proposals[0] for action in proposals[1:]):
                    index = choose(candidate_rows(frame,history,obs,proposals))
                else:
                    index = 0
                action = proposals[index]
            else:
                action = pending
            history.append(dict(frame,action=action))
            pending = s.tick_action(world,action,base.third.EXECUTOR,update,counts,review=review)
        result=s.output(world,counts)
        if collect:
            final_distance=dist(body.position,goal["xy"])
    result["reviews"]=dict(reviews)
    if collect:
        result["distance_m"] = final_distance
        result["progress_m"] = start_distance-final_distance
        result["forward_progress_m"] = float(np.maximum(0., -np.diff([start_distance, *distances, final_distance])).sum())
        result["choice_periods"] = np.diff([*choice_ticks, result["ticks"]]).tolist()
        rewards = (-long.dense.PER_METRE*np.diff([*distances,final_distance])).astype(np.float32)
        if len(rewards):
            rewards[-1] += result["success"]
        return result, np.asarray(x_rows,np.float32).reshape(-1,CHOICES,base.SIZE), np.asarray(chosen,int), np.asarray(logp,np.float32), rewards
    return result


def compare(job):
    key, time_ticks = job
    baseline = base.episode(key, actor=("basic", WEIGHTS), greedy=True)[0]
    outcomes = {}
    for mode in MODES:
        started = time.perf_counter()
        pass_through = episode(key,WEIGHTS,mode,review_policy="continue",time_ticks=time_ticks)
        pass_through["wall_seconds"] = time.perf_counter()-started
        for field in ("history","ticks","decisions","termination","success","counts"):
            if pass_through[field] != baseline[field]:
                raise ValueError(f"{key}: {mode} continue diverged at {field}")
        outcomes[mode] = pass_through
    for mode in MODES[1:]:
        started = time.perf_counter()
        switched = episode(key,WEIGHTS,mode,review_policy="switch",time_ticks=time_ticks)
        switched["wall_seconds"] = time.perf_counter()-started
        outcomes[f"{mode}_switch"] = switched
    save(OUT/"rooms"/(key+".json"),outcomes)
    return outcomes


def shared_setup(parent_round):
    plan = base.setup()
    source = long.PARENT if parent_round == long.START else long.OUT
    actor = long.previous.model_path(source, parent_round)
    state = long.previous.state_path(source, parent_round)
    digits = 2 if source == long.PARENT else 3
    record = base.read(source/"metrics"/f"r{parent_round:0{digits}d}.json")
    assert long.ppo.sha(actor) == base.read(source/"locked.json")[str(parent_round)] == record["actor_sha"]
    assert long.ppo.sha(state) == record["state_sha"]
    return plan, actor, state


@lru_cache(maxsize=3)
def load_weights(path):
    with np.load(path) as arrays:
        return tuple(arrays[f"a{i}"] for i in range(6))


def from_checkpoint(state):
    model = Policy()
    optimizer = torch.optim.Adam([
        dict(params=model.actor.parameters(), lr=3e-5),
        dict(params=model.critic.parameters(), lr=3e-4),
    ])
    long.previous.restore(state, model, optimizer)
    # The basic parent never received this input. Introduce it with no initial preference.
    weight = model.actor[0].weight
    with torch.no_grad():
        weight[:, CONTINUE_FEATURE].zero_()
        for name in ("exp_avg", "exp_avg_sq"):
            optimizer.state[weight][name][:, CONTINUE_FEATURE].zero_()
    return model, optimizer


def export(path, model):
    arrays = {f"a{i}": p.detach().numpy().T.copy() for i, p in enumerate(model.actor.parameters())}
    base.ppo.r.exp.save_npz(path, **arrays)


def preflight(count,workers,time_ticks):
    global WEIGHTS
    _, actor, state = shared_setup(long.START)
    WEIGHTS = load_weights(actor)
    dev = base.read(base.third.OUT/"protocol.json")["dev_content"]
    keys = sorted(dev,key=lambda k:base.third.pair.key("event-preflight/"+k))[:count]
    model, _ = from_checkpoint(state)
    smoke_actor = OUT/"smoke_actor.npz"
    smoke_actor.parent.mkdir(parents=True, exist_ok=True)
    export(smoke_actor, model)
    with ProcessPoolExecutor(workers,mp_context=mp.get_context("fork")) as pool:
        rows = list(pool.map(compare,[(key,time_ticks) for key in keys]))
        smoke = {mode:smoke_update(pool,keys,mode,time_ticks,smoke_actor,state) for mode in MODES}
    summary = {}
    for mode in (*MODES,"time_switch","progress_switch"):
        modes = [row[mode] for row in rows]
        reviews = Counter()
        for row in modes:
            reviews.update(row["reviews"])
        summary[mode] = dict(success=sum(r["success"] for r in modes),reviews=dict(reviews),
                             ticks=sum(r["ticks"] for r in modes),
                             wall_seconds=sum(r["wall_seconds"] for r in modes))
    report = dict(n=len(rows),review_ticks=time_ticks,continue_parity=True,modes=summary,ppo_smoke=smoke)
    save(OUT/"summary.json",report)
    print(report,flush=True)


def collect_row(job):
    key, fixture, mode, stage, index, time_ticks, actor = job
    rng = np.random.default_rng([long.SEED, stage, index, 1549])
    if fixture is not None:
        base.ROOMS[key] = fixture
    try:
        result, x, actions, logp, rewards = episode(
            key, load_weights(actor), mode, rng=rng, time_ticks=time_ticks, collect=True
        )
    finally:
        if fixture is not None:
            del base.ROOMS[key]
    # The signed-progress PPO learner consumes a reward vector at each route decision.
    return (key, result["success"], result["ticks"], x, actions, logp, result["distance_m"],
            rewards, long.dense.PER_METRE * result["progress_m"], result["forward_progress_m"],
            result["reviews"], result["choice_periods"])


def smoke_update(pool, keys, mode, time_ticks, actor, state):
    model, optimizer = from_checkpoint(state)
    stage = long.START + 1
    jobs = [(k, None, mode, stage, i, time_ticks, actor) for i,k in enumerate(keys)]
    rows = list(pool.map(collect_row, jobs))
    review_choices=sum(int(np.count_nonzero(row[3][:,CONTINUE,-1])) for row in rows)
    if mode != "original" and not review_choices:
        raise ValueError(f"{mode} smoke run supplied no review choices")
    stats = long.dense.fit(model,optimizer,rows,stage)
    assert stats["old_logp_error"]<1e-3
    return dict(choices=stats["n"],review_choices=review_choices,
                old_logp_error=stats["old_logp_error"],updates=stats["updates"])


def greedy(job):
    key, mode, time_ticks, actor = job
    if actor is None:
        return base.episode(key, teacher=mode)[0]
    return episode(key, load_weights(actor), mode, time_ticks=time_ticks)


def evaluate(pool, path, mode, actor, keys, time_ticks, *, expert_sha=None):
    policy_sha = long.ppo.sha(actor) if actor is not None else expert_sha
    if path.is_file():
        existing = base.read(path)
        assert existing["policy_sha"] == policy_sha and set(existing["rooms"]) == set(keys)
        return
    results = list(pool.map(greedy, [(key, mode, time_ticks, actor) for key in keys]))
    save(path, dict(n=len(keys), success=sum(result["success"] for result in results),
                    policy_sha=policy_sha, rooms=dict(zip(keys, results, strict=True))))


def actor_path(out, mode, stage):
    name = "shared" if not stage else f"{mode}_{stage:04d}"
    return out/"actors"/(name+".npz")


def state_path(out, mode, stage):
    return out/"state"/mode/f"r{stage:04d}.pt"


def metrics_path(out, mode, stage):
    return out/"metrics"/mode/f"r{stage:04d}.json"


def save_training_round(out, mode, stage, model, optimizer, metrics):
    actor = actor_path(out, mode, stage)
    state = state_path(out, mode, stage)
    export(actor, model)
    state.parent.mkdir(parents=True, exist_ok=True)
    long.ppo.legacy.save_checkpoint(state, dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
    save(metrics_path(out, mode, stage), dict(metrics, actor_sha=long.ppo.sha(actor), state_sha=long.ppo.sha(state)))
    if stage > 1:
        state_path(out, mode, stage-1).unlink(missing_ok=True)
        if (stage-1) % 20:
            actor_path(out, mode, stage-1).unlink(missing_ok=True)


def restore_latest(out, mode, model, optimizer, rounds):
    files = sorted((out/"metrics"/mode).glob("r*.json"))
    stage = len(files)
    assert stage <= rounds and files == [metrics_path(out, mode, i) for i in range(1, stage+1)]
    if stage:
        record = base.read(files[-1])
        state = state_path(out, mode, stage)
        assert long.ppo.sha(state) == record["state_sha"]
        assert long.ppo.sha(actor_path(out, mode, stage)) == record["actor_sha"]
        long.previous.restore(state, model, optimizer)
    return stage


def evaluate_panel(pool, out, keys, stage, initial, time_ticks, experts, *, interim=False):
    folder = out/("interim_evaluation" if interim else "evaluation")
    for mode in MODES:
        evaluate(pool, folder/f"{mode}.json", mode, actor_path(out, mode, stage), keys, time_ticks)
    evaluate(pool, folder/"parent.json", "original", initial, keys, time_ticks)
    for expert, digest in experts.items():
        evaluate(pool, folder/f"{expert}.json", expert, None, keys, time_ticks, expert_sha=digest)
    results = {mode:base.read(folder/f"{mode}.json") for mode in (*MODES,"parent",*experts)}
    original = [results["original"]["rooms"][key]["success"] for key in keys]
    paired = {}
    for mode in MODES[1:]:
        outcomes = [results[mode]["rooms"][key]["success"] for key in keys]
        paired[mode] = dict(gained=sum(a>b for a,b in zip(outcomes,original,strict=True)),
                            lost=sum(a<b for a,b in zip(outcomes,original,strict=True)))
    save(out/("interim.json" if interim else "confirmation.json"), dict(
        episodes_per_arm=stage*long.BATCH, rooms=len(keys),
        successes={mode:value["success"] for mode,value in results.items()}, versus_original=paired,
    ))


def train_comparison(out, rounds, workers, time_ticks):
    """Train all arms on each shared room batch and evaluate locked final actors."""
    plan, actor, state = shared_setup(long.END)
    extension = base.read(long.OUT/"protocol.json")
    diagnostic = extension["diagnostic"]
    used = s.collect_identities() | set(base.ROOMS) | set(extension["confirm"])
    used.update(s.phase.input_hash(fixture) for fixture in base.ROOMS.values())
    for path in (long.OUT/"metrics").glob("r*.json"):
        used.update(base.read(path)["rooms"])
    test = base.get_rooms(662344521)
    interim = base.get_rooms(662344522) if rounds > INTERIM_ROUND else {}
    if (len(test) != 200 or test.keys() & interim.keys() or
            used.intersection(test.keys() | interim.keys()) or (interim and len(interim) != 200)):
        raise ValueError("event PPO needs distinct new confirmation rooms")
    base.ROOMS.update(test)
    base.ROOMS.update(interim)
    used.update(test)
    used.update(interim)
    schedule = list(range(662344701, 662344701+rounds))
    source_paths = (
        "tools/cap_abc_event_ppo.py",
        "tools/cap_abc_semantic_ppo.py",
        "tools/cap_option_semantic_factorial.py",
        "experiments/planar_tasks/semantic_rl/environment.py",
        "experiments/planar_tasks/semantic_rl/duration_execution.py",
        "experiments/planar_tasks/semantic_rl/push_phases.py",
    )
    repo = Path(__file__).resolve().parents[1]
    protocol = dict(
        arms=list(MODES), seed=long.SEED, parent_episode=long.END*long.BATCH,
        actor_sha=long.ppo.sha(actor), state_sha=long.ppo.sha(state),
        profile=plan["profile"], executor=plan["executor"], experts=plan["expert_sha"],
        time_ticks=time_ticks, progress_m=.20, contact_ticks=3, room_seeds=schedule,
        batch_size=long.BATCH, diagnostic=diagnostic, confirmation=sorted(test),
        interim_round=INTERIM_ROUND if interim else None, interim_confirmation=sorted(interim),
        reward="Signed distance change times 0.10, terminal success +1, same as parent",
        sources={path:long.ppo.sha(repo/path) for path in source_paths},
    )
    manifest = out/"protocol.json"
    if manifest.is_file():
        if base.read(manifest) != protocol:
            raise ValueError("existing protocol differs from requested comparison")
    else:
        save(manifest, protocol)
    learners = {mode:from_checkpoint(state) for mode in MODES}
    initial = actor_path(out, "original", 0)
    if not initial.is_file():
        export(initial, learners["original"][0])
    completed = {mode:restore_latest(out, mode, model, optimizer, rounds)
                 for mode,(model,optimizer) in learners.items()}
    experts = {name:plan["expert_sha"][name] for name in base.EXPERTS}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("fork")) as pool:
        for mode in MODES:
            evaluate(pool, out/"diagnostic"/mode/"r0000.json", mode, initial,
                     diagnostic, time_ticks)
        for i, seed in enumerate(schedule, 1):
            if i <= min(completed.values()):
                rooms = base.read(metrics_path(out, "original", i))["rooms"]
                assert len(rooms) == len(set(rooms)) == long.BATCH and used.isdisjoint(rooms)
                assert all(base.read(metrics_path(out, mode, i))["rooms"] == rooms for mode in MODES[1:])
                used.update(rooms)
                if i % 20 == 0 or i == rounds:
                    for mode in MODES:
                        evaluate(pool, out/"diagnostic"/mode/f"r{i:04d}.json", mode,
                                 actor_path(out, mode, i), diagnostic, time_ticks)
                if i == INTERIM_ROUND and interim:
                    evaluate_panel(pool, out, sorted(interim), i, initial, time_ticks, experts, interim=True)
                continue
            batch = long.sample(seed, used)
            room_keys = [key for key,_ in batch]
            stage = long.END + i
            started = time.perf_counter()
            active = {mode:[] for mode in MODES if i > completed[mode]}
            for j, (key, fixture) in enumerate(batch):
                for mode, futures in active.items():
                    job = (key, fixture, mode, stage, j, time_ticks, actor_path(out, mode, i-1))
                    futures.append(pool.submit(collect_row, job))
            for mode, (model, optimizer) in learners.items():
                if i <= completed[mode]:
                    assert base.read(metrics_path(out, mode, i))["rooms"] == room_keys
                    continue
                rows = [f.result() for f in active[mode]]
                stats = long.dense.fit(model, optimizer, rows, stage)
                counts = Counter()
                for row in rows:
                    counts.update(row[10])
                spans = np.concatenate([np.asarray(row[11], int) for row in rows])
                save_training_round(out, mode, i, model, optimizer, dict(
                    stats, rooms=room_keys, reviews=dict(counts),
                    decision_ticks_median=float(np.median(spans)),
                    wall_seconds=time.perf_counter()-started,
                ))
                print(f"{mode}: round {i}/{rounds}, {stats['success']}/{len(rows)}, {counts['switched']} switches", flush=True)
            if i % 20 == 0 or i == rounds:
                for mode in MODES:
                    evaluate(pool, out/"diagnostic"/mode/f"r{i:04d}.json", mode,
                             actor_path(out, mode, i), diagnostic, time_ticks)
            if i == INTERIM_ROUND and interim:
                evaluate_panel(pool, out, sorted(interim), i, initial, time_ticks, experts, interim=True)
        evaluate_panel(pool, out, sorted(test), rounds, initial, time_ticks, experts)
    save(out/"finished.json",dict(rounds=rounds, episodes_per_arm=rounds*long.BATCH,
                                  actors={mode:long.ppo.sha(actor_path(out, mode, rounds)) for mode in MODES}))


if __name__ == "__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--out", type=Path)
    p.add_argument("--rooms",type=int,default=24)
    p.add_argument("--workers",type=int,default=6)
    p.add_argument("--time-ticks",type=int)
    p.add_argument("--train-rounds", type=int, default=0)
    args=p.parse_args()
    torch.set_num_threads(1)
    if not 1 <= args.rooms <= 200 or args.workers < 1 or (args.time_ticks is not None and args.time_ticks < 1) or args.train_rounds < 0:
        p.error("require positive room, worker and time settings and a nonnegative training budget")
    if args.train_rounds:
        if args.out is None or args.time_ticks is None:
            p.error("--train-rounds requires an --out directory and an explicit --time-ticks")
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out/".driver.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            train_comparison(args.out, args.train_rounds, args.workers, args.time_ticks)
    else:
        if args.out is not None:
            OUT = args.out
        preflight(args.rooms,args.workers,args.time_ticks or 4)
