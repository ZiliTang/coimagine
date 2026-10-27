"""Option-level PPO for the frozen v4.5 random-start contract.

The trainer only reads the environment's public API: ``observation_vector`` (the legal 760-dim
joint view), ``joint_step`` (boundary reward / periods / termination / success). The return is
semi-Markov: a skill of n control periods contributes its dense reward ``r`` and the next
decision is discounted by ``gamma_step ** n``. A skill exit is *not* an episode end; only the
task termination (success / failure / time budget) stops the return. Nothing privileged is fed
to the policy: the observation is exactly the env's legal vector.

Usage:
    python experiments/option_rl_v45/train.py --iterations 2 --episodes 2 --eval-roots 2
    python experiments/option_rl_v45/train.py --hours 4.5 --out outputs/option_rl_v45/run1
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coimagine.random_start import env as E, layout, spec  # noqa: E402

CONTROL_DT = spec.CONTROL_DT
ACTION_KINDS = tuple(action.kind for action in spec.ROBOT_ACTIONS)
#: factorised action: the type one-hot (7) and the scale one-hot (4) are separate policy
#: heads, exactly the contract's ``index = type * SCALE_COUNT + scale``. Exploration is 7 + 4
#: decisions per robot instead of a flat 28-way softmax, while every PUSH/REPOSITION distance
#: (including the 1.0 m long push and the 1.0 m reposition) stays available.
TYPE_COUNT = spec.ROBOT_ACTION_TYPE_COUNT
SCALE_COUNT = spec.OPTION_SCALE_COUNT


def type_mask(mask28) -> np.ndarray:
    """A type is feasible when at least one of its scale entries is feasible."""
    mask28 = np.asarray(mask28, dtype=bool)
    return np.asarray([bool(mask28[t * SCALE_COUNT:(t + 1) * SCALE_COUNT].any())
                       for t in range(TYPE_COUNT)], dtype=bool)


GAMMA_STEP = 0.99 ** CONTROL_DT
HIDDEN = 256
#: Hard safety cap only. The task budget is 90 s and the shortest option is 2 s, so a normal
#: episode terminates far below this. Previously the runner stopped silently at 24 skills and
#: skipped the env's failure penalty, so ``WAIT[2 s] x 24`` ended at 48 s for roughly -2.9
#: while every honest 90 s failure cost about -7.4: early exit was cheaper than the task
#: budget. Hitting the cap now charges the truncation failure explicitly, so no policy can
#: buy return by ending the episode early.
MAX_SKILLS = 120
TRUNCATION_PENALTY = spec.FAILURE_PENALTY


class RunningMeanStd:
    """The standard observation normaliser; frozen at evaluation time."""

    def __init__(self, dim: int):
        self.mean = np.zeros(dim, dtype=np.float64)
        self.var = np.ones(dim, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64)
        batch_mean, batch_var = batch.mean(axis=0), batch.var(axis=0)
        count = batch.shape[0]
        delta = batch_mean - self.mean
        total = self.count + count
        self.mean = self.mean + delta * count / total
        m2 = self.var * self.count + batch_var * count + delta ** 2 * self.count * count / total
        self.var = m2 / total
        self.count = total

    def normalise(self, x: np.ndarray) -> np.ndarray:
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10.0, 10.0).astype(np.float32)


class Policy(nn.Module):
    """Two factorised heads per robot: the option type (7) and its scale (4)."""

    def __init__(self, obs_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.type0 = nn.Linear(hidden, TYPE_COUNT)
        self.scale0 = nn.Linear(hidden, SCALE_COUNT)
        self.type1 = nn.Linear(hidden, TYPE_COUNT)
        self.scale1 = nn.Linear(hidden, SCALE_COUNT)
        self.value = nn.Linear(hidden, 1)
        # the heads share the trunk so each robot can use the other's view (the joint state)
        for head in (self.type0, self.scale0, self.type1, self.scale1):
            nn.init.orthogonal_(head.weight, 0.01)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return (self.type0(h), self.scale0(h)), (self.type1(h), self.scale1(h)), \
               self.value(h).squeeze(-1)


@dataclass
class Step:
    obs: np.ndarray
    a0: int              # env action index = type * SCALE_COUNT + scale
    a1: int
    logp: float
    value: float
    reward: float
    discount: float
    mask0: np.ndarray
    mask1: np.ndarray
    terminal: bool = False


def action_mask(observation, action_kinds) -> np.ndarray:
    """Public feasibility mask: a push/reposition needs a visible patch; wait/follow always fit."""
    mask = np.asarray([kind in ("wait", "follow") for kind in action_kinds], dtype=bool)
    if observation is not None and getattr(observation, "patch", None) is not None:
        mask[:] = True
    return mask


def observe_vector(env):
    """Returns (vector, mask0, mask1) from the robot's own legal observations."""
    obs0 = env.observe(0, env.total_periods, env.total_periods * CONTROL_DT)
    obs1 = env.observe(1, env.total_periods, env.total_periods * CONTROL_DT)
    # ``env._last_progress`` is updated on every control period inside ``joint_step``, even
    # when ``record=False``. Reading ``env.rows`` here used to feed zeros to both progress
    # columns because the training runner never records, so pass ``None`` and let the env use
    # its own legal per-robot signed progress / target.
    mask0 = action_mask(obs0, ACTION_KINDS)
    mask1 = action_mask(obs1, ACTION_KINDS)
    return np.asarray(env.observation_vector(obs0, obs1, None), dtype=np.float32), mask0, mask1


def run_episode(policy: Policy, instance, rng: np.random.Generator,
                obs_stats: RunningMeanStd) -> Tuple[List[Step], Dict]:
    env = E.RandomStartEnv(instance)
    initial = max(env.object_distance(), 1e-6)
    obs, mask0, mask1 = observe_vector(env)
    steps: List[Step] = []
    total_reward = 0.0
    components = {key: 0.0 for key in
                  ("progress", "distance_cost", "time_cost", "success_bonus", "failure_penalty")}
    success = False
    terminated = False
    for _ in range(MAX_SKILLS):
        obs_stats.update(obs[None, :])
        x = torch.as_tensor(obs_stats.normalise(obs)).float().unsqueeze(0)
        tm0, tm1 = type_mask(mask0), type_mask(mask1)
        with torch.no_grad():
            (ty0, sc0), (ty1, sc1), value = policy(x)
            d_ty0 = torch.distributions.Categorical(logits=ty0.masked_fill(~torch.as_tensor(tm0), -1e9))
            d_sc0 = torch.distributions.Categorical(logits=sc0)
            d_ty1 = torch.distributions.Categorical(logits=ty1.masked_fill(~torch.as_tensor(tm1), -1e9))
            d_sc1 = torch.distributions.Categorical(logits=sc1)
            t0, s0 = int(d_ty0.sample()), int(d_sc0.sample())
            t1, s1 = int(d_ty1.sample()), int(d_sc1.sample())
            a0, a1 = t0 * SCALE_COUNT + s0, t1 * SCALE_COUNT + s1
            logp = float(d_ty0.log_prob(torch.tensor(t0)) + d_sc0.log_prob(torch.tensor(s0))
                         + d_ty1.log_prob(torch.tensor(t1)) + d_sc1.log_prob(torch.tensor(s1)))
        boundary = env.joint_step(a0, a1, record=False)
        periods = int(boundary["periods"])
        reward = float(boundary["reward"]["total"])
        success = bool(boundary["success"])
        total_reward += reward
        for key in components:
            components[key] += float(boundary["reward"].get(key, 0.0))
        terminated = boundary["termination"] is not None
        steps.append(Step(obs=obs_stats.normalise(obs), a0=a0, a1=a1, logp=logp,
                          value=float(value.item()), reward=reward,
                          discount=GAMMA_STEP ** periods,
                          mask0=tm0, mask1=tm1, terminal=terminated))
        if terminated:
            break
        obs, mask0, mask1 = observe_vector(env)
    exit_reason = env.termination
    if not terminated:
        # Loop guard only: never let the runner end an episode uncharged. The skill cap is
        # far above what a 90 s budget can reach with 2 s options, so this branch is a safety
        # net; if it ever fires, the missing task-budget failure is charged here.
        assert steps, "an episode must contain at least one skill"
        steps[-1].reward += TRUNCATION_PENALTY
        steps[-1].terminal = True
        total_reward += TRUNCATION_PENALTY
        components["failure_penalty"] += TRUNCATION_PENALTY
        exit_reason = "trainer_skill_cap"
    return steps, {"return": total_reward, "success": success,
                   "progress": float(np.clip((initial - env.object_distance()) / initial, -1.0, 1.0)),
                   "exit_reason": exit_reason, "components": components}


def returns_and_advantages(steps: List[Step]) -> Tuple[np.ndarray, np.ndarray]:
    returns = np.zeros(len(steps), dtype=np.float32)
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        if steps[index].terminal:
            # a terminal step has no successor: this also resets the accumulator at every
            # episode boundary, so one episode's return cannot leak into the previous one
            running = steps[index].reward
        else:
            running = steps[index].reward + steps[index].discount * running
        returns[index] = running
    values = np.asarray([step.value for step in steps], dtype=np.float32)
    return returns, returns - values


def warm_start(policy, obs_stats, instances, steps, rng, batch=128, lr=1e-3):
    """Behaviour-cloning initialisation from the *legal* goal heuristic.

    The heuristic reads only the robot's own observation (patch normal + goal direction) and the
    public menu; it is an initialisation, never an oracle input to the policy.
    """
    from experiments.option_rl_v45.fixed_policies import _goal_action
    samples = []
    for instance in instances:
        env = E.RandomStartEnv(instance)
        for _ in range(8):
            observations = env.boundary_observations()
            a0 = int(_goal_action(observations[0]))
            a1 = int(_goal_action(observations[1]))
            obs, _m0, _m1 = observe_vector(env)
            samples.append((obs, a0, a1))
            env.joint_step(a0, a1, record=False)
            if env.termination is not None:
                break
    if not samples:
        return 0
    obs_stats.update(np.stack([sample[0] for sample in samples]))
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    for _ in range(steps):
        index = rng.integers(0, len(samples), size=min(batch, len(samples)))
        x = torch.as_tensor(np.stack([obs_stats.normalise(samples[i][0]) for i in index])).float()
        a0 = torch.tensor([samples[i][1] for i in index])
        a1 = torch.tensor([samples[i][2] for i in index])
        (ty0, sc0), (ty1, sc1), _ = policy(x)
        loss = (F.cross_entropy(ty0, a0 // SCALE_COUNT) + F.cross_entropy(sc0, a0 % SCALE_COUNT)
                + F.cross_entropy(ty1, a1 // SCALE_COUNT) + F.cross_entropy(sc1, a1 % SCALE_COUNT))
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    return len(samples)


def train(*, iterations: int, episodes_per_iteration: int, hours: float, seed: int,
          out_dir: Path, eval_roots: int, simple_scene: bool = False, easy_goal=None,
          anneal_iterations: int = 0, bc_steps: int = 0, lr: float = 3e-4, epochs: int = 4,
          minibatch: int = 256, clip: float = 0.2, value_coef: float = 0.5,
          entropy_coef: float = 0.01, eval_every: int = 5) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_instances = layout.build_split("L0", "train", count=64)[0]
    if simple_scene:
        from experiments.option_rl_v45.scenes import sample_scenes
        train_instances = sample_scenes(seed * 1000 + 1, 64, "train", easy_goal)
        # the evaluation always uses the full random goal, never the curriculum distribution
        eval_instances = (sample_scenes(seed * 1000 + 2, eval_roots, "test") if eval_roots else [])
    else:
        train_instances = layout.build_split("L0", "train", count=64)[0]
        eval_instances = (layout.build_split("L0", "test", count=eval_roots)[0] if eval_roots else [])
    obs_dim = len(observe_vector(E.RandomStartEnv(train_instances[0]))[0])
    obs_stats = RunningMeanStd(obs_dim)
    policy = Policy(obs_dim)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history: List[Dict] = []
    if bc_steps > 0:
        count = warm_start(policy, obs_stats, train_instances, bc_steps, rng)
        print(f"warm start: {count} heuristic samples, {bc_steps} steps", flush=True)

    def save(path: Path) -> None:
        torch.save({"state_dict": policy.state_dict(), "obs_dim": obs_dim,
                    "obs_mean": obs_stats.mean, "obs_var": obs_stats.var,
                    "action_count": spec.ROBOT_ACTION_COUNT, "hidden": HIDDEN,
                    "seed": seed, "history": history}, path)

    def evaluate() -> Dict:
        if not eval_instances:
            return {"success_rate": None, "mean_progress": None}
        successes, progresses = [], []
        for instance in eval_instances:
            env = E.RandomStartEnv(instance)
            initial = max(env.object_distance(), 1e-6)
            obs, mask0, mask1 = observe_vector(env)
            success = False
            for _ in range(MAX_SKILLS):
                x = torch.as_tensor(obs_stats.normalise(obs)).float().unsqueeze(0)
                with torch.no_grad():
                    (ty0, sc0), (ty1, sc1), _ = policy(x)
                    ty0 = ty0.masked_fill(~torch.as_tensor(type_mask(mask0)), -1e9)
                    ty1 = ty1.masked_fill(~torch.as_tensor(type_mask(mask1)), -1e9)
                    a0 = int(torch.argmax(ty0)) * SCALE_COUNT + int(torch.argmax(sc0))
                    a1 = int(torch.argmax(ty1)) * SCALE_COUNT + int(torch.argmax(sc1))
                boundary = env.joint_step(a0, a1, record=False)
                success = bool(boundary["success"])
                if boundary["termination"] is not None:
                    break
                obs, mask0, mask1 = observe_vector(env)
            successes.append(float(success))
            progresses.append(float(np.clip((initial - env.object_distance()) / initial, -1.0, 1.0)))
        return {"success_rate": float(np.mean(successes)),
                "mean_progress": float(np.mean(progresses))}

    for iteration in range(1, iterations + 1):
        if simple_scene and anneal_iterations > 0:
            # annealing curriculum: the goal distance grows towards the full arena so the
            # policy does not overfit the near-goal distribution; the evaluation stays full
            from experiments.option_rl_v45.scenes import sample_scenes
            fraction = min(1.0, iteration / float(anneal_iterations))
            max_distance = 0.85 + 2.2 * fraction
            train_instances = sample_scenes(seed * 1000 + 17 + iteration, 64, "train",
                                            (0.35, max_distance))
        episodes = []
        for _ in range(episodes_per_iteration):
            instance = train_instances[int(rng.integers(0, len(train_instances)))]
            episodes.append(run_episode(policy, instance, rng, obs_stats))
        steps = [step for episode, _ in episodes for step in episode]
        returns, advantages = returns_and_advantages(steps)
        if len(steps) >= 4 and float(advantages.std()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        obs_t = torch.as_tensor(np.stack([step.obs for step in steps]))
        a0_t = torch.tensor([step.a0 for step in steps])
        a1_t = torch.tensor([step.a1 for step in steps])
        mask0_t = torch.as_tensor(np.stack([step.mask0 for step in steps]))
        mask1_t = torch.as_tensor(np.stack([step.mask1 for step in steps]))
        old_logp = torch.tensor([step.logp for step in steps], dtype=torch.float32)
        ret_t = torch.as_tensor(returns)
        adv_t = torch.as_tensor(advantages)
        n = len(steps)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, minibatch):
                idx = order[start:start + minibatch]
                (ty0, sc0), (ty1, sc1), value = policy(obs_t[idx])
                d_ty0 = torch.distributions.Categorical(
                    logits=ty0.masked_fill(~mask0_t[idx], -1e9))
                d_sc0 = torch.distributions.Categorical(logits=sc0)
                d_ty1 = torch.distributions.Categorical(
                    logits=ty1.masked_fill(~mask1_t[idx], -1e9))
                d_sc1 = torch.distributions.Categorical(logits=sc1)
                t0_l, s0_l = a0_t[idx] // SCALE_COUNT, a0_t[idx] % SCALE_COUNT
                t1_l, s1_l = a1_t[idx] // SCALE_COUNT, a1_t[idx] % SCALE_COUNT
                logp = (d_ty0.log_prob(t0_l) + d_sc0.log_prob(s0_l)
                        + d_ty1.log_prob(t1_l) + d_sc1.log_prob(s1_l))
                ratio = torch.exp(logp - old_logp[idx])
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - ret_t[idx]).pow(2).mean()
                entropy = (d_ty0.entropy().mean() + d_sc0.entropy().mean()
                           + d_ty1.entropy().mean() + d_sc1.entropy().mean())
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                optimiser.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                stats["policy_loss"] += float(policy_loss.detach())
                stats["value_loss"] += float(value_loss.detach())
                stats["entropy"] += float(entropy.detach())
                stats["updates"] += 1
        for key in ("policy_loss", "value_loss", "entropy"):
            stats[key] /= max(1, stats["updates"])
        components = {f"reward_{key}": float(np.mean([e["components"][key] for _, e in episodes]))
                      for key in ("progress", "distance_cost", "time_cost",
                                  "success_bonus", "failure_penalty")}
        entry = {"iteration": iteration,
                 "explore_return": float(np.mean([e["return"] for _, e in episodes])),
                 "explore_success": float(np.mean([e["success"] for _, e in episodes])),
                 "explore_progress": float(np.mean([e["progress"] for _, e in episodes])),
                 "success_rate": None, "mean_progress": None, **stats, **components,
                 "steps": n, "seconds": time.time() - started}
        if iteration % eval_every == 0 or iteration == 1:
            entry.update(evaluate())
            save(out_dir / "checkpoint.pt")
        history.append(entry)
        # persist every iteration so an interrupted run keeps the dense record
        (out_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
        try:
            from experiments.option_rl_v45.monitor import write_dashboard
            write_dashboard(history, out_dir / "dashboard.html", subtitle=str(out_dir))
        except Exception as error:      # the dashboard must never stop a training run
            print(f"dashboard skipped: {error}", flush=True)
        if entry["success_rate"] is not None:
            print(f"iter {iteration:4d} return={entry['explore_return']:+.3f} "
                  f"success={entry['explore_success']:.3f} progress={entry['explore_progress']:+.3f} "
                  f"val_success={entry['success_rate']} val_prog={entry['mean_progress']:+.3f} "
                  f"({entry['seconds']:.0f}s)", flush=True)
        if time.time() - started >= hours * 3600.0:
            print(f"time budget reached at iteration {iteration}", flush=True)
            break
    checkpoint = out_dir / "policy.pt"
    save(checkpoint)
    final = evaluate()
    (out_dir / "result.json").write_text(json.dumps(
        {"checkpoint": str(checkpoint), "iterations": len(history), "seconds": time.time() - started,
         "final": final, "history": history}, indent=1) + "\n")
    print(f"done: {checkpoint} final={final}", flush=True)
    return {"checkpoint": str(checkpoint), "final": final, "history": history}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--hours", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-roots", type=int, default=16)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--entropy", type=float, default=0.01,
                        help="entropy coefficient (0.05-0.1 recommended until PUSH appears)")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--easy-goal", action="store_true",
                        help="curriculum: train with the goal centre 0.35-0.85 m from the object; "
                             "the evaluation still uses the full random goal")
    parser.add_argument("--anneal-iterations", type=int, default=0,
                        help="annealing curriculum: grow the train goal distance over this many "
                             "iterations (0 = fixed --easy-goal)")
    parser.add_argument("--bc-steps", type=int, default=0,
                        help="behaviour-cloning warm start from the legal goal heuristic")
    parser.add_argument("--simple-scene", action="store_true",
                        help="one empty 4x3 arena, random starts and a random goal centre away "
                             "from the walls (the training scenario recommended for v4.6)")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/option_rl_v45/run")
    args = parser.parse_args()
    train(iterations=args.iterations, episodes_per_iteration=args.episodes, hours=args.hours,
          seed=args.seed, out_dir=args.out, eval_roots=args.eval_roots,
          simple_scene=args.simple_scene, eval_every=args.eval_every,
          easy_goal=(0.35, 0.85) if args.easy_goal else None,
          anneal_iterations=args.anneal_iterations, bc_steps=args.bc_steps,
          entropy_coef=args.entropy, lr=args.lr)


if __name__ == "__main__":
    main()
