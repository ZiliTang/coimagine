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

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from coimagine.random_start import env as E, layout, spec  # noqa: E402

CONTROL_DT = spec.CONTROL_DT
ACTION_KINDS = tuple(action.kind for action in spec.ROBOT_ACTIONS)
GAMMA_STEP = 0.99 ** CONTROL_DT
HIDDEN = 256
MAX_SKILLS = 24


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
    def __init__(self, obs_dim: int, hidden: int = HIDDEN, act: int = spec.ROBOT_ACTION_COUNT):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh())
        self.pi0 = nn.Linear(hidden, act)
        self.pi1 = nn.Linear(hidden, act)
        self.value = nn.Linear(hidden, 1)
        # the two heads share the trunk so each robot can use the other's view (the joint state)
        for head in (self.pi0, self.pi1):
            nn.init.orthogonal_(head.weight, 0.01)

    def forward(self, x: torch.Tensor):
        h = self.body(x)
        return self.pi0(h), self.pi1(h), self.value(h).squeeze(-1)


@dataclass
class Step:
    obs: np.ndarray
    a0: int
    a1: int
    logp: float
    value: float
    reward: float
    discount: float


def action_mask(observation, action_kinds) -> np.ndarray:
    """Public feasibility mask: a push/reposition needs a visible patch; wait/follow always fit."""
    mask = np.asarray([kind in ("wait", "follow") for kind in action_kinds], dtype=bool)
    if observation is not None and getattr(observation, "patch", None) is not None:
        mask[:] = True
    return mask


def observe_vector(env, fraction: float = 0.0):
    """Returns (vector, mask0, mask1) from the robot's own legal observations."""
    obs0 = env.observe(0, env.total_periods, env.total_periods * CONTROL_DT)
    obs1 = env.observe(1, env.total_periods, env.total_periods * CONTROL_DT)
    """The legal 760-dim joint view; ``fraction`` is unused (kept for call compatibility)."""
    obs0 = env.observe(0, env.total_periods, env.total_periods * CONTROL_DT)
    obs1 = env.observe(1, env.total_periods, env.total_periods * CONTROL_DT)
    progress = {}
    if env.rows:
        last = env.rows[-1]
        for robot in (0, 1):
            info = last.get(f"r{robot}", {})
            progress[robot] = (float(info.get("signed_progress") or 0.0),
                               float(info.get("progress_target_distance") or 0.0))
    mask0 = action_mask(obs0, ACTION_KINDS)
    mask1 = action_mask(obs1, ACTION_KINDS)
    return np.asarray(env.observation_vector(obs0, obs1, progress), dtype=np.float32), mask0, mask1


def run_episode(policy: Policy, instance, rng: np.random.Generator,
                obs_stats: RunningMeanStd) -> Tuple[List[Step], Dict]:
    env = E.RandomStartEnv(instance)
    initial = max(env.object_distance(), 1e-6)
    progress = 0.0
    obs, mask0, mask1 = observe_vector(env, progress)
    steps: List[Step] = []
    total_reward = 0.0
    components = {key: 0.0 for key in
                  ("progress", "distance_cost", "time_cost", "success_bonus", "failure_penalty")}
    success = False
    for _ in range(MAX_SKILLS):
        obs_stats.update(obs[None, :])
        x = torch.as_tensor(obs_stats.normalise(obs)).float().unsqueeze(0)
        with torch.no_grad():
            logits0, logits1, value = policy(x)
            logits0 = logits0.masked_fill(~torch.as_tensor(mask0), -1e9)
            logits1 = logits1.masked_fill(~torch.as_tensor(mask1), -1e9)
            d0 = torch.distributions.Categorical(logits=logits0)
            d1 = torch.distributions.Categorical(logits=logits1)
            a0, a1 = int(d0.sample()), int(d1.sample())
            logp = float(d0.log_prob(torch.tensor(a0)) + d1.log_prob(torch.tensor(a1)))
        boundary = env.joint_step(a0, a1, record=False)
        periods = int(boundary["periods"])
        reward = float(boundary["reward"]["total"])
        success = bool(boundary["success"])
        total_reward += reward
        for key in components:
            components[key] += float(boundary["reward"].get(key, 0.0))
        steps.append(Step(obs=obs_stats.normalise(obs), a0=a0, a1=a1, logp=logp,
                          value=float(value.item()), reward=reward,
                          discount=GAMMA_STEP ** periods))
        if boundary["termination"] is not None:
            break
        progress = float(np.clip((initial - env.object_distance()) / initial, -1.0, 1.0))
        obs, mask0, mask1 = observe_vector(env, progress)
    return steps, {"return": total_reward, "success": success,
                   "progress": float(np.clip((initial - env.object_distance()) / initial, -1.0, 1.0)),
                   "components": components}


def returns_and_advantages(steps: List[Step]) -> Tuple[np.ndarray, np.ndarray]:
    returns = np.zeros(len(steps), dtype=np.float32)
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        running = steps[index].reward + steps[index].discount * running
        returns[index] = running
    values = np.asarray([step.value for step in steps], dtype=np.float32)
    return returns, returns - values


def train(*, iterations: int, episodes_per_iteration: int, hours: float, seed: int,
          out_dir: Path, eval_roots: int, lr: float = 3e-4, epochs: int = 4,
          minibatch: int = 256, clip: float = 0.2, value_coef: float = 0.5,
          entropy_coef: float = 0.01, eval_every: int = 5) -> Dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train_instances = layout.build_split("L0", "train", count=64)[0]
    eval_instances = (layout.build_split("L0", "test", count=eval_roots)[0] if eval_roots else [])
    obs_dim = len(observe_vector(E.RandomStartEnv(train_instances[0]), 0.0)[0])
    obs_stats = RunningMeanStd(obs_dim)
    policy = Policy(obs_dim)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    history: List[Dict] = []

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
            obs, mask0, mask1 = observe_vector(env, 0.0)
            success = False
            for _ in range(MAX_SKILLS):
                x = torch.as_tensor(obs_stats.normalise(obs)).float().unsqueeze(0)
                with torch.no_grad():
                    logits0, logits1, _ = policy(x)
                    logits0 = logits0.masked_fill(~torch.as_tensor(mask0), -1e9)
                    logits1 = logits1.masked_fill(~torch.as_tensor(mask1), -1e9)
                    a0, a1 = int(torch.argmax(logits0)), int(torch.argmax(logits1))
                boundary = env.joint_step(a0, a1, record=False)
                success = bool(boundary["success"])
                if boundary["termination"] is not None:
                    break
                obs, mask0, mask1 = observe_vector(env, float(np.clip(
                    (initial - env.object_distance()) / initial, -1.0, 1.0)))
            successes.append(float(success))
            progresses.append(float(np.clip((initial - env.object_distance()) / initial, -1.0, 1.0)))
        return {"success_rate": float(np.mean(successes)),
                "mean_progress": float(np.mean(progresses))}

    for iteration in range(1, iterations + 1):
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
        old_logp = torch.tensor([step.logp for step in steps], dtype=torch.float32)
        ret_t = torch.as_tensor(returns)
        adv_t = torch.as_tensor(advantages)
        n = len(steps)
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
        for _ in range(epochs):
            order = torch.randperm(n)
            for start in range(0, n, minibatch):
                idx = order[start:start + minibatch]
                logits0, logits1, value = policy(obs_t[idx])
                d0 = torch.distributions.Categorical(logits=logits0)
                d1 = torch.distributions.Categorical(logits=logits1)
                logp = d0.log_prob(a0_t[idx]) + d1.log_prob(a1_t[idx])
                ratio = torch.exp(logp - old_logp[idx])
                unclipped = ratio * adv_t[idx]
                clipped = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[idx]
                policy_loss = -torch.min(unclipped, clipped).mean()
                value_loss = (value - ret_t[idx]).pow(2).mean()
                entropy = d0.entropy().mean() + d1.entropy().mean()
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
        if iteration % eval_every == 0 or iteration == 1:
            evaluation = evaluate()
            components = {f"reward_{key}": float(np.mean([e["components"][key] for _, e in episodes]))
                          for key in ("progress", "distance_cost", "time_cost",
                                      "success_bonus", "failure_penalty")}
            entry = {"iteration": iteration,
                     "explore_return": float(np.mean([e["return"] for _, e in episodes])),
                     "explore_success": float(np.mean([e["success"] for _, e in episodes])),
                     "explore_progress": float(np.mean([e["progress"] for _, e in episodes])),
                     **evaluation, **stats, **components,
                     "steps": n, "seconds": time.time() - started}
            history.append(entry)
            save(out_dir / "checkpoint.pt")
            try:
                from experiments.option_rl_v45.monitor import write_dashboard
                write_dashboard(history, out_dir / "dashboard.html", subtitle=str(out_dir))
            except Exception as error:      # the dashboard must never stop a training run
                print(f"dashboard skipped: {error}", flush=True)
            print(f"iter {iteration:4d} return={entry['explore_return']:+.3f} "
                  f"success={entry['explore_success']:.3f} progress={entry['explore_progress']:+.3f} "
                  f"val_success={entry['success_rate']} ({entry['seconds']:.0f}s)", flush=True)
        if time.time() - started >= hours * 3600.0:
            print(f"time budget reached at iteration {iteration}", flush=True)
            break
        (out_dir / "history.json").write_text(json.dumps(history, indent=1) + "\n")
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
    parser.add_argument("--out", type=Path, default=ROOT / "outputs/option_rl_v45/run")
    args = parser.parse_args()
    train(iterations=args.iterations, episodes_per_iteration=args.episodes, hours=args.hours,
          seed=args.seed, out_dir=args.out, eval_roots=args.eval_roots,
          eval_every=args.eval_every)


if __name__ == "__main__":
    main()
