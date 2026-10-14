"""Recurrent PPO at the option level, with semi-Markov discounting.

Four things are kept strictly apart, because folding them together is how a variable-length option
framework silently produces wrong returns:

* **skill exit** -- the option's own terminal condition. Not an episode end, no bootstrap change.
* **task success / failure** -- a real terminal, so the return stops there.
* **task time limit** -- part of the task definition; treated as a real terminal.
* **rollout truncation** -- a buffer cut, which *does* bootstrap from the value estimate.

A skill of ``n`` control periods contributes ``sum_j gamma_step**j r_j`` and the next decision is
discounted by ``gamma_step ** n``; the shaping term is ``gamma_step**n * Phi(next) - Phi(current)``
and the same progress is never paid twice. Returns are computed **within** an episode's boundary
chain, so a truncated episode bootstraps and a terminated one does not.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_repair_optionrl_v1.option_env import (  # noqa: E402
    MENU,
    OptionRLConfig,
    OptionRLEnv,
    default_goal,
)
from experiments.night_repair_optionrl_v1.policy import RecurrentActorCritic  # noqa: E402
from coimagine.local_contact_option.roots import RootSpec, build_roots  # noqa: E402

OUT = ROOT / "outputs/night_repair_optionrl_v1/option_rl"


@dataclass
class Decision:
    """One boundary decision with everything needed to recompute its advantage."""

    boundary_obs_index: int
    action: int
    log_prob: float
    value: float
    reward: float          # the semi-Markov discounted reward collected inside the skill
    discount: float        # gamma_step ** n, applied to the next decision
    skill_steps: int


@dataclass
class Episode:
    decisions: List[Decision]
    observations: List[np.ndarray]
    terminated: bool
    truncated: bool
    success: bool
    root_id: str
    scene: str
    goal: Tuple[float, float]
    total_reward: float
    goal_distance_start: float
    goal_distance_end: float
    skill_count: int
    exit_reasons: List[str] = field(default_factory=list)


def legal_actions(root: Optional[RootSpec]) -> List[int]:
    """The public menu entries this scene can execute. Never filtered by hidden truth."""

    if root is None:
        return list(range(len(MENU)))
    active = set(getattr(root, "active_robots", (0, 1)))
    return [index for index, action in enumerate(MENU) if set(action.active).issubset(active)]


def collect_episode(env: OptionRLEnv, policy: RecurrentActorCritic, root: RootSpec,
                    goal: np.ndarray, *, rng: np.random.Generator, epsilon: float = 0.0,
                    deterministic: bool = False, max_skills: Optional[int] = None) -> Episode:
    """Run one episode. ``epsilon`` is the probability of a **random legal skill**, not action noise."""

    observation = env.reset(root, goal)
    allowed = legal_actions(root)
    decisions: List[Decision] = []
    observations: List[np.ndarray] = [observation]
    hidden = policy.initial_hidden(1)
    start_distance = float(np.hypot(env.privileged_object()[0] - goal[0],
                                    env.privileged_object()[1] - goal[1]))
    limit = max_skills or env.config.max_skills
    terminated = truncated = success = False
    total = 0.0
    exit_reasons: List[str] = []
    while True:
        with torch.no_grad():
            logits, value, hidden = policy.step(torch.tensor(observation[None, :]), hidden)
            logits = logits[0]
            mask = torch.full((len(MENU),), float("-inf"))
            mask[allowed] = 0.0
            masked = logits + mask
            probabilities = torch.softmax(masked, dim=-1)
            if deterministic:
                action = int(torch.argmax(masked))
            elif epsilon > 0.0 and rng.random() < epsilon:
                action = int(allowed[rng.integers(0, len(allowed))])
            else:
                action = int(torch.multinomial(probabilities, 1))
            log_prob = float(torch.log(probabilities[action].clamp(min=1e-12)))
        decisions.append(Decision(boundary_obs_index=len(observations) - 1, action=action,
                                  log_prob=log_prob, value=float(value),
                                  reward=0.0, discount=1.0, skill_steps=0))
        result = env.step(action)
        decisions[-1].reward = float(result.reward)
        decisions[-1].discount = float(result.info["discount"])
        decisions[-1].skill_steps = int(result.skill_steps)
        total += float(result.reward)
        if result.inner_observations.shape[0]:
            for row in result.inner_observations:
                observations.append(row)
        observations.append(result.observation)
        exit_reasons.append(str(env.boundaries[-1].get("exit_reason", "")))
        hidden = hidden.detach()
        if result.terminated:
            terminated, success = True, True
            break
        if result.truncated or len(decisions) >= limit:
            truncated = True
            break
        observation = result.observation
    end_distance = float(np.hypot(env.privileged_object()[0] - goal[0],
                                  env.privileged_object()[1] - goal[1]))
    return Episode(decisions=decisions, observations=observations, terminated=terminated,
                   truncated=truncated, success=success,
                   root_id=str(getattr(root, "root_id", "unknown")),
                   scene=str(getattr(root, "interaction_class", "unknown")),
                   goal=(float(goal[0]), float(goal[1])), total_reward=total,
                   goal_distance_start=start_distance, goal_distance_end=end_distance,
                   skill_count=len(decisions), exit_reasons=exit_reasons)


def discounted_returns(episode: Episode, bootstrap_value: float) -> np.ndarray:
    """Semi-Markov returns, computed inside the episode's own boundary chain."""

    returns = np.zeros(len(episode.decisions), dtype=np.float64)
    running = 0.0 if episode.terminated else bootstrap_value
    for index in range(len(episode.decisions) - 1, -1, -1):
        decision = episode.decisions[index]
        running = decision.reward + decision.discount * running
        returns[index] = running
    return returns


def evaluate(policy: RecurrentActorCritic, roots: Sequence[RootSpec], goal_distance: float, *,
             seed: int, epsilon: float = 0.0, deterministic: bool = False,
             max_skills: Optional[int] = None) -> Dict[str, object]:
    """Validation on fixed roots with a fixed action rule; exploration is off unless asked for."""

    rng = np.random.default_rng(seed)
    successes = 0
    distances, counts, rewards = [], [], []
    per_scene: Dict[str, Dict[str, float]] = {}
    for root in roots:
        env = OptionRLEnv(OptionRLConfig(seed=seed))
        episode = collect_episode(env, policy, root, default_goal(root, goal_distance), rng=rng,
                                  epsilon=epsilon, deterministic=deterministic,
                                  max_skills=max_skills)
        successes += int(episode.success)
        distances.append(episode.goal_distance_end - episode.goal_distance_start)
        counts.append(episode.skill_count)
        rewards.append(episode.total_reward)
        # the capability gap is scene-specific, so a pooled success rate would hide it
        scene = per_scene.setdefault(episode.scene, {"episodes": 0, "successes": 0,
                                                     "progress": [], "skills": []})
        scene["episodes"] += 1
        scene["successes"] += int(episode.success)
        scene["progress"].append(episode.goal_distance_end - episode.goal_distance_start)
        scene["skills"].append(episode.skill_count)
    for scene in per_scene.values():
        scene["success_rate"] = scene["successes"] / max(1, scene["episodes"])
        scene["mean_net_progress_m"] = float(np.mean(scene["progress"]))
        scene["median_skills"] = float(np.median(scene["skills"]))
        scene.pop("progress")
        scene.pop("skills")
    return {"episodes": len(roots), "successes": successes,
            "success_rate": successes / max(1, len(roots)),
            "mean_net_progress_m": float(np.mean(distances)),
            "median_skills": float(np.median(counts)),
            "mean_return": float(np.mean(rewards)),
            "per_scene": per_scene}


def update(policy: RecurrentActorCritic, optimiser, episodes: Sequence[Episode], *,
           epochs: int = 4, clip: float = 0.2, value_coef: float = 0.5, entropy_coef: float = 0.01,
           bootstrap_values: Sequence[float] = None, batch: int = 4, seed: int = 0) -> Dict[str, float]:
    """PPO with the recurrent state rebuilt by replaying each episode's observation sequence."""

    torch.manual_seed(seed)
    observations, boundaries, actions, old_log_probs, advantages, returns = [], [], [], [], [], []
    for position, episode in enumerate(episodes):
        values = discounted_returns(episode, bootstrap_values[position]
                                    if bootstrap_values else 0.0)
        baseline = np.asarray([d.value for d in episode.decisions])
        advantage = values - baseline
        if len(advantage) > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        observations.append(np.stack(episode.observations))
        boundaries.append([d.boundary_obs_index for d in episode.decisions])
        actions.append([d.action for d in episode.decisions])
        old_log_probs.append([d.log_prob for d in episode.decisions])
        advantages.append(advantage)
        returns.append(values)
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "updates": 0}
    order = np.arange(len(episodes))
    for _epoch in range(epochs):
        np.random.default_rng(seed + _epoch).shuffle(order)
        for start in range(0, len(order), batch):
            selection = order[start:start + batch]
            optimiser.zero_grad(set_to_none=True)
            policy_loss = torch.zeros(())
            value_loss = torch.zeros(())
            entropy_total = torch.zeros(())
            for index in selection:
                sequence = torch.tensor(observations[index][None, :, :], dtype=torch.float32)
                output = policy(sequence)
                logits = output["logits"][0]
                values = output["values"][0]
                boundary = torch.tensor(boundaries[index], dtype=torch.long)
                chosen = logits[boundary]
                log_probs = torch.log_softmax(chosen, dim=-1)
                action_tensor = torch.tensor(actions[index], dtype=torch.long)
                new_log_prob = log_probs[torch.arange(len(action_tensor)), action_tensor]
                old = torch.tensor(old_log_probs[index], dtype=torch.float32)
                ratio = torch.exp(new_log_prob - old)
                advantage = torch.tensor(advantages[index], dtype=torch.float32)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage
                policy_loss = policy_loss - torch.min(unclipped, clipped).mean()
                target = torch.tensor(returns[index], dtype=torch.float32)
                value_loss = value_loss + ((values[boundary] - target) ** 2).mean()
                entropy_total = entropy_total + torch.distributions.Categorical(
                    logits=chosen).entropy().mean()
            count = max(1, len(selection))
            loss = policy_loss / count + value_coef * value_loss / count \
                - entropy_coef * entropy_total / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            stats["policy_loss"] += float(policy_loss.detach()) / count
            stats["value_loss"] += float(value_loss.detach()) / count
            stats["entropy"] += float(entropy_total.detach()) / count
            stats["updates"] += 1
    for key in ("policy_loss", "value_loss", "entropy"):
        stats[key] /= max(1, stats["updates"])
    return stats


def train(*, iterations: int = 30, episodes_per_iteration: int = 8, hidden: int = 128,
          lr: float = 3e-4, goal_distance: float = 0.45, seed: int = 0, epsilon_train: float = 0.1,
          val_roots: int = 64, test_roots: int = 64, dev_roots: int = 32,
          max_skills: int = 12, out_dir: Path = OUT, budget_seconds: Optional[float] = None,
          log: bool = True) -> Dict[str, object]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    # roots are split by scenario family and index, never by frame; the same root's exploration
    # variants stay in one split
    # the split is by scenario family and root index, never by frame; a root and all of its
    # exploration variants live entirely inside one split
    dev = build_roots("dev", 6)[:dev_roots] if dev_roots else []
    pool = build_roots("test", 24)          # 6 classes x 24 = 144 roots
    validation = pool[:val_roots] if val_roots else []
    test = pool[val_roots:val_roots + test_roots] if test_roots else []
    overlap = {r.root_id for r in validation} & {r.root_id for r in test}
    assert not overlap, "validation and test roots must be disjoint"
    from experiments.night_repair_optionrl_v1.option_env import ObservationSpec
    policy = RecurrentActorCritic(ObservationSpec().width, len(MENU), hidden=hidden)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    history: List[Dict[str, object]] = []
    started = time.time()
    for iteration in range(1, iterations + 1):
        episodes: List[Episode] = []
        for _ in range(episodes_per_iteration):
            root = dev[int(rng.integers(0, len(dev)))]
            env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30)), max_skills=max_skills))
            episodes.append(collect_episode(env, policy, root, default_goal(root, goal_distance),
                                            rng=rng, epsilon=epsilon_train, max_skills=max_skills))
        bootstraps = []
        for episode in episodes:
            bootstraps.append(0.0 if episode.terminated else float(episode.decisions[-1].value)
                              if episode.decisions else 0.0)
        stats = update(policy, optimiser, episodes, bootstrap_values=bootstraps, seed=seed + iteration)
        if iteration % 5 == 0 or iteration == 1 or iteration == iterations:
            report = evaluate(policy, validation, goal_distance, seed=seed + iteration,
                              deterministic=True, max_skills=max_skills)
            entry = {"iteration": iteration, "explore_return": float(np.mean(
                [e.total_reward for e in episodes])),
                "explore_success": float(np.mean([e.success for e in episodes])),
                "val_success_rate": report["success_rate"],
                "val_net_progress_m": report["mean_net_progress_m"],
                "val_median_skills": report["median_skills"], **stats,
                "seconds": time.time() - started}
            history.append(entry)
            if log:
                print(f"  iter {iteration:3d} explore_return={entry['explore_return']:+.3f} "
                      f"val_success={entry['val_success_rate']:.3f} "
                      f"val_progress={entry['val_net_progress_m']:+.4f} "
                      f"({entry['seconds']:.0f}s)", flush=True)
        if budget_seconds and time.time() - started > budget_seconds:
            break
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = out_dir / f"policy_seed{seed}.pt"
    torch.save({"state_dict": policy.state_dict(), "observation_width": ObservationSpec().width,
                "action_count": len(MENU), "hidden": hidden, "seed": seed,
                "goal_distance": goal_distance, "history": history,
                "menu": [action.label() for action in MENU]}, checkpoint)
    result = {"checkpoint": str(checkpoint.relative_to(ROOT)), "iterations": len(history),
              "history": history, "seconds": time.time() - started,
              "dev_roots": len(dev), "val_roots": len(validation), "test_roots": len(test),
              "val": evaluate(policy, validation, goal_distance, seed=seed, deterministic=True,
                              max_skills=max_skills),
              "test": evaluate(policy, test, goal_distance, seed=seed, deterministic=True,
                               max_skills=max_skills)}
    return result


def random_baseline(roots: Sequence[RootSpec], goal_distance: float, *, seed: int = 0,
                    max_skills: int = 12) -> Dict[str, object]:
    """The comparison the task book asks for: the same task rules, a uniform legal policy."""

    rng = np.random.default_rng(seed)
    successes = 0
    progress, counts, rewards = [], [], []
    for root in roots:
        env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30)), max_skills=max_skills))
        goal = default_goal(root, goal_distance)
        env.reset(root, goal)
        allowed = legal_actions(root)
        total = 0.0
        start = float(np.hypot(env.privileged_object()[0] - goal[0],
                               env.privileged_object()[1] - goal[1]))
        used = 0
        while True:
            action = int(allowed[rng.integers(0, len(allowed))])
            result = env.step(action)
            total += result.reward
            used += 1
            if result.terminated:
                successes += 1
                break
            if result.truncated or used >= max_skills:
                break
        end = float(np.hypot(env.privileged_object()[0] - goal[0],
                             env.privileged_object()[1] - goal[1]))
        progress.append(end - start)
        counts.append(used)
        rewards.append(total)
    return {"episodes": len(roots), "successes": successes,
            "success_rate": successes / max(1, len(roots)),
            "mean_net_progress_m": float(np.mean(progress)),
            "median_skills": float(np.median(counts)), "mean_return": float(np.mean(rewards))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--random-baseline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.random_baseline:
        pool = build_roots("test", 24)
        report = {"validation": random_baseline(pool[:64], args.goal_distance, seed=args.seed),
                  "test": random_baseline(pool[64:128], args.goal_distance, seed=args.seed + 1)}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        (Path(args.out) / "random_baseline.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        print("RANDOM_BASELINE_OK")
        return 0
    result = train(iterations=args.iterations, episodes_per_iteration=args.episodes,
                   hidden=args.hidden, lr=args.lr, seed=args.seed,
                   goal_distance=args.goal_distance, budget_seconds=args.budget_seconds,
                   out_dir=Path(args.out))
    (Path(args.out) / f"train_result_seed{args.seed}.json").write_text(
        json.dumps(result, indent=1, default=float))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=1, default=float))
    print("PPO_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
