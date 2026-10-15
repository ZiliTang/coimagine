"""Recurrent PPO on the corrected rollout.

On-policy sampling only (``epsilon = 0``): the update recomputes the action distribution through the
same ``rollout``/``advance`` path and the same mask, so the importance ratio is exactly one before
any parameter change. The epsilon mixture is a **data-collection** tool and is not used for training.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_repair_optionrl_v1.option_env import (  # noqa: E402
    MENU,
    OptionRLConfig,
    OptionRLEnv,
    ObservationSpec,
    default_goal,
)
from experiments.night_repair2_v1.rl import (  # noqa: E402
    BOOTSTRAPS,
    Episode,
    RecurrentPolicy,
    legal_actions,
    masked_logits,
    returns,
    rollout,
)
from coimagine.local_contact_option.roots import RootSpec, build_roots  # noqa: E402

OUT = ROOT / "outputs/night_repair2_v1/option_rl"


def collect(policy: RecurrentPolicy, roots: Sequence[RootSpec], goal_distance: float, *,
            seed: int, count: Optional[int] = None, deterministic: bool = False,
            epsilon: float = 0.0, max_skills: int = 12) -> List[Episode]:
    rng = np.random.default_rng(seed)
    episodes: List[Episode] = []
    pool = list(roots)
    if count is not None:
        picks = rng.integers(0, len(pool), size=count)
    else:
        picks = np.arange(len(pool))
    for pick in picks:
        root = pool[int(pick)]
        env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30)), max_skills=max_skills))
        episodes.append(rollout(env, policy, root, default_goal(root, goal_distance), rng=rng,
                                epsilon=epsilon, deterministic=deterministic, max_skills=max_skills))
    return episodes


def summarise(episodes: Sequence[Episode]) -> Dict[str, object]:
    """Every quantity the task book asks for, including both signs of the distance change."""

    if not episodes:
        return {"episodes": 0}
    per_scene: Dict[str, List[Episode]] = {}
    for episode in episodes:
        per_scene.setdefault(episode.scene, []).append(episode)
    outcomes: Dict[str, int] = {}
    for episode in episodes:
        outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1
    return {
        "episodes": len(episodes),
        "success_rate": float(np.mean([e.success for e in episodes])),
        "mean_end_distance_m": float(np.mean([e.end_distance for e in episodes])),
        "mean_start_distance_m": float(np.mean([e.start_distance for e in episodes])),
        "mean_progress_towards_goal_m": float(np.mean(
            [e.net_progress_towards_goal for e in episodes])),
        "mean_distance_change_m": float(np.mean([e.net_distance_change for e in episodes])),
        "median_skills": float(np.median([len(e.decisions) for e in episodes])),
        "mean_return": float(np.mean([e.total_reward for e in episodes])),
        "outcomes": outcomes,
        "per_scene": {scene: {
            "episodes": len(group),
            "success_rate": float(np.mean([e.success for e in group])),
            "mean_progress_towards_goal_m": float(np.mean(
                [e.net_progress_towards_goal for e in group])),
            "mean_end_distance_m": float(np.mean([e.end_distance for e in group]))}
            for scene, group in sorted(per_scene.items())},
    }


def ppo_update(policy: RecurrentPolicy, optimiser, episodes: Sequence[Episode], *,
               epochs: int = 4, clip: float = 0.2, value_coef: float = 0.5,
               entropy_coef: float = 0.01, seed: int = 0) -> Dict[str, float]:
    """One PPO epoch set. The ratio is computed from the same masked distribution as sampling."""

    observations = [torch.tensor(e.observations, dtype=torch.float32) for e in episodes]
    # the importance ratio must be exactly one *before* any parameter change; measuring it inside
    # the update loop would mix in the steps already taken for earlier episodes
    pre_update_ratios = []
    with torch.no_grad():
        for index, episode in enumerate(episodes):
            hidden = policy.initial_hidden(1)
            consumed = 0
            for decision in episode.decisions:
                target = decision.buffer_index + 1
                hidden = policy.advance(hidden, observations[index][consumed:target])
                consumed = target
                logits, _value = policy.heads(hidden)
                log_probs = torch.log_softmax(masked_logits(logits[0], decision.allowed), dim=-1)
                pre_update_ratios.append(float(torch.exp(log_probs[decision.action]
                                                         - decision.policy_logprob)))
    stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0, "updates": 0,
             "mean_ratio_before_update": float(np.mean(pre_update_ratios)),
             "max_abs_ratio_minus_one": float(np.max(np.abs(np.asarray(pre_update_ratios) - 1.0)))}
    for epoch in range(epochs):
        for index, episode in enumerate(episodes):
            values = returns(episode)
            baseline = np.asarray([d.value for d in episode.decisions])
            advantage = values - baseline
            if len(advantage) > 1 and advantage.std() > 1e-8:
                advantage = (advantage - advantage.mean()) / advantage.std()
            optimiser.zero_grad(set_to_none=True)
            hidden = policy.initial_hidden(1)
            consumed = 0
            policy_loss = torch.zeros(())
            value_loss = torch.zeros(())
            entropy = torch.zeros(())
            kl = torch.zeros(())
            for position, decision in enumerate(episode.decisions):
                target = decision.buffer_index + 1
                hidden = policy.advance(hidden, observations[index][consumed:target])
                consumed = target
                logits, value = policy.heads(hidden)
                # the same mask the sampler used; without it the probabilities differ by ~19x
                masked = masked_logits(logits[0], decision.allowed)
                log_probs = torch.log_softmax(masked, dim=-1)
                new_log_prob = log_probs[decision.action]
                old = torch.tensor(decision.policy_logprob, dtype=torch.float32)
                ratio = torch.exp(new_log_prob - old)
                advantage_t = torch.tensor(advantage[position], dtype=torch.float32)
                unclipped = ratio * advantage_t
                clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip) * advantage_t
                policy_loss = policy_loss - torch.min(unclipped, clipped)
                value_loss = value_loss + (value - torch.tensor(values[position],
                                                                dtype=torch.float32)) ** 2
                entropy = entropy + torch.distributions.Categorical(logits=masked).entropy()
                kl = kl + (old - new_log_prob).detach()
            count = max(1, len(episode.decisions))
            loss = policy_loss / count + value_coef * value_loss / count \
                - entropy_coef * entropy / count
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimiser.step()
            stats["policy_loss"] += float(policy_loss.detach()) / count
            stats["value_loss"] += float(value_loss.detach()) / count
            stats["entropy"] += float(entropy.detach()) / count
            stats["kl"] += float(kl.detach()) / count
            stats["updates"] += 1
    for key in ("policy_loss", "value_loss", "entropy", "kl"):
        stats[key] /= max(1, stats["updates"])
    return stats


def train(*, iterations: int = 60, episodes_per_iteration: int = 8, hidden: int = 128,
          lr: float = 3e-4, goal_distance: float = 0.45, seed: int = 0,
          dev_roots: int = 24, val_roots: int = 64, test_roots: int = 64, max_skills: int = 12,
          validate_every: int = 5, out_dir: Path = OUT, budget_seconds: Optional[float] = None,
          log: bool = True, resume_from: Optional[Path] = None,
          fail_after: Optional[int] = None) -> Dict[str, object]:
    """``fail_after`` deliberately aborts the run so a resume can be tested for real."""

    torch.manual_seed(seed)
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = build_roots("dev", 6)[:dev_roots]
    pool = build_roots("test", 24)
    validation, test = pool[:val_roots], pool[val_roots:val_roots + test_roots]
    assert not ({r.root_id for r in validation} & {r.root_id for r in test})
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=hidden)
    optimiser = torch.optim.AdamW(policy.parameters(), lr=lr, weight_decay=0.0)
    history: List[Dict[str, object]] = []
    best = {"success": -1.0, "state": None, "iteration": -1}
    start_iteration = 1
    resumed = False
    if resume_from is not None and Path(resume_from).exists():
        saved = torch.load(resume_from, map_location="cpu", weights_only=False)
        policy.load_state_dict(saved["state_dict"])
        if "optimiser" in saved:
            optimiser.load_state_dict(saved["optimiser"])
        if "torch_rng" in saved:
            torch.set_rng_state(saved["torch_rng"])
        history = list(saved.get("history", []))
        start_iteration = int(saved.get("iteration", 0)) + 1
        resumed = True
        if log:
            print(f"  resumed from {Path(resume_from).name} at iteration {start_iteration}",
                  flush=True)
    started = time.time()
    for iteration in range(start_iteration, iterations + 1):
        episodes = collect(policy, dev, goal_distance, seed=seed + iteration,
                           count=episodes_per_iteration, max_skills=max_skills)
        stats = ppo_update(policy, optimiser, episodes, seed=seed + iteration)
        if iteration % validate_every == 0 or iteration == 1 or iteration == iterations:
            report = summarise(collect(policy, validation, goal_distance,
                                       seed=seed + 1000 + iteration, deterministic=True,
                                       max_skills=max_skills))
            entry = {"iteration": iteration, "explore_return": float(np.mean(
                [e.total_reward for e in episodes])),
                "explore_success": float(np.mean([e.success for e in episodes])),
                "val_success_rate": report["success_rate"],
                "val_progress_towards_goal_m": report["mean_progress_towards_goal_m"],
                "val_end_distance_m": report["mean_end_distance_m"],
                "seconds": time.time() - started, **stats}
            history.append(entry)
            if report["success_rate"] > best["success"]:
                best = {"success": report["success_rate"], "iteration": iteration,
                        "state": {k: v.detach().clone() for k, v in policy.state_dict().items()}}
            if log:
                print(f"  iter {iteration:3d} explore_ret={entry['explore_return']:+.3f} "
                      f"val_succ={entry['val_success_rate']:.3f} "
                      f"val_progress={entry['val_progress_towards_goal_m']:+.3f} "
                      f"ratio={entry['mean_ratio_before_update']:.6f} "
                      f"max|r-1|={entry['max_abs_ratio_minus_one']:.2e} "
                      f"kl={entry['kl']:+.4f} ({entry['seconds']:.0f}s)", flush=True)
        if fail_after is not None and iteration >= fail_after:
            # an intentional interruption: the resumable checkpoint has already been written by the
            # normal path only at the end, so write it here exactly as a real crash-survivor would
            torch.save({"state_dict": policy.state_dict(), "optimiser": optimiser.state_dict(),
                        "observation_width": ObservationSpec().width, "action_count": len(MENU),
                        "hidden": hidden, "seed": seed, "iteration": iteration,
                        "torch_rng": torch.get_rng_state(), "goal_distance": goal_distance,
                        "max_skills": max_skills, "history": history,
                        "resume_kind": "optimiser + iteration + torch rng",
                        "menu": [action.label() for action in MENU]},
                       out_dir / f"policy_resumable_seed{seed}.pt")
            raise KeyboardInterrupt(f"intentional interruption after iteration {iteration}")
        if budget_seconds and time.time() - started > budget_seconds:
            break
    # two checkpoints with different jobs: the best inference weights and a fully resumable state
    inference_path = out_dir / f"policy_best_seed{seed}.pt"
    torch.save({"state_dict": best["state"] if best["state"] else policy.state_dict(),
                "observation_width": ObservationSpec().width, "action_count": len(MENU),
                "hidden": hidden, "seed": seed, "goal_distance": goal_distance,
                "selected_on": "highest validation success rate",
                "best_iteration": best["iteration"],
                "menu": [action.label() for action in MENU]}, inference_path)
    resumable_path = out_dir / f"policy_resumable_seed{seed}.pt"
    torch.save({"state_dict": policy.state_dict(), "optimiser": optimiser.state_dict(),
                "observation_width": ObservationSpec().width, "action_count": len(MENU),
                "hidden": hidden, "seed": seed, "iteration": len(history),
                "torch_rng": torch.get_rng_state(),
                "goal_distance": goal_distance, "max_skills": max_skills,
                "history": history, "resume_kind": "optimiser + iteration + torch rng",
                "menu": [action.label() for action in MENU]}, resumable_path)
    validation_report = summarise(collect(policy, validation, goal_distance, seed=seed + 7,
                                          deterministic=True, max_skills=max_skills))
    test_report = summarise(collect(policy, test, goal_distance, seed=seed + 8,
                                    deterministic=True, max_skills=max_skills))
    random_validation = summarise(collect(None, validation, goal_distance, seed=seed + 7,
                                          max_skills=max_skills))
    random_test = summarise(collect(None, test, goal_distance, seed=seed + 8,
                                    max_skills=max_skills))
    return {"seed": seed, "iterations": len(history), "history": history,
            "resumed": resumed, "start_iteration": start_iteration,
            "seconds": time.time() - started,
            "inference_checkpoint": str(inference_path.relative_to(ROOT))
            if str(inference_path).startswith(str(ROOT)) else str(inference_path),
            "resumable_checkpoint": str(resumable_path.relative_to(ROOT))
            if str(resumable_path).startswith(str(ROOT)) else str(resumable_path),
            "dev_roots": len(dev), "val_roots": len(validation), "test_roots": len(test),
            "trained": {"validation": validation_report, "test": test_report},
            "random": {"validation": random_validation, "test": random_test}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=60)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--budget-seconds", type=float, default=None)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
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
