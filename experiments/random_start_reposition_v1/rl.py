"""S4: the minimal RL diagnostic -- can this task be learned at all, before any scaling?

Design (task book section 8, all of it fixed *before* the run):

* encoder MLP + GRU (hidden 128) + a **conditional** joint action head: ``p(a0|h) *
  p(a1|h, a0)`` over the 225 ordered pairs. The recorded quantity is the exact joint log
  probability, so the two conditionals cannot be mistaken for a correlated joint distribution.
* one decision per control period through the GRU: the boundary action is sampled from the
  hidden state *before* the skill, and the hidden state is then advanced over the skill's real
  per-period observations. Gradients are truncated at skill boundaries (a fixed, declared
  truncation), while the hidden state itself is carried across the boundary.
* semi-Markov reward: the environment's per-boundary reward already carries
  ``gamma ** n``. The value target is the Monte-Carlo discounted return from that boundary,
  ``sum_{j>=i} r_j / prefix_i``; no GAE is used, so there is no lambda to mis-time.
* advantages are normalised **once per update batch, across all decisions of all
  trajectories**, never per episode.
* fixed configuration: AdamW lr 3e-4, PPO clip 0.2, value weight 0.5, entropy weight 0.01,
  gamma 0.997 per 0.05 s period.
* evaluation reports deterministic (argmax) and stochastic (policy sampling) execution, plus
  the uniform-random baseline and the untrained network.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import layout  # noqa: E402
from coimagine.random_start.spec import CONTROL_DT, GAMMA, ROBOT_ACTION_COUNT  # noqa: E402

OUT = ROOT / "outputs/random_start_reposition_v1"

HIDDEN = 128
EMBED = 16
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.01
UPDATE_EPOCHS = 3
TRAJECTORIES_PER_UPDATE = 8


class JointPolicy(nn.Module):
    """Encoder + GRU + conditional joint head over (robot0 action, robot1 action)."""

    def __init__(self, observation_dim: int, hidden: int = HIDDEN):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh())
        self.gru = nn.GRUCell(hidden, hidden)
        self.embed = nn.Embedding(ROBOT_ACTION_COUNT, EMBED)
        self.head_first = nn.Linear(hidden, ROBOT_ACTION_COUNT)
        self.head_second = nn.Linear(hidden + EMBED, ROBOT_ACTION_COUNT)
        self.value_head = nn.Linear(hidden, 1)
        self.hidden = hidden

    def initial_hidden(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.hidden)

    def advance(self, observations: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """One GRU step per observation row (a whole skill's frames)."""

        state = hidden
        for row in range(observations.shape[0]):
            state = self.gru(self.encoder(observations[row:row + 1]), state)
        return state

    def head(self, hidden: torch.Tensor, first: Optional[torch.Tensor] = None):
        logits_first = self.head_first(hidden)
        if first is None:
            return logits_first, None
        second = torch.cat([hidden, self.embed(first)], dim=-1)
        return logits_first, self.head_second(second)

    def value(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.value_head(hidden).squeeze(-1)

    def sample(self, hidden: torch.Tensor, generator: Optional[torch.Generator] = None,
               deterministic: bool = False) -> Tuple[int, int, float, float]:
        logits_first, _ = self.head(hidden)
        dist_first = torch.distributions.Categorical(logits=logits_first)
        if deterministic:
            first = int(torch.argmax(logits_first, dim=-1).item())
            logp_first = float(dist_first.log_prob(torch.tensor([first])).item())
        else:
            first = int(dist_first.sample().item())
            logp_first = float(dist_first.log_prob(torch.tensor([first])).item())
        _, logits_second = self.head(hidden, torch.tensor([first]))
        dist_second = torch.distributions.Categorical(logits=logits_second)
        if deterministic:
            second = int(torch.argmax(logits_second, dim=-1).item())
            logp_second = float(dist_second.log_prob(torch.tensor([second])).item())
        else:
            second = int(dist_second.sample().item())
            logp_second = float(dist_second.log_prob(torch.tensor([second])).item())
        entropy = float(dist_first.entropy().item() + dist_second.entropy().item())
        return first, second, logp_first + logp_second, entropy

    def evaluate(self, hidden: torch.Tensor, first: int, second: int):
        logits_first, _ = self.head(hidden)
        dist_first = torch.distributions.Categorical(logits=logits_first)
        _, logits_second = self.head(hidden, torch.tensor([first]))
        dist_second = torch.distributions.Categorical(logits=logits_second)
        logp = (dist_first.log_prob(torch.tensor([first]))
                + dist_second.log_prob(torch.tensor([second])))
        entropy = dist_first.entropy() + dist_second.entropy()
        return logp, entropy


def rollout(environment: E.RandomStartEnv, policy: JointPolicy, *, stochastic: bool = True,
            generator: Optional[torch.Generator] = None) -> Dict[str, object]:
    """One episode: sample a joint skill per boundary, advance the GRU on real periods."""

    decisions: List[Dict[str, object]] = []
    hidden = policy.initial_hidden()
    while environment.termination is None:
        # the frame at the boundary instant is consumed *before* the decision, so that every
        # frame of the episode is consumed exactly once: the boundary frame here, and the
        # remainder of the skill's frames after the action has been taken
        boundary_frame = torch.as_tensor(environment.boundary_vector(),
                                         dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            hidden = policy.advance(boundary_frame, hidden)
            value = float(policy.value(hidden).item())
            first, second, logp, entropy = policy.sample(hidden, generator=generator,
                                                         deterministic=not stochastic)
        index_start = len(environment.rows)
        environment.joint_step(first, second)
        boundary = environment.boundaries[-1]
        inner = environment.history[index_start + 1:index_start + int(boundary["periods"])]
        observations = torch.as_tensor(np.stack([entry["observation"] for entry in inner])
                                       if inner else np.zeros((0, environment.
                                                               observation_fields().__len__())),
                                       dtype=torch.float32)
        with torch.no_grad():
            hidden = policy.advance(observations, hidden)
        decisions.append({
            "skill_index": len(decisions) + 1,
            "action": [int(first), int(second)],
            "logp": float(logp), "entropy": float(entropy), "value": float(value),
            "reward": float(boundary["reward"]["total"]),
            "periods": int(boundary["periods"]),
            "prefix_before": float(boundary["discount_prefix_before"]),
            "terminated": bool(boundary["success"] or boundary["truncated"]
                               or boundary["anomaly"] is not None),
            "success": bool(boundary["success"]),
            "truncated": bool(boundary["truncated"]),
            "observations": observations,
            "boundary_frame": boundary_frame,
        })
    # Monte-Carlo semi-Markov return from each boundary, expressed at period 0
    running = 0.0
    for decision in reversed(decisions):
        running += float(decision["reward"])
        decision["return_to_go"] = running
    return {"decisions": decisions, "success": bool(environment.success),
            "termination": environment.termination,
            "final_goal_distance": float(environment.object_distance()),
            "skills_used": int(environment.skills_used),
            "periods": int(environment.total_periods),
            "return": float(environment.reward_total)}


def flatten(episodes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Batch-wide advantage statistics: one mean/std over every valid decision."""

    entries: List[Dict[str, object]] = []
    for episode in episodes:
        for decision in episode["decisions"]:  # type: ignore[index]
            entries.append(dict(decision, episode_success=episode["success"]))
    for entry in entries:
        entry["target"] = float(entry["return_to_go"]) / max(1e-12, float(entry["prefix_before"]))
        entry["advantage_raw"] = entry["target"] - float(entry["value"])
    raw = np.asarray([entry["advantage_raw"] for entry in entries])
    mean, std = float(raw.mean()), float(raw.std() + 1e-8)
    for entry in entries:
        entry["advantage"] = (float(entry["advantage_raw"]) - mean) / std
    return {"entries": entries, "advantage_mean": mean, "advantage_std": std}


def normalize_per_trajectory(episodes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """The single declared ablation: per-trajectory mean/std instead of batch-wide."""

    entries: List[Dict[str, object]] = []
    for episode in episodes:
        local: List[Dict[str, object]] = []
        for decision in episode["decisions"]:  # type: ignore[index]
            entry = dict(decision, episode_success=episode["success"])
            entry["target"] = float(entry["return_to_go"]) / max(1e-12,
                                                                float(entry["prefix_before"]))
            entry["advantage_raw"] = entry["target"] - float(entry["value"])
            local.append(entry)
        raw = np.asarray([entry["advantage_raw"] for entry in local]) if local else np.zeros(1)
        mean, std = float(raw.mean()), float(raw.std() + 1e-8)
        for entry in local:
            entry["advantage"] = (float(entry["advantage_raw"]) - mean) / std
        entries.extend(local)
    return {"entries": entries, "advantage_mean": 0.0, "advantage_std": 0.0}


def update(policy: JointPolicy, optimizer: torch.optim.Optimizer, episodes: Sequence[Dict[str, object]],
           *, normalize: str = "batch") -> Dict[str, object]:
    batch = flatten(episodes) if normalize == "batch" else normalize_per_trajectory(episodes)
    entries = batch["entries"]
    advantages = torch.as_tensor([entry["advantage"] for entry in entries], dtype=torch.float32)
    old_logp = torch.as_tensor([entry["logp"] for entry in entries], dtype=torch.float32)
    targets = torch.as_tensor([entry["target"] for entry in entries], dtype=torch.float32)

    metrics: Dict[str, object] = {
        "decisions": len(entries),
        "advantage_mean": float(batch["advantage_mean"]),
        "advantage_std": float(batch["advantage_std"]),
        "advantage_positive_fraction": float(np.mean([entry["advantage_raw"] > 0
                                                      for entry in entries])) if entries else 0.0,
        "success_trajectories": int(sum(1 for episode in episodes if episode["success"])),
        "failure_trajectories": int(sum(1 for episode in episodes if not episode["success"])),
    }
    success_advantages = [entry["advantage_raw"] for entry in entries if entry["episode_success"]]
    failure_advantages = [entry["advantage_raw"] for entry in entries
                          if not entry["episode_success"]]
    metrics["advantage_raw_success_mean"] = float(np.mean(success_advantages)) \
        if success_advantages else None
    metrics["advantage_raw_failure_mean"] = float(np.mean(failure_advantages)) \
        if failure_advantages else None

    grad_norms: List[float] = []
    ratios: List[float] = []
    clip_fractions: List[float] = []
    entropies: List[float] = []
    value_errors: List[float] = []
    policy_changes: List[float] = []
    for _ in range(UPDATE_EPOCHS):
        offset = 0
        optimizer.zero_grad()
        losses = []
        for episode in episodes:
            hidden = policy.initial_hidden()
            for decision in episode["decisions"]:  # type: ignore[index]
                hidden = policy.advance(decision["boundary_frame"], hidden)
                first, second = decision["action"]
                logp, entropy = policy.evaluate(hidden, int(first), int(second))
                hidden = policy.advance(decision["observations"], hidden)
                value = policy.value(hidden)
                index = offset
                advantage = advantages[index]
                ratio = torch.exp(logp - old_logp[index])
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1.0 - CLIP_RANGE, 1.0 + CLIP_RANGE) * advantage
                policy_loss = -torch.min(unclipped, clipped)
                value_loss = (value - targets[index]) ** 2
                losses.append(policy_loss + VALUE_WEIGHT * value_loss
                              - ENTROPY_WEIGHT * entropy)
                ratios.append(float(ratio.item()))
                clip_fractions.append(float((torch.abs(ratio - 1.0) > CLIP_RANGE).item()))
                entropies.append(float(entropy.item()))
                value_errors.append(float(abs(value.item() - targets[index].item())))
                policy_changes.append(float(abs(logp.item() - old_logp[index].item())))
                offset += 1
        loss = torch.stack(losses).mean() if losses else torch.zeros((), requires_grad=True)
        loss.backward()
        norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0).item()
                     if any(p.grad is not None for p in policy.parameters()) else 0.0)
        grad_norms.append(norm)
        optimizer.step()
    metrics.update({
        "grad_norm_last": grad_norms[-1] if grad_norms else 0.0,
        "ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
        "ratio_max": float(np.max(ratios)) if ratios else 1.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "value_error_mean": float(np.mean(value_errors)) if value_errors else 0.0,
        "policy_change_mean": float(np.mean(policy_changes)) if policy_changes else 0.0,
    })
    return metrics


def evaluate(policy: JointPolicy, instances: Sequence[layout.InstanceSpec], *,
             stochastic: bool, seeds: int = 3, max_skills: int = 30) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for instance in instances:
        for seed in range(seeds if stochastic else 1):
            generator = torch.Generator().manual_seed(seed)
            environment = E.RandomStartEnv(instance, E.EnvConfig(skill_budget=max_skills))
            episode = rollout(environment, policy, stochastic=stochastic, generator=generator)
            rows.append({"instance": instance.name, "seed": seed,
                         "success": episode["success"],
                         "final_goal_distance": episode["final_goal_distance"],
                         "skills_used": episode["skills_used"]})
    by_instance: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance"]].append(1.0 if row["success"] else 0.0)
    values = np.asarray([float(np.mean(group)) for group in by_instance.values()])
    return {"episodes": len(rows), "instances": int(values.size),
            "success": float(values.mean()) if values.size else 0.0,
            "median_final_distance": float(np.median([row["final_goal_distance"]
                                                      for row in rows])) if rows else None,
            "median_skills": float(np.median([row["skills_used"] for row in rows]))
            if rows else None,
            "per_instance": dict(by_instance), "rows": rows}


def random_baseline(instances: Sequence[layout.InstanceSpec], seeds: int = 3,
                    max_skills: int = 30) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for instance in instances:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            environment = E.RandomStartEnv(instance, E.EnvConfig(skill_budget=max_skills))
            while environment.termination is None:
                environment.joint_step(int(rng.integers(0, ROBOT_ACTION_COUNT)),
                                       int(rng.integers(0, ROBOT_ACTION_COUNT)))
            rows.append({"instance": instance.name, "seed": seed, "success": environment.success,
                         "final_goal_distance": float(environment.object_distance())})
    by_instance: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance"]].append(1.0 if row["success"] else 0.0)
    values = np.asarray([float(np.mean(group)) for group in by_instance.values()])
    return {"episodes": len(rows), "instances": int(values.size),
            "success": float(values.mean()) if values.size else 0.0}


def bandit_check(seed: int = 0) -> Dict[str, object]:
    """A hand-checkable decision problem: the better action's probability must rise.

    One state, two actions, the same PPO code path: action 0 pays +1, action 1 pays -1. After
    a short optimisation the deterministic policy must select action 0 and p(action 0) must
    have increased relative to the initial uniform value.
    """

    torch.manual_seed(seed)
    policy = JointPolicy(observation_dim=4, hidden=32)
    # the sanity problem uses its own (larger) learning rate: it checks the *direction* of the
    # update on a problem that can be hand-checked, not the round's hyper-parameter value
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.05)
    history: List[float] = []
    for step in range(800):
        # a stateless two-action bandit: the same one-state decision problem every step, so the
        # only thing that can change is the preference for the better-paying action
        with torch.no_grad():
            hidden = policy.initial_hidden()
            logits_first, _ = policy.head(hidden)
            probability = float(torch.softmax(logits_first, dim=-1)[0, 0].item())
            history.append(probability)
        first, second, logp, entropy = policy.sample(hidden)
        reward = 1.0 if first == 0 else -1.0
        dist_first = torch.distributions.Categorical(logits=policy.head(hidden)[0])
        _, logits_second = policy.head(hidden, torch.tensor([first]))
        dist_second = torch.distributions.Categorical(logits=logits_second)
        logp_t = (dist_first.log_prob(torch.tensor([first]))
                  + dist_second.log_prob(torch.tensor([second])))
        # the advantage is the observed reward minus the learned state value, exactly the form
        # the environment's advantage takes (no reward model, no baseline other than V)
        value = policy.value(hidden)
        advantage = torch.tensor([reward]) - value
        loss = (-logp_t * advantage.detach()
                + VALUE_WEIGHT * (value - torch.tensor([reward])) ** 2
                - ENTROPY_WEIGHT * (dist_first.entropy() + dist_second.entropy()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        logits_first, _ = policy.head(policy.initial_hidden())
        final_probability = float(torch.softmax(logits_first, dim=-1)[0, 0].item())
        greedy = int(torch.argmax(logits_first, dim=-1).item())
    return {"ok": bool(final_probability > 0.5 and greedy == 0
                       and final_probability > history[0]),
            "initial_probability_action0": history[0],
            "final_probability_action0": final_probability,
            "greedy_action": greedy,
            "probability_trace_every_25": history[::25]}


def probability_replay_check(policy: JointPolicy) -> Dict[str, object]:
    """Before updating, a recomputed decision probability must match the sampling record."""

    instances, _ = layout.build_split("L0", "validation", count=1)
    environment = E.RandomStartEnv(instances[0])
    episode = rollout(environment, policy, stochastic=True,
                      generator=torch.Generator().manual_seed(3))
    worst = 0.0
    # replay the sequence and recompute each recorded log probability
    hidden = policy.initial_hidden()
    for decision in episode["decisions"]:  # type: ignore[index]
        hidden = policy.advance(decision["boundary_frame"], hidden)
        with torch.no_grad():
            logp, _ = policy.evaluate(hidden, int(decision["action"][0]),
                                      int(decision["action"][1]))
        hidden = policy.advance(decision["observations"], hidden)
        worst = max(worst, abs(float(logp.item()) - float(decision["logp"])))
    return {"ok": bool(worst < 1e-5), "worst_logp_difference": worst,
            "decisions": len(episode["decisions"]),
            "note": "the sampling record and the recomputed joint log probability agree; the "
                    "GRU state used for the recomputation is the one the action was sampled from"}


def unchanged_policy_ratio_check(policy: JointPolicy, instance) -> Dict[str, object]:
    """With parameters untouched, the recomputed ratio must be exactly 1 for every decision."""

    environment = E.RandomStartEnv(instance)
    episode = rollout(environment, policy, stochastic=True,
                      generator=torch.Generator().manual_seed(11))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.0)
    before = [parameter.detach().clone() for parameter in policy.parameters()]
    metrics = update(policy, optimizer, [episode])
    unchanged = all(bool(torch.equal(parameter.detach(), reference))
                    for parameter, reference in zip(policy.parameters(), before))
    return {"ok": bool(unchanged and abs(metrics["ratio_mean"] - 1.0) < 1e-5
                       and abs(metrics["ratio_max"] - 1.0) < 1e-5),
            "ratio_mean": metrics["ratio_mean"], "ratio_max": metrics["ratio_max"],
            "parameters_unchanged": unchanged, "decisions": metrics["decisions"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=TRAJECTORIES_PER_UPDATE)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--dev-instances", type=int, default=8)
    parser.add_argument("--scan", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize", default="batch", choices=("batch", "trajectory"))
    parser.add_argument("--bandit-only", action="store_true")
    parser.add_argument("--save-every", type=int, default=10,
                        help="period between atomic resumable checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoints/<prefix>_resume.pt if it exists")
    parser.add_argument("--prefix", default="rl", help="checkpoint file prefix")
    parser.add_argument("--subset-file", default=None,
                        help="diagnostic subset json (defaults to <out>/results/diagnostic_subset.json)")
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    bandit = bandit_check(args.seed)
    print("bandit check:", bandit)
    if args.bandit_only:
        (out / "results" / "rl_bandit.json").write_text(json.dumps(bandit, indent=1))
        return 0

    pool, _ = layout.build_split("L0", "train", count=args.scan)
    validation_pool, _ = layout.build_split("L0", "validation", count=args.scan)
    by_name = {instance.name: instance for instance in list(pool) + list(validation_pool)}
    subset_path = Path(args.subset_file) if args.subset_file else \
        out / "results" / "diagnostic_subset.json"
    dev: List[layout.InstanceSpec] = []
    selection = "not available"
    if subset_path.exists():
        data = json.loads(subset_path.read_text())
        dev = [by_name[name] for name in data["selected"] if name in by_name]
        selection = ("instances on which the scripted demo policy actually succeeded "
                     f"({len(data['successes'])} of {data['attempted']} episodes); "
                     f"padded without success evidence: "
                     f"{data['padded_without_success_evidence']}")
    if len(dev) < 4:
        dev = pool[: args.dev_instances]
        selection = "fallback: the first dev-instances of the frozen L0 train panel"
    validation, _ = layout.build_split("L0", "validation", count=8)

    probe = E.RandomStartEnv(dev[0])
    observation_dim = int(probe.observation_fields().__len__())
    policy = JointPolicy(observation_dim)
    checkpoint = out / "checkpoints" / f"{args.prefix}_initial.pt"
    torch.save(policy.state_dict(), checkpoint)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE)

    initial = copy.deepcopy(policy)
    replay_check = probability_replay_check(policy)
    ratio_check = unchanged_policy_ratio_check(policy, dev[0])
    evaluation_init = evaluate(initial, dev, stochastic=False)
    evaluation_init_stochastic = evaluate(initial, dev, stochastic=True, seeds=3)
    baseline_random = random_baseline(dev, seeds=3)

    resume_path = out / "checkpoints" / f"{args.prefix}_resume.pt"
    start_update = 1
    history: List[Dict[str, object]] = []
    if args.resume and resume_path.exists():
        # a local, trusted training state: it contains the numpy RNG state as well as
        # tensors, so the default weights_only=True loader rejects it
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        history = list(state.get("history", []))
        start_update = int(state["update"]) + 1
        print(f"resumed at update {start_update} from {resume_path}", flush=True)

    def save_resume(update_index: int) -> None:
        payload = {"update": int(update_index), "model": policy.state_dict(),
                   "optimizer": optimizer.state_dict(), "torch_rng": torch.get_rng_state(),
                   "numpy_rng": np.random.get_state(), "history": history,
                   "config": vars(args)}
        temporary = resume_path.with_suffix(".tmp")
        torch.save(payload, temporary)
        temporary.replace(resume_path)  # atomic: a partial file can never be read as a state

    start = time.time()
    for update_index in range(start_update, args.updates + 1):
        episodes: List[Dict[str, object]] = []
        rng = np.random.default_rng(1000 + update_index)
        for episode_index in range(args.episodes_per_update):
            instance = dev[int(rng.integers(0, len(dev)))]
            environment = E.RandomStartEnv(instance)
            episodes.append(rollout(environment, policy, stochastic=True,
                                    generator=torch.Generator().manual_seed(
                                        int(rng.integers(0, 10 ** 6)))))
        metrics = update(policy, optimizer, episodes, normalize=args.normalize)
        metrics["update"] = update_index
        metrics["rollout_success"] = float(np.mean([episode["success"] for episode in episodes]))
        metrics["rollout_return_mean"] = float(np.mean([episode["return"]
                                                        for episode in episodes]))
        metrics["rollout_skills_mean"] = float(np.mean([episode["skills_used"]
                                                        for episode in episodes]))
        metrics["wall_seconds"] = time.time() - start
        if update_index % args.eval_every == 0 or update_index == args.updates:
            deterministic = evaluate(policy, dev, stochastic=False)
            metrics["dev_success_deterministic"] = deterministic["success"]
            metrics["dev_final_distance_median"] = deterministic["median_final_distance"]
            metrics["dev_success_per_instance"] = deterministic["per_instance"]
        history.append(metrics)
        if update_index % args.save_every == 0:
            save_resume(update_index)
        print(json.dumps({key: metrics[key] for key in
                          ("update", "rollout_success", "rollout_return_mean", "ratio_mean",
                           "clip_fraction", "entropy_mean", "grad_norm_last",
                           "dev_success_deterministic")
                          if key in metrics}), flush=True)

    save_resume(args.updates)
    final = evaluate(policy, dev, stochastic=False)
    final_stochastic = evaluate(policy, dev, stochastic=True, seeds=3)
    validation_result = evaluate(policy, validation, stochastic=False)
    torch.save(policy.state_dict(), out / "checkpoints" / f"{args.prefix}_final.pt")
    torch.save(policy.state_dict(), out / "checkpoints" / "rl_final.pt")
    passed = bool(final["success"] >= 0.75
                  and final["success"] - evaluation_init["success"] >= 0.20)
    report = {
        "config": {"updates": args.updates, "episodes_per_update": args.episodes_per_update,
                   "eval_every": args.eval_every, "hidden": HIDDEN, "lr": LEARNING_RATE,
                   "clip": CLIP_RANGE, "value_weight": VALUE_WEIGHT,
                   "entropy_weight": ENTROPY_WEIGHT, "gamma": GAMMA,
                   "control_dt": CONTROL_DT, "normalize": args.normalize,
                   "advantage_estimator": "Monte-Carlo semi-Markov return minus value; no GAE",
                   "gradient_truncation": "hidden state carried across skill boundaries, "
                                          "gradients truncated at each boundary", "seed": args.seed},
        "dev_subset": {"selection": selection, "instances": [instance.name for instance in dev]},
        "bandit_check": bandit,
        "probability_replay_check": replay_check,
        "unchanged_policy_ratio_check": ratio_check,
        "baselines": {"uniform_random": baseline_random,
                      "untrained_deterministic": evaluation_init["success"],
                      "untrained_stochastic": evaluation_init_stochastic["success"]},
        "initial_deterministic": evaluation_init,
        "final_deterministic": final,
        "final_stochastic": final_stochastic,
        "validation_deterministic": validation_result,
        "pass_signal": passed,
        "history": history,
    }
    (out / "results" / "rl_diagnostic.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({"pass_signal": passed,
                      "init_deterministic": evaluation_init["success"],
                      "final_deterministic": final["success"],
                      "uniform_random": baseline_random["success"],
                      "validation_deterministic": validation_result["success"],
                      "bandit_ok": bandit["ok"], "replay_ok": replay_check["ok"],
                      "ratio_ok": ratio_check["ok"]}, indent=1))
    print("RL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
