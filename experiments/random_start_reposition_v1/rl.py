"""S4: the minimal RL diagnostic -- can this task be learned at all, before any scaling?

Design (task book section 8, all of it fixed *before* the run):

* encoder MLP + GRU (hidden 128) + a **conditional** joint action head: ``p(a0|h) *
  p(a1|h, a0)`` over the 225 ordered pairs. The recorded quantity is the exact joint log
  probability, so the two conditionals cannot be mistaken for a correlated joint distribution.
* one decision per control period through the GRU: the boundary action is sampled from the
  hidden state *before* the skill, and the hidden state is then advanced over the skill's real
   per-period observations. After reading each decision's actor/critic heads, its hidden state
   is detached before advancing the skill's inner frames. Thus the next decision can train on
   those intervening frames, but not back through the previous decision. Hidden values carry.
* semi-Markov reward: the environment's per-boundary reward already carries
  ``gamma ** n``. The value target is the Monte-Carlo discounted return from that boundary,
  ``sum_{j>=i} r_j / prefix_i``; no GAE is used, so there is no lambda to mis-time.
* advantages are normalised **once per update batch, across all decisions of all
  trajectories**, never per episode.
* fixed configuration: AdamW lr 3e-4 (explicitly no weight decay), PPO clip 0.2, value
   weight 0.5, no entropy bonus (entropy is logged only),
   gamma 1.0 for the finite-horizon dense-reward-v2 task (no terminal potential).
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
from coimagine.random_start.spec import (  # noqa: E402
    CONTROL_DT, GAMMA, JOINT_ACTION_COUNT, ROBOT_ACTION_COUNT, ENV_VERSION)

OUT = ROOT / "outputs/random_start_time_budget_v3"

HIDDEN = 128
EMBED = 16
LEARNING_RATE = 3e-4
CLIP_RANGE = 0.2
VALUE_WEIGHT = 0.5
ENTROPY_WEIGHT = 0.0  # log entropy, but do not add a second optimisation preference
UPDATE_EPOCHS = 3
TRAJECTORIES_PER_UPDATE = 8

#: The one training variable of the bounded long-run comparison.
#: ``shared`` (arm A) is the existing baseline: the value loss backpropagates into the
#: encoder/GRU. ``stopped`` (arm B) feeds the value head a detached hidden state, so the value
#: head still trains but the critic cannot move the shared representation. Everything else --
#: network, parameters, forward value function, targets, advantages, optimizer, clipping -- is
#: identical. This is NOT an independent critic network and it is not claimed to be better.
CRITIC_MODES = {"shared": True, "stopped": False}


class JointPolicy(nn.Module):
    """Encoder + GRU + conditional joint head over (robot0 action, robot1 action).

    ``critic_shared_grad`` is an explicit, serialisable architecture switch. It does not add
    or remove parameters; it only changes whether the value loss sees a differentiable shared
    hidden state. Arm B therefore has exactly the same state_dict shape as arm A.
    """

    def __init__(self, observation_dim: int, hidden: int = HIDDEN,
                 critic_shared_grad: bool = True):
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
        self.observation_dim = int(observation_dim)
        self.critic_shared_grad = bool(critic_shared_grad)

    def contract(self) -> Dict[str, object]:
        """Everything a checkpoint/resume must compare, apart from the weights themselves."""

        return {"model": "JointPolicy", "observation_dim": self.observation_dim,
                "hidden": self.hidden, "critic_shared_grad": self.critic_shared_grad,
                "critic_mode": ("shared" if self.critic_shared_grad else "stopped")}

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
        """Value read-out; arm B detaches the shared hidden state here, and only here.

        The value head keeps all of its parameters and receives the value loss normally. The
        actor path is untouched: it reads the same non-detached hidden state.
        """

        if not self.critic_shared_grad:
            hidden = hidden.detach()
        return self.value_head(hidden).squeeze(-1)

    def sample(self, hidden: torch.Tensor, generator: Optional[torch.Generator] = None,
               deterministic: bool = False) -> Tuple[int, int, float, float]:
        logits_first, _ = self.head(hidden)
        dist_first = torch.distributions.Categorical(logits=logits_first)
        if deterministic:
            first = int(torch.argmax(logits_first, dim=-1).item())
            logp_first = float(dist_first.log_prob(torch.tensor([first])).item())
        else:
            first = int(torch.multinomial(dist_first.probs, 1, generator=generator).item())
            logp_first = float(dist_first.log_prob(torch.tensor([first])).item())
        _, logits_second = self.head(hidden, torch.tensor([first]))
        dist_second = torch.distributions.Categorical(logits=logits_second)
        if deterministic:
            second = int(torch.argmax(logits_second, dim=-1).item())
            logp_second = float(dist_second.log_prob(torch.tensor([second])).item())
        else:
            second = int(torch.multinomial(dist_second.probs, 1, generator=generator).item())
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
    # Diagnostics below are read from the recorded public boundaries only; no hidden truth is
    # used for a policy input. "short" is the task-book definition: a joint skill of at most
    # two control periods (0.10 s).
    boundaries = environment.boundaries
    durations = np.asarray([int(boundary["periods"]) for boundary in boundaries], dtype=np.int64)
    exits = Counter(end["exit_reason"]
                    for boundary in boundaries for end in boundary["skill_ends"].values())
    action_indices = [int(b["action_index"][0]) * ROBOT_ACTION_COUNT
                      + int(b["action_index"][1]) for b in boundaries]
    repeat_runs: List[int] = []
    if action_indices:
        run = 1
        for previous, current in zip(action_indices, action_indices[1:]):
            if previous == current:
                run += 1
            else:
                repeat_runs.append(run)
                run = 1
        repeat_runs.append(run)
    repeats = sum(a["action_index"] == b["action_index"]
                  for a, b in zip(boundaries, boundaries[1:]))
    total_periods = int(durations.sum())
    short_periods = int(durations[durations <= 2].sum())
    return {"decisions": decisions, "success": bool(environment.success),
            "termination": environment.termination,
            "final_goal_distance": float(environment.object_distance()),
            "skills_used": int(environment.skills_used),
            "periods": int(environment.total_periods),
            "return": float(environment.reward_total),
            "reward_components": dict(environment.reward_components),
            "exits": dict(exits),
            "short_decisions": int(np.sum(durations <= 2)),
            "short_time_fraction": (float(short_periods / total_periods)
                                    if total_periods else 0.0),
            "repeated_action_pairs": int(repeats),
            "longest_repeat_run": int(max(repeat_runs)) if repeat_runs else 0,
            "mean_repeat_run": float(np.mean(repeat_runs)) if repeat_runs else 0.0,
            "median_periods": float(np.median(durations)) if durations.size else 0.0,
            "action_indices": action_indices}


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


def decision_group(entry: Dict[str, object]) -> str:
    if entry["episode_success"]:
        return "success"
    return "failure_short" if entry["periods"] <= 2 else "failure_long"


def advantage_group_statistics(episodes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Raw and batch-normalised advantage statistics per outcome group, per update."""

    entries = flatten(episodes)["entries"]
    groups = ("success", "failure_short", "failure_long")
    report: Dict[str, object] = {}
    for group in groups:
        rows = [entry for entry in entries if decision_group(entry) == group]
        if not rows:
            continue
        raw = np.asarray([entry["advantage_raw"] for entry in rows])
        normalised = np.asarray([entry["advantage"] for entry in rows])
        report[group] = {"decisions": len(rows),
                         "raw_mean": float(raw.mean()), "raw_std": float(raw.std()),
                         "raw_positive_fraction": float(np.mean(raw > 0)),
                         "normalized_mean": float(normalised.mean()),
                         "normalized_std": float(normalised.std())}
    return report


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


def decision_states(policy: JointPolicy, episode: Dict[str, object]):
    """Replay pre-action states with one-skill truncated backpropagation.

    Each head reads exactly the state used by rollout. Detach *after* the heads, not after the
    inner observations: the latter would prevent future losses from training on those frames.
    No observation values are dropped or reset by the gradient truncation.
    """
    hidden = policy.initial_hidden()
    for decision in episode["decisions"]:
        hidden = policy.advance(decision["boundary_frame"], hidden)
        yield decision, hidden
        hidden = policy.advance(decision["observations"], hidden.detach())


def value_fit_metrics(policy: JointPolicy, episodes: Sequence[Dict[str, object]]) -> Dict[str, float]:
    """Pre-action value quality on the current rollout batch (not held-out accuracy)."""
    targets = np.asarray([entry["target"] for entry in flatten(episodes)["entries"]])
    predictions = []
    with torch.no_grad():
        for episode in episodes:
            for _decision, hidden in decision_states(policy, episode):
                predictions.append(float(policy.value(hidden).item()))
    residual = targets - np.asarray(predictions)
    variance = float(np.var(targets))
    return {"mse": float(np.mean(residual ** 2)),
            "explained_variance": float(1.0 - np.var(residual) / variance)
            if variance > 1e-12 else None}


def sustained_pass(history, initial_success: float) -> bool:
    """The latest three fixed-panel evaluation points must all clear both thresholds."""
    points = [row["dev_success_deterministic"] for row in history
              if "dev_success_deterministic" in row]
    return len(points) >= 3 and all(value >= 0.75 and value - initial_success >= 0.20
                                    for value in points[-3:])


# ---------------------------------------------------------------------------
# distribution diagnostics: the exact 225-way joint distribution, not just the
# sampled action's probability
# ---------------------------------------------------------------------------


def joint_log_distribution(policy: JointPolicy, hidden: torch.Tensor) -> torch.Tensor:
    """Log-probability of every one of the 225 ordered joint actions.

    The model is p(a0) * p(a1 | a0), so the exact joint density is the outer sum of the two
    conditional log-softmax tables. The joint entropy logged during training is likewise
    H(a0) + H(a1 | a0) = H(a0, a1); the two conditionals are not mixed up with a correlated
    joint distribution.
    """

    with torch.no_grad():
        first = torch.log_softmax(policy.head_first(hidden), dim=-1)[0]
        second = policy.head_second(torch.cat(
            [hidden.expand(ROBOT_ACTION_COUNT, -1),
             policy.embed(torch.arange(ROBOT_ACTION_COUNT))], dim=-1))
        second = torch.log_softmax(second, dim=-1)
        return (first[:, None] + second).flatten()


def joint_distribution_panel(policy: JointPolicy,
                             episodes: Sequence[Dict[str, object]]) -> List[torch.Tensor]:
    """Exact [n_decisions, 225] log-probability table per trajectory."""

    rows: List[torch.Tensor] = []
    for episode in episodes:
        path = [joint_log_distribution(policy, hidden)
                for _decision, hidden in decision_states(policy, episode)]
        rows.append(torch.stack(path) if path
                    else torch.zeros((0, JOINT_ACTION_COUNT)))
    return rows


def joint_kl(before: Sequence[torch.Tensor],
             after: Sequence[torch.Tensor]) -> Dict[str, Optional[float]]:
    """Exact KL(before || after) over the 225 joint actions, two aggregation units."""

    per_path = [torch.sum(a.exp() * (a - b), dim=-1) for a, b in zip(before, after)]
    decisions = [float(value) for path in per_path for value in path]
    trajectories = [float(path.mean()) for path in per_path if path.numel()]
    return {"by_decision_mean": float(np.mean(decisions)) if decisions else None,
            "by_trajectory_mean": float(np.mean(trajectories)) if trajectories else None,
            "max_by_decision": float(np.max(decisions)) if decisions else None,
            "decisions": len(decisions), "trajectories": len(trajectories)}


def path_scores(policy: JointPolicy, episodes: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Recorded-action log probability and exact joint table for a fixed history.

    Only recorded legal observations and executed actions are read; these histories never
    enter training or model selection. They are a fixed-history forgetting diagnostic.
    """

    scores: List[Dict[str, object]] = []
    for episode in episodes:
        logps, hits, tables = [], [], []
        for decision, hidden in decision_states(policy, episode):
            logp, _ = policy.evaluate(hidden, *decision["action"])
            logps.append(float(logp.detach()))
            hits.append(policy.sample(hidden, deterministic=True)[:2] == tuple(decision["action"]))
            tables.append(joint_log_distribution(policy, hidden))
        scores.append({"instance": episode.get("instance"), "success": episode.get("success"),
                       "mean_logp": float(np.mean(logps)) if logps else None,
                       "accuracy": float(np.mean(hits)) if hits else None,
                       "joint_logs": torch.stack(tables) if tables
                       else torch.zeros((0, JOINT_ACTION_COUNT))})
    return scores


def fixed_history_change(before: Sequence[Dict[str, object]],
                         after: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for a, b in zip(before, after):
        kls = torch.sum(a["joint_logs"].exp() * (a["joint_logs"] - b["joint_logs"]), dim=-1)
        rows.append({"instance": a["instance"], "success": a["success"],
                     "mean_logp_before": a["mean_logp"], "mean_logp_after": b["mean_logp"],
                     "mean_logp_change": (b["mean_logp"] - a["mean_logp"]
                                          if a["mean_logp"] is not None else None),
                     "action_agreement_before": a["accuracy"],
                     "action_agreement_after": b["accuracy"],
                     "mean_exact_joint_KL": float(kls.mean()) if kls.numel() else None,
                     "max_exact_joint_KL": float(kls.max()) if kls.numel() else None})
    return rows


# ---------------------------------------------------------------------------
# grouped gradients on the real batch (diagnostic; never updates the model)
# ---------------------------------------------------------------------------


def _flat_grad(parameters, gradients) -> torch.Tensor:
    return torch.cat([(g if g is not None else torch.zeros_like(p)).detach().flatten()
                      for p, g in zip(parameters, gradients)])


def _cosine(a: torch.Tensor, b: torch.Tensor) -> Optional[float]:
    denominator = a.norm() * b.norm()
    return float(torch.dot(a, b) / denominator) if float(denominator) > 1e-15 else None


def grouped_gradient_audit(policy: JointPolicy,
                           episodes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    """Decompose the first-epoch PPO gradient of the *actual* batch by outcome group.

    Every group shares the batch-wide advantage normalisation and the same decision
    denominator, so the gradients are additive up to floating point. The critic part is the
    production value weight times the squared-error loss; for arm B
    ``critic_shared_norm`` is exactly zero by construction.
    """

    entries = flatten(episodes)["entries"]
    parameters = list(policy.parameters())
    size = sum(parameter.numel() for parameter in parameters)
    names = ["success", "failure_short", "failure_long"]
    vectors = {f"{kind}/{name}": torch.zeros(size)
               for kind in ("actor", "critic") for name in names}
    counts: Counter = Counter()
    offset = 0
    replay_error = 0.0
    for episode in episodes:
        grouped: Dict[str, List[torch.Tensor]] = defaultdict(list)
        for decision, hidden in decision_states(policy, episode):
            entry = entries[offset]
            offset += 1
            group = ("success" if episode["success"] else
                     ("failure_short" if decision["periods"] <= 2 else "failure_long"))
            counts[group] += 1
            logp, _ = policy.evaluate(hidden, *decision["action"])
            replay_error = max(replay_error, abs(float(logp.detach()) - decision["logp"]))
            ratio = torch.exp(logp - decision["logp"])
            advantage = entry["advantage"]
            actor = -torch.minimum(ratio * advantage,
                                   ratio.clamp(1 - CLIP_RANGE, 1 + CLIP_RANGE) * advantage)
            critic = VALUE_WEIGHT * (policy.value(hidden) - entry["target"]) ** 2
            grouped[f"actor/{group}"].append(actor)
            grouped[f"critic/{group}"].append(critic)
        keys = list(grouped)
        for index, key in enumerate(keys):
            loss = torch.stack(grouped[key]).sum() / len(entries)
            gradient = torch.autograd.grad(loss, parameters, allow_unused=True,
                                           retain_graph=index < len(keys) - 1)
            vectors[key] += _flat_grad(parameters, gradient)
    shared = torch.cat([torch.full((parameter.numel(),),
                                   name.startswith(("encoder.", "gru.")), dtype=torch.bool)
                        for name, parameter in policy.named_parameters()])
    actor = sum(vectors[f"actor/{name}"] for name in names)
    critic = sum(vectors[f"critic/{name}"] for name in names)
    total = actor + critic
    report: Dict[str, object] = {
        "decisions": len(entries), "counts": dict(counts),
        "logp_replay_max_abs": replay_error,
        "gradient_norms": {key: float(value.norm()) for key, value in vectors.items()},
        "shared_gradient_norms": {key: float(value[shared].norm())
                                  for key, value in vectors.items()},
        "actor_norm": float(actor.norm()), "weighted_critic_norm": float(critic.norm()),
        "shared_actor_norm": float(actor[shared].norm()),
        "shared_critic_norm": float(critic[shared].norm()),
        "shared_actor_critic_cosine": _cosine(actor[shared], critic[shared]),
        "total_preclip_norm": float(total.norm()),
        "global_clip_scale": min(1.0, 5.0 / (float(total.norm()) + 1e-6)),
        "critic_mode": ("shared" if policy.critic_shared_grad else "stopped"),
    }
    for name in names:
        if not counts[name]:
            continue
        group_entries = [entry for entry in entries
                         if ("success" if entry["episode_success"] else
                             ("failure_short" if entry["periods"] <= 2
                              else "failure_long")) == name]
        report.setdefault("advantage_by_group", {})[name] = {
            "raw_mean": float(np.mean([entry["advantage_raw"] for entry in group_entries])),
            "normalized_mean": float(np.mean([entry["advantage"] for entry in group_entries])),
            "raw_positive_fraction": float(np.mean([entry["advantage_raw"] > 0
                                                    for entry in group_entries]))}
    return report


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
    grad_clip_scales: List[float] = []
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
            for decision, hidden in decision_states(policy, episode):
                first, second = decision["action"]
                logp, entropy = policy.evaluate(hidden, int(first), int(second))
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
        grad_clip_scales.append(min(1.0, 5.0 / (norm + 1e-6)))
        optimizer.step()
    metrics.update({
        "grad_norm_last": grad_norms[-1] if grad_norms else 0.0,
        "grad_clip_scale_last": grad_clip_scales[-1] if grad_clip_scales else 0.0,
        "ratio_mean": float(np.mean(ratios)) if ratios else 1.0,
        "ratio_max": float(np.max(ratios)) if ratios else 1.0,
        "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "value_error_mean": float(np.mean(value_errors)) if value_errors else 0.0,
        "policy_change_mean": float(np.mean(policy_changes)) if policy_changes else 0.0,
    })
    return metrics


def evaluate(policy: JointPolicy, instances: Sequence[layout.InstanceSpec], *,
              stochastic: bool, seeds: int = 3) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for instance in instances:
        for seed in range(seeds if stochastic else 1):
            generator = torch.Generator().manual_seed(seed)
            environment = E.RandomStartEnv(instance)
            episode = rollout(environment, policy, stochastic=stochastic, generator=generator)
            rows.append({"instance": instance.name, "seed": seed,
                          "success": episode["success"],
                          "return": episode["return"],
                          "periods": episode["periods"],
                         "final_goal_distance": episode["final_goal_distance"],
                         "skills_used": episode["skills_used"]})
    by_instance: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        by_instance[row["instance"]].append(1.0 if row["success"] else 0.0)
    values = np.asarray([float(np.mean(group)) for group in by_instance.values()])
    return {"episodes": len(rows), "instances": int(values.size),
             "success": float(values.mean()) if values.size else 0.0,
             "return_mean": float(np.mean([row["return"] for row in rows])) if rows else None,
            "median_final_distance": float(np.median([row["final_goal_distance"]
                                                      for row in rows])) if rows else None,
            "median_skills": float(np.median([row["skills_used"] for row in rows]))
            if rows else None,
            "per_instance": dict(by_instance), "rows": rows}


def random_baseline(instances: Sequence[layout.InstanceSpec], seeds: int = 3) -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    for instance in instances:
        for seed in range(seeds):
            rng = np.random.default_rng(seed)
            environment = E.RandomStartEnv(instance)
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
    parser.add_argument("--scan", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--eval-stochastic", action="store_true",
                        help="also evaluate three fixed behaviour seeds at each evaluation point")
    parser.add_argument("--normalize", default="batch", choices=("batch", "trajectory"))
    parser.add_argument("--save-every", type=int, default=10,
                        help="period between atomic resumable checkpoints")
    parser.add_argument("--resume", action="store_true",
                        help="continue from checkpoints/<prefix>_resume.pt if it exists")
    parser.add_argument("--prefix", default="rl", help="checkpoint file prefix")
    parser.add_argument("--subset-file", default=None,
                        help="diagnostic subset json (defaults to <out>/results/diagnostic_subset.json)")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    pool, _ = layout.build_split("L0", "train", count=args.scan)
    by_name = {instance.name: instance for instance in pool}
    subset_path = Path(args.subset_file) if args.subset_file else \
        ROOT / "outputs/random_start_reposition_v1/results/diagnostic_subset.json"
    dev: List[layout.InstanceSpec] = []
    selection = "not available"
    if subset_path.exists():
        data = json.loads(subset_path.read_text())
        dev = [by_name[name] for name in data["selected"] if name in by_name]
        selection = ("instances on which the scripted demo policy actually succeeded "
                     f"({len(data['successes'])} of {data['attempted']} episodes); "
                     f"padded without success evidence: "
                      f"{data['padded_without_success_evidence']}")
        if len(dev) != len(data["selected"]):
            raise ValueError("Explicit diagnostic subset contains unavailable instances")
        if not dev or len(set(data["selected"])) != len(dev):
            raise ValueError("Diagnostic subset must be nonempty with unique train instances")
        if not set(data["selected"]).issubset(data["successes"]):
            raise ValueError("Diagnostic subset contains instances without recorded success evidence")
    else:
        raise FileNotFoundError(subset_path)
    validation, _ = layout.build_split("L0", "validation", count=8)

    probe = E.RandomStartEnv(dev[0])
    observation_dim = int(probe.observation_fields().__len__())
    policy = JointPolicy(observation_dim)
    checkpoint = out / "checkpoints" / f"{args.prefix}_initial.pt"
    if checkpoint.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {checkpoint}; choose a fresh --out")
    if not checkpoint.exists():
        torch.save(policy.state_dict(), checkpoint)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=LEARNING_RATE, weight_decay=0.0)

    initial = copy.deepcopy(policy)
    replay_check = probability_replay_check(policy)
    ratio_check = unchanged_policy_ratio_check(policy, dev[0])
    evaluation_init = evaluate(initial, dev, stochastic=False)
    evaluation_init_stochastic = evaluate(initial, dev, stochastic=True, seeds=3)
    baseline_random = random_baseline(dev, seeds=3)
    if not (replay_check["ok"] and ratio_check["ok"]):
        raise RuntimeError("Training preflight failed; do not spend the training budget")

    resume_path = out / "checkpoints" / f"{args.prefix}_resume.pt"
    start_update = 1
    history: List[Dict[str, object]] = []
    if args.resume and not resume_path.exists():
        raise FileNotFoundError(resume_path)
    if args.resume and resume_path.exists():
        # a local, trusted training state: it contains the numpy RNG state as well as
        # tensors, so the default weights_only=True loader rejects it
        state = torch.load(resume_path, map_location="cpu", weights_only=False)
        if state.get("env_version") != ENV_VERSION:
            raise ValueError("Cannot resume a checkpoint from a different reward contract")
        policy.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        np.random.set_state(state["numpy_rng"])
        history = list(state.get("history", []))
        start_update = int(state["update"]) + 1
        print(f"resumed at update {start_update} from {resume_path}", flush=True)

    def save_resume(update_index: int) -> None:
        payload = {"update": int(update_index), "env_version": ENV_VERSION, "model": policy.state_dict(),
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
        if any(episode["termination"] in E.ANOMALY_TERMINATIONS for episode in episodes):
            raise RuntimeError("Simulation anomaly in rollout; stop instead of learning from cheap failure")
        critic_before = value_fit_metrics(policy, episodes)
        metrics = update(policy, optimizer, episodes, normalize=args.normalize)
        metrics["critic_before"] = critic_before
        metrics["critic_after"] = value_fit_metrics(policy, episodes)
        if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
            raise FloatingPointError("Non-finite model parameters; bounded run stopped")
        metrics["update"] = update_index
        metrics["rollout_success"] = float(np.mean([episode["success"] for episode in episodes]))
        metrics["rollout_return_mean"] = float(np.mean([episode["return"]
                                                        for episode in episodes]))
        metrics["rollout_skills_mean"] = float(np.mean([episode["skills_used"]
                                                         for episode in episodes]))
        metrics["rollout_periods_mean"] = float(np.mean([episode["periods"] for episode in episodes]))
        metrics["rollout_final_distance_mean"] = float(np.mean([
            episode["final_goal_distance"] for episode in episodes]))
        metrics["rollout_terminations"] = dict(Counter(episode["termination"] for episode in episodes))
        metrics["wall_seconds"] = time.time() - start
        if update_index % args.eval_every == 0 or update_index == args.updates:
            deterministic = evaluate(policy, dev, stochastic=False)
            metrics["dev_success_deterministic"] = deterministic["success"]
            metrics["dev_final_distance_median"] = deterministic["median_final_distance"]
            metrics["dev_success_per_instance"] = deterministic["per_instance"]
            metrics["dev_return_mean"] = deterministic["return_mean"]
            metrics["dev_evaluation"] = deterministic
            if args.eval_stochastic:
                metrics["dev_stochastic_evaluation"] = evaluate(policy, dev, stochastic=True, seeds=3)
            torch.save(policy.state_dict(), out / "checkpoints" / f"{args.prefix}_update_{update_index:04d}.pt")
        history.append(metrics)
        with (out / "results" / f"{args.prefix}_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
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
    passed = sustained_pass(history, evaluation_init["success"])
    report = {
        "config": {"updates": args.updates, "episodes_per_update": args.episodes_per_update,
                   "eval_every": args.eval_every, "hidden": HIDDEN, "lr": LEARNING_RATE,
                   "clip": CLIP_RANGE, "value_weight": VALUE_WEIGHT,
                    "entropy_weight": ENTROPY_WEIGHT, "gamma": GAMMA,
                    "weight_decay": 0.0,
                    "control_dt": CONTROL_DT, "normalize": args.normalize,
                    "threads": args.threads, "eval_stochastic": args.eval_stochastic,
                    "env_version": ENV_VERSION, "reward_config": probe.config.as_dict(),
                   "advantage_estimator": "Monte-Carlo semi-Markov return minus value; no GAE",
                    "gradient_truncation": "detach after each pre-action actor/critic head; "
                                           "next decision trains on intervening skill frames",
                    "training_semantics_version": "pre_action_value_tbptt_v2",
                    "seed": args.seed},
        "dev_subset": {"selection": selection, "instances": [instance.name for instance in dev]},
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
                       "replay_ok": replay_check["ok"],
                      "ratio_ok": ratio_check["ok"]}, indent=1))
    print("RL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
