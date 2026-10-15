"""Corrected option-level RL: one rollout path, one mask, one probability definition.

What was wrong before, and what this module does instead:

* **History.** The sampler advanced the recurrent state only at skill boundaries while the update
  replayed the whole sequence, so the recomputed probabilities differed from the sampled ones by
  roughly a factor of twenty. Here a single ``advance`` helper consumes the observation buffer, and
  both the sampler and the update call it, so the hidden state at decision *i* is by construction the
  state after consuming ``buffer[:index_i + 1]``.
* **Masking.** Probabilities are taken from ``masked_logits`` inside the one function that both paths
  use; there is no second, unmasked softmax in the update.
* **Behaviour versus policy probability.** When exploration mixes in a uniform legal action, the
  recorded ``behaviour_logprob`` is the log-probability of the action under the **mixture actually
  used**, and ``policy_logprob`` is kept separately. Training uses on-policy sampling
  (``epsilon=0``), so the two coincide; the mixture is reserved for later data collection.
* **Rewards.** A skill of ``n`` control periods contributes
  ``gamma**n * Phi(next) - Phi(current) - c * sum_j gamma**j`` -- the boundary shaping form the task
  book prescribes, with the time cost discounted per period, and the same ``gamma`` for both.
* **Termination.** ``success`` and ``task_budget`` are real terminals; only a **sampling cut**
  bootstraps, and it bootstraps from the value of the final legal history, not from the value at the
  last decision.
"""

from __future__ import annotations

import math
import sys
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
    ObservationSpec,
    default_goal,
)
from coimagine.local_contact_option.roots import RootSpec  # noqa: E402

#: outcomes, kept apart because they bootstrap differently
OUTCOME_SUCCESS = "success"
OUTCOME_TASK_BUDGET = "task_budget"
OUTCOME_TRUNCATED = "sampling_truncation"
#: a real terminal does not bootstrap; only a sampling cut does
BOOTSTRAPS = {OUTCOME_TRUNCATED}


class RecurrentPolicy(nn.Module):
    """Encoder + GRU + discrete policy head + value head over the same history."""

    def __init__(self, observation_width: int, action_count: int, hidden: int = 128,
                 encoder: int = 128):
        super().__init__()
        self.observation_width = observation_width
        self.action_count = action_count
        self.hidden_size = hidden
        self.encoder = nn.Sequential(nn.Linear(observation_width, encoder), nn.Tanh(),
                                     nn.Linear(encoder, encoder), nn.Tanh())
        self.gru = nn.GRUCell(encoder, hidden)
        self.policy = nn.Linear(hidden, action_count)
        self.value = nn.Linear(hidden, 1)

    def initial_hidden(self, batch: int = 1, device=None) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, device=device or "cpu")

    def advance(self, hidden: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
        """Consume observations in order. The only place the recurrent state moves."""

        for index in range(observations.shape[0]):
            hidden = self.gru(self.encoder(observations[index:index + 1]), hidden)
        return hidden

    def heads(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.policy(hidden), self.value(hidden).squeeze(-1)


def masked_logits(logits: torch.Tensor, allowed: Sequence[int]) -> torch.Tensor:
    """The single masking rule; both the sampler and the update go through it."""

    mask = torch.full_like(logits, float("-inf"))
    mask[list(allowed)] = 0.0
    return logits + mask


@dataclass
class Decision:
    buffer_index: int
    action: int
    behaviour_logprob: float
    policy_logprob: float
    value: float
    reward: float
    discount: float
    skill_steps: int
    allowed: Tuple[int, ...]
    #: the reward broken into independently checkable parts: the potential difference, the time
    #: cost and the one-off task event. ``reward == shaping - time_cost + terminal`` must hold, and
    #: the verifier re-adds them from the recorded state rather than from the total.
    shaping: float = 0.0
    time_cost: float = 0.0
    terminal: float = 0.0
    task_event: Optional[str] = None
    #: the object's position before and after the skill, so the reward can be re-derived from
    #: recorded state rather than from the reward itself (the earlier "hand check" substituted the
    #: reward back into its own formula and would have passed even with the success bonus missing)
    object_before: Tuple[float, float] = (0.0, 0.0)
    object_after: Tuple[float, float] = (0.0, 0.0)
    #: True when the caller forced the action (branch collection). A forced action is an
    #: intervention, so its probability is not the policy's and must not be used as one.
    forced: bool = False


@dataclass
class Episode:
    observations: np.ndarray
    decisions: List[Decision]
    final_value: float
    outcome: str
    success: bool
    root_id: str
    scene: str
    goal: Tuple[float, float]
    start_distance: float
    end_distance: float
    total_reward: float
    exit_reasons: List[str] = field(default_factory=list)
    physics_steps: int = 0
    #: how many leading buffer frames came from the parent trajectory (branch collection)
    prefix_length: int = 0

    @property
    def net_distance_change(self) -> float:
        """``end - start``. **Positive means the object ended farther from the goal.**"""

        return self.end_distance - self.start_distance

    @property
    def net_progress_towards_goal(self) -> float:
        """``start - end``. Positive means the object got closer; the sign the report should quote."""

        return self.start_distance - self.end_distance


def legal_actions(root: Optional[RootSpec]) -> List[int]:
    if root is None:
        return list(range(len(MENU)))
    active = set(getattr(root, "active_robots", (0, 1)))
    return [index for index, action in enumerate(MENU) if set(action.active).issubset(active)]


def behaviour_probabilities(policy_probs: torch.Tensor, allowed: Sequence[int],
                            epsilon: float) -> torch.Tensor:
    """``(1 - eps) * pi + eps * Uniform(legal)`` -- the distribution actually sampled from."""

    if epsilon <= 0.0:
        return policy_probs
    mixture = policy_probs * (1.0 - epsilon)
    mixture = mixture.clone()
    weight = epsilon / len(allowed)
    for index in allowed:
        mixture[index] = mixture[index] + weight
    return mixture / mixture.sum()


def rollout(env: OptionRLEnv, policy: Optional[RecurrentPolicy], root: RootSpec,
            goal: np.ndarray, *, rng: np.random.Generator, epsilon: float = 0.0,
            deterministic: bool = False, max_skills: Optional[int] = None,
            snapshot=None, forced_first_action: Optional[int] = None,
            prefix: Optional[np.ndarray] = None,
            counters: Optional[Dict[str, int]] = None) -> Episode:
    """One episode on one world. Used for sampling, for validation and for data collection.

    ``snapshot`` restores a saved mid-trajectory state **after** the reset has configured the scene,
    and ``forced_first_action`` makes the first decision the caller's choice rather than the
    policy's. Both exist for branch collection: the previous version reset the environment after
    restoring, so every branch silently started from the initial state, and the chosen candidate was
    never passed to the executor at all.
    """

    observation = env.reset(root, goal)
    if counters:
        # a branch continues the parent's counters, otherwise the goal-hold and skill accounting
        # restart and the branch is not a continuation of the same task
        env.skills_used = int(counters.get("skills_used", 0))
        env.hold = int(counters.get("hold", 0))
        env.steps = int(counters.get("physics_steps", 0))
    if snapshot is not None:
        env.env.restore(snapshot, exact=True)
        observation = env._observation()
        # the restored state, not the root's initial state, defines the branch's starting distance
        pose = env.privileged_object()
        env.initial_object = np.asarray(pose, dtype=np.float64)
        env.start_distance = float(np.hypot(pose[0] - goal[0], pose[1] - goal[1]))
    allowed = legal_actions(root)
    # a branch starts from the parent's observation history, so the stored episode carries the
    # prefix that produced the decision rather than only a family name
    buffer: List[np.ndarray] = ([np.asarray(frame, dtype=np.float32) for frame in prefix]
                                if prefix is not None else []) + [observation]
    hidden = policy.initial_hidden(1) if policy is not None else None
    consumed = 0
    decisions: List[Decision] = []
    limit = max_skills or env.config.max_skills
    gamma = env.config.gamma_step
    cost = env.config.time_cost_per_step
    potential = env.config.potential_weight
    start_phi = potential * _phi(env, goal)
    current_phi = start_phi
    total = 0.0
    exit_reasons: List[str] = []
    outcome = OUTCOME_TRUNCATED
    success = False
    physics = 0
    while True:
        # ---- the one place the recurrent state advances: consume every buffered frame ----
        if policy is not None:
            with torch.no_grad():
                pending = torch.tensor(np.stack(buffer[consumed:]), dtype=torch.float32)
                hidden = policy.advance(hidden, pending)
                consumed = len(buffer)
                logits, value = policy.heads(hidden)
                masked = masked_logits(logits[0], allowed)
                policy_probs = torch.softmax(masked, dim=-1)
                sample_probs = behaviour_probabilities(policy_probs, allowed, epsilon)
                # exactly one mechanism: draw from the mixture itself. An extra ``if rng < eps``
                # branch on top of the mixture would square the exploration rate
                # (eps=0.1 -> 0.19, eps=0.3 -> 0.51) and make the recorded probability wrong.
                forced = forced_first_action is not None and not decisions
                if forced:
                    action = int(forced_first_action)
                elif deterministic:
                    action = int(torch.argmax(masked))
                else:
                    action = int(torch.multinomial(sample_probs, 1))
                behaviour_logprob = float(torch.log(sample_probs[action].clamp(min=1e-12)))
                policy_logprob = float(torch.log(policy_probs[action].clamp(min=1e-12)))
                value_scalar = float(value)
        else:
            if forced_first_action is not None and not decisions:
                action = int(forced_first_action)
            else:
                action = int(allowed[rng.integers(0, len(allowed))])
            behaviour_logprob = policy_logprob = -math.log(len(allowed))
            value_scalar = 0.0
            forced = forced_first_action is not None and not decisions
        decisions.append(Decision(buffer_index=len(buffer) - 1, action=action,
                                  behaviour_logprob=behaviour_logprob,
                                  policy_logprob=policy_logprob, value=value_scalar,
                                  reward=0.0, discount=1.0, skill_steps=0,
                                  allowed=tuple(allowed), forced=bool(forced)))
        pose_before = env.privileged_object()[:2].copy()
        result = env.step(action)
        inner = result.inner_observations
        # the inner frames already end at the state the skill finished in, so no extra boundary
        # frame is appended: that extra frame was what inflated the time axis by one period per skill
        if inner.shape[0]:
            buffer.extend(inner)
        else:
            buffer.append(result.observation)
        steps = max(1, int(result.skill_steps))
        discount = float(gamma ** steps)
        next_phi = potential * _phi(env, goal)
        shaping = discount * next_phi - current_phi
        time_cost = cost * _discount_sum(gamma, steps)
        event = result.info.get("task_event")
        # the one-off task reward lands on the last period of the skill, so it carries gamma**(n-1)
        # inside the skill in addition to the cross-skill discount already applied by the return
        terminal_discount = float(gamma ** (steps - 1))
        terminal = 0.0
        if event == "success":
            terminal = terminal_discount * env.config.success_bonus
        elif event == "budget_exhausted":
            terminal = terminal_discount * env.config.failure_penalty
        reward = shaping - time_cost + terminal
        current_phi = next_phi
        decisions[-1].reward = reward
        decisions[-1].discount = discount
        decisions[-1].skill_steps = steps
        decisions[-1].shaping = shaping
        decisions[-1].time_cost = time_cost
        decisions[-1].terminal = terminal
        decisions[-1].task_event = event
        decisions[-1].object_before = (float(pose_before[0]), float(pose_before[1]))
        decisions[-1].object_after = (float(env.privileged_object()[0]),
                                      float(env.privileged_object()[1]))
        total += reward
        physics += steps
        exit_reasons.append(str(env.boundaries[-1].get("exit_reason", "")))
        if result.terminated:
            outcome, success = OUTCOME_SUCCESS, True
            break
        if result.truncated:
            outcome = OUTCOME_TASK_BUDGET if result.info.get("task_budget_expired") else \
                OUTCOME_TRUNCATED
            break
        if len(decisions) >= limit:
            outcome = OUTCOME_TRUNCATED
            break
    # ---- the value a sampling cut bootstraps from: the final legal history, not the last decision
    if policy is not None:
        with torch.no_grad():
            pending = torch.tensor(np.stack(buffer[consumed:]), dtype=torch.float32)
            hidden = policy.advance(hidden, pending)
            final_value = float(policy.heads(hidden)[1])
    else:
        final_value = 0.0
    pose = env.privileged_object()
    end_distance = float(np.hypot(pose[0] - goal[0], pose[1] - goal[1]))
    return Episode(observations=np.stack(buffer), decisions=decisions, final_value=final_value,
                   prefix_length=(0 if prefix is None else len(prefix)),
                   outcome=outcome, success=success,
                   root_id=str(getattr(root, "root_id", "unknown")),
                   scene=str(getattr(root, "interaction_class", "unknown")),
                   goal=(float(goal[0]), float(goal[1])), start_distance=env.start_distance,
                   end_distance=end_distance, total_reward=total, exit_reasons=exit_reasons,
                   physics_steps=physics)


def _phi(env: OptionRLEnv, goal: np.ndarray) -> float:
    pose = env.privileged_object()
    return -float(np.hypot(pose[0] - goal[0], pose[1] - goal[1]))


def _discount_sum(gamma: float, steps: int) -> float:
    """``sum_{j=0}^{n-1} gamma**j`` -- the per-period time cost with the same clock as the shaping."""

    if abs(1.0 - gamma) < 1e-12:
        return float(steps)
    return (1.0 - gamma ** steps) / (1.0 - gamma)


def returns(episode: Episode) -> np.ndarray:
    """Semi-Markov returns inside the episode's own decision chain."""

    values = np.zeros(len(episode.decisions), dtype=np.float64)
    running = episode.final_value if episode.outcome in BOOTSTRAPS else 0.0
    for index in range(len(episode.decisions) - 1, -1, -1):
        decision = episode.decisions[index]
        running = decision.reward + decision.discount * running
        values[index] = running
    return values


def replay_logprobs(policy: RecurrentPolicy, episode: Episode) -> Tuple[np.ndarray, np.ndarray]:
    """Recompute the action log-probabilities by replaying the episode through the same path.

    With unchanged parameters this must reproduce the sampled values exactly; that equality is the
    property the previous implementation violated and the one the verifier now asserts.
    """

    observations = torch.tensor(episode.observations, dtype=torch.float32)
    hidden = policy.initial_hidden(1)
    consumed = 0
    behaviour: List[float] = []
    policy_probs_out: List[float] = []
    with torch.no_grad():
        for decision in episode.decisions:
            target = decision.buffer_index + 1
            pending = observations[consumed:target]
            hidden = policy.advance(hidden, pending)
            consumed = target
            logits, value = policy.heads(hidden)
            masked = masked_logits(logits[0], decision.allowed)
            probabilities = torch.softmax(masked, dim=-1)
            behaviour.append(float(torch.log(probabilities[decision.action].clamp(min=1e-12))))
            policy_probs_out.append(float(value))
    return np.asarray(behaviour), np.asarray(policy_probs_out)
