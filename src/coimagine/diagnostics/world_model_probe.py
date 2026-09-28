"""Evaluation of the learned world models: multi-step prediction and closed-loop planning.

Two questions:

1. **Does imagined rollout stay accurate?** Roll the model forward ``k`` steps from a real
   observation and compare the imagined quantities with what the simulator did. Reported for
   the latent model, the direct model, and a no-change baseline, so the numbers mean something.
2. **Can the model be planned with?** Replace the simulator in the planner's lookahead with the
   learned model, keep the simulator for execution, and compare success and cost against the
   exact belief-space reference and the full-information oracle.

The planner's lookahead *never* touches the simulator in the learned conditions: the only thing
it sees is what the model imagines. That is the whole point of the exercise.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import numpy as np

from ..grid.dynamics import execute
from ..grid.planner import PHYSICS, DEFAULT_FAIL_COST
from ..grid.sensing import SensorNoise, apply_noise, ray_cast
from ..grid.state import Scene, enumerate_hidden_physics
from ..learning.data import _joint_for, generate_situations, pose_to_index
from ..learning.features import ACTION_IDS, OBSERVATION_DIM, observation_vector
from ..learning.world_model import (
    N_CELLS,
    DirectWorldModel,
    LatentWorldModel,
    load_world_model,
    one_hot_actions,
    save_world_model,
)
from ..learning.world_model_data import collect_trajectories

#: The planner's horizon when it plans with a learned model. Kept short on purpose: imagined
#: accuracy decays, and the report should show where.
PLAN_DEPTH = 2
#: How much a step of imagined progress towards the goal is worth, in the same units as the
#: action costs. Mirrors the exact planner's progress weight so the comparison is about the
#: model, not about a differently tuned objective.
PROGRESS_WEIGHT = 3.0
#: Charged when the object is not on the goal at the horizon, like the exact planner's cut.
FAIL_COST = DEFAULT_FAIL_COST


@dataclass
class StepPrediction:
    """What a model imagined versus what the simulator did."""

    depth: int
    observation_error: float
    pose_correct: int
    moved_correct: int
    pose_top1: int
    physics_correct: int


@dataclass
class RolloutAccuracy:
    depths: List[int] = field(default_factory=list)
    observation_error: Dict[str, List[float]] = field(default_factory=dict)
    pose_accuracy: Dict[str, List[float]] = field(default_factory=dict)
    moved_accuracy: Dict[str, List[float]] = field(default_factory=dict)
    samples: int = 0
    #: Reference levels, so accuracy numbers are readable: a pose head that always answers
    #: "not moved" scores the base rate and has learned nothing.
    chance_pose: List[float] = field(default_factory=list)
    base_moved: List[float] = field(default_factory=list)


def evaluate_rollout_accuracy(
    models: Dict[str, object],
    *,
    n_episodes: int = 60,
    max_depth: int = 4,
    seed: int = 90_000,
    policy: str = "mixed",
) -> RolloutAccuracy:
    """Roll each model forward from real observations and score it at every depth."""

    trajectories = collect_trajectories(n_episodes=n_episodes, horizon=max_depth + 1, seed=seed)
    accuracy = RolloutAccuracy(depths=list(range(1, max_depth + 1)))
    for name in list(models) + ["no_change", "simulator"]:
        accuracy.observation_error[name] = [0.0] * max_depth
        accuracy.pose_accuracy[name] = [0.0] * max_depth
        accuracy.moved_accuracy[name] = [0.0] * max_depth

    for name in list(models) + ["no_change", "simulator"]:
        accuracy.pose_accuracy.setdefault(name, [0.0] * max_depth)
        accuracy.moved_accuracy.setdefault(name, [0.0] * max_depth)
    chance_pose = [0.0] * max_depth
    base_moved = [0.0] * max_depth

    counts = [0] * max_depth
    for trajectory in trajectories:
        steps = trajectory.transitions
        if len(steps) < max_depth:
            continue
        accuracy.samples += 1
        start = steps[0]
        observation = start.observation
        actions = [step.action for step in steps[:max_depth]]
        truths = steps[:max_depth]

        imagined: Dict[str, List[np.ndarray]] = {}
        heads: Dict[str, List[Dict[str, np.ndarray]]] = {}
        for name, model in models.items():
            # One imagined rollout per condition, obtained the same way a planner would.
            rollout = model.imagine(observation, actions)
            heads[name] = rollout
            imagined[name] = [step["observation"][0] for step in rollout]
        imagined["no_change"] = [observation] * max_depth

        for depth in range(max_depth):
            counts[depth] += 1
            truth = truths[depth]
            base_moved[depth] += float(truth.moved)
            for name in list(models) + ["no_change", "simulator"]:
                if name == "simulator":
                    predicted_observation = truth.next_observation
                else:
                    predicted_observation = imagined[name][depth]
                error = float(
                    ((predicted_observation - truth.next_observation) ** 2).mean()
                )
                accuracy.observation_error[name][depth] += error
                # What the planner actually consumes: where the object will be, and whether
                # this action does anything. A raw observation MSE is dominated by the static
                # cells and is not the quantity a planner needs.
                if name in heads:
                    pose_logits = heads[name][depth]["pose_logits"][0]
                    accuracy.pose_accuracy[name][depth] += float(
                        pose_logits.argmax() == truth.next_object_cell
                    )
                    logit = float(heads[name][depth]["moved_logit"][0])
                    probability = 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))
                    accuracy.moved_accuracy[name][depth] += float(
                        (probability >= 0.5) == (truth.moved >= 0.5)
                    )
                elif name == "simulator":
                    accuracy.pose_accuracy[name][depth] += 1.0
                    accuracy.moved_accuracy[name][depth] += 1.0
    for name in accuracy.observation_error:
        for depth in range(max_depth):
            count = max(1, counts[depth])
            accuracy.observation_error[name][depth] /= count
            accuracy.pose_accuracy[name][depth] /= count
            accuracy.moved_accuracy[name][depth] /= count
    accuracy.chance_pose = [1.0 / 36.0] * max_depth
    accuracy.base_moved = [value / max(1, counts[d]) for d, value in enumerate(base_moved)]
    return accuracy


# ---------------------------------------------------------------------------
# Planning with a learned model
# ---------------------------------------------------------------------------


def _goal_cells(scene: Scene) -> frozenset:
    """The goal region, which every robot knows from the prior map."""

    return frozenset(scene.static.target)


def _worst_cell_distance(origin: Tuple[int, int], goal: frozenset) -> float:
    """Steps needed for the worst cell of a 2x2 object at ``origin`` to reach the goal."""

    cells = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
    if all(cell in goal for cell in cells):
        return 0.0
    return float(min(max(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in cells))


def pose_index_to_origin(index: int) -> Tuple[int, int]:
    return (index % 6, index // 6)


def _heuristic_from_pose_logits(pose_logits: np.ndarray, goal: frozenset) -> float:
    """Expected worst-cell distance under the model's predicted pose distribution."""

    shifted = pose_logits - pose_logits.max()
    probabilities = np.exp(shifted) / np.exp(shifted).sum()
    total = 0.0
    for index, probability in enumerate(probabilities):
        if probability < 1e-4:
            continue
        total += probability * _worst_cell_distance(pose_index_to_origin(index), goal)
    return float(total)


def plan_with_model(
    model,
    observation: np.ndarray,
    goal: frozenset,
    *,
    depth: int = PLAN_DEPTH,
    progress_weight: float = PROGRESS_WEIGHT,
    fail_cost: float = FAIL_COST,
) -> Tuple[int, List[Dict[str, object]]]:
    """Choose an action by imagining forward with the learned model.

    Returns the chosen action index and, for the visualiser, the score of every first action
    together with its imagined prediction. The simulator is not consulted anywhere here.
    """

    branches: List[Dict[str, object]] = []
    for action in range(len(ACTION_IDS)):
        first = model.predict_next(observation, action)
        # cost of the first step, plus the best continuation from the imagined state
        best_continuation = None
        if depth <= 1:
            terminal = first
        else:
            for continuation in range(len(ACTION_IDS)):
                second = model.predict_next(first["observation"], continuation)
                value = float(second["cost"][0]) + progress_weight * _heuristic_from_pose_logits(
                    second["pose_logits"][0], goal
                )
                if best_continuation is None or value < best_continuation[0]:
                    best_continuation = (value, continuation, second)
            terminal = best_continuation[2]
        expected_distance = _heuristic_from_pose_logits(terminal["pose_logits"][0], goal)
        # A model that predicts the object has arrived is claiming success; otherwise charge the
        # failure cost exactly as the exact planner's cut does.
        terminal_cost = 0.0 if expected_distance < 0.5 else fail_cost
        score = (
            float(first["cost"][0])
            + (0.0 if best_continuation is None else best_continuation[0])
            + progress_weight * expected_distance
            + terminal_cost
        )
        branches.append(
            {
                "action": ACTION_IDS[action],
                "action_index": action,
                "score": score,
                "first_cost": float(first["cost"][0]),
                "expected_distance": expected_distance,
                "imagined_observation": first["observation"][0],
                "pose_logits": first["pose_logits"][0],
                "moved_probability": float(
                    1.0 / (1.0 + np.exp(-np.clip(first["moved_logit"][0], -30, 30)))
                ),
                "continuation": None if best_continuation is None else best_continuation[1],
            }
        )
    chosen = min(branches, key=lambda b: (b["score"], b["action"]))
    return int(chosen["action_index"]), branches


@dataclass
class ClosedLoopRow:
    policy: str
    episodes: int
    success_rate: float
    mean_cost: float
    cost_per_success: float
    mean_steps: float


def run_closed_loop(
    model=None,
    *,
    policy: str = "learned_model",
    n_episodes: int = 60,
    horizon: int = 8,
    radius: int = 5,
    seed: int = 90_000,
    lead_in: int = 2,
    depth: int = PLAN_DEPTH,
) -> ClosedLoopRow:
    """Run episodes where the planner imagines with ``model`` and the simulator executes."""

    situations = generate_situations(
        n_scenes=n_episodes, lead_in=lead_in, seed=seed, physics_pool=enumerate_hidden_physics()
    )
    noise = SensorNoise()
    rng = np.random.default_rng(seed)
    successes: List[int] = []
    costs: List[float] = []
    step_counts: List[int] = []

    for situation in situations:
        scene = situation.scene
        goal = _goal_cells(scene)
        history_a = list(situation.history_a)
        public_history = list(situation.public_history)
        total = 0.0
        steps = 0
        for step in range(horizon):
            if scene.is_goal():
                break
            scan = apply_noise(ray_cast(scene, 0, radius=radius), noise, rng)
            observation = np.asarray(
                observation_vector(scan, scene.robot_pose(0), tuple(history_a), tuple(public_history))
            )
            if policy == "learned_model":
                action_index, _branches = plan_with_model(model, observation, goal, depth=depth)
                action_id = ACTION_IDS[action_index]
            elif policy == "random":
                action_id = ACTION_IDS[int(rng.integers(len(ACTION_IDS)))]
            elif policy == "always_co_push":
                action_id = "co_push:E"
            else:
                raise ValueError(policy)
            successor, result = execute(scene, _joint_for(scene, action_id))
            total += float(result.cost)
            steps += 1
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            scene = successor
        successes.append(int(scene.is_goal()))
        costs.append(total)
        step_counts.append(steps)

    rate = float(np.mean(successes))
    mean_cost = float(np.mean(costs))
    return ClosedLoopRow(
        policy=policy,
        episodes=len(successes),
        success_rate=rate,
        mean_cost=mean_cost,
        cost_per_success=(mean_cost / rate if rate > 0 else float("inf")),
        mean_steps=float(np.mean(step_counts)),
    )


def format_closed_loop(rows: Sequence[ClosedLoopRow]) -> str:
    lines = [
        "| planner | episodes | success rate | mean cost | cost per success | mean steps |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        per = "n/a" if row.cost_per_success == float("inf") else f"{row.cost_per_success:.2f}"
        lines.append(
            f"| {row.policy} | {row.episodes} | {row.success_rate:.3f} | "
            f"{row.mean_cost:.3f} | {per} | {row.mean_steps:.2f} |"
        )
    return "\n".join(lines)


def format_accuracy(accuracy: RolloutAccuracy) -> str:
    lines = [
        "Mean squared error of the imagined observation against what the simulator produced, "
        "by imagined depth. `no change` repeats the current observation.",
        "",
        "| model | " + " | ".join(f"step {d}" for d in accuracy.depths) + " |",
        "|---" * (len(accuracy.depths) + 1) + "|",
    ]
    for name, errors in accuracy.observation_error.items():
        cells = " | ".join(f"{value:.4f}" for value in errors)
        lines.append(f"| {name} | {cells} |")
    lines += [
        "",
        "Object-position accuracy and action-outcome accuracy at each imagined depth -- the "
        "quantities the planner actually reads.",
        "",
        "| model | " + " | ".join(f"pose@{d}" for d in accuracy.depths) + " | "
        + " | ".join(f"moved@{d}" for d in accuracy.depths) + " |",
        "|---" * (2 * len(accuracy.depths) + 1) + "|",
    ]
    for name in accuracy.pose_accuracy:
        pose = " | ".join(f"{v:.3f}" for v in accuracy.pose_accuracy[name])
        moved = " | ".join(f"{v:.3f}" for v in accuracy.moved_accuracy[name])
        lines.append(f"| {name} | {pose} | {moved} |")
    lines.append(
        "| reference | "
        + " | ".join(f"{v:.3f}" for v in accuracy.chance_pose)
        + " | "
        + " | ".join(f"{v:.3f}" for v in accuracy.base_moved)
        + " |"
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The diagnostic that explains the planning failure
# ---------------------------------------------------------------------------


@dataclass
class ActionSensitivity:
    """How much the model's imagination depends on the action.

    This is the quantity that decides whether a learned model can be planned with. A model can
    be an excellent state estimator -- predicting where the object is -- and still be useless to
    a planner, if it predicts the *same* consequence for every candidate action.
    """

    distinct_poses: float          # how many different next-cells it names across the actions
    pose_accuracy: float           # accuracy under the action that was actually taken
    branch_entropy: float          # mean entropy of the pose distribution, per branch
    expected_cell_spread: float    # spread of the expected cell across branches, in cells
    branch_variance: float         # variance of the imagined observation across branches
    error_at_true_action: float    # squared error of the imagined observation under the truth
    error_no_change: float         # squared error of repeating the current observation


def measure_action_sensitivity(
    model, trajectories: Sequence[Trajectory], *, stride: int = 1
) -> ActionSensitivity:
    """Imagine every action from real states and measure how much the answer moves."""

    steps = [s for t in trajectories for s in t.transitions][::stride]
    observations = np.stack([s.observation for s in steps])
    next_pose = np.asarray([s.next_object_cell for s in steps])
    taken = np.asarray([s.action for s in steps])
    true_next = np.stack([s.next_observation for s in steps])

    distinct: List[int] = []
    accuracy: List[int] = []
    entropies: List[float] = []
    spreads: List[float] = []
    variances: List[float] = []
    errors_true: List[float] = []
    errors_hold: List[float] = []

    for index in range(len(steps)):
        rollout = [model.predict_next(observations[index : index + 1], a) for a in range(len(ACTION_IDS))]
        logits = np.stack([entry["pose_logits"][0] for entry in rollout])
        tops = logits.argmax(axis=1)
        distinct.append(len(set(tops.tolist())))
        accuracy.append(int(tops[taken[index]] == next_pose[index]))
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        entropies.append(float(-(probabilities * np.log(probabilities + 1e-12)).sum(axis=1).mean()))
        expected = probabilities @ np.arange(N_CELLS)
        spreads.append(float(expected.std()))
        imagined = np.stack([entry["observation"][0] for entry in rollout])
        variances.append(float(imagined.var(axis=0).mean()))
        errors_true.append(float(((imagined[taken[index]] - true_next[index]) ** 2).mean()))
        errors_hold.append(float(((observations[index] - true_next[index]) ** 2).mean()))

    return ActionSensitivity(
        distinct_poses=float(np.mean(distinct)),
        pose_accuracy=float(np.mean(accuracy)),
        branch_entropy=float(np.mean(entropies)),
        expected_cell_spread=float(np.mean(spreads)),
        branch_variance=float(np.mean(variances)),
        error_at_true_action=float(np.mean(errors_true)),
        error_no_change=float(np.mean(errors_hold)),
    )


def format_action_sensitivity(rows: Dict[str, ActionSensitivity]) -> str:
    lines = [
        "How much the imagination depends on the action. `distinct next poses` counts how many "
        "different object cells the model names across the six candidate actions from one "
        "state: a value near 1 means the model is indifferent to the action. `uniform` is the "
        "entropy of a flat 36-way distribution.",
        "",
        "| model | distinct next poses | pose accuracy | branch entropy | expected-cell spread | "
        "branch variance | error at true action | error of no change |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row.distinct_poses:.2f} of 6 | {row.pose_accuracy:.3f} | "
            f"{row.branch_entropy:.2f} | {row.expected_cell_spread:.3f} | "
            f"{row.branch_variance:.5f} | {row.error_at_true_action:.5f} | "
            f"{row.error_no_change:.5f} |"
        )
    lines.append(
        f"| uniform reference | - | {1 / N_CELLS:.3f} | {np.log(N_CELLS):.2f} | - | - | - | - |"
    )
    return "\n".join(lines)
