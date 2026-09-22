"""Closed-loop rollout: does a learned value actually drive an episode?

Rounds 9-11 compared policies at single decision points. This module closes the loop:
the chosen action is executed in the simulator, the robot takes a fresh scan and
accumulates real feedback, and the policy decides again. Nothing about the future
observation is assumed.

Policies compared on identical episodes:

* ``oracle_true_state`` -- the true state and true model: the ceiling;
* ``true_model_belief`` -- the true model, but the contact physics only as the robot's
  own posterior. This is the equal-information reference: the best any method can do
  without knowing the hidden physics;
* ``learned_value`` -- the network's predicted value, argmin;
* ``learned_moved`` -- the network's predicted "will it move", argmax (the round-9
  baseline, kept so the contrast survives into the closed loop).

The released flag is tracked per hypothesis by replaying the committed action sequence
inside each hypothesis, so release is handled rather than excluded.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid.dynamics import FeedbackNoise, feedback_likelihood
from ..grid.planner import (
    PHYSICS,
    ContactContext,
    DEFAULT_FAIL_COST,
    HypKey,
    _execute_cached,
    central_q_values,
    full_value,
    implied_history,
)
from ..grid.sensing import OccupancyScan, SensorNoise, apply_noise, ray_cast
from ..grid.state import Scene
from ..learning.data import candidates_for, generate_situations, pose_to_index
from ..learning.data import sample_vector, _joint_for, action_labels
from ..learning.features import ACTION_IDS
from ..learning.mlp import MultiHeadMLP, train
from .decision_probe import (
    CHANNEL_ENABLES,
    CHANNEL_MOVED,
    CHANNEL_PROGRESS,
    _argmin,
    _features_for,
    _value_of_action,
    posterior_over_physics,
    situation_quantities,
)

POLICIES: Tuple[str, ...] = (
    "oracle_true_state",
    "true_model_belief",
    "learned_value",
    "learned_moved",
)

HORIZON = 8

#: Lookahead depth used inside the closed loop. Depth 1 cannot value preparation at all
#: (release only pays off through the push that follows it), and depth 2 cannot contain a
#: reposition-plus-two-pushes plan, which is three actions: measured, depth 2 fails a
#: latched solvable scene that depth 3 solves in four steps.
DEPTH = 3


@dataclass
class StepTrace:
    """One decision boundary of a closed-loop episode, for rendering."""

    t: int
    scene: Scene
    scan: Optional[object]
    action: str
    y_public: str
    feedback: Dict[int, str]
    goal: bool


@dataclass
class EpisodeRecord:
    scene_id: int
    policy: str
    success: int
    cost: float
    steps: int
    true_physics: str
    constrained: int
    heavy: int
    teammate_slips: int
    #: Whether the true-state oracle solves this episode at all. Some generated
    #: episodes are simply unsolvable (both robots slipping gives zero capacity), and
    #: scoring every policy on them only dilutes the comparison.
    solvable: int = 0


@dataclass
class PolicySummary:
    name: str
    n: int
    success_rate: float
    mean_cost: float
    mean_steps: float
    #: Mean cost divided by success rate: the cost of actually finishing a task.
    cost_per_success: float


@dataclass
class RolloutProbeResult:
    summaries: List[PolicySummary]
    by_mode: Dict[str, List[PolicySummary]]
    records: List[EpisodeRecord] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def _replayed_scene(initial_ctx: ContactContext, skills, hypothesis: HypKey) -> Scene:
    """The scene one hypothesis implies after replaying the committed actions.

    This is what makes release honest: the hypothesis' own ``released`` transition is
    produced by the same dynamics, not assumed.
    """

    scene = initial_ctx.scene(hypothesis)
    for skill in skills:
        scene, _result = _execute_cached(scene, skill)
    return scene


def _belief_values(
    initial_ctx: ContactContext,
    skills,
    history_a: Sequence[str],
    history_b: Sequence[str],
    candidates,
    *,
    depth: int,
    fail_cost: float,
    noise: Optional[FeedbackNoise] = None,
) -> Dict[str, float]:
    """Belief-weighted value of each candidate at the current decision point."""

    noise = noise or FeedbackNoise()

    weights: Dict[int, float] = {}
    for index in range(len(PHYSICS)):
        likelihood = 1.0
        for robot_id, observed in ((0, history_a), (1, history_b)):
            nominal = implied_history(
                initial_ctx, skills, (index, False), robot_id
            )
            if len(nominal) != len(observed):
                likelihood = 0.0
                break
            for seen, expected in zip(observed, nominal):
                likelihood *= feedback_likelihood(seen, expected, noise)
        weights[index] = likelihood
    total = sum(weights.values())
    if total <= 0.0:
        weights = {index: 1.0 for index in weights}
        total = float(len(weights))

    # The reference is the exact belief-space backup, not a per-hypothesis average with a
    # frozen belief. The frozen version cannot value an informative action, so it hedges
    # and idles; see the round-16 correction.
    belief: Dict[object, float] = {}
    for index, weight in weights.items():
        if weight <= 0.0:
            continue
        scene = _replayed_scene(initial_ctx, skills, (index, False))
        belief[(scene.obj.origin, scene.robots, (index, scene.released))] = weight / total
    static_ctx = ContactContext(
        initial_ctx.static, initial_ctx.origin, initial_ctx.robots
    )
    return central_q_values(
        static_ctx, belief, candidates, depth, fail_cost=fail_cost
    )


def run_episode(
    initial_scene: Scene,
    policy: str,
    model: MultiHeadMLP,
    *,
    radius: int = 5,
    horizon: int = HORIZON,
    depth: int = 2,
    fail_cost: float = DEFAULT_FAIL_COST,
    sensor_noise: Optional[SensorNoise] = None,
    feedback_noise: Optional[FeedbackNoise] = None,
    rng: Optional[np.random.Generator] = None,
    trace: Optional[List[StepTrace]] = None,
) -> Tuple[int, float, int]:
    """Run one closed-loop episode and return ``(success, cost, steps)``.

    Pass a list as ``trace`` to record every boundary; the renderer uses it to show what a
    learned policy did next to the reference.
    """

    candidates = candidates_for()
    sensor_noise = sensor_noise or SensorNoise()
    feedback_noise = feedback_noise or FeedbackNoise()
    rng = rng or np.random.default_rng(0)

    initial_ctx = ContactContext(
        initial_scene.static, initial_scene.obj.origin, initial_scene.robots
    )
    true_hyp: HypKey = (PHYSICS.index(initial_scene.physics), False)

    current = initial_scene
    skills: List = []
    history_a: List[str] = []
    history_b: List[str] = []
    public_history: List[str] = []
    total_cost = 0.0

    for step in range(horizon):
        if current.is_goal():
            return 1, total_cost, step

        scan: OccupancyScan = apply_noise(
            ray_cast(current, 0, radius=radius), sensor_noise, rng
        )
        if policy == "oracle_true_state":
            ctx = ContactContext(current.static, current.obj.origin, current.robots)
            values = {
                action: _value_of_action(
                    ctx, true_hyp, action, candidates, depth=depth, fail_cost=fail_cost
                )
                for action in ACTION_IDS
            }
            action_id = _argmin(values)
        elif policy == "true_model_belief":
            values = _belief_values(
                initial_ctx, tuple(skills), tuple(history_a), tuple(history_b),
                candidates, depth=depth, fail_cost=fail_cost, noise=feedback_noise,
            )
            action_id = _argmin(values)
        else:
            obs = np.asarray(
                [
                    sample_vector(
                        scan, current.robot_pose(0), tuple(history_a),
                        tuple(public_history), action,
                    )
                    for action in ACTION_IDS
                ],
                dtype=np.float64,
            )
            if policy == "learned_value":
                predicted = model.predict_value(
                    obs[:, : model.obs_dim], obs[:, model.obs_dim:]
                )
                action_id = _argmin(
                    {a: float(predicted[i]) for i, a in enumerate(ACTION_IDS)}
                )
            elif policy in (
                "learned_moved", "learned_progress", "learned_progress_enable"
            ):
                outcomes = model.predict_outcomes(
                    obs[:, : model.obs_dim], obs[:, model.obs_dim:]
                )
                if policy == "learned_moved":
                    scores = {
                        a: float(outcomes[i][CHANNEL_MOVED])
                        for i, a in enumerate(ACTION_IDS)
                    }
                elif policy == "learned_progress":
                    scores = {
                        a: float(outcomes[i][CHANNEL_PROGRESS])
                        for i, a in enumerate(ACTION_IDS)
                    }
                else:
                    # progress plus enablement: this is the only learned rule that can
                    # choose a preparation action such as release, because release does not
                    # move the object and so never scores on the motion channel.
                    scores = {
                        a: float(
                            outcomes[i][CHANNEL_PROGRESS]
                            + outcomes[i][CHANNEL_ENABLES]
                        )
                        for i, a in enumerate(ACTION_IDS)
                    }
                action_id = max(ACTION_IDS, key=lambda a: (scores[a], a))
            else:
                raise ValueError(policy)

        joint = _joint_for(current, action_id)
        successor, result = _execute_cached(current, joint)
        total_cost += result.cost
        if trace is not None:
            trace.append(
                StepTrace(
                    t=step,
                    scene=current,
                    scan=scan,
                    action=action_id,
                    y_public=result.y_public,
                    feedback=dict(result.feedback),
                    goal=successor.is_goal(),
                )
            )
        skills.append(joint)
        history_a.append(result.feedback[0])
        history_b.append(result.feedback[1])
        public_history.append(result.y_public)
        current = successor
        # The released flag is part of the state, so the oracle's own model must
        # track it. Leaving it pinned at the initial value made a successful release
        # invisible to the oracle: it kept believing the object was still latched,
        # released again, and never pushed.
        true_hyp = (true_hyp[0], successor.released)

    return int(current.is_goal()), total_cost, horizon


def _summaries(name: str, records: Sequence[EpisodeRecord]) -> List[PolicySummary]:
    out: List[PolicySummary] = []
    for policy in POLICIES:
        subset = [r for r in records if r.policy == policy]
        if not subset:
            out.append(
                PolicySummary(policy, 0, float("nan"), float("nan"), float("nan"), float("nan"))
            )
            continue
        success_rate = float(np.mean([r.success for r in subset]))
        mean_cost = float(np.mean([r.cost for r in subset]))
        out.append(
            PolicySummary(
                name=policy,
                n=len(subset),
                success_rate=success_rate,
                mean_cost=mean_cost,
                mean_steps=float(np.mean([r.steps for r in subset])),
                cost_per_success=(
                    mean_cost / success_rate if success_rate > 0 else float("inf")
                ),
            )
        )
    return out


def fit_value_model(
    situations: Sequence,
    *,
    epochs: int = 400,
    seed: int = 0,
    depth: int = DEPTH,
) -> MultiHeadMLP:
    """Fit the value model on a set of decision points.

    Shared by every closed-loop experiment so an ablation changes only the training data,
    never the training procedure.
    """

    import numpy as np

    features: List[Tuple[float, ...]] = []
    targets: List[List[float]] = []
    values: List[float] = []
    physics_index: List[int] = []
    pose_index: List[int] = []
    for situation in situations:
        _ctx, _true, belief_values, _bp, _be = situation_quantities(
            situation, depth=depth, fail_cost=DEFAULT_FAIL_COST
        )
        for action in ACTION_IDS:
            moved, progress, enables = action_labels(situation.scene, action)
            features.append(_features_for(situation, action))
            targets.append([moved, progress, enables])
            values.append(belief_values[action])
            physics_index.append(PHYSICS.index(situation.physics))
            pose_index.append(pose_to_index(situation.scene.obj.origin))

    x = np.asarray(features, dtype=np.float64)
    model = MultiHeadMLP(seed=seed)
    train(
        model,
        x[:, : model.obs_dim],
        x[:, model.obs_dim:],
        np.asarray(targets, dtype=np.float64),
        np.asarray(physics_index, dtype=np.int64),
        np.asarray(pose_index, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        epochs=epochs,
        seed=seed,
    )
    return model


def run_rollout_probe(
    *,
    n_train_situations: int = 200,
    n_episodes: int = 60,
    epochs: int = 400,
    seed: int = 0,
    horizon: int = HORIZON,
    depth: int = DEPTH,
    lead_in: int = 2,
    feedback_noise: float = 0.0,
) -> RolloutProbeResult:
    """Train a value model, then run every policy on the same set of episodes."""

    started = time.perf_counter()
    # Training must cover the state distribution the closed loop actually visits.
    # Episodes start from a *fresh* scene (lead_in 0) but drift forward, so training on
    # post-lead-in situations alone leaves the closed-loop states out of distribution.
    train_situations = list(
        generate_situations(
            n_scenes=n_train_situations, lead_in=lead_in, seed=seed,
            feedback_noise=feedback_noise,
        )
    ) + list(
        generate_situations(
            n_scenes=n_train_situations // 2, lead_in=0, seed=seed + 77,
            feedback_noise=feedback_noise,
        )
    )
    features: List[Tuple[float, ...]] = []
    targets: List[List[float]] = []
    values: List[float] = []
    physics_index: List[int] = []
    pose_index: List[int] = []
    for situation in train_situations:
        # The value target must be computed at the same lookahead depth as the
        # planner that will consume it. At depth 1 the target cannot see that release
        # enables a later push, so it labels release as strictly worse than waiting --
        # and a model trained on that target learns to idle.
        _ctx, _true, belief_values, _bp, _be = situation_quantities(
            situation, depth=depth, fail_cost=DEFAULT_FAIL_COST
        )
        for action in ACTION_IDS:
            moved, progress, enables = action_labels(situation.scene, action)
            features.append(_features_for(situation, action))
            targets.append([moved, progress, enables])
            values.append(belief_values[action])
            physics_index.append(PHYSICS.index(situation.physics))
            pose_index.append(pose_to_index(situation.scene.obj.origin))

    x = np.asarray(features, dtype=np.float64)
    model = MultiHeadMLP(seed=seed)
    train(
        model,
        x[:, : model.obs_dim],
        x[:, model.obs_dim:],
        np.asarray(targets, dtype=np.float64),
        np.asarray(physics_index, dtype=np.int64),
        np.asarray(pose_index, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        epochs=epochs,
        seed=seed,
    )

    episodes = generate_situations(
        n_scenes=n_episodes, lead_in=0, seed=seed + 500_000,
        feedback_noise=feedback_noise,
    )
    fb_noise = FeedbackNoise(flip_prob=feedback_noise)
    records: List[EpisodeRecord] = []
    for episode_index, situation in enumerate(episodes):
        oracle_success, _cost, _steps = run_episode(
            situation.scene,
            "oracle_true_state",
            model,
            horizon=horizon,
            depth=depth,
            feedback_noise=fb_noise,
            rng=np.random.default_rng(episode_index),
        )
        for policy in POLICIES:
            success, cost, steps = run_episode(
                situation.scene,
                policy,
                model,
                horizon=horizon,
                depth=depth,
                feedback_noise=fb_noise,
                rng=np.random.default_rng(episode_index),
            )
            records.append(
                EpisodeRecord(
                    scene_id=situation.scene_id,
                    policy=policy,
                    success=success,
                    cost=cost,
                    steps=steps,
                    true_physics=situation.physics.label,
                    constrained=int(situation.physics.constraint == "constrained"),
                    heavy=int(situation.physics.resistance >= 2),
                    teammate_slips=int(situation.physics.traction[1] == "slip"),
                    solvable=oracle_success,
                )
            )

    groups: Dict[str, List[EpisodeRecord]] = defaultdict(list)
    for record in records:
        if record.solvable:
            groups["solvable"].append(record)
        if record.constrained:
            groups["constrained"].append(record)
        if record.heavy:
            groups["heavy"].append(record)
        if record.teammate_slips:
            groups["teammate_slips"].append(record)

    return RolloutProbeResult(
        summaries=_summaries("all", records),
        by_mode={name: _summaries(name, group) for name, group in sorted(groups.items())},
        records=records,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_train_situations": len(train_situations),
            "n_episodes": len(episodes),
            "epochs": epochs,
            "horizon": horizon,
            "depth": depth,
            "seed": seed,
            "feedback_noise": feedback_noise,
            "wall_clock_seconds": time.perf_counter() - started,
        },
    )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_rollout_markdown(result: RolloutProbeResult) -> str:
    lines = [
        "| group | policy | n | success rate | mean cost | mean steps | cost per success |",
        "|---|---|---|---|---|---|---|",
    ]
    def row(group: str, summary: PolicySummary) -> str:
        per = "n/a" if summary.cost_per_success == float("inf") else f"{summary.cost_per_success:.2f}"
        return (
            f"| {group} | {summary.name} | {summary.n} | {summary.success_rate:.3f} | "
            f"{summary.mean_cost:.3f} | {summary.mean_steps:.2f} | {per} |"
        )

    for summary in result.summaries:
        lines.append(row("all", summary))
    for group, summaries in result.by_mode.items():
        for summary in summaries:
            lines.append(row(group, summary))
    return "\n".join(lines)


def write_rollout_report(result: RolloutProbeResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "rollout_probe.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "summaries": [asdict(s) for s in result.summaries],
                "by_mode": {k: [asdict(s) for s in v] for k, v in result.by_mode.items()},
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "rollout_probe.md"
    md_path.write_text(
        "# Closed-loop rollout: does a learned value drive an episode?\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- trained on {result.meta['n_train_situations']} situations, "
        f"{result.meta['n_episodes']} episodes per policy, horizon {result.meta['horizon']}\n"
        f"- wall clock: {result.meta['wall_clock_seconds']:.1f}s\n\n"
        + format_rollout_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths


# ---------------------------------------------------------------------------
# Mode ablation at the episode level
# ---------------------------------------------------------------------------


@dataclass
class ModeAblationSummary:
    mode: str
    model: str
    n: int
    success_rate: float
    cost_per_success: float


def run_mode_ablation(
    *,
    n_train_situations: int = 150,
    n_episodes: int = 150,
    epochs: int = 400,
    seed: int = 0,
    horizon: int = HORIZON,
    policy: str = "learned_moved",
) -> List[ModeAblationSummary]:
    """Train with a physics mode removed from the data and measure it on that mode.

    The question is whether **training coverage** of a contact mode determines what the
    policy can do on episodes containing it, once the leaked channel is gone. Two modes are
    ablated in turn -- the latched/constraint mode and heavy resistance -- and each is
    evaluated on held-out episodes that actually contain it.
    """

    from ..grid.state import enumerate_hidden_physics

    modes = {
        "latched": lambda p: p.constraint == "constrained",
        "heavy": lambda p: p.resistance >= 2,
    }
    everything = enumerate_hidden_physics()
    out: List[ModeAblationSummary] = []

    for mode, predicate in modes.items():
        pool_full = tuple(everything)
        pool_without = tuple(p for p in everything if not predicate(p))

        models = {
            "trained_with": fit_value_model(
                generate_situations(
                    n_scenes=n_train_situations, lead_in=2, seed=seed,
                    physics_pool=pool_full,
                ),
                epochs=epochs, seed=seed,
            ),
            "trained_without": fit_value_model(
                generate_situations(
                    n_scenes=n_train_situations, lead_in=2, seed=seed,
                    physics_pool=pool_without,
                ),
                epochs=epochs, seed=seed,
            ),
        }

        episodes = generate_situations(
            n_scenes=n_episodes, lead_in=0, seed=seed + 500_000,
            physics_pool=tuple(p for p in everything if predicate(p)),
        )
        for name, model in models.items():
            successes: List[int] = []
            costs: List[float] = []
            for index, situation in enumerate(episodes):
                success, cost, _steps = run_episode(
                    situation.scene,
                    policy,
                    model,
                    horizon=horizon,
                    rng=np.random.default_rng(index),
                )
                successes.append(success)
                costs.append(cost)
            rate = float(np.mean(successes)) if successes else float("nan")
            mean_cost = float(np.mean(costs)) if costs else float("nan")
            out.append(
                ModeAblationSummary(
                    mode=mode,
                    model=f"{name} [{policy}]",
                    n=len(episodes),
                    success_rate=rate,
                    cost_per_success=(mean_cost / rate if rate > 0 else float("inf")),
                )
            )
    return out


def format_mode_ablation_markdown(summaries: Sequence[ModeAblationSummary]) -> str:
    lines = [
        "Each row is the same closed-loop policy, differing only in whether the ablated "
        "mode was present in its training data. Episodes contain that mode.",
        "",
        "| ablated mode | training data | episodes | success rate | cost per success |",
        "|---|---|---|---|---|",
    ]
    for summary in summaries:
        per = (
            "n/a"
            if summary.cost_per_success == float("inf")
            else f"{summary.cost_per_success:.2f}"
        )
        lines.append(
            f"| {summary.mode} | {summary.model} | {summary.n} | "
            f"{summary.success_rate:.3f} | {per} |"
        )
    return "\n".join(lines)
