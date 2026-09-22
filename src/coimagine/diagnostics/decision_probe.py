"""Decision-level probe: what does a learned action model cost a planner?

The comparison is built so that two different failures can be told apart:

* a **decision rule** that is wrong for the task (predicting motion instead of
  progress, or ignoring preparation), measured by the *matched analytic* rule that
  uses the same rule with the correct model;
* a **model** that is wrong, measured as the extra regret the learned model incurs
  under the identical rule.

Action values are computed by the simulator, so a preparation action such as release
earns its place through the dynamics rather than through an assertion. Regret is the
true-model value of the chosen action minus that of the best action.
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
    full_value,
    implied_history,
)
from ..learning.data import (
    Situation,
    action_labels,
    candidates_for,
    generate_situations,
    pose_to_index,
    sample_vector,
)
from ..learning.features import ACTION_IDS
from ..learning.mlp import MultiHeadMLP, train

#: Policy names in report order. ``value_optimal`` and the ``analytic_*`` rows are the
#: equal-information references; the ``learned_*`` rows are the models under test.
POLICIES: Tuple[str, ...] = (
    "value_optimal",
    "analytic_progress",
    "analytic_progress_enable",
    "learned_moved",
    "learned_progress",
    "learned_progress_enable",
    "learned_value",
)

#: Channel indices of the action head.
CHANNEL_MOVED, CHANNEL_PROGRESS, CHANNEL_ENABLES = 0, 1, 2


@dataclass
class DecisionRecord:
    scene_id: int
    true_physics: str
    oracle_action: str
    oracle_value: float
    chosen: Dict[str, str]
    regret: Dict[str, float]
    teammate_slips: int
    constrained: int
    heavy: int


@dataclass
class PolicySummary:
    name: str
    n: int
    mean_regret: float
    agreement: float


@dataclass
class DecisionProbeResult:
    summaries: List[PolicySummary]
    by_mode: Dict[str, List[PolicySummary]]
    records: List[DecisionRecord] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def _value_of_action(
    ctx: ContactContext,
    hyp: HypKey,
    action_id: str,
    candidates,
    *,
    depth: int,
    fail_cost: float,
) -> float:
    candidate = candidates.by_id(action_id)
    successor, result = _execute_cached(ctx.scene(hyp), candidate.joint)
    if successor.is_goal():
        return result.cost
    return result.cost + full_value(
        ctx,
        (hyp[0], successor.released),
        candidates,
        depth - 1,
        fail_cost=fail_cost,
        start_origin=successor.obj.origin,
        start_robots=successor.robots,
    )


def posterior_over_physics(
    situation: Situation, *, noise: Optional[FeedbackNoise] = None
) -> Dict[int, float]:
    """Posterior over the 16 contact types given the robot's own lead-in feedback.

    Computed by simulating each hypothesis through the *same* lead-in actions and
    scoring the observed feedback under the feedback channel. The prior is uniform,
    and a hypothesis whose implied history cannot be formed gets zero weight.
    """

    noise = noise or FeedbackNoise()
    initial_ctx = ContactContext(
        situation.initial_scene.static,
        situation.initial_scene.obj.origin,
        situation.initial_scene.robots,
    )
    weights: Dict[int, float] = {}
    for index in range(len(PHYSICS)):
        hypothesis = (index, False)
        likelihood = 1.0
        for robot_id, observed in ((0, situation.history_a), (1, situation.history_b)):
            nominal = implied_history(initial_ctx, situation.skills, hypothesis, robot_id)
            if len(nominal) != len(observed):
                likelihood = 0.0
                break
            for seen, expected in zip(observed, nominal):
                likelihood *= feedback_likelihood(seen, expected, noise)
        weights[index] = likelihood
    total = sum(weights.values())
    if total <= 0.0:
        return {index: 1.0 / len(weights) for index in weights}
    return {index: value / total for index, value in weights.items()}


def _features_for(situation: Situation, action_id: str) -> Tuple[float, ...]:
    return sample_vector(
        situation.scan,
        situation.scene.robot_pose(0),
        situation.history_a,
        situation.history_b,
        action_id,
    )


def _argmin(values: Dict[str, float]) -> str:
    return min(ACTION_IDS, key=lambda a: (values[a], a))


def _argmax(values: Dict[str, float]) -> str:
    return max(ACTION_IDS, key=lambda a: (values[a], a))


def situation_quantities(
    situation: Situation,
    *,
    depth: int = 1,
    fail_cost: float = DEFAULT_FAIL_COST,
    candidates=None,
):
    """Everything the probe needs about one decision point, under one code path.

    Training targets and evaluation references are produced here, so a learned value
    head is trained against exactly the quantity the reference policy minimises.
    """

    candidates = candidates if candidates is not None else candidates_for()
    ctx = ContactContext(
        situation.scene.static, situation.scene.obj.origin, situation.scene.robots
    )
    true_hyp = (PHYSICS.index(situation.physics), False)
    true_values = {
        action: _value_of_action(
            ctx, true_hyp, action, candidates, depth=depth, fail_cost=fail_cost
        )
        for action in ACTION_IDS
    }
    posterior = posterior_over_physics(situation)
    belief_values = {action: 0.0 for action in ACTION_IDS}
    belief_progress = {action: 0.0 for action in ACTION_IDS}
    belief_enables = {action: 0.0 for action in ACTION_IDS}
    for index, weight in posterior.items():
        if weight <= 0.0:
            continue
        hypothesis = (index, False)
        scene = ctx.scene(hypothesis)
        for action in ACTION_IDS:
            belief_values[action] += weight * _value_of_action(
                ctx, hypothesis, action, candidates, depth=depth, fail_cost=fail_cost
            )
            _moved, progress, enables = action_labels(scene, action)
            belief_progress[action] += weight * progress
            belief_enables[action] += weight * enables
    return ctx, true_values, belief_values, belief_progress, belief_enables


def evaluate_decisions(
    situations: Sequence[Situation],
    model: MultiHeadMLP,
    *,
    depth: int = 1,
    fail_cost: float = DEFAULT_FAIL_COST,
) -> List[DecisionRecord]:
    candidates = candidates_for()
    records: List[DecisionRecord] = []

    for situation in situations:
        (
            _ctx,
            true_values,
            belief_values,
            belief_progress,
            belief_enables,
        ) = situation_quantities(
            situation, depth=depth, fail_cost=fail_cost, candidates=candidates
        )
        oracle_action = _argmin(true_values)

        obs = np.asarray(
            [_features_for(situation, action) for action in ACTION_IDS], dtype=np.float64
        )
        outcomes = model.predict_outcomes(obs[:, : model.obs_dim], obs[:, model.obs_dim:])
        learned = {action: outcomes[i] for i, action in enumerate(ACTION_IDS)}

        chosen = {
            "value_optimal": _argmin(belief_values),
            "analytic_progress": _argmax(belief_progress),
            "analytic_progress_enable": _argmax(
                {a: belief_progress[a] + belief_enables[a] for a in ACTION_IDS}
            ),
            "learned_moved": _argmax(
                {a: float(learned[a][CHANNEL_MOVED]) for a in ACTION_IDS}
            ),
            "learned_progress": _argmax(
                {a: float(learned[a][CHANNEL_PROGRESS]) for a in ACTION_IDS}
            ),
            "learned_progress_enable": _argmax(
                {
                    a: float(learned[a][CHANNEL_PROGRESS] + learned[a][CHANNEL_ENABLES])
                    for a in ACTION_IDS
                }
            ),
        }
        predicted_value = model.predict_value(
            obs[:, : model.obs_dim], obs[:, model.obs_dim:]
        )
        chosen["learned_value"] = _argmin(
            {a: float(predicted_value[i]) for i, a in enumerate(ACTION_IDS)}
        )
        oracle_value = true_values[oracle_action]
        records.append(
            DecisionRecord(
                scene_id=situation.scene_id,
                true_physics=situation.physics.label,
                oracle_action=oracle_action,
                oracle_value=oracle_value,
                chosen=chosen,
                regret={
                    name: true_values[action] - oracle_value
                    for name, action in chosen.items()
                },
                teammate_slips=int(situation.physics.traction[1] == "slip"),
                constrained=int(situation.physics.constraint == "constrained"),
                heavy=int(situation.physics.resistance >= 2),
            )
        )
    return records


def _summaries(name: str, records: Sequence[DecisionRecord]) -> List[PolicySummary]:
    if not records:
        return [PolicySummary(p, 0, float("nan"), float("nan")) for p in POLICIES]
    return [
        PolicySummary(
            name=policy,
            n=len(records),
            mean_regret=float(np.mean([r.regret[policy] for r in records])),
            agreement=float(
                np.mean([r.chosen[policy] == r.oracle_action for r in records])
            ),
        )
        for policy in POLICIES
    ]


def run_decision_probe(
    *,
    n_train_situations: int = 300,
    n_test_situations: int = 120,
    lead_in: int = 2,
    epochs: int = 400,
    seed: int = 0,
    depth: int = 1,
    exclude_slipping_from_training: bool = False,
) -> DecisionProbeResult:
    started = time.perf_counter()
    train_situations = generate_situations(
        n_scenes=n_train_situations, lead_in=lead_in, seed=seed
    )
    test_situations = generate_situations(
        n_scenes=n_test_situations, lead_in=lead_in, seed=seed + 10_000
    )
    if exclude_slipping_from_training:
        train_situations = [
            s for s in train_situations if s.physics.traction[1] != "slip"
        ]

    features: List[Tuple[float, ...]] = []
    targets: List[List[float]] = []
    values: List[float] = []
    physics_index: List[int] = []
    pose_index: List[int] = []
    for situation in train_situations:
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
    t = np.asarray(targets, dtype=np.float64)
    model = MultiHeadMLP(seed=seed)
    train(
        model,
        x[:, : model.obs_dim],
        x[:, model.obs_dim:],
        t,
        np.asarray(physics_index, dtype=np.int64),
        np.asarray(pose_index, dtype=np.int64),
        np.asarray(values, dtype=np.float64),
        epochs=epochs,
        seed=seed,
    )

    records = evaluate_decisions(test_situations, model, depth=depth)
    groups: Dict[str, List[DecisionRecord]] = defaultdict(list)
    for record in records:
        if record.teammate_slips:
            groups["teammate_slips"].append(record)
        if record.constrained:
            groups["constrained"].append(record)
        if record.heavy:
            groups["heavy"].append(record)

    return DecisionProbeResult(
        summaries=_summaries("all", records),
        by_mode={
            name: _summaries(name, group) for name, group in sorted(groups.items())
        },
        records=records,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_train_situations": len(train_situations),
            "n_test_situations": len(test_situations),
            "lead_in": lead_in,
            "epochs": epochs,
            "depth": depth,
            "seed": seed,
            "exclude_slipping_from_training": exclude_slipping_from_training,
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


def format_decision_markdown(result: DecisionProbeResult) -> str:
    lines = [
        "| group | policy | n | mean regret | agreement with oracle |",
        "|---|---|---|---|---|",
    ]
    for summary in result.summaries:
        lines.append(
            f"| all | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
            f"{summary.agreement:.3f} |"
        )
    for group, summaries in result.by_mode.items():
        for summary in summaries:
            lines.append(
                f"| {group} | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
                f"{summary.agreement:.3f} |"
            )
    return "\n".join(lines)


def write_decision_report(result: DecisionProbeResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "decision_probe.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "summaries": [asdict(s) for s in result.summaries],
                "by_mode": {
                    k: [asdict(s) for s in v] for k, v in result.by_mode.items()
                },
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "decision_probe.md"
    md_path.write_text(
        "# Decision-level probe: what does a learned action model cost?\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_train_situations']} training situations, "
        f"{result.meta['n_test_situations']} test situations, lead-in "
        f"{result.meta['lead_in']}, lookahead depth {result.meta['depth']}\n"
        f"- wall clock: {result.meta['wall_clock_seconds']:.1f}s\n\n"
        + format_decision_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
