"""Decision-level probe: what does a learned feasibility rule cost a planner?

Round 8 evaluated the learned head per sample. This module evaluates it where it
matters: at the decision.

For every held-out decision point we score each candidate action under the **true**
physics, which gives

* the oracle action -- the argmin, i.e. regret zero by construction;
* the **analytic** action -- the argmin of the belief-weighted value, which is the
  equal-information rule the planner would use with a correct model;
* the **learned** action -- the argmax of the network's predicted feasibility.

The metric is regret: the true-model value of the chosen action minus that of the
oracle. That keeps the comparison honest, because the value of a *preparation* action
such as release is computed by the simulator, not asserted by us -- so a policy that
cannot represent "release is worth doing" shows up as regret rather than as a hidden
modelling choice.
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
    full_value,
    implied_history,
)
from ..grid.state import HiddenPhysics
from ..learning.data import Situation, candidates_for, generate_situations, pose_to_index
from ..learning.features import ACTION_IDS, sample_vector
from ..learning.mlp import MultiHeadMLP, train
from ..learning.data import sample_to_arrays


@dataclass
class DecisionRecord:
    scene_id: int
    true_physics: str
    oracle_action: str
    analytic_action: str
    learned_action: str
    analytic_greedy_action: str
    oracle_value: float
    analytic_value: float
    learned_value: float
    analytic_regret: float
    learned_regret: float
    analytic_greedy_value: float
    analytic_greedy_regret: float
    teammate_slips: int
    constrained: int
    heavy: int


@dataclass
class DecisionSummary:
    name: str
    n: int
    mean_analytic_regret: float
    mean_learned_regret: float
    analytic_agreement: float
    learned_agreement: float
    analytic_success: float
    learned_success: float
    analytic_greedy_regret: float
    analytic_greedy_agreement: float


@dataclass
class DecisionProbeResult:
    overall: DecisionSummary
    by_mode: Dict[str, DecisionSummary]
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
    """True-model value of taking ``action_id`` now, one step plus the heuristic."""

    from ..grid.planner import _execute_cached

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
    situation: Situation,
    *,
    noise: Optional[FeedbackNoise] = None,
) -> Dict[int, float]:
    """The robot's posterior over the 16 contact types given its lead-in feedback.

    Computed the honest way: simulate each hypothesis through the *same* lead-in
    actions and score the observed feedback under the feedback channel.
    """

    noise = noise or FeedbackNoise()
    initial_ctx = ContactContext(
        situation.initial_scene.static,
        situation.initial_scene.obj.origin,
        situation.initial_scene.robots,
    )
    weights: Dict[int, float] = {}
    for index, _physics in enumerate(PHYSICS):
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
        oracle_action = min(true_values, key=lambda a: (true_values[a], a))

        posterior = posterior_over_physics(situation)
        belief_values = {action: 0.0 for action in ACTION_IDS}
        for index, weight in posterior.items():
            if weight <= 0.0:
                continue
            hypothesis = (index, False)
            for action in ACTION_IDS:
                belief_values[action] += weight * _value_of_action(
                    ctx, hypothesis, action, candidates, depth=depth, fail_cost=fail_cost
                )
        analytic_action = min(belief_values, key=lambda a: (belief_values[a], a))

        # The *matched* analytic baseline: the same greedy feasibility rule the
        # network implements, but with the analytic success probability from the
        # belief. Comparing learned against analytic this way isolates the model,
        # while comparing either against the oracle isolates the decision rule.
        from ..grid.planner import _execute_cached

        moved_prob = {action: 0.0 for action in ACTION_IDS}
        for index, weight in posterior.items():
            if weight <= 0.0:
                continue
            hypothesis = (index, False)
            for action in ACTION_IDS:
                successor, result = _execute_cached(
                    ctx.scene(hypothesis), candidates.by_id(action).joint
                )
                moved_prob[action] += weight * (1.0 if result.y_public == "moved" else 0.0)
        analytic_greedy_action = max(moved_prob, key=lambda a: (moved_prob[a], a))

        obs = np.asarray(
            [_features_for(situation, action) for action in ACTION_IDS], dtype=np.float64
        )
        prob = model.predict_feasibility(obs[:, : model.obs_dim], obs[:, model.obs_dim:])
        learned_index = int(np.argmax(prob))
        learned_action = ACTION_IDS[learned_index]

        oracle_value = true_values[oracle_action]
        records.append(
            DecisionRecord(
                scene_id=situation.scene_id,
                true_physics=situation.physics.label,
                oracle_action=oracle_action,
                analytic_action=analytic_action,
                learned_action=learned_action,
                analytic_greedy_action=analytic_greedy_action,
                oracle_value=oracle_value,
                analytic_value=true_values[analytic_action],
                learned_value=true_values[learned_action],
                analytic_regret=true_values[analytic_action] - oracle_value,
                learned_regret=true_values[learned_action] - oracle_value,
                analytic_greedy_value=true_values[analytic_greedy_action],
                analytic_greedy_regret=true_values[analytic_greedy_action] - oracle_value,
                teammate_slips=int(situation.physics.traction[1] == "slip"),
                constrained=int(situation.physics.constraint == "constrained"),
                heavy=int(situation.physics.resistance >= 2),
            )
        )
    return records


def _summarise(name: str, records: Sequence[DecisionRecord]) -> DecisionSummary:
    if not records:
        return DecisionSummary(
            name, 0, float("nan"), float("nan"), float("nan"), float("nan"),
            float("nan"), float("nan"), float("nan"), float("nan"),
        )
    n = len(records)
    return DecisionSummary(
        name=name,
        n=n,
        mean_analytic_regret=float(np.mean([r.analytic_regret for r in records])),
        mean_learned_regret=float(np.mean([r.learned_regret for r in records])),
        analytic_agreement=float(np.mean([r.analytic_action == r.oracle_action for r in records])),
        learned_agreement=float(np.mean([r.learned_action == r.oracle_action for r in records])),
        analytic_success=float(np.mean([r.analytic_regret <= 1e-9 for r in records])),
        learned_success=float(np.mean([r.learned_regret <= 1e-9 for r in records])),
        analytic_greedy_regret=float(np.mean([r.analytic_greedy_regret for r in records])),
        analytic_greedy_agreement=float(
            np.mean([r.analytic_greedy_action == r.oracle_action for r in records])
        ),
    )


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
        train_situations = [s for s in train_situations if s.physics.traction[1] != "slip"]

    # Build the training tensors from the same situations the evaluation uses.
    samples = []
    for situation in train_situations:
        for action in ACTION_IDS:
            from ..grid.dynamics import execute as _execute

            from ..learning.data import _joint_for

            successor, result = _execute(situation.scene, _joint_for(situation.scene, action))
            samples.append(
                (
                    _features_for(situation, action),
                    1.0 if result.y_public == "moved" else 0.0,
                    PHYSICS.index(situation.physics),
                    pose_to_index(situation.scene.obj.origin),
                )
            )
    x = np.asarray([s[0] for s in samples], dtype=np.float64)
    y = np.asarray([s[1] for s in samples], dtype=np.float64)
    physics_index = np.asarray([s[2] for s in samples], dtype=np.int64)
    pose_index = np.asarray([s[3] for s in samples], dtype=np.int64)

    model = MultiHeadMLP(seed=seed)
    train(
        model, x[:, : model.obs_dim], x[:, model.obs_dim:], y,
        physics_index, pose_index, epochs=epochs, seed=seed,
    )

    records = evaluate_decisions(test_situations, model, depth=depth)
    overall = _summarise("all", records)
    groups: Dict[str, List[DecisionRecord]] = defaultdict(list)
    for record in records:
        if record.teammate_slips:
            groups["teammate_slips"].append(record)
        if record.constrained:
            groups["constrained"].append(record)
        if record.heavy:
            groups["heavy"].append(record)
    by_mode = {name: _summarise(name, group) for name, group in sorted(groups.items())}

    return DecisionProbeResult(
        overall=overall,
        by_mode=by_mode,
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
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_decision_markdown(result: DecisionProbeResult) -> str:
    lines = [
        "| group | n | value-optimal regret | analytic-greedy regret | learned-greedy regret |",
        "|---|---|---|---|---|",
    ]
    for summary in [result.overall] + list(result.by_mode.values()):
        lines.append(
            f"| {summary.name} | {summary.n} | {summary.mean_analytic_regret:.3f} | "
            f"{summary.analytic_greedy_regret:.3f} | {summary.mean_learned_regret:.3f} |"
        )
    lines += [
        "",
        "| group | n | value-optimal agreement | analytic-greedy agreement | learned-greedy agreement |",
        "|---|---|---|---|---|",
    ]
    for summary in [result.overall] + list(result.by_mode.values()):
        lines.append(
            f"| {summary.name} | {summary.n} | {summary.analytic_agreement:.3f} | "
            f"{summary.analytic_greedy_agreement:.3f} | {summary.learned_agreement:.3f} |"
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
                "overall": asdict(result.overall),
                "by_mode": {k: asdict(v) for k, v in result.by_mode.items()},
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)

    md_path = out / "decision_probe.md"
    md_path.write_text(
        "# Decision-level probe: what does a learned feasibility rule cost?\n\n"
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
