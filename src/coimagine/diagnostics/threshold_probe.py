"""Threshold probe: infer the force a push needs, when force costs something.

Round 18 formulated the continuous case as binary classification over an (angle, force)
grid and it failed: 3.7% of the labels are positive, so the model either predicts "never
feasible" or over-predicts, and the region is wrong either way.

The physics says the right object is a **scalar**: inside the angular window a push works
exactly when ``force >= resistance / sum(traction)``. That single number is what the robot
cannot observe, so the probe regresses it directly, with two targets:

* **can push at all** -- whether a push at maximum force works (false when the robot
  slips, when the object is latched, or when the direction is blocked);
* **force threshold** -- the smallest force that works, regressed only where a push is
  possible (else the target is infinite and is masked out).

The threshold is decision-relevant only if force costs something, which it does here:
a push at force ``f`` costs ``1 + f``. So pushing too softly fails and pushing too hard
wastes energy, and the optimal choice is exactly the threshold. That makes the metric a
**regret in the task's own units**, not an abstract error.
"""

from __future__ import annotations

import json
import subprocess
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid.continuous import FORCE_MAX, FORCE_MIN, force_threshold, push_outcome
from ..grid.planner import PHYSICS
from ..learning.continuous_data import continuous_features
from ..learning.data import generate_situations
from ..learning.features import OBSERVATION_DIM
from ..learning.mlp import MultiHeadMLP, train

#: Base cost of a push; the total cost of a push at force f is BASE + f, so a heavier
#: push really is more expensive. Declared here and used identically for every policy.
PUSH_BASE_COST = 1.0

#: Regret charged when the policy pushes and the push fails, or declines a push that was
#: possible. Fixed before the comparison and identical for every policy.
FAIL_PENALTY = 2.0

#: Pushing direction used by the probe: straight east, inside the angular window.
PROBE_ANGLE = 0.0

BASELINES: Tuple[str, ...] = ("oracle", "learned", "always_max", "always_min")


@dataclass
class ThresholdRecord:
    scene_id: int
    true_physics: str
    can_push: int
    true_threshold: float
    predicted_can_push: float
    predicted_threshold: float
    mechanism: str
    regret: Dict[str, float]
    force_used: Dict[str, float]


@dataclass
class PolicySummary:
    name: str
    n: int
    mean_regret: float
    success_rate: float


@dataclass
class ThresholdProbeResult:
    summaries: List[PolicySummary]
    by_mechanism: Dict[str, List[PolicySummary]]
    threshold_mae: float
    can_push_balanced_accuracy: float
    n_pushable: int
    records: List[ThresholdRecord] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


def _targets_for(situation):
    """The two supervised targets for one decision point, plus the observables."""

    scene = situation.scene
    at_max = push_outcome(scene, {0: FORCE_MAX}, PROBE_ANGLE)
    can_push = int(at_max.moved)
    threshold = force_threshold(scene, [0])
    return scene, can_push, threshold


def run_threshold_probe(
    *,
    n_train_situations: int = 250,
    n_test_situations: int = 120,
    epochs: int = 500,
    seed: int = 0,
    lead_in: int = 2,
) -> ThresholdProbeResult:
    started = time.perf_counter()
    train_situations = generate_situations(
        n_scenes=n_train_situations, lead_in=lead_in, seed=seed
    )
    test_situations = generate_situations(
        n_scenes=n_test_situations, lead_in=lead_in, seed=seed + 70_000
    )

    def build(situations):
        features: List[Tuple[float, ...]] = []
        can_push: List[float] = []
        thresholds: List[float] = []
        mask: List[float] = []
        physics_index: List[int] = []
        poses: List[int] = []
        from ..learning.data import pose_to_index

        for situation in situations:
            scene, pushable, threshold = _targets_for(situation)
            features.append(
                continuous_features(
                    situation.scan,
                    scene.robot_pose(0),
                    situation.history_a,
                    situation.history_b,
                    PROBE_ANGLE,
                    FORCE_MAX,
                )
            )
            can_push.append(float(pushable))
            thresholds.append(threshold if np.isfinite(threshold) else 0.0)
            mask.append(float(pushable))
            physics_index.append(PHYSICS.index(situation.physics))
            poses.append(pose_to_index(scene.obj.origin))
        x = np.asarray(features, dtype=np.float64)
        return (
            x[:, :OBSERVATION_DIM],
            x[:, OBSERVATION_DIM:],
            np.asarray(can_push, dtype=np.float64),
            np.asarray(thresholds, dtype=np.float64),
            np.asarray(mask, dtype=np.float64),
            np.asarray(physics_index, dtype=np.int64),
            np.asarray(poses, dtype=np.int64),
        )

    obs, action, can_push, thresholds, mask, physics_index, poses = build(train_situations)
    positive_rate = float(can_push.mean())
    model = MultiHeadMLP(
        seed=seed,
        action_dim=action.shape[1],
        positive_weight=(1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0,
    )
    targets = np.concatenate(
        [can_push[:, None], np.zeros((len(can_push), 2))], axis=1
    )
    train(
        model, obs, action, targets, physics_index, poses,
        thresholds=thresholds, threshold_mask=mask,
        epochs=epochs, seed=seed,
    )

    # -- evaluation on held-out situations -------------------------------------
    records: List[ThresholdRecord] = []
    for situation in test_situations:
        scene, pushable, threshold = _targets_for(situation)
        feature = np.asarray(
            [
                continuous_features(
                    situation.scan, scene.robot_pose(0), situation.history_a,
                    situation.history_b, PROBE_ANGLE, FORCE_MAX,
                )
            ],
            dtype=np.float64,
        )
        obs_one = feature[:, :OBSERVATION_DIM]
        act_one = feature[:, OBSERVATION_DIM:]
        prob_push = float(model.predict_outcomes(obs_one, act_one)[0, 0])
        predicted_threshold = float(model.predict_threshold(obs_one, act_one)[0])
        predicted_threshold = float(
            min(max(predicted_threshold, FORCE_MIN), FORCE_MAX)
        )
        pushes = prob_push >= 0.5

        mechanism = []
        if situation.physics.constraint == "constrained":
            mechanism.append("latched")
        if situation.physics.resistance >= 2:
            mechanism.append("heavy")
        if situation.physics.traction[0] == "slip":
            mechanism.append("self_slips")
        if situation.physics.traction[1] == "slip":
            mechanism.append("teammate_slips")

        def regret_for(pushes: bool, force: float) -> Tuple[float, float, float]:
            """Regret, success and the force spent for one policy's choice."""

            if not pushes:
                if pushable:
                    return FAIL_PENALTY, 0.0, 0.0
                return 0.0, 0.0, 0.0
            outcome = push_outcome(scene, {0: force}, PROBE_ANGLE)
            if not outcome.moved:
                return FAIL_PENALTY + force, 0.0, force
            return force - threshold, 1.0, force

        choices = {
            "oracle": (pushable, threshold if np.isfinite(threshold) else 0.0),
            "learned": (pushes, predicted_threshold),
            "always_max": (True, FORCE_MAX),
            "always_min": (True, 0.25),
        }
        regret: Dict[str, float] = {}
        force_used: Dict[str, float] = {}
        for name, (does_push, force) in choices.items():
            value, _success, spent = regret_for(bool(does_push), force)
            regret[name] = value
            force_used[name] = spent

        records.append(
            ThresholdRecord(
                scene_id=situation.scene_id,
                true_physics=situation.physics.label,
                can_push=pushable,
                true_threshold=threshold if np.isfinite(threshold) else float("inf"),
                predicted_can_push=prob_push,
                predicted_threshold=predicted_threshold,
                mechanism="+".join(mechanism) if mechanism else "plain",
                regret=regret,
                force_used=force_used,
            )
        )

    pushable_records = [r for r in records if r.can_push]
    threshold_mae = (
        float(
            np.mean(
                [abs(r.predicted_threshold - r.true_threshold) for r in pushable_records]
            )
        )
        if pushable_records
        else float("nan")
    )
    correct = [
        (r.predicted_can_push >= 0.5) == bool(r.can_push) for r in records
    ]
    recalls = []
    if any(r.can_push for r in records):
        recalls.append(
            float(
                np.mean(
                    [
                        (r.predicted_can_push >= 0.5)
                        for r in records
                        if r.can_push
                    ]
                )
            )
        )
    if any(not r.can_push for r in records):
        recalls.append(
            float(
                np.mean(
                    [
                        (r.predicted_can_push < 0.5)
                        for r in records
                        if not r.can_push
                    ]
                )
            )
        )

    def summarise(name: str, subset: Sequence[ThresholdRecord]) -> List[PolicySummary]:
        out = []
        for policy in BASELINES:
            if not subset:
                out.append(PolicySummary(policy, 0, float("nan"), float("nan")))
                continue
            out.append(
                PolicySummary(
                    name=policy,
                    n=len(subset),
                    mean_regret=float(np.mean([r.regret[policy] for r in subset])),
                    success_rate=float(
                        np.mean([r.regret[policy] <= r.true_threshold + 1e-9 for r in subset])
                    ),
                )
            )
        return out

    groups: Dict[str, List[ThresholdRecord]] = defaultdict(list)
    for record in records:
        groups[record.mechanism].append(record)

    return ThresholdProbeResult(
        summaries=summarise("all", records),
        by_mechanism={k: summarise(k, v) for k, v in sorted(groups.items())},
        threshold_mae=threshold_mae,
        can_push_balanced_accuracy=float(np.mean(recalls)) if recalls else float("nan"),
        n_pushable=len(pushable_records),
        records=records,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_train_situations": len(train_situations),
            "n_test_situations": len(test_situations),
            "epochs": epochs,
            "positive_rate": positive_rate,
            "push_base_cost": PUSH_BASE_COST,
            "fail_penalty": FAIL_PENALTY,
            "wall_clock_seconds": time.perf_counter() - started,
        },
    )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_threshold_markdown(result: ThresholdProbeResult) -> str:
    lines = [
        f"Pushable decision points: {result.n_pushable} of {len(result.records)}. "
        f"Can-push balanced accuracy {result.can_push_balanced_accuracy:.3f}; "
        f"threshold MAE on pushable points {result.threshold_mae:.3f} force units.",
        "",
        "| group | policy | n | mean regret | success rate |",
        "|---|---|---|---|---|",
    ]
    for summary in result.summaries:
        lines.append(
            f"| all | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
            f"{summary.success_rate:.3f} |"
        )
    for group, summaries in result.by_mechanism.items():
        for summary in summaries:
            lines.append(
                f"| {group} | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
                f"{summary.success_rate:.3f} |"
            )
    return "\n".join(lines)


def write_threshold_report(result: ThresholdProbeResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "threshold_probe.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "summaries": [asdict(s) for s in result.summaries],
                "by_mechanism": {k: [asdict(s) for s in v] for k, v in result.by_mechanism.items()},
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "threshold_probe.md"
    md_path.write_text(
        "# Threshold probe: inferring the force a push needs\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_train_situations']} training / "
        f"{result.meta['n_test_situations']} held-out situations, "
        f"{result.meta['epochs']} epochs\n"
        f"- push costs {result.meta['push_base_cost']} + force; failure penalty "
        f"{result.meta['fail_penalty']}\n"
        f"- wall clock: {result.meta['wall_clock_seconds']:.1f}s\n\n"
        + format_threshold_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
