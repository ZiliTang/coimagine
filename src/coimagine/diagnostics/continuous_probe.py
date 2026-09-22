"""Continuous-parameter probe: can a learned model recover the feasible region?

The discrete probe could only ask "does this action work" per sample. With continuous
parameters the question becomes geometric: **for which (direction, force) does a push
work?** The answer is a region whose force threshold is set by the hidden contact
physics, so a model that has not inferred the physics puts the boundary in the wrong
place even when its per-sample accuracy looks acceptable.

Metrics, all on held-out scenes:

* per-sample balanced accuracy (the weak metric, reported for continuity);
* **region IoU** -- per scene, the overlap between the predicted feasible set over the
  (angle, force) grid and the true one;
* **threshold error** -- predicted minus true minimum feasible force inside the angular
  window, which is the quantity the physics actually controls;
* **decision success** -- choose the (angle, force) the model likes best and check in the
  simulator whether it works.
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
from ..grid.continuous import FORCE_MAX, push_outcome
from ..grid.planner import PHYSICS
from ..learning.continuous_data import (
    DEFAULT_ANGLES,
    DEFAULT_FORCES,
    ContinuousSample,
    generate_continuous_dataset,
    to_arrays,
)
from ..learning.mlp import MultiHeadMLP, train

#: A push is treated as feasible at this probability. 0.5 is the natural threshold for a
#: calibrated binary head; the probe reports the threshold-free metrics alongside.
DECISION_THRESHOLD = 0.5


@dataclass
class ProbeMetrics:
    name: str
    n_samples: int
    n_scenes: int
    accuracy: float
    balanced_accuracy: float
    bce: float
    region_iou: float
    threshold_error: float
    decision_success: float
    physics_accuracy: float
    pose_accuracy: float


@dataclass
class ContinuousProbeResult:
    overall: ProbeMetrics
    by_mechanism: Dict[str, ProbeMetrics]
    meta: Dict[str, object] = field(default_factory=dict)


def _group(samples: Sequence[ContinuousSample]) -> Dict[int, List[ContinuousSample]]:
    out: Dict[int, List[ContinuousSample]] = defaultdict(list)
    for sample in samples:
        out[sample.scene_id].append(sample)
    return out


def _region_iou(
    samples: Sequence[ContinuousSample], prob: np.ndarray, angles: Sequence[float], forces: Sequence[float]
) -> float:
    """Per-scene IoU averaged over scenes, using the analytical ground truth."""

    by_scene = _group(samples)
    scores: List[float] = []
    for scene_samples in by_scene.values():
        cells = len(angles) * len(forces)
        if len(scene_samples) != cells:
            continue
        indices = {(round(s.angle, 6), round(s.force, 6)): i for i, s in enumerate(scene_samples)}
        pred = np.zeros(cells, dtype=bool)
        true = np.zeros(cells, dtype=bool)
        for (angle, force), i in indices.items():
            pred[i] = prob[i] >= DECISION_THRESHOLD
            true[i] = bool(scene_samples[i].moved)
        union = np.logical_or(pred, true).sum()
        if union == 0:
            scores.append(1.0)
            continue
        scores.append(float(np.logical_and(pred, true).sum() / union))
    return float(np.mean(scores)) if scores else float("nan")


def _threshold_error(
    samples: Sequence[ContinuousSample], prob: np.ndarray
) -> float:
    """Predicted minus true minimum feasible force, averaged over scenes.

    Restricted to pushes inside the angular window, where a feasible force exists for at
    least some scene; otherwise the quantity is undefined.
    """

    by_scene = _group(samples)
    errors: List[float] = []
    for scene_samples in by_scene.values():
        true_forces = [s.force for s in scene_samples if s.moved]
        if not true_forces:
            continue
        predicted = [
            s.force
            for i, s in enumerate(scene_samples)
            if prob[i] >= DECISION_THRESHOLD
        ]
        if not predicted:
            errors.append(FORCE_MAX)
            continue
        errors.append(min(predicted) - min(true_forces))
    return float(np.mean(errors)) if errors else float("nan")


def _decision_success(
    samples: Sequence[ContinuousSample], prob: np.ndarray
) -> float:
    """Choose the model's preferred (angle, force) per scene and test it in the simulator.

    Ties are broken towards the **lowest force**, which is the sensible preference: a
    push should use no more force than it needs.
    """

    by_scene = _group(samples)
    successes: List[float] = []
    for scene_samples in by_scene.values():
        best_index = None
        best_key = None
        for i, sample in enumerate(scene_samples):
            key = (-prob[i], sample.force)
            if best_key is None or key < best_key:
                best_key = key
                best_index = i
        if best_index is None:
            continue
        chosen = scene_samples[best_index]
        successes.append(float(chosen.moved))
    return float(np.mean(successes)) if successes else float("nan")


def _evaluate(name: str, samples: Sequence[ContinuousSample], model: MultiHeadMLP,
              angles: Sequence[float], forces: Sequence[float]) -> ProbeMetrics:
    obs, action, y, physics_index, pose_index = to_arrays(samples)
    prob = model.predict_outcomes(obs, action)[:, 0]
    pred = (prob >= DECISION_THRESHOLD).astype(float)
    accuracy = float((pred == y[:, 0]).mean())
    recalls = []
    if (y[:, 0] == 1).any():
        recalls.append(float((pred[y[:, 0] == 1] == 1).mean()))
    if (y[:, 0] == 0).any():
        recalls.append(float((pred[y[:, 0] == 0] == 0).mean()))
    balanced = float(np.mean(recalls)) if recalls else float("nan")
    eps = 1e-12
    bce = float(
        -(y[:, 0] * np.log(prob + eps) + (1 - y[:, 0]) * np.log(1 - prob + eps)).mean()
    )
    phys = model.predict_physics(obs).argmax(axis=1)
    pose_logits = model.forward(obs, np.zeros_like(action))["pose_logit"]
    cache = {"pose_logit": pose_logits}
    return ProbeMetrics(
        name=name,
        n_samples=len(samples),
        n_scenes=len(_group(samples)),
        accuracy=accuracy,
        balanced_accuracy=balanced,
        bce=bce,
        region_iou=_region_iou(samples, prob, angles, forces),
        threshold_error=_threshold_error(samples, prob),
        decision_success=_decision_success(samples, prob),
        physics_accuracy=float((phys == physics_index).mean()),
        pose_accuracy=float((cache["pose_logit"].argmax(axis=1) == pose_index).mean()),
    )


def run_continuous_probe(
    *,
    n_train_scenes: int = 200,
    n_test_scenes: int = 60,
    epochs: int = 400,
    seed: int = 0,
    lead_in: int = 2,
    exclude_slipping_from_training: bool = False,
    angles: Sequence[float] = DEFAULT_ANGLES,
    forces: Sequence[float] = DEFAULT_FORCES,
) -> ContinuousProbeResult:
    started = time.perf_counter()
    train_samples = generate_continuous_dataset(
        n_scenes=n_train_scenes, lead_in=lead_in, seed=seed, angles=angles, forces=forces
    )
    test_samples = generate_continuous_dataset(
        n_scenes=n_test_scenes, lead_in=lead_in, seed=seed + 90_000, angles=angles, forces=forces
    )
    if exclude_slipping_from_training:
        train_samples = [s for s in train_samples if s.physics.traction[1] != "slip"]

    obs, action, y, physics_index, pose_index = to_arrays(train_samples)
    # The labels are heavily imbalanced (a few percent feasible), so an unweighted BCE is
    # minimised by predicting "never feasible". The weight is set from the training rate
    # and reported; it is never tuned on the test set.
    positive_rate = float(y.mean()) if len(y) else 0.0
    positive_weight = (1.0 - positive_rate) / positive_rate if positive_rate > 0 else 1.0
    model = MultiHeadMLP(
        seed=seed, action_dim=action.shape[1], positive_weight=positive_weight
    )
    # Only the feasibility channel is meaningful here; progress and enablement are
    # discrete-action notions and are not defined for a continuous push.
    targets = np.concatenate([y, np.zeros_like(y), np.zeros_like(y)], axis=1)
    train(
        model, obs, action, targets, physics_index, pose_index,
        epochs=epochs, seed=seed,
    )

    overall = _evaluate("all", test_samples, model, angles, forces)
    groups: Dict[str, List[ContinuousSample]] = defaultdict(list)
    for sample in test_samples:
        mechanism = []
        if sample.physics.constraint == "constrained":
            mechanism.append("latched")
        if sample.physics.resistance >= 2:
            mechanism.append("heavy")
        if sample.physics.traction[1] == "slip":
            mechanism.append("teammate_slips")
        groups["+".join(mechanism) if mechanism else "plain"].append(sample)
    by_mechanism = {
        name: _evaluate(name, subset, model, angles, forces)
        for name, subset in sorted(groups.items())
    }

    return ContinuousProbeResult(
        overall=overall,
        by_mechanism=by_mechanism,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_train_scenes": n_train_scenes,
            "n_test_scenes": n_test_scenes,
            "epochs": epochs,
            "lead_in": lead_in,
            "exclude_slipping_from_training": exclude_slipping_from_training,
            "positive_rate": positive_rate,
            "positive_weight": positive_weight,
            "angles": list(angles),
            "forces": list(forces),
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


def format_continuous_markdown(result: ContinuousProbeResult) -> str:
    lines = [
        "| group | scenes | region IoU | force-threshold error | decision success | balanced accuracy |",
        "|---|---|---|---|---|---|",
    ]
    for metrics in [result.overall] + list(result.by_mechanism.values()):
        lines.append(
            f"| {metrics.name} | {metrics.n_scenes} | {metrics.region_iou:.3f} | "
            f"{metrics.threshold_error:+.3f} | {metrics.decision_success:.3f} | "
            f"{metrics.balanced_accuracy:.3f} |"
        )
    lines += [
        "",
        "| group | physics decode | pose decode | per-sample accuracy |",
        "|---|---|---|---|",
    ]
    for metrics in [result.overall] + list(result.by_mechanism.values()):
        lines.append(
            f"| {metrics.name} | {metrics.physics_accuracy:.3f} | "
            f"{metrics.pose_accuracy:.3f} | {metrics.accuracy:.3f} |"
        )
    return "\n".join(lines)


def write_continuous_report(result: ContinuousProbeResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "continuous_probe.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "overall": asdict(result.overall),
                "by_mechanism": {k: asdict(v) for k, v in result.by_mechanism.items()},
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "continuous_probe.md"
    md_path.write_text(
        "# Continuous-parameter probe: is the feasible region recovered?\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_train_scenes']} training scenes, "
        f"{result.meta['n_test_scenes']} held out, {result.meta['epochs']} epochs\n"
        f"- {len(result.meta['angles'])} angles x {len(result.meta['forces'])} forces "
        f"per scene\n"
        f"- wall clock: {result.meta['wall_clock_seconds']:.1f}s\n\n"
        + format_continuous_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
