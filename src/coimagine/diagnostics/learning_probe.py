"""The learning probe: can a model discover the contact modes from data?

Two questions, both measured on scenes the model never saw:

1. **In distribution** -- can it predict which joint action will work, and do its
   privileged decoder heads recover the object placement and the contact physics?
2. **Mode ablation** -- if training data never contains a contact mode (here, the
   teammate slipping), does the model fail on that mode at test time compared with a
   model that has seen it, even though the test samples are unseen by both?

The second question is the one that connects the learning programme to the round-7
finding that a model class which cannot express the teammate's traction fails
outright. If the ablation shows the same collapse, then coverage of the contact modes
-- not raw prediction accuracy -- is the thing that determines whether rollout works.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid.state import HiddenPhysics
from ..learning.data import Dataset, generate_dataset, pose_to_index, sample_to_arrays
from ..learning.mlp import MultiHeadMLP, train

#: The contact mode held out for the ablation: the teammate (robot B) slipping.
def teammate_slips(sample) -> bool:
    return sample.physics.traction[1] == "slip"


@dataclass
class SplitMetrics:
    name: str
    n: int
    feasibility_accuracy: float
    feasibility_balanced_accuracy: float
    feasibility_bce: float
    physics_accuracy: float
    pose_accuracy: float
    positive_rate: float


@dataclass
class ProbeResult:
    in_distribution: SplitMetrics
    narrow_on_rare_mode: SplitMetrics
    wide_on_rare_mode: SplitMetrics
    majority_baseline: float
    n_train: int
    n_test: int
    wall_clock_seconds: float
    meta: Dict[str, object] = field(default_factory=dict)


def _evaluate(model: MultiHeadMLP, samples: Sequence) -> SplitMetrics:
    x, y, physics_index = sample_to_arrays(samples)
    action = x[:, model.obs_dim:]
    obs = x[:, : model.obs_dim]
    prob = model.predict_feasibility(obs, action)
    pred = (prob >= 0.5).astype(float)

    accuracy = float((pred == y).mean())
    positives = y == 1
    negatives = y == 0
    recalls = []
    if positives.any():
        recalls.append(float((pred[positives] == 1).mean()))
    if negatives.any():
        recalls.append(float((pred[negatives] == 0).mean()))
    balanced = float(np.mean(recalls)) if recalls else float("nan")

    eps = 1e-12
    bce = float(-(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)).mean())

    phys_pred = model.predict_physics(obs).argmax(axis=1)
    physics_accuracy = float((phys_pred == physics_index).mean())

    pose_index = np.asarray([pose_to_index(s.object_origin) for s in samples], dtype=np.int64)
    cache = model.forward(obs, np.zeros_like(action))
    pose_accuracy = float((cache["pose_logit"].argmax(axis=1) == pose_index).mean())

    return SplitMetrics(
        name="",
        n=len(samples),
        feasibility_accuracy=accuracy,
        feasibility_balanced_accuracy=balanced,
        feasibility_bce=bce,
        physics_accuracy=physics_accuracy,
        pose_accuracy=pose_accuracy,
        positive_rate=float(y.mean()),
    )


def run_learning_probe(
    *,
    n_scenes: int = 400,
    lead_in: int = 2,
    radius: int = 5,
    epochs: int = 400,
    seed: int = 0,
    test_fraction: float = 0.3,
    verbose: bool = False,
) -> ProbeResult:
    started = time.perf_counter()
    dataset = generate_dataset(n_scenes=n_scenes, lead_in=lead_in, radius=radius, seed=seed)
    scene_ids = sorted({s.scene_id for s in dataset.samples})
    cutoff = int(len(scene_ids) * (1.0 - test_fraction))
    train_scenes = set(scene_ids[:cutoff])
    test_scenes = set(scene_ids[cutoff:])

    train_set = dataset.filter(lambda s: s.scene_id in train_scenes)
    test_set = dataset.filter(lambda s: s.scene_id in test_scenes)
    rare_train = train_set.filter(lambda s: not teammate_slips(s))
    rare_test = test_set.filter(teammate_slips)

    def fit(subset: Dataset) -> MultiHeadMLP:
        x, y, physics_index = sample_to_arrays(subset.samples)
        pose_index = np.asarray(
            [pose_to_index(s.object_origin) for s in subset.samples], dtype=np.int64
        )
        model = MultiHeadMLP(seed=seed)
        train(
            model, x[:, : model.obs_dim], x[:, model.obs_dim:], y,
            physics_index, pose_index,
            epochs=epochs, seed=seed, verbose=verbose,
        )
        return model

    wide = fit(train_set)
    narrow = fit(rare_train)

    in_dist = _evaluate(wide, test_set.samples)
    in_dist.name = "in_distribution"
    narrow_metrics = _evaluate(narrow, rare_test.samples)
    narrow_metrics.name = "narrow_model_trained_without_slipping"
    wide_metrics = _evaluate(wide, rare_test.samples)
    wide_metrics.name = "wide_model_trained_with_slipping"

    x, y, _ = sample_to_arrays(test_set.samples)
    majority = float(max(y.mean(), 1.0 - y.mean()))

    return ProbeResult(
        in_distribution=in_dist,
        narrow_on_rare_mode=narrow_metrics,
        wide_on_rare_mode=wide_metrics,
        majority_baseline=majority,
        n_train=len(train_set),
        n_test=len(test_set.samples),
        wall_clock_seconds=time.perf_counter() - started,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_scenes": n_scenes,
            "lead_in": lead_in,
            "radius": radius,
            "epochs": epochs,
            "seed": seed,
            "rare_train_samples": len(rare_train),
            "rare_test_samples": len(rare_test),
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


def format_probe_markdown(result: ProbeResult) -> str:
    lines: List[str] = []
    lines.append(
        f"Train {result.n_train} samples, test {result.n_test} samples, "
        f"majority-class baseline {result.majority_baseline:.3f}."
    )
    lines.append("")
    lines.append("| evaluation set | n | feasibility accuracy | balanced accuracy | BCE |")
    lines.append("|---|---|---|---|---|")
    for metrics in (result.in_distribution, result.wide_on_rare_mode, result.narrow_on_rare_mode):
        lines.append(
            f"| {metrics.name} | {metrics.n} | {metrics.feasibility_accuracy:.3f} | "
            f"{metrics.feasibility_balanced_accuracy:.3f} | {metrics.feasibility_bce:.3f} |"
        )
    lines.append("")
    lines.append("| evaluation set | physics decode | pose decode | positive rate |")
    lines.append("|---|---|---|---|")
    for metrics in (result.in_distribution, result.wide_on_rare_mode, result.narrow_on_rare_mode):
        lines.append(
            f"| {metrics.name} | {metrics.physics_accuracy:.3f} | "
            f"{metrics.pose_accuracy:.3f} | {metrics.positive_rate:.3f} |"
        )
    lines.append("")
    delta = result.wide_on_rare_mode.feasibility_balanced_accuracy - (
        result.narrow_on_rare_mode.feasibility_balanced_accuracy
    )
    lines.append(
        f"On the held-out contact mode, training with that mode improves balanced "
        f"accuracy by **{delta:+.3f}** over a model that never saw it."
    )
    return "\n".join(lines)


def write_probe_report(result: ProbeResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    json_path = out / "learning_probe.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "in_distribution": asdict(result.in_distribution),
                "narrow_on_rare_mode": asdict(result.narrow_on_rare_mode),
                "wide_on_rare_mode": asdict(result.wide_on_rare_mode),
                "majority_baseline": result.majority_baseline,
                "n_train": result.n_train,
                "n_test": result.n_test,
                "wall_clock_seconds": result.wall_clock_seconds,
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)

    md_path = out / "learning_probe.md"
    md_path.write_text(
        "# Learning probe: can a model discover the contact modes?\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_scenes']} scenes, lead-in {result.meta['lead_in']}, "
        f"scan radius {result.meta['radius']}, {result.meta['epochs']} epochs\n"
        f"- rare-mode training samples: {result.meta['rare_train_samples']}, "
        f"rare-mode test samples: {result.meta['rare_test_samples']}\n"
        f"- wall clock: {result.wall_clock_seconds:.1f}s\n\n"
        + format_probe_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
