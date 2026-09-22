"""Multi-step feasibility: does training on imagined rollout steps help at depth?

The alignment document proposed augmenting the model with a "feasibility prediction loss at
several imagined steps". This module tests the claim directly, and in the form in which it
can actually fail.

Two models share an architecture and a training budget:

* **shallow** -- trained only on depth-0 states, i.e. the immediate decision;
* **multi-step** -- trained on states taken from depths ``0 .. H-1`` of real rollouts,
  predicting for each candidate action whether it will work *from there*.

Both are then evaluated at every depth, on held-out scenes. A planner consumes feasibility
estimates at depth, so the question is whether the shallow model's estimates degrade as the
rollout advances and whether multi-step training arrests that.

Training states come from real rollouts: the simulator is advanced with the action the
analytic one-step value prefers under the true physics, so the states visited are the ones a
competent planner actually reaches, not a random walk.
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
from ..grid.dynamics import execute
from ..grid.planner import (
    PHYSICS,
    ContactContext,
    DEFAULT_FAIL_COST,
    _execute_cached,
)
from .decision_probe import _value_of_action
from ..learning.data import (
    candidates_for,
    generate_situations,
    pose_to_index,
    sample_vector,
    _joint_for,
)
from ..learning.features import ACTION_IDS, OBSERVATION_DIM
from ..learning.mlp import MultiHeadMLP, train

#: Rollout depths to evaluate at. Depth 0 is the immediate decision.
DEPTHS: Tuple[int, ...] = (0, 1, 2)


@dataclass
class DepthMetrics:
    model: str
    depth: int
    n_samples: int
    balanced_accuracy: float
    regret: float


@dataclass
class MultiStepResult:
    metrics: List[DepthMetrics]
    meta: Dict[str, object] = field(default_factory=dict)


def _rollout_states(
    *, n_scenes: int, horizon: int, seed: int, lead_in: int = 2
) -> List[Tuple[int, Tuple[float, ...], Tuple[float, ...], int, int, int]]:
    """(depth, observation, action, moved, scene_id, is_goal) tuples from real rollouts.

    One sample per candidate action per visited state, so the model sees what each option
    would have done from exactly the states a planner reaches.
    """

    situations = generate_situations(n_scenes=n_scenes, lead_in=lead_in, seed=seed)
    candidates = candidates_for()
    out: List[Tuple[int, Tuple[float, ...], Tuple[float, ...], int, int, int]] = []

    for situation in situations:
        scene = situation.scene
        history_a = list(situation.history_a)
        public_history = list(situation.public_history)
        for depth in range(horizon):
            if scene.is_goal():
                break
            obs = sample_vector(
                situation.scan, scene.robot_pose(0), tuple(history_a),
                tuple(public_history), ACTION_IDS[0],
            )
            observation = obs[:OBSERVATION_DIM]
            for action_id in ACTION_IDS:
                successor, result = _execute_cached(scene, _joint_for(scene, action_id))
                action_vector = obs[OBSERVATION_DIM:]
                # the action part of the feature vector is rebuilt per candidate
                candidate_vector = sample_vector(
                    situation.scan, scene.robot_pose(0), tuple(history_a),
                    tuple(public_history), action_id,
                )[OBSERVATION_DIM:]
                out.append(
                    (
                        depth,
                        observation,
                        candidate_vector,
                        int(result.y_public == "moved"),
                        situation.scene_id,
                        int(scene.is_goal()),
                    )
                )

            # advance with the action the analytic one-step value prefers
            ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
            hyp = (PHYSICS.index(scene.physics), False)
            values = {
                action: _value_of_action(
                    ctx, hyp, action, candidates, depth=1, fail_cost=DEFAULT_FAIL_COST
                )
                for action in ACTION_IDS
            }
            best = min(ACTION_IDS, key=lambda a: (values[a], a))
            successor, result = execute(scene, _joint_for(scene, best))
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            scene = successor
    return out


def _arrays(rows):
    observation = np.asarray([r[1] for r in rows], dtype=np.float64)
    action = np.asarray([r[2] for r in rows], dtype=np.float64)
    moved = np.asarray([[r[3]] for r in rows], dtype=np.float64)
    return observation, action, moved


def _evaluate(model: MultiHeadMLP, rows, model_name: str, depth: int) -> DepthMetrics:
    if not rows:
        return DepthMetrics(model_name, depth, 0, float("nan"), float("nan"))
    observation, action, moved = _arrays(rows)
    prob = model.predict_outcomes(observation, action)[:, 0]
    predicted = prob >= 0.5
    label = moved[:, 0] >= 0.5
    recalls = []
    if label.any():
        recalls.append(float((predicted[label] == True).mean()))  # noqa: E712
    if (~label).any():
        recalls.append(float((predicted[~label] == False).mean()))  # noqa: E712

    # decision regret: per state, pick the action with the highest predicted probability
    # and compare with the best action by the true one-step value
    grouped: Dict[int, List[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        grouped[row[4]].append(index)
    regrets: List[float] = []
    for scene_id, indices in grouped.items():
        best_index = max(indices, key=lambda i: (prob[i], -ACTION_IDS.index("wait")))
        # true one-step value is the negated moved label here: moving is better
        true_best = max(indices, key=lambda i: (label[i], -i))
        regrets.append(float(label[true_best]) - float(label[best_index]))
    return DepthMetrics(
        model=model_name,
        depth=depth,
        n_samples=len(rows),
        balanced_accuracy=float(np.mean(recalls)) if recalls else float("nan"),
        regret=float(np.mean(regrets)) if regrets else float("nan"),
    )


def run_multistep_probe(
    *,
    n_train_scenes: int = 200,
    n_test_scenes: int = 80,
    horizon: int = 3,
    epochs: int = 400,
    seed: int = 0,
    budget_match: bool = True,
) -> MultiStepResult:
    started = time.perf_counter()
    train_rows = _rollout_states(
        n_scenes=n_train_scenes, horizon=horizon, seed=seed
    )
    test_rows = _rollout_states(
        n_scenes=n_test_scenes, horizon=horizon, seed=seed + 60_000
    )

    shallow_rows = [r for r in train_rows if r[0] == 0]
    if budget_match:
        # The multi-step model sees more samples simply because rollouts visit more states.
        # Give the shallow model the same number of depth-0 samples, drawn from extra
        # scenes, so the comparison is about the *loss* and not about data quantity.
        target = len(train_rows)
        attempt = 0
        while len(shallow_rows) < target and attempt < 8:
            attempt += 1
            extra = _rollout_states(
                n_scenes=max(1, (target - len(shallow_rows)) // 6 + 1),
                horizon=1,
                seed=seed + 1_000 * attempt,
            )
            shallow_rows.extend(r for r in extra if r[0] == 0)
        shallow_rows = shallow_rows[:target]
    observation, action, moved = _arrays(shallow_rows)
    targets = np.concatenate([moved, np.zeros((len(moved), 2))], axis=1)
    shallow = MultiHeadMLP(seed=seed, obs_dim=observation.shape[1], action_dim=action.shape[1])
    train(shallow, observation, action, targets, epochs=epochs, seed=seed)

    observation, action, moved = _arrays(train_rows)
    targets = np.concatenate([moved, np.zeros((len(moved), 2))], axis=1)
    multistep = MultiHeadMLP(seed=seed, obs_dim=observation.shape[1], action_dim=action.shape[1])
    train(multistep, observation, action, targets, epochs=epochs, seed=seed)

    metrics: List[DepthMetrics] = []
    for depth in DEPTHS:
        for name, model in (("shallow", shallow), ("multi_step", multistep)):
            metrics.append(
                _evaluate(
                    model,
                    [r for r in test_rows if r[0] == depth],
                    name,
                    depth,
                )
            )

    return MultiStepResult(
        metrics=metrics,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_train_scenes": n_train_scenes,
            "n_test_scenes": n_test_scenes,
            "horizon": horizon,
            "epochs": epochs,
            "shallow_train_samples": len(shallow_rows),
            "budget_match": budget_match,
            "multistep_train_samples": len(train_rows),
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


def format_multistep_markdown(result: MultiStepResult) -> str:
    lines = [
        f"Shallow model trained on {result.meta['shallow_train_samples']} depth-0 samples; "
        f"multi-step model on {result.meta['multistep_train_samples']} samples spread over "
        f"depths 0-{result.meta['horizon'] - 1}.",
        "",
        "| depth | model | n | balanced accuracy | decision regret |",
        "|---|---|---|---|---|",
    ]
    for metrics in result.metrics:
        lines.append(
            f"| {metrics.depth} | {metrics.model} | {metrics.n_samples} | "
            f"{metrics.balanced_accuracy:.3f} | {metrics.regret:.3f} |"
        )
    return "\n".join(lines)


def write_multistep_report(result: MultiStepResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "multistep.json"
    json_path.write_text(
        json.dumps(
            {"meta": result.meta, "metrics": [asdict(m) for m in result.metrics]},
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "multistep.md"
    md_path.write_text(
        "# Multi-step feasibility: does training at depth help at depth?\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_train_scenes']} training / {result.meta['n_test_scenes']} "
        f"held-out scenes, horizon {result.meta['horizon']}\n"
        f"- wall clock: {result.meta['wall_clock_seconds']:.1f}s\n\n"
        + format_multistep_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
