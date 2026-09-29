"""How much does the distribution shift from training to closed-loop execution cost?

The world model reaches 1.000 object-position accuracy on held-out trajectories drawn from the
**data-collection** policy. The planner does not follow that policy, so its own states are a
different distribution. Two questions, both quantitative:

1. **How much accuracy does the shift cost?** One-step object-cell accuracy measured with the *same
   protocol* on both distributions: on trajectories from the collection policy, and on the states the
   planner actually stands on during execution.
2. **How far off is it?** Interpretable state features (object-to-goal distance, robot-to-object
   distance, contact, released, visible object cells) compared between the two, plus the distance from
   each state to the nearest state seen in training, so "off distribution" is a number rather than an
   assertion. The collection-policy sample is the baseline for unseen but in-distribution data.

Memory note: the nearest-neighbour pass is chunked. Building the full
``(n_queries, n_reference, observation_dim)`` array at once needs several gigabytes and killed an
earlier run of this probe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..grid.dynamics import FEEDBACK_SYMBOLS, execute
from ..grid.sensing import ray_cast
from ..grid.state import enumerate_hidden_physics
from ..learning.belief_model import BeliefWorldModel
from ..learning.data import _joint_for, generate_situations, pose_to_index
from ..learning.features import ACTION_IDS, CELL_KINDS, GRID, N_CELL_BLOCKS, observation_vector
from ..learning.world_model import N_CELLS, N_PHYSICS
from ..learning.world_model_data import ACTION_MIXTURE, collect_trajectories
from .belief_probe import fuse_hidden_state, plan_with_search, theta_index

PHYSICS_TABLE = tuple(enumerate_hidden_physics())
_OBJECT_BLOCK = CELL_KINDS.index("object") + 1
#: Cap on the reference set for the nearest-neighbour pass, and the chunk size, both for memory.
NN_REFERENCE_CAP = 3000
NN_CHUNK = 64

FEATURE_NAMES = (
    "object-to-goal distance",
    "robot-to-object distance",
    "robot in contact",
    "object released",
    "object cells visible",
)


def _one_hot(index: int) -> np.ndarray:
    out = np.zeros((1, N_PHYSICS))
    out[0, index] = 1.0
    return out


def _visible_object_cells(observation: np.ndarray) -> int:
    plane = observation[: GRID * GRID * N_CELL_BLOCKS].reshape(GRID, GRID, N_CELL_BLOCKS)
    return int(plane[:, :, _OBJECT_BLOCK].sum())


def _features(scene, observation: np.ndarray, goal: frozenset) -> Tuple[float, ...]:
    origin = scene.obj.origin
    members = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
    distance = min(max(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in members)
    robot = scene.robot_pose(0)
    contact = float(robot in {(origin[0] - 1, origin[1]), (origin[0] - 1, origin[1] + 1)})
    return (
        float(distance),
        float(abs(robot[0] - origin[0]) + abs(robot[1] - origin[1])),
        contact,
        float(scene.released),
        float(_visible_object_cells(observation)),
    )


@dataclass
class DistributionSample:
    name: str
    observations: List[np.ndarray] = field(default_factory=list)
    features: List[Tuple[float, ...]] = field(default_factory=list)
    correct_true_params: int = 0
    correct_end_to_end: int = 0
    steps: int = 0

    @property
    def accuracy_true_params(self) -> float:
        return self.correct_true_params / max(1, self.steps)

    @property
    def accuracy_end_to_end(self) -> float:
        return self.correct_end_to_end / max(1, self.steps)

    def feature_means(self) -> List[float]:
        if not self.features:
            return [float("nan")] * len(FEATURE_NAMES)
        return [float(v) for v in np.asarray(self.features).mean(axis=0)]


def _score(model, dynamic, action: int, truth: int, physics: int, belief) -> Tuple[int, int]:
    """One-step next-cell prediction, under the true parameters and under the current belief."""

    predicted_true = int(
        model.step(dynamic, action, _one_hot(physics))[0, 1 : 1 + N_CELLS].argmax()
    )
    predicted_e2e = int(model.step(dynamic, action, belief)[0, 1 : 1 + N_CELLS].argmax())
    return int(predicted_true == truth), int(predicted_e2e == truth)


def sample_collection_distribution(
    model: BeliefWorldModel,
    *,
    n_scenes: int = 300,
    horizon: int = 8,
    seed: int = 90_000,
    name: str = "collection policy (held out)",
) -> DistributionSample:
    """States produced by the data-collection policy, scored with the same protocol."""

    sample = DistributionSample(name=name)
    situations = generate_situations(
        n_scenes=n_scenes, lead_in=2, seed=seed, physics_pool=PHYSICS_TABLE
    )
    rng = np.random.default_rng(seed)
    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            continue
        goal = frozenset(scene.static.target)
        history_a = list(situation.history_a)
        public_history = list(situation.public_history)
        for _ in range(horizon):
            if scene.is_goal():
                break
            observation = np.asarray(
                observation_vector(
                    ray_cast(scene, 0, radius=5), scene.robot_pose(0),
                    tuple(history_a), tuple(public_history),
                )
            )
            action_id = ACTION_MIXTURE[int(rng.integers(len(ACTION_MIXTURE)))]
            action = ACTION_IDS.index(action_id)
            successor, result = execute(scene, _joint_for(scene, action_id))
            truth = pose_to_index(successor.obj.origin)
            belief = model.parameter_belief(observation)
            # Score the *current* state estimate from the observation directly, with no propagation:
            # propagating under the encoder belief would add its own drift and confound the
            # comparison between the two distributions.
            here = model.dynamic_state(observation)
            good_true, good_e2e = _score(
                model, here, action, truth, theta_index(scene.physics), belief
            )
            sample.correct_true_params += good_true
            sample.correct_end_to_end += good_e2e
            sample.steps += 1
            sample.observations.append(observation)
            sample.features.append(_features(scene, observation, goal))
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            scene = successor
    return sample


def sample_execution_distribution(
    model: BeliefWorldModel,
    *,
    n_scenes: int = 200,
    horizon: int = 8,
    seed: int = 55_000,
    depth: int = 2,
    beam: int = 1,
    name: str = "closed-loop execution",
) -> DistributionSample:
    """States the planner actually stands on, scored the same way."""

    sample = DistributionSample(name=name)
    situations = generate_situations(
        n_scenes=n_scenes, lead_in=0, seed=seed, physics_pool=PHYSICS_TABLE
    )
    probe_action = ACTION_IDS.index("probe:0:E")
    for situation in situations:
        scene = situation.scene
        if scene.is_goal():
            continue
        goal = frozenset(scene.static.target)
        successor, result = execute(scene, _joint_for(scene, "probe:0:E"))
        if successor.is_goal():
            continue
        observation = np.asarray(
            observation_vector(ray_cast(successor, 0, radius=5), successor.robot_pose(0), (), ())
        )
        belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
        belief = model.update_belief(
            belief, model.dynamic_state(observation), probe_action,
            FEEDBACK_SYMBOLS.index(result.feedback[0]),
        )
        carried = model.step(model.dynamic_state(observation), probe_action, belief)
        history_a = [result.feedback[0]]
        public_history = [result.y_public]
        current = successor

        for _ in range(horizon):
            if current.is_goal():
                break
            observation = np.asarray(
                observation_vector(
                    ray_cast(current, 0, radius=5), current.robot_pose(0),
                    tuple(history_a), tuple(public_history),
                )
            )
            dynamic = fuse_hidden_state(model.dynamic_state(observation), carried)
            chosen, _ = plan_with_search(
                model, observation, goal, belief=belief, dynamic=dynamic, depth=depth, beam=beam
            )
            next_scene, result = execute(current, _joint_for(current, ACTION_IDS[chosen]))
            truth = pose_to_index(next_scene.obj.origin)
            good_true, good_e2e = _score(
                model, dynamic, chosen, truth, theta_index(current.physics), belief
            )
            sample.correct_true_params += good_true
            sample.correct_end_to_end += good_e2e
            sample.steps += 1
            sample.observations.append(observation)
            sample.features.append(_features(current, observation, goal))

            belief = model.update_belief(
                belief, dynamic, chosen, FEEDBACK_SYMBOLS.index(result.feedback[0])
            )
            carried = model.step(dynamic, chosen, belief)
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            current = next_scene
    return sample


def nearest_training_distance(reference: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Distance from each query to the nearest reference state, chunked to bound memory."""

    # Floor the scale. A near-constant dimension (many are) would otherwise divide by ~0 and make
    # one dimension dominate the whole distance -- an earlier run reported a mean of 10846.
    scale = np.maximum(reference.std(axis=0), 0.05)
    scaled_reference = reference / scale
    out = np.empty(len(queries))
    for start in range(0, len(queries), NN_CHUNK):
        block = queries[start : start + NN_CHUNK] / scale
        distances = np.linalg.norm(
            block[:, None, :] - scaled_reference[None, :, :], axis=2
        )
        out[start : start + len(block)] = distances.min(axis=1)
    return out


@dataclass
class ShiftProbe:
    collection: DistributionSample
    execution: DistributionSample
    novelty: Dict[str, float] = field(default_factory=dict)


def run_shift_probe(
    model: BeliefWorldModel,
    *,
    n_collection_scenes: int = 300,
    n_execution_scenes: int = 200,
    horizon: int = 8,
    depth: int = 2,
    beam: int = 1,
    seed: int = 0,
    execution_seed: int = 55_000,
    train_reference_scenes: int = 400,
) -> ShiftProbe:
    collection = sample_collection_distribution(
        model, n_scenes=n_collection_scenes, horizon=horizon, seed=seed + 90_000
    )
    execution = sample_execution_distribution(
        model, n_scenes=n_execution_scenes, horizon=horizon,
        seed=execution_seed, depth=depth, beam=beam,
    )
    reference_trajectories = collect_trajectories(
        n_episodes=train_reference_scenes, horizon=horizon, seed=seed
    )
    reference = np.stack(
        [s.observation for t in reference_trajectories for s in t.transitions]
    )
    if len(reference) > NN_REFERENCE_CAP:
        picks = np.linspace(0, len(reference) - 1, NN_REFERENCE_CAP).astype(int)
        reference = reference[picks]

    probe = ShiftProbe(collection=collection, execution=execution)
    for name, sample in (
        ("collection policy (held out)", collection),
        ("closed-loop execution", execution),
    ):
        if not sample.observations:
            continue
        distances = nearest_training_distance(reference, np.stack(sample.observations))
        probe.novelty[name] = float(distances.mean())
        probe.novelty[name + " p90"] = float(np.percentile(distances, 90))
    return probe


def format_shift_probe(probe: ShiftProbe) -> str:
    lines = [
        "**One-step object-cell accuracy on each distribution.** `transition only` hands over the "
        "true parameters, which isolates how well the *transition* generalises; `end to end` is what "
        "the planner actually has (inferred belief and carried hidden state).",
        "",
        "| distribution | steps | transition only | end to end |",
        "|---|---|---|---|",
    ]
    for sample in (probe.collection, probe.execution):
        lines.append(
            f"| {sample.name} | {sample.steps} | {sample.accuracy_true_params:.3f} | "
            f"{sample.accuracy_end_to_end:.3f} |"
        )
    lines += [
        "",
        "**How different the states are** (mean per state):",
        "",
        "| feature | collection policy | closed-loop execution |",
        "|---|---|---|",
    ]
    collection_means = probe.collection.feature_means()
    execution_means = probe.execution.feature_means()
    for index, name in enumerate(FEATURE_NAMES):
        lines.append(f"| {name} | {collection_means[index]:.3f} | {execution_means[index]:.3f} |")
    lines += [
        "",
        "**Novelty**: mean distance to the nearest state seen in training, in units of the training "
        "spread per dimension. The collection-policy row is the baseline for unseen but "
        "in-distribution data.",
        "",
        "| distribution | mean nearest-training distance | 90th percentile |",
        "|---|---|---|",
    ]
    for name in ("collection policy (held out)", "closed-loop execution"):
        if name in probe.novelty:
            lines.append(
                f"| {name} | {probe.novelty[name]:.2f} | "
                f"{probe.novelty.get(name + ' p90', float('nan')):.2f} |"
            )
    return "\n".join(lines)
