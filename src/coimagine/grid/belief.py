"""Per-robot Bayesian belief over the hidden object pose and hidden physics.

The hypothesis space is finite::

    h = (object origin, resistance, constraint mode, traction of both robots, released)

The object's true pose is **never** broadcast, so each robot maintains its own
posterior from its own observations. Two metrics are reported separately, as the
protocol requires:

* **localization** -- how well the robot knows *where* the object is;
* **physics belief** -- how well it knows *what the object is like*.

Conflating them would let a method look bad because it cannot see rather than
because of its message or planning.

The transition is deterministic given a hypothesis (the simulator has no process
noise in Stage 2); all uncertainty is in the state and in the sensing channels. The
filter therefore just moves mass between hypotheses and reweights it by the
published likelihoods. The scan likelihood scores the cells the robot actually
reported; occlusion-induced differences in *which* cells are reported are not yet
modelled, which is recorded as a limitation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .dynamics import (
    NO_FEEDBACK_NOISE,
    FeedbackNoise,
    JointSkill,
    execute,
    feedback_likelihood,
)
from .sensing import KIND_FREE, KIND_OBJECT, KIND_WALL, OccupancyScan, SensorNoise
from .state import (
    Cell,
    HiddenPhysics,
    ObjectState,
    RobotState,
    Scene,
    StaticMap,
    enumerate_hidden_physics,
)

Belief = Dict["Hypothesis", float]


@dataclass(frozen=True)
class Hypothesis:
    origin: Cell
    physics: HiddenPhysics
    released: bool = False


# --------------------------------------------------------------------------
# Hypothesis space
# --------------------------------------------------------------------------


def valid_object_origins(
    static: StaticMap,
    *,
    blocked: Iterable[Cell] = (),
    width: int = 2,
    height: int = 2,
) -> Tuple[Cell, ...]:
    """Every 2x2 placement that avoids prior-known walls and blocked cells.

    ``blocked`` normally holds the (publicly known) robot poses: a hypothesis that
    puts the object on top of a robot is physically impossible, and dropping it is
    based on public information only.
    """

    blocked_set = frozenset(blocked)
    origins: List[Cell] = []
    for y in range(static.height - height + 1):
        for x in range(static.width - width + 1):
            cells = ObjectState((x, y), width, height).cells
            if all(not static.is_wall(c) for c in cells) and not (cells & blocked_set):
                origins.append((x, y))
    return tuple(origins)


def uniform_prior(
    static: StaticMap,
    *,
    robot_poses: Sequence[Cell] = (),
    physics_values: Optional[Sequence[HiddenPhysics]] = None,
    released_values: Tuple[bool, ...] = (False,),
) -> Belief:
    physics_values = tuple(physics_values) if physics_values is not None else enumerate_hidden_physics()
    origins = valid_object_origins(static, blocked=robot_poses)
    if not origins:
        raise ValueError("No valid object placement fits this map.")
    weight = 1.0 / (len(origins) * len(physics_values) * len(released_values))
    belief: Belief = {}
    for origin in origins:
        for physics in physics_values:
            for released in released_values:
                belief[Hypothesis(origin, physics, released)] = weight
    return belief


def normalize(belief: Mapping[Hypothesis, float]) -> Belief:
    total = sum(belief.values())
    if total <= 0.0:
        return {}
    return {h: w / total for h, w in belief.items() if w > 0.0}


# --------------------------------------------------------------------------
# Scan likelihood
# --------------------------------------------------------------------------


def scan_log_likelihood(
    scan: OccupancyScan,
    origin: Cell,
    static: StaticMap,
    noise: SensorNoise,
    *,
    width: int = 2,
    height: int = 2,
) -> float:
    """Log ``P(scan | object at ``origin``)`` under the stated sensing channel.

    An ``object`` report is a hard positive (the channel never invents one); a
    ``free`` report penalises hypotheses that place the object there.
    """

    cells = ObjectState(origin, width, height).cells
    drop = noise.drop_prob
    false_free = noise.false_free_prob
    false_occupied = noise.false_occupied_prob
    logp = 0.0
    for cell, kind in scan.observed.items():
        if kind == KIND_OBJECT:
            if cell not in cells:
                return -math.inf
            logp += math.log(max(1e-300, (1.0 - drop) * (1.0 - false_free)))
        elif kind == KIND_FREE:
            if cell in cells:
                if false_free <= 0.0:
                    return -math.inf
                logp += math.log(max(1e-300, (1.0 - drop) * false_free))
            else:
                logp += math.log(max(1e-300, (1.0 - drop) * (1.0 - false_occupied)))
        elif kind == KIND_WALL:
            if cell in cells:
                return -math.inf
        # Robot cells carry no information about the object.
    return logp


def update_with_scan(
    belief: Belief,
    scan: OccupancyScan,
    static: StaticMap,
    noise: SensorNoise = SensorNoise(),
) -> Belief:
    """Reweight the belief with the robot's own occupancy scan."""

    updated: Belief = {}
    for hypothesis, weight in belief.items():
        logp = scan_log_likelihood(scan, hypothesis.origin, static, noise)
        if logp == -math.inf:
            continue
        updated[hypothesis] = weight * math.exp(logp)
    return normalize(updated)


# --------------------------------------------------------------------------
# Advancing on one executed joint skill
# --------------------------------------------------------------------------


def _scene_for(
    static: StaticMap,
    hypothesis: Hypothesis,
    robots: Sequence[RobotState],
) -> Optional[Scene]:
    try:
        return Scene(
            static=static,
            obj=ObjectState(hypothesis.origin),
            robots=tuple(robots),
            physics=hypothesis.physics,
            released=hypothesis.released,
        )
    except ValueError:
        # The hypothesis contradicts the publicly known robot poses (or the map),
        # so it is impossible rather than merely unlikely.
        return None


def advance(
    belief: Belief,
    static: StaticMap,
    robots: Sequence[RobotState],
    joint: JointSkill,
    *,
    robot_id: int,
    observed_feedback: str,
    y_public: str,
    noise: FeedbackNoise = NO_FEEDBACK_NOISE,
    probe_cost: float = 0.3,
) -> Belief:
    """Move and reweight the belief after one executed joint skill.

    ``observed_feedback`` is the executing robot's own private symbol; ``y_public``
    is the shared motion event. A hypothesis that predicts a different public event
    is eliminated; feedback is reweighted through the published channel.
    """

    updated: Belief = {}
    for hypothesis, weight in belief.items():
        scene = _scene_for(static, hypothesis, robots)
        if scene is None:
            continue
        successor, result = execute(scene, joint, probe_cost=probe_cost)
        if result.y_public != y_public:
            continue
        nominal = result.feedback[robot_id]
        likelihood = feedback_likelihood(observed_feedback, nominal, noise)
        if likelihood <= 0.0:
            continue
        new_hypothesis = Hypothesis(
            successor.obj.origin, successor.physics, successor.released
        )
        updated[new_hypothesis] = updated.get(new_hypothesis, 0.0) + weight * likelihood
    return normalize(updated)


# --------------------------------------------------------------------------
# Marginals and the two separate error metrics
# --------------------------------------------------------------------------


def localization_marginal(belief: Belief) -> Dict[Cell, float]:
    out: Dict[Cell, float] = {}
    for hypothesis, weight in belief.items():
        out[hypothesis.origin] = out.get(hypothesis.origin, 0.0) + weight
    return out


def physics_marginal(belief: Belief) -> Dict[HiddenPhysics, float]:
    out: Dict[HiddenPhysics, float] = {}
    for hypothesis, weight in belief.items():
        out[hypothesis.physics] = out.get(hypothesis.physics, 0.0) + weight
    return out


def released_probability(belief: Belief) -> float:
    return sum(w for h, w in belief.items() if h.released)


def most_likely_origin(belief: Belief) -> Optional[Cell]:
    marginal = localization_marginal(belief)
    if not marginal:
        return None
    return max(marginal.items(), key=lambda kv: (kv[1], kv[0]))[0]


def most_likely_physics(belief: Belief) -> Optional[HiddenPhysics]:
    marginal = physics_marginal(belief)
    if not marginal:
        return None
    return max(marginal.items(), key=lambda kv: (kv[1], kv[0]))[0]


def localization_metrics(belief: Belief, true_origin: Cell) -> Dict[str, float]:
    """How well the robot knows *where* the object is."""

    marginal = localization_marginal(belief)
    if not marginal:
        return {
            "p_correct_origin": 0.0,
            "expected_origin_distance": float("nan"),
            "expected_cell_error": float("nan"),
        }
    true_cells = ObjectState(true_origin).cells
    expected_distance = 0.0
    expected_cell_error = 0.0
    for origin, weight in marginal.items():
        expected_distance += weight * (
            abs(origin[0] - true_origin[0]) + abs(origin[1] - true_origin[1])
        )
        expected_cell_error += weight * len(true_cells ^ ObjectState(origin).cells)
    return {
        "p_correct_origin": marginal.get(true_origin, 0.0),
        "expected_origin_distance": expected_distance,
        "expected_cell_error": expected_cell_error / max(1, len(true_cells)),
    }


def physics_metrics(belief: Belief, true_physics: HiddenPhysics) -> Dict[str, float]:
    """How well the robot knows *what the object is like*."""

    marginal = physics_marginal(belief)
    if not marginal:
        return {"p_correct_physics": 0.0, "brier_physics": float("nan")}
    types = enumerate_hidden_physics()
    p_correct = marginal.get(true_physics, 0.0)
    brier = 0.0
    for physics in types:
        target = 1.0 if physics == true_physics else 0.0
        brier += (marginal.get(physics, 0.0) - target) ** 2
    return {"p_correct_physics": p_correct, "brier_physics": brier}
