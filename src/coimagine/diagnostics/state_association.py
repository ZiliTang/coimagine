"""Two short action chains, compared step by step between state-propagation schemes.

The question this answers
------------------------
When the planner imagines forward, does it keep, for every possible world, the state that
world has actually reached -- or does it mix the worlds together and then keep unrolling the
mixture?  The second is what the learned factored pipeline does: ``BeliefWorldModel.step``
and ``step_actions`` copy **one** marginal dynamic vector 16 times, hand it to the transition
head together with each parameter one-hot, and mix the results by ``b(theta)``.  A world in
which the release succeeded and a world in which it failed are then represented by a single
"released probability", and the next step is evaluated from combinations that never occurred.

What is exact here
------------------
``ExactWorldChain`` keeps an explicit joint over ``(origin, robots, theta, released)`` and
weight, executes the exact tabular transition **per world**, and aggregates only when asked
for a distribution.  It never decodes a mixture into point states.  Two variants are built
from it:

* ``step_exact``  -- per-world propagation (the reference);
* ``step_mixed``  -- collapse the joint to ``b(theta)`` times a marginal world distribution,
  recombine, and unroll from the recombined worlds.  This is the exact-dynamics analogue of
  what the learned model does, so it measures the cost of the factorisation **with zero
  learning error**.

``LearnedChain`` runs the same two propagation schemes on a frozen checkpoint:

* ``step_current``   -- the shipped behaviour (one vector, ``repeat`` 16, mix);
* ``step_per_theta`` -- 16 separate vectors, each advanced under its own parameter one-hot,
  mixed only for reporting.  This is a pure propagation change: no retraining, same network.

Its residual approximation is stated, not hidden: per-parameter state vectors preserve the
``theta <-> (cell, released)`` association but not the ``(cell, released)`` association
*within* a parameter, so it is not an exact joint belief.

First round deliberately bypasses the learned state estimator: every condition starts from
the same explicit initial state (``dynamic_vector_from_state``).  That is a privileged
diagnostic for locating the error, not a performance experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.dynamics import execute
from ..grid.sensing import ray_cast
from ..grid.planner import PHYSICS
from ..grid.state import Cell, ObjectState, RobotState, Scene, StaticMap
from ..learning.belief_model import DYNAMIC_DIM, N_PHYSICS
from ..learning.data import _joint_for, pose_to_index
from ..learning.features import ACTION_IDS, observation_vector

N_CELLS = 36
_EPS = 1e-12

#: ``(origin, robots, theta, released)``: one possible world.
WorldKey = Tuple[Cell, Tuple[RobotState, ...], int, bool]


def dynamic_vector_from_state(
    origin: Cell, released: bool, robots: Sequence[RobotState]
) -> np.ndarray:
    """The model's 41-dim dynamic vector for an explicitly known world state.

    Privileged by construction: this is how a *known* state is handed to the network while
    ``dyn_head`` is bypassed, so that a deviation cannot be blamed on the state estimator.
    Layout matches ``belief_model.DYNAMIC_DIM``: released probability, 36 object cells, four
    robot coordinates divided by ``_POSE_SCALE``.
    """

    vector = np.zeros((1, DYNAMIC_DIM))
    vector[0, 0] = 1.0 if released else 0.0
    vector[0, 1 + pose_to_index(origin)] = 1.0
    poses = [coordinate / 5.0 for robot in robots for coordinate in robot.pose]
    vector[0, 1 + N_CELLS :] = poses
    return vector


def _position_distribution(weights: Dict) -> np.ndarray:
    """Marginal over object cells from any weighted store keyed on a world."""

    distribution = np.zeros(N_CELLS)
    for key, weight in weights.items():
        origin = key[0]
        distribution[pose_to_index(origin)] += weight
    return distribution


def _released_by_theta(weights: Dict) -> np.ndarray:
    out = np.zeros(N_PHYSICS)
    for key, weight in weights.items():
        if key[3]:
            out[key[2]] += weight
    return out


def _goal_by_theta(weights: Dict, target: frozenset) -> np.ndarray:
    out = np.zeros(N_PHYSICS)
    for key, weight in weights.items():
        if ObjectState(key[0]).within(target):
            out[key[2]] += weight
    return out


def _goal_probability(weights: Dict, target: frozenset) -> float:
    return float(sum(w for key, w in weights.items() if ObjectState(key[0]).within(target)))


def _theta_marginal(weights: Dict) -> np.ndarray:
    out = np.zeros(N_PHYSICS)
    for key, weight in weights.items():
        out[key[2]] += weight
    return out


class ExactWorldChain:
    """Exact tabular transition with an explicit per-world joint (the reference)."""

    def __init__(
        self,
        static: StaticMap,
        initial_worlds: Dict[WorldKey, float],
        theta_prior: Optional[np.ndarray] = None,
    ) -> None:
        self.static = static
        self.worlds: Dict[WorldKey, float] = dict(initial_worlds)
        self.theta_prior = (
            np.asarray(theta_prior, dtype=float).copy()
            if theta_prior is not None
            else _theta_marginal(self.worlds)
        )
        self.mixed_worlds: Dict[WorldKey, float] = dict(initial_worlds)

    # -- per-world propagation (reference) ----------------------------------

    def step_exact(self, action_id: str) -> None:
        joint = _joint_for(None, action_id)
        updated: Dict[WorldKey, float] = {}
        for (origin, robots, theta, released), weight in self.worlds.items():
            scene = Scene(
                self.static, ObjectState(origin), robots, PHYSICS[theta], released
            )
            successor, _result = execute(scene, joint)
            key: WorldKey = (
                successor.obj.origin,
                successor.robots,
                theta,
                successor.released,
            )
            updated[key] = updated.get(key, 0.0) + weight
        self.worlds = updated

    # -- factorised propagation (the exact analogue of the learned mix) -----

    def step_mixed(self, action_id: str) -> None:
        """Collapse to ``b(theta)`` x marginal world, then unroll the recombined worlds.

        Every possible world keeps its own weight in the reference; here the parameter and the
        dynamic state are treated as independent and recombined, which is the exact-dynamics
        form of "one marginal vector, repeated per parameter".
        """

        joint = _joint_for(None, action_id)
        theta = _theta_marginal(self.mixed_worlds)
        marginal: Dict[Tuple[Cell, Tuple[RobotState, ...], bool], float] = {}
        for (origin, robots, _theta, released), weight in self.mixed_worlds.items():
            key = (origin, robots, released)
            marginal[key] = marginal.get(key, 0.0) + weight
        total = sum(marginal.values())
        if total > 0.0:
            marginal = {key: value / total for key, value in marginal.items()}
        total_theta = theta.sum()
        if total_theta > 0.0:
            theta = theta / total_theta

        updated: Dict[WorldKey, float] = {}
        for (origin, robots, released), world_weight in marginal.items():
            for index in range(N_PHYSICS):
                parameter_weight = float(theta[index])
                if parameter_weight <= 0.0 or world_weight <= 0.0:
                    continue
                scene = Scene(
                    self.static, ObjectState(origin), robots, PHYSICS[index], released
                )
                successor, _result = execute(scene, joint)
                key: WorldKey = (
                    successor.obj.origin,
                    successor.robots,
                    index,
                    successor.released,
                )
                updated[key] = updated.get(key, 0.0) + parameter_weight * world_weight
        norm = sum(updated.values())
        if norm > 0.0:
            updated = {key: value / norm for key, value in updated.items()}
        self.mixed_worlds = updated


class LearnedChain:
    """The frozen checkpoint under two propagation schemes."""

    def __init__(self, model, initial_vector: np.ndarray, belief: np.ndarray) -> None:
        self.model = model
        self.belief = np.asarray(belief, dtype=float).reshape(1, N_PHYSICS)
        self.belief = self.belief / max(self.belief.sum(), _EPS)
        #: shipped behaviour: one marginal vector, repeated per parameter inside ``step``
        self.current = np.array(initial_vector, dtype=float)
        #: propagation change: one vector per parameter, no recombination before advancing
        self.per_theta = np.repeat(np.array(initial_vector, dtype=float), N_PHYSICS, axis=0)

    def step_current(self, action: int) -> None:
        self.current = np.atleast_2d(
            self.model.step(self.current, action, self.belief)
        )

    def step_per_theta(self, action: int) -> None:
        advanced = np.zeros((N_PHYSICS, DYNAMIC_DIM))
        for index in range(N_PHYSICS):
            onehot = np.zeros((1, N_PHYSICS))
            onehot[0, index] = 1.0
            advanced[index] = self.model.step(
                self.per_theta[index : index + 1], action, onehot
            )[0]
        # recombine only for reporting; the next step still advances the per-parameter rows
        self.per_theta = advanced


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------


@dataclass
class StepSummary:
    step: int
    action: str
    position: np.ndarray
    goal_probability: float
    released_probability: float
    released_by_theta: np.ndarray
    goal_by_theta: np.ndarray

    def as_dict(self) -> Dict[str, object]:
        return {
            "step": self.step,
            "action": self.action,
            "position": [round(float(v), 6) for v in self.position],
            "goal_probability": round(self.goal_probability, 6),
            "released_probability": round(self.released_probability, 6),
            "released_by_theta": [round(float(v), 6) for v in self.released_by_theta],
            "goal_by_theta": [round(float(v), 6) for v in self.goal_by_theta],
        }


def _summarise_exact(
    weights: Dict[WorldKey, float], target: frozenset, step: int, action: str
) -> StepSummary:
    """Conditional ``P(released | theta)`` and ``P(goal | theta)`` from the per-world joint."""

    theta = _theta_marginal(weights)
    released_joint = _released_by_theta(weights)
    goal_joint = _goal_by_theta(weights, target)
    conditional_released = np.array(
        [released_joint[i] / theta[i] if theta[i] > _EPS else 0.0 for i in range(N_PHYSICS)]
    )
    conditional_goal = np.array(
        [goal_joint[i] / theta[i] if theta[i] > _EPS else 0.0 for i in range(N_PHYSICS)]
    )
    return StepSummary(
        step=step,
        action=action,
        position=_position_distribution(weights),
        goal_probability=_goal_probability(weights, target),
        released_probability=float(released_joint.sum()),
        released_by_theta=conditional_released,
        goal_by_theta=conditional_goal,
    )


def _summarise_learned(
    vectors: np.ndarray,
    belief: np.ndarray,
    target: frozenset,
    step: int,
    action: str,
    *,
    per_theta: bool,
) -> StepSummary:
    """The same quantities from the model's vectors.

    ``per_theta=False`` is the shipped scheme: one vector, so ``P(released | theta)`` and
    ``P(goal | theta)`` are forced equal for every parameter -- that equality *is* the lost
    association, and it is reported rather than hidden.  ``per_theta=True`` reads the
    per-parameter rows.  Either way a row is a within-parameter distribution over object
    cells and a released probability, so ``(cell, released)`` correlation inside one
    parameter is still not represented; the exact reference keeps it.
    """

    vectors = np.atleast_2d(vectors)
    goal_indices = [pose_to_index(origin) for origin in _goal_origins(target)]
    if per_theta:
        densities = []
        released = np.zeros(N_PHYSICS)
        goal = np.zeros(N_PHYSICS)
        for index in range(N_PHYSICS):
            density = np.clip(vectors[index, 1 : 1 + N_CELLS], 0.0, None)
            density = density / max(density.sum(), _EPS)
            densities.append(density)
            released[index] = float(np.clip(vectors[index, 0], 0.0, 1.0))
            goal[index] = float(density[goal_indices].sum()) if goal_indices else 0.0
        marginal = np.einsum("t,td->d", belief[0], np.stack(densities))
        released_probability = float(belief[0] @ released)
        goal_probability = float(belief[0] @ goal)
    else:
        density = np.clip(vectors[0, 1 : 1 + N_CELLS], 0.0, None)
        marginal = density / max(density.sum(), _EPS)
        released_probability = float(np.clip(vectors[0, 0], 0.0, 1.0))
        goal_probability = float(marginal[goal_indices].sum()) if goal_indices else 0.0
        released = np.full(N_PHYSICS, released_probability)
        goal = np.full(N_PHYSICS, goal_probability)
    return StepSummary(
        step=step,
        action=action,
        position=marginal,
        goal_probability=goal_probability,
        released_probability=released_probability,
        released_by_theta=released,
        goal_by_theta=goal,
    )


def _goal_origins(target: frozenset) -> Tuple[Cell, ...]:
    origins = []
    for y in range(0, 5):
        for x in range(0, 5):
            members = {(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)}
            if members <= target:
                origins.append((x, y))
    return tuple(origins)


# ---------------------------------------------------------------------------
# The four-condition comparison
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChainSpec:
    name: str
    physics: object  # HiddenPhysics
    origin: Cell
    actions: Tuple[str, ...]
    description: str


def default_chains():
    """Two short chains with the failure hypotheses kept in the belief.

    Both start from the same explicit state under a **uniform 16-way prior**, all
    unreleased.  The object starts three cells from the goal so that the goal is reached only
    on the last step, which keeps the chain out of the region where the encoder has no
    training data (trajectories stop at the goal).  The release chain is the user's example:
    worlds where robot A grips become released, worlds where it slips stay locked, and both
    are carried.
    """

    from ..grid.state import HiddenPhysics

    return (
        ChainSpec(
            name="no_release",
            physics=HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip")),
            origin=(1, 2),
            actions=("co_push:E", "co_push:E", "co_push:E"),
            description="continuous push, no release needed",
        ),
        ChainSpec(
            name="release_then_push",
            physics=HiddenPhysics(resistance=1, constraint="constrained", traction=("grip", "grip")),
            origin=(1, 2),
            actions=("release:0:E", "co_push:E", "co_push:E", "co_push:E"),
            description="release first, then continuous push",
        ),
    )


def _encode_true_state(model, scene: Scene) -> np.ndarray:
    """The model's own dynamic vector for the observation of a *known true* state.

    The transition head is trained on the encoder's soft output, not on a one-hot; feeding a
    hard one-hot is off-distribution and changes the prediction (measured: for the state
    after one co-push the one-hot input makes the network name a different cell while the
    encoded input names the true one).  Using the encoder on the true observation keeps the
    state estimator as the only thing that can be wrong here, and its accuracy is reported
    separately so a state-estimation failure is not read as a transition failure.
    """

    observation = np.asarray(
        observation_vector(
            ray_cast(scene, 0, radius=5), scene.robot_pose(0), (), ()
        )
    )
    return np.atleast_2d(model.dynamic_state(observation))


def _initial_worlds(scene: Scene, *, theta_only: Optional[int] = None):
    worlds: Dict[WorldKey, float] = {}
    if theta_only is None:
        weight = 1.0 / N_PHYSICS
        for index in range(N_PHYSICS):
            worlds[(scene.obj.origin, scene.robots, index, scene.released)] = weight
    else:
        worlds[(scene.obj.origin, scene.robots, theta_only, scene.released)] = 1.0
    return worlds


def _tv(a: np.ndarray, b: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(a) - np.asarray(b)).sum())


def run_chain_probe(model, chain: ChainSpec, *, tolerance: float = 0.02) -> Dict[str, object]:
    """Run all four conditions on one chain and locate the first clear deviation.

    Conditions, in the order the user asked for them:

    1. ``learned_single_step_true_theta`` -- true state re-read every step, true parameter;
    2. ``learned_recursion_true_theta``    -- true parameter, own prediction fed back;
    3. ``exact_mixed``                     -- exact dynamics, per-world vs factorised;
    4. ``learned_current`` / ``learned_per_theta`` -- frozen network, shipped vs per-parameter
       propagation, under the same uniform prior.
    """

    from ..learning.data import probe_scene

    scene = probe_scene(chain.origin, chain.physics)
    target = frozenset(scene.static.target)
    theta_true = PHYSICS.index(chain.physics)
    onehot_true = np.zeros((1, N_PHYSICS))
    onehot_true[0, theta_true] = 1.0

    reference = ExactWorldChain(scene.static, _initial_worlds(scene))
    reference_true = ExactWorldChain(scene.static, _initial_worlds(scene, theta_only=theta_true))
    exact_mixed = ExactWorldChain(scene.static, _initial_worlds(scene))

    initial_vector = _encode_true_state(model, scene)
    uniform = np.full(N_PHYSICS, 1.0 / N_PHYSICS)
    learned = LearnedChain(model, initial_vector, uniform)
    learned_true_state = LearnedChain(model, initial_vector, onehot_true)

    def encoder_row(current: Scene) -> Dict[str, object]:
        encoded = _encode_true_state(model, current)
        cells = np.asarray(encoded[0, 1 : 1 + N_CELLS], dtype=float)
        top = int(np.argmax(cells))
        return {
            "true_cell": list(current.obj.origin),
            "argmax_cell": [top % 6, top // 6],
            "probability_on_true_cell": round(float(cells[pose_to_index(current.obj.origin)]), 6),
            "true_released": bool(current.released),
            "released_probability": round(float(np.clip(encoded[0, 0], 0.0, 1.0)), 6),
        }

    steps: List[Dict[str, object]] = [
        {
            "step": 0,
            "action": None,
            "encoder": encoder_row(scene),
            "reference": _summarise_exact(reference.worlds, target, 0, "").as_dict(),
            "reference_true_theta": _summarise_exact(reference_true.worlds, target, 0, "").as_dict(),
            "exact_mixed": _summarise_exact(exact_mixed.mixed_worlds, target, 0, "").as_dict(),
            "learned_current": _summarise_learned(
                learned.current, learned.belief, target, 0, "", per_theta=False
            ).as_dict(),
            "learned_per_theta": _summarise_learned(
                learned.per_theta, learned.belief, target, 0, "", per_theta=True
            ).as_dict(),
            "learned_single_step_true_theta": _summarise_learned(
                initial_vector, onehot_true, target, 0, "", per_theta=False
            ).as_dict(),
            "learned_recursion_true_theta": _summarise_learned(
                initial_vector, onehot_true, target, 0, "", per_theta=False
            ).as_dict(),
        }
    ]

    true_scene = scene
    for step, action_id in enumerate(chain.actions, start=1):
        action = ACTION_IDS.index(action_id)

        reference.step_exact(action_id)
        reference_true.step_exact(action_id)
        exact_mixed.step_mixed(action_id)
        learned.step_current(action)
        learned.step_per_theta(action)

        # condition 1: the true state re-encoded and fed again, under the true parameter
        single = np.atleast_2d(model.step(_encode_true_state(model, true_scene), action, onehot_true))
        # condition 2: the same chain, but the network's own prediction is fed back
        learned_true_state.step_current(action)

        true_scene, _result = execute(true_scene, _joint_for(None, action_id))
        # the encoder row describes the same post-action state the summaries describe
        encoder = encoder_row(true_scene)

        steps.append(
            {
                "step": step,
                "action": action_id,
                "encoder": encoder,
                "reference": _summarise_exact(reference.worlds, target, step, action_id).as_dict(),
                "reference_true_theta": _summarise_exact(
                    reference_true.worlds, target, step, action_id
                ).as_dict(),
                "exact_mixed": _summarise_exact(
                    exact_mixed.mixed_worlds, target, step, action_id
                ).as_dict(),
                "learned_current": _summarise_learned(
                    learned.current, learned.belief, target, step, action_id, per_theta=False
                ).as_dict(),
                "learned_per_theta": _summarise_learned(
                    learned.per_theta, learned.belief, target, step, action_id, per_theta=True
                ).as_dict(),
                "learned_single_step_true_theta": _summarise_learned(
                    single, onehot_true, target, step, action_id, per_theta=False
                ).as_dict(),
                "learned_recursion_true_theta": _summarise_learned(
                    learned_true_state.current, onehot_true, target, step, action_id,
                    per_theta=False,
                ).as_dict(),
            }
        )

    conditions = (
        "exact_mixed",
        "learned_current",
        "learned_per_theta",
        "learned_single_step_true_theta",
        "learned_recursion_true_theta",
    )
    deviations: Dict[str, List[Dict[str, object]]] = {name: [] for name in conditions}
    for record in steps:
        reference_position = np.asarray(record["reference"]["position"])
        reference_goal = record["reference"]["goal_probability"]
        for name in conditions:
            entry = record[name]
            position = np.asarray(entry["position"])
            released_gap = float(
                np.max(
                    np.abs(
                        np.asarray(entry["released_by_theta"])
                        - np.asarray(record["reference"]["released_by_theta"])
                    )
                )
            )
            goal_by_theta_gap = float(
                np.max(
                    np.abs(
                        np.asarray(entry["goal_by_theta"])
                        - np.asarray(record["reference"]["goal_by_theta"])
                    )
                )
            )
            deviations[name].append(
                {
                    "step": record["step"],
                    "position_tv": _tv(position, reference_position),
                    "goal_probability": entry["goal_probability"],
                    "goal_probability_gap": entry["goal_probability"] - reference_goal,
                    "max_released_by_theta_gap": released_gap,
                    "max_goal_by_theta_gap": goal_by_theta_gap,
                }
            )

    first_deviation: Dict[str, object] = {}
    for name, rows in deviations.items():
        flagged = [
            row
            for row in rows
            if row["position_tv"] > tolerance or abs(row["goal_probability_gap"]) > tolerance
        ]
        first_deviation[name] = flagged[0] if flagged else None

    return {
        "chain": chain.name,
        "description": chain.description,
        "true_physics": chain.physics.label,
        "true_physics_index": theta_true,
        "origin": list(chain.origin),
        "robots": [list(r.pose) for r in scene.robots],
        "actions": list(chain.actions),
        "prior": "uniform over the 16 contact types, all unreleased",
        "tolerance": tolerance,
        "steps": steps,
        "deviations": deviations,
        "first_deviation": first_deviation,
        "reference_released_by_theta": steps[-1]["reference"]["released_by_theta"],
        "reference_goal_by_theta": steps[-1]["reference"]["goal_by_theta"],
    }


# ---------------------------------------------------------------------------
# The key single-step transition, on real states
# ---------------------------------------------------------------------------


def key_transition_probe(model, *, n_episodes: int = 400, horizon: int = 8, seed: int = 0) -> Dict[str, object]:
    """Is the transition right on the pushes that actually move the object?

    Aggregate one-step accuracy is dominated by the easy "nothing moves" case: in this data
    about 87% of contact actions do not move the object, so a model that always answers
    "still" scores high.  This probe splits the two populations and reports the productive
    pushes separately, under the **true parameter** (network error alone) and under a
    uniform parameter belief (network plus mixture).

    Reference is the simulator's own outcome, which is deterministic given the true
    parameter, so "predicted move" should be 1.0 on the true-move population under the true
    parameter.
    """

    from ..learning.world_model_data import collect_trajectories

    trajectories = collect_trajectories(
        n_episodes=n_episodes, horizon=horizon, seed=seed, lead_in=2
    )
    transitions = [step for trajectory in trajectories for step in trajectory.transitions]
    uniform = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    actions = {
        "co_push:E": ACTION_IDS.index("co_push:E"),
        "push:0:E": ACTION_IDS.index("push:0:E"),
        "push:1:E": ACTION_IDS.index("push:1:E"),
    }
    out: Dict[str, object] = {
        "n_transitions": len(transitions),
        "n_episodes": n_episodes,
        "conditions": {},
    }
    for action_id, action_index in actions.items():
        rows = [step for step in transitions if step.action == action_index]
        entry: Dict[str, object] = {"n": len(rows), "encoder_cell_top1": 0}
        for condition in ("true_theta", "uniform_belief"):
            entry[condition] = {
                "true_moved": 0,
                "true_not_moved": 0,
                "predicted_move_on_true_move": 0,
                "cell_top1_on_true_move": 0,
                "mean_move_probability_on_true_move": 0.0,
                "predicted_move_on_true_not_moved": 0,
            }
        for step in rows:
            dynamic = model.dynamic_state(step.observation)
            entry["encoder_cell_top1"] += int(
                int(np.argmax(dynamic[0, 1 : 1 + N_CELLS])) == step.object_cell
            )
            for condition, belief in (
                ("true_theta", np.eye(N_PHYSICS)[[step.physics]]),
                ("uniform_belief", uniform),
            ):
                states = model.step_actions(dynamic, belief)
                cells = np.asarray(states[action_index][1 : 1 + N_CELLS], dtype=float)
                total = cells.sum()
                cells = cells / total if total > 0 else cells
                move_probability = float(cells.sum() - cells[step.object_cell])
                bucket = entry[condition]
                if step.moved:
                    bucket["true_moved"] += 1
                    bucket["predicted_move_on_true_move"] += int(move_probability >= 0.5)
                    bucket["cell_top1_on_true_move"] += int(
                        int(np.argmax(cells)) == step.next_object_cell
                    )
                    bucket["mean_move_probability_on_true_move"] += move_probability
                else:
                    bucket["true_not_moved"] += 1
                    bucket["predicted_move_on_true_not_moved"] += int(move_probability >= 0.5)
        for condition in ("true_theta", "uniform_belief"):
            bucket = entry[condition]
            moved = max(1, bucket["true_moved"])
            not_moved = max(1, bucket["true_not_moved"])
            bucket["predicted_move_rate_on_true_move"] = (
                bucket["predicted_move_on_true_move"] / moved
            )
            bucket["cell_top1_rate_on_true_move"] = bucket["cell_top1_on_true_move"] / moved
            bucket["mean_move_probability_on_true_move"] /= moved
            bucket["predicted_move_rate_on_true_not_moved"] = (
                bucket["predicted_move_on_true_not_moved"] / not_moved
            )
        entry["encoder_cell_top1_rate"] = entry["encoder_cell_top1"] / max(1, len(rows))
        out["conditions"][action_id] = entry
    return out
