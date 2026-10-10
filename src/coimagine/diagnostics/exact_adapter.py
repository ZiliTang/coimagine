"""The exact tabular world model behind the learned planner's interface.

Why this exists
---------------
The planner used by the world-model line (``world_model_probe.plan_with_model``) reads a
model through exactly two methods::

    predict_next(observation, action)   -> per-step dict
    continue_from(predicted, action)    -> per-step dict

with keys ``pose_logits``, ``moved_logit``, ``cost``, ``observation`` (and whatever else the
model wants to carry). :class:`ExactBeliefModel` implements **that same interface** with the
tabular dynamics and an explicit Bayes posterior over the hidden contact physics.  The
planner code, the candidate set, the scoring rule, the execution loop, the termination test
and the tie-break are therefore identical between the learned and the exact condition: the
only thing swapped is the model.  That is what isolates *search/scoring* error from *model*
error.

Information boundary
--------------------
The adapter is built from deployment-legal quantities only:

* the robot's own observation vector (scan plane + own pose + private feedback histogram +
  public event histogram),
* the ordered action sequence and robot 0's own feedback history,
* the public static map, both robot poses and the public event history.

It never receives a :class:`~coimagine.grid.state.Scene`, the true
:class:`~coimagine.grid.state.HiddenPhysics`, the teammate's private feedback, or the
simulator.  Every step marginalises the exact transition over the current posterior -- the
"compute the transition under every belief hypothesis and combine by legal probability"
requirement -- instead of reading the hidden parameters.

Two conditions are offered:

``condition_on_feedback=True``
    the full exact posterior over the 16 contact types given the observation history;
``condition_on_feedback=False``
    the *same exact dynamics* but the physics posterior frozen at the uniform prior.  This
    second condition uses strictly **less** information than the learned model (which at
    least sees the feedback histogram), so if it still solves episodes the learned model is
    the thing at fault, not the planner and not the information available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.dynamics import FEEDBACK_SYMBOLS, execute
from ..grid.planner import PHYSICS
from ..grid.sensing import (
    KIND_FREE,
    KIND_OBJECT,
    KIND_ROBOT,
    OccupancyScan,
    ray_cast,
)
from ..grid.state import Cell, ObjectState, RobotState, Scene, StaticMap
from ..learning.data import _joint_for
from ..learning.features import (
    ACTION_IDS,
    CELL_KINDS,
    GRID,
    N_CELL_BLOCKS,
    OBSERVATION_DIM,
    observation_vector,
)

N_CELLS = GRID * GRID
_EPS = 1e-12
_LOGIT_CLIP = 30.0

#: ``(origin, robots, physics index, released)`` -- robots are part of the key because a
#: successful push advances the pushers, so different hypotheses can end a step at
#: different poses.
ExactKey = Tuple[Cell, Tuple[RobotState, ...], int, bool]


def pose_to_index(origin: Cell) -> int:
    return origin[1] * GRID + origin[0]


def scan_from_observation(
    observation: np.ndarray, own_pose: Cell, *, radius: int, robot_id: int = 0
) -> OccupancyScan:
    """Rebuild the occupancy scan from the observation vector.

    The vector's first ``GRID * GRID * N_CELL_BLOCKS`` entries are the one-hot scan plane,
    so this is lossless for the cells the robot actually observed.  Cells outside the
    radius or occluded stay absent, i.e. unknown, exactly as in the runtime scan.
    """

    plane = np.asarray(observation, dtype=np.float64)[: GRID * GRID * N_CELL_BLOCKS]
    observed: Dict[Cell, str] = {}
    for index in range(N_CELLS):
        block = plane[index * N_CELL_BLOCKS : (index + 1) * N_CELL_BLOCKS]
        kind_index = int(np.argmax(block))
        if kind_index == 0 or block[kind_index] <= 0.0:
            continue
        y, x = divmod(index, GRID)
        observed[(x, y)] = CELL_KINDS[kind_index - 1]
    return OccupancyScan(robot_id, own_pose, radius, observed)


def object_origins_from_scan(
    scan: OccupancyScan, static: StaticMap
) -> Tuple[Tuple[Cell, ...], bool]:
    """Every 2x2 placement consistent with what the robot's own scan saw.

    Returns the candidate origins and whether the object was seen at all.  A placement is
    consistent when it contains every observed object cell and no cell the scan reported as
    free or as the teammate.  With the noiseless scans used by the closed loop this is a
    single origin; with a noisy or occluded scan it is a distribution, which the adapter
    carries as an explicit prior.
    """

    observed_object = {c for c, k in scan.observed.items() if k == KIND_OBJECT}
    contradicting = {
        c for c, k in scan.observed.items() if k in (KIND_FREE, KIND_ROBOT)
    }
    candidates: List[Cell] = []
    for oy in range(static.height - 1):
        for ox in range(static.width - 1):
            cells = {(ox + dx, oy + dy) for dx in (0, 1) for dy in (0, 1)}
            if any(static.is_wall(c) for c in cells):
                continue
            if not observed_object <= cells:
                continue
            if cells & contradicting:
                continue
            candidates.append((ox, oy))
    return tuple(candidates), bool(observed_object)


def _shift_origin_back(origin: Cell, public_history: Sequence[str]) -> Cell:
    """The object's origin before the recorded public events.

    Every object displacement in this action set is one cell east (all pushes are
    ``push:<r>:E``), so a public ``moved`` at step ``t`` means the object was one cell west
    of where it is now.  This is public information and lets the adapter reconstruct the
    initial pose without being handed the simulator's initial state.
    """

    moves = sum(1 for event in public_history if event == "moved")
    return (origin[0] - moves, origin[1])


@dataclass
class ExactStep:
    """One imagined step, in the learned models' per-step format."""

    cost: np.ndarray
    pose_logits: np.ndarray
    moved_logit: np.ndarray
    observation: np.ndarray
    exact_belief: Dict[ExactKey, float]


class ExactBeliefModel:
    """Exact tabular transition + Bayes filter behind ``predict_next`` / ``continue_from``."""

    def __init__(
        self,
        observation: np.ndarray,
        *,
        static: StaticMap,
        robots: Tuple[RobotState, ...],
        initial_robots: Tuple[RobotState, ...],
        actions: Sequence[str] = (),
        history_a: Sequence[str] = (),
        public_history: Sequence[str] = (),
        radius: int = 5,
        condition_on_feedback: bool = True,
        initial_robots_are_public: bool = True,
    ) -> None:
        self.observation = np.asarray(observation, dtype=np.float64)
        self.static = static
        self.robots = tuple(robots)
        self.initial_robots = tuple(initial_robots)
        self.actions = tuple(actions)
        self.history_a = tuple(history_a)
        self.public_history = tuple(public_history)
        self.radius = radius
        self.condition_on_feedback = condition_on_feedback
        #: Recorded so the report can say exactly how much public information the exact
        #: condition was given about the team's own initial configuration.
        self.initial_robots_are_public = initial_robots_are_public

        scan = scan_from_observation(
            self.observation, self.robots[0].pose, radius=radius
        )
        origins, self.object_observed = object_origins_from_scan(scan, static)
        if not origins:
            # No consistent placement: fall back to every in-bounds, wall-free 2x2 origin.
            origins = tuple(
                (ox, oy)
                for oy in range(static.height - 1)
                for ox in range(static.width - 1)
                if not any(
                    static.is_wall((ox + dx, oy + dy)) for dx in (0, 1) for dy in (0, 1)
                )
            )
        self.origins = origins
        self.origin_weight = {origin: 1.0 / len(origins) for origin in origins}
        #: ``(initial origin, physics index) -> final released flag`` from the replay.
        self._released: Dict[Tuple[Cell, int], bool] = {}

        if condition_on_feedback:
            physics_weight = self._posterior_from_history(origins)
        else:
            # Same exact dynamics, but the parameters stay at the prior.  The released
            # chain is still tracked exactly, since it is part of the state, not the
            # parameter belief.
            for origin in origins:
                initial = _shift_origin_back(origin, self.public_history)
                for index in range(len(PHYSICS)):
                    self._fill_released(initial, origin, index)
            physics_weight = {index: 1.0 for index in range(len(PHYSICS))}

        belief: Dict[ExactKey, float] = {}
        for origin, origin_w in self.origin_weight.items():
            for index, physics_w in physics_weight.items():
                if physics_w <= 0.0:
                    continue
                released = self._released.get((origin, index), False)
                key: ExactKey = (origin, self.robots, index, released)
                belief[key] = belief.get(key, 0.0) + origin_w * physics_w
        self._belief = _normalise(belief)

    # -- history conditioning ------------------------------------------------

    def _posterior_from_history(
        self, origins: Sequence[Cell]
    ) -> Dict[int, float]:
        """``P(physics | own feedback history, public events, current observed pose)``.

        Each hypothesis is replayed forward from the reconstructed initial pose under the
        exact dynamics; a hypothesis survives iff the replay reproduces robot 0's private
        feedback, the public event history and the object pose the robot can see now.  The
        teammate's traction is not identifiable from any of this, so it correctly stays at
        its prior.
        """

        weights: Dict[int, float] = {}
        for origin in origins:
            initial = _shift_origin_back(origin, self.public_history)
            for index in range(len(PHYSICS)):
                likelihood = self._replay_likelihood(initial, origin, index)
                if likelihood > 0.0:
                    weights[index] = weights.get(index, 0.0) + likelihood
        if not weights:
            return {index: 1.0 for index in range(len(PHYSICS))}
        return weights

    def _replay_likelihood(self, initial: Cell, current: Cell, index: int) -> float:
        scene = Scene(
            static=self.static,
            obj=ObjectState(initial),
            robots=self.initial_robots,
            physics=PHYSICS[index],
            released=False,
        )
        for action_id, observed_feedback, observed_public in zip(
            self.actions, self.history_a, self.public_history
        ):
            scene, result = execute(scene, _joint_for(None, action_id))
            if result.feedback[0] != observed_feedback:
                return 0.0
            if result.y_public != observed_public:
                return 0.0
        if scene.obj.origin != current:
            return 0.0
        self._released[(current, index)] = scene.released
        return 1.0

    def _fill_released(self, initial: Cell, current: Cell, index: int) -> bool:
        """Track the released chain exactly even when the parameters are not conditioned."""

        scene = Scene(
            static=self.static,
            obj=ObjectState(initial),
            robots=self.initial_robots,
            physics=PHYSICS[index],
            released=False,
        )
        for action_id in self.actions:
            scene, _result = execute(scene, _joint_for(None, action_id))
        self._released[(current, index)] = scene.released
        return scene.released

    # -- the learned models' interface --------------------------------------

    def predict_next(self, observation: np.ndarray, action: int) -> Dict[str, object]:
        return self._predict(
            self._belief,
            ACTION_IDS[action],
            extend_observation=True,
            history_a=self.history_a,
            public_history=self.public_history,
        )

    def continue_from(self, predicted: Dict[str, object], action: int) -> Dict[str, object]:
        belief = predicted.get("exact_belief") or self._belief
        history_a = predicted.get("history_a", self.history_a)
        public_history = predicted.get("public_history", self.public_history)
        return self._predict(
            belief,
            ACTION_IDS[action],
            # ``plan_with_model`` never reads the imagined *observation* of a continuation
            # (it only reads cost and pose_logits), so computing 16 extra ray casts there
            # would be pure waste.
            extend_observation=False,
            history_a=history_a,
            public_history=public_history,
        )

    def _predict(
        self,
        belief: Dict[ExactKey, float],
        action_id: str,
        *,
        extend_observation: bool,
        history_a: Sequence[str],
        public_history: Sequence[str],
    ) -> Dict[str, object]:
        joint = _joint_for(None, action_id)
        cost = 0.0
        moved = 0.0
        pose_weights = np.zeros(N_CELLS)
        expected_observation = np.zeros(OBSERVATION_DIM)
        posterior: Dict[ExactKey, float] = {}

        for (origin, robots, index, released), weight in belief.items():
            scene = Scene(
                static=self.static,
                obj=ObjectState(origin),
                robots=robots,
                physics=PHYSICS[index],
                released=released,
            )
            successor, result = execute(scene, joint)
            cost += weight * float(result.cost)
            moved += weight * (1.0 if result.y_public == "moved" else 0.0)
            pose_weights[pose_to_index(successor.obj.origin)] += weight
            key: ExactKey = (
                successor.obj.origin,
                successor.robots,
                index,
                successor.released,
            )
            posterior[key] = posterior.get(key, 0.0) + weight
            if extend_observation:
                scan = ray_cast(successor, 0, radius=self.radius)
                vector = observation_vector(
                    scan,
                    successor.robot_pose(0),
                    tuple(history_a) + (result.feedback[0],),
                    tuple(public_history) + (result.y_public,),
                )
                expected_observation += weight * np.asarray(vector, dtype=np.float64)

        probability = float(np.clip(moved, 0.0, 1.0))
        moved_logit = float(
            np.clip(
                np.log(max(probability, _EPS) / max(1.0 - probability, _EPS)),
                -_LOGIT_CLIP,
                _LOGIT_CLIP,
            )
        )
        return {
            "cost": np.asarray([cost], dtype=np.float64),
            "pose_logits": np.log(np.maximum(pose_weights, _EPS))[None, :],
            "moved_logit": np.asarray([moved_logit], dtype=np.float64),
            "observation": expected_observation[None, :],
            "exact_belief": _normalise(posterior),
            "history_a": tuple(history_a),
            "public_history": tuple(public_history),
        }

    # -- diagnostics ---------------------------------------------------------

    @property
    def belief(self) -> Dict[ExactKey, float]:
        return dict(self._belief)

    def belief_summary(self) -> Dict[str, object]:
        physics_mass: Dict[int, float] = {}
        for (_origin, _robots, index, _released), weight in self._belief.items():
            physics_mass[index] = physics_mass.get(index, 0.0) + weight
        return {
            "n_entries": len(self._belief),
            "object_observed": self.object_observed,
            "n_origins": len(self.origins),
            "physics_mass": {k: v for k, v in sorted(physics_mass.items())},
            "max_physics_mass": max(physics_mass.values()) if physics_mass else 0.0,
        }


def _normalise(weights: Dict[object, float]) -> Dict:
    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in weights.items() if v > 0.0}


def exact_model_factory(*, condition_on_feedback: bool = True):
    """Build a per-boundary factory compatible with ``run_closed_loop``.

    ``run_closed_loop`` calls the factory once per decision boundary with the deployment
    inputs it has at that boundary: the current observation, the public static map, both
    robot poses, the executed action sequence and the two histories.  The simulator scene
    is not among them.
    """

    def build(context: Dict[str, object]) -> ExactBeliefModel:
        return ExactBeliefModel(
            context["observation"],
            static=context["static"],
            robots=context["robots"],
            initial_robots=context["initial_robots"],
            actions=context["actions"],
            history_a=context["history_a"],
            public_history=context["public_history"],
            radius=int(context.get("radius", 5)),
            condition_on_feedback=condition_on_feedback,
        )

    return build


# ---------------------------------------------------------------------------
# The exact model behind the factored (belief-space) planner's interface
# ---------------------------------------------------------------------------


class ExactFactoredModel:
    """Exact tabular model behind :class:`~coimagine.learning.belief_model.BeliefWorldModel`.

    The factored planner (``belief_probe.plan_with_search`` / ``plan_with_belief`` and the
    episode drivers ``run_search_episode`` / ``run_belief_episode``) consumes a model through
    a different set of methods: ``parameter_belief``, ``dynamic_state``, ``step``,
    ``step_actions``, ``cost_actions``, ``update_belief`` and ``predict_outcome``.
    This class implements all of them from the exact tables, so those planners run
    **unchanged** with an exact model instead of the learned one.  That is the swap that
    separates search/scoring error from model error on the line that produced the headline
    depth sweep.

    State layout (identical to the learned model): ``[released, 36 object cells, 4 robot
    pose coordinates / 5]``.  The parameter belief is the 16-way distribution over the
    hidden contact types.

    Exactness
    ---------
    ``parameter_belief`` / ``dynamic_state`` are exact given the observation; the first
    step of the lookahead -- the step that selects the action -- operates on the point-mass
    state the scan gives, so its transition, likelihood and cost are exact.  Deeper levels
    mix the object-cell marginal rather than the joint ``(cell, theta)`` distribution, which
    is the planner's *own* approximation and is applied identically to the learned model.

    ``released`` is unobservable (a successful release leaves the scan unchanged), so it is
    threaded forward by ``step`` exactly as the learned pipeline does, not re-read from the
    observation.
    """

    def __init__(
        self,
        *,
        static: StaticMap,
        radius: int = 5,
        condition_on_feedback: bool = True,
        cell_threshold: float = 0.02,
        feedback_threshold: float = 1e-9,
    ) -> None:
        self.static = static
        self.radius = radius
        self.condition_on_feedback = condition_on_feedback
        self.cell_threshold = cell_threshold
        self.feedback_threshold = feedback_threshold
        #: Exact last observation, kept so ``update_belief`` can resolve the state when it
        #: is called with the vector the planner carries.
        self._last_observation: Optional[np.ndarray] = None

    # -- observation decoding ------------------------------------------------

    def _decode(self, observation: np.ndarray) -> Tuple[Cell, Tuple[Cell, Cell]]:
        observation = np.asarray(observation, dtype=np.float64)
        scan = scan_from_observation(
            observation, (0, 0), radius=self.radius, robot_id=0
        )
        own = (
            int(round(float(observation[GRID * GRID * N_CELL_BLOCKS]) * (GRID - 1))),
            int(round(float(observation[GRID * GRID * N_CELL_BLOCKS + 1]) * (GRID - 1))),
        )
        teammate = None
        for cell, kind in scan.observed.items():
            if kind == KIND_ROBOT and cell != own:
                teammate = cell
                break
        if teammate is None:
            teammate = own
        return own, (own, teammate)

    def _origins(self, observation: np.ndarray) -> Dict[Cell, float]:
        observation = np.asarray(observation, dtype=np.float64)
        own, _poses = self._decode(observation)
        scan = scan_from_observation(observation, own, radius=self.radius)
        origins, _seen = object_origins_from_scan(scan, self.static)
        if not origins:
            return {}
        weight = 1.0 / len(origins)
        return {origin: weight for origin in origins}

    # -- belief-model interface ---------------------------------------------

    def parameter_belief(self, observation: np.ndarray) -> np.ndarray:
        """Uniform over the 16 contact types: without a history nothing is identifiable."""

        self._last_observation = np.asarray(observation, dtype=np.float64)
        return np.full((1, len(PHYSICS)), 1.0 / len(PHYSICS))

    def dynamic_state(self, observation: np.ndarray) -> np.ndarray:
        self._last_observation = np.asarray(observation, dtype=np.float64)
        origins = self._origins(observation)
        _own, poses = self._decode(observation)
        state = np.zeros((1, _DYNAMIC_DIM))
        for origin, weight in origins.items():
            state[0, 1 + pose_to_index(origin)] += weight
        state[0, 0] = 0.0  # an episode starts unreleased; ``step`` threads it afterwards
        state[0, 1 + N_CELLS :] = _pose_vector(poses)
        return state

    def step(self, dynamic: np.ndarray, action: int, belief: np.ndarray) -> np.ndarray:
        return self.step_actions(dynamic, belief)[action : action + 1]

    def step_actions(self, dynamic: np.ndarray, belief: np.ndarray) -> np.ndarray:
        dynamic = np.atleast_2d(dynamic)
        belief = np.atleast_2d(belief)
        released_probability = float(np.clip(dynamic[0, 0], 0.0, 1.0))
        origins = _sparse_cells(dynamic[0, 1 : 1 + N_CELLS], self.cell_threshold)
        poses = _decode_poses(dynamic[0, 1 + N_CELLS :])

        out = np.zeros((len(ACTION_IDS), _DYNAMIC_DIM))
        for action, action_id in enumerate(ACTION_IDS):
            joint = _joint_for(None, action_id)
            accumulated = np.zeros(_DYNAMIC_DIM)
            for index in range(len(PHYSICS)):
                weight = float(belief[0, index])
                if weight <= 0.0:
                    continue
                for origin, cell_weight in origins.items():
                    robots = _project_poses(poses, origin, self.static)
                    for released, released_weight in (
                        (False, 1.0 - released_probability),
                        (True, released_probability),
                    ):
                        if released_weight <= 0.0:
                            continue
                        scene = Scene(
                            static=self.static,
                            obj=ObjectState(origin),
                            robots=robots,
                            physics=PHYSICS[index],
                            released=released,
                        )
                        successor, result = execute(scene, joint)
                        mass = weight * cell_weight * released_weight
                        accumulated[0] += mass * float(successor.released)
                        accumulated[1 + pose_to_index(successor.obj.origin)] += mass
                        accumulated[1 + N_CELLS :] += mass * _pose_vector(
                            (successor.robot_pose(0), successor.robot_pose(1))
                        )
            out[action] = accumulated
        cells = out[:, 1 : 1 + N_CELLS]
        totals = cells.sum(axis=1, keepdims=True)
        cells /= np.where(totals > 0.0, totals, 1.0)
        return out

    def cost_actions(self, states: np.ndarray, belief: np.ndarray) -> np.ndarray:
        """Exact action cost.  It is action-determined here, so the state is irrelevant."""

        from ..grid.dynamics import DURATION_PROBE, DURATION_PUSH

        costs = []
        for action_id in ACTION_IDS:
            if action_id == "wait":
                costs.append(0.0)
            elif action_id.startswith("probe"):
                costs.append(DURATION_PROBE)
            else:
                costs.append(DURATION_PUSH)
        return np.asarray(costs)

    def predict_outcome(
        self, dynamic: np.ndarray, action: int, belief: np.ndarray
    ) -> np.ndarray:
        dynamic = np.atleast_2d(dynamic)
        belief = np.atleast_2d(belief)
        released_probability = float(np.clip(dynamic[0, 0], 0.0, 1.0))
        origins = _sparse_cells(dynamic[0, 1 : 1 + N_CELLS], self.cell_threshold)
        poses = _decode_poses(dynamic[0, 1 + N_CELLS :])
        action_id = ACTION_IDS[action]
        joint = _joint_for(None, action_id)
        outcome = np.zeros((1, _N_OBSERVATION_SYMBOLS))
        for index in range(len(PHYSICS)):
            weight = float(belief[0, index])
            if weight <= 0.0:
                continue
            for origin, cell_weight in origins.items():
                robots = _project_poses(poses, origin, self.static)
                for released, released_weight in (
                    (False, 1.0 - released_probability),
                    (True, released_probability),
                ):
                    if released_weight <= 0.0:
                        continue
                    scene = Scene(
                        static=self.static,
                        obj=ObjectState(origin),
                        robots=robots,
                        physics=PHYSICS[index],
                        released=released,
                    )
                    _successor, result = execute(scene, joint)
                    symbol = _FEEDBACK_INDEX[result.feedback[0]]
                    outcome[0, symbol] += weight * cell_weight * released_weight
        total = outcome.sum()
        if total > 0.0:
            outcome /= total
        return outcome

    def update_belief(
        self,
        belief: np.ndarray,
        prior_state: np.ndarray,
        action: int,
        symbol: int,
    ) -> np.ndarray:
        """Exact ``b'(theta) ∝ P(y | a, s_theta', theta) b(theta)``.

        ``prior_state`` is the planner's dynamic vector; the state it encodes is a point
        mass whenever the planner is at the boundary it is choosing for, which is the case
        for every call ``plan_with_search`` makes with a first-step branch.
        """

        belief = np.atleast_2d(belief)
        prior_state = np.atleast_2d(prior_state)
        if symbol >= len(FEEDBACK_SYMBOLS):
            return belief
        expected = FEEDBACK_SYMBOLS[symbol]
        released_probability = float(np.clip(prior_state[0, 0], 0.0, 1.0))
        origins = _sparse_cells(prior_state[0, 1 : 1 + N_CELLS], self.cell_threshold)
        poses = _decode_poses(prior_state[0, 1 + N_CELLS :])
        action_id = ACTION_IDS[action]
        joint = _joint_for(None, action_id)

        posterior = np.zeros_like(belief)
        for index in range(len(PHYSICS)):
            weight = float(belief[0, index])
            if weight <= 0.0:
                continue
            likelihood = 0.0
            for origin, cell_weight in origins.items():
                robots = _project_poses(poses, origin, self.static)
                for released, released_weight in (
                    (False, 1.0 - released_probability),
                    (True, released_probability),
                ):
                    if released_weight <= 0.0:
                        continue
                    scene = Scene(
                        static=self.static,
                        obj=ObjectState(origin),
                        robots=robots,
                        physics=PHYSICS[index],
                        released=released,
                    )
                    _successor, result = execute(scene, joint)
                    if result.feedback[0] == expected:
                        likelihood += cell_weight * released_weight
            posterior[0, index] = weight * likelihood
        total = posterior.sum()
        if total <= 0.0:
            return belief
        return posterior / total


# ---------------------------------------------------------------------------
# Helpers shared by the factored adapter
# ---------------------------------------------------------------------------

try:  # pragma: no cover - import-time only
    from ..learning.belief_model import (
        DYNAMIC_DIM as _DYNAMIC_DIM,
        N_OBSERVATION_SYMBOLS as _N_OBSERVATION_SYMBOLS,
    )
except Exception:  # pragma: no cover
    _DYNAMIC_DIM = 1 + N_CELLS + 4
    _N_OBSERVATION_SYMBOLS = 6

_FEEDBACK_INDEX = {symbol: index for index, symbol in enumerate(FEEDBACK_SYMBOLS)}


def _pose_vector(poses: Tuple[Cell, Cell]) -> np.ndarray:
    return np.asarray(
        [coordinate / 5.0 for pose in poses for coordinate in pose], dtype=np.float64
    )


def _decode_poses(values: np.ndarray) -> Tuple[RobotState, RobotState]:
    coordinates = [int(round(float(value) * 5.0)) for value in np.asarray(values).ravel()[:4]]
    first = RobotState(0, (coordinates[0], coordinates[1]))
    second = RobotState(1, (coordinates[2], coordinates[3]))
    return (first, second)


def _sparse_cells(distribution: np.ndarray, threshold: float) -> Dict[Cell, float]:
    values = np.asarray(distribution, dtype=np.float64)
    keep = {index: float(value) for index, value in enumerate(values) if value > threshold}
    if not keep:
        index = int(np.argmax(values))
        return {pose_index_to_origin(index): 1.0}
    total = sum(keep.values())
    return {pose_index_to_origin(index): value / total for index, value in keep.items()}


def _project_poses(
    poses: Tuple[RobotState, RobotState], origin: Cell, static: StaticMap
) -> Tuple[RobotState, RobotState]:
    """Snap decoded point-estimate poses onto the legal state space.

    The dynamic vector stores robot positions as an average over mixture components, so the
    rounded cell can land on a wall or inside the object.  The planner's own state interface
    only admits legal poses, so project instead of raising.
    """

    object_cells = ObjectState(origin).cells
    used: set = set()
    projected: List[RobotState] = []
    for robot in poses:
        cell = robot.pose
        if (
            not static.in_bounds(cell)
            or static.is_wall(cell)
            or cell in object_cells
            or cell in used
        ):
            best = None
            best_distance = None
            for candidate in static.cells():
                if static.is_wall(candidate) or candidate in object_cells or candidate in used:
                    continue
                distance = abs(candidate[0] - cell[0]) + abs(candidate[1] - cell[1])
                if best_distance is None or distance < best_distance:
                    best, best_distance = candidate, distance
            if best is not None:
                cell = best
        used.add(cell)
        projected.append(RobotState(robot.robot_id, cell))
    return (projected[0], projected[1])


def pose_index_to_origin(index: int) -> Cell:
    return (index % GRID, index // GRID)


class HybridFactoredModel:
    """Two factored models glued at the interface, for the module-replacement experiment.

    Both the learned :class:`~coimagine.learning.belief_model.BeliefWorldModel` and
    :class:`ExactFactoredModel` consume and produce the *same* state layout (released,
    36 object cells, 4 robot coordinates) and the same 16-way parameter belief, so one
    module can be swapped without touching the planner.  The transition side supplies
    ``dynamic_state``, ``step``, ``step_actions`` and ``cost_actions``; the observation
    side supplies ``parameter_belief``, ``predict_outcome`` and ``update_belief``.

    Mixing requires identical state semantics.  Here they hold: both sides read and write
    the declared dynamic vector and the declared parameter index.  Where they did not, the
    replacement would not be interpretable and would be reported as such instead.
    """

    def __init__(self, transition, observation) -> None:
        self._transition = transition
        self._observation = observation
        self.transition_source = type(transition).__name__
        self.observation_source = type(observation).__name__

    def parameter_belief(self, observation):
        return self._observation.parameter_belief(observation)

    def dynamic_state(self, observation):
        return self._transition.dynamic_state(observation)

    def step(self, dynamic, action, belief):
        return self._transition.step(dynamic, action, belief)

    def step_actions(self, dynamic, belief):
        return self._transition.step_actions(dynamic, belief)

    def cost_actions(self, states, belief):
        return self._transition.cost_actions(states, belief)

    def predict_outcome(self, dynamic, action, belief):
        return self._observation.predict_outcome(dynamic, action, belief)

    def update_belief(self, belief, prior_state, action, symbol):
        return self._observation.update_belief(belief, prior_state, action, symbol)
