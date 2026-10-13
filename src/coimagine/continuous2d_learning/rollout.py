"""Fixed-sequence evaluation with the new scalar score (contract C1).

Two evaluators that call the **same** :func:`contract.score_components`:

* :func:`evaluate_sequence_exact` runs the true simulator on an independent planner env
  restored from the snapshot with exact prefix replay;
* :func:`evaluate_sequence_learned` rolls the learned model recursively from the same
  starting boundary state and never queries the simulator.

Both absorb success and failure identically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.backend import BodyState, Snapshot
from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import (
    TERMINATION_NUMERIC,
    TERMINATION_OUT_OF_BOUNDS,
    TERMINATION_PENETRATION,
    TERMINATION_SUCCESS,
    TERMINATION_WINDOW,
    Continuous2DEnv,
)
from ..continuous2d.scenes import SceneSpec
from ..continuous2d.skills import JointSpec
from .contract import (
    ScoreComponents,
    geometric_violation,
    passage_distance,
    score_components,
    state_corners,
)
from .features import BoundaryState
from .model import JointWindowMLP, predict_window

EXIT_SUCCESS = "success"
EXIT_PHYSICAL_FAILURE = "physical_failure"
EXIT_INVALID = "invalid_prediction"
EXIT_WINDOW_COMPLETE = "window_complete"
EXIT_NON_FINITE = "non_finite_prediction"


@dataclass
class SequenceEvaluation:
    candidate_id: int
    j: float
    components: ScoreComponents
    exit_class: str
    windows_executed: int
    terminal_state: Optional[BoundaryState]
    predicted_windows: List[Dict[str, object]] = field(default_factory=list)
    trajectory: List[Dict[str, object]] = field(default_factory=list)
    physics_steps: int = 0
    prefix_replay_steps: int = 0
    prediction_nodes: int = 0

    @property
    def success(self) -> bool:
        return self.exit_class == EXIT_SUCCESS

    @property
    def finite(self) -> bool:
        return self.exit_class != EXIT_NON_FINITE

    @property
    def feasible(self) -> bool:
        """A3: only ``window_complete`` and ``success`` may be executed in closed loop."""

        return self.exit_class in (EXIT_WINDOW_COMPLETE, EXIT_SUCCESS)


def _absorbed_boundary(state: BoundaryState, config: Continuous2DConfig) -> Optional[str]:
    """Absorbed task-level outcome of an already-terminal boundary state, if any."""

    if state.success_counter >= config.physics.success_hold_time:
        return EXIT_SUCCESS
    return None


def _absorbed_start(snapshot, config: Continuous2DConfig) -> Optional[str]:
    """Absorbed outcome of an already-terminal snapshot, from its termination or hold timer."""

    if snapshot.termination == TERMINATION_SUCCESS:
        return EXIT_SUCCESS
    if snapshot.termination in (
        TERMINATION_NUMERIC,
        TERMINATION_OUT_OF_BOUNDS,
        TERMINATION_PENETRATION,
    ):
        return EXIT_PHYSICAL_FAILURE
    if float(snapshot.success_counter) >= config.physics.success_hold_time:
        return EXIT_SUCCESS
    return None


def state_dict(state: Optional[BoundaryState]) -> Optional[Dict[str, object]]:
    if state is None:
        return None
    return state.as_dict()


def boundary_from_dict(payload: Dict[str, object]) -> BoundaryState:
    return BoundaryState(
        object=body_from_list(payload["object"]),
        robots=(body_from_list(payload["robots"][0]), body_from_list(payload["robots"][1])),  # type: ignore[index]
        success_counter=float(payload.get("success_counter", 0.0)),
    )


def body_from_list(values) -> BodyState:
    return BodyState(*[float(value) for value in values])




def _distance_from(state: BoundaryState, scene: SceneSpec, config: Continuous2DConfig) -> float:
    physics = config.physics
    corners = state_corners(state, (physics.object_width, physics.object_height))
    return passage_distance(
        (state.object.x, state.object.y, state.object.angle),
        (physics.object_width, physics.object_height),
        scene.goal,
        scene.obstacles,
        corners,
    )


def distance_of_state(state: BoundaryState, scene: SceneSpec, config: Continuous2DConfig) -> float:
    """Public alias of the terminal-distance rule used by the exact and learned sides."""

    return _distance_from(state, scene, config)


def _finish(
    candidate_id: int,
    *,
    success: bool,
    time: float,
    distance: float,
    failure: bool,
    exit_class: str,
    terminal_state: Optional[BoundaryState],
    scene: SceneSpec,
    predicted_windows: List[Dict[str, object]] | None = None,
    trajectory: List[Dict[str, object]] | None = None,
    physics_steps: int = 0,
    prefix_replay_steps: int = 0,
    prediction_nodes: int = 0,
) -> SequenceEvaluation:
    components = score_components(
        success=success,
        time=time,
        distance=distance,
        failure=failure,
        diagonal=(scene.arena[0] ** 2 + scene.arena[1] ** 2) ** 0.5,
    )
    return SequenceEvaluation(
        candidate_id=candidate_id,
        j=components.j,
        components=components,
        exit_class=exit_class,
        windows_executed=len(predicted_windows or []),
        terminal_state=terminal_state,
        predicted_windows=predicted_windows or [],
        trajectory=trajectory or [],
        physics_steps=physics_steps,
        prefix_replay_steps=prefix_replay_steps,
        prediction_nodes=prediction_nodes,
    )


def evaluate_sequence_exact(
    planner: Continuous2DEnv,
    snapshot: Snapshot,
    scene: SceneSpec,
    sequence: Sequence[JointSpec],
    config: Continuous2DConfig,
    *,
    candidate_id: int,
) -> SequenceEvaluation:
    """Execute one fixed sequence from the true snapshot with exact prefix replay."""

    substeps = config.physics.substeps
    prefix_steps = len(snapshot.prefix) * substeps
    planner.restore(snapshot, exact=True)
    time_used = 0.0
    physics_steps = 0
    windows: List[Dict[str, object]] = []
    trajectory: List[Dict[str, object]] = []
    terminal: Optional[BoundaryState] = None
    exit_class = EXIT_WINDOW_COMPLETE
    success = False
    failure = False

    # A3 absorption: a snapshot that is already terminal is absorbed, and no window is run.
    absorbed = _absorbed_start(snapshot, config)
    if absorbed is not None:
        final = planner.export_state()
        terminal = BoundaryState(
            object=final.object, robots=final.robots, success_counter=final.success_counter
        )
        success = absorbed == "success"
        failure = absorbed == "physical_failure"
        distance = 0.0 if success else _distance_from(terminal, scene, config)
        return _finish(
            candidate_id,
            success=success,
            time=0.0,
            distance=distance,
            failure=failure,
            exit_class=EXIT_SUCCESS if success else EXIT_PHYSICAL_FAILURE,
            terminal_state=terminal,
            scene=scene,
            predicted_windows=windows,
            trajectory=trajectory,
            physics_steps=0,
            prefix_replay_steps=prefix_steps,
        )

    for index, joint_spec in enumerate(sequence):
        if planner.termination is not None:
            break
        before_snapshot = planner.export_state()
        result = planner.execute_joint_skill(joint_spec)
        time_used += float(result.cost_components["time"])
        physics_steps += len(result.steps) * substeps
        windows.append(
            {
                "window": index,
                "duration": float(result.duration),
                "termination": result.termination,
                "termination_class": (
                    "success"
                    if result.termination == TERMINATION_SUCCESS
                    else "physical_failure"
                    if result.termination in (TERMINATION_OUT_OF_BOUNDS, TERMINATION_PENETRATION)
                    else "window_complete"
                ),
            }
        )
        trajectory.append(
            {
                "window": index,
                "before": {
                    "object": list(before_snapshot.object.as_tuple()),
                    "robots": [list(state.as_tuple()) for state in before_snapshot.robots],
                },
                "after": {
                    "object": list(result.final.object.as_tuple()),
                    "robots": [list(state.as_tuple()) for state in result.final.robots],
                },
                "duration": float(result.duration),
                "termination": result.termination,
            }
        )
        if result.success:
            success = True
            exit_class = EXIT_SUCCESS
            break
        if result.termination != TERMINATION_WINDOW:
            failure = True
            exit_class = EXIT_PHYSICAL_FAILURE
            break

    final = planner.export_state()
    terminal = BoundaryState(
        object=final.object, robots=final.robots, success_counter=final.success_counter
    )
    if success:
        distance = 0.0
    else:
        distance = _distance_from(terminal, scene, config)
    return _finish(
        candidate_id,
        success=success,
        time=time_used,
        distance=distance,
        failure=failure,
        exit_class=exit_class,
        terminal_state=terminal,
        scene=scene,
        predicted_windows=windows,
        trajectory=trajectory,
        physics_steps=physics_steps,
        prefix_replay_steps=prefix_steps,
    )


def evaluate_sequence_learned(
    model: JointWindowMLP,
    scene: SceneSpec,
    start: BoundaryState,
    sequence: Sequence[JointSpec],
    config: Continuous2DConfig,
    *,
    candidate_id: int,
) -> SequenceEvaluation:
    """Recursively roll the learned model over one fixed sequence.

    The model never sees the true successor. An invalid prediction stops the chain and the
    raw (violating) predicted state supplies the terminal distance, so hallucinated progress
    is measured instead of hidden.
    """

    physics = config.physics
    state = start
    time_used = 0.0
    success = False
    failure = False
    invalid = False
    non_finite = False
    terminal: Optional[BoundaryState] = None
    windows: List[Dict[str, object]] = []
    trajectory: List[Dict[str, object]] = []
    nodes = 0

    # A3 absorption: an already-successful or already-failed start is absorbed without
    # calling the model at all.
    absorbed = _absorbed_boundary(start, config)
    if absorbed is not None:
        success = absorbed == "success"
        failure = absorbed == "physical_failure"
        terminal = start
        return _finish(
            candidate_id,
            success=success,
            time=0.0,
            distance=0.0 if success else _distance_from(start, scene, config),
            failure=failure,
            exit_class=EXIT_SUCCESS if success else EXIT_PHYSICAL_FAILURE,
            terminal_state=terminal,
            scene=scene,
            predicted_windows=windows,
            trajectory=trajectory,
            prediction_nodes=0,
        )

    for index, joint_spec in enumerate(sequence):
        prediction = predict_window(model, scene, state, joint_spec, physics)
        nodes += 1
        time_used += prediction.duration
        violation = geometric_violation(
            prediction.next_state.object, prediction.next_state.robots, scene, physics
        )
        windows.append(
            {
                "window": index,
                "duration": prediction.duration,
                "termination": prediction.termination,
                "success_hold": prediction.success_hold,
                "termination_logits": list(prediction.termination_logits),
                "penetration": violation["penetration"],
                "violation": violation["violation"],
                "reasons": list(violation["reasons"]),
            }
        )
        trajectory.append(
            {
                "window": index,
                "before": state.as_dict(),
                "after": prediction.next_state.as_dict(),
                "duration": prediction.duration,
                "termination": prediction.termination,
                "violation": bool(violation["violation"]),
            }
        )
        terminal = prediction.next_state
        # A3 priority: geometry first. A predicted success may not bypass the legality check.
        if not violation["finite"]:
            non_finite = True
            terminal = state
            break
        if violation["violation"]:
            invalid = True
            break
        if prediction.termination == "success":
            success = True
            break
        if prediction.termination == "physical_failure":
            failure = True
            break
        state = prediction.next_state
        terminal = state

    if success:
        distance = 0.0
        exit_class = EXIT_SUCCESS
    elif non_finite:
        distance = _distance_from(terminal or state, scene, config)
        exit_class = EXIT_NON_FINITE
    elif invalid:
        distance = _distance_from(terminal or state, scene, config)
        exit_class = EXIT_INVALID
    elif failure:
        distance = _distance_from(terminal or state, scene, config)
        exit_class = EXIT_PHYSICAL_FAILURE
    else:
        distance = _distance_from(terminal or state, scene, config)
        exit_class = EXIT_WINDOW_COMPLETE

    return _finish(
        candidate_id,
        success=success,
        time=time_used,
        distance=distance,
        failure=bool(invalid or failure or non_finite),
        exit_class=exit_class,
        terminal_state=terminal,
        scene=scene,
        predicted_windows=windows,
        trajectory=trajectory,
        prediction_nodes=nodes,
    )
