"""Fixed trajectory score shared by the exact environment and the learned model.

See CONTRACT section 3. The exact environment and the learned model both run through
:func:`rollout` and :func:`score_trajectory`; the learned side supplies a step function that
may return an invalid prediction, which is recorded explicitly rather than repaired.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from .contract import GridMap, GridState, is_goal, terminal_distance
from .dynamics import transition_exact

StepFn = Callable[[GridState, int], Tuple[Optional[GridState], bool]]


@dataclass(frozen=True)
class TrajectoryResult:
    success: bool
    executed_steps: int
    terminal_distance: float
    invalid_prediction: int
    states: Tuple[GridState, ...]
    actions: Tuple[int, ...]


@dataclass(frozen=True)
class ScoreComponents:
    success: int
    executed_steps: int
    terminal_distance: float
    invalid_prediction: int
    J: float

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "executed_steps": self.executed_steps,
            "terminal_distance": self.terminal_distance,
            "invalid_prediction": self.invalid_prediction,
            "J": self.J,
        }


def exact_step(grid_map: GridMap) -> StepFn:
    def _step(state: GridState, joint_action: int) -> Tuple[Optional[GridState], bool]:
        next_state, _info = transition_exact(state, joint_action, grid_map)
        return next_state, True

    return _step


def rollout(
    initial: GridState,
    sequence: Sequence[int],
    grid_map: GridMap,
    step_fn: StepFn,
) -> TrajectoryResult:
    """Execute a fixed joint-action sequence; no re-selection happens mid-sequence."""

    if is_goal(initial, grid_map):
        # Only reachable in unit tests / constructed states; success is immediate, 0 steps.
        return TrajectoryResult(True, 0, 0.0, 0, (initial,), ())
    state = initial
    states = [initial]
    actions = []
    executed = 0
    invalid = 0
    success = False
    for joint_action in sequence:
        next_state, valid = step_fn(state, joint_action)
        executed += 1
        actions.append(joint_action)
        if not valid or next_state is None:
            invalid = 1
            break
        state = next_state
        states.append(state)
        if is_goal(state, grid_map):
            success = True
            break
    distance = 0.0 if success else terminal_distance(state, grid_map)
    return TrajectoryResult(success, executed, distance, invalid, tuple(states), tuple(actions))


def rollout_exact(initial: GridState, sequence: Sequence[int], grid_map: GridMap) -> TrajectoryResult:
    return rollout(initial, sequence, grid_map, exact_step(grid_map))


def score_trajectory(result: TrajectoryResult) -> ScoreComponents:
    """The one scoring entry used by both the exact environment and the learned model."""

    success = int(result.success)
    J = (
        10.0 * success
        - float(result.executed_steps)
        - 0.5 * float(result.terminal_distance)
        - 10.0 * float(result.invalid_prediction)
    )
    return ScoreComponents(
        success=success,
        executed_steps=int(result.executed_steps),
        terminal_distance=float(result.terminal_distance),
        invalid_prediction=int(result.invalid_prediction),
        J=float(J),
    )
