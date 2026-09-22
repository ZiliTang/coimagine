"""Stage 2 dynamics: joint skills, contact geometry, cost and local feedback.

A boundary is one **joint skill**: each robot picks one local skill, both execute
synchronously, and robots that finish early hold. The boundary cost is the longest
local duration (a reposition is charged its full collision-free path length; there
is no free teleport).

Local skills
------------
``PUSH(d)``
    Push the object one cell in direction ``d``. Legal only from a contact cell on
    the opposite face. A single robot has capacity 1; two robots pushing the same
    direction have capacity 2. Per-robot traction can remove a pusher's capacity.
``CO_PUSH`` is therefore expressed as both robots choosing ``PUSH(d)``.
``RELEASE(d)``
    Lateral nudge from an adjacent cell. It does not move the object; it performs
    the explicit contact-mode transition ``constrained -> free``. It is a real
    physical action with a cost, not an information bonus.
``PROBE(d)``
    A deliberately short push attempt: no displacement, but it produces load/slip
    feedback. This is the identification action.
``HOLD`` / ``WAIT``
    No displacement, zero cost. ``HOLD`` keeps a robot at the contact; ``WAIT`` is
    an explicit idle.

Hidden conditions decide the outcome: resistance, the constrained mode, and each
robot's traction. Everything is read from the published transition; there is no
information-gain or enabling reward anywhere.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .state import (
    Cell,
    Direction,
    HiddenPhysics,
    ObjectState,
    RobotState,
    Scene,
    add,
    manhattan,
)

# -- local skill kinds -----------------------------------------------------

SKILL_PUSH = "push"
SKILL_RELEASE = "release"
SKILL_PROBE = "probe"
SKILL_REPOSITION = "reposition"
SKILL_HOLD = "hold"
SKILL_WAIT = "wait"

LOCAL_SKILLS = (
    SKILL_PUSH,
    SKILL_RELEASE,
    SKILL_PROBE,
    SKILL_REPOSITION,
    SKILL_HOLD,
    SKILL_WAIT,
)

# -- local feedback symbols ------------------------------------------------

OBS_DROVE = "drove"
OBS_STRAINED = "strained"
OBS_SLIPPED = "slipped"
OBS_IDLE = "idle"

FEEDBACK_SYMBOLS = (OBS_DROVE, OBS_STRAINED, OBS_SLIPPED, OBS_IDLE)

# -- public event ----------------------------------------------------------

Y_MOVED = "moved"
Y_STALLED = "stalled"

# -- durations -------------------------------------------------------------

DURATION_PUSH = 1.0
DURATION_RELEASE = 1.0
DURATION_HOLD = 0.0
DURATION_WAIT = 0.0
#: Default cost of the short diagnostic push.
DURATION_PROBE = 0.3

# -- failure reason codes --------------------------------------------------

FAIL_NOT_AT_CONTACT = "not_at_contact"
FAIL_MIXED_DIRECTIONS = "mixed_push_directions"
FAIL_NO_GRIP = "no_effective_grip"
FAIL_INSUFFICIENT_CAPACITY = "insufficient_capacity"
FAIL_CONSTRAINED = "constrained_contact"
FAIL_BLOCKED = "destination_blocked"
FAIL_NO_EFFECT = "no_effect"


@dataclass(frozen=True)
class LocalSkill:
    kind: str
    direction: Optional[Direction] = None
    target: Optional[Cell] = None

    def __post_init__(self) -> None:
        if self.kind not in LOCAL_SKILLS:
            raise ValueError(f"Unknown local skill kind {self.kind!r}.")
        if self.kind in (SKILL_PUSH, SKILL_RELEASE, SKILL_PROBE) and self.direction is None:
            raise ValueError(f"{self.kind} requires a direction.")
        if self.kind == SKILL_REPOSITION and self.target is None:
            raise ValueError("reposition requires a target cell.")

    @property
    def label(self) -> str:
        if self.kind in (SKILL_PUSH, SKILL_RELEASE, SKILL_PROBE):
            return f"{self.kind}:{self.direction.label}"  # type: ignore[union-attr]
        if self.kind == SKILL_REPOSITION:
            return f"reposition:{self.target}"
        return self.kind


@dataclass(frozen=True)
class JointSkill:
    skills: Tuple[LocalSkill, LocalSkill]

    def __post_init__(self) -> None:
        if len(self.skills) != 2:
            raise ValueError("A joint skill needs exactly one local skill per robot.")

    def for_robot(self, robot_id: int) -> LocalSkill:
        return self.skills[robot_id]

    @property
    def label(self) -> str:
        return " | ".join(
            f"{'AB'[i]}:{skill.label}" for i, skill in enumerate(self.skills)
        )


@dataclass(frozen=True)
class FeedbackNoise:
    """Optional noise on the local load/slip feedback.

    ``flip_prob`` replaces the nominal symbol by a uniformly random different one.
    The belief filter uses the same channel, so the generative model and the filter
    can never drift apart.
    """

    flip_prob: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.flip_prob < 1.0:
            raise ValueError("flip_prob must lie in [0, 1)")


NO_FEEDBACK_NOISE = FeedbackNoise()


@dataclass(frozen=True)
class StepResult:
    cost: float
    y_public: str
    feedback: Mapping[int, str]
    success: bool
    released: bool
    failures: Tuple[str, ...] = ()
    detail: Mapping[str, object] = field(default_factory=dict)

    @property
    def object_moved(self) -> bool:
        return self.y_public == Y_MOVED


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def shortest_path(
    scene: Scene,
    start: Cell,
    goal: Cell,
    *,
    blocked: Sequence[Cell] = (),
) -> Optional[Tuple[Cell, ...]]:
    """Four-connected collision-free path inside the prior free space."""

    static = scene.static
    blocked_set = frozenset(blocked)
    if not static.in_bounds(goal) or static.is_wall(goal) or goal in blocked_set:
        return None
    if start == goal:
        return (start,)

    queue = deque([start])
    parent: Dict[Cell, Optional[Cell]] = {start: None}
    while queue:
        current = queue.popleft()
        for direction in (Direction.NORTH, Direction.EAST, Direction.SOUTH, Direction.WEST):
            nxt = add(current, direction.delta)
            if nxt in parent:
                continue
            if not static.in_bounds(nxt) or static.is_wall(nxt) or nxt in blocked_set:
                continue
            parent[nxt] = current
            if nxt == goal:
                path = [goal]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])  # type: ignore[arg-type]
                return tuple(reversed(path))
            queue.append(nxt)
    return None


def push_contact_ok(scene: Scene, robot_id: int, direction: Direction) -> bool:
    """The robot sits on the face opposite to ``direction``."""

    pose = scene.robot_pose(robot_id)
    return add(pose, direction.delta) in scene.obj.cells


def release_contact_ok(scene: Scene, robot_id: int) -> bool:
    """The robot is adjacent to the object (any face)."""

    pose = scene.robot_pose(robot_id)
    return any(manhattan(pose, cell) == 1 for cell in scene.obj.cells)


def _destination_clear(
    scene: Scene,
    direction: Direction,
    *,
    extra_blocked: Sequence[Cell] = (),
) -> bool:
    static = scene.static
    moved = scene.obj.translated(direction.delta)
    extra = frozenset(extra_blocked)
    for cell in moved.cells:
        if not static.in_bounds(cell) or static.is_wall(cell):
            return False
        if cell in extra:
            return False
    for robot in scene.robots:
        if robot.pose in moved.cells:
            return False
    return True


# --------------------------------------------------------------------------
# Execution
# --------------------------------------------------------------------------


def _nominal_feedback(
    physics: HiddenPhysics,
    robot_id: int,
    skill: LocalSkill,
    *,
    object_moved: bool,
    probe_would_succeed: bool,
    release_succeeded: bool,
) -> str:
    if skill.kind in (SKILL_HOLD, SKILL_WAIT, SKILL_REPOSITION):
        return OBS_IDLE
    if physics.traction[robot_id] == "slip":
        return OBS_SLIPPED
    if skill.kind == SKILL_RELEASE:
        return OBS_DROVE if release_succeeded else OBS_STRAINED
    if skill.kind == SKILL_PROBE:
        return OBS_DROVE if probe_would_succeed else OBS_STRAINED
    # A real push: drove exactly when the object actually moved.
    return OBS_DROVE if object_moved else OBS_STRAINED


def sample_feedback(
    nominal: str, noise: FeedbackNoise, rng: random.Random
) -> str:
    if noise.flip_prob <= 0.0:
        return nominal
    if rng.random() >= noise.flip_prob:
        return nominal
    alternatives = [s for s in FEEDBACK_SYMBOLS if s != nominal]
    return rng.choice(alternatives)


def feedback_likelihood(observed: str, nominal: str, noise: FeedbackNoise) -> float:
    """``P(observed | nominal)`` for the feedback channel."""

    if noise.flip_prob <= 0.0:
        return 1.0 if observed == nominal else 0.0
    if observed == nominal:
        return 1.0 - noise.flip_prob
    return noise.flip_prob / (len(FEEDBACK_SYMBOLS) - 1)


def execute(
    scene: Scene,
    joint: JointSkill,
    *,
    probe_cost: float = DURATION_PROBE,
    noise: FeedbackNoise = NO_FEEDBACK_NOISE,
    rng: Optional[random.Random] = None,
) -> Tuple[Scene, StepResult]:
    """Execute one joint skill and return the successor scene and the result."""

    rng = rng or random.Random(0)
    physics = scene.physics
    failures: List[str] = []

    # -- 1. repositioning: paths are planned before anything moves ----------
    robot_moves: Dict[int, Tuple[Cell, ...]] = {}
    durations: Dict[int, float] = {}
    moving_cells: List[Cell] = []
    for robot in scene.robots:
        skill = joint.for_robot(robot.robot_id)
        if skill.kind == SKILL_REPOSITION:
            blocked = [
                cell
                for other in scene.robots
                if other.robot_id != robot.robot_id
                for cell in (other.pose,)
            ] + list(scene.obj.cells)
            path = shortest_path(scene, robot.pose, skill.target, blocked=blocked)
            if path is None:
                failures.append(f"robot{robot.robot_id}:{FAIL_BLOCKED}")
                durations[robot.robot_id] = 0.0
                continue
            robot_moves[robot.robot_id] = path
            moving_cells.extend(path[1:])
            durations[robot.robot_id] = float(len(path) - 1)
        elif skill.kind == SKILL_PUSH:
            durations[robot.robot_id] = DURATION_PUSH
        elif skill.kind == SKILL_PROBE:
            durations[robot.robot_id] = probe_cost
        elif skill.kind == SKILL_RELEASE:
            durations[robot.robot_id] = DURATION_RELEASE
        else:
            durations[robot.robot_id] = DURATION_HOLD

    # -- 2. pushes and releases --------------------------------------------
    pushers = [
        robot.robot_id
        for robot in scene.robots
        if joint.for_robot(robot.robot_id).kind == SKILL_PUSH
    ]
    push_directions = {
        joint.for_robot(r).direction for r in pushers
    }
    object_moved = False
    if pushers:
        if len(push_directions) != 1:
            failures.append(FAIL_MIXED_DIRECTIONS)
        else:
            direction = next(iter(push_directions))
            assert direction is not None
            contacts_ok = all(push_contact_ok(scene, r, direction) for r in pushers)
            if not contacts_ok:
                failures.append(FAIL_NOT_AT_CONTACT)
            capacity = physics.effective_capacity(tuple(pushers))
            if capacity == 0:
                failures.append(FAIL_NO_GRIP)
            elif capacity < physics.resistance:
                failures.append(FAIL_INSUFFICIENT_CAPACITY)
            if physics.constraint == "constrained" and not scene.released:
                failures.append(FAIL_CONSTRAINED)
            if not failures and _destination_clear(
                scene, direction, extra_blocked=moving_cells
            ):
                object_moved = True

    release_succeeded = False
    for robot in scene.robots:
        skill = joint.for_robot(robot.robot_id)
        if skill.kind != SKILL_RELEASE:
            continue
        if not release_contact_ok(scene, robot.robot_id):
            failures.append(f"robot{robot.robot_id}:{FAIL_NOT_AT_CONTACT}")
            continue
        if physics.constraint == "constrained" and not scene.released:
            if physics.traction[robot.robot_id] == "slip":
                failures.append(f"robot{robot.robot_id}:{FAIL_NO_GRIP}")
            else:
                release_succeeded = True
        else:
            failures.append(f"robot{robot.robot_id}:{FAIL_NO_EFFECT}")

    new_released = scene.released or release_succeeded

    # -- 3. new object and robot poses --------------------------------------
    if object_moved:
        direction = next(iter(push_directions))
        assert direction is not None
        new_obj = scene.obj.translated(direction.delta)
    else:
        direction = None
        new_obj = scene.obj

    new_robots: List[RobotState] = []
    push_delta = direction.delta if (object_moved and pushers) else None
    for robot in scene.robots:
        if robot.robot_id in robot_moves:
            path = robot_moves[robot.robot_id]
            heading = robot.heading
            if len(path) >= 2:
                delta = (path[-1][0] - path[-2][0], path[-1][1] - path[-2][1])
                for candidate in Direction:
                    if candidate.delta == delta:
                        heading = candidate
            new_robots.append(RobotState(robot.robot_id, path[-1], heading))
        else:
            skill = joint.for_robot(robot.robot_id)
            heading = robot.heading
            if skill.kind in (SKILL_PUSH, SKILL_RELEASE, SKILL_PROBE) and skill.direction:
                heading = skill.direction
            pose = robot.pose
            # A pusher walks forward with the object, so a multi-step push does not
            # need a reposition between every cell. It advances only into a legal
            # cell; otherwise it stays put.
            if push_delta is not None and skill.kind == SKILL_PUSH:
                candidate_cell = add(pose, push_delta)
                legal = (
                    scene.static.in_bounds(candidate_cell)
                    and not scene.static.is_wall(candidate_cell)
                    and candidate_cell not in new_obj.cells
                    and all(
                        candidate_cell != other.pose
                        for other in scene.robots
                        if other.robot_id != robot.robot_id
                    )
                )
                if legal:
                    pose = candidate_cell
            new_robots.append(RobotState(robot.robot_id, pose, heading))

    # -- 4. feedback --------------------------------------------------------
    def probe_would_succeed(robot_id: int, direction: Optional[Direction]) -> bool:
        """Would a *full* push by this robot alone move the object?

        Used only to define the nominal short-push feedback. It reads the true
        transition, exactly like every other outcome.
        """

        if direction is None or physics.traction[robot_id] == "slip":
            return False
        if not push_contact_ok(scene, robot_id, direction):
            return False
        if physics.effective_capacity((robot_id,)) < physics.resistance:
            return False
        if physics.constraint == "constrained" and not scene.released:
            return False
        return _destination_clear(scene, direction, extra_blocked=moving_cells)

    nominal: Dict[int, str] = {}
    feedback: Dict[int, str] = {}
    for robot in scene.robots:
        skill = joint.for_robot(robot.robot_id)
        nominal[robot.robot_id] = _nominal_feedback(
            physics,
            robot.robot_id,
            skill,
            object_moved=object_moved,
            probe_would_succeed=probe_would_succeed(robot.robot_id, skill.direction),
            release_succeeded=release_succeeded,
        )
        feedback[robot.robot_id] = sample_feedback(
            nominal[robot.robot_id], noise, rng
        )

    successor = Scene(
        static=scene.static,
        obj=new_obj,
        robots=tuple(new_robots),
        physics=physics,
        released=new_released,
    )
    result = StepResult(
        cost=max(durations.values()) if durations else 0.0,
        y_public=Y_MOVED if object_moved else Y_STALLED,
        feedback=feedback,
        success=object_moved or release_succeeded,
        released=new_released,
        failures=tuple(failures),
        detail={
            "pushers": tuple(pushers),
            "capacity": physics.effective_capacity(tuple(pushers)) if pushers else 0,
            "object_moved": object_moved,
            "release_succeeded": release_succeeded,
            "nominal_feedback": dict(nominal),
        },
    )
    return successor, result
