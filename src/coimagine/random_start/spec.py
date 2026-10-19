"""Frozen numbers and the action table of the random-start round.

Every value here is either inherited from the frozen ``continuous2d`` configuration or is a
first-version engineering setting of this round. ``SPEC.md`` copies this table; nothing in the
skill or environment code may use a hard-coded literal that is not declared here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

from ..continuous2d.config import Continuous2DConfig, default_config

# ---------------------------------------------------------------------------
# inherited physics (SI). Read from the frozen config so a change there cannot
# silently diverge from this module.
# ---------------------------------------------------------------------------

_PHYSICS = default_config().physics
CONTROL_DT: float = _PHYSICS.control_dt                       # s, 0.05
PHYSICS_DT: float = _PHYSICS.physics_dt                       # s, 1/120
SUBSTEPS: int = _PHYSICS.substeps                             # physics steps per control period
ROBOT_RADIUS: float = _PHYSICS.robot_radius                   # m, 0.18
ROBOT_MASS: float = _PHYSICS.robot_mass                       # kg, 4.0
OBJECT_WIDTH: float = _PHYSICS.object_width                   # m, 1.0
OBJECT_HEIGHT: float = _PHYSICS.object_height                 # m, 0.25
OBJECT_MASS: float = _PHYSICS.object_mass                     # kg, 3.0
MAX_FORCE: float = _PHYSICS.max_force                         # N, 15.0 (both robots, each period)
FRICTION: float = _PHYSICS.friction
ELASTICITY: float = _PHYSICS.elasticity
LINEAR_DAMPING: float = _PHYSICS.linear_damping
ANGULAR_DAMPING: float = _PHYSICS.angular_damping
MAX_ACCEL: float = MAX_FORCE / ROBOT_MASS                     # m/s^2, 3.75 (no torque channel)
BOUNDARY_RADIUS: float = 0.05                                 # m, arena wall capsule radius

CONFIG_HASH: str = default_config().config_hash()

# ---------------------------------------------------------------------------
# arena and maps
# ---------------------------------------------------------------------------

ARENA: Tuple[float, float] = (3.2, 2.2)                       # m, identical for both maps
#: L1: one axis-aligned rectangular obstacle, centre (1.60, 1.10) m. The published size is the
#: **full** size; the engine's ``Poly.create_box`` also takes the full size, and
#: ``scenes.Box`` stores half sizes, so the two halves are 0.10 x 0.30 here.
L1_CENTRE: Tuple[float, float] = (1.60, 1.10)
L1_FULL_WIDTH: float = 0.20
L1_FULL_HEIGHT: float = 0.60
MAP_NAMES: Tuple[str, ...] = ("L0", "L1")

# ---------------------------------------------------------------------------
# sampling contract (task book section 2.2)
# ---------------------------------------------------------------------------

MAX_SAMPLE_ATTEMPTS: int = 2000        # candidate draws per instance, hard cap
ROBOT_DRAWS_PER_STAGE: int = 40        # candidate draws per robot inside one outer attempt
OBJECT_EDGE_MARGIN: float = 0.10       # m, object to boundary/obstacle initial clearance
ROBOT_OBJECT_GAP_MIN: float = 0.12     # m, robot outer edge to object boundary
ROBOT_OBJECT_GAP_MAX: float = 0.50     # m
ROBOT_CLEARANCE: float = 0.05          # m, robot to obstacle / boundary / other robot
GOAL_RADIUS: float = 0.20              # m
GOAL_DISTANCE_MIN: float = 0.80        # m, object initial centre to goal centre
GOAL_DISTANCE_MAX: float = 1.20
GOAL_TERMINAL_CLEARANCE: float = 0.05  # m, object placed at the goal must clear everything
SAMPLE_GOAL_DRAWS: int = 40

SPLIT_SIZES: Dict[str, int] = {"train": 256, "validation": 64, "test": 64}
SPLIT_SEEDS: Dict[str, int] = {"L0": 20261015, "L1": 20261016}

# ---------------------------------------------------------------------------
# sensing (task book section 3.1)
# ---------------------------------------------------------------------------

LIDAR_RANGE: float = 1.20        # m
LIDAR_RAYS: int = 72             # equal angular spacing over 360 degrees
LIDAR_HZ: float = 20.0           # Hz, one full scan per control period
PATCH_GAP_TOLERANCE: float = 0.03  # m, range continuity for grouping returns into a surface
PATCH_MIN_POINTS: int = 3          # a fitted normal needs at least three returns
#: Range-continuity band for grouping returns: ``PATCH_GAP_TOLERANCE + factor * d * dtheta``.
#: A planar face at a grazing angle changes range by ``d * dtheta * tan(phi)`` between
#: neighbouring rays, so an absolute band alone would shred it into single rays.
PATCH_RELATIVE_FACTOR: float = 2.5
#: A linear fit is accepted when its RMS perpendicular residual is below this. Exact
#: geometry plus a planar face gives ~0; a run spanning a corner does not.
MAX_FIT_RMS: float = 0.005
NORMAL_FILTER_TAU: float = 0.10    # s
NORMAL_JUMP_LIMIT_DEG: float = 30.0  # a larger change is a corner/face transition
CONTACT_LOSS_TOLERANCE: float = 0.15  # s of continuous contact loss inside PUSH -> abort
#: How long PUSH may keep pressing the visible surface without the contact sensor ever
#: reporting the object before it gives up. PUSH is entered on a *sensed contact*, so this
#: only fires when the surface is visible but physically unreachable.
CONTACT_SEEK_SECONDS: float = 1.00
PATCH_LOSS_TOLERANCE: float = 1.00    # s of continuous object-patch loss before giving up
OBSERVATION_FAIL_TOLERANCE: float = 1.00  # s, "lost the object" re-acquisition budget

# ---------------------------------------------------------------------------
# skills (task book section 4)
# ---------------------------------------------------------------------------

ALPHA_DEG: Tuple[float, ...] = (-15.0, 0.0, 15.0)
PUSH_SPEEDS: Tuple[float, ...] = (0.08, 0.16, 0.26)          # m/s commanded surface speed
REPOSITION_DIRECTIONS: Tuple[str, ...] = ("CCW", "CW")
REPOSITION_DISTANCES: Tuple[float, ...] = (0.25, 0.50)       # m along the visible boundary
WAIT_DURATIONS: Tuple[float, ...] = (0.50, 1.00)             # s
AGENTS: Tuple[int, int] = (0, 1)
ROBOT_ACTION_COUNT: int = 15
JOINT_ACTION_COUNT: int = ROBOT_ACTION_COUNT ** 2            # 225, unique per ordered pair

SKILL_MAX_SECONDS: float = 8.0        # per robot skill: seek + approach + execute + brake
JOINT_MAX_SECONDS: float = 8.0        # synchronous joint window
PUSH_ACTIVE_SECONDS: float = 2.0      # accumulated *object-contact* seconds (PUSH_TARGET_ACTIVE)
PUSH_WALL_SECONDS: float = 3.0        # wall-clock cap of the PUSH stage (PUSH_WALL_LIMIT)
APPROACH_TIMEOUT: float = 6.0         # s, raised from the inherited 5.0 s (see CHANGELOG)
APPROACH_MIN_SPEED: float = 0.02      # m/s, inherited from the staged protocol
#: Raised from the inherited 0.08 m/s. The inherited value was tuned for a fixed 0.35 m
#: staging standoff; a random start may be up to 0.50 m clear of the object, and 0.08 m/s
#: cannot close that inside the 8 s skill at all. The physics, the force limit and the
#: acceleration bound are untouched: this is the commanded approach speed only.
APPROACH_MAX_SPEED: float = 0.20      # m/s
APPROACH_GAIN: float = 2.0            # 1/s, gap-proportional approach speed
BRAKE_MAX_SECONDS: float = 0.80       # s of braking before the skill ends regardless
BRAKE_SPEED: float = 0.03             # m/s, below this the robot counts as braked
NO_PROGRESS_SECONDS: float = 2.0      # continuous seconds without progress -> blocked
REPOSITION_GAP: float = 0.10          # m, target robot-edge to object-surface gap
REPOSITION_SPEED_MAX: float = 0.12    # m/s, tangential speed cap
REPOSITION_ALIGN_TOLERANCE: float = 0.035  # m, gap error considered aligned
REPOSITION_SPEED: float = 0.12        # m/s, tangential command (equal to the contract cap)
#: The MOVE stage may start while the gap is still converging: the radial correction runs
#: inside MOVE as well, so waiting for exact alignment first would waste most of the 8 s
#: window on a 0.50 m reposition (0.50 m / 0.12 m/s is already 4.2 s of pure travel).
REPOSITION_ALIGN_ENTRY: float = 0.15  # m, gap error at which MOVE may start
REPOSITION_GAP_GAIN: float = 1.0      # 1/s, radial gap correction gain
#: Dead band around the target gap: inside it the radial correction is zero, so the tangential
#: motion cannot press the disc into the surface when the boundary curves away from the tangent.
REPOSITION_GAP_DEADBAND: float = 0.012
A_REF_MAX: float = 0.5                # m/s^2, inherited reference-velocity ramp limit
KV: float = 40.0                      # N s / m, inherited velocity feedback gain
FIND_SECONDS: float = 1.50            # s, bounded local seek before not_observed
FIND_SPEED: float = 0.06              # m/s, seek speed (same force interface, no new actuator)
SAFETY_GAP: float = 0.02              # m, robot-to-static/teammate measured gap -> safety abort
OBSTACLE_AVOID_GAIN: float = 2.0      # 1/s, repulsion used by REPOSITION near a static hit
OBSTACLE_AVOID_MARGIN: float = 0.12   # m, distance at which the repulsion starts

# ---------------------------------------------------------------------------
# task, budget, reward (task book section 5)
# ---------------------------------------------------------------------------

SUCCESS_DISTANCE: float = 0.20        # m, object centre to goal centre
SUCCESS_LINEAR: float = 0.02          # m/s
SUCCESS_ANGULAR: float = 0.05         # rad/s
SUCCESS_HOLD_SECONDS: float = 0.50    # s of continuous true physical time
TIME_BUDGET_SECONDS: float = 90.0
GAMMA: float = 1.0                    # finite-horizon undiscounted task, no delayed-failure benefit
TIME_COST: float = 0.002              # per control period, in reward units
SUCCESS_BONUS: float = 10.0
FAILURE_PENALTY: float = -2.0
PROGRESS_WEIGHT: float = 1.0          # reward / m, real distance decrease (including final period)
DISTANCE_COST: float = 0.02           # reward / (m s), integrated at CONTROL_DT

# v3.1 repairs PUSH's seek timeout (skill-local time, never episode elapsed time).
# Rewards, physical horizon and observation layout are unchanged. Old weights may
# be evaluated explicitly, but training/branch restores must not cross versions.
ENV_VERSION: str = "random-start-time-budget-v3.1"

# ---------------------------------------------------------------------------
# the 15 discrete robot actions and the 225 joint actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RobotAction:
    """One of the 15 discrete skills of one robot."""

    kind: str                     # "push" | "reposition" | "wait"
    alpha_deg: float = 0.0
    speed: float = 0.0
    direction: str = ""
    distance: float = 0.0
    duration: float = 0.0
    label: str = ""
    index: int = -1

    def parameters(self) -> Dict[str, float]:
        return {"alpha_deg": self.alpha_deg, "speed": self.speed,
                "cw": 1.0 if self.direction == "CW" else 0.0,
                "distance": self.distance, "duration": self.duration}


def action_table() -> Tuple[RobotAction, ...]:
    """The public menu, in a frozen order: 9 PUSH, 4 REPOSITION, 2 WAIT_BRAKE."""

    out = []
    for alpha in ALPHA_DEG:
        for speed in PUSH_SPEEDS:
            out.append(RobotAction(
                kind="push", alpha_deg=alpha, speed=speed,
                label=f"PUSH(a={alpha:+.0f}deg,v={speed:.2f})"))
    for direction in REPOSITION_DIRECTIONS:
        for distance in REPOSITION_DISTANCES:
            out.append(RobotAction(
                kind="reposition", direction=direction, distance=distance,
                label=f"REPOSITION_{direction}({distance:.2f}m)"))
    for duration in WAIT_DURATIONS:
        out.append(RobotAction(kind="wait", duration=duration,
                               label=f"WAIT_BRAKE({duration:.2f}s)"))
    table = []
    for index, action in enumerate(out):
        table.append(RobotAction(**{**action.__dict__, "index": index}))
    if len(table) != ROBOT_ACTION_COUNT:
        raise RuntimeError(f"action table has {len(table)} entries, expected {ROBOT_ACTION_COUNT}")
    return tuple(table)


ROBOT_ACTIONS: Tuple[RobotAction, ...] = action_table()


def joint_action_index(first: int, second: int) -> int:
    """Unique index of the ordered pair (robot0 action, robot1 action)."""

    if not (0 <= first < ROBOT_ACTION_COUNT and 0 <= second < ROBOT_ACTION_COUNT):
        raise ValueError(f"robot action out of range: {(first, second)}")
    return first * ROBOT_ACTION_COUNT + second


def joint_action(first: int, second: int) -> Tuple[RobotAction, RobotAction]:
    return ROBOT_ACTIONS[first], ROBOT_ACTIONS[second]


def joint_action_label(first: int, second: int) -> str:
    a0, a1 = joint_action(first, second)
    return f"{a0.label}|{a1.label}"


def degrees(value: float) -> float:
    return math.degrees(value)
