"""Frozen continuous-2D scene library: S0 free-space transport and S1 narrow passage.

Only two task families are in scope. S0 is for tuning the controller and the physics
parameters; it is not evidence about long-horizon planning. S1 forces preparation: the
object starts with its long axis across the passage, so a straight translation cannot fit
through the gap and the orientation must be changed by real contact before it can pass.

Each family has one development scene plus five frozen layout variants. The variants are
listed explicitly (not sampled at run time) so the evaluated set cannot drift and no hard
case can be dropped.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional, Tuple


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    angle: float = 0.0


@dataclass(frozen=True)
class Box:
    """Axis-aligned when ``angle == 0``; rotated boxes are supported by the backend."""

    cx: float
    cy: float
    half_w: float
    half_h: float
    angle: float = 0.0


@dataclass(frozen=True)
class SceneSpec:
    name: str
    category: str
    variant: int
    arena: Tuple[float, float]
    obstacles: Tuple[Box, ...]
    goal: Box
    object_pose: Pose
    robot_poses: Tuple[Pose, Pose]
    description: str = ""
    scene_version: str = "v1"
    object_width: float = 1.0
    object_height: float = 0.25
    robot_radius: float = 0.18

    @property
    def gap_width(self) -> float:
        """The free opening in the passage wall, for the record."""

        if len(self.obstacles) != 2:
            return float("nan")
        lower, upper = sorted(self.obstacles, key=lambda box: box.cy)
        return (upper.cy - upper.half_h) - (lower.cy + lower.half_h)


#: Object footprint (width along its local x, height along its local y), from PhysicsConfig.
OBJECT_WIDTH = 1.0
OBJECT_HEIGHT = 0.25
#: Passage opening. Wider than the object's short side (0.25) and narrower than its long
#: side (1.0), so the initial diagonal orientation is blocked and an aligned one is not.
#: The passable angular window is about +-28 degrees around an aligned orientation.
PASSAGE_GAP = 0.70
_ARENA_S0 = (3.2, 2.2)
_ARENA_S1 = (3.0, 2.0)
_WALL_X = 1.60
_WALL_HALF = 0.09


#: Closeout fix (task A1). The arena boundary is a capsule of radius 0.05, so a robot
#: starting at x < 0.235 overlaps it. The three S0 scenes below started at 0.22/0.17/0.19
#: and are moved to the smallest position satisfying the uniform 0.005 m non-contact
#: margin; S0_v2's second robot additionally had to move in y because the object margin
#: would otherwise bind. Before/after coordinates and clearances are recorded in
#: docs/closeout_v1/scene_legality.md. Only the robots moved: object, goal, obstacles and
#: all physics parameters are unchanged.
_ROBOT_X_FIXED = 0.235001
_S0_V2_ROBOT_B_Y_EXTRA = 0.0386701858489629


def _s0(name: str, variant: int, *, object_dx: float, object_dy: float, angle: float,
        robot_dy: float, goal_dx: float, robot_x: Optional[float] = None,
        robot_b_y_extra: float = 0.0) -> SceneSpec:
    base_x = 0.22 + object_dx if robot_x is None else robot_x
    version = "closeout_v2" if (robot_x is not None or robot_b_y_extra) else "v1"
    return SceneSpec(
        name=name,
        category="S0",
        variant=variant,
        arena=_ARENA_S0,
        obstacles=(),
        goal=Box(2.35 + goal_dx, 1.10, 0.65, 0.60),
        object_pose=Pose(0.95 + object_dx, 1.10 + object_dy, angle),
        robot_poses=(
            Pose(base_x, 0.90 + object_dy + robot_dy),
            Pose(base_x, 1.30 + object_dy + robot_dy + robot_b_y_extra),
        ),
        description="free-space transport: contact push, off-centre push, two-sided response",
        scene_version=version,
    )


def _s1(name: str, variant: int, *, gap_center: float, object_dx: float, object_dy: float,
        angle: float, robot_dy: float) -> SceneSpec:
    width, height = _ARENA_S1
    lower_height = gap_center - PASSAGE_GAP / 2.0
    upper_start = gap_center + PASSAGE_GAP / 2.0
    lower = Box(_WALL_X, lower_height / 2.0, _WALL_HALF, lower_height / 2.0)
    upper = Box(_WALL_X, (upper_start + height) / 2.0, _WALL_HALF, (height - upper_start) / 2.0)
    return SceneSpec(
        name=name,
        category="S1",
        variant=variant,
        arena=_ARENA_S1,
        obstacles=(lower, upper),
        goal=Box(2.25, gap_center, 0.55, 0.55),
        object_pose=Pose(0.95 + object_dx, gap_center + object_dy, angle),
        robot_poses=(
            Pose(0.30 + object_dx, gap_center - 0.45 + robot_dy),
            Pose(1.30 + object_dx, gap_center + 0.70 + robot_dy),
        ),
        description="narrow passage: the long axis must be rotated before the object fits",
    )


S0_DEVELOPMENT = _s0("S0_dev", 0, object_dx=0.0, object_dy=0.0, angle=0.0, robot_dy=0.0,
                     goal_dx=0.0, robot_x=_ROBOT_X_FIXED)
S0_VARIANTS: Tuple[SceneSpec, ...] = (
    S0_DEVELOPMENT,
    _s0("S0_v1", 1, object_dx=0.06, object_dy=0.05, angle=0.0, robot_dy=0.02, goal_dx=0.05),
    _s0("S0_v2", 2, object_dx=-0.05, object_dy=-0.04, angle=0.0, robot_dy=-0.03, goal_dx=-0.05,
        robot_x=_ROBOT_X_FIXED, robot_b_y_extra=_S0_V2_ROBOT_B_Y_EXTRA),
    _s0("S0_v3", 3, object_dx=0.03, object_dy=0.07, angle=0.0, robot_dy=0.04, goal_dx=0.02),
    _s0("S0_v4", 4, object_dx=-0.03, object_dy=-0.06, angle=0.0, robot_dy=-0.05, goal_dx=-0.03,
        robot_x=_ROBOT_X_FIXED),
)

_ALIGNMENT = math.radians(55.0)
S1_DEVELOPMENT = _s1("S1_dev", 0, gap_center=1.00, object_dx=0.0, object_dy=0.0,
                     angle=_ALIGNMENT, robot_dy=0.0)
S1_VARIANTS: Tuple[SceneSpec, ...] = (
    S1_DEVELOPMENT,
    _s1("S1_v1", 1, gap_center=1.03, object_dx=0.02, object_dy=0.02,
        angle=_ALIGNMENT + 0.05, robot_dy=0.02),
    _s1("S1_v2", 2, gap_center=0.97, object_dx=-0.02, object_dy=-0.02,
        angle=_ALIGNMENT - 0.05, robot_dy=-0.02),
    _s1("S1_v3", 3, gap_center=1.01, object_dx=0.01, object_dy=-0.02,
        angle=_ALIGNMENT + 0.03, robot_dy=-0.03),
    _s1("S1_v4", 4, gap_center=0.99, object_dx=-0.01, object_dy=0.02,
        angle=_ALIGNMENT - 0.03, robot_dy=0.03),
)


def default_scenes() -> Tuple[SceneSpec, ...]:
    """All ten frozen scenes, S0 then S1, in a stable order."""

    return S0_VARIANTS + S1_VARIANTS


#: Optional ``name -> Optional[SceneSpec]`` resolver installed by
#: ``coimagine.coverage_scale.scene_registry``. It is ``None`` unless a run explicitly
#: activates a registry, so the default behaviour of the frozen library is unchanged.
_SCENE_RESOLVER = None


def set_scene_resolver(resolver) -> None:
    """Install (or clear, with ``None``) the extra scene resolver.

    The resolver is consulted before the frozen library; returning ``None`` falls through to
    the ten original names. This is the "minimal compatible mechanism" of the coverage-scale
    round: every existing call site keeps working, and the original names still resolve to
    their frozen definitions.
    """

    global _SCENE_RESOLVER
    _SCENE_RESOLVER = resolver


def scene_by_name(name: str) -> SceneSpec:
    if _SCENE_RESOLVER is not None:
        extra = _SCENE_RESOLVER(name)
        if extra is not None:
            return extra
    for scene in default_scenes():
        if scene.name == name:
            return scene
    raise KeyError(name)


def with_object_pose(scene: SceneSpec, pose: Pose) -> SceneSpec:
    """Used only by reset/restore and the manual witness; never to teleport mid-episode."""

    return replace(scene, object_pose=pose)
