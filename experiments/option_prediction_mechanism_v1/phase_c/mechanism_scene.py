"""The explicit long-rectangle *mechanism* scene for phase C (information ambiguity).

Why a new scene is needed
-------------------------
The frozen continuous-2D object is 1.0 x 0.25 m and the frozen local scan range is 0.6 m
(``coimagine.local_contact_option.contract.SCAN_RANGE``).  For any legal contact stance the
object's short (end) faces of that object fall inside the 0.6 m scan, so the object's
endpoints are always visible and a tangential translation of the object along its own long
edge is *immediately observable*.  CONTRACT section 3.2 explicitly permits one explicit long
rectangle for exactly this reason; this module builds it.  Nothing about the old benchmarks
is changed, and this scene is never registered in the scene library.

What is held fixed inside a pair
--------------------------------
Shape, size, mass, friction, damping, force limits, goal, obstacles, arena and the **global
robot poses** are identical in the two worlds.  Only ``object_pose`` differs, translated by
``shift`` along the object's own long (x) axis.  The visible part of the contact edge keeps
the same world coordinates to machine precision, while the object's centre of mass moves, so
the identical commanded contact force acts at a different lever arm.  The robots are placed
so that this is the only difference: their poses and velocities are exactly equal in both
worlds, which is required because the robots' own global poses are known and shareable.

Scene conventions
-----------------
The object's long axis is along x (``object_len`` is the long side, ``object_thick`` the
short one).  The two long edges are horizontal at ``y = edge_y -+ thick/2``.  Robot 0 stands
just below the lower long edge, robot 1 just above the upper long edge, both facing the edge,
so each sees a straight unbounded-looking segment whose range returns do not depend on the
object's tangential position.  Every body is kept clear of the arena boundary capsules
(radius 0.05 m).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Tuple

from coimagine.continuous2d.config import Continuous2DConfig, PhysicsConfig, default_config
from coimagine.continuous2d.scenes import Box, Pose, SceneSpec

#: Defaults of the long-rectangle mechanism.  ``object_len`` is chosen so that the short
#: faces stay outside the 0.6 m scan for every robot stance used below; that is verified per
#: candidate pair by ``perception.observe(...).robots[i].scan.short_face_visible``.
OBJECT_LEN = 3.00
OBJECT_THICK = 0.30
EDGE_Y = 2.00
ARENA = (14.00, 4.00)
ROBOT_R = 0.18
#: Initial standoff between the robot hull and the contact edge.  Small enough that the push
#: controller is already in its contact branch, so a single skill window applies real force
#: from the first control period; the exact value is recorded in every result.
CONTACT_STANDOFF = 0.02


def edge_lines(edge_y: float = EDGE_Y, thick: float = OBJECT_THICK) -> Tuple[float, float]:
    return (edge_y - thick / 2.0, edge_y + thick / 2.0)


@dataclass(frozen=True)
class MechanismParams:
    """One mechanism candidate.  World A is as written; world B shifts the object by ``shift``."""

    name: str
    object_centre_x: float
    shift: float
    goal: Box
    obstacles: Tuple[Box, ...] = ()
    object_len: float = OBJECT_LEN
    object_thick: float = OBJECT_THICK
    edge_y: float = EDGE_Y
    robot_x: Tuple[float, float] = (6.00, 8.00)
    arena: Tuple[float, float] = ARENA
    standoff: float = CONTACT_STANDOFF

    def edges(self) -> Tuple[float, float]:
        return edge_lines(self.edge_y, self.object_thick)

    def robot_poses(self) -> Tuple[Pose, Pose]:
        lower, upper = self.edges()
        return (
            Pose(self.robot_x[0], lower - ROBOT_R - self.standoff, math.pi / 2.0),
            Pose(self.robot_x[1], upper + ROBOT_R + self.standoff, -math.pi / 2.0),
        )

    def _scene(self, world: str) -> SceneSpec:
        delta = 0.0 if world == "A" else self.shift
        return SceneSpec(
            name=f"{self.name}_{world}",
            category="mechanism_C",
            variant=0 if world == "A" else 1,
            arena=self.arena,
            obstacles=tuple(self.obstacles),
            goal=self.goal,
            object_pose=Pose(self.object_centre_x + delta, self.edge_y, 0.0),
            robot_poses=self.robot_poses(),
            description="phase-C long-rectangle mechanism scene (not a benchmark scene)",
            scene_version="mechanism_C_v1",
            object_width=self.object_len,
            object_height=self.object_thick,
            robot_radius=ROBOT_R,
        )

    def scenes(self) -> Tuple[SceneSpec, SceneSpec]:
        return (self._scene("A"), self._scene("B"))

    def object_pose(self, world: str) -> Tuple[float, float, float]:
        delta = 0.0 if world == "A" else self.shift
        return (self.object_centre_x + delta, self.edge_y, 0.0)

    def short_face_x(self) -> Tuple[float, float]:
        half = self.object_len / 2.0
        return (self.object_centre_x - half, self.object_centre_x + half)

    def replace(self, **changes) -> "MechanismParams":
        return replace(self, **changes)


def lower_edge_contact_y(params: MechanismParams) -> float:
    return params.edges()[0]


def upper_edge_contact_y(params: MechanismParams) -> float:
    return params.edges()[1]


#: One joint-skill window.  The phase-C mechanism uses a longer window than the frozen 2.0 s
#: benchmark window so that a single skill can actually reach the goal; otherwise every
#: payoff is a tiny progress difference and no completion bit can ever flip.  The controller,
#: physics, force limits and the success rule are unchanged, and the value is recorded in
#: every result.
WINDOW_STEPS = 120  # 6.0 s per skill at the frozen 0.05 s control period


def mechanism_config(params: MechanismParams) -> Continuous2DConfig:
    """The frozen physics/control configuration with the mechanism's object size and window.

    ``PymunkBackend`` builds the object from ``PhysicsConfig.object_width/object_height``, not
    from the scene, so the long-rectangle geometry must be written into the physics config as
    well as into the scene.  Everything except the object size and the window length is the
    frozen default.  The resulting hash is recorded in every result record.
    """

    base = default_config()
    physics = PhysicsConfig(**{
        **base.physics.__dict__,
        "object_width": params.object_len,
        "object_height": params.object_thick,
        "window_steps": WINDOW_STEPS,
    })
    return Continuous2DConfig(
        physics=physics,
        control=base.control,
        candidates=base.candidates,
        mpc=base.mpc,
        seeds=base.seeds,
    )
