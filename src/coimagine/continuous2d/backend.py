"""Pymunk/Chipmunk backend: the only place that touches the physics engine.

Pinned engine: ``pymunk 6.9.0`` (installed from the official wheel; verified against the
6.9 API). No custom collision engine is written, and nothing outside this module may write
a body position during an episode: the environment interacts with the world through bounded
forces only.

Snapshot semantics
------------------
Chipmunk does not expose an exact clone of a stepped space, and restoring pose and velocity
alone does not restore the cached contact impulses. ``export_state`` therefore stores both
the body state *and* the applied-control prefix. ``restore`` rebuilds a fresh space and
writes the body state (fast, independent clones); ``restore_exact`` rebuilds and replays the
prefix (exact by construction). :func:`snapshot_equivalence` measures whether the fast path
is within tolerance of the exact one, and the acceptance report records the answer instead
of assuming it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import pymunk

from .config import PhysicsConfig
from .scenes import Box, SceneSpec

VecTuple = Tuple[float, float]

#: Numerical tolerances for verifying an exact prefix replay against the snapshot it came
#: from. Position, angle and the two velocity channels are separate so a failure names the
#: quantity that drifted; the default is the strict same-platform replay tolerance.
RESTORE_TOLERANCES: Dict[str, float] = {
    "position": 1e-9,
    "angle": 1e-9,
    "linear_velocity": 1e-9,
    "angular_velocity": 1e-9,
}


class SnapshotRestoreError(RuntimeError):
    """Exact prefix replay did not reproduce the snapshot within the declared tolerances."""


@dataclass(frozen=True)
class BodyState:
    x: float
    y: float
    angle: float
    vx: float
    vy: float
    omega: float

    def as_tuple(self) -> Tuple[float, ...]:
        return (self.x, self.y, self.angle, self.vx, self.vy, self.omega)


@dataclass
class Snapshot:
    """Everything needed to reproduce the rest of one task episode.

    Includes the task-level state that lives outside the physics engine
    (``success_counter``, ``termination``, ``controller_state``), the full scene
    configuration (so a custom ``SceneSpec`` restores without a name lookup), the physics
    configuration, and the applied-control ``prefix`` that ``restore_exact`` replays.
    """

    scene_config: SceneSpec
    physics: PhysicsConfig
    config_hash: str
    tick: int
    object: BodyState
    robots: Tuple[BodyState, BodyState]
    success_counter: float = 0.0
    termination: Optional[str] = None
    controller_state: Dict[str, object] = field(default_factory=dict)
    prefix: Tuple[Tuple[VecTuple, VecTuple], ...] = ()

    @property
    def scene_name(self) -> str:
        return self.scene_config.name

    def as_key(self) -> Tuple:
        return (
            self.tick,
            self.object.as_tuple(),
            tuple(state.as_tuple() for state in self.robots),
            self.success_counter,
            self.termination,
        )


def _body_state(body: pymunk.Body) -> BodyState:
    return BodyState(
        float(body.position.x),
        float(body.position.y),
        float(body.angle),
        float(body.velocity.x),
        float(body.velocity.y),
        float(body.angular_velocity),
    )


def _compare_body(expected: BodyState, actual: BodyState) -> Dict[str, float]:
    return {
        "position": max(abs(expected.x - actual.x), abs(expected.y - actual.y)),
        "angle": abs(expected.angle - actual.angle),
        "linear_velocity": max(abs(expected.vx - actual.vx), abs(expected.vy - actual.vy)),
        "angular_velocity": abs(expected.omega - actual.omega),
    }


class PymunkBackend:
    """A single stepped world plus its reset/restore/step interface."""

    def __init__(
        self,
        scene: SceneSpec,
        physics: PhysicsConfig,
        config_hash: str = "",
    ):
        self.scene = scene
        self.physics = physics
        self.object_size = (physics.object_width, physics.object_height)
        self.config_hash = config_hash
        self.space: Optional[pymunk.Space] = None
        self.tick = 0
        self.prefix: List[Tuple[VecTuple, VecTuple]] = []
        self._build()

    # -- construction ------------------------------------------------------
    def _build(self, scene: Optional[SceneSpec] = None, physics: Optional[PhysicsConfig] = None) -> None:
        scene = scene or self.scene
        physics = physics or self.physics
        self.scene = scene
        self.physics = physics
        self.object_size = (physics.object_width, physics.object_height)
        space = pymunk.Space()
        space.iterations = 30
        width, height = scene.arena

        self.boundary_shapes: List[pymunk.Shape] = []
        boundary = 0.05
        for start, end in (
            ((0.0, 0.0), (width, 0.0)),
            ((width, 0.0), (width, height)),
            ((width, height), (0.0, height)),
            ((0.0, height), (0.0, 0.0)),
        ):
            segment = pymunk.Segment(space.static_body, start, end, boundary)
            segment.friction = physics.friction
            segment.elasticity = physics.elasticity
            space.add(segment)
            self.boundary_shapes.append(segment)

        self.obstacle_shapes: List[pymunk.Shape] = []
        for index, box in enumerate(scene.obstacles):
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = (box.cx, box.cy)
            body.angle = box.angle
            shape = pymunk.Poly.create_box(body, (2 * box.half_w, 2 * box.half_h))
            shape.friction = physics.friction
            shape.elasticity = physics.elasticity
            shape.collision_type = 1
            space.add(body, shape)
            self.obstacle_shapes.append(shape)

        self.static_shapes: List[pymunk.Shape] = [*self.boundary_shapes, *self.obstacle_shapes]

        object_mass = physics.object_mass
        object_moment = pymunk.moment_for_box(object_mass, self.object_size)
        self.object_body = pymunk.Body(object_mass, object_moment)
        self.object_body.position = (scene.object_pose.x, scene.object_pose.y)
        self.object_body.angle = scene.object_pose.angle
        self.object_shape = pymunk.Poly.create_box(self.object_body, self.object_size)
        self.object_shape.friction = physics.friction
        self.object_shape.elasticity = physics.elasticity
        self.object_shape.collision_type = 2
        space.add(self.object_body, self.object_shape)

        self.robot_bodies: List[pymunk.Body] = []
        self.robot_shapes: List[pymunk.Shape] = []
        for robot_id, pose in enumerate(scene.robot_poses):
            mass = physics.robot_mass
            moment = pymunk.moment_for_circle(mass, 0.0, physics.robot_radius)
            body = pymunk.Body(mass, moment)
            body.position = (pose.x, pose.y)
            body.angle = pose.angle
            shape = pymunk.Circle(body, physics.robot_radius)
            shape.friction = physics.friction
            shape.elasticity = physics.elasticity
            shape.collision_type = 3 + robot_id
            space.add(body, shape)
            self.robot_bodies.append(body)
            self.robot_shapes.append(shape)

        self.space = space

    # -- state -------------------------------------------------------------
    def object_state(self) -> BodyState:
        return _body_state(self.object_body)

    def robot_states(self) -> Tuple[BodyState, BodyState]:
        return (_body_state(self.robot_bodies[0]), _body_state(self.robot_bodies[1]))

    def export_state(
        self,
        *,
        controller_state: Optional[Dict[str, object]] = None,
        success_counter: float = 0.0,
        termination: Optional[str] = None,
    ) -> Snapshot:
        return Snapshot(
            scene_config=self.scene,
            physics=self.physics,
            config_hash=self.config_hash,
            tick=self.tick,
            object=self.object_state(),
            robots=self.robot_states(),
            success_counter=success_counter,
            termination=termination,
            controller_state=dict(controller_state or {}),
            prefix=tuple(self.prefix),
        )

    def _write_states(self, object_state: BodyState, robots: Sequence[BodyState]) -> None:
        for body, state in zip([self.object_body, *self.robot_bodies], [object_state, *robots]):
            body.position = (state.x, state.y)
            body.angle = state.angle
            body.velocity = (state.vx, state.vy)
            body.angular_velocity = state.omega
            body.force = (0.0, 0.0)
            body.torque = 0.0
        # Drop any cached contact information from the previous life of this space.
        self.space.reindex_shapes_for_body(self.object_body)
        for body in self.robot_bodies:
            self.space.reindex_shapes_for_body(body)

    def approximate_restore(self, snapshot: Snapshot) -> None:
        """Development-only fast clone: fresh space with the pose and velocity written in.

        This discards the engine's cached contact state and is therefore **not** the same
        world as the one the snapshot came from. It is named ``approximate`` so no caller
        can mistake it for exact; the main experiment and the exact backend never use it.
        """

        self._build(scene=snapshot.scene_config, physics=snapshot.physics)
        self.prefix = list(snapshot.prefix)
        self._write_states(snapshot.object, snapshot.robots)
        self.tick = snapshot.tick

    def restore_exact(
        self,
        snapshot: Snapshot,
        *,
        tolerances: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Rebuild the initial scene, replay the applied-control prefix, and verify.

        Returns the per-quantity maximum deviation between the replay and the snapshot.
        Raises :class:`SnapshotRestoreError` when a deviation exceeds ``tolerances``; it
        never silently projects the world back onto the snapshot.
        """

        tolerances = tolerances or RESTORE_TOLERANCES
        self._build(scene=snapshot.scene_config, physics=snapshot.physics)
        self.tick = 0
        self.prefix = []
        for control in snapshot.prefix:
            self.apply_and_step(control)
        if self.tick != snapshot.tick:
            raise SnapshotRestoreError(
                f"prefix replay produced tick {self.tick}, expected {snapshot.tick}"
            )

        object_deviation = _compare_body(snapshot.object, self.object_state())
        robot_deviations = [
            _compare_body(expected, actual)
            for expected, actual in zip(snapshot.robots, self.robot_states())
        ]
        deviation = {
            quantity: max([object_deviation[quantity], *(row[quantity] for row in robot_deviations)])
            for quantity in tolerances
        }
        violations = {
            quantity: value
            for quantity, value in deviation.items()
            if value > tolerances[quantity]
        }
        if violations:
            raise SnapshotRestoreError(
                "exact prefix replay deviates from the snapshot beyond tolerance: "
                + ", ".join(f"{name}={value:.3e}>{tolerances[name]:.1e}" for name, value in violations.items())
            )
        return deviation

    # -- stepping ----------------------------------------------------------
    def clamp_force(self, force: VecTuple) -> VecTuple:
        limit = self.physics.max_force
        norm = math.hypot(force[0], force[1])
        if norm > limit and norm > 0.0:
            scale = limit / norm
            return (force[0] * scale, force[1] * scale)
        return force

    def _apply_forces(self, control: Tuple[VecTuple, VecTuple]) -> None:
        physics = self.physics
        for body, force in zip(self.robot_bodies, control):
            clamped = self.clamp_force(force)
            body.force = clamped
            body.torque = 0.0
        for body in (self.object_body, *self.robot_bodies):
            body.force = (
                body.force.x - physics.linear_damping * body.velocity.x,
                body.force.y - physics.linear_damping * body.velocity.y,
            )
            body.torque += -physics.angular_damping * body.angular_velocity

    def apply_and_step(self, control: Tuple[VecTuple, VecTuple]) -> None:
        """One control period: ``substeps`` physics steps with the control held constant."""

        for _ in range(self.physics.substeps):
            self._apply_forces(control)
            self.space.step(self.physics.physics_dt)
        self.tick += 1
        self.prefix.append((tuple(control[0]), tuple(control[1])))

    # -- diagnostics -------------------------------------------------------
    def contacts(self) -> List[Dict[str, object]]:
        """Every dynamic-body contact, classified by collision pair.

        Four classes are reported separately: ``robot_object``, ``robot_robot``,
        ``robot_wall`` (arena boundary and static obstacles) and ``object_wall``. A
        penetration statistic that only looked at the first two would miss a robot or the
        object being driven into a wall.
        """

        out: List[Dict[str, object]] = []

        def _record(pair: str, robot_id, point) -> None:
            out.append(
                {
                    "pair": pair,
                    "robot_id": robot_id,
                    "point": (float(point.point_a.x), float(point.point_a.y)),
                    "distance": float(point.distance),
                }
            )

        for robot_id, shape in enumerate(self.robot_shapes):
            for point in shape.shapes_collide(self.object_shape).points:
                _record("robot_object", robot_id, point)
        for point in self.robot_shapes[0].shapes_collide(self.robot_shapes[1]).points:
            _record("robot_robot", -1, point)
        for robot_id, shape in enumerate(self.robot_shapes):
            for wall in self.static_shapes:
                for point in shape.shapes_collide(wall).points:
                    _record("robot_wall", robot_id, point)
        for wall in self.static_shapes:
            for point in self.object_shape.shapes_collide(wall).points:
                _record("object_wall", -1, point)
        return out

    def penetration_by_pair(self) -> Dict[str, float]:
        """Maximum penetration depth per collision class (0.0 when no overlap)."""

        classes = ("robot_object", "robot_robot", "robot_wall", "object_wall")
        worst = {name: 0.0 for name in classes}
        for entry in self.contacts():
            depth = max(0.0, -float(entry["distance"]))
            name = str(entry["pair"])
            if name in worst:
                worst[name] = max(worst[name], depth)
        return worst

    def max_penetration(self) -> float:
        depths = [max(0.0, -float(entry["distance"])) for entry in self.contacts()]
        return max(depths) if depths else 0.0

    def object_corners(self) -> Tuple[VecTuple, ...]:
        half_w, half_h = self.object_size[0] / 2.0, self.object_size[1] / 2.0
        angle = self.object_body.angle
        cos, sin = math.cos(angle), math.sin(angle)
        corners = []
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
            corners.append(
                (
                    self.object_body.position.x + dx * cos - dy * sin,
                    self.object_body.position.y + dx * sin + dy * cos,
                )
            )
        return tuple(corners)

    def finite(self) -> bool:
        values = [self.object_state().as_tuple(), *(state.as_tuple() for state in self.robot_states())]
        return all(math.isfinite(value) for state in values for value in state)

    def in_bounds(self) -> bool:
        width, height = self.scene.arena
        margin = self.physics.out_of_bounds_margin
        shapes = [self.object_body, *self.robot_bodies]
        for body in shapes:
            if not (-margin <= body.position.x <= width + margin):
                return False
            if not (-margin <= body.position.y <= height + margin):
                return False
        for corner in self.object_corners():
            if not (-margin <= corner[0] <= width + margin):
                return False
            if not (-margin <= corner[1] <= height + margin):
                return False
        return True


def snapshot_equivalence(
    backend: PymunkBackend, snapshot: Snapshot, control: Tuple[VecTuple, VecTuple]
) -> Dict[str, float]:
    """How far the development-only approximate clone drifts from exact prefix replay."""

    backend.approximate_restore(snapshot)
    backend.apply_and_step(control)
    fast_object = backend.object_state().as_tuple()
    fast_robots = tuple(state.as_tuple() for state in backend.robot_states())

    backend.restore_exact(snapshot)
    backend.apply_and_step(control)
    exact_object = backend.object_state().as_tuple()
    exact_robots = tuple(state.as_tuple() for state in backend.robot_states())

    return {
        "object_max_abs_diff": max(abs(a - b) for a, b in zip(fast_object, exact_object)),
        "robot_max_abs_diff": max(
            abs(a - b) for row_fast, row_exact in zip(fast_robots, exact_robots)
            for a, b in zip(row_fast, row_exact)
        ),
    }
