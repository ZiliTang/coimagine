"""The three feedback skills and the observation/skill contracts.

Only three local skills exist this round:

``move_to(world_xy)``
    Track a target position under the force limit (PD on position and velocity).
``push_contact(object_id, contact_local_xy, push_direction_world, force_limit)``
    Approach a contact point on the object's surface using the current object pose, then
    push along a world direction. The contact point is re-derived from the object's current
    pose every control period, so the skill follows a rotating object instead of aiming at a
    frozen point. Both the approach and the push are bounded forces; the robot can never be
    placed at the contact point and cannot pass through the object.
``hold``
    Damped braking to rest (``hold_semantics`` in the control config). It is never mixed
    with a position hold.

A controller may use the current full observation as feedback; it may not query the
simulator's future.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple, Union

from .backend import BodyState
from .config import CONTROLLER_VERSION, ControlConfig, PhysicsConfig
from .geometry import plan_path
from .scenes import Box

Vec = Tuple[float, float]


def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm(a: Vec) -> float:
    return math.hypot(a[0], a[1])


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _unit(vector: Vec) -> Vec:
    length = _norm(vector)
    if length < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return (vector[0] / length, vector[1] / length)


# ---------------------------------------------------------------------------
# Skill specs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MoveTo:
    target: Vec
    kind: str = "move_to"

    @property
    def label(self) -> str:
        return f"move_to({self.target[0]:.2f},{self.target[1]:.2f})"


@dataclass(frozen=True)
class PushContact:
    object_id: int
    contact_local: Vec
    push_direction_world: Vec
    force_limit: float
    kind: str = "push_contact"

    @property
    def label(self) -> str:
        direction = _unit(self.push_direction_world)
        return (
            f"push_contact(local=({self.contact_local[0]:+.2f},{self.contact_local[1]:+.2f}),"
            f"dir=({direction[0]:+.2f},{direction[1]:+.2f}),F={self.force_limit:.1f})"
        )


@dataclass(frozen=True)
class Hold:
    kind: str = "hold"

    @property
    def label(self) -> str:
        return "hold"


Skill = Union[MoveTo, PushContact, Hold]
JointSpec = Tuple[Skill, Skill]

#: Controller-internal state carried inside a snapshot and a data record.
ControllerState = Dict[str, object]


def skill_label(skill: Skill) -> str:
    return skill.label


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Observation:
    tick: int
    time: float
    robots: Tuple[BodyState, BodyState]
    object: BodyState
    goal: Box
    arena: Tuple[float, float]
    obstacles: Tuple[Box, ...]
    window_progress: float = 0.0
    controller_state: Tuple[ControllerState, ControllerState] = ({}, {})
    config_hash: str = ""
    contact_points: Tuple[Tuple[float, ...], ...] = ()
    object_size: Tuple[float, float] = (1.0, 0.25)

    def object_pose(self) -> Tuple[float, float, float]:
        return (self.object.x, self.object.y, self.object.angle)

    def robot_pose(self, robot_id: int) -> Tuple[float, float]:
        state = self.robots[robot_id]
        return (state.x, state.y)

    def robot_velocity(self, robot_id: int) -> Vec:
        state = self.robots[robot_id]
        return (state.vx, state.vy)

    def object_corners(self) -> Tuple[Vec, ...]:
        return _corners(self.object, self.object_size)


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------


def _corners(object_state: BodyState, size: Tuple[float, float]) -> Tuple[Vec, ...]:
    half_w, half_h = size[0] / 2.0, size[1] / 2.0
    cos, sin = math.cos(object_state.angle), math.sin(object_state.angle)
    return tuple(
        (
            object_state.x + dx * cos - dy * sin,
            object_state.y + dx * sin + dy * cos,
        )
        for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h))
    )


def _waypoint(
    observation: Observation,
    robot_id: int,
    target: Vec,
    state: ControllerState,
    physics: PhysicsConfig,
) -> Tuple[Vec, ControllerState]:
    """Current waypoint of a cached collision-free route to ``target``."""

    path = state.get("path")
    if path is None or state.get("anchor") != target:
        path = plan_path(
            observation.robot_pose(robot_id),
            target,
            arena=observation.arena,
            obstacles=observation.obstacles,
            object_corners=observation.object_corners(),
            robot_radius=physics.robot_radius,
        )
        state = {"path": list(path), "anchor": target}
    path = [tuple(point) for point in state["path"]]  # type: ignore[index]
    position = observation.robot_pose(robot_id)
    while len(path) > 1 and _norm(_sub(path[0], position)) < 0.10:
        path.pop(0)
    state["path"] = path
    return path[0], state


def contact_world(object_state: BodyState, local: Vec) -> Vec:
    cos, sin = math.cos(object_state.angle), math.sin(object_state.angle)
    return (
        object_state.x + local[0] * cos - local[1] * sin,
        object_state.y + local[0] * sin + local[1] * cos,
    )


def contact_point_velocity(object_state: BodyState, contact: Vec) -> Vec:
    """Linear velocity of the material point of the object at ``contact``."""

    rx = contact[0] - object_state.x
    ry = contact[1] - object_state.y
    return (
        object_state.vx - object_state.omega * ry,
        object_state.vy + object_state.omega * rx,
    )


def control(
    skill: Skill,
    observation: Observation,
    robot_id: int,
    control_config: ControlConfig,
    physics: PhysicsConfig,
    *,
    state: ControllerState,
) -> Tuple[Vec, ControllerState]:
    """Return the commanded force and the updated controller state."""

    if isinstance(skill, MoveTo):
        return _move_to(skill, observation, robot_id, control_config, physics, state)
    if isinstance(skill, PushContact):
        return _push_contact(skill, observation, robot_id, control_config, physics, state)
    if isinstance(skill, Hold):
        return _hold(observation, robot_id, control_config, physics), {"mode": "hold"}
    raise TypeError(f"unknown skill {skill!r}")


def _move_to(
    skill: MoveTo,
    observation: Observation,
    robot_id: int,
    control_config: ControlConfig,
    physics: PhysicsConfig,
    state: ControllerState,
) -> Tuple[Vec, ControllerState]:
    waypoint, state = _waypoint(observation, robot_id, skill.target, state, physics)
    position = observation.robot_pose(robot_id)
    velocity = observation.robot_velocity(robot_id)
    error = _sub(waypoint, position)
    force = (
        control_config.move_kp * error[0] - control_config.move_kd * velocity[0],
        control_config.move_kp * error[1] - control_config.move_kd * velocity[1],
    )
    state["mode"] = "move_to"
    return _clamp_force(force, physics.max_force), state


def _hold(
    observation: Observation,
    robot_id: int,
    control_config: ControlConfig,
    physics: PhysicsConfig,
) -> Vec:
    velocity = observation.robot_velocity(robot_id)
    force = (
        -control_config.hold_kd * velocity[0],
        -control_config.hold_kd * velocity[1],
    )
    return _clamp_force(force, physics.max_force)


def _push_contact(
    skill: PushContact,
    observation: Observation,
    robot_id: int,
    control_config: ControlConfig,
    physics: PhysicsConfig,
    state: ControllerState,
) -> Tuple[Vec, ControllerState]:
    direction = _unit(skill.push_direction_world)
    perpendicular = (-direction[1], direction[0])
    contact = contact_world(observation.object, skill.contact_local)
    desired = (
        contact[0] - direction[0] * physics.robot_radius,
        contact[1] - direction[1] * physics.robot_radius,
    )
    position = observation.robot_pose(robot_id)
    velocity = observation.robot_velocity(robot_id)
    gap = _dot(_sub(contact, position), direction) - physics.robot_radius

    mode = str(state.get("mode", "approach"))
    if mode == "approach" and gap <= 0.005:
        mode = "push"

    if mode == "approach":
        # Route around the object instead of driving through it. The route is planned once
        # per window towards the desired point, and abandoned for a direct PD once close.
        distance_to_desired = _norm(_sub(desired, position))
        if distance_to_desired < 0.25:
            target = desired
        else:
            target, state = _waypoint(observation, robot_id, desired, state, physics)
        error = _sub(target, position)
        force = (
            control_config.move_kp * error[0] - control_config.move_kd * velocity[0],
            control_config.move_kp * error[1] - control_config.move_kd * velocity[1],
        )
        limit = min(physics.max_force, skill.force_limit)
        state["mode"] = "approach"
        return _clamp_force(force, limit), state

    material_velocity = contact_point_velocity(observation.object, contact)
    relative_along = _dot(_sub(material_velocity, velocity), direction)
    forward = control_config.push_k_distance * max(0.0, gap) + control_config.push_k_velocity * (
        control_config.push_speed - relative_along
    )
    forward = _clamp(forward, 0.0, skill.force_limit)
    lateral_error = _dot(_sub(desired, position), perpendicular)
    lateral_velocity = _dot(velocity, perpendicular)
    lateral = (
        control_config.push_lateral_kp * lateral_error
        - control_config.push_lateral_kd * lateral_velocity
    )
    force = (
        direction[0] * forward + perpendicular[0] * lateral,
        direction[1] * forward + perpendicular[1] * lateral,
    )
    return _clamp_force(force, min(physics.max_force, skill.force_limit)), {"mode": "push"}


def _clamp_force(force: Vec, limit: float) -> Vec:
    norm = _norm(force)
    if norm > limit and norm > 0.0:
        scale = limit / norm
        return (force[0] * scale, force[1] * scale)
    return force


def is_done(
    skill: Skill,
    observation: Observation,
    robot_id: int,
    control_config: ControlConfig,
) -> bool:
    """Whether the skill's own termination condition is met this control period."""

    if isinstance(skill, MoveTo):
        error = _sub(skill.target, observation.robot_pose(robot_id))
        speed = _norm(observation.robot_velocity(robot_id))
        return _norm(error) <= control_config.move_position_tolerance and speed <= control_config.move_speed_tolerance
    if isinstance(skill, Hold):
        speed = _norm(observation.robot_velocity(robot_id))
        return speed <= control_config.hold_speed_tolerance
    return False


def version() -> str:
    return CONTROLLER_VERSION


# ---------------------------------------------------------------------------
# Serialisation (schema v2): structured, full precision, no parsing of labels
# ---------------------------------------------------------------------------


def skill_to_dict(skill: Skill) -> Dict[str, object]:
    """Serialise a skill to a structured, full-precision dictionary."""

    if isinstance(skill, MoveTo):
        return {"kind": "move_to", "target": [float(skill.target[0]), float(skill.target[1])]}
    if isinstance(skill, PushContact):
        return {
            "kind": "push_contact",
            "object_id": int(skill.object_id),
            "contact_local": [float(skill.contact_local[0]), float(skill.contact_local[1])],
            "push_direction_world": [
                float(skill.push_direction_world[0]),
                float(skill.push_direction_world[1]),
            ],
            "force_limit": float(skill.force_limit),
        }
    if isinstance(skill, Hold):
        return {"kind": "hold"}
    raise TypeError(f"unknown skill {skill!r}")


def skill_from_dict(payload: Dict[str, object]) -> Skill:
    """Rebuild a skill from :func:`skill_to_dict` without reading its label."""

    kind = payload["kind"]
    if kind == "move_to":
        target = payload["target"]
        return MoveTo((float(target[0]), float(target[1])))  # type: ignore[index]
    if kind == "push_contact":
        contact = payload["contact_local"]  # type: ignore[index]
        direction = payload["push_direction_world"]  # type: ignore[index]
        return PushContact(
            object_id=int(payload["object_id"]),  # type: ignore[arg-type]
            contact_local=(float(contact[0]), float(contact[1])),  # type: ignore[index]
            push_direction_world=(float(direction[0]), float(direction[1])),  # type: ignore[index]
            force_limit=float(payload["force_limit"]),  # type: ignore[arg-type]
        )
    if kind == "hold":
        return Hold()
    raise ValueError(f"unknown serialised skill kind {kind!r}")


def joint_spec_to_dicts(joint_spec: JointSpec) -> List[Dict[str, object]]:
    return [skill_to_dict(joint_spec[0]), skill_to_dict(joint_spec[1])]


def joint_spec_from_dicts(payloads: Sequence[Dict[str, object]]) -> JointSpec:
    return (skill_from_dict(payloads[0]), skill_from_dict(payloads[1]))
