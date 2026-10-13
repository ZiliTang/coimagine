"""Deterministic input/target encoding for the single-scale window model.

Layout is frozen in ``docs/forecast_baseline_v1/CONTRACT.md`` section 2 (B1) and is
implemented exactly once here. Nothing in this module reads hidden information: the input
is built from the robot's own window-boundary state (its own scan of the world), the public
static geometry and the full-precision joint skill parameters. The target is the real
successor, used only as a label.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from ..continuous2d.backend import BodyState
from ..continuous2d.config import PhysicsConfig
from ..continuous2d.scenes import SceneSpec
from ..continuous2d.skills import Hold, JointSpec, MoveTo, PushContact, Skill
from .contract import (
    ANGULAR_VELOCITY_SCALE,
    DURATION_SCALE,
    HOLD_SCALE,
    MAX_OBSTACLE_SLOTS,
    TERMINATION_INDEX,
    VELOCITY_SCALE,
    wrap_angle,
)

INPUT_DIM = 57
STATE_DIM = 18
PARAM_DIM = 5
SKILL_DIM = 3 + PARAM_DIM
TERMINATION_DIM = 3

BODY_FIELDS = ("object", "robot0", "robot1")


@dataclass(frozen=True)
class BoundaryState:
    """One window boundary: the three bodies plus the task-level success hold timer."""

    object: BodyState
    robots: Tuple[BodyState, BodyState]
    success_counter: float = 0.0

    def bodies(self) -> Tuple[BodyState, BodyState, BodyState]:
        return (self.object, self.robots[0], self.robots[1])

    def as_dict(self) -> Dict[str, object]:
        return {
            "object": list(self.object.as_tuple()),
            "robots": [list(state.as_tuple()) for state in self.robots],
            "success_counter": self.success_counter,
        }


def skill_features(skill: Skill, scene: SceneSpec, physics: PhysicsConfig) -> List[float]:
    """One skill as ``kind one-hot(3) + 5 full-precision parameters``."""

    width, height = scene.arena
    half_w = physics.object_width / 2.0
    half_h = physics.object_height / 2.0
    out = [0.0] * SKILL_DIM
    if isinstance(skill, MoveTo):
        out[0] = 1.0
        out[3] = skill.target[0] / width
        out[4] = skill.target[1] / height
    elif isinstance(skill, PushContact):
        out[1] = 1.0
        out[3] = skill.contact_local[0] / half_w
        out[4] = skill.contact_local[1] / half_h
        out[5] = skill.push_direction_world[0]
        out[6] = skill.push_direction_world[1]
        out[7] = skill.force_limit / physics.max_force
    elif isinstance(skill, Hold):
        out[2] = 1.0
    else:  # pragma: no cover - the skill union has exactly three members
        raise TypeError(f"unknown skill {skill!r}")
    return out


def _body_features(state: BodyState, width: float, height: float) -> List[float]:
    return [
        state.x / width,
        state.y / height,
        math.sin(state.angle),
        math.cos(state.angle),
        state.vx / VELOCITY_SCALE,
        state.vy / VELOCITY_SCALE,
        state.omega / ANGULAR_VELOCITY_SCALE,
    ]


def encode_input(
    scene: SceneSpec,
    state: BoundaryState,
    joint_spec: JointSpec,
    physics: PhysicsConfig,
    *,
    window_max_duration: float = DURATION_SCALE,
) -> List[float]:
    width, height = scene.arena
    features: List[float] = []
    for body in state.bodies():
        features += _body_features(body, width, height)
    features.append(state.success_counter / HOLD_SCALE)
    goal = scene.goal
    features += [goal.cx / width, goal.cy / height, goal.half_w / width, goal.half_h / height]
    features += [width, height]
    for index in range(MAX_OBSTACLE_SLOTS):
        if index < len(scene.obstacles):
            box = scene.obstacles[index]
            features += [
                box.cx / width, box.cy / height, box.half_w / width, box.half_h / height,
            ]
        else:
            features += [0.0, 0.0, 0.0, 0.0]
    for index in range(MAX_OBSTACLE_SLOTS):
        features.append(1.0 if index < len(scene.obstacles) else 0.0)
    features += [physics.object_width / width, physics.object_height / height]
    features.append(window_max_duration / DURATION_SCALE)
    features += skill_features(joint_spec[0], scene, physics)
    features += skill_features(joint_spec[1], scene, physics)
    if len(features) != INPUT_DIM:  # pragma: no cover - guarded by the unit test
        raise AssertionError(f"input layout produced {len(features)} values, expected {INPUT_DIM}")
    return features


def encode_state_target(previous: BoundaryState, next_state: BoundaryState, scene: SceneSpec) -> List[float]:
    """The 18 normalised per-body state deltas (positions, wrapped angle, velocities)."""

    width, height = scene.arena
    out: List[float] = []
    for before, after in zip(previous.bodies(), next_state.bodies()):
        out += [
            (after.x - before.x) / width,
            (after.y - before.y) / height,
            wrap_angle(after.angle - before.angle) / math.pi,
            (after.vx - before.vx) / VELOCITY_SCALE,
            (after.vy - before.vy) / VELOCITY_SCALE,
            (after.omega - before.omega) / ANGULAR_VELOCITY_SCALE,
        ]
    return out


def apply_state_delta(
    state: BoundaryState, delta: Sequence[float], scene: SceneSpec
) -> BoundaryState:
    """Apply the model's 18-value delta to a boundary state (angles wrapped)."""

    width, height = scene.arena
    bodies = []
    for index, body in enumerate(state.bodies()):
        base = index * 6
        bodies.append(
            BodyState(
                x=body.x + delta[base + 0] * width,
                y=body.y + delta[base + 1] * height,
                angle=wrap_angle(body.angle + delta[base + 2] * math.pi),
                vx=body.vx + delta[base + 3] * VELOCITY_SCALE,
                vy=body.vy + delta[base + 4] * VELOCITY_SCALE,
                omega=body.omega + delta[base + 5] * ANGULAR_VELOCITY_SCALE,
            )
        )
    return BoundaryState(object=bodies[0], robots=(bodies[1], bodies[2]), success_counter=state.success_counter)


def termination_index(name: str) -> int:
    return TERMINATION_INDEX[name]
