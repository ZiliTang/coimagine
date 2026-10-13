"""Finite joint-candidate template, generated from public geometry only.

For each of the object's four faces the generator samples contact points along the face and
pairs each with the **inward** face normal as the push direction, so the robot stands
outside the object and pushes into the surface. Off-centre contacts therefore produce real
torque: pushing near an end rotates the object, pushing through the face centre translates
it. Two staging ``move_to`` points and one ``hold`` complete the per-robot option list.

Nothing here reads hidden physics (there is none in the continuous environment), a future
state, the true goal score, or a scene id. The same template is the only source of
candidates for the planner, the random control and the manual witness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .config import CandidateConfig, ControlConfig, PhysicsConfig
from .scenes import Box
from .skills import Hold, JointSpec, MoveTo, Observation, PushContact, Skill

Vec = Tuple[float, float]


@dataclass(frozen=True)
class JointCandidate:
    candidate_id: int
    skills: JointSpec
    label: str


def _rotate(vector: Vec, angle: float) -> Vec:
    cos, sin = math.cos(angle), math.sin(angle)
    return (vector[0] * cos - vector[1] * sin, vector[0] * sin + vector[1] * cos)


def _add(a: Vec, b: Vec) -> Vec:
    return (a[0] + b[0], a[1] + b[1])


def _scale(a: Vec, k: float) -> Vec:
    return (a[0] * k, a[1] * k)


def per_robot_options(
    observation: Observation,
    candidates: CandidateConfig,
    control: ControlConfig,
    physics: PhysicsConfig,
) -> List[Skill]:
    """The per-robot option list: pushes, staging moves, hold -- in a fixed order."""

    object_state = observation.object
    angle = object_state.angle
    half_w = observation.object_size[0] / 2.0
    half_h = observation.object_size[1] / 2.0

    # Local outward normals and the face half-length to sample along.
    faces = (
        ((1.0, 0.0), half_h),
        ((-1.0, 0.0), half_h),
        ((0.0, 1.0), half_w),
        ((0.0, -1.0), half_w),
    )
    samples = max(1, candidates.contact_samples_per_face)
    if samples == 1:
        offsets = (0.0,)
    else:
        offsets = tuple(
            2.0 * index / (samples - 1) - 1.0 for index in range(samples)
        )

    options: List[Skill] = []
    for normal_local, face_half in faces[: candidates.faces_per_robot]:
        tangent_local = (-normal_local[1], normal_local[0])
        for offset in offsets:
            local = _add(_scale(normal_local, half_w if normal_local[0] else half_h),
                         _scale(tangent_local, offset * 0.55 * face_half))
            inward = _scale(_rotate(normal_local, angle), -1.0)
            for level in candidates.force_levels:
                options.append(
                    PushContact(
                        object_id=0,
                        contact_local=(round(local[0], 6), round(local[1], 6)),
                        push_direction_world=(round(inward[0], 6), round(inward[1], 6)),
                        force_limit=physics.max_force * level,
                    )
                )

    # Staging points: outside the two faces whose outward normal is most anti-aligned with
    # the goal direction, i.e. the sides a robot would use to push the object goal-ward.
    goal_direction = _unit(
        (observation.goal.cx - object_state.x, observation.goal.cy - object_state.y)
    )
    ranked = sorted(
        faces,
        key=lambda face: _dot(_rotate(face[0], angle), goal_direction),
    )
    for normal_local, _half in ranked[:2]:
        face_center = _add(
            (object_state.x, object_state.y),
            _rotate(_scale(normal_local, half_w if normal_local[0] else half_h), angle),
        )
        normal_world = _rotate(normal_local, angle)
        stage = _add(face_center, _scale(normal_world, physics.robot_radius + candidates.staging_standoff))
        options.append(MoveTo((round(stage[0], 6), round(stage[1], 6))))

    options.append(Hold())
    return options


def joint_candidates(
    observation: Observation,
    candidates: CandidateConfig,
    control: ControlConfig,
    physics: PhysicsConfig,
) -> List[JointCandidate]:
    """Every pair of per-robot options, in a fixed stable order, capped at the budget."""

    options = per_robot_options(observation, candidates, control, physics)
    out: List[JointCandidate] = []
    candidate_id = 0
    for first in options:
        for second in options:
            out.append(
                JointCandidate(candidate_id, (first, second), f"A:{first.label} | B:{second.label}")
            )
            candidate_id += 1
            if candidate_id >= candidates.max_candidates:
                return out
    return out


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _unit(vector: Vec) -> Vec:
    length = math.hypot(vector[0], vector[1])
    if length < 1e-9:
        return (1.0, 0.0)
    return (vector[0] / length, vector[1] / length)
