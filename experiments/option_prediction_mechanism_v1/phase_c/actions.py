"""The joint action set of one phase-C candidate pair.

CONTRACT section 3.3 fixes the budget: one common set of **at most eight** deployable joint
skills per pair, single skills first, then at most two-skill sequences, no deeper.

Parameterisation.  A skill is defined by an offset ``d`` along the *visible* long edge and a
push direction, both expressed in the object frame of world A.  ``PushContact`` stores the
contact in the object frame, so the same skill resolves to the same place on the object in
both worlds; because the two worlds share the same relative robot/object edge geometry, the
skill is also identical **relative to each robot**.  It is therefore realisable, and produces
the same commanded force, in both worlds.  No skill reads the hidden offset, and the two
worlds receive exactly the same command sequence.

What changes between the worlds is only where the object's centre of mass lies relative to
that shared contact geometry -- the lever arm of the applied force.  That is the entire
mechanism under test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from coimagine.continuous2d.skills import (
    Hold,
    JointSpec,
    MoveTo,
    PushContact,
    skill_to_dict,
)

from mechanism_scene import MechanismParams

Vec = Tuple[float, float]


@dataclass(frozen=True)
class NamedJointSkill:
    skill_id: str
    label: str
    joint: JointSpec
    #: The visible-edge description, for the record (not read by the simulator).
    provenance: Dict[str, object] = None  # type: ignore[assignment]

    def as_dict(self) -> Dict[str, object]:
        return {
            "skill_id": self.skill_id,
            "label": self.label,
            "joint_skill_specs": [skill_to_dict(self.joint[0]), skill_to_dict(self.joint[1])],
            "provenance": dict(self.provenance or {}),
        }


def edge_contact(params: MechanismParams, robot_id: int, offset: float) -> Tuple[float, float]:
    """Object-frame contact on the edge seen by ``robot_id`` at a deployable reference.

    ``offset`` is measured from ``params.reference_x``, a point of the known static map, along
    the long axis.  The object-frame coordinate is filled in only because ``PushContact``
    stores it that way; the *decision* is "place my contact point at world x =
    reference_x + offset on the visible edge", which the robot can compute from its own pose,
    the visible edge geometry and the shared static map.  ``reference_x`` does not depend on
    the object pose, so the same skill resolves to the **same world contact point** in both
    worlds of a pair.
    """

    half = params.object_thick / 2.0
    y = -half if robot_id == 0 else half
    return (round(params.reference_x + offset - params.object_centre_x, 6), round(y, 6))


def skill_world_contact(params: MechanismParams, offset: float) -> float:
    """The single world x that a skill with reference offset ``offset`` touches."""

    return params.reference_x + offset


def push_on_edge(params: MechanismParams, robot_id: int, offset: float,
                 direction: Vec, force_limit: float) -> PushContact:
    return PushContact(
        object_id=0,
        contact_local=edge_contact(params, robot_id, offset),
        push_direction_world=direction,
        force_limit=force_limit,
    )


def _skill(skill_id: str, label: str, joint: JointSpec, **provenance) -> NamedJointSkill:
    return NamedJointSkill(skill_id=skill_id, label=label, joint=joint, provenance=provenance)


# ---------------------------------------------------------------------------
# set 1: two-robot symmetric (translation) and antisymmetric (spin) couples
# ---------------------------------------------------------------------------


def symmetric_action_set(params: MechanismParams, *, force_limit: float,
                         offsets: Sequence[float] = (-0.8, 0.0, 0.8)) -> List[NamedJointSkill]:
    """Six to eight joint skills: translation attempts and spin couples.

    ``along_*``: both robots push the same tangential way (rigid translation, zero net torque
    when the contacts are symmetric about the centre of mass).
    ``spin_*``: the robots push the same way *perpendicular* to the edge on opposite sides
    (a pure couple: zero net force, torque 2*d*F).
    ``press_*``: both push inward (opposing normals: near-zero net force, small torque).
    """

    skills: List[NamedJointSkill] = []
    for index, offset in enumerate(offsets):
        skills.append(_skill(
            f"along_pos_{index}", f"both +x at d={offset:+.2f}",
            (push_on_edge(params, 0, offset, (1.0, 0.0), force_limit),
             push_on_edge(params, 1, offset, (1.0, 0.0), force_limit)),
            offset=offset, kind="along_pos"))
    for index, offset in enumerate(offsets):
        skills.append(_skill(
            f"along_neg_{index}", f"both -x at d={offset:+.2f}",
            (push_on_edge(params, 0, offset, (-1.0, 0.0), force_limit),
             push_on_edge(params, 1, offset, (-1.0, 0.0), force_limit)),
            offset=offset, kind="along_neg"))
    skills.append(_skill(
        "spin_pos", "lower +y / upper -y at d=0",
        (push_on_edge(params, 0, 0.0, (0.0, 1.0), force_limit),
         push_on_edge(params, 1, 0.0, (0.0, -1.0), force_limit)),
        offset=0.0, kind="spin_pos"))
    skills.append(_skill(
        "spin_neg", "lower -y / upper +y at d=0",
        (push_on_edge(params, 0, 0.0, (0.0, -1.0), force_limit),
         push_on_edge(params, 1, 0.0, (0.0, 1.0), force_limit)),
        offset=0.0, kind="spin_neg"))
    return skills


# ---------------------------------------------------------------------------
# set 2: one-sided pushes (teammate holds)
# ---------------------------------------------------------------------------


def one_sided_action_set(params: MechanismParams, *, force_limit: float,
                         offsets: Sequence[float] = (-0.8, 0.0, 0.8)) -> List[NamedJointSkill]:
    skills: List[NamedJointSkill] = []
    for index, offset in enumerate(offsets):
        skills.append(_skill(
            f"lower_y_{index}", f"lower +y at d={offset:+.2f}, teammate holds",
            (push_on_edge(params, 0, offset, (0.0, 1.0), force_limit), Hold()),
            offset=offset, kind="lower_y"))
    for index, offset in enumerate(offsets):
        skills.append(_skill(
            f"upper_y_{index}", f"upper -y at d={offset:+.2f}, teammate holds",
            (Hold(), push_on_edge(params, 1, offset, (0.0, -1.0), force_limit)),
            offset=offset, kind="upper_y"))
    skills.append(_skill(
        "lower_x", "lower +x at d=0, teammate holds",
        (push_on_edge(params, 0, 0.0, (1.0, 0.0), force_limit), Hold()),
        offset=0.0, kind="lower_x"))
    skills.append(_skill(
        "upper_x", "upper +x at d=0, teammate holds",
        (Hold(), push_on_edge(params, 1, 0.0, (1.0, 0.0), force_limit)),
        offset=0.0, kind="upper_x"))
    return skills


# ---------------------------------------------------------------------------
# information actions: a short move along the edge before pushing
# ---------------------------------------------------------------------------


def move_along_edge(params: MechanismParams, robot_id: int, offset: float,
                    force_limit: float) -> MoveTo:
    """A legal short re-position of one robot along the visible edge (an information action)."""

    from .mechanism_scene import ROBOT_R

    lower, upper = params.edges()
    y = (lower - ROBOT_R - 0.03) if robot_id == 0 else (upper + ROBOT_R + 0.03)
    target_x = params.object_centre_x + offset
    return MoveTo(target=(round(target_x, 6), round(y, 6)))
