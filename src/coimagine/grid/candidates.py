"""Public joint-candidate generation for Stage 2.

The candidate **ID space** is a deterministic function of public information only:
the prior static map and the (publicly known) robot poses. It never depends on the
object's estimated pose or on the hidden physics. That is what lets both robots
agree on the same IDs and the same commitment rule while scoring the candidates with
**different private beliefs** (``docs/2d_protocol_spec.md`` section 9).

A candidate is a *template*: a joint skill whose executability is checked at
execution time. A robot that believes the object is elsewhere may still commit a
push; the boundary then fails to move the object and the cost is charged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

from .dynamics import (
    SKILL_HOLD,
    SKILL_PROBE,
    SKILL_PUSH,
    SKILL_RELEASE,
    SKILL_REPOSITION,
    SKILL_WAIT,
    JointSkill,
    LocalSkill,
)
from .state import Cell, Direction, Scene, StaticMap, manhattan

DIRECTIONS: Tuple[Direction, ...] = (
    Direction.NORTH,
    Direction.EAST,
    Direction.SOUTH,
    Direction.WEST,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    joint: JointSkill

    @property
    def kind(self) -> str:
        return self.candidate_id.split(":", 1)[0]


@dataclass(frozen=True)
class CandidateSet:
    candidates: Tuple[Candidate, ...]

    def __post_init__(self) -> None:
        ids = [c.candidate_id for c in self.candidates]
        if len(ids) != len(set(ids)):
            raise ValueError("Candidate IDs must be unique.")

    def ids(self) -> Tuple[str, ...]:
        return tuple(c.candidate_id for c in self.candidates)

    def by_id(self, candidate_id: str) -> Candidate:
        for candidate in self.candidates:
            if candidate.candidate_id == candidate_id:
                return candidate
        raise KeyError(candidate_id)

    def __len__(self) -> int:
        return len(self.candidates)


def _distance_to_goal(static: StaticMap, cell: Cell) -> int:
    if not static.target:
        centre = (static.width // 2, static.height // 2)
        return manhattan(cell, centre)
    return min(manhattan(cell, goal) for goal in static.target)


def public_anchor_cells(
    static: StaticMap, robot_poses: Sequence[Cell], *, count: int = 4
) -> Tuple[Cell, ...]:
    """A small deterministic set of reposition anchors from public information."""

    occupied = frozenset(robot_poses)
    free = [
        cell
        for cell in sorted(static.free_cells())
        if cell not in occupied
    ]
    free.sort(key=lambda c: (_distance_to_goal(static, c), c[1], c[0]))
    if count <= 0 or len(free) <= count:
        return tuple(free)
    stride = len(free) / float(count)
    picked: List[Cell] = []
    for index in range(count):
        cell = free[min(int(index * stride), len(free) - 1)]
        if cell not in picked:
            picked.append(cell)
    return tuple(picked)


def generate_candidates(
    static: StaticMap,
    robot_poses: Sequence[Cell],
    *,
    max_reposition_targets: int = 4,
) -> CandidateSet:
    """The public joint-candidate set for one decision boundary."""

    if len(robot_poses) != 2:
        raise ValueError("Stage 2 is fixed at two robots.")

    candidates: List[Candidate] = []

    # Same-direction cooperative push: the only way to reach capacity 2.
    for direction in DIRECTIONS:
        joint = JointSkill(
            (LocalSkill(SKILL_PUSH, direction), LocalSkill(SKILL_PUSH, direction))
        )
        candidates.append(Candidate(f"co_push:{direction.label}", joint))

    for robot_id in (0, 1):
        other = 1 - robot_id
        for direction in DIRECTIONS:
            push_skills = [LocalSkill(SKILL_HOLD), LocalSkill(SKILL_HOLD)]
            push_skills[robot_id] = LocalSkill(SKILL_PUSH, direction)
            candidates.append(
                Candidate(f"push:{robot_id}:{direction.label}", JointSkill(tuple(push_skills)))
            )

            probe_skills = [LocalSkill(SKILL_HOLD), LocalSkill(SKILL_HOLD)]
            probe_skills[robot_id] = LocalSkill(SKILL_PROBE, direction)
            candidates.append(
                Candidate(f"probe:{robot_id}:{direction.label}", JointSkill(tuple(probe_skills)))
            )

            release_skills = [LocalSkill(SKILL_HOLD), LocalSkill(SKILL_HOLD)]
            release_skills[robot_id] = LocalSkill(SKILL_RELEASE, direction)
            candidates.append(
                Candidate(
                    f"release:{robot_id}:{direction.label}", JointSkill(tuple(release_skills))
                )
            )

    candidates.append(
        Candidate("wait", JointSkill((LocalSkill(SKILL_WAIT), LocalSkill(SKILL_WAIT))))
    )

    for anchor in public_anchor_cells(
        static, robot_poses, count=max_reposition_targets
    ):
        for robot_id in (0, 1):
            skills = [LocalSkill(SKILL_HOLD), LocalSkill(SKILL_HOLD)]
            skills[robot_id] = LocalSkill(SKILL_REPOSITION, target=anchor)
            candidates.append(
                Candidate(
                    f"reposition:{robot_id}:{anchor[0]},{anchor[1]}",
                    JointSkill(tuple(skills)),
                )
            )

    candidates.sort(key=lambda c: c.candidate_id)
    return CandidateSet(tuple(candidates))


def generate_candidates_for_scene(scene: Scene, **kwargs) -> CandidateSet:
    return generate_candidates(
        scene.static, tuple(r.pose for r in scene.robots), **kwargs
    )
