"""Dataset generation for the learning probe.

Each sample is one *(situation, candidate action)* pair:

* **inputs** -- the robot's own ray scan, its own feedback history, public poses, the
  public action template. Nothing privileged.
* **labels** -- whether the action actually works in the simulator (the feasibility
  target), plus privileged decoder targets: the true object placement and the true
  contact physics.

The privileged labels are what let us ask the two questions that matter:

1. does the model predict feasibility, including when the cause is an invisible
   contact mode?
2. does its latent carry the contact mode at all, or only the geometry?
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..grid import scene_from_ascii
from ..grid.dynamics import (
    FEEDBACK_SYMBOLS,
    FeedbackNoise,
    SKILL_HOLD,
    SKILL_PROBE,
    SKILL_PUSH,
    SKILL_RELEASE,
    SKILL_WAIT,
    JointSkill,
    LocalSkill,
    execute,
)
from ..grid.planner import PHYSICS
from ..grid.sensing import OccupancyScan, SensorNoise, apply_noise, ray_cast
from ..grid.state import Cell, Direction, HiddenPhysics, Scene, enumerate_hidden_physics
from .features import ACTION_IDS, sample_vector

#: Grid side length used by the probe scenes.
GRID = 6

#: Valid object placements for the probe scene, i.e. free 2x2 blocks.
#: Columns 4 and 5 are the goal region, so the object starts at x <= 3.
TARGET_COLUMNS = (GRID - 2, GRID - 1)


def valid_origins(scene: Scene) -> Tuple[Cell, ...]:
    """Object placements for which both robots can stand on the west face.

    Staging the robots at the object's west face is deliberate: it removes contact
    geometry from the probe so that the only thing left to explain is the invisible
    contact physics.
    """

    origins: List[Cell] = []
    for x in range(1, GRID - 3):
        for y in range(GRID - 1):
            if x + 1 >= GRID or y + 1 >= GRID:
                continue
            cells = {(x + dx, y + dy) for dx in (0, 1) for dy in (0, 1)}
            if any(scene.static.is_wall(c) for c in cells):
                continue
            if cells & scene.static.target:
                continue
            origins.append((x, y))
    return tuple(origins)


def probe_scene(origin: Cell, physics: HiddenPhysics) -> Scene:
    """The object at ``origin``, the goal to the east, both robots on its west face."""

    grid = [["."] * GRID for _ in range(GRID)]
    for dx in (0, 1):
        for dy in (0, 1):
            grid[origin[1] + dy][origin[0] + dx] = "O"
    for y in range(GRID):
        for x in TARGET_COLUMNS:
            grid[y][x] = "T"
    grid[origin[1]][origin[0] - 1] = "A"
    grid[origin[1] + 1][origin[0] - 1] = "B"
    rows = ["".join(grid[y]) for y in range(GRID - 1, -1, -1)]
    return scene_from_ascii(rows, name=f"probe_{origin[0]}_{origin[1]}", hidden=physics)


#: The pushing actions used when asking whether an action enabled anything.
PUSH_ACTIONS: Tuple[str, ...] = ("push:0:E", "push:1:E", "co_push:E")


def _min_distance_to_target(scene: Scene) -> int:
    return min(
        abs(c[0] - t[0]) + abs(c[1] - t[1])
        for c in scene.obj.cells
        for t in scene.static.target
    )


def action_labels(scene: Scene, action_id: str) -> Tuple[int, int, int]:
    """The three supervised outcomes of taking ``action_id`` in ``scene``.

    1. **moved** -- the object physically moved this step;
    2. **progress** -- the object ended *closer* to the goal region than it started,
       which is the thing a planner actually wants, and which stops a greedy rule from
       pushing the object straight past the goal;
    3. **enables** -- some push that failed before this action now succeeds, which is
       how a preparation action such as release earns its place even though it does not
       move anything itself.
    """

    joint = _joint_for(scene, action_id)
    successor, result = execute(scene, joint)
    moved = int(result.y_public == "moved")
    before = _min_distance_to_target(scene)
    after = _min_distance_to_target(successor)
    progress = int(after < before)

    enables = 0
    for push in PUSH_ACTIONS:
        _, before_result = execute(scene, _joint_for(scene, push))
        if before_result.y_public == "moved":
            continue
        _, after_result = execute(successor, _joint_for(successor, push))
        if after_result.y_public == "moved":
            enables = 1
            break
    return moved, progress, enables


@dataclass(frozen=True)
class FeasibilitySample:
    """One (situation, action) pair with its labels."""

    features: Tuple[float, ...]
    feasible: int
    object_origin: Cell
    physics: HiddenPhysics
    action_id: str
    moved: int
    scene_id: int = 0
    progress: int = 0
    enables: int = 0


@dataclass(frozen=True)
class Situation:
    """A decision point: the state after a lead-in, plus what the robot observed."""

    scene: Scene                      # the state *at* the decision point
    initial_scene: Scene              # the state before the lead-in
    skills: Tuple[JointSkill, ...]    # the lead-in actions that were executed
    history_a: Tuple[str, ...]
    history_b: Tuple[str, ...]
    physics: HiddenPhysics
    scan: OccupancyScan
    scene_id: int


def generate_situations(
    *,
    n_scenes: int = 200,
    lead_in: int = 2,
    radius: int = 5,
    scan_noise: float = 0.0,
    feedback_noise: float = 0.0,
    seed: int = 0,
    physics_pool: Optional[Sequence[HiddenPhysics]] = None,
    lead_in_actions: Sequence[str] = ("push:0:E", "push:1:E", "co_push:E", "probe:0:E"),
) -> List[Situation]:
    """Decision points reached by driving each scene through a real lead-in episode.

    ``lead_in_actions`` deliberately excludes ``release`` so that the released flag is
    constant across a situation; that keeps the equal-information baseline honest
    without inventing a release tracker.
    """

    rng = random.Random(seed)
    pool = tuple(physics_pool) if physics_pool is not None else enumerate_hidden_physics()
    scan_noise_model = SensorNoise(drop_prob=scan_noise)
    fb_noise = FeedbackNoise(flip_prob=feedback_noise)
    out: List[Situation] = []

    built = 0
    attempts = 0
    while built < n_scenes and attempts < n_scenes * 40:
        attempts += 1
        physics = rng.choice(pool)
        origins = valid_origins(probe_scene((2, 2), physics))
        if not origins:
            continue
        scene = probe_scene(rng.choice(origins), physics)
        built += 1

        current = scene
        history_a: List[str] = []
        history_b: List[str] = []
        skills: List[JointSkill] = []
        for _ in range(lead_in):
            action_id = rng.choice(tuple(lead_in_actions))
            joint = _joint_for(current, action_id)
            skills.append(joint)
            successor, result = execute(current, joint, noise=fb_noise, rng=rng)
            history_a.append(result.feedback[0])
            history_b.append(result.feedback[1])
            current = successor
            if current.is_goal():
                break
        scan = apply_noise(ray_cast(current, 0, radius=radius), scan_noise_model, rng)
        out.append(
            Situation(
                scene=current,
                initial_scene=scene,
                skills=tuple(skills),
                history_a=tuple(history_a),
                history_b=tuple(history_b),
                physics=physics,
                scan=scan,
                scene_id=built - 1,
            )
        )
    return out


def candidates_for(action_ids: Sequence[str] = ACTION_IDS):
    """Build the candidate set for the probe's action template."""

    from ..grid.candidates import Candidate, CandidateSet

    return CandidateSet(tuple(Candidate(a, _joint_for(None, a)) for a in action_ids))


@dataclass
class Dataset:
    samples: List[FeasibilitySample] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.samples)

    def physics_labels(self) -> Tuple[HiddenPhysics, ...]:
        return tuple(s.physics for s in self.samples)

    def filter(self, predicate) -> "Dataset":
        return Dataset([s for s in self.samples if predicate(s)])

    def take(self, count: int) -> "Dataset":
        return Dataset(self.samples[:count])


def generate_dataset(
    *,
    n_scenes: int = 60,
    lead_in: int = 2,
    radius: int = 5,
    scan_noise: float = 0.0,
    feedback_noise: float = 0.0,
    seed: int = 0,
    physics_pool: Optional[Sequence[HiddenPhysics]] = None,
) -> Dataset:
    """Roll out a labelled dataset of (situation, action) pairs.

    Each scene is first driven through a short **lead-in** episode of real actions, so
    the robots accumulate genuine contact feedback before the samples are taken. That
    is what makes the invisible physics inferable at all: on a fresh scene nothing has
    touched the object yet, so no model could tell a heavy object from a light one.

    The scan is the robot's own; each feasibility label comes from executing the action
    in the simulator with the true physics.
    """

    rng = random.Random(seed)
    pool = tuple(physics_pool) if physics_pool is not None else enumerate_hidden_physics()
    scan_noise_model = SensorNoise(drop_prob=scan_noise)
    fb_noise = FeedbackNoise(flip_prob=feedback_noise)
    samples: List[FeasibilitySample] = []

    scenes_built = 0
    attempts = 0
    while scenes_built < n_scenes and attempts < n_scenes * 40:
        attempts += 1
        physics = rng.choice(pool)
        base = probe_scene((2, 2), physics)
        origins = valid_origins(base)
        if not origins:
            continue
        origin = rng.choice(origins)
        scene = probe_scene(origin, physics)
        scenes_built += 1

        current = scene
        history_a: List[str] = []
        history_b: List[str] = []
        for _ in range(lead_in):
            action_id = rng.choice(ACTION_IDS)
            successor, result = execute(
                current, _joint_for(current, action_id), noise=fb_noise, rng=rng
            )
            history_a.append(result.feedback[0])
            history_b.append(result.feedback[1])
            current = successor
            if current.is_goal():
                break

        scan = apply_noise(ray_cast(current, 0, radius=radius), scan_noise_model, rng)
        for action_id in ACTION_IDS:
            moved, progress, enables = action_labels(current, action_id)
            samples.append(
                FeasibilitySample(
                    features=sample_vector(
                        scan, current.robot_pose(0), tuple(history_a), tuple(history_b), action_id
                    ),
                    feasible=moved,
                    object_origin=current.obj.origin,
                    physics=physics,
                    action_id=action_id,
                    moved=moved,
                    scene_id=scenes_built - 1,
                    progress=progress,
                    enables=enables,
                )
            )
    return Dataset(samples)


def _joint_for(scene: Optional[Scene], action_id: str) -> JointSkill:
    """Build the joint skill for a candidate id."""

    east = Direction.EAST
    hold = LocalSkill(SKILL_HOLD)
    if action_id == "push:0:E":
        return JointSkill((LocalSkill(SKILL_PUSH, east), hold))
    if action_id == "push:1:E":
        return JointSkill((hold, LocalSkill(SKILL_PUSH, east)))
    if action_id == "co_push:E":
        return JointSkill((LocalSkill(SKILL_PUSH, east), LocalSkill(SKILL_PUSH, east)))
    if action_id == "probe:0:E":
        return JointSkill((LocalSkill(SKILL_PROBE, east), hold))
    if action_id == "release:0:E":
        return JointSkill((LocalSkill(SKILL_RELEASE, east), hold))
    if action_id == "wait":
        wait = LocalSkill(SKILL_WAIT)
        return JointSkill((wait, wait))
    raise ValueError(action_id)


def sample_to_arrays(samples: Sequence[FeasibilitySample]):
    """Split a sample list into input matrix, feasibility labels and aux labels."""

    import numpy as np

    if not samples:
        raise ValueError("no samples")
    x = np.asarray([s.features for s in samples], dtype=np.float64)
    y = np.asarray([s.feasible for s in samples], dtype=np.float64)
    targets = np.asarray(
        [[s.moved, s.progress, s.enables] for s in samples], dtype=np.float64
    )
    physics_index = np.asarray([PHYSICS.index(s.physics) for s in samples], dtype=np.int64)
    return x, y, physics_index, targets


def pose_to_index(origin: Cell, grid: int = GRID) -> int:
    """Flattened cell index used as the pose decoder's target."""

    return origin[1] * grid + origin[0]
