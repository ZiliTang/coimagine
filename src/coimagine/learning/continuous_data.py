"""Data for the continuous-parameter probe.

Each sample is an (observation, push direction, force) triple labelled with whether the
push actually works. The observation is the robot's own scan and feedback history only.
The labels include the privileged contact physics, used as decoder targets, so the probe
can ask whether the latent carries the mode that sets the feasible region's boundary.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.continuous import FORCE_MAX, push_outcome
from ..grid.dynamics import FEEDBACK_SYMBOLS, FeedbackNoise
from ..grid.planner import PHYSICS, _execute_cached
from ..grid.sensing import OccupancyScan, SensorNoise, apply_noise, ray_cast
from ..grid.state import Cell, HiddenPhysics, Scene, enumerate_hidden_physics
from .data import _joint_for, pose_to_index, probe_scene, valid_origins
from .features import CELL_KINDS, FEEDBACK_SYMBOLS as _SYMBOLS, GRID, OBSERVATION_DIM
from .features import feedback_counts

#: Default grids. The angular grid spans the east Voronoi cell plus one outside sample on
#: each side, so a model that never learned the window edge is visible as an error.
DEFAULT_ANGLES: Tuple[float, ...] = tuple(float(a) for a in range(-60, 61, 10))
DEFAULT_FORCES: Tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0)

#: Continuous action encoding appended to the observation features.
CONTINUOUS_ACTION_DIM = 3


@dataclass(frozen=True)
class ContinuousSample:
    features: Tuple[float, ...]
    moved: int
    angle: float
    force: float
    physics: HiddenPhysics
    object_origin: Cell
    scene_id: int


def continuous_action_vector(angle: float, force: float) -> Tuple[float, float, float]:
    import math

    return (
        math.sin(math.radians(angle)),
        math.cos(math.radians(angle)),
        force / FORCE_MAX,
    )


def continuous_features(
    scan: OccupancyScan,
    own_pose: Cell,
    feedback_a: Sequence[str],
    feedback_b: Sequence[str],
    angle: float,
    force: float,
) -> Tuple[float, ...]:
    from .features import observation_vector

    return (
        observation_vector(scan, own_pose, feedback_a, feedback_b)
        + continuous_action_vector(angle, force)
    )


def continuous_feature_dim() -> int:
    return OBSERVATION_DIM + CONTINUOUS_ACTION_DIM


def generate_continuous_dataset(
    *,
    n_scenes: int = 120,
    lead_in: int = 2,
    radius: int = 5,
    angles: Sequence[float] = DEFAULT_ANGLES,
    forces: Sequence[float] = DEFAULT_FORCES,
    seed: int = 0,
    physics_pool: Optional[Sequence[HiddenPhysics]] = None,
    feedback_noise: float = 0.0,
) -> List[ContinuousSample]:
    """Scenes driven through a short lead-in, then labelled over the (angle, force) grid."""

    rng = random.Random(seed)
    pool = tuple(physics_pool) if physics_pool is not None else enumerate_hidden_physics()
    fb_noise = FeedbackNoise(flip_prob=feedback_noise)
    samples: List[ContinuousSample] = []

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
        for _ in range(lead_in):
            action_id = rng.choice(("push:0:E", "push:1:E", "co_push:E", "probe:0:E"))
            successor, result = _execute_cached(
                current, _joint_for(current, action_id)
            )
            history_a.append(result.feedback[0])
            history_b.append(result.feedback[1])
            current = successor
            if current.is_goal():
                break

        scan = apply_noise(ray_cast(current, 0, radius=radius), SensorNoise(), rng)
        for angle in angles:
            for force in forces:
                outcome = push_outcome(current, {0: force}, angle)
                samples.append(
                    ContinuousSample(
                        features=continuous_features(
                            scan, current.robot_pose(0), tuple(history_a), tuple(history_b),
                            angle, force,
                        ),
                        moved=int(outcome.moved),
                        angle=angle,
                        force=force,
                        physics=physics,
                        object_origin=current.obj.origin,
                        scene_id=built - 1,
                    )
                )
    return samples


def to_arrays(samples: Sequence[ContinuousSample]):
    """Split into observation features, continuous action inputs, moved labels and aux labels."""

    x = np.asarray([s.features for s in samples], dtype=np.float64)
    obs = x[:, :OBSERVATION_DIM]
    action = x[:, OBSERVATION_DIM:]
    y = np.asarray([[s.moved] for s in samples], dtype=np.float64)
    physics_index = np.asarray([PHYSICS.index(s.physics) for s in samples], dtype=np.int64)
    pose_index = np.asarray([pose_to_index(s.object_origin) for s in samples], dtype=np.int64)
    return obs, action, y, physics_index, pose_index
