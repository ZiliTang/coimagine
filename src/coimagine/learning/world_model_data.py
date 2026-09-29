"""Collect rollout trajectories from the gridworld for world-model training.

Each trajectory is a contiguous run of simulator steps, recorded from **robot A's own point of
view**: its scan, its own feedback history and the public event history. Nothing privileged is
an input; the object's cell and the contact physics are recorded as decoder targets only.

Actions are drawn from a mixture that is deliberately not uniform: pushing and co-pushing are
over-sampled, because a uniform policy spends most of its steps failing to reach the object and
the interesting transitions -- contact, slipping, the object actually moving -- would be rare.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import List, Optional, Sequence

import numpy as np

from ..grid.dynamics import execute
from ..grid.planner import PHYSICS
from ..grid.sensing import SensorNoise, apply_noise, ray_cast
from ..grid.state import HiddenPhysics, RobotState, Scene, enumerate_hidden_physics
from .data import _joint_for, generate_situations, pose_to_index, probe_scene, valid_origins
from .features import ACTION_IDS, observation_vector
from .world_model import Trajectory, Transition

#: Action mixture: contact-forming actions dominate.
ACTION_MIXTURE = (
    "push:0:E", "push:0:E", "push:1:E", "co_push:E", "co_push:E", "co_push:E",
    "probe:0:E", "release:0:E", "wait",
)


def _observation(scene: Scene, history_a, public_history, rng, radius: int, noise: SensorNoise):
    scan = apply_noise(ray_cast(scene, 0, radius=radius), noise, rng)
    return np.asarray(
        observation_vector(scan, scene.robot_pose(0), tuple(history_a), tuple(public_history))
    )


def collect_trajectories(
    *,
    n_episodes: int = 120,
    horizon: int = 8,
    seed: int = 0,
    radius: int = 5,
    scan_noise: float = 0.0,
    lead_in: int = 2,
    physics_pool: Optional[Sequence[HiddenPhysics]] = None,
) -> List[Trajectory]:
    """Roll out episodes and record every transition from robot A's viewpoint."""

    situations = generate_situations(
        n_scenes=n_episodes, lead_in=lead_in, seed=seed, physics_pool=physics_pool
    )
    noise = SensorNoise(drop_prob=scan_noise)
    rng = random.Random(seed + 1)
    np_rng = np.random.default_rng(seed)

    trajectories: List[Trajectory] = []
    for situation in situations:
        scene = situation.scene
        history_a = list(situation.history_a)
        public_history = list(situation.public_history)
        trajectory = Trajectory()
        for _ in range(horizon):
            if scene.is_goal():
                trajectory.goal_reached = True
                break
            observation = _observation(scene, history_a, public_history, np_rng, radius, noise)
            action_id = rng.choice(ACTION_MIXTURE)
            action_index = ACTION_IDS.index(action_id)
            successor, result = execute(scene, _joint_for(scene, action_id))
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            next_observation = _observation(
                successor, history_a, public_history, np_rng, radius, noise
            )
            trajectory.transitions.append(
                Transition(
                    observation=observation,
                    action=action_index,
                    next_observation=next_observation,
                    moved=int(result.y_public == "moved"),
                    cost=float(result.cost),
                    physics=PHYSICS.index(scene.physics),
                    object_cell=pose_to_index(scene.obj.origin),
                    next_physics=PHYSICS.index(successor.physics),
                    next_object_cell=pose_to_index(successor.obj.origin),
                    released=int(scene.released),
                    next_released=int(successor.released),
                )
            )
            scene = successor
        if trajectory.transitions:
            trajectories.append(trajectory)
    return trajectories


def collect_evaluation_pairs(
    *,
    n_episodes: int = 60,
    seed: int = 0,
    radius: int = 5,
    lead_in: int = 2,
) -> List[Trajectory]:
    """A small held-out set drawn from a disjoint seed range."""

    return collect_trajectories(
        n_episodes=n_episodes, horizon=8, seed=seed + 90_000, radius=radius, lead_in=lead_in
    )
