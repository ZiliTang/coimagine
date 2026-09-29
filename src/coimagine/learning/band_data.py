"""Collect a **band** of data rather than a line: a solving trajectory with all actions fanned out.

The planner does not ask "what happens if I follow the good policy". At every state it visits it
asks **"what happens if I do each of the six candidate actions here"** -- and it acts on the answer
for the five it rejects. Data collected by following one good policy answers only the first
question. Data collected by following an i.i.d. action mixture, which is what the repository had,
answers it for states the good policy never reaches and leaves the productive regime almost empty:
the object actually moves in 6.4% of those transitions, and the pattern the search evaluates -- four
consecutive pushes with at least two movements -- occurs 3 times in 1687 windows.

What the planner's *query distribution* actually is, therefore:

    (states a competent policy visits)  x  (all six actions at each of those states)

This module collects exactly that. The trajectory advances only along actions a privileged scripted
policy has verified to make progress, so the states stay productive and movement-rich; and at every
one of those states all six actions are executed in a copy of the world and recorded. The cost is six
times the simulation, and it covers the good actions along the trajectory *and* the counterfactual
bad ones at the same states.

The exploratory mixture is still worth keeping at some fraction: the planner also needs to know what
happens in the null regime, and most candidate actions still do nothing.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence

import numpy as np

from ..grid.dynamics import execute
from ..grid.planner import PHYSICS
from ..grid.sensing import SensorNoise, apply_noise, ray_cast
from ..grid.state import HiddenPhysics, Scene
from .data import _joint_for, generate_situations, pose_to_index
from .features import ACTION_IDS, observation_vector
from .world_model import Trajectory, Transition
from .world_model_data import _observation

#: Actions a privileged policy will try in order when looking for one that advances the task.
ADVANCING_CANDIDATES = ("co_push:E", "push:0:E", "push:1:E")


def solving_action(scene: Scene) -> str:
    """A privileged scripted policy: whichever action demonstrably advances the task.

    Deliberately verified by simulation rather than reasoned about, so it cannot be wrong about the
    geometry: if no push moves the object, release if that changes anything, otherwise wait. Used
    only to *place* the trajectory in the productive regime; the labels all come from the simulator.
    """

    for action_id in ADVANCING_CANDIDATES:
        successor, _result = execute(scene, _joint_for(scene, action_id))
        if successor.obj.origin != scene.obj.origin:
            return action_id
    successor, _result = execute(scene, _joint_for(scene, "release:0:E"))
    if successor.released and not scene.released:
        return "release:0:E"
    return "wait"


def _transition_from(scene, successor, result, observation, next_observation, action_index):
    return Transition(
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
        feedback_symbol=result.feedback[0],
        robot_poses=tuple(float(c) for robot in scene.robots for c in robot.pose),
        next_robot_poses=tuple(float(c) for robot in successor.robots for c in robot.pose),
    )


def collect_band_trajectories(
    *,
    n_episodes: int = 120,
    horizon: int = 8,
    seed: int = 0,
    radius: int = 5,
    scan_noise: float = 0.0,
    lead_in: int = 2,
    physics_pool: Optional[Sequence[HiddenPhysics]] = None,
    fan_out: bool = True,
) -> List[Trajectory]:
    """Follow a solving trajectory, recording all six actions' outcomes at every state."""

    situations = generate_situations(
        n_scenes=n_episodes, lead_in=lead_in, seed=seed, physics_pool=physics_pool
    )
    noise = SensorNoise(drop_prob=scan_noise)
    np_rng = np.random.default_rng(seed)

    trajectories: List[Trajectory] = []
    for situation in situations:
        scene = situation.scene
        history_a = list(situation.history_a)
        public_history = list(situation.public_history)
        # Two records come out of the same walk, and keeping them separate matters:
        #
        #   * the **advancing chain** -- the transitions along the solving policy, in temporal
        #     order. This is the only place consecutive successful pushes appear, and the
        #     multi-step rollout loss needs them to be contiguous.
        #   * the **counterfactual fan-out** -- the rejected actions at the same states, as
        #     one-step records. These teach the model what the bad actions do, which is what the
        #     planner asks about when it enumerates candidates.
        #
        # An earlier version mixed both into one flat list, which destroyed the temporal order:
        # four consecutive pushes could never appear, because five other actions were interleaved
        # between every pair of them.
        chain = Trajectory()
        counterfactual: List[Trajectory] = []
        for _ in range(horizon):
            if scene.is_goal():
                chain.goal_reached = True
                break
            observation = _observation(scene, history_a, public_history, np_rng, radius, noise)
            advancing = solving_action(scene)
            for action_id in ACTION_IDS:
                successor, result = execute(scene, _joint_for(scene, action_id))
                next_history = history_a + [result.feedback[0]]
                next_public = public_history + [result.y_public]
                next_observation = _observation(
                    successor, next_history, next_public, np_rng, radius, noise
                )
                record = _transition_from(
                    scene, successor, result, observation, next_observation,
                    ACTION_IDS.index(action_id),
                )
                if action_id == advancing:
                    chain.transitions.append(record)
                elif fan_out:
                    counterfactual.append(Trajectory(transitions=[record]))
            successor, result = execute(scene, _joint_for(scene, advancing))
            history_a.append(result.feedback[0])
            public_history.append(result.y_public)
            scene = successor
        if chain.transitions:
            trajectories.append(chain)
        trajectories.extend(counterfactual)
    return trajectories


def band_statistics(
    trajectories: Sequence[Trajectory], *, chains_only: bool = False
) -> dict:
    """The composition of a collected set, for reporting rather than for scoring.

    ``chains_only`` restricts the count to trajectories longer than one step, i.e. the advancing
    chains rather than the one-step counterfactual fan-out. The two have different jobs -- the
    chains carry the multi-step temporal signal, the fan-out teaches the rejected actions -- so
    reporting them together hides whether the chains are actually productive.
    """

    if chains_only:
        trajectories = [t for t in trajectories if len(t.transitions) > 1]
    steps = [s for t in trajectories for s in t.transitions]
    if not steps:
        return {"transitions": 0}
    moved = sum(s.moved for s in steps)
    pushes = sum(
        1 for s in steps if ACTION_IDS[s.action] in ("push:0:E", "push:1:E", "co_push:E")
    )
    # windows of four consecutive pushes with at least two movements: the pattern the search
    # evaluates when it imagines driving the object to the goal
    windows = productive = 0
    push_indices = {ACTION_IDS.index(a) for a in ("push:0:E", "push:1:E", "co_push:E")}
    for trajectory in trajectories:
        actions = [s.action for s in trajectory.transitions]
        moved_flags = [s.moved for s in trajectory.transitions]
        for start in range(len(actions) - 3):
            windows += 1
            if all(actions[start + k] in push_indices for k in range(4)) and sum(
                moved_flags[start : start + 4]
            ) >= 2:
                productive += 1
    return {
        "transitions": len(steps),
        "moved_fraction": moved / len(steps),
        "push_fraction": pushes / len(steps),
        "productive_4windows": productive,
        "windows": windows,
        "productive_fraction": productive / max(1, windows),
    }


def mixed_dataset(
    band: Sequence[Trajectory],
    exploratory: Sequence[Trajectory],
    *,
    band_repeats: int = 1,
    exploratory_repeats: int = 1,
) -> List[Trajectory]:
    """The band plus some of the exploratory mixture.

    The band answers the planner's counterfactual questions in the productive regime; the
    exploratory data keeps the null regime populated, because most candidate actions still do
    nothing and the model has to be right about that too.
    """

    out: List[Trajectory] = []
    for _ in range(band_repeats):
        out.extend(band)
    for _ in range(exploratory_repeats):
        out.extend(exploratory)
    return out
