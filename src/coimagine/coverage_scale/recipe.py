"""The fixed per-root collection recipe (task book B1) and its window records.

Every stored transition is a real ``execute_joint_skill`` result of the live environment. The
exact simulator is used **offline only**, to pick which first window the guided trajectory
executes; no exact prediction is ever stored as a supervision label.

Recipe, identical at every scale and for the narrow-coverage control:

* one random trajectory of up to 8 windows from the same legal root;
* one one-step exactly-guided trajectory of up to 8 windows, each window chosen from at most
  16 distinct joint skills of the same root library by the frozen scalar ``J``;
* the last two non-terminal window starts of the guided trajectory become anchors;
* each anchor is branched with 4 distinct first actions (the guided action, both-hold, and two
  uniform root-library draws, topped up from the library in a stable random order); each
  branch runs up to 2 windows, the second repeating with probability 0.5.

That is at most ``8 + 8 + 2 x 4 x 2 = 32`` real windows per root. A trajectory stops on real
success or real failure and never continues from an absorbing state.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.config import CONTROLLER_VERSION, Continuous2DConfig
from ..continuous2d.data import code_version
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.legality import scene_content_hash
from ..continuous2d.scenes import SceneSpec
from ..continuous2d.skills import Hold, JointSpec, joint_spec_to_dicts
from ..continuous2d_learning.contract import termination_class
from ..continuous2d_learning.pools import root_library
from ..continuous2d_learning.rollout import evaluate_sequence_exact
from .sampler import RootSpec

RECIPE_VERSION = "coverage-scale-recipe-v1"
RANDOM_WINDOWS = 8
GUIDE_WINDOWS = 8
GUIDE_K = 16
ANCHORS_PER_ROOT = 2
BRANCHES_PER_ANCHOR = 4
BRANCH_WINDOWS = 2
BRANCH_REPEAT_PROBABILITY = 0.5
MAX_REAL_WINDOWS = RANDOM_WINDOWS + GUIDE_WINDOWS + ANCHORS_PER_ROOT * BRANCHES_PER_ANCHOR * BRANCH_WINDOWS


def _recipe_seed(root_id: str, stream: str) -> int:
    digest = hashlib.sha256(f"{RECIPE_VERSION}|{root_id}|{stream}".encode()).hexdigest()
    return int(digest[:8], 16)


def _spec_key(spec: JointSpec) -> str:
    return json.dumps(joint_spec_to_dicts(spec), sort_keys=True)


def _distinct_specs(candidates, rng: random.Random, count: int) -> List[JointSpec]:
    """At most ``count`` distinct joint skills, in a stable shuffled order."""

    unique: List[JointSpec] = []
    seen = set()
    for candidate in candidates:
        key = _spec_key(candidate.skills)
        if key not in seen:
            seen.add(key)
            unique.append(candidate.skills)
    rng.shuffle(unique)
    return unique[:count]


def _window_record(
    *,
    root: RootSpec,
    scene: SceneSpec,
    episode_id: str,
    seed: int,
    trajectory: str,
    window_index: int,
    joint_spec: JointSpec,
    before,
    result,
    config: Continuous2DConfig,
    candidate_id: Optional[int] = None,
) -> Dict[str, object]:
    contacts = [point for step in result.steps for point in step.contacts]
    had_contact = any(str(point[0]) == "robot_object" for point in contacts)
    penetration_by_pair: Dict[str, float] = {}
    for step in result.steps:
        for key, value in step.penetration_by_pair.items():
            penetration_by_pair[key] = max(penetration_by_pair.get(key, 0.0), float(value))
    record: Dict[str, object] = {
        "schema_version": 3,
        "record_version": RECIPE_VERSION,
        "episode_id": episode_id,
        "root_id": root.root_id,
        "scene_id": scene.name,
        "scene_version": scene.scene_version,
        "scene_content_hash": scene_content_hash(scene),
        "family": root.family,
        "mode": root.mode,
        "split": root.split,
        "seed": seed,
        "trajectory": trajectory,
        "window_index": window_index,
        "t": before.tick,
        "state": {
            "boundary": "before_window",
            "robots": [state.as_tuple() for state in before.robots],
            "object": before.object.as_tuple(),
        },
        "next_state": {
            "boundary": "after_window",
            "robots": [state.as_tuple() for state in result.final.robots],
            "object": result.final.object.as_tuple(),
        },
        "joint_skill_specs": joint_spec_to_dicts(joint_spec),
        "joint_skill_labels": [joint_spec[0].label, joint_spec[1].label],
        "duration": result.duration,
        "duration_steps": len(result.steps),
        "termination": result.termination,
        "termination_class": termination_class(result.termination),
        "success": int(result.success),
        "success_counter_before": before.success_counter,
        "success_counter_after": result.final.success_counter,
        "contact": {
            "had_robot_object_contact": had_contact,
            "max_penetration": max((step.penetration for step in result.steps), default=0.0),
            "penetration_by_pair": penetration_by_pair,
        },
        "cost_components": dict(result.cost_components),
        "config_hash": config.config_hash(),
        "controller_version": CONTROLLER_VERSION,
        "code_version": code_version(),
        "source": {"kind": trajectory, "candidate_id": candidate_id},
    }
    record["transition_id"] = _transition_id(record)
    return record


def _transition_id(record: Dict[str, object]) -> str:
    """Stable content id of one real transition, for exact-duplicate removal."""

    payload = {
        "scene": record["scene_content_hash"],
        "state": record["state"],
        "next_state": record["next_state"],
        "action": record["joint_skill_specs"],
        "termination": record["termination"],
        "success": record["success"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:24]


def _terminal(result) -> bool:
    return result.termination != "window_complete"


def collect_root(
    root: RootSpec,
    config: Continuous2DConfig,
    *,
    live: Optional[Continuous2DEnv] = None,
    planner: Optional[Continuous2DEnv] = None,
) -> Dict[str, object]:
    """Run the frozen recipe on one root and return its real windows plus statistics."""

    scene = root.scene
    live = live or Continuous2DEnv(config)
    planner = planner or Continuous2DEnv(config)
    live.reset(scene)

    windows: List[Dict[str, object]] = []
    guide_trial_steps = 0
    guide_prefix_steps = 0
    guide_clock = 0.0
    started = time.time()

    # ---- random trajectory ------------------------------------------------
    rng = random.Random(_recipe_seed(root.root_id, "random"))
    previous: Optional[JointSpec] = None
    for window_index in range(RANDOM_WINDOWS):
        if live.termination is not None:
            break
        observation = live.observation()
        if window_index == 0 or rng.random() < 0.5:
            spec = rng.choice(root_library(observation, config)).skills
        else:
            spec = previous  # type: ignore[assignment]
        previous = spec
        before = live.export_state()
        result = live.execute_joint_skill(spec)
        windows.append(
            _window_record(
                root=root, scene=scene, episode_id=f"{root.root_id}/random", seed=rng.randrange(2**31),
                trajectory="random", window_index=window_index, joint_spec=spec,
                before=before, result=result, config=config,
            )
        )
        if _terminal(result):
            break

    # ---- one-step exactly guided trajectory --------------------------------
    live.reset(scene)
    rng = random.Random(_recipe_seed(root.root_id, "guide"))
    guided: List[Dict[str, object]] = []
    snapshots: List[object] = []
    observations: List[object] = []
    specs: List[JointSpec] = []
    non_terminal: List[int] = []
    guide_termination = "window_complete"
    no_feasible = 0
    for window_index in range(GUIDE_WINDOWS):
        if live.termination is not None:
            break
        observation = live.observation()
        snapshot = live.export_state()
        options = _distinct_specs(root_library(observation, config), rng, GUIDE_K)
        trial_started = time.time()
        evaluations = [
            evaluate_sequence_exact(planner, snapshot, scene, (spec,), config, candidate_id=index)
            for index, spec in enumerate(options)
        ]
        guide_clock += time.time() - trial_started
        guide_trial_steps += sum(evaluation.physics_steps for evaluation in evaluations)
        guide_prefix_steps += sum(evaluation.prefix_replay_steps for evaluation in evaluations)
        feasible = [index for index, evaluation in enumerate(evaluations) if evaluation.feasible]
        if not options or not feasible:
            guide_termination = "planner_no_feasible_plan"
            no_feasible = 1
            break
        best = max(feasible, key=lambda index: (evaluations[index].j, -index))
        spec = options[best]
        snapshots.append(snapshot)
        observations.append(observation)
        specs.append(spec)
        before = snapshot
        result = live.execute_joint_skill(spec)
        guided.append(
            _window_record(
                root=root, scene=scene, episode_id=f"{root.root_id}/guide", seed=rng.randrange(2**31),
                trajectory="guide", window_index=window_index, joint_spec=spec,
                before=before, result=result, config=config, candidate_id=best,
            )
        )
        if not _terminal(result):
            non_terminal.append(window_index)
        guide_termination = result.termination
        if _terminal(result):
            break
    windows.extend(guided)

    # ---- anchors: the last two non-terminal guided window starts -----------
    anchor_indices = non_terminal[-ANCHORS_PER_ROOT:]
    branch_windows: List[Dict[str, object]] = []
    for anchor_rank, window_index in enumerate(anchor_indices):
        snapshot = snapshots[window_index]
        observation = observations[window_index]
        expert = specs[window_index]
        rng = random.Random(_recipe_seed(root.root_id, f"branch{anchor_rank}"))
        library = root_library(observation, config)
        first_actions: List[JointSpec] = []
        keys = set()

        def _add(spec: JointSpec) -> None:
            key = _spec_key(spec)
            if key not in keys:
                keys.add(key)
                first_actions.append(spec)

        _add(expert)
        _add((Hold(), Hold()))
        attempts = 0
        while len(first_actions) < 2 and attempts < 200 and library:
            attempts += 1
            _add(rng.choice(library).skills)
        shuffled = [_candidate.skills for _candidate in library]
        rng.shuffle(shuffled)
        for spec in shuffled:
            if len(first_actions) >= BRANCHES_PER_ANCHOR:
                break
            _add(spec)
        for branch_id, first in enumerate(first_actions[:BRANCHES_PER_ANCHOR]):
            live.restore(snapshot, exact=True)
            current = first
            for step in range(BRANCH_WINDOWS):
                if live.termination is not None:
                    break
                if step == 0:
                    spec = first
                elif rng.random() < BRANCH_REPEAT_PROBABILITY:
                    spec = current
                else:
                    spec = rng.choice(root_library(live.observation(), config)).skills
                current = spec
                before = live.export_state()
                result = live.execute_joint_skill(spec)
                branch_windows.append(
                    _window_record(
                        root=root, scene=scene,
                        episode_id=f"{root.root_id}/a{window_index}/b{branch_id}",
                        seed=rng.randrange(2**31), trajectory="branch", window_index=step,
                        joint_spec=spec, before=before, result=result, config=config,
                        candidate_id=branch_id,
                    )
                )
                if _terminal(result):
                    break
    windows.extend(branch_windows)

    stats = {
        **root.as_metadata(),
        "scene_content_hash": scene_content_hash(scene),
        "recipe_version": RECIPE_VERSION,
        "windows_total": len(windows),
        "windows_random": len(windows) - len(guided) - len(branch_windows),
        "windows_guide": len(guided),
        "windows_branch": len(branch_windows),
        "anchors": len(anchor_indices),
        "guide_termination": guide_termination,
        "no_feasible_plan": no_feasible,
        "success": int(any(int(record["success"]) for record in windows)),
        "guide_success": int(any(int(record["success"]) for record in guided)),
        "contact_windows": sum(int(record["contact"]["had_robot_object_contact"]) for record in windows),
        "failure_windows": sum(int(record["termination_class"] == "physical_failure") for record in windows),
        "guide_trial_physics_steps": guide_trial_steps,
        "guide_trial_prefix_steps": guide_prefix_steps,
        "guide_plan_wall_clock": guide_clock,
        "wall_clock": time.time() - started,
        "max_real_windows": MAX_REAL_WINDOWS,
    }
    return {"root": root, "windows": windows, "stats": stats}
