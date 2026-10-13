"""Phase C: guided successful trajectories, anchors, branch data and the random control.

The exact model is used here **offline only**, to produce real executed trajectories. No
exact model prediction is ever used as a supervision label: every stored transition is a
real ``execute_joint_skill`` result from the live environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.candidates import JointCandidate, joint_candidates
from ..continuous2d.config import CONTROLLER_VERSION, Continuous2DConfig
from ..continuous2d.data import code_version
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.legality import scene_content_hash
from ..continuous2d.scenes import SceneSpec, scene_by_name
from ..continuous2d.skills import Hold, JointSpec, joint_spec_to_dicts
from ..continuous2d_learning.contract import termination_class
from ..continuous2d_learning.features import BoundaryState, skill_features
from ..continuous2d_learning.pools import candidate_pool, root_library
from ..continuous2d_learning.records import CollectionConfig, SPLIT_SCENES, collect_episode, episode_seed
from ..continuous2d_learning.rollout import evaluate_sequence_exact

GUIDE_DEPTH = 3
GUIDE_K = 128
GUIDE_HORIZON = 12
MAX_ANCHORS = 6
BRANCHES_PER_ANCHOR = 16
BRANCH_CONTINUATION = 2
RANDOM_ADD_SALT = "random-add-v1"
RECORD_VERSION = "data-upgrade-v1"

TRAIN_SCENES = SPLIT_SCENES["train"]
VAL_SCENES = SPLIT_SCENES["val"]
TEST_SCENES = SPLIT_SCENES["test"]


def _pool_seed(scene: str, window: int, episode_seed_value: int) -> int:
    digest = hashlib.sha256(
        f"{scene}|{window}|{episode_seed_value}|forecast-pool-v1".encode()
    ).hexdigest()
    return int(digest[:8], 16) % (2**31 - 1)


def _branch_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _window_record(
    *,
    scene: SceneSpec,
    split: str,
    episode_id: str,
    seed: int,
    window_index: int,
    joint_spec: JointSpec,
    before,
    result,
    config: Continuous2DConfig,
    source: Dict[str, object],
) -> Dict[str, object]:
    contacts = [point for step in result.steps for point in step.contacts]
    had_contact = any(str(point[0]) == "robot_object" for point in contacts)
    penetration_by_pair: Dict[str, float] = {}
    for step in result.steps:
        for key, value in step.penetration_by_pair.items():
            penetration_by_pair[key] = max(penetration_by_pair.get(key, 0.0), float(value))
    return {
        "schema_version": 2,
        "record_version": RECORD_VERSION,
        "episode_id": episode_id,
        "scene_id": scene.name,
        "scene_version": scene.scene_version,
        "scene_content_hash": scene_content_hash(scene),
        "split": split,
        "seed": seed,
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
        "source": dict(source),
    }


# ---------------------------------------------------------------------------
# C2: guided trajectories
# ---------------------------------------------------------------------------


def guide_episode(
    scene_name: str,
    episode_seed_value: int,
    *,
    split: str,
    config: Continuous2DConfig,
    planner: Optional[Continuous2DEnv] = None,
    live: Optional[Continuous2DEnv] = None,
) -> Dict[str, object]:
    """One ``exact_fixed_pool`` episode (H=3, K=128), fully recorded for offline use."""

    scene = scene_by_name(scene_name)
    live = live or Continuous2DEnv(config)
    planner = planner or Continuous2DEnv(config)
    live.reset(scene)
    started = time.time()
    plan_clock = 0.0
    windows: List[Dict[str, object]] = []
    snapshots: List[object] = []
    observations: List[object] = []
    specs: List[JointSpec] = []
    exact_steps = 0
    prefix_steps = 0
    termination = "window_complete"
    no_feasible = False

    for window in range(GUIDE_HORIZON):
        if live.termination is not None:
            break
        observation = live.observation()
        snapshot = live.export_state()
        seed = _pool_seed(scene_name, window, episode_seed_value)
        pool = candidate_pool(observation, config, GUIDE_DEPTH, seed, cap=GUIDE_K)
        plan_started = time.time()
        evaluations = [
            evaluate_sequence_exact(planner, snapshot, scene, sequence, config, candidate_id=index)
            for index, sequence in enumerate(pool)
        ]
        plan_clock += time.time() - plan_started
        exact_steps += sum(evaluation.physics_steps for evaluation in evaluations)
        prefix_steps += sum(evaluation.prefix_replay_steps for evaluation in evaluations)
        feasible = [index for index, evaluation in enumerate(evaluations) if evaluation.feasible]
        if not pool or not feasible:
            termination = "planner_no_feasible_plan"
            no_feasible = True
            break
        best = max(feasible, key=lambda index: (evaluations[index].j, -evaluations[index].candidate_id))
        joint_spec = pool[best][0]
        snapshots.append(snapshot)
        observations.append(observation)
        specs.append(joint_spec)
        before = snapshot
        result = live.execute_joint_skill(joint_spec)
        windows.append(
            _window_record(
                scene=scene, split=split, episode_id=f"guide/{scene_name}/{episode_seed_value}",
                seed=episode_seed_value, window_index=window, joint_spec=joint_spec,
                before=before, result=result, config=config,
                source={"kind": "guide", "teacher_episode_id": f"{scene_name}/g{episode_seed_value}", "anchor_id": None, "branch_id": None},
            )
        )
        termination = result.termination
        if result.termination in ("success", "numeric_failure", "out_of_bounds", "penetration_limit"):
            break

    success = any(int(record["success"]) for record in windows)
    return {
        "scene_id": scene_name,
        "split": split,
        "episode_seed": episode_seed_value,
        "teacher_episode_id": f"{scene_name}/g{episode_seed_value}",
        "success": int(success),
        "termination": termination,
        "no_feasible_plan": int(no_feasible),
        "windows": windows,
        "snapshots": snapshots,
        "observations": observations,
        "first_specs": specs,
        "exact_simulation_steps": exact_steps,
        "prefix_replay_steps": prefix_steps,
        "plan_wall_clock": plan_clock,
        "wall_clock": time.time() - started,
    }


def collect_guides(
    scenes: Sequence[str],
    *,
    config: Continuous2DConfig,
    split_for_scene,
    seeds: Sequence[int] = (0, 1),
    fallback_seeds: Sequence[int] = (2, 3),
    max_successes_per_scene: int = 2,
    max_episodes: int = 40,
) -> Dict[str, object]:
    """Guided trajectories for every scene, respecting the 40-episode budget."""

    planner = Continuous2DEnv(config)
    live = Continuous2DEnv(config)
    runs: List[Dict[str, object]] = []
    attempts: List[Dict[str, object]] = []
    spent = 0
    for scene_name in scenes:
        successes = 0
        used_seeds: List[int] = []
        for phase, seed_group in (("primary", seeds), ("fallback", fallback_seeds)):
            if successes >= max_successes_per_scene:
                break
            for episode_seed_value in seed_group:
                if successes >= max_successes_per_scene or spent >= max_episodes:
                    break
                run = guide_episode(
                    scene_name, episode_seed_value, split=split_for_scene(scene_name),
                    config=config, planner=planner, live=live,
                )
                spent += 1
                used_seeds.append(episode_seed_value)
                attempts.append(
                    {
                        "scene_id": scene_name, "episode_seed": episode_seed_value, "success": run["success"],
                        "windows": len(run["windows"]), "termination": run["termination"],
                        "phase": phase, "wall_clock": run["wall_clock"],
                    }
                )
                if run["success"]:
                    successes += 1
                    runs.append(run)
        if successes == 0:
            attempts.append({"scene_id": scene_name, "warning": "no successful guided trajectory"})
    return {
        "runs": runs,
        "attempts": attempts,
        "episodes_spent": spent,
        "budget": max_episodes,
    }


# ---------------------------------------------------------------------------
# C3: anchors
# ---------------------------------------------------------------------------


def anchor_indices(n_windows: int, max_anchors: int = MAX_ANCHORS) -> List[int]:
    """Last two non-terminal window starts first, then evenly spaced ones, capped."""

    if n_windows <= 0:
        return []
    if n_windows <= max_anchors:
        return list(range(n_windows))
    keep = [n_windows - 2, n_windows - 1]
    rest: List[int] = []
    slots = max_anchors * 2
    for k in range(slots):
        index = round(k * (n_windows - 1) / max(1, slots - 1))
        if index not in keep and index not in rest:
            rest.append(index)
    chosen: List[int] = []
    for index in keep + rest:
        if index not in chosen:
            chosen.append(index)
    return sorted(chosen[:max_anchors])


def build_anchors(run: Dict[str, object], *, max_anchors: int = MAX_ANCHORS) -> List[Dict[str, object]]:
    windows = run["windows"]
    indices = anchor_indices(len(windows), max_anchors)
    anchors = []
    for index in indices:
        record = windows[index]
        anchors.append(
            {
                "anchor_id": f"{run['teacher_episode_id']}/a{index}",
                "teacher_episode_id": run["teacher_episode_id"],
                "scene_id": run["scene_id"],
                "split": run["split"],
                "window_index": index,
                "snapshot": run["snapshots"][index],
                "observation": run["observations"][index],
                "expert_joint_spec": run["first_specs"][index],
                "state": record["state"],
                "success_counter": record["success_counter_before"],
            }
        )
    return anchors


# ---------------------------------------------------------------------------
# C4: action branches
# ---------------------------------------------------------------------------


def _spec_vector(joint_spec: JointSpec, scene: SceneSpec, physics) -> List[float]:
    return skill_features(joint_spec[0], scene, physics) + skill_features(joint_spec[1], scene, physics)


def _squared_distance(first: Sequence[float], second: Sequence[float]) -> float:
    return float(sum((a - b) ** 2 for a, b in zip(first, second)))


def branch_first_actions(
    observation,
    scene: SceneSpec,
    expert_spec: JointSpec,
    config: Continuous2DConfig,
    rng: random.Random,
    *,
    count: int = BRANCHES_PER_ANCHOR,
) -> List[JointSpec]:
    """1 expert + 1 hold-hold + 4 nearest + 10 uniform, deduplicated and topped up."""

    root = root_library(observation, config)
    options: List[JointCandidate] = list(root)
    keys = set()
    chosen: List[JointSpec] = []

    def add(spec: JointSpec) -> bool:
        key = json.dumps(joint_spec_to_dicts(spec), sort_keys=True)
        if key in keys:
            return False
        keys.add(key)
        chosen.append(spec)
        return True

    add(expert_spec)
    add((Hold(), Hold()))

    expert_vector = _spec_vector(expert_spec, scene, config.physics)
    ranked = sorted(
        options,
        key=lambda candidate: (
            _squared_distance(_spec_vector(candidate.skills, scene, config.physics), expert_vector),
            candidate.candidate_id,
        ),
    )
    added = 0
    for candidate in ranked:
        if added >= 4:
            break
        if add(candidate.skills):
            added += 1

    attempts = 0
    while len(chosen) < count and attempts < count * 40 and options:
        attempts += 1
        add(rng.choice(options).skills)
    return chosen[:count]


def collect_branches(
    anchor: Dict[str, object],
    *,
    config: Continuous2DConfig,
    planner: Continuous2DEnv,
    live: Continuous2DEnv,
) -> List[Dict[str, object]]:
    """Execute every first action from the same anchor snapshot, then up to 2 more windows."""

    scene = scene_by_name(str(anchor["scene_id"]))
    seed = _branch_seed(anchor["split"], anchor["scene_id"], anchor["teacher_episode_id"], anchor["anchor_id"])
    rng = random.Random(seed)
    first_actions = branch_first_actions(
        anchor["observation"], scene, anchor["expert_joint_spec"], config, rng
    )
    records: List[Dict[str, object]] = []
    for branch_id, first in enumerate(first_actions):
        live.restore(anchor["snapshot"], exact=True)
        previous = first
        for step in range(1 + BRANCH_CONTINUATION):
            if live.termination is not None:
                break
            observation = live.observation()
            if step == 0:
                joint_spec = first
            elif rng.random() < 0.5:
                joint_spec = previous
            else:
                joint_spec = rng.choice(root_library(observation, config)).skills
            previous = joint_spec
            before = live.export_state()
            result = live.execute_joint_skill(joint_spec)
            records.append(
                _window_record(
                    scene=scene, split=str(anchor["split"]),
                    episode_id=f"branch/{anchor['anchor_id']}/{branch_id}",
                    seed=seed, window_index=step, joint_spec=joint_spec,
                    before=before, result=result, config=config,
                    source={
                        "kind": "branch",
                        "teacher_episode_id": anchor["teacher_episode_id"],
                        "anchor_id": anchor["anchor_id"],
                        "branch_id": branch_id,
                        "branch_window": step,
                    },
                )
            )
            if result.termination != "window_complete":
                break
    return records


# ---------------------------------------------------------------------------
# C5: random control
# ---------------------------------------------------------------------------


def collect_random_add(
    scene_name: str,
    target_transitions: int,
    *,
    config: Continuous2DConfig,
    attempt_cap: int,
    max_windows: int = 12,
    start_index: int = 0,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Original collection policy, until ``target_transitions`` new windows are reached."""

    collection = CollectionConfig(
        episodes_per_scene=max_windows, max_windows=max_windows, full_step_episodes=0,
        seed_salt=RANDOM_ADD_SALT,
    )
    accepted: List[Dict[str, object]] = []
    attempts = 0
    episode_index = start_index
    while len(accepted) < target_transitions and attempts < attempt_cap:
        episode_id = f"rand/{scene_name}/{episode_index:04d}"
        records, _summary = collect_episode(
            scene_name, episode_id, "train", config, collection, keep_steps=False
        )
        attempts += len(records)
        episode_index += 1
        for record in records:
            record.setdefault("source", {})
            record["source"] = {
                "kind": "random_add",
                "teacher_episode_id": None,
                "anchor_id": None,
                "branch_id": None,
            }
            accepted.append(record)
            if len(accepted) >= target_transitions:
                break
    report = {
        "scene_id": scene_name,
        "target_transitions": target_transitions,
        "attempted_windows": attempts,
        "attempt_cap": attempt_cap,
        "accepted_windows": len(accepted),
        "shortfall": max(0, target_transitions - len(accepted)),
        "episodes": episode_index - start_index,
    }
    return accepted[:target_transitions], report
