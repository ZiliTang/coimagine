"""Window-level data collection for the single-scale continuous world model (contract B3).

Collection behaviour, seeds and split are frozen in
``docs/forecast_baseline_v1/CONTRACT.md``. This module never filters an action by predicted
or exact return, never continues from a terminal state, and never mixes in an exact-MPC
trajectory.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.candidates import joint_candidates
from ..continuous2d.config import CONTROLLER_VERSION, Continuous2DConfig, default_config
from ..continuous2d.data import code_version, write_jsonl
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.legality import scene_content_hash
from ..continuous2d.scenes import scene_by_name
from ..continuous2d.skills import JointSpec, joint_spec_to_dicts
from .contract import (
    ROTATION_THRESHOLD,
    TRANSLATION_THRESHOLD,
    termination_class,
    wrap_angle,
)

SEED_SALT = "forecast-baseline-v1"
RECORD_VERSION = "forecast-baseline-v1"

#: Whole-scene split (contract B3). No episode fragment crosses a split.
SPLIT_SCENES: Dict[str, Tuple[str, ...]] = {
    "train": ("S0_dev", "S0_v1", "S0_v2", "S1_dev", "S1_v1", "S1_v2"),
    "val": ("S0_v3", "S1_v3"),
    "test": ("S0_v4", "S1_v4"),
}


@dataclass(frozen=True)
class CollectionConfig:
    episodes_per_scene: int = 20
    max_windows: int = 12
    #: Probability of drawing a fresh candidate instead of repeating the previous window.
    fresh_candidate_probability: float = 0.5
    #: Episodes per scene that also keep the full per-control-step record.
    full_step_episodes: int = 1
    seed_salt: str = SEED_SALT

    def as_dict(self) -> Dict[str, object]:
        return {
            "episodes_per_scene": self.episodes_per_scene,
            "max_windows": self.max_windows,
            "fresh_candidate_probability": self.fresh_candidate_probability,
            "full_step_episodes": self.full_step_episodes,
            "seed_salt": self.seed_salt,
        }


def episode_seed(scene_name: str, episode_id: str, salt: str = SEED_SALT) -> int:
    digest = hashlib.sha256(f"{salt}|{scene_name}|{episode_id}".encode()).hexdigest()
    return int(digest[:8], 16)


def _step_records(result) -> List[Dict[str, object]]:
    steps: List[Dict[str, object]] = []
    for step_index, step in enumerate(result.steps):
        steps.append(
            {
                "step_index": step_index,
                "tick": step.tick,
                "time": step.time,
                "state": {
                    "robots": [state.as_tuple() for state in step.before_robots],
                    "object": step.before_object.as_tuple(),
                },
                "next_state": {
                    "robots": [state.as_tuple() for state in step.robots],
                    "object": step.object.as_tuple(),
                },
                "applied_control": [list(step.applied_control[0]), list(step.applied_control[1])],
                "skill_labels": list(step.skill_labels),
                "penetration": step.penetration,
                "penetration_by_pair": dict(step.penetration_by_pair),
                "contacts": [list(point) for point in step.contacts],
            }
        )
    return steps


def _choose_joint_spec(
    observation,
    candidates,
    rng: random.Random,
    window_index: int,
    previous: Optional[JointSpec],
    collection: CollectionConfig,
) -> JointSpec:
    """The frozen collection rule, shared by collection and prefix replay."""

    if window_index == 0 or rng.random() < collection.fresh_candidate_probability:
        return rng.choice(candidates).skills
    assert previous is not None
    return previous


def episode_boundaries(
    scene_name: str,
    episode_id: str,
    split: str,
    config: Continuous2DConfig,
    collection: CollectionConfig,
) -> List[Dict[str, object]]:
    """Re-run one episode deterministically and return every window-start boundary.

    The collection rule is a pure function of ``(scene, episode_id, seed_salt)``, so the
    boundary snapshots reproduced here are the ones the collection run saw. Each entry
    carries the replayable snapshot, the chosen joint skill spec and the observation; these
    are the only inputs the diagnostic uses to rebuild a start.
    """

    scene = scene_by_name(scene_name)
    seed = episode_seed(scene_name, episode_id, collection.seed_salt)
    rng = random.Random(seed)
    env = Continuous2DEnv(config)
    env.reset(scene)
    boundaries: List[Dict[str, object]] = []
    previous: Optional[JointSpec] = None
    for window_index in range(collection.max_windows):
        if env.termination is not None:
            break
        observation = env.observation()
        candidates = joint_candidates(
            observation, config.candidates, config.control, config.physics
        )
        joint_spec = _choose_joint_spec(
            observation, candidates, rng, window_index, previous, collection
        )
        previous = joint_spec
        snapshot = env.export_state()
        boundaries.append(
            {
                "scene_id": scene_name,
                "episode_id": episode_id,
                "split": split,
                "seed": seed,
                "window_index": window_index,
                "snapshot": snapshot,
                "observation": observation,
                "joint_spec": joint_spec,
            }
        )
        env.execute_joint_skill(joint_spec)
    return boundaries


def collect_episode(
    scene_name: str,
    episode_id: str,
    split: str,
    config: Continuous2DConfig,
    collection: CollectionConfig,
    *,
    keep_steps: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Run one episode under the frozen collection behaviour and return its window records."""

    scene = scene_by_name(scene_name)
    seed = episode_seed(scene_name, episode_id, collection.seed_salt)
    rng = random.Random(seed)
    env = Continuous2DEnv(config)
    env.reset(scene)
    records: List[Dict[str, object]] = []
    previous: Optional[JointSpec] = None

    for window_index in range(collection.max_windows):
        if env.termination is not None:
            break
        observation = env.observation()
        candidates = joint_candidates(
            observation, config.candidates, config.control, config.physics
        )
        joint_spec: JointSpec = _choose_joint_spec(
            observation, candidates, rng, window_index, previous, collection
        )
        previous = joint_spec

        before = env.export_state()
        result = env.execute_joint_skill(joint_spec)

        contacts = [
            point
            for step in result.steps
            for point in step.contacts
        ]
        had_contact = any(str(point[0]) == "robot_object" for point in contacts)
        max_penetration = max((step.penetration for step in result.steps), default=0.0)
        penetration_by_pair: Dict[str, float] = {}
        for step in result.steps:
            for key, value in step.penetration_by_pair.items():
                penetration_by_pair[key] = max(penetration_by_pair.get(key, 0.0), float(value))

        record: Dict[str, object] = {
            "schema_version": 2,
            "record_version": RECORD_VERSION,
            "episode_id": episode_id,
            "scene_id": scene_name,
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
                "max_penetration": max_penetration,
                "penetration_by_pair": penetration_by_pair,
            },
            "cost_components": dict(result.cost_components),
            "config_hash": config.config_hash(),
            "controller_version": CONTROLLER_VERSION,
            "code_version": code_version(),
        }
        if keep_steps:
            record["steps"] = _step_records(result)
        records.append(record)
        if result.termination not in ("window_complete",):
            break

    summary = {
        "episode_id": episode_id,
        "scene_id": scene_name,
        "split": split,
        "seed": seed,
        "windows": len(records),
        "success": int(any(int(record["success"]) for record in records)),
        "termination": records[-1]["termination"] if records else "none",
        "full_steps_recorded": bool(keep_steps),
        "scene_version": scene.scene_version,
        "scene_content_hash": scene_content_hash(scene),
    }
    return records, summary


# ---------------------------------------------------------------------------
# Coverage statistics (contract B3 thresholds)
# ---------------------------------------------------------------------------


def _combination(specs: Sequence[Dict[str, object]]) -> str:
    kinds = [str(spec["kind"]) for spec in specs]
    pushes = kinds.count("push_contact")
    moves = kinds.count("move_to")
    holds = kinds.count("hold")
    if pushes == 2:
        return "double_push"
    if pushes == 1 and holds == 1:
        return "single_push"
    if pushes == 1 and moves == 1:
        return "push_move"
    if pushes == 1:
        return "single_push"
    if moves >= 1 and holds >= 1:
        return "move_hold"
    if moves == 2:
        return "move_move"
    if holds == 2:
        return "hold_hold"
    return "mixed"


def window_groups(record: Dict[str, object]) -> Dict[str, bool]:
    """Group flags for one window, using only the frozen thresholds."""

    before = record["state"]
    after = record["next_state"]
    dx = float(after["object"][0]) - float(before["object"][0])
    dy = float(after["object"][1]) - float(before["object"][1])
    translation = (dx * dx + dy * dy) ** 0.5
    rotation = abs(wrap_angle(float(after["object"][2]) - float(before["object"][2])))
    translated = translation > TRANSLATION_THRESHOLD
    rotated = rotation > ROTATION_THRESHOLD
    return {
        "object_translation": translated,
        "object_rotation": rotated,
        "stationary": not translated and not rotated,
        "robot_moved": any(
            abs(float(after["robots"][i][j]) - float(before["robots"][i][j])) > 1e-9
            for i in (0, 1)
            for j in (0, 1)
        ),
        "contact": bool(record["contact"]["had_robot_object_contact"]),
        "success": bool(int(record["success"])),
        "failure": str(record["termination_class"]) == "physical_failure",
    }


def coverage(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    per_split: Dict[str, Dict[str, object]] = {}
    for record in records:
        split = str(record["split"])
        row = per_split.setdefault(
            split,
            {
                "windows": 0,
                "episodes": set(),
                "object_translation": 0,
                "object_rotation": 0,
                "stationary": 0,
                "robot_moved": 0,
                "contact": 0,
                "success": 0,
                "failure": 0,
                "combinations": {},
                "terminations": {},
                "scenes": set(),
            },
        )
        row["windows"] += 1
        row["episodes"].add(str(record["episode_id"]))
        row["scenes"].add(str(record["scene_id"]))
        flags = window_groups(record)
        for key in ("object_translation", "object_rotation", "stationary", "robot_moved", "contact", "success", "failure"):
            row[key] += int(bool(flags[key]))
        combination = _combination(record["joint_skill_specs"])  # type: ignore[arg-type]
        row["combinations"][combination] = row["combinations"].get(combination, 0) + 1
        termination = str(record["termination"])
        row["terminations"][termination] = row["terminations"].get(termination, 0) + 1
    out: Dict[str, object] = {}
    for split, row in per_split.items():
        row = dict(row)
        row["episodes"] = len(row["episodes"])
        row["scenes"] = sorted(row["scenes"])
        row["windows"] = int(row["windows"])
        row["thresholds"] = {
            "translation_m": TRANSLATION_THRESHOLD,
            "rotation_rad": ROTATION_THRESHOLD,
        }
        out[split] = row
    return out


def collect_dataset(
    out_dir: Path,
    *,
    config: Optional[Continuous2DConfig] = None,
    collection: Optional[CollectionConfig] = None,
) -> Dict[str, object]:
    """Collect all 200 episodes, write ``windows.jsonl``/``episodes.jsonl``/``coverage.json``."""

    config = config or default_config()
    collection = collection or CollectionConfig()
    started = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    windows: List[Dict[str, object]] = []
    episodes: List[Dict[str, object]] = []
    for split in ("train", "val", "test"):
        for scene_name in SPLIT_SCENES[split]:
            for index in range(collection.episodes_per_scene):
                episode_id = f"{scene_name}/{index:02d}"
                keep_steps = index < collection.full_step_episodes
                records, summary = collect_episode(
                    scene_name, episode_id, split, config, collection, keep_steps=keep_steps
                )
                windows.extend(records)
                episodes.append(summary)
    write_jsonl(out_dir / "windows.jsonl", windows)
    write_jsonl(out_dir / "episodes.jsonl", episodes)
    coverage_stats = coverage(windows)
    (out_dir / "coverage.json").write_text(json.dumps(coverage_stats, indent=2, sort_keys=True))
    manifest = {
        "record_version": RECORD_VERSION,
        "collection": collection.as_dict(),
        "config": config.as_dict(),
        "split_scenes": {key: list(value) for key, value in SPLIT_SCENES.items()},
        "episodes": len(episodes),
        "windows": len(windows),
        "wall_clock": time.time() - started,
        "coverage": coverage_stats,
        "code_version": code_version(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
