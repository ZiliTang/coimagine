"""Trajectory export, schema version 2.

Changes from schema 1 (kept as a separate version; old files are never overwritten):

* skills are stored **structured and at full precision** (``kind`` plus every parameter),
  so a skill can be rebuilt without parsing its human-readable label;
* ``applied_control`` records **both** robots, as ``[control_step, robot_id, [fx, fy]]``
  entries, and every ``steps`` entry carries both robots' forces plus its own
  ``state``/``next_state`` pair;
* the success hold counter, the termination reason and the controller state are recorded;
* the scene, physics, controller and code versions are referenced, not implied.

``windows.jsonl`` is one record per synchronous joint-skill window; ``episodes.jsonl``
orders them. :func:`segments` slices contiguous clips inside a single episode only.

All records this round are ``dev``. No learning dataset is produced.
"""

from __future__ import annotations

import json
import subprocess
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from .config import CONTROLLER_VERSION, Continuous2DConfig
from .mpc import EpisodeResult
from .skills import joint_spec_to_dicts

SCHEMA_VERSION = 2


@lru_cache(maxsize=1)
def code_version() -> str:
    """Best-effort code reference for a data record; ``unknown`` without git."""

    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return f"{head}{'-dirty' if dirty else ''}"
    except Exception:
        return "unknown"


def window_records(
    episode_id: str,
    scene_name: str,
    seed: int,
    episode: EpisodeResult,
    config: Continuous2DConfig,
    *,
    split: str = "dev",
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    index = 0
    for entry in episode.trace:
        # Episodes store SkillResults; a raw TrajectoryResult is accepted too.
        results = getattr(entry, "results", None) or (entry,)
        for result in results:
            steps: List[Dict[str, object]] = []
            applied_control: List[List[object]] = []
            for step_index, step in enumerate(result.steps):
                applied_control.append(
                    [step_index, 0, [float(step.applied_control[0][0]), float(step.applied_control[0][1])]]
                )
                applied_control.append(
                    [step_index, 1, [float(step.applied_control[1][0]), float(step.applied_control[1][1])]]
                )
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
                        "applied_control": [
                            list(step.applied_control[0]),
                            list(step.applied_control[1]),
                        ],
                        "skill_labels": list(step.skill_labels),
                        "penetration": step.penetration,
                        "penetration_by_pair": dict(step.penetration_by_pair),
                        "contacts": [list(point) for point in step.contacts],
                    }
                )
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "scene_id": scene_name,
                    "seed": seed,
                    "split": split,
                    "window_index": index,
                    "t": result.initial.tick,
                    "state": {
                        "boundary": "before_window",
                        "robots": [state.as_tuple() for state in result.initial.robots],
                        "object": result.initial.object.as_tuple(),
                    },
                    "next_state": {
                        "boundary": "after_window",
                        "robots": [state.as_tuple() for state in result.final.robots],
                        "object": result.final.object.as_tuple(),
                    },
                    "joint_skill_specs": joint_spec_to_dicts(result.joint_spec),
                    "joint_skill_labels": [result.joint_spec[0].label, result.joint_spec[1].label],
                    "controller_state": result.initial.controller_state,
                    "applied_control": applied_control,
                    "duration": result.duration,
                    "termination": result.termination,
                    "success": int(result.success),
                    "success_counter": result.final.success_counter,
                    "cost_components": dict(result.cost_components),
                    "physics_config_hash": config.config_hash(),
                    "config_hash": result.initial.config_hash,
                    "scene_name": result.initial.scene_config.name,
                    "controller_version": result.controller_version,
                    "controller_version_id": CONTROLLER_VERSION,
                    "code_version": code_version(),
                    "steps": steps,
                }
            )
            index += 1
    return records


def episode_record(
    episode_id: str,
    scene_name: str,
    seed: int,
    episode: EpisodeResult,
    config: Continuous2DConfig,
    *,
    split: str = "dev",
) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": episode_id,
        "scene_id": scene_name,
        "seed": seed,
        "split": split,
        "windows": len(episode.trace),
        "success": int(episode.success),
        "termination": episode.termination,
        "no_feasible_plan": int(episode.no_feasible_plan),
        "success_time": episode.success_time,
        "control_usage": episode.control_usage,
        "terminal_distance": episode.terminal_distance,
        "window_indices": list(range(len(episode.trace))),
        "physics_config_hash": config.config_hash(),
        "controller_version": CONTROLLER_VERSION,
        "code_version": code_version(),
        "plan_nodes": episode.plan_nodes,
        "candidate_simulated_steps": episode.candidate_simulated_steps,
        "prefix_replay_steps": episode.prefix_replay_steps,
        "live_physics_steps": episode.live_physics_steps,
        "failure_distribution": dict(episode.failure_distribution),
        "decisions": [
            {
                "window_index": decision.window_index,
                "chosen_candidate_id": decision.chosen_candidate_id,
                "sequence_length": decision.sequence_length,
                "predicted_sequence_labels": list(decision.predicted_sequence_labels),
                "predicted_success_time": decision.predicted_success_time,
                "predicted_final_key": (
                    list(decision.predicted_final_key) if decision.predicted_final_key is not None else None
                ),
                "key": list(decision.key) if decision.key is not None else None,
                "termination": decision.termination,
                "node_candidates": decision.node_candidates,
                "plan_wall_clock": decision.plan_wall_clock,
            }
            for decision in episode.decisions
        ],
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
            count += 1
    return count


def segments(
    windows: Sequence[Dict[str, object]], length: int
) -> List[List[Dict[str, object]]]:
    """Contiguous ``length``-window clips; segments never cross an episode boundary."""

    episode_ids = {str(record["episode_id"]) for record in windows}
    if len(episode_ids) > 1:
        raise ValueError(f"segments must come from one episode, got {sorted(episode_ids)}")
    ordered = sorted(windows, key=lambda record: int(record["window_index"]))
    return [list(ordered[start : start + length]) for start in range(len(ordered) - length + 1)]
