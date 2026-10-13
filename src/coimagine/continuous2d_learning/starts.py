"""Evaluation starts for the fixed-candidate ranking diagnostic (contract C2).

One legal non-terminal initial state per test scene plus five non-terminal window starts
ordered by ``SHA256("continuous-audit-v1|" + episode_id + "|" + window_index)``. Selection
never uses model error, distance or return.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ..continuous2d.config import Continuous2DConfig, default_config
from ..continuous2d.environment import Continuous2DEnv
from ..continuous2d.scenes import scene_by_name
from .records import (
    CollectionConfig,
    SPLIT_SCENES,
    episode_boundaries,
)

AUDIT_SALT = "continuous-audit-v1"
TEST_SCENES = SPLIT_SCENES["test"]


@dataclass
class EvalStart:
    start_id: str
    kind: str
    scene_id: str
    episode_id: Optional[str]
    window_index: Optional[int]
    seed: Optional[int]
    success_counter: float
    state_key: str
    snapshot_hash: str
    physical_state: Dict[str, object]
    snapshot: object
    observation: object

    def recoverable(self) -> Dict[str, object]:
        return {
            "start_id": self.start_id,
            "kind": self.kind,
            "scene_id": self.scene_id,
            "episode_id": self.episode_id,
            "window_index": self.window_index,
            "seed": self.seed,
            "success_counter": self.success_counter,
            "state_key": self.state_key,
            "snapshot_hash": self.snapshot_hash,
            "physical_state": self.physical_state,
        }


def _physical_key(snapshot) -> str:
    def _round(values) -> Tuple[float, ...]:
        return tuple(round(float(value), 12) for value in values)

    payload = {
        "object": _round(snapshot.object.as_tuple()),
        "robots": [_round(state.as_tuple()) for state in snapshot.robots],
        "success_counter": round(float(snapshot.success_counter), 12),
    }
    return json.dumps(payload, sort_keys=True)


def _snapshot_hash(snapshot) -> str:
    return hashlib.sha256(_physical_key(snapshot).encode()).hexdigest()[:16]


def _order_key(episode_id: str, window_index: int) -> str:
    return hashlib.sha256(f"{AUDIT_SALT}|{episode_id}|{window_index}".encode()).hexdigest()


def _make_start(
    start_id: str,
    kind: str,
    scene_id: str,
    snapshot,
    observation,
    *,
    episode_id: Optional[str] = None,
    window_index: Optional[int] = None,
    seed: Optional[int] = None,
) -> EvalStart:
    return EvalStart(
        start_id=start_id,
        kind=kind,
        scene_id=scene_id,
        episode_id=episode_id,
        window_index=window_index,
        seed=seed,
        success_counter=float(snapshot.success_counter),
        state_key=_physical_key(snapshot),
        snapshot_hash=_snapshot_hash(snapshot),
        physical_state={
            "object": list(snapshot.object.as_tuple()),
            "robots": [list(state.as_tuple()) for state in snapshot.robots],
        },
        snapshot=snapshot,
        observation=observation,
    )


def select_eval_starts(
    windows_path: Path,
    *,
    config: Optional[Continuous2DConfig] = None,
    collection: Optional[CollectionConfig] = None,
    window_starts_per_scene: int = 5,
    scenes: Sequence[str] = TEST_SCENES,
) -> List[EvalStart]:
    config = config or default_config()
    collection = collection or CollectionConfig()
    records: List[Dict[str, object]] = []
    with windows_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record["split"] == "test" and record["scene_id"] in scenes:
                records.append(record)
    by_scene: Dict[str, List[Dict[str, object]]] = {name: [] for name in scenes}
    for record in records:
        by_scene[str(record["scene_id"])].append(record)

    starts: List[EvalStart] = []
    for scene_name in scenes:
        env = Continuous2DEnv(config)
        env.reset(scene_by_name(scene_name))
        initial = _make_start(f"{scene_name}/initial", "initial", scene_name, env.export_state(), env.observation())
        selected = [initial]
        seen = {initial.state_key}

        ordered = sorted(
            by_scene[scene_name],
            key=lambda record: _order_key(str(record["episode_id"]), int(record["window_index"])),
        )
        boundary_cache: Dict[str, List[Dict[str, object]]] = {}
        for record in ordered:
            if len(selected) >= 1 + window_starts_per_scene:
                break
            episode_id = str(record["episode_id"])
            window_index = int(record["window_index"])
            if episode_id not in boundary_cache:
                boundary_cache[episode_id] = episode_boundaries(
                    scene_name, episode_id, "test", config, collection
                )
            boundaries = boundary_cache[episode_id]
            if window_index >= len(boundaries):
                continue
            entry = boundaries[window_index]
            snapshot = entry["snapshot"]
            key = _physical_key(snapshot)
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                _make_start(
                    f"{scene_name}/w{window_index:02d}/{episode_id}",
                    "window",
                    scene_name,
                    snapshot,
                    entry["observation"],
                    episode_id=episode_id,
                    window_index=window_index,
                    seed=int(entry["seed"]),
                )
            )
        starts.extend(selected)
    return starts
