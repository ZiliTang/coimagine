"""A deterministic ``scene_id -> SceneSpec`` registry, loadable across processes.

The frozen library only knew ten names (``scene_by_name``). Random coverage needs every path
-- encoding, decoding, geometry checks, scoring, prediction, evaluation -- to read the *same*
full specification. This module provides the minimal compatible mechanism asked for in the
task book (section A1):

* full :class:`SceneSpec` serialisation round-trips exactly (floats go through ``repr``);
* a registry maps ``scene_id`` to a spec and is written to / read from one JSON file;
* :meth:`SceneRegistry.activate` installs the registry as the resolver consulted by
  :func:`coimagine.continuous2d.scenes.scene_by_name`, so no existing call site has to
  change and the ten original names still resolve to their frozen definitions.

``scene_id``, ``split`` and generation seed are metadata only. They never enter the model
input; the input is still built exclusively from the geometry the frozen encoder reads.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from ..continuous2d.scenes import Box, Pose, SceneSpec, default_scenes

SCENE_REGISTRY_VERSION = "coverage-scale-scene-registry-v1"


def box_to_dict(box: Box) -> Dict[str, float]:
    return {
        "cx": float(box.cx),
        "cy": float(box.cy),
        "half_w": float(box.half_w),
        "half_h": float(box.half_h),
        "angle": float(box.angle),
    }


def box_from_dict(payload: Dict[str, float]) -> Box:
    return Box(
        float(payload["cx"]),
        float(payload["cy"]),
        float(payload["half_w"]),
        float(payload["half_h"]),
        float(payload.get("angle", 0.0)),
    )


def pose_to_dict(pose: Pose) -> Dict[str, float]:
    return {"x": float(pose.x), "y": float(pose.y), "angle": float(pose.angle)}


def pose_from_dict(payload: Dict[str, float]) -> Pose:
    return Pose(float(payload["x"]), float(payload["y"]), float(payload.get("angle", 0.0)))


def scene_to_dict(scene: SceneSpec) -> Dict[str, object]:
    """Full, lossless serialisation of a :class:`SceneSpec`."""

    return {
        "name": scene.name,
        "category": scene.category,
        "variant": int(scene.variant),
        "arena": [float(scene.arena[0]), float(scene.arena[1])],
        "obstacles": [box_to_dict(box) for box in scene.obstacles],
        "goal": box_to_dict(scene.goal),
        "object_pose": pose_to_dict(scene.object_pose),
        "robot_poses": [pose_to_dict(pose) for pose in scene.robot_poses],
        "description": scene.description,
        "scene_version": scene.scene_version,
        "object_width": float(scene.object_width),
        "object_height": float(scene.object_height),
        "robot_radius": float(scene.robot_radius),
    }


def scene_from_dict(payload: Dict[str, object]) -> SceneSpec:
    arena = payload["arena"]  # type: ignore[index]
    robots = payload["robot_poses"]  # type: ignore[index]
    return SceneSpec(
        name=str(payload["name"]),
        category=str(payload["category"]),
        variant=int(payload["variant"]),  # type: ignore[arg-type]
        arena=(float(arena[0]), float(arena[1])),  # type: ignore[index]
        obstacles=tuple(box_from_dict(box) for box in payload["obstacles"]),  # type: ignore[arg-type]
        goal=box_from_dict(payload["goal"]),  # type: ignore[arg-type]
        object_pose=pose_from_dict(payload["object_pose"]),  # type: ignore[arg-type]
        robot_poses=(pose_from_dict(robots[0]), pose_from_dict(robots[1])),  # type: ignore[index]
        description=str(payload.get("description", "")),
        scene_version=str(payload.get("scene_version", "v1")),
        object_width=float(payload.get("object_width", 1.0)),
        object_height=float(payload.get("object_height", 0.25)),
        robot_radius=float(payload.get("robot_radius", 0.18)),
    )


@dataclass
class SceneRegistry:
    """Insertion-ordered ``name -> SceneSpec`` map with exact JSON round-trip."""

    specs: Dict[str, SceneSpec]

    def __init__(self, specs: Optional[Dict[str, SceneSpec]] = None):
        self.specs = dict(specs or {})

    # -- construction ------------------------------------------------------
    @classmethod
    def from_scenes(cls, scenes: Iterable[SceneSpec]) -> "SceneRegistry":
        registry = cls()
        for scene in scenes:
            registry.add(scene)
        return registry

    def add(self, scene: SceneSpec) -> None:
        existing = self.specs.get(scene.name)
        if existing is not None and scene_to_dict(existing) != scene_to_dict(scene):
            raise ValueError(f"scene_id {scene.name!r} already registered with other content")
        self.specs[scene.name] = scene

    # -- lookup ------------------------------------------------------------
    def get(self, name: str) -> SceneSpec:
        try:
            return self.specs[name]
        except KeyError as error:  # pragma: no cover - message only
            raise KeyError(name) from error

    def try_get(self, name: str) -> Optional[SceneSpec]:
        return self.specs.get(name)

    def __contains__(self, name: object) -> bool:
        return name in self.specs

    def __len__(self) -> int:
        return len(self.specs)

    def __iter__(self) -> Iterator[SceneSpec]:
        return iter(self.specs.values())

    def names(self) -> List[str]:
        return list(self.specs)

    # -- persistence -------------------------------------------------------
    def to_payload(self) -> Dict[str, object]:
        return {
            "registry_version": SCENE_REGISTRY_VERSION,
            "count": len(self.specs),
            "scenes": [scene_to_dict(self.specs[name]) for name in self.specs],
        }

    @classmethod
    def from_payload(cls, payload: Dict[str, object]) -> "SceneRegistry":
        version = payload.get("registry_version")
        if version != SCENE_REGISTRY_VERSION:
            raise ValueError(f"unknown scene registry version {version!r}")
        registry = cls()
        for entry in payload["scenes"]:  # type: ignore[index]
            registry.add(scene_from_dict(entry))  # type: ignore[arg-type]
        return registry

    def save(self, path: Path) -> str:
        from ..fit_capacity.trainer import file_sha256

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_payload(), indent=1, sort_keys=True))
        temporary.replace(path)
        return file_sha256(path)

    @classmethod
    def load(cls, path: Path) -> "SceneRegistry":
        return cls.from_payload(json.loads(path.read_text()))

    # -- activation --------------------------------------------------------
    def activate(self) -> "SceneRegistry":
        """Install this registry as the resolver used by ``scene_by_name``."""

        from ..continuous2d import scenes as scenes_module

        scenes_module.set_scene_resolver(self.try_get)
        return self


def load_registry_file(path: Path, *, activate: bool = True) -> SceneRegistry:
    """Load a registry written by :meth:`SceneRegistry.save` and optionally activate it."""

    registry = SceneRegistry.load(path)
    if activate:
        registry.activate()
    return registry


def full_registry(random_scenes: Sequence[SceneSpec] = ()) -> SceneRegistry:
    """The ten frozen scenes plus every random scene of this round."""

    registry = SceneRegistry.from_scenes(default_scenes())
    for scene in random_scenes:
        registry.add(scene)
    return registry
