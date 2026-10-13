"""Disk-backed, nested root plan (task book A2/B2).

The plan for a split is generated once and cached as ``roots_<split>.jsonl`` plus a
``scenes_<split>.json`` registry. Caching is what makes "a prefix of the plan is a valid
smaller plan" true across processes and restarts: the *k*-th accepted root never depends on
how many roots a later run happens to ask for.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Sequence

from ..continuous2d.scenes import SceneSpec
from .sampler import RootSpec, sample_roots
from .scene_registry import SceneRegistry, scene_to_dict


def _root_to_row(root: RootSpec) -> Dict[str, object]:
    row = root.as_metadata()
    row["scene"] = root.scene  # replaced by the registry on write
    return row


def resolve_scene(registry: SceneRegistry, root: RootSpec) -> RootSpec:
    """Give ``root`` a scene name that is unique inside ``registry``.

    ``scene_id`` is ``{family}_r{seed:09d}`` and the seed is a 24-bit hash of
    ``(split, family, mode, ordinal)``, so two different cells can collide once the plan grows
    past a few thousand roots (birthday bound). When the colliding name is already registered
    with *different* content the root is renamed deterministically -- first by adding the
    proposal mode, then the cell ordinal -- and the renamed spec is what the shard records
    carry. Roots whose name is free or already identical are untouched, so every
    previously materialised scene keeps its original id byte for byte.
    """

    scene = root.scene
    existing = registry.try_get(scene.name)
    if existing is None or scene_to_dict(existing) == scene_to_dict(scene):
        registry.add(scene)
        return root
    for candidate in (
        f"{root.family}_{root.mode}_r{root.seed:09d}",
        f"{root.family}_{root.mode}_r{root.seed:09d}_o{root.ordinal:05d}",
    ):
        existing = registry.try_get(candidate)
        if existing is not None and scene_to_dict(existing) != scene_to_dict(scene):
            continue
        renamed = replace(scene, name=candidate)
        registry.add(renamed)
        return replace(root, scene=renamed, scene_id=candidate)
    raise RuntimeError(f"could not give root {root.root_id} a unique scene id")


class RootPlan:
    """Append-only, deterministic per-split root sequence."""

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, List[RootSpec]] = {}

    def _metadata_path(self, split: str) -> Path:
        return self.root_dir / f"roots_{split}.jsonl"

    def registry_path(self, split: str) -> Path:
        return self.root_dir / f"scenes_{split}.json"

    def _read(self, split: str) -> List[RootSpec]:
        """Read the plan, dropping repeated rows.

        The file is append-only, so a previous bug (``ensure`` re-sampling the plan from zero
        and appending it again) could leave duplicate ``root_id`` rows behind. Reading keeps
        the **first** occurrence of each ``root_id``, which is exactly the canonical plan
        prefix, so a corrupted file still yields the correct sequence.
        """

        path = self._metadata_path(split)
        if not path.exists():
            return []
        registry = SceneRegistry.load(self.registry_path(split))
        roots: List[RootSpec] = []
        seen = set()
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                root_id = str(row["root_id"])
                if root_id in seen:
                    continue
                seen.add(root_id)
                roots.append(
                    RootSpec(
                        root_id=root_id,
                        scene_id=str(row["scene_id"]),
                        split=str(row["split"]),
                        family=str(row["family"]),
                        mode=str(row["mode"]),
                        ordinal=int(row["ordinal"]),
                        seed=int(row["seed"]),
                        scene=registry.get(str(row["scene_id"])),
                        attempts=int(row["attempts"]),
                    )
                )
        return roots

    def ensure(self, split: str, count: int) -> List[RootSpec]:
        """Materialise at least ``count`` roots and return exactly the first ``count``.

        The plan is a pure function of ``(split, index)``. The store is extended by taking the
        canonical plan's **next** block, ``sample_roots(split, count)[len(known):count]``, and
        appending only that block. Re-sampling from zero and appending everything (the
        behaviour this replaces) would have silently duplicated the whole prefix.
        """

        if split not in self._cache:
            self._cache[split] = self._read(split)
        roots = self._cache[split]
        if len(roots) < count:
            canonical = sample_roots(split, count)
            if len(canonical) < count:
                raise RuntimeError(f"plan for {split} only produced {len(canonical)} of {count} roots")
            extension = canonical[len(roots):count]
            if [root.root_id for root in canonical[: len(roots)]] != [root.root_id for root in roots]:
                raise RuntimeError(
                    f"stored {split} plan disagrees with the canonical plan; refusing to extend"
                )
            registry = (
                SceneRegistry.load(self.registry_path(split))
                if self.registry_path(split).exists() else SceneRegistry()
            )
            for position, root in enumerate(extension):
                extension[position] = resolve_scene(registry, root)
            registry.save(self.registry_path(split))
            with self._metadata_path(split).open("a") as handle:
                for root in extension:
                    handle.write(json.dumps(root.as_metadata(), sort_keys=True) + "\n")
            roots = roots + extension
            self._cache[split] = roots
        return roots[:count]

    def peek(self, split: str) -> List[RootSpec]:
        if split not in self._cache:
            self._cache[split] = self._read(split)
        return self._cache[split]


def load_registries(root_dir: Path, splits: Sequence[str]) -> SceneRegistry:
    """Merge the per-split scene registries into one and activate it."""

    merged = SceneRegistry()
    for split in splits:
        path = Path(root_dir) / f"scenes_{split}.json"
        if not path.exists():
            continue
        for scene in SceneRegistry.load(path):
            merged.add(scene)
    return merged


def scene_of(root_dir: Path, split: str, scene_id: str) -> SceneSpec:
    return SceneRegistry.load(Path(root_dir) / f"scenes_{split}.json").get(scene_id)


#: The six original training initial states, used by the equal-size narrow-coverage control.
NARROW_SCENES = ("S0_dev", "S1_dev", "S0_v1", "S1_v1", "S0_v2", "S1_v2")


class NarrowPlan:
    """A ``RootPlan``-compatible source of collection *rounds* over the six old initial states.

    Each round is one collection episode of the frozen B1 recipe started from one of the six
    fixed initial states. The initial state is never altered; extra data comes only from extra
    rounds, so the number of *independent initial states* stays exactly six and is reported.
    """

    def __init__(self, root_dir: Path):
        self.root_dir = Path(root_dir)

    def ensure(self, split: str, count: int) -> List[RootSpec]:
        from ..continuous2d.scenes import scene_by_name
        from .sampler import RootSpec

        roots: List[RootSpec] = []
        for index in range(count):
            slot = index % len(NARROW_SCENES)
            round_index = index // len(NARROW_SCENES)
            scene = scene_by_name(NARROW_SCENES[slot])
            roots.append(
                RootSpec(
                    root_id=f"narrow/{NARROW_SCENES[slot]}/{round_index:05d}",
                    scene_id=scene.name,
                    split="narrow",
                    family=scene.category,
                    mode="fixed",
                    ordinal=index,
                    seed=index,
                    scene=scene,
                    attempts=1,
                )
            )
        return roots[:count]

    def peek(self, split: str) -> List[RootSpec]:
        return self.ensure(split, 0)
