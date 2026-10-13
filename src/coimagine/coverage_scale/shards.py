"""Sharded, crash-safe storage for the collected windows (task book B2/B7).

One shard covers a fixed block of whole roots of the plan, so a shard is always a complete-root
prefix slice. A shard only counts once its ``_done`` marker exists; that marker is written
last, atomically, after the window file and its metadata are on disk. Workers never write the
same shard, so no cross-process lock is needed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from ..continuous2d.config import Continuous2DConfig
from ..continuous2d.environment import Continuous2DEnv
from .plan import RootPlan
from .recipe import RECIPE_VERSION, collect_root
from .sampler import RootSpec

ROOTS_PER_SHARD = 16
SHARD_LAYOUT_VERSION = "coverage-scale-shards-v1"


def shard_dir(root_dir: Path, split: str) -> Path:
    return Path(root_dir) / "shards" / split


def shard_count_for(roots: int) -> int:
    return (roots + ROOTS_PER_SHARD - 1) // ROOTS_PER_SHARD


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def done_marker(root_dir: Path, split: str, shard: int) -> Path:
    return shard_dir(root_dir, split) / f"_done_{shard:05d}.json"


def is_done(root_dir: Path, split: str, shard: int) -> bool:
    return done_marker(root_dir, split, shard).exists()


def scan_done(root_dir: Path, split: str) -> List[int]:
    directory = shard_dir(root_dir, split)
    if not directory.exists():
        return []
    indices = []
    for path in directory.glob("_done_*.json"):
        try:
            indices.append(int(path.stem.split("_")[-1]))
        except ValueError:  # pragma: no cover - malformed marker
            continue
    return sorted(indices)


def collect_shard(
    root_dir: Path,
    split: str,
    shard: int,
    config: Continuous2DConfig,
    plan: Optional[RootPlan] = None,
    *,
    force: bool = False,
) -> Dict[str, object]:
    """Collect (or re-collect) one shard of whole roots. Returns the shard metadata."""

    if is_done(root_dir, split, shard) and not force:
        return json.loads(done_marker(root_dir, split, shard).read_text())

    plan = plan or RootPlan(root_dir)
    start = shard * ROOTS_PER_SHARD
    stop = start + ROOTS_PER_SHARD
    roots: Sequence[RootSpec] = plan.ensure(split, stop)

    live = Continuous2DEnv(config)
    planner = Continuous2DEnv(config)
    directory = shard_dir(root_dir, split)
    directory.mkdir(parents=True, exist_ok=True)
    windows_path = directory / f"shard_{shard:05d}.jsonl"
    temporary = windows_path.with_suffix(".jsonl.tmp")

    stats: List[Dict[str, object]] = []
    count = 0
    with temporary.open("w") as handle:
        for root in roots[start:stop]:
            payload = collect_root(root, config, live=live, planner=planner)
            for window in payload["windows"]:
                handle.write(json.dumps(window, sort_keys=True) + "\n")
                count += 1
            stats.append(payload["stats"])
    os.replace(temporary, windows_path)

    meta = {
        "layout_version": SHARD_LAYOUT_VERSION,
        "recipe_version": RECIPE_VERSION,
        "split": split,
        "shard": shard,
        "roots": [root.root_id for root in roots[start:stop]],
        "root_count": len(stats),
        "windows": count,
        "stats": stats,
        "config_hash": config.config_hash(),
    }
    _atomic_write_text(directory / f"shard_{shard:05d}.meta.json", json.dumps(meta, indent=1, sort_keys=True))
    _atomic_write_text(done_marker(root_dir, split, shard), json.dumps(
        {"shard": shard, "split": split, "windows": count, "root_count": len(stats)}
    ))
    return meta


def iter_shard_records(root_dir: Path, split: str, shard: int) -> Iterator[Dict[str, object]]:
    path = shard_dir(root_dir, split) / f"shard_{shard:05d}.jsonl"
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def shard_records(root_dir: Path, split: str, shard: int) -> List[Dict[str, object]]:
    return list(iter_shard_records(root_dir, split, shard))


def root_grouped(records: Sequence[Dict[str, object]]) -> List[Tuple[str, List[Dict[str, object]]]]:
    """Group a contiguous record stream by its root, preserving order."""

    grouped: List[Tuple[str, List[Dict[str, object]]]] = []
    for record in records:
        root_id = str(record["root_id"])
        if not grouped or grouped[-1][0] != root_id:
            grouped.append((root_id, []))
        grouped[-1][1].append(record)
    return grouped


def contiguous_done(root_dir: Path, split: str) -> List[int]:
    """The longest *gap-free* prefix ``0, 1, ...`` of finished shards.

    Shards finish out of order because the workers take every ``n``-th task of the queue, so
    ``scan_done`` can contain holes. Concatenating a holed set would silently skip a block of
    roots and would make the tier prefix change as later shards complete. Only the gap-free
    prefix is a stable, strictly nested sequence, so that is what every tier is built from.
    """

    indices: List[int] = []
    shard = 0
    while is_done(root_dir, split, shard):
        indices.append(shard)
        shard += 1
    return indices


def tier_records(
    root_dir: Path, split: str, target_windows: int, *, dedupe: bool = True
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """The shortest complete-root prefix of the split reaching ``target_windows``.

    Only the gap-free prefix of finished shards is read, and the cut is always at a root
    boundary, so a larger target's record list is a strict extension of a smaller target's list
    and the same prefix is returned no matter how many further shards have finished since.
    """

    done = contiguous_done(root_dir, split)
    collected: List[Dict[str, object]] = []
    seen = set()
    duplicates = 0
    shards_used = 0
    previous_root: Optional[str] = None
    cut = False
    for shard in done:
        for record in iter_shard_records(root_dir, split, shard):
            root_id = str(record["root_id"])
            if root_id != previous_root:
                # A root boundary. Stop only here, so the cut is always a whole-root prefix.
                if previous_root is not None and len(collected) >= target_windows:
                    cut = True
                    break
                previous_root = root_id
            if dedupe:
                key = str(record.get("transition_id", ""))
                if key and key in seen:
                    duplicates += 1
                    continue
                if key:
                    seen.add(key)
            collected.append(record)
        shards_used += 1
        if cut:
            break
    roots_included = len({str(record["root_id"]) for record in collected})
    info = {
        "target_windows": target_windows,
        "actual_windows": len(collected),
        "independent_roots": roots_included,
        "shards_used": shards_used,
        "shards_available": len(done),
        "duplicate_transitions_removed": duplicates,
        "reached_target": len(collected) >= target_windows,
    }
    return collected, info


def select_monitor(
    records: Sequence[Dict[str, object]], limit: int = 4096
) -> List[Dict[str, object]]:
    """A fixed ``train_monitor`` subset: whole roots in ascending root-hash order, cut at ``limit``.

    The cut is at an exact window count so the subset is nested across tiers; the last root is
    truncated, which keeps the size at or below the frozen ``limit`` instead of overshooting it
    by a whole root.
    """

    import hashlib

    grouped = root_grouped(records)
    ordered = sorted(
        grouped,
        key=lambda item: hashlib.sha256(f"coverage-scale-monitor-v1|{item[0]}".encode()).hexdigest(),
    )
    selected: List[Dict[str, object]] = []
    for _, windows in ordered:
        if len(selected) >= limit:
            break
        selected.extend(windows)
    return selected[:limit]
