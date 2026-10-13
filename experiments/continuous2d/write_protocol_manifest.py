"""Write ``docs/protocol_fix_v1/manifest.json`` for this round (read-only over the repo).

    python experiments/continuous2d/write_protocol_manifest.py --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]

KEY_SOURCES = (
    "src/coimagine/continuous2d/backend.py",
    "src/coimagine/continuous2d/environment.py",
    "src/coimagine/continuous2d/mpc.py",
    "src/coimagine/continuous2d/data.py",
    "src/coimagine/continuous2d/skills.py",
    "src/coimagine/continuous2d/candidates.py",
    "src/coimagine/continuous2d/scoring.py",
    "src/coimagine/continuous2d/config.py",
    "src/coimagine/diagnostics/grid_selection_audit.py",
    "experiments/continuous2d/run_baseline.py",
    "experiments/continuous2d/run_physics_acceptance.py",
    "experiments/continuous2d/compare_baselines.py",
    "experiments/grid_selection_audit/write_errata.py",
    "tests/test_protocol_fix.py",
    "tests/test_continuous2d.py",
    "pyproject.toml",
)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _versions() -> Dict[str, str]:
    import matplotlib
    import numpy
    import pymunk
    import pytest
    import yaml

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "PyYAML": yaml.__version__,
        "pymunk": pymunk.version,
        "pytest": pytest.__version__,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out", default="docs/protocol_fix_v1/manifest.json")
    args = parser.parse_args()

    from coimagine.continuous2d import default_config

    config = default_config()
    old_manifest_path = ROOT / "docs/migration_v1/manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text()) if old_manifest_path.exists() else {}
    checkpoint = ROOT / "outputs/grid/world_model/belief.npz"
    run_dir = ROOT / "outputs/protocol_fix_v1" / args.run_id

    manifest = {
        "generated_by": "experiments/continuous2d/write_protocol_manifest.py",
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "dirty_files": [line for line in _git("status", "--porcelain").splitlines() if line],
        "versions": _versions(),
        "config_hash": config.config_hash(),
        "schema_version": 2,
        "sources_sha256": {path: _sha256(ROOT / path) for path in KEY_SOURCES if (ROOT / path).exists()},
        "checkpoints": {
            "outputs/grid/world_model/belief.npz": {
                "sha256": _sha256(checkpoint),
                "read_only": True,
            }
        },
        "old_manifest_versions": old_manifest.get("versions", {}),
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.exists() else None,
        "run_dir_contents": sorted(
            str(path.relative_to(ROOT)) for path in run_dir.rglob("*") if path.is_file()
        ) if run_dir.exists() else [],
        "protocol_changes": [
            "snapshot carries scene_config, physics, success_counter, termination, deep-copied controller_state and the applied-control prefix",
            "restore defaults to exact prefix replay with per-quantity tolerance verification; approximate clone renamed approximate_restore and unused by the experiment",
            "live and planner environments separated; live executes directly and is never rolled out",
            "success absorbed in search; physical failures terminal and not expanded; explicit planner_no_feasible_plan",
            "schema_version=2 structured skills and both robots' applied control",
            "wall/boundary contacts included in penetration, reported per collision class",
            "pymunk==6.9.0 declared in pyproject dependencies",
            "Grid conclusion generator writes a factual table and defers mechanism judgement",
        ],
        "unchanged_scientific_conditions": [
            "physics parameters, object size/mass, friction, damping, time steps, force limit",
            "scene layouts, skill control law, candidate template and order",
            "depth (1/3), beam (3), episode horizon (12), success tolerances, original scoring and passage_cost_to_go formula",
            "the Grid model, dynamics, scoring, candidates and all old raw results/checkpoints",
        ],
    }
    target = ROOT / args.out
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: manifest[k] for k in ("head", "config_hash", "schema_version", "run_dir")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
