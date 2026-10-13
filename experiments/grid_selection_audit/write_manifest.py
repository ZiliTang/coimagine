"""Regenerate ``docs/migration_v1/manifest.json`` from the live repository.

Read-only with respect to every legacy asset: it hashes checkpoints and records git
state, versions and the frozen defaults. Run it again after the continuous defaults
change so the manifest always describes the current tree.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CHECKPOINTS = (
    "outputs/grid/world_model/belief.npz",
    "outputs/grid/world_model/belief_band.npz",
    "outputs/grid/world_model/belief_big.npz",
    "outputs/grid/world_model/belief_exploration.npz",
    "outputs/grid/world_model/direct.npz",
    "outputs/grid/world_model/latent.npz",
    "outputs/grid/world_model/latent_released.npz",
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


def _versions() -> dict:
    import matplotlib
    import numpy
    import pymunk
    import yaml

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "PyYAML": yaml.__version__,
        "pymunk": pymunk.version,
    }


def build() -> dict:
    dirty = [line for line in _git("status", "--porcelain").splitlines() if line]
    manifest = {
        "generated_by": "experiments/grid_selection_audit/write_manifest.py",
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "dirty_files": dirty,
            "note": "HEAD is read live; no historical commit is assumed.",
        },
        "checkpoints": {
            path: {
                "sha256": _sha256(ROOT / path),
                "size": (ROOT / path).stat().st_size,
                "role": (
                    "primary frozen checkpoint for the legacy-grid selection audit"
                    if path == "outputs/grid/world_model/belief.npz"
                    else "available legacy checkpoint (read-only)"
                ),
            }
            for path in CHECKPOINTS
        },
        "versions": _versions(),
        "legacy_grid": {
            "scene_library": "coimagine.grid.scenes (open_transport, pillar_occlusion, "
            "narrow_passage, corner_goal; mechanism scenes A0-A3)",
            "action_order": ["push:0:E", "push:1:E", "co_push:E", "probe:0:E", "release:0:E", "wait"],
            "step_cost": {"push/release": 1.0, "probe": 0.3, "hold/wait": 0.0},
            "observation_dim": 188,
            "dynamic_dim": 41,
            "physics_types": 16,
            "success": "the 2x2 object's cells are a subset of the goal region (absorbing)",
            "scalar_return": "J = 10*1[in goal] - cumulative_cost - 0.5*terminal_manhattan_distance",
        },
        "continuous2d_defaults": _continuous_defaults(),
        "commands": {
            "tests": "source .venv/bin/activate && pytest",
            "phase_A_check": "python experiments/grid_selection_audit/phase_a_check.py",
            "phase_B_audit": "python experiments/grid_selection_audit/run_audit.py",
            "phase_C_physics": "python experiments/continuous2d/run_physics_acceptance.py",
            "phase_D_baseline": "python experiments/continuous2d/run_baseline.py",
        },
    }
    return manifest


def _continuous_defaults() -> dict:
    """Read the continuous config if Phase C exists, else record it as pending."""

    try:
        from coimagine.continuous2d.config import default_config

        return default_config().as_dict()
    except Exception as error:  # pragma: no cover - before Phase C lands
        return {"status": "pending Phase C", "detail": repr(error)}


def main() -> None:
    target = ROOT / "docs" / "migration_v1" / "manifest.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(build(), indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {target.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
