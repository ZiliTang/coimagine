"""Write ``docs/forecast_baseline_v1/manifest.json`` from the finished run.

    python experiments/forecast_baseline/write_manifest.py --run-id <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILES = (
    "src/coimagine/continuous2d_learning/__init__.py",
    "src/coimagine/continuous2d_learning/contract.py",
    "src/coimagine/continuous2d_learning/features.py",
    "src/coimagine/continuous2d_learning/records.py",
    "src/coimagine/continuous2d_learning/model.py",
    "src/coimagine/continuous2d_learning/evaluation.py",
    "src/coimagine/continuous2d_learning/rollout.py",
    "src/coimagine/continuous2d_learning/pools.py",
    "src/coimagine/continuous2d_learning/starts.py",
    "src/coimagine/continuous2d_learning/diagnostic.py",
    "src/coimagine/continuous2d_learning/closed_loop.py",
    "src/coimagine/deterministic_grid/audit.py",
    "experiments/forecast_baseline/collect_data.py",
    "experiments/forecast_baseline/train_models.py",
    "experiments/forecast_baseline/model_eval.py",
    "experiments/forecast_baseline/run_diagnostic.py",
    "experiments/forecast_baseline/run_closed_loop.py",
    "experiments/deterministic_grid/run_audit_v2.py",
    "tests/test_forecast_baseline.py",
    "tests/test_grid_audit_v2.py",
    "docs/forecast_baseline_v1/CONTRACT.md",
    "docs/forecast_baseline_v1/STATUS.md",
    "docs/forecast_baseline_v1/report.md",
    "docs/forecast_baseline_v1/errata.md",
    "docs/forecast_baseline_v1/A_grid_audit_v2.md",
    "experiments/review_bundle/make_forecast_baseline_bundle.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--out-root", default="outputs/forecast_baseline_v1")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    from coimagine.continuous2d.config import default_config
    from coimagine.continuous2d.legality import legality_report, scene_content_hash
    from coimagine.continuous2d.scenes import default_scenes
    from coimagine.continuous2d_learning import SCORE_VERSION
    from coimagine.continuous2d_learning.contract import (
        DURATION_SCALE,
        FAILURE_PENALTY,
        HOLD_SCALE,
        SUCCESS_REWARD,
        TIME_DIVISOR,
        VELOCITY_SCALE,
        ANGULAR_VELOCITY_SCALE,
    )
    from coimagine.continuous2d_learning.records import SPLIT_SCENES, CollectionConfig
    from coimagine.continuous2d_learning.model import ModelConfig

    config = default_config()
    manifest = {
        "run_id": args.run_id,
        "contract": "docs/forecast_baseline_v1/CONTRACT.md",
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _git(["status", "--porcelain"]).splitlines(),
        "python": platform.python_version(),
        "dependency_versions": {},
        "source_sha256": {},
        "scenes": [],
        "continuous2d_config": config.as_dict(),
        "score": {
            "version": SCORE_VERSION,
            "success_reward": SUCCESS_REWARD,
            "time_divisor": TIME_DIVISOR,
            "failure_penalty": FAILURE_PENALTY,
        },
        "normalisation": {
            "velocity_scale": VELOCITY_SCALE,
            "angular_velocity_scale": ANGULAR_VELOCITY_SCALE,
            "hold_scale": HOLD_SCALE,
            "duration_scale": DURATION_SCALE,
            "positions": "arena-relative, angles as sin/cos, angle deltas wrapped by pi",
        },
        "collection": CollectionConfig().as_dict(),
        "model_config": ModelConfig().as_dict(),
        "split_scenes": {key: list(value) for key, value in SPLIT_SCENES.items()},
    }
    try:
        import numpy
        import pymunk
        import torch

        manifest["dependency_versions"] = {
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "pymunk": pymunk.version,
        }
    except Exception as error:  # pragma: no cover
        manifest["dependency_versions"] = {"error": str(error)}

    for relative in SOURCE_FILES:
        path = ROOT / relative
        manifest["source_sha256"][relative] = _sha256(path) if path.exists() else "MISSING"

    for scene in default_scenes():
        report = legality_report(scene)
        manifest["scenes"].append(
            {
                "name": scene.name,
                "category": scene.category,
                "variant": scene.variant,
                "scene_version": scene.scene_version,
                "content_hash": scene_content_hash(scene),
                "legal": report["legal"],
                "min_clearance": report["min_clearance"],
            }
        )

    for name in ("dataset/windows.jsonl", "dataset/episodes.jsonl", "dataset/manifest.json",
                 "dataset_summary.json", "training_summary.json", "test_metrics.json",
                 "diagnostic/coverage.json", "diagnostic/aggregate.json",
                 "diagnostic/summary.json", "closed_loop/aggregate.json"):
        path = run_dir / name
        manifest.setdefault("run_file_sha256", {})[name] = _sha256(path) if path.exists() else "NOT_RUN"

    models_dir = run_dir / "models"
    manifest["checkpoints"] = {}
    if models_dir.exists():
        for path in sorted(models_dir.glob("seed*.pt")):
            manifest["checkpoints"][path.name] = _sha256(path)

    coverage_path = run_dir / "diagnostic" / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text())
        manifest["candidate_pool"] = {
            "source": "existing joint_candidates (public observation only)",
            "cap": 128,
            "rules": "H=1 dedup root then seed-shuffle; H>1 = 32 repeated root skills + 32 staging-then-push templates + uniform random fill",
            "verdict": coverage["summary"]["verdict"],
            "starts_with_d_lowering_feasible": coverage["summary"]["starts_with_d_lowering_feasible"],
            "n_starts": coverage["summary"]["n_starts"],
        }

    destination = ROOT / "docs" / "forecast_baseline_v1" / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {destination}")
    print(json.dumps({key: manifest[key] for key in ("git_head", "score")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
