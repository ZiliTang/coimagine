"""Write ``docs/data_upgrade_v1/manifest.json`` from the finished run."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

SOURCE_FILES = (
    "src/coimagine/data_upgrade/__init__.py",
    "src/coimagine/data_upgrade/branch_data.py",
    "src/coimagine/data_upgrade/coverage.py",
    "src/coimagine/data_upgrade/fit_check.py",
    "src/coimagine/data_upgrade/fixed_candidates.py",
    "src/coimagine/data_upgrade/geometry_audit.py",
    "src/coimagine/data_upgrade/group_eval.py",
    "src/coimagine/data_upgrade/metrics.py",
    "src/coimagine/data_upgrade/training.py",
    "src/coimagine/continuous2d_learning/closed_loop.py",
    "src/coimagine/continuous2d_learning/contract.py",
    "src/coimagine/continuous2d_learning/diagnostic.py",
    "src/coimagine/continuous2d_learning/model.py",
    "src/coimagine/continuous2d_learning/rollout.py",
    "src/coimagine/deterministic_grid/audit.py",
    "experiments/data_upgrade/run_branch_collection.py",
    "experiments/data_upgrade/run_cases.py",
    "experiments/data_upgrade/run_coverage.py",
    "experiments/data_upgrade/run_fit_check.py",
    "experiments/data_upgrade/run_fixed_candidates.py",
    "experiments/data_upgrade/run_geometry_audit.py",
    "experiments/data_upgrade/run_group_closed_loop.py",
    "experiments/data_upgrade/run_group_eval.py",
    "experiments/data_upgrade/run_old_model_audit.py",
    "experiments/data_upgrade/run_selfcheck.py",
    "experiments/data_upgrade/run_training_groups.py",
    "experiments/deterministic_grid/run_audit_v2.py",
    "tests/test_data_upgrade.py",
    "tests/test_grid_audit_v2.py",
    "docs/data_upgrade_v1/CONTRACT.md",
    "docs/data_upgrade_v1/STATUS.md",
    "docs/data_upgrade_v1/report.md",
    "docs/data_upgrade_v1/errata.md",
    "docs/data_upgrade_v1/A1_A2_grid_fixes.md",
    "docs/data_upgrade_v1/A_grid_audit_v2.md",
)

RUN_FILES = (
    "geometry/summary.json", "geometry/true_state_check.json", "geometry/engine_agreement.json",
    "geometry/convention_probe.json", "geometry/predicted_state_check.json",
    "fit_check/fit_check.json",
    "old_models/old_model_metrics.json",
    "branch_data/collection_summary.json", "branch_data/guides.json",
    "coverage/coverage.json", "coverage/coverage_table.md",
    "training_groups.json", "training_groups_originalval.json",
    "group_eval/group_eval_combined.json", "group_eval/group_eval_origval.json",
    "fixed_candidates/reference_verification.json",
    "fixed_candidates/aggregate_combined.json", "fixed_candidates/aggregate_origval.json",
    "fixed_candidates/metrics_table_combined.md", "fixed_candidates/metrics_table_origval.md",
    "closed_loop/aggregate.json", "closed_loop/summary.json",
    "closed_loop_groups_combined/aggregate.json", "closed_loop_groups_combined/summary.json",
    "closed_loop_groups_origval/aggregate.json", "closed_loop_groups_origval/summary.json",
    "cases/cases_origval.json", "selfcheck.json",
)

LARGE_FILES = (
    "branch_data/branch_windows.jsonl", "branch_data/random_add_windows.jsonl",
    "branch_data/guide_windows.jsonl", "branch_data/branch_train_matched.jsonl",
    "branch_data/random_train_matched.jsonl", "branch_data/anchors.jsonl",
    "fixed_candidates/rows_combined.jsonl", "fixed_candidates/rows_origval.jsonl",
    "closed_loop_groups_combined/episodes.jsonl", "closed_loop_groups_combined/windows.jsonl",
    "closed_loop_groups_origval/episodes.jsonl", "closed_loop_groups_origval/windows.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id

    from coimagine.continuous2d.config import default_config
    from coimagine.continuous2d.legality import legality_report, scene_content_hash
    from coimagine.continuous2d.scenes import default_scenes
    from coimagine.continuous2d_learning import SCORE_VERSION
    from coimagine.continuous2d_learning.contract import (
        CHAIN_ANGLE_TOLERANCE, CHAIN_POSITION_TOLERANCE, DURATION_SCALE, FAILURE_PENALTY,
        HOLD_SCALE, SUCCESS_REWARD, TIME_DIVISOR, ANGULAR_VELOCITY_SCALE, VELOCITY_SCALE,
    )
    from coimagine.continuous2d_learning.model import ModelConfig
    from coimagine.continuous2d_learning.records import SPLIT_SCENES

    config = default_config()
    manifest = {
        "run_id": args.run_id,
        "contract": "docs/data_upgrade_v1/CONTRACT.md",
        "git_head": _git(["rev-parse", "HEAD"]),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"]),
        "git_status": _git(["status", "--porcelain"]).splitlines(),
        "python": platform.python_version(),
        "source_sha256": {},
        "continuous2d_config": config.as_dict(),
        "score": {"version": SCORE_VERSION, "success_reward": SUCCESS_REWARD,
                  "time_divisor": TIME_DIVISOR, "failure_penalty": FAILURE_PENALTY},
        "normalisation": {"velocity_scale": VELOCITY_SCALE, "angular_velocity_scale": ANGULAR_VELOCITY_SCALE,
                          "hold_scale": HOLD_SCALE, "duration_scale": DURATION_SCALE},
        "chain_tolerances": {"position_m": CHAIN_POSITION_TOLERANCE, "angle_rad": CHAIN_ANGLE_TOLERANCE},
        "model_config": ModelConfig().as_dict(),
        "split_scenes": {key: list(value) for key, value in SPLIT_SCENES.items()},
        "scenes": [],
        "checkpoint_rules": {
            "primary": "0.5 * mean original val loss + 0.5 * mean new-branch val loss",
            "secondary": "mean original val loss only (uniform across groups; the primary rule degenerates for groups without branch training data)",
        },
    }
    try:
        import numpy, pymunk, torch
        manifest["dependency_versions"] = {"numpy": numpy.__version__, "torch": torch.__version__, "pymunk": pymunk.version}
    except Exception as error:
        manifest["dependency_versions"] = {"error": str(error)}

    for relative in SOURCE_FILES:
        path = ROOT / relative
        manifest["source_sha256"][relative] = _sha256(path) if path.exists() else "MISSING"
    for scene in default_scenes():
        report = legality_report(scene)
        manifest["scenes"].append({"name": scene.name, "scene_version": scene.scene_version,
                                   "content_hash": scene_content_hash(scene), "legal": report["legal"],
                                   "min_clearance": report["min_clearance"]})
    manifest["run_file_sha256"] = {}
    for name in RUN_FILES:
        path = run_dir / name
        manifest["run_file_sha256"][name] = _sha256(path) if path.exists() else "NOT_RUN"
    manifest["large_files_not_in_bundle"] = {
        name: (_sha256(run_dir / name) if (run_dir / name).exists() else "NOT_RUN") for name in LARGE_FILES
    }
    manifest["checkpoints"] = {}
    for subdir in ("models", "models_originalval"):
        for path in sorted((run_dir / subdir).glob("*.pt")):
            manifest["checkpoints"][f"{subdir}/{path.name}"] = _sha256(path)
    grid = ROOT / "outputs/data_upgrade_v1/grid-audit-v2"
    manifest["grid_run_file_sha256"] = {
        name: (_sha256(grid / name) if (grid / name).exists() else "NOT_RUN")
        for name in ("summary.json", "hash_check.json", "metrics_table_v2.md", "decomposition.json")
    }
    destination = ROOT / "docs/data_upgrade_v1/manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print("wrote", destination)
    print("missing sources:", [k for k, v in manifest["source_sha256"].items() if v == "MISSING"])
    print("NOT_RUN run files:", [k for k, v in manifest["run_file_sha256"].items() if v == "NOT_RUN"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
