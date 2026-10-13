"""Build ``data_upgrade_v1_review.zip`` (read-only; nothing is re-run)."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "data_upgrade_v1_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_data_upgrade_bundle"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

DIRECTORIES = (
    "src/coimagine/data_upgrade",
    "src/coimagine/continuous2d_learning",
    "experiments/data_upgrade",
)
FILES = (
    "src/coimagine/deterministic_grid/audit.py",
    "src/coimagine/diagnostics/grid_audit_v2.py",
    "experiments/deterministic_grid/run_audit_v2.py",
    "experiments/forecast_baseline/run_closed_loop.py",
    "tests/test_data_upgrade.py",
    "tests/test_grid_audit_v2.py",
    "tests/test_continuous2d.py",
    "docs/data_upgrade_v1/CONTRACT.md",
    "docs/data_upgrade_v1/STATUS.md",
    "docs/data_upgrade_v1/report.md",
    "docs/data_upgrade_v1/errata.md",
    "docs/data_upgrade_v1/A1_A2_grid_fixes.md",
    "docs/data_upgrade_v1/A_grid_audit_v2.md",
    "docs/data_upgrade_v1/manifest.json",
    "docs/forecast_baseline_v1/STATUS.md",
    "pyproject.toml",
)
RUN_FILES = (
    "geometry/summary.json", "geometry/true_state_check.json", "geometry/engine_agreement.json",
    "geometry/convention_probe.json", "geometry/predicted_state_check.json",
    "geometry/misclassified_true_states.jsonl",
    "fit_check/fit_check.json",
    "old_models/old_model_metrics.json",
    "branch_data/collection_summary.json", "branch_data/guides.json",
    "coverage/coverage.json", "coverage/coverage_table.md",
    "training_groups.json", "training_groups_originalval.json",
    "group_eval/group_eval_combined.json", "group_eval/group_eval_origval.json",
    "fixed_candidates/reference_verification.json",
    "fixed_candidates/aggregate_combined.json", "fixed_candidates/aggregate_origval.json",
    "fixed_candidates/metrics_table_combined.md", "fixed_candidates/metrics_table_origval.md",
    "closed_loop/aggregate.json", "closed_loop/summary.json", "closed_loop/episodes.jsonl",
    "closed_loop_groups_combined/aggregate.json", "closed_loop_groups_combined/episodes.jsonl",
    "closed_loop_groups_origval/aggregate.json", "closed_loop_groups_origval/episodes.jsonl",
    "cases/cases_origval.json", "selfcheck.json",
    "branch_data/anchors.jsonl",
)
CHECKPOINTS = (
    "models/Old_seed0.pt", "models/Old_Random_seed0.pt", "models/Old_Branch_seed0.pt",
    "models/Old_seed1.pt", "models/Old_Random_seed1.pt", "models/Old_Branch_seed1.pt",
    "models/Old_seed2.pt", "models/Old_Random_seed2.pt", "models/Old_Branch_seed2.pt",
)
LARGE_FILES = (
    "branch_data/branch_windows.jsonl", "branch_data/random_add_windows.jsonl",
    "branch_data/guide_windows.jsonl", "branch_data/branch_train_matched.jsonl",
    "branch_data/random_train_matched.jsonl",
    "fixed_candidates/rows_combined.jsonl", "fixed_candidates/rows_origval.jsonl",
    "closed_loop_groups_combined/windows.jsonl", "closed_loop_groups_origval/windows.jsonl",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(relative: str, stage: Path, copied: List[str]) -> None:
    source = ROOT / relative
    if not source.exists():
        return
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    copied.append(relative)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    args = parser.parse_args()
    run = ROOT / args.out_root / args.run_id

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    copied: List[str] = []
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*.py")):
            if any(part in SKIP_DIRS for part in path.parts):
                continue
            _copy(str(path.relative_to(ROOT)), STAGE, copied)
    for relative in FILES:
        _copy(relative, STAGE, copied)
    for relative in RUN_FILES:
        _copy(str((run / relative).relative_to(ROOT)), STAGE, copied)
    for relative in CHECKPOINTS:
        _copy(str((run / relative).relative_to(ROOT)), STAGE, copied)
    grid = ROOT / "outputs/data_upgrade_v1/grid-audit-v2"
    for name in ("summary.json", "hash_check.json", "metrics_table_v2.md", "group_stats.json",
                 "decomposition.json", "traced_cases.json", "recursive_vs_teacher.json", "eval_states_v2.json"):
        _copy(str((grid / name).relative_to(ROOT)), STAGE, copied)

    missing = {
        "not_copied_large_files": [
            {"path": str((run / name).relative_to(ROOT)),
             "sha256": _sha256(run / name) if (run / name).exists() else "MISSING",
             "restore": "python experiments/data_upgrade/run_branch_collection.py --run-id data-upgrade-20261012 (branch_data) "
                        "or python experiments/data_upgrade/run_fixed_candidates.py --run-id data-upgrade-20261012 (rows)"}
            for name in LARGE_FILES
        ],
        "not_copied_checkpoints": [
            {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
            for path in sorted((run / "models_originalval").glob("*.pt"))
        ],
        "note": "Every number in docs/data_upgrade_v1/ comes from the copied run files or from the listed large files.",
    }
    (STAGE / "MISSING.md").write_text("# Not copied into this bundle\n\n" + json.dumps(missing, indent=2) + "\n")

    lines = [
        "# data_upgrade_v1 review bundle", "",
        "Fresh-run commands:", "", "```bash",
        "python experiments/data_upgrade/run_geometry_audit.py --run-id data-upgrade-20261012",
        "python experiments/data_upgrade/run_fit_check.py --run-id data-upgrade-20261012",
        "python experiments/data_upgrade/run_old_model_audit.py --run-id data-upgrade-20261012",
        "python experiments/data_upgrade/run_branch_collection.py --run-id data-upgrade-20261012",
        "python experiments/data_upgrade/run_coverage.py --run-id data-upgrade-20261012",
        "python experiments/data_upgrade/run_training_groups.py --run-id data-upgrade-20261012",
        "python experiments/data_upgrade/run_training_groups.py --run-id data-upgrade-20261012 --val-branch off --model-subdir models_originalval",
        "python experiments/data_upgrade/run_group_eval.py --run-id data-upgrade-20261012 --tag combined",
        "python experiments/data_upgrade/run_group_eval.py --run-id data-upgrade-20261012 --model-subdir models_originalval --tag origval",
        "python experiments/data_upgrade/run_fixed_candidates.py --run-id data-upgrade-20261012 --tag combined --include-old",
        "python experiments/data_upgrade/run_fixed_candidates.py --run-id data-upgrade-20261012 --model-subdir models_originalval --tag origval",
        "python experiments/data_upgrade/run_group_closed_loop.py --run-id data-upgrade-20261012 --tag combined",
        "python experiments/data_upgrade/run_group_closed_loop.py --run-id data-upgrade-20261012 --model-subdir models_originalval --tag origval",
        "python experiments/data_upgrade/run_cases.py --run-id data-upgrade-20261012 --tag origval",
        "python experiments/data_upgrade/run_selfcheck.py --run-id data-upgrade-20261012",
        "pytest tests/test_data_upgrade.py tests/test_grid_audit_v2.py tests/test_forecast_baseline.py -q",
        "```", "", "## Files", "",
    ]
    hashes = {}
    for relative in sorted(copied):
        hashes[relative] = _sha256(STAGE / relative)
        lines.append(f"- `{relative}`")
    (STAGE / "REVIEW_INDEX.md").write_text("\n".join(lines) + "\n")
    (STAGE / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    bundle = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE))
    with zipfile.ZipFile(bundle) as archive:
        count = len(archive.namelist())
    print(json.dumps({"bundle": str(bundle), "files": count, "sha256": _sha256(bundle),
                      "size_bytes": bundle.stat().st_size}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
