"""Build ``forecast_baseline_review.zip`` from the finished run (read-only).

    python experiments/review_bundle/make_forecast_baseline_bundle.py --run-id <id>

The bundle carries the new source, tests, contract, manifest and the core result files.
Very large raw tables (the full dataset window table and the per-candidate record table) are
listed in ``MISSING.md`` with their in-repo path instead of being copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "forecast_baseline_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_forecast_bundle"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

DIRECTORIES = (
    "src/coimagine/continuous2d_learning",
    "experiments/forecast_baseline",
)
FILES = (
    "src/coimagine/deterministic_grid/audit.py",
    "src/coimagine/diagnostics/grid_audit_v2.py",
    "experiments/deterministic_grid/run_audit_v2.py",
    "tests/test_forecast_baseline.py",
    "tests/test_grid_audit_v2.py",
    "docs/forecast_baseline_v1/CONTRACT.md",
    "docs/forecast_baseline_v1/STATUS.md",
    "docs/forecast_baseline_v1/report.md",
    "docs/forecast_baseline_v1/errata.md",
    "docs/forecast_baseline_v1/A_grid_audit_v2.md",
    "docs/forecast_baseline_v1/manifest.json",
    "pyproject.toml",
    "AGENTS.md",
)

RUN_FILES = (
    "dataset/manifest.json",
    "dataset/coverage.json",
    "dataset/episodes.jsonl",
    "dataset_summary.json",
    "training_summary.json",
    "test_metrics.json",
    "diagnostic/config.json",
    "diagnostic/eval_starts.json",
    "diagnostic/summary.json",
    "diagnostic/aggregate.json",
    "diagnostic/coverage.json",
    "diagnostic/per_start.json",
    "diagnostic/recursive.json",
    "diagnostic/traced_cases.json",
    "diagnostic/metrics_table.md",
    "closed_loop/summary.json",
    "closed_loop/aggregate.json",
    "closed_loop/episodes.jsonl",
    "closed_loop/windows.jsonl",
)

LARGE_FILES = (
    "dataset/windows.jsonl",
    "diagnostic/per_run.jsonl",
    "diagnostic/candidate_records.jsonl",
)

GRID_RUN_FILES = (
    "group_stats.json",
    "eval_states_v2.json",
    "decomposition.json",
    "traced_cases.json",
    "recursive_vs_teacher.json",
    "summary.json",
    "hash_check.json",
    "metrics_table_v2.md",
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
    parser.add_argument("--run-id", default="forecast-20261012")
    parser.add_argument("--grid-run-id", default="grid-audit-v2")
    args = parser.parse_args()

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

    run = ROOT / "outputs/forecast_baseline_v1" / args.run_id
    for relative in RUN_FILES:
        _copy(str((run / relative).relative_to(ROOT)), STAGE, copied)
    grid_run = ROOT / "outputs/forecast_baseline_v1" / args.grid_run_id
    for name in GRID_RUN_FILES:
        _copy(str((grid_run / name).relative_to(ROOT)), STAGE, copied)

    missing = {
        "not_copied_large_files": [
            {
                "path": str((run / relative).relative_to(ROOT)),
                "sha256": _sha256(run / relative) if (run / relative).exists() else "MISSING",
                "reason": "large raw table; kept in the repository, reconstructible by the listed command",
            }
            for relative in LARGE_FILES
        ],
        "grid_run_missing": [
            name for name in GRID_RUN_FILES if not (grid_run / name).exists()
        ],
        "note": "Every result number in docs/forecast_baseline_v1/ comes from the run files above.",
    }
    (STAGE / "MISSING.md").write_text(
        "# Not copied into this bundle\n\n" + json.dumps(missing, indent=2) + "\n"
    )

    index_lines = [
        "# forecast_baseline_v1 review bundle",
        "",
        "Fresh-run commands:",
        "",
        "```bash",
        "python experiments/forecast_baseline/collect_data.py --run-id forecast-20261012",
        "python experiments/forecast_baseline/train_models.py --run-id forecast-20261012",
        "python experiments/forecast_baseline/model_eval.py --run-id forecast-20261012",
        "python experiments/forecast_baseline/run_diagnostic.py --run-id forecast-20261012",
        "python experiments/forecast_baseline/run_closed_loop.py --run-id forecast-20261012",
        "python experiments/deterministic_grid/run_audit_v2.py --run-id grid-audit-v2",
        "pytest tests/test_forecast_baseline.py tests/test_grid_audit_v2.py -q",
        "```",
        "",
        "## Files",
        "",
    ]
    hashes = {}
    for relative in sorted(copied):
        hashes[relative] = _sha256(STAGE / relative)
        index_lines.append(f"- `{relative}`")
    (STAGE / "REVIEW_INDEX.md").write_text("\n".join(index_lines) + "\n")
    (STAGE / "SHA256SUMS.json").write_text(json.dumps(hashes, indent=2, sort_keys=True))

    bundle = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE))
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
    print(json.dumps({"bundle": str(bundle), "files": len(names), "sha256": _sha256(bundle)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
