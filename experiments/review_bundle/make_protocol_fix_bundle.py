"""Build the small protocol-fix review ZIP (read-only; no experiment is re-run).

    python experiments/review_bundle/make_protocol_fix_bundle.py --run-id <id>

Includes the changed source, config and tests in full (untracked new files included
whole), the round's documents, the current manifest, the result tables, the success-hold
restoration regression result, key trajectories and the first-divergence records, plus
``working_tree.patch`` for tracked modifications, a SHA256 list and a missing list.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "CoImagine_protocol_fix_v1_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_protocol_fix_bundle"

SOURCE = (
    "src/coimagine/continuous2d",
    "src/coimagine/diagnostics/grid_selection_audit.py",
    "experiments/continuous2d",
    "experiments/grid_selection_audit/write_errata.py",
    "tests/test_protocol_fix.py",
    "tests/test_continuous2d.py",
    "pyproject.toml",
    "README.md",
)
DOCS = "docs/protocol_fix_v1"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _gather_sources(stage: Path) -> List[str]:
    copied: List[str] = []
    for entry in SOURCE:
        path = ROOT / entry
        if path.is_file():
            _copy(path, stage / entry)
            copied.append(entry)
            continue
        for child in sorted(path.rglob("*")):
            if not child.is_file() or any(part in SKIP_DIRS for part in child.parts):
                continue
            relative = child.relative_to(ROOT)
            _copy(child, stage / relative)
            copied.append(str(relative))
    return copied


def _gather_run(stage: Path, run_id: str) -> Dict[str, object]:
    run = ROOT / "outputs/protocol_fix_v1" / run_id
    copied: List[str] = []
    for name in (
        "continuous2d/results.csv",
        "continuous2d/results.json",
        "continuous2d/summary.json",
        "continuous2d/results_table.md",
        "continuous2d/episodes.jsonl",
        "comparison/before_after.csv",
        "comparison/before_after.md",
        "comparison/first_divergence.jsonl",
        "physics_check/physics_acceptance.json",
        "run_info.md",
        "targeted_tests/summary.json",
        "targeted_tests/junit.xml",
    ):
        source = run / name
        if source.exists():
            _copy(source, stage / "outputs/protocol_fix_v1" / run_id / name)
            copied.append(name)

    # window-level summary for every episode, and full per-step records for the key ones
    windows_path = run / "continuous2d" / "windows.jsonl"
    summary_lines: List[str] = []
    full_lines: List[str] = []
    key_prefixes = (
        "mpc_depth1-S0_v2", "mpc_depth3-S0_v2",
        "mpc_depth3-S1_v2", "mpc_depth1-S1_v3", "mpc_depth3-S1_v4",
        "mpc_depth1-S1_dev", "mpc_depth3-S1_dev",
        "mpc_depth1-S1_v1", "mpc_depth3-S1_v1", "mpc_depth1-S0_dev",
    )
    if windows_path.exists():
        with windows_path.open() as handle:
            for line in handle:
                record = json.loads(line)
                summary = {key: value for key, value in record.items() if key != "steps"}
                summary["step_count"] = len(record.get("steps", []))
                summary_lines.append(json.dumps(summary))
                if str(record["episode_id"]).startswith(key_prefixes):
                    full_lines.append(line.rstrip("\n"))
    target_dir = stage / "outputs/protocol_fix_v1" / run_id / "continuous2d"
    target_dir.mkdir(parents=True, exist_ok=True)
    (target_dir / "windows_summary.jsonl").write_text("\n".join(summary_lines) + "\n")
    (target_dir / "windows_key_full.jsonl").write_text("\n".join(full_lines) + "\n")
    copied += ["continuous2d/windows_summary.jsonl", "continuous2d/windows_key_full.jsonl"]

    # the success-hold restoration regression result, extracted for quick review
    acceptance = run / "physics_check" / "physics_acceptance.json"
    if acceptance.exists():
        report = json.loads(acceptance.read_text())
        extract = {
            "success_hold_restore": report.get("s1_hold_restore"),
            "snapshot_equivalence": report.get("snapshot_equivalence"),
            "wall_penetration_coverage": report.get("wall_penetration_coverage"),
            "s1_witness": {
                key: report["s1_witness"][key]
                for key in ("planner_success", "replay_success", "replay_windows")
                if key in report.get("s1_witness", {})
            },
            "checks": report.get("checks"),
        }
        target = stage / "outputs/protocol_fix_v1" / run_id / "SUCCESS_HOLD_RESTORE.json"
        target.write_text(json.dumps(extract, indent=2))
        copied.append("SUCCESS_HOLD_RESTORE.json")
    return {"run_id": run_id, "copied": copied}


def _write_patch(stage: Path) -> None:
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    (stage / "working_tree.patch").write_text(patch)


def _write_hashes(stage: Path) -> int:
    hash_file = stage / "SHA256SUMS.txt"
    rows = []
    for path in sorted(stage.rglob("*")):
        if path.is_file() and path != hash_file:
            rows.append(f"{_sha256(path)}  {path.relative_to(stage).as_posix()}")
    hash_file.write_text("\n".join(rows) + "\n")
    return len(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)

    copied = _gather_sources(STAGE)
    for item in sorted((ROOT / DOCS).glob("*")):
        if item.is_file():
            _copy(item, STAGE / DOCS / item.name)
            copied.append(str(item.relative_to(ROOT)))
    run_info = _gather_run(STAGE, args.run_id)
    _write_patch(STAGE)

    (STAGE / "MISSING.md").write_text(
        "# 缺失项\n\n"
        "- `CoImagine_快速代码审查_v1.md`：任务书引用但仓库不存在。\n"
        "- S1_v1 失败的 GIF：从未生成，且本轮不重新渲染。\n"
        "- 三窗口候选计划与 beam 中间预测状态：仍未逐步落盘（只保存每次决策选中的完整预测序列）。\n"
        "- 单机器人贡献对照：实验未做。\n"
        "- 完整 `windows.jsonl`（26 MB）：本包只放 `windows_summary.jsonl`（全部窗口摘要）与"
        "`windows_key_full.jsonl`（关键场景逐控制步）；完整文件在仓库 "
        "`outputs/protocol_fix_v1/<run_id>/continuous2d/windows.jsonl`。\n"
        "- 旧 Grid 原始数据与旧检查点不在包内（未改动，保留在仓库）。\n",
        encoding="utf-8",
    )
    hashed = _write_hashes(STAGE)

    target = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE).as_posix())
    readable = zipfile.ZipFile(target).testzip() is None
    size = target.stat().st_size
    print(json.dumps({
        "bundle": str(target),
        "size_mb": round(size / 1e6, 2),
        "hashed_files": hashed,
        "sources": len(copied),
        "run": run_info["run_id"],
        "readable": readable,
    }, indent=2))
    return 0 if readable else 1


if __name__ == "__main__":
    raise SystemExit(main())
