"""Build ``CoImagine_review_bundle_v1.zip`` for the external code review.

Read-only with respect to production code and evidence: it only copies existing files,
hashes them and writes the three hand-off documents plus a manifest. It never runs an
experiment, never trains, never edits a source file, and never creates a git commit.

    python experiments/review_bundle/make_bundle.py
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[2]
BUNDLE_NAME = "CoImagine_review_bundle_v1.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_review_bundle_v1"

#: Explicit include list. Globs are relative to the repository root. Anything matching a
#: SKIP_PATTERN is dropped even if a glob matched it.
INCLUDE = (
    "src/coimagine/**/*",
    "experiments/grid_selection_audit/*",
    "experiments/continuous2d/*",
    "tests/test_grid_selection_audit.py",
    "tests/test_continuous2d.py",
    "README.md",
    "AGENTS.md",
    "pyproject.toml",
    "STATUS.md",
    "docs/migration_v1/*",
    "docs/CoImagine_Agent_迁移任务书_v1.md",
    "docs/CoImagine_代码审查材料交接要求_v1.md",
    "outputs/grid/world_model/belief.npz",
    "outputs/migration_v1/phaseB-full/grid_selection_audit/summary.csv",
    "outputs/migration_v1/phaseB-full/grid_selection_audit/checks.json",
    "outputs/migration_v1/phaseB-full/grid_selection_audit/conclusion.md",
    "outputs/migration_v1/phaseB-full/grid_selection_audit/candidates.jsonl",
    "outputs/migration_v1/phaseB-full/grid_selection_audit/figures/*.png",
    "outputs/migration_v1/phaseD-final/continuous2d/results.csv",
    "outputs/migration_v1/phaseD-final/continuous2d/results.json",
    "outputs/migration_v1/phaseD-final/continuous2d/summary.json",
    "outputs/migration_v1/phaseD-final/continuous2d/results_table.md",
    "outputs/migration_v1/phaseD-final/continuous2d/episodes.jsonl",
    "outputs/migration_v1/phaseD-final/continuous2d/windows.jsonl",
    "outputs/migration_v1/phaseD-final/continuous2d/media/S1_dev_mpc_success.gif",
    "outputs/migration_v1/physics_check/physics_acceptance.json",
    "outputs/migration_v1/beam_diagnostic.log",
    "outputs/migration_v1/phaseB_regen.log",
    "outputs/migration_v1/phaseD_final.log",
    "outputs/migration_v1/phaseD_full.log",
)

SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache", "node_modules"}
SKIP_SUFFIXES = {".pyc", ".pyo"}

REVIEW_TEMPLATES = {
    "REVIEW_INDEX.md": "experiments/review_bundle/REVIEW_INDEX.md",
    "RUN_REVIEW.md": "experiments/review_bundle/RUN_REVIEW.md",
}

#: Files the task asks for that are known not to exist; recorded, not fabricated.
EXPECTED_ABSENT = {
    "requirements.txt / requirements*.txt / lock file": "仓库不存在这些文件；pyproject.toml 是唯一依赖声明",
    "tests/conftest.py": "仓库不存在该文件；pytest 通过 pyproject.toml 的 pythonpath 配置",
    "outputs/migration_v1/phaseD-final/continuous2d/media/S1_v1_failure.gif":
        "从未生成；本轮不重新渲染",
    "完整三窗口候选计划与 beam 中间预测状态":
        "当前实现只记录执行结果，未记录每节点的候选计划/中间预测；未记录，未补日志重跑",
    "单机器人贡献对照数据": "实验未做（仅双机器人联合技能）",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _skip(path: Path) -> bool:
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    return path.suffix in SKIP_SUFFIXES


def collect() -> List[Path]:
    found: List[Path] = []
    for pattern in INCLUDE:
        for path in sorted(ROOT.glob(pattern)):
            if not path.is_file() or _skip(path):
                continue
            found.append(path)
    # de-duplicate while preserving order
    seen = set()
    unique: List[Path] = []
    for path in found:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def copy_files(files: List[Path]) -> List[str]:
    copied: List[str] = []
    for source in files:
        relative = source.relative_to(ROOT)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(str(relative))
    return copied


def current_versions() -> Dict[str, str]:
    import matplotlib
    import numpy
    import pymunk
    import yaml

    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "PyYAML": yaml.__version__,
        "pymunk": pymunk.version,
        "pytest": _pytest_version(),
    }


def _pytest_version() -> str:
    import pytest

    return pytest.__version__


def production_changes_after_reports(copied: List[str]) -> Dict[str, object]:
    """Which source files changed after the report artifacts were written.

    A file whose mtime is newer than both the phase-D results and the migration report was
    changed after the evidence was produced; otherwise it is reported as consistent. This
    is a timestamp comparison, not a content trace, and is labelled as such.
    """

    markers = [
        ROOT / "outputs/migration_v1/phaseD-final/continuous2d/results.csv",
        ROOT / "docs/migration_v1/report.md",
    ]
    existing = [path for path in markers if path.exists()]
    if not existing:
        return {"status": "unknown", "reason": "no report marker found"}
    marker_time = min(path.stat().st_mtime for path in existing)
    changed = [
        name for name in copied
        if name.endswith(".py") and (ROOT / name).stat().st_mtime > marker_time
    ]
    return {
        "status": "timestamp-comparison",
        "marker_time": datetime.fromtimestamp(marker_time, timezone.utc).isoformat(),
        "source_files_changed_after_marker": changed,
        "note": "pyproject.toml/old files are tracked by git; new files are copied whole",
    }


def write_snapshot(copied: List[str], stage: Path) -> Dict[str, object]:
    manifest_path = ROOT / "docs/migration_v1/manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    checkpoint = ROOT / "outputs/grid/world_model/belief.npz"
    snapshot = {
        "packaged_at": datetime.now(timezone.utc).astimezone().isoformat(),
        "timezone": time.tzname[0],
        "head": _git("rev-parse", "HEAD"),
        "git_status_short": _git("status", "--short").splitlines(),
        "versions_actual_environment": current_versions(),
        "versions_in_manifest": manifest.get("versions", {}),
        "production_code_after_reports": production_changes_after_reports(copied),
        "checkpoint": {
            "path": "outputs/grid/world_model/belief.npz",
            "sha256": _sha256(checkpoint),
            "manifest_sha256": manifest.get("checkpoints", {})
            .get("outputs/grid/world_model/belief.npz", {})
            .get("sha256"),
        },
        "included_result_dirs": sorted(
            {
                str(Path(name).parent)
                for name in copied
                if name.startswith("outputs/migration_v1/")
            }
        ),
        "copied_repository_files": len(copied),
        "missing_files": EXPECTED_ABSENT,
        "report_commit_represents_run_code": (
            "HEAD 627d5b8 只能代表已提交的旧代码；本轮新增源码、配置、脚本与测试均为未跟踪"
            "文件，工作区同时有 README.md 的未提交修改（见 working_tree.patch）。因此不能把 "
            "HEAD 当成完整实验版本；完整版本 = HEAD + 工作区（本包逐文件复制）。"
        ),
        "media_policy": (
            "只附 1 段现成 GIF（S1 成功）。S1_v1 失败 GIF 不存在，未重新渲染；"
            "S1_v1 失败的逐控制步轨迹仍在 phaseD-final/continuous2d/windows.jsonl。"
        ),
    }
    review = stage / "review"
    review.mkdir(parents=True, exist_ok=True)
    (review / "SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    return snapshot


def write_patch(stage: Path) -> str:
    patch = subprocess.run(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, text=True
    ).stdout
    (stage / "review" / "working_tree.patch").write_text(patch)
    return patch


def write_hashes(stage: Path) -> int:
    hash_file = stage / "review" / "SHA256SUMS.txt"
    rows = []
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path == hash_file:
            continue
        rows.append(f"{_sha256(path)}  {path.relative_to(stage).as_posix()}")
    hash_file.write_text("\n".join(rows) + "\n")
    return len(rows)


def verify(stage: Path) -> Dict[str, object]:
    hash_file = stage / "review" / "SHA256SUMS.txt"
    mismatches = []
    checked = 0
    for line in hash_file.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split("  ", 1)
        target = stage / name
        if not target.exists() or _sha256(target) != digest:
            mismatches.append(name)
        checked += 1
    patch = stage / "review" / "working_tree.patch"
    return {"checked": checked, "mismatches": mismatches, "patch_nonempty": patch.stat().st_size > 0}


def make_zip(stage: Path, target: Path) -> Tuple[int, int]:
    count = 0
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(stage.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(stage).as_posix())
                count += 1
    return count, target.stat().st_size


def main() -> int:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    files = collect()
    copied = copy_files(files)

    # hand-off documents
    review = STAGE / "review"
    review.mkdir(parents=True, exist_ok=True)
    for target_name, template in REVIEW_TEMPLATES.items():
        shutil.copy2(ROOT / template, review / target_name)
    shutil.copy2(Path(__file__), review / "make_bundle.py")

    snapshot = write_snapshot(copied, STAGE)
    write_patch(STAGE)
    hashed = write_hashes(STAGE)
    verification = verify(STAGE)

    target = ROOT / BUNDLE_NAME
    count, size = make_zip(STAGE, target)
    readable = zipfile.ZipFile(target).testzip() is None

    patch_nonempty = verification.pop("patch_nonempty", False)
    print(json.dumps({
        "bundle": str(target),
        "size_mb": round(size / 1e6, 2),
        "files_in_zip": count,
        "hashed_files": hashed,
        "verification": verification,
        "patch_nonempty": patch_nonempty,
        "zip_readable": readable,
        "head": snapshot["head"],
        "missing": list(snapshot["missing_files"]),
    }, indent=2, ensure_ascii=False))
    return 0 if readable and not verification["mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
