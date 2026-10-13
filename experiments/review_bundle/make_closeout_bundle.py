"""Build ``continuous_closeout_review.zip`` (read-only; no experiment is re-run).

    python experiments/review_bundle/make_closeout_bundle.py --run-id <id>
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
BUNDLE_NAME = "continuous_closeout_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_closeout_bundle"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

SOURCES = (
    "src/coimagine/continuous2d",
    "src/coimagine/diagnostics/grid_selection_audit.py",
    "experiments/continuous2d",
    "experiments/grid_selection_audit/write_errata.py",
    "tests/test_protocol_fix.py",
    "tests/test_continuous2d.py",
    "pyproject.toml",
    "README.md",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(relative: str, stage: Path) -> str:
    source = ROOT / relative
    target = stage / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return relative


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--protocol-run", default="20261012-160943")
    args = parser.parse_args()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    run = ROOT / "outputs/closeout_v1" / args.run_id
    protocol = ROOT / "outputs/protocol_fix_v1" / args.protocol_run

    copied: List[str] = []
    for entry in SOURCES:
        path = ROOT / entry
        if path.is_file():
            copied.append(_copy(entry, STAGE))
            continue
        for child in sorted(path.rglob("*")):
            if child.is_file() and not any(part in SKIP_DIRS for part in child.parts):
                copied.append(_copy(str(child.relative_to(ROOT)), STAGE))
    for docs in ("docs/closeout_v1", "docs/protocol_fix_v1"):
        for child in sorted((ROOT / docs).glob("*")):
            if child.is_file():
                copied.append(_copy(str(child.relative_to(ROOT)), STAGE))

    run_files = [
        "scene_legality.json",
        "scene_legality.md",
        "targeted/continuous2d/results.csv",
        "targeted/continuous2d/results.json",
        "targeted/continuous2d/summary.json",
        "targeted/continuous2d/results_table.md",
        "targeted/continuous2d/episodes.jsonl",
        "targeted/continuous2d/windows.jsonl",
        "combined/results.csv",
        "combined/results_table.md",
        "combined/summary.json",
    ]
    for name in run_files:
        if (run / name).exists():
            copied.append(_copy(str((run / name).relative_to(ROOT)), STAGE))
    protocol_files = [
        "physics_check/physics_acceptance.json",
        "targeted_tests/summary.json",
        "targeted_tests/junit.xml",
        "comparison/before_after.csv",
        "comparison/first_divergence.jsonl",
    ]
    for name in protocol_files:
        if (protocol / name).exists():
            copied.append(_copy(str((protocol / name).relative_to(ROOT)), STAGE))

    # manifest for the frozen closeout version
    from coimagine.continuous2d.legality import scene_content_hash
    from coimagine.continuous2d.scenes import default_scenes

    def git(*argv: str) -> str:
        return subprocess.run(["git", *argv], cwd=ROOT, capture_output=True, text=True).stdout.strip()

    manifest = {
        "head": git("rev-parse", "HEAD"),
        "dirty_files": [line for line in git("status", "--porcelain").splitlines() if line],
        "scene_version_library": "closeout_v2",
        "scenes": {
            scene.name: {"scene_version": scene.scene_version, "content_hash": scene_content_hash(scene)}
            for scene in default_scenes()
        },
        "modified_scenes": ["S0_dev", "S0_v2", "S0_v4"],
        "source_sha256": {name: _sha256(STAGE / name) for name in copied if name.endswith(".py")},
    }
    manifest_path = STAGE / "docs/closeout_v1/manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    copied.append("docs/closeout_v1/manifest.json")

    patch = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    (STAGE / "working_tree.patch").write_text(patch)

    (STAGE / "MISSING.md").write_text(
        "# 缺失项\n\n"
        "- 外部独立代码审查结论：尚未提供，本轮只交付材料与自测证据。\n"
        "- S1_v1 depth3 失败与 depth1 成功的现象未解决（非本轮目标）。\n"
        "- 三窗口候选计划与 beam 中间预测状态仍未逐步落盘；逐步预测序列见 schema v2 记录。\n"
        "- 旧 Grid 原始数据、旧检查点不在包内（未改动）。\n",
        encoding="utf-8",
    )

    hash_file = STAGE / "SHA256SUMS.txt"
    rows = [
        f"{_sha256(path)}  {path.relative_to(STAGE).as_posix()}"
        for path in sorted(STAGE.rglob("*"))
        if path.is_file() and path != hash_file
    ]
    hash_file.write_text("\n".join(rows) + "\n")

    target = ROOT / BUNDLE_NAME
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE).as_posix())
    readable = zipfile.ZipFile(target).testzip() is None
    print(json.dumps({
        "bundle": str(target),
        "size_mb": round(target.stat().st_size / 1e6, 2),
        "files": len(rows),
        "head": manifest["head"],
        "readable": readable,
    }, indent=2))
    return 0 if readable else 1


if __name__ == "__main__":
    raise SystemExit(main())
