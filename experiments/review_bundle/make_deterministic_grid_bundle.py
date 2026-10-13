"""Build ``deterministic_grid_review.zip`` (read-only; no experiment is re-run).

    python experiments/review_bundle/make_deterministic_grid_bundle.py --run-id <id>
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
BUNDLE_NAME = "deterministic_grid_review.zip"
STAGE = Path(tempfile.gettempdir()) / "coimagine_deterministic_grid_bundle"
SKIP_DIRS = {"__pycache__", ".git", ".venv", ".pytest_cache"}

SOURCES = (
    "src/coimagine/deterministic_grid",
    "experiments/deterministic_grid",
    "tests/test_deterministic_grid.py",
    "docs/deterministic_grid_v1",
    "pyproject.toml",
    "README.md",
)
RUN_FILES = (
    "env_checks.json",
    "dataset/config.json",
    "dataset/stats.json",
    "dataset/splits.jsonl",
    "dataset/manifest.json",
    "dataset/samples.jsonl",
    "training/summary.json",
    "training/seed0.json",
    "training/seed1.json",
    "training/seed2.json",
    "models/seed0.pt",
    "models/seed1.pt",
    "models/seed2.pt",
    "diagnostic/config.json",
    "diagnostic/metrics.json",
    "diagnostic/metrics_table.md",
    "diagnostic/per_run.jsonl",
    "diagnostic/candidate_records.jsonl",
    "diagnostic/chosen_traces.jsonl",
    "diagnostic/teacher_forced_vs_recursive.json",
    "diagnostic/eval_states.json",
    "diagnostic/summary.json",
    "figures/fig_k_selection_bias.png",
    "figures/fig_k_true_return.png",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--out-root", default="outputs/deterministic_grid_v1")
    args = parser.parse_args()
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True)
    run = ROOT / args.out_root / args.run_id

    copied: List[str] = []
    for entry in SOURCES:
        source = ROOT / entry
        if source.is_file():
            target = STAGE / entry
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            copied.append(entry)
            continue
        for child in sorted(source.rglob("*")):
            if child.is_file() and not any(part in SKIP_DIRS for part in child.parts):
                relative = child.relative_to(ROOT)
                target = STAGE / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(child, target)
                copied.append(str(relative))
    missing: List[str] = []
    for name in RUN_FILES:
        source = run / name
        if not source.exists():
            missing.append(name)
            continue
        target = STAGE / args.out_root / args.run_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(f"{args.out_root}/{args.run_id}/{name}")

    patch = subprocess.run(["git", "diff", "--binary", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout
    (STAGE / "working_tree.patch").write_text(patch)
    (STAGE / "MISSING.md").write_text(
        "# 缺失项\n\n"
        + ("\n".join(f"- `{name}`：运行目录中不存在。" for name in missing) if missing else "- 无（合同要求的运行文件全部在包内）。")
        + "\n- 代码外部独立复核结论：尚未提供。\n"
        + "- 概率分布递归、不确定性规划、多尺度方法：本轮明确未做（CONTRACT 只定义点预测基线）。\n",
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
        "missing_run_files": missing,
        "readable": readable,
    }, indent=2, ensure_ascii=False))
    return 0 if readable else 1


if __name__ == "__main__":
    raise SystemExit(main())
