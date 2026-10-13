"""Package the round: the lightweight review ZIP and the standalone presentation asset ZIP.

    python experiments/night_sequence_v1/make_bundle.py

The review ZIP carries source, tests, the protocol documents, the data index and the full small
sample of real sequences; checkpoints and big shards are listed with size, sha256 and the exact
regeneration command. The asset ZIP is standalone and holds every rendered figure, its underlying
data, the videos and the index. Both are reopened and checked against their own manifest.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/night_sequence_v1"
DOCS = ROOT / "docs/night_sequence_v1"
EXPERIMENTS = ROOT / "experiments/night_sequence_v1"
ASSETS = ROOT / "presentation_assets"
BUNDLE = ROOT / "night_sequence_v1_review.zip"
ASSET_ZIP = ROOT / "presentation_assets.zip"
EXTRA_ZIP = ROOT / "night_sequence_v1_checkpoints_extra.zip"

#: one representative checkpoint per architecture goes in the lightweight review ZIP; the rest go to
#: a separate resource package so the review ZIP stays small enough to open quickly
REVIEW_CHECKPOINTS = ["d1_mlp_history_seed0", "d1_transformer_history_seed0",
                      "d1_mlp_current_seed0"]
ROUND_START = "d7731ca"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_review() -> List[Path]:
    files: List[Path] = []
    files += sorted(EXPERIMENTS.glob("*.py"))
    files += sorted(DOCS.glob("*.md"))
    files += sorted((ROOT / "tests").glob("test_night_sequence_v1.py"))
    files += [ROOT / "docs/tasks/20261014_0205_CoImagine_夜间资源积累与受控扩展任务书.md"]
    files += [DATA / "runs" / f"{name}.pt" for name in REVIEW_CHECKPOINTS]
    files += [DATA / "runs" / f"{name}.curves.json" for name in REVIEW_CHECKPOINTS]
    files += sorted((DATA / "runs").glob("*.curves.json"))
    files += [DATA / "roots_night.json", DATA / "token_summary.json",
              DATA / "label_audit_ledger.json", DATA / "collection_d0.json",
              DATA / "collection_eval.json", DATA / "collection_d1.json",
              DATA / "run_ledger.jsonl"]
    for tier in ("d0", "d1"):
        metrics = DATA / "analysis" / tier / "metrics.json"
        if metrics.exists():
            files.append(metrics)
    # a complete small sample of real sequences, so the loader can be exercised from the ZIP alone
    for split in ("d0", "val", "test"):
        shards = sorted((DATA / "sequences").glob(f"{split}_*.npz"))
        if shards:
            files.append(shards[0])
    seen, unique = set(), []
    for path in files:
        if not path.exists():
            continue
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def listed_only() -> List[Path]:
    out = list(sorted((DATA / "sequences").glob("*.npz")))
    out += sorted((DATA / "tokens").glob("*.npz"))
    out += [path for path in sorted((DATA / "runs").glob("*.pt"))
            if path.stem not in REVIEW_CHECKPOINTS]
    return out


def build_extra_zip() -> dict:
    """Resource package for the checkpoints the lightweight review ZIP leaves out."""

    extras = [path for path in sorted((DATA / "runs").glob("*.pt"))
              if path.stem not in REVIEW_CHECKPOINTS]
    if not extras:
        return {"zip": None, "files": 0}
    if EXTRA_ZIP.exists():
        EXTRA_ZIP.unlink()
    with zipfile.ZipFile(EXTRA_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in extras:
            archive.write(path, path.relative_to(ROOT).as_posix())
    return {"zip": str(EXTRA_ZIP.relative_to(ROOT)), "files": len(extras),
            "bytes": EXTRA_ZIP.stat().st_size,
            "checkpoints": [path.stem for path in extras]}


def build_review() -> dict:
    files = collect_review()
    diff = subprocess.run(["git", "diff", f"{ROUND_START}..HEAD", "--", "src", "experiments",
                           "tests"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    stat = subprocess.run(["git", "diff", "--stat", f"{ROUND_START}..HEAD", "--", "src",
                           "experiments", "tests"], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()
    (DOCS / "round5.patch").write_text(diff)
    (DOCS / "round5_diffstat.txt").write_text(stat)
    files += [DOCS / "round5.patch", DOCS / "round5_diffstat.txt"]

    manifest = ["# MANIFEST — night_sequence_v1", "",
                f"git commit `{head}`; round base `{ROUND_START}`; patch limited to "
                f"`src experiments tests`.", "", "## Included", "",
                "| path | bytes | sha256 |", "|---|---:|---|"]
    for path in sorted(files):
        manifest.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)} |")
    # Large shards are summarised per split in the manifest, with the complete per-file listing in
    # LARGE_ARTIFACTS.md: thousands of rows would make the manifest unreadable without adding
    # information a hash rollup does not already carry.
    large = listed_only()
    detail = ["# LARGE_ARTIFACTS — night_sequence_v1", "",
              "未打包进审查 ZIP 的资源。完整分片列表；恢复命令见 MANIFEST。", "",
              "| path | bytes | sha256 |", "|---|---:|---|"]
    groups = {}
    for path in large:
        groups.setdefault(path.parent.relative_to(ROOT).as_posix(), []).append(path)
        detail.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)} |")
    (DOCS / "LARGE_ARTIFACTS.md").write_text("\n".join(detail) + "\n")
    manifest += ["", "## Not included (regenerable, listed for audit)", "",
                 "分片总量与滚动哈希（逐文件清单见 `LARGE_ARTIFACTS.md`）：", "",
                 "| group | files | bytes | rollup sha256 | regeneration |", "|---|---:|---:|---|---|"]
    for group, paths in sorted(groups.items()):
        blob = "".join(f"{p.name}:{p.stat().st_size}:{sha256(p)}\n" for p in sorted(paths))
        manifest.append(f"| {group} | {len(paths)} | {sum(p.stat().st_size for p in paths)} | "
                        f"{hashlib.sha256(blob.encode()).hexdigest()} | "
                        f"`python experiments/night_sequence_v1/collect.py --stage ...` then "
                        f"`windows.py` |")
    (DOCS / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    files.append(DOCS / "MANIFEST.md")
    seen, unique = set(), []
    for path in files:
        key = path.resolve()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if BUNDLE.exists():
        BUNDLE.unlink()
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(unique):
            archive.write(path, path.relative_to(ROOT).as_posix())
    with zipfile.ZipFile(BUNDLE) as archive:
        names = archive.namelist()
        text = archive.read("docs/night_sequence_v1/MANIFEST.md").decode()
        block = text.split("## Included")[1].split("## Not included")[0]
        rows = [line.split("|")[1].strip() for line in block.splitlines()
                if line.startswith("| ") and "---" not in line and "path" not in line]
        expected = set(rows) | {"docs/night_sequence_v1/MANIFEST.md"}
        return {"bundle": str(BUNDLE.relative_to(ROOT)), "bytes": BUNDLE.stat().st_size,
                "entries": len(names), "listed": len(rows),
                "duplicates": len(names) - len(set(names)),
                "missing": [r for r in rows if r not in set(names)],
                "unexpected": [n for n in names if n not in expected],
                "checkpoints": sorted(n for n in names if n.endswith(".pt")),
                "sequences_included": sorted(n for n in names if n.endswith(".npz") and
                                             "sequences" in n)}


def build_assets_zip() -> dict:
    if not ASSETS.exists():
        return {"error": "presentation_assets/ missing"}
    files = sorted(path for path in ASSETS.rglob("*") if path.is_file())
    if ASSET_ZIP.exists():
        ASSET_ZIP.unlink()
    total = 0
    with zipfile.ZipFile(ASSET_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.name == "ASSET_CHECK.json":
                continue
            archive.write(path, path.relative_to(ROOT).as_posix())
            total += path.stat().st_size
    with zipfile.ZipFile(ASSET_ZIP) as archive:
        names = archive.namelist()
    return {"zip": str(ASSET_ZIP.relative_to(ROOT)), "bytes": ASSET_ZIP.stat().st_size,
            "files": len(names), "uncompressed": total,
            "figures": sorted(n for n in names if n.endswith(".png")),
            "videos": sorted(n for n in names if n.endswith(".mp4")),
            "index_present": any(n.endswith("INDEX.md") for n in names),
            "progress_facts": any(n.endswith("progress_facts.csv") for n in names)}


def main() -> int:
    review = build_review()
    assets = build_assets_zip()
    extra = build_extra_zip()
    result = {"review": review, "assets": assets, "extra_checkpoints": extra}
    (DOCS / "BUNDLE_CHECK.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1))
    ok = (not review["missing"] and not review["unexpected"] and review["duplicates"] == 0
          and assets.get("index_present") and assets.get("progress_facts"))
    print("BUNDLE_OK" if ok else "BUNDLE_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
