"""Build the two archives for local_contact_option_repair_v2.

    python experiments/local_contact_option/make_bundle_repair.py

``local_contact_option_repair_v2_review.zip`` carries the code, tests, docs, the **complete**
3840-record test JSONL, the continuation records, the metric implementations and the
comparison tables. ``local_contact_option_repair_v2_animations.zip`` carries the 32 single
videos, the 8 quads, the 8 preview GIFs, the offline index and the animation logs, so the
animations never depend on re-running anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/local_contact_option_repair_v2/local-contact-option-repair-v2"

SOURCE = [
    "src/coimagine/local_contact_option/__init__.py",
    "src/coimagine/local_contact_option/contract.py",
    "src/coimagine/local_contact_option/sensing.py",
    "src/coimagine/local_contact_option/control.py",
    "src/coimagine/local_contact_option/skill.py",
    "src/coimagine/local_contact_option/delay.py",
    "src/coimagine/local_contact_option/option.py",
    "src/coimagine/local_contact_option/roots.py",
    "src/coimagine/local_contact_option/runner.py",
    "src/coimagine/local_contact_option/metrics.py",
    "experiments/local_contact_option/audit_old.py",
    "experiments/local_contact_option/run_test.py",
    "experiments/local_contact_option/run_continuation.py",
    "experiments/local_contact_option/make_metrics.py",
    "experiments/local_contact_option/make_comparison.py",
    "experiments/local_contact_option/make_animations.py",
    "experiments/local_contact_option/make_bundle_repair.py",
    "tests/test_local_contact_option.py",
    "tests/test_local_contact_option_repair.py",
]
DOCS = [
    "docs/local_contact_option_repair_v2/report.md",
    "docs/local_contact_option_repair_v2/STATUS.md",
    "docs/local_contact_option_repair_v2/CONTRACT_revision.md",
    "docs/local_contact_option_repair_v2/errata.md",
    "docs/local_contact_option_repair_v2/changed_files.md",
    "docs/local_contact_option_repair_v2/ANIMATION_GUIDE.md",
    "docs/local_contact_option_repair_v2/manifest.json",
]
RUN_SMALL = [
    "metrics.json", "tables.md", "comparison.json", "comparison.md", "continuation.json",
    "roots_dev.json", "roots_test.json", "dev_kv_selection.json",
    "summary_dev.json", "summary_test.json", "animation_index.csv", "animation_index.json",
    "index.html",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_review(archive: Path) -> dict:
    staging = Path(tempfile.mkdtemp(prefix="lco_repair_review_"))
    missing = []
    for name in SOURCE + DOCS:
        source = ROOT / name
        if source.exists():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            missing.append(name)
    for name in RUN_SMALL:
        source = RUN / name
        if source.exists():
            target = staging / "outputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for name in ("executions_test.jsonl", "executions_dev.jsonl"):
        source = RUN / name
        if source.exists():
            target = staging / "executions" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    logs = staging / "logs_sample"
    logs.mkdir(parents=True, exist_ok=True)
    for path in sorted((RUN / "logs" / "test").glob("*.jsonl"))[:32]:
        shutil.copy2(path, logs / path.name)

    total_logs = sum(path.stat().st_size for path in (RUN / "logs").rglob("*.jsonl"))
    missing_doc = {
        "note": "the archive carries the complete 3840-record test JSONL and all dev records; "
                "the per-control-period logs are sampled (32 of them) because they are ~380 MB",
        "logs": {
            "path": str((RUN / "logs").relative_to(ROOT)),
            "files": sum(1 for _ in (RUN / "logs").rglob("*.jsonl")),
            "bytes": total_logs,
            "sha256_of_first": sha256(sorted((RUN / "logs" / "test").glob("*.jsonl"))[0]),
            "recovery": "python experiments/local_contact_option/run_test.py "
                        "--run-id local-contact-option-repair-v2 "
                        "--out-root outputs/local_contact_option_repair_v2 --split test --workers 12",
        },
        "executions": {
            "test": {"bytes": (RUN / "executions_test.jsonl").stat().st_size,
                     "sha256": sha256(RUN / "executions_test.jsonl")},
            "dev": {"bytes": (RUN / "executions_dev.jsonl").stat().st_size,
                    "sha256": sha256(RUN / "executions_dev.jsonl")},
            "note": "both are inside this archive as well",
        },
    }
    (staging / "MISSING.md").write_text(
        "# Large artifacts left in the working tree\n\n"
        "Everything listed here is regenerated deterministically by the recovery command.\n\n"
        "```json\n" + json.dumps(missing_doc, indent=1) + "\n```\n"
        + ("\n## Expected but absent\n\n" + "\n".join(f"- {name}" for name in missing) + "\n"
           if missing else "")
    )
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as handle:
        count = len(handle.namelist())
    return {"archive": archive.name, "files": count, "bytes": archive.stat().st_size,
            "missing": missing}


def build_animations(archive: Path) -> dict:
    staging = Path(tempfile.mkdtemp(prefix="lco_repair_anim_"))
    figures = RUN / "figures"
    for kind, folder in (("singles", "singles"), ("quads", "quads"), ("gif", "gif")):
        source = figures / folder
        if not source.exists():
            continue
        target = staging / "figures" / folder
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if path.is_file():
                shutil.copy2(path, target / path.name)
    for name in ("index.html", "animation_index.csv", "animation_index.json", "roots_test.json"):
        source = RUN / name
        if source.exists():
            shutil.copy2(source, staging / name)
    guide = ROOT / "docs/local_contact_option_repair_v2/ANIMATION_GUIDE.md"
    if guide.exists():
        shutil.copy2(guide, staging / "ANIMATION_GUIDE.md")
    # the eight case logs, so the videos can be traced back to the executions
    sample = staging / "case_logs"
    sample.mkdir(exist_ok=True)
    index = json.loads((RUN / "animation_index.json").read_text()) if (RUN / "animation_index.json").exists() else {"cases": []}
    for case in index.get("cases", []):
        name = f"{case['root_id'].replace('/', '_')}__{case['variant']}.jsonl"
        source = RUN / "logs" / "test" / name
        if source.exists():
            shutil.copy2(source, sample / name)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as handle:
        count = len(handle.namelist())
    return {"archive": archive.name, "files": count, "bytes": archive.stat().st_size}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-zip", default="local_contact_option_repair_v2_review.zip")
    parser.add_argument("--animation-zip", default="local_contact_option_repair_v2_animations.zip")
    args = parser.parse_args()
    review = build_review(ROOT / args.review_zip)
    animations = build_animations(ROOT / args.animation_zip)
    print(json.dumps({"review": review, "animations": animations}, indent=1))
    print("BUNDLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
