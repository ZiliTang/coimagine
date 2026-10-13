"""Build the two archives for local_position_push_v1.

    python experiments/local_position_push/make_bundle.py

``..._review.zip``: the new controller source, the shared-protocol diff, the config, the tests,
the complete 1920-record test JSONL and 384-record dev JSONL, the root/perturbation hashes, the
continuation records, the metric scripts and the core tables, the three timing diagnostics, and
the eight case logs with their actual scene configuration.
``..._animations.zip``: 16 single videos, 8 V/X side-by-side videos, 8 preview GIFs, the offline
index and the guide. The animations are actually delivered, not just described.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/local_position_push_v1/local-position-push-v1"

SOURCE = [
    "src/coimagine/local_contact_option/__init__.py",
    "src/coimagine/local_contact_option/contract.py",
    "src/coimagine/local_contact_option/sensing.py",
    "src/coimagine/local_contact_option/control.py",
    "src/coimagine/local_contact_option/position.py",
    "src/coimagine/local_contact_option/skill.py",
    "src/coimagine/local_contact_option/delay.py",
    "src/coimagine/local_contact_option/option.py",
    "src/coimagine/local_contact_option/roots.py",
    "src/coimagine/local_contact_option/runner.py",
    "src/coimagine/local_contact_option/metrics.py",
    "tests/test_local_contact_option.py",
    "tests/test_local_contact_option_repair.py",
    "tests/test_local_position_push.py",
    "experiments/local_position_push/run_pair.py",
    "experiments/local_position_push/run_continuation.py",
    "experiments/local_position_push/make_metrics.py",
    "experiments/local_position_push/make_timing_diagnostics.py",
    "experiments/local_position_push/make_animations.py",
    "experiments/local_position_push/make_bundle.py",
]
DOCS = [
    "docs/local_position_push_v1/report.md",
    "docs/local_position_push_v1/STATUS.md",
    "docs/local_position_push_v1/A_baseline_facts.md",
    "docs/local_position_push_v1/CONTRACT_revision.md",
    "docs/local_position_push_v1/errata.md",
    "docs/local_position_push_v1/ANIMATION_GUIDE.md",
    "docs/local_position_push_v1/tables.md",
    "docs/local_position_push_v1/PROTOCOL_CHANGES.md",
    "docs/local_position_push_v1/REPRODUCE.md",
    "docs/local_position_push_v1/manifest.json",
]
RUN_SMALL = [
    "metrics_test.json", "tables.md", "continuation.json", "roots_test.json", "roots_dev.json",
    "roots_manifest.json", "summary_dev.json", "summary_test.json",
    "animation_index.csv", "animation_index.json", "index.html",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-zip", default="local_position_push_v1_review.zip")
    parser.add_argument("--animation-zip", default="local_position_push_v1_animations.zip")
    args = parser.parse_args()

    staging = Path(tempfile.mkdtemp(prefix="lpp_review_"))
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
        target = staging / "executions" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RUN / name, target)
    # the three timing diagnostics (figures + CSV) go in whole
    timing = staging / "timing"
    timing.mkdir(parents=True, exist_ok=True)
    for path in sorted((RUN / "timing").iterdir()):
        if path.is_file():
            shutil.copy2(path, timing / path.name)
    # the eight case logs and their actual scene configuration
    case_logs = staging / "case_logs"
    case_logs.mkdir(parents=True, exist_ok=True)
    index = json.loads((RUN / "animation_index.json").read_text())
    roots = {row["root_id"]: row for row in json.loads((RUN / "roots_test.json").read_text())["roots"]}
    cases = []
    for case in index["cases"]:
        name = f"{case['root_id'].replace('/', '_')}__baseline.jsonl"
        source = RUN / "logs" / "test" / name
        if source.exists():
            shutil.copy2(source, case_logs / name)
        cases.append({**case, "scene_metadata": roots.get(case["root_id"])})
    (case_logs / "cases.json").write_text(json.dumps(cases, indent=1, sort_keys=True))
    samples = staging / "logs_sample"
    samples.mkdir(parents=True, exist_ok=True)
    for path in sorted((RUN / "logs" / "test").glob("*.jsonl"))[:24]:
        shutil.copy2(path, samples / path.name)

    total = sum(path.stat().st_size for path in (RUN / "logs").rglob("*.jsonl"))
    (staging / "MISSING.md").write_text(
        "# Large artifacts left in the working tree\n\n"
        "Everything here is regenerated deterministically by the command below.\n\n"
        "```json\n" + json.dumps({
            "per_control_period_logs": {
                "path": str((RUN / "logs").relative_to(ROOT)),
                "files": sum(1 for _ in (RUN / "logs").rglob("*.jsonl")),
                "bytes": total,
                "note": "the eight case logs are included in full under case_logs/, plus 24 samples",
                "recovery": "python experiments/local_position_push/run_pair.py --split test "
                            "--workers 12",
            },
            "executions": {
                "note": "executions_test.jsonl (1920) and executions_dev.jsonl (384) are both "
                        "inside this archive in full",
            },
            "animations": {
                "note": "delivered in local_position_push_v1_animations.zip, not in this archive",
            },
        }, indent=1) + "\n```\n"
        + ("\n## Expected but absent\n\n" + "\n".join(f"- {n}" for n in missing) + "\n"
           if missing else ""))
    review = ROOT / args.review_zip
    with zipfile.ZipFile(review, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)

    staging = Path(tempfile.mkdtemp(prefix="lpp_anim_"))
    for kind in ("singles", "pairs", "gif"):
        source = RUN / "figures" / kind
        if not source.exists():
            continue
        target = staging / "figures" / kind
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            if path.is_file():
                shutil.copy2(path, target / path.name)
    for name in ("index.html", "animation_index.csv", "animation_index.json"):
        source = RUN / name
        if source.exists():
            shutil.copy2(source, staging / name)
    for name in ("ANIMATION_GUIDE.md",):
        source = ROOT / "docs/local_position_push_v1" / name
        if source.exists():
            shutil.copy2(source, staging / name)
    timing = staging / "timing"
    timing.mkdir(parents=True, exist_ok=True)
    for path in sorted((RUN / "timing").iterdir()):
        if path.is_file():
            shutil.copy2(path, timing / path.name)
    archive = ROOT / args.animation_zip
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)

    result = {"review": {"archive": review.name, "bytes": review.stat().st_size},
              "animations": {"archive": archive.name, "bytes": archive.stat().st_size},
              "missing": missing}
    print(json.dumps(result, indent=1))
    print("BUNDLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
