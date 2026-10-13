"""Build ``local_contact_option_v1_review.zip``.

    python experiments/local_contact_option/make_bundle.py
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
RUN = ROOT / "outputs/local_contact_option_v1/local-contact-option-v1"

SOURCE = [
    "src/coimagine/local_contact_option/__init__.py",
    "src/coimagine/local_contact_option/contract.py",
    "src/coimagine/local_contact_option/sensing.py",
    "src/coimagine/local_contact_option/control.py",
    "src/coimagine/local_contact_option/skill.py",
    "src/coimagine/local_contact_option/option.py",
    "src/coimagine/local_contact_option/roots.py",
    "src/coimagine/local_contact_option/runner.py",
    "src/coimagine/local_contact_option/metrics.py",
    "experiments/local_contact_option/audit_old.py",
    "experiments/local_contact_option/run_dev.py",
    "experiments/local_contact_option/run_test.py",
    "experiments/local_contact_option/make_metrics.py",
    "experiments/local_contact_option/run_continuation.py",
    "experiments/local_contact_option/make_gifs.py",
    "experiments/local_contact_option/make_bundle.py",
    "tests/test_local_contact_option.py",
]
DOCS = [
    "docs/local_contact_option_v1/CONTRACT.md",
    "docs/local_contact_option_v1/STATUS.md",
    "docs/local_contact_option_v1/report.md",
    "docs/local_contact_option_v1/errata.md",
    "docs/local_contact_option_v1/tables.md",
    "docs/local_contact_option_v1/manifest.json",
    "docs/local_contact_option_v1/A_implementation_audit.md",
    "docs/local_contact_option_v1/A_implementation_audit.json",
]
RUN_FILES = [
    "dev_kv_selection.json", "metrics.json", "tables.md", "continuation.json",
    "roots_manifest.json", "roots_dev.json", "roots_test.json",
    "summary_dev.json", "summary_test.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", default="local_contact_option_v1_review.zip")
    args = parser.parse_args()
    staging = Path(tempfile.mkdtemp(prefix="lco_bundle_"))
    missing = []
    for name in SOURCE + DOCS:
        source = ROOT / name
        if source.exists():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        else:
            missing.append(name)
    for name in RUN_FILES:
        source = RUN / name
        if source.exists():
            target = staging / "outputs" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    for figure in sorted((RUN / "figures").glob("*.gif")):
        target = staging / "figures" / figure.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(figure, target)

    # per-execution JSONL: the dev set in full, plus the first 800 test records (the rest is
    # restored by the runner, which is deterministic)
    executions = staging / "executions"
    executions.mkdir(parents=True, exist_ok=True)
    dev_path = RUN / "executions_dev.jsonl"
    if dev_path.exists():
        shutil.copy2(dev_path, executions / "executions_dev.jsonl")
    test_path = RUN / "executions_test.jsonl"
    if test_path.exists():
        with test_path.open() as handle, (executions / "executions_test_head800.jsonl").open("w") as out:
            for index, line in enumerate(handle):
                if index >= 800:
                    break
                out.write(line)

    # one representative per-control-period log per class, plus a declared sample rule
    logs = staging / "logs_sample"
    logs.mkdir(parents=True, exist_ok=True)
    copied = 0
    for path in sorted((RUN / "logs" / "test").glob("*__baseline.jsonl")):
        shutil.copy2(path, logs / path.name)
        copied += 1
        if copied >= 24:
            break

    large = []
    for path in sorted((RUN / "logs").rglob("*.jsonl")):
        if path.name in {entry.name for entry in logs.glob("*.jsonl")}:
            continue
    total = sum(path.stat().st_size for path in (RUN / "logs").rglob("*.jsonl"))
    large.append({
        "path": str((RUN / "logs").relative_to(ROOT)),
        "files": sum(1 for _ in (RUN / "logs").rglob("*.jsonl")),
        "bytes": total,
        "recovery": "python experiments/local_contact_option/run_test.py --split test --workers 12 "
                    "(and --split dev); the runner is deterministic and rewrites the logs",
        "note": f"24 of these files, one per (root, baseline variant), are in logs_sample/",
    })
    large.append({
        "path": str(test_path.relative_to(ROOT)) if test_path.exists() else "executions_test.jsonl",
        "bytes": test_path.stat().st_size if test_path.exists() else 0,
        "sha256": sha256(test_path) if test_path.exists() else None,
        "note": "the archive carries the first 800 of 3840 records; the full file stays in the "
                "working tree and is reproduced by the same command",
    })
    (staging / "MISSING.md").write_text(
        "# Large artifacts left in the working tree\n\n"
        "Everything here is regenerated deterministically by the listed command.\n\n"
        "```json\n" + json.dumps(large, indent=1) + "\n```\n"
        + ("\n## Expected but absent\n\n" + "\n".join(f"- {name}" for name in missing) + "\n"
           if missing else "")
    )
    archive = ROOT / args.zip
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                handle.write(path, path.relative_to(staging))
    shutil.rmtree(staging)
    with zipfile.ZipFile(archive) as handle:
        count = len(handle.namelist())
    print(json.dumps({"archive": str(archive), "files": count,
                      "bytes": archive.stat().st_size, "missing": missing}, indent=1))
    print("BUNDLE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
