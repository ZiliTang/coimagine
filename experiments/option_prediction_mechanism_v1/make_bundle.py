"""Build the compact round-3 review bundle, and verify that it really contains what it claims.

    python experiments/option_prediction_mechanism_v1/make_bundle.py

What goes in (the review found the first bundle lacking these, so they are now explicit):

* the **source changes**: every modified ``src/coimagine/local_contact_option/*.py`` file plus a
  unified patch (``git diff <round-start>..HEAD``) and its ``--stat``;
* the round's scripts **and their markdown** (field table, phase READMEs);
* all three round-3 test files;
* the **A pairing results**, the **B training reports and ledgers**, the C tables, and the frozen
  C result;
* a **small sample of raw execution records**: the eight command branches of one train root, one
  validation root and one test root, and one per-period A log;

and lists -- without copying -- the large artifacts (execution JSONL, roots/commands, checkpoints,
per-period logs) with size, sha256 and how to regenerate them.

The build ends by re-opening the archive and asserting that every path the manifest lists under
"Included" is actually present, so the manifest cannot drift from the zip again.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1"
DOCS = ROOT / "docs/option_prediction_mechanism_v1"
EXPERIMENTS = ROOT / "experiments/option_prediction_mechanism_v1"
OUT = ROOT / "option_prediction_mechanism_v1_review.zip"
SMALL_LIMIT = 2 * 1024 * 1024
#: the round-2 delivery commit; the patch below is exactly this round's diff
ROUND_START = "aedffae"

SMALL_ARTIFACTS = [
    RUN / "a3_dev.json", RUN / "a3_dev.md", RUN / "a3_test.json", RUN / "a3_test.md",
    RUN / "a3_test_compare.json", RUN / "a3_test_compare.md",
    RUN / "phase_b/ledger.json", RUN / "phase_b/verification.json", RUN / "phase_b/spot_check.json",
    RUN / "phase_b/train_report_1.md", RUN / "phase_b/train_report_1.json",
    RUN / "phase_b/README.md",
    RUN / "phase_b_tier2/ledger.json", RUN / "phase_b_tier2/verification.json",
    RUN / "phase_b_tier2/train_report_2.md", RUN / "phase_b_tier2/train_report_2.json",
    ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_table.md",
    ROOT / "outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_result.json",
    ROOT / "outputs/option_prediction_mechanism_v1/phase_c_frozen/frozen_c_recomputed.json",
    RUN / "phase_b/target_audit.json",
    RUN / "phase_b_tier2/action_sensitivity.json",
    DOCS / "CORRECTIONS.md",
]
#: source changed by this round (the A fix lives here)
SRC_FILES = [
    "src/coimagine/local_contact_option/contract.py",
    "src/coimagine/local_contact_option/control.py",
    "src/coimagine/local_contact_option/option.py",
    "src/coimagine/local_contact_option/metrics.py",
]
#: raw record samples: eight command branches of one root per split, plus one A per-period log
SAMPLE_GLOBS = [
    (RUN / "phase_b", "executions_train.jsonl", 1),
    (RUN / "phase_b", "executions_val.jsonl", 1),
    (RUN / "phase_b", "executions_test.jsonl", 1),
]
A_LOG_SAMPLES = [
    RUN / "logs/w020_test/test_solo_central_000__baseline.jsonl",
    RUN / "logs/w0_test/test_solo_central_000__baseline.jsonl",
]
LIST_ONLY = [
    RUN / "phase_b/executions_train.jsonl", RUN / "phase_b/executions_val.jsonl",
    RUN / "phase_b/executions_test.jsonl", RUN / "phase_b/records_tier1.jsonl",
    RUN / "phase_b/roots.json", RUN / "phase_b/commands.json",
    RUN / "phase_b_tier2/executions_train.jsonl", RUN / "phase_b_tier2/executions_val.jsonl",
    RUN / "phase_b_tier2/executions_test.jsonl", RUN / "phase_b_tier2/roots.json",
    RUN / "phase_b_tier2/commands.json",
    ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_records.json",
    ROOT / "outputs/option_prediction_mechanism_v1/phase_c/phase_c_belief_reference.json",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()




def collect() -> Tuple[List[Path], List[Path]]:
    copied: List[Path] = []
    # the manifest and its own check are written by this script; never bundle them as inputs
    skip = {"MANIFEST.md", "BUNDLE_CHECK.json", "round3.patch", "round3_diffstat.txt"}
    copied.extend(sorted(path for path in DOCS.glob("*") if path.name not in skip))
    copied.extend(sorted(EXPERIMENTS.rglob("*.py")))
    copied.extend(sorted(EXPERIMENTS.rglob("*.md")))
    copied.extend(ROOT / path for path in SRC_FILES)
    copied.extend(sorted((ROOT / "tests").glob("test_option_prediction_mechanism_*.py")))
    for path in SMALL_ARTIFACTS + A_LOG_SAMPLES:
        if path.exists():
            copied.append(path)

    # raw record samples, one root per split (eight branches of that root)
    sample_dir = DOCS / "samples"
    sample_dir.mkdir(exist_ok=True)
    for base, name, _count in SAMPLE_GLOBS:
        source = base / name
        if not source.exists():
            continue
        target = sample_dir / name.replace("executions_", "sample_")
        records: List[dict] = []
        root = None
        for line in source.open():
            record = json.loads(line)
            if root is None:
                root = record["root_id"]
            if record["root_id"] != root:
                break
            records.append(record)
        target.write_text("\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n")
        copied.append(target)

    # the round's source patch, so the review can diff rather than trust
    patch = DOCS / "round3.patch"
    code_paths = ["src", "experiments", "tests"]
    diff = subprocess.run(["git", "diff", f"{ROUND_START}..HEAD", "--", *code_paths], cwd=ROOT,
                          capture_output=True, text=True, check=False).stdout
    patch.write_text(diff)
    copied.append(patch)
    stat = subprocess.run(["git", "diff", "--stat", f"{ROUND_START}..HEAD", "--", *code_paths],
                          cwd=ROOT, capture_output=True, text=True, check=False).stdout
    (DOCS / "round3_diffstat.txt").write_text(stat)
    copied.append(DOCS / "round3_diffstat.txt")

    # a file can be reached both by a glob and by an explicit entry; never ship it twice
    seen: set = set()
    unique: List[Path] = []
    for path in copied:
        key = path.resolve()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)
    copied = unique
    listed: List[Path] = [path for path in copied if path.stat().st_size > SMALL_LIMIT]
    copied = [path for path in copied if path.stat().st_size <= SMALL_LIMIT]
    for path in LIST_ONLY + [RUN / "phase_b/checkpoints", RUN / "phase_b_tier2/checkpoints"]:
        if path.is_dir():
            listed.extend(sorted(item for item in path.rglob("*") if item.is_file()))
        elif path.exists():
            listed.append(path)
    return copied, listed, _directory_summaries()


def _directory_summaries() -> List[dict]:
    """Per-period log trees are huge; summarise them instead of listing every file."""

    out: List[dict] = []
    for base in (RUN / "logs",):
        if not base.exists():
            continue
        for sub in sorted(item for item in base.iterdir() if item.is_dir()):
            files = [item for item in sub.rglob("*") if item.is_file()]
            if not files:
                continue
            digest = hashlib.sha256()
            for item in sorted(files):
                digest.update(str(item.relative_to(ROOT)).encode())
                digest.update(str(item.stat().st_size).encode())
            out.append({"path": str(sub.relative_to(ROOT)), "files": len(files),
                        "bytes": sum(item.stat().st_size for item in files),
                        "sha256_of_filelist": digest.hexdigest()[:16]})
    return out


def main() -> int:
    copied, listed, directories = collect()
    lines = ["# option_prediction_mechanism_v1 — review bundle manifest", "",
             f"built {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"git commit {_git()}", f"round diff base {ROUND_START}", "",
             "## Included (present inside the zip)", "",
             "| path | bytes | sha256 |", "|---|---:|---|"]
    for path in copied:
        lines.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)[:16]} |")
    lines += ["", "## Not included (listed only: large per-period logs / checkpoints)", "",
              "| path | bytes | sha256 | how to regenerate |", "|---|---:|---|---|"]
    for path in listed:
        if not path.exists():
            continue
        lines.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)[:16]} | "
                     f"see docs/option_prediction_mechanism_v1/REPRODUCE.md |")
    lines += ["", "## Not included (per-period log trees, summarised)", "",
              "| directory | files | bytes | sha256 of file list |", "|---|---:|---:|---|"]
    for entry in directories:
        lines.append(f"| {entry['path']} | {entry['files']} | {entry['bytes']} | "
                     f"{entry['sha256_of_filelist']} |")
    lines += ["", "Everything listed above is regenerated by the commands in REPRODUCE.md.", ""]
    manifest = DOCS / "MANIFEST.md"
    manifest.write_text("\n".join(lines))

    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in copied + [manifest]:
            archive.write(path, path.relative_to(ROOT))

    # verify: every manifest "Included" row must be inside the archive
    with zipfile.ZipFile(OUT) as archive:
        names = set(archive.namelist())
    missing = [str(path.relative_to(ROOT)) for path in copied
               if str(path.relative_to(ROOT)) not in names]
    extra = sorted(names - {str(path.relative_to(ROOT)) for path in copied}
                   - {str(manifest.relative_to(ROOT))})
    report = {"bundle": str(OUT), "included_in_manifest": len(copied) + 1,
              "entries_in_zip": len(names), "bytes": OUT.stat().st_size,
              "listed_only": len([p for p in listed if p.exists()]),
              "log_directories_summarised": len(directories),
              "missing_from_zip": missing, "unexpected_in_zip": extra}
    (DOCS / "BUNDLE_CHECK.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))
    if missing:
        print("BUNDLE_MISMATCH")
        return 1
    print("BUNDLE_OK")
    return 0


def _git() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True, check=False).stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
