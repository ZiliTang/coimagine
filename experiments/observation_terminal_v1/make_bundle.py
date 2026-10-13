"""Build the round-4 review bundle and verify that the manifest matches what is inside.

    python experiments/observation_terminal_v1/make_bundle.py

Frozen checkpoints (~0.4 MB each) and the normalisation export are packed directly, as the task
book requires. Large per-period logs are **not** packed; they are listed with size, sha256 and the
exact regeneration command instead. After writing the zip the script reopens it and checks that
every "Included" row of the generated manifest really exists, that there are no duplicate names and
that nothing unexpected slipped in.
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
DOCS = ROOT / "docs/observation_terminal_v1"
OUT = ROOT / "outputs/observation_terminal_v1"
EXPERIMENTS = ROOT / "experiments/observation_terminal_v1"
BUNDLE = ROOT / "observation_terminal_v1_review.zip"
TIER2 = ROOT / ("outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/"
                "phase_b_tier2")
ROUND_START = "c216be6"

#: packed outright
SMALL_ARTIFACTS = [
    ROOT / "docs/tasks/20261014_0040_CoImagine_可观测终态预测任务书.md",
    OUT / "label_audit_ledger.json",
    OUT / "replay_verification.json",
    OUT / "terminal_report_check.json",
    OUT / "terminal_report_check.md",
    OUT / "terminal_report_train.json",
    OUT / "terminal_report_train.md",
    OUT / "normalisation.json",
    OUT / "spot_checks.json",
    OUT / "test_root_dump.json",
    OUT / "terminal_observations_test.jsonl",
    OUT / "checkpoints/deployable_seed0.pt",
    OUT / "checkpoints/deployable_seed1.pt",
    OUT / "checkpoints/deployable_seed2.pt",
    TIER2 / "roots.json",
    TIER2 / "ledger.json",
]

#: referenced but too large to pack
LISTED_ONLY = [
    OUT / "terminal_observations_train.jsonl",
    OUT / "terminal_observations_val.jsonl",
    TIER2 / "executions_train.jsonl",
    TIER2 / "executions_val.jsonl",
    TIER2 / "executions_test.jsonl",
]

REGENERATE = {
    "terminal_observations_train.jsonl": "python experiments/observation_terminal_v1/audit_labels.py --all --workers 12",
    "terminal_observations_val.jsonl": "python experiments/observation_terminal_v1/audit_labels.py --all --workers 12",
    "executions_train.jsonl": "produced by the previous round's phase_b collect_b.py; already in the repository, do not re-run",
    "executions_val.jsonl": "produced by the previous round's phase_b collect_b.py; already in the repository, do not re-run",
    "executions_test.jsonl": "produced by the previous round's phase_b collect_b.py; already in the repository, do not re-run",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect() -> List[Path]:
    copied: List[Path] = list(SMALL_ARTIFACTS)
    copied += sorted(DOCS.glob("*.md"))
    copied += sorted(EXPERIMENTS.glob("*.py"))
    copied += sorted((ROOT / "experiments/option_prediction_mechanism_v1/phase_c_frozen").glob("*.py"))
    copied += sorted((ROOT / "tests").glob("test_observation_terminal_v1.py"))
    copied += sorted((ROOT / "tests").glob("test_option_prediction_mechanism_c.py"))
    copied += sorted((ROOT / "tests").glob("test_option_prediction_mechanism_b.py"))
    copied += sorted((ROOT / "tests").glob("test_option_prediction_mechanism_a.py"))
    copied += sorted((ROOT / "src/coimagine/local_contact_option").glob("*.py"))
    seen, unique = set(), []
    for path in copied:
        key = path.resolve()
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        unique.append(path)
    return unique


def main() -> int:
    files = collect()
    included = [path for path in files]
    diff = subprocess.run(["git", "diff", f"{ROUND_START}..HEAD", "--", "src", "experiments",
                           "tests"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    stat = subprocess.run(["git", "diff", "--stat", f"{ROUND_START}..HEAD", "--", "src",
                           "experiments", "tests"], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                          check=False).stdout.strip()
    (DOCS / "round4.patch").write_text(diff)
    (DOCS / "round4_diffstat.txt").write_text(stat)
    files += [DOCS / "round4.patch", DOCS / "round4_diffstat.txt"]

    manifest = [
        "# MANIFEST — observation_terminal_v1",
        "",
        f"git commit `{head}`; round base `{ROUND_START}`; patch limited to `src experiments tests`.",
        "",
        "## Included",
        "",
        "| path | bytes | sha256 |",
        "|---|---:|---|",
    ]
    for path in sorted(files):
        manifest.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)} |")
    manifest += ["", "## Not included (regenerable, listed for audit)", "",
                 "| path | bytes | sha256 | regeneration |", "|---|---:|---|---|"]
    for path in LISTED_ONLY:
        if path.exists():
            manifest.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)} "
                            f"| `{REGENERATE[path.name]}` |")
    (DOCS / "MANIFEST.md").write_text("\n".join(manifest) + "\n")
    files.append(DOCS / "MANIFEST.md")
    # re-runs pick up the previous MANIFEST.md through the docs glob: dedupe once more
    final, seen_final = [], set()
    for path in files:
        key = path.resolve()
        if key in seen_final:
            continue
        seen_final.add(key)
        final.append(path)
    files = final

    if BUNDLE.exists():
        BUNDLE.unlink()
    with zipfile.ZipFile(BUNDLE, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.write(path, path.relative_to(ROOT).as_posix())

    with zipfile.ZipFile(BUNDLE) as archive:
        names = archive.namelist()
        duplicated = len(names) - len(set(names))
        text = archive.read("docs/observation_terminal_v1/MANIFEST.md").decode()
        block = text.split("## Included")[1].split("## Not included")[0]
        rows = [line.split("|")[1].strip() for line in block.splitlines()
                if line.startswith("| ") and "---" not in line and "path" not in line]
        missing = [row for row in rows if row not in set(names)]
        # the manifest itself is written after its own table, so it is expected but unlisted
        expected = set(rows) | {"docs/observation_terminal_v1/MANIFEST.md"}
        unexpected = [name for name in names if name not in expected]
        result = {
            "bundle": str(BUNDLE), "bytes": BUNDLE.stat().st_size,
            "entries_in_zip": len(names), "included_in_manifest": len(rows),
            "duplicated_names": duplicated, "missing_from_zip": missing,
            "unexpected_in_zip": unexpected,
            "checkpoints_included": sorted(name for name in names if name.endswith(".pt")),
            "listed_only": [path.name for path in LISTED_ONLY],
        }
    (DOCS / "BUNDLE_CHECK.json").write_text(json.dumps(result, indent=1, sort_keys=True))
    print(json.dumps(result, indent=1))
    ok = not missing and not unexpected and duplicated == 0
    print("BUNDLE_OK" if ok else "BUNDLE_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
