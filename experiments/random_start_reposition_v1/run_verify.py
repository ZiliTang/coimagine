"""Run the section-7.2 semantics checks and write results/invariants.json."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import layout, verify  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/random_start_reposition_v1"))
    args = parser.parse_args()
    out = Path(args.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    instances, _ = layout.build_split("L0", "validation", count=4)
    report = verify.run_all(instances)
    (out / "results" / "invariants.json").write_text(json.dumps(report, indent=1, default=str))
    print(json.dumps({name: check["ok"] for name, check in report["checks"].items()}, indent=1))
    print(f"passed {report['passed']}/{report['total']}")
    print("VERIFY_OK" if report["all_ok"] else "VERIFY_FAILED")
    return 0 if report["all_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
