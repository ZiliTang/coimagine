"""Development phase: select and freeze the velocity-feedback gain Kv.

    python experiments/local_contact_option/run_dev.py --workers 8

Only the development roots are used. The rule is frozen before the run: take the **smallest**
gain that satisfies every acceptance check below, prioritising contact retention over
penetration and oscillation. No test root is read here, and the gain cannot change afterwards.

Acceptance for a candidate gain ``Kv`` (all on development roots, baseline variant, arm N
unless stated):

1. contact retention: the fraction of nominally pushable executions that reach 2.0 s of
   effective contact is at least 0.90;
2. penetration: no execution exceeds the frozen 0.05 m penetration limit;
3. oscillation: at most 10% of executions spend more than 20% of their push periods above
   1.5x the commanded speed.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.local_contact_option.contract import (  # noqa: E402
    KV_DEV_CANDIDATES,
    OptionConfig,
)
from coimagine.local_contact_option.roots import (  # noqa: E402
    CLASSES,
    DEV_PER_CLASS,
    build_roots,
    manifest,
)
from coimagine.local_contact_option.runner import root_metadata, run_split  # noqa: E402

PUSHABLE_CLASSES = ("solo_central", "solo_offset", "dual_same_side", "dual_rotate")
CONTACT_RETENTION_MIN = 0.90
PENETRATION_LIMIT = 0.05
OVERSHOOT_EXECUTION_MAX = 0.10
OVERSHOOT_PERIOD_MAX = 0.20


def acceptance(records, *, kv: float) -> dict:
    dev_push = [
        record for record in records
        if record["arm"] == "N" and record["variant"] == "baseline"
        and record["interaction_class"] in PUSHABLE_CLASSES
    ]
    retention = (
        sum(1 for record in dev_push if record["metrics"]["reached_effective_push"]) / len(dev_push)
        if dev_push else None
    )
    penetration_bad = sum(
        1 for record in records
        if record["metrics"].get("max_penetration", 0.0) > PENETRATION_LIMIT
    )
    overshoot_cases = sum(
        1 for record in records
        if record["metrics"].get("speed_overshoot_ratio", 0.0) > OVERSHOOT_PERIOD_MAX
    )
    overshoot_ratio = overshoot_cases / max(1, len(records))
    checks = {
        "contact_retention": None if retention is None else retention >= CONTACT_RETENTION_MIN,
        "penetration_within_limit": penetration_bad == 0,
        "oscillation_below_limit": overshoot_ratio <= OVERSHOOT_EXECUTION_MAX,
    }
    return {
        "kv": kv,
        "dev_pushable_executions": len(dev_push),
        "contact_retention": retention,
        "penetration_violations": penetration_bad,
        "overshoot_execution_ratio": overshoot_ratio,
        "checks": checks,
        "qualified": all(value is True for value in checks.values()),
        "order": {"contact_retention": -(retention or 0.0),
                  "penetration_violations": penetration_bad,
                  "overshoot_execution_ratio": overshoot_ratio},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-v1")
    parser.add_argument("--out-root", default="outputs/local_contact_option_v1")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--gains", default=",".join(str(value) for value in KV_DEV_CANDIDATES))
    args = parser.parse_args()

    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dev_roots = build_roots("dev", DEV_PER_CLASS)
    (out_dir / "roots_dev.json").write_text(json.dumps({
        "manifest": manifest(dev_roots, []),
        "roots": [root_metadata(root) for root in dev_roots],
    }, indent=1, sort_keys=True))

    gains = [float(value) for value in args.gains.split(",") if value]
    trials = []
    selected = None
    for kv in sorted(gains):
        option = OptionConfig(kv=kv)
        summary = run_split(
            dev_roots, out_dir=out_dir, option_config=option, variants=["baseline"],
            arms=("N", "F"), workers=args.workers, tag=f"dev_kv{int(kv)}",
        )
        records = [json.loads(line) for line in Path(summary["executions_path"]).open()]
        verdict = acceptance(records, kv=kv)
        verdict["summary"] = {key: value for key, value in summary.items() if key != "option_config"}
        trials.append(verdict)
        print(f"Kv={kv}: retention={verdict['contact_retention']} "
              f"penetration_violations={verdict['penetration_violations']} "
              f"overshoot_ratio={verdict['overshoot_execution_ratio']:.3f} "
              f"qualified={verdict['qualified']}", flush=True)
        if selected is None and verdict["qualified"]:
            selected = kv
    if selected is None:
        selected = min(trials, key=lambda row: (row["order"]["contact_retention"],
                                                row["order"]["penetration_violations"],
                                                row["order"]["overshoot_execution_ratio"]))["kv"]
        fallback = True
    else:
        fallback = False
    payload = {
        "candidates": gains, "selected_kv": selected, "fallback_by_ranking": fallback,
        "rule": {
            "contact_retention_min": CONTACT_RETENTION_MIN,
            "penetration_limit": PENETRATION_LIMIT,
            "overshoot_execution_max": OVERSHOOT_EXECUTION_MAX,
            "overshoot_period_max": OVERSHOOT_PERIOD_MAX,
            "pushable_classes": list(PUSHABLE_CLASSES),
            "order": "smallest qualified gain; contact retention first, then penetration, "
                     "then oscillation",
        },
        "trials": trials,
    }
    (out_dir / "dev_kv_selection.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({"selected_kv": selected, "fallback": fallback}, indent=1))
    print("DEV_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
