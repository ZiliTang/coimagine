"""Phase A runner: V-only execution under a frozen settle configuration.

    python experiments/option_prediction_mechanism_v1/phase_a/run_a.py \
        --split dev --window 0.20 --tag window020

Only the settle estimator / timestamp handling differ between tags; the controller, forces,
stage timing, thresholds and whitelist are the frozen round-3 values. ``--window 0`` reproduces
the pre-round-3 estimator exactly (the "old protocol" arm of the paired comparison).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.roots import (  # noqa: E402
    DEV_PER_CLASS,
    TEST_PER_CLASS,
    build_roots,
    manifest,
)
from coimagine.local_contact_option.runner import root_metadata, run_split  # noqa: E402

#: the frozen v3 V controller, unchanged since round 1
BASE = dict(kv=25.0, a_ref_max=3.0, approach_max_speed=0.30, approach_contact_speed=0.06,
            approach_gain=4.0, ready_precontact_speed=0.02)
VARIANTS = ("baseline", "pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05", "noise", "delay", "param")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="option-prediction-mechanism-v1")
    parser.add_argument("--out-root", default="outputs/option_prediction_mechanism_v1")
    parser.add_argument("--split", default="dev", choices=("dev", "test"))
    parser.add_argument("--window", type=float, default=0.0)
    parser.add_argument("--min-baseline", type=float, default=0.10)
    parser.add_argument("--max-sample-gap", type=float, default=0.15)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    option = replace(
        OptionConfig(**BASE),
        settle_window=args.window,
        settle_window_min_baseline=args.min_baseline,
        settle_max_sample_gap=args.max_sample_gap,
    )
    per_class = TEST_PER_CLASS if args.split == "test" else DEV_PER_CLASS
    roots = build_roots(args.split, per_class)
    (out_dir / f"roots_{args.split}.json").write_text(json.dumps({
        "frozen_config": {
            "controller": BASE, "force_limit": option.force_limit,
            "settle_surface": option.settle_surface, "settle_hold": option.settle_hold,
            "settle_linear": option.settle_linear, "settle_angular": option.settle_angular,
            "brake_settle_limit": option.brake_settle_limit,
            "round3": {"settle_window": args.window, "min_baseline": args.min_baseline,
                       "max_sample_gap": args.max_sample_gap,
                       "surface_freshness": option.surface_freshness
                       if hasattr(option, "surface_freshness") else 0.20},
        },
        "manifest": manifest([], roots),
        "roots": [root_metadata(root) for root in roots],
    }, indent=1, sort_keys=True))
    print(f"{args.split}: {len(roots)} roots x {len(VARIANTS)} variants x 1 arm", flush=True)
    payload = run_split(
        roots, out_dir=out_dir, option_config=option, variants=list(VARIANTS), arms=["V"],
        workers=args.workers, tag=args.tag,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"},
                     indent=1, default=str))
    print("RUN_A_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
