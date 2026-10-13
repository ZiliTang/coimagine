"""Round 2 paired V/X experiment with an explicit settle-reliability configuration.

    python experiments/local_position_push_v2/run_pair.py --split test --workers 12 \
        --settle-window 0.30 --brake-hold-force 0.0 --settle-surface 0.02

Every run records the full ``option_config`` it used, so the settle-reliability condition is a
recorded parameter change and not an edit smuggled into the middle of the round. The physics,
whitelist and frozen scenes are untouched; the only differences from round 1 live in the
BRAKE/SETTLE protocol (which estimator decides "quiet", how long it may take, and whether the
brake keeps holding the object).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.roots import (  # noqa: E402
    DEV_PER_CLASS,
    TEST_PER_CLASS,
    build_roots,
    manifest,
)
from coimagine.local_contact_option.runner import root_metadata, run_split  # noqa: E402

ARMS = ("V", "X")
VARIANTS = ("baseline", "pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05", "noise", "delay", "param")

#: the frozen v3 baseline controls, identical to round 1
BASE = dict(kv=25.0, a_ref_max=3.0, approach_max_speed=0.30, approach_contact_speed=0.06,
            approach_gain=4.0, ready_precontact_speed=0.02)


def build_config(args) -> OptionConfig:
    return replace(
        OptionConfig(**BASE),
        settle_surface=args.settle_surface,
        settle_window=args.settle_window,
        settle_window_min_baseline=args.settle_window_min_baseline,
        brake_hold_force=args.brake_hold_force,
        brake_settle_limit=args.brake_settle_limit,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v2")
    parser.add_argument("--out-root", default="outputs/local_position_push_v2")
    parser.add_argument("--split", default="test", choices=("dev", "test"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--log-every", type=int, default=1)
    # settle-reliability condition (round 2). Defaults are the round-2 protocol.
    parser.add_argument("--settle-surface", type=float, default=0.02)
    parser.add_argument("--settle-window", type=float, default=0.30)
    parser.add_argument("--settle-window-min-baseline", type=float, default=0.10)
    parser.add_argument("--brake-hold-force", type=float, default=0.0)
    parser.add_argument("--brake-settle-limit", type=float, default=4.0)
    args = parser.parse_args()

    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    option = build_config(args)
    per_class = TEST_PER_CLASS if args.split == "test" else DEV_PER_CLASS
    roots = build_roots(args.split, per_class)
    target = out_dir / (f"roots_{args.split}.json" if args.split == "test" else "roots_dev.json")
    target.write_text(json.dumps({
        "frozen_config": {
            "baseline": BASE, "kp": option.kp, "kd": option.kd, "e_max": option.e_max,
            "beta_ramp": option.beta_ramp, "force_limit": option.force_limit,
            "patch_support_margin": option.patch_support_margin,
            "delta_per_speed": option.kd / option.kp,
            "round2_settle": {
                "settle_surface": option.settle_surface,
                "settle_window": option.settle_window,
                "settle_window_min_baseline": option.settle_window_min_baseline,
                "brake_hold_force": option.brake_hold_force,
                "brake_settle_limit": option.brake_settle_limit,
            },
        },
        "manifest": manifest([], roots),
        "roots": [root_metadata(root) for root in roots],
    }, indent=1, sort_keys=True))
    (out_dir / "roots_manifest.json").write_text(json.dumps(manifest([], roots), indent=1,
                                                           sort_keys=True))
    print(f"{args.split}: {len(roots)} roots x {len(args.variants.split(','))} variants x "
          f"{len(args.arms.split(','))} arms", flush=True)
    payload = run_split(
        roots, out_dir=out_dir, option_config=option,
        variants=args.variants.split(","), arms=args.arms.split(","),
        workers=args.workers, log_every=args.log_every, tag=args.split,
    )
    print(json.dumps({key: value for key, value in payload.items() if key != "records"},
                     indent=1, default=str))
    print("RUN_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
