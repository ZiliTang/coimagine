"""C: the paired V/X experiment.

    python experiments/local_position_push/run_pair.py --split dev
    python experiments/local_position_push/run_pair.py --split test

Arm V is the frozen v3 velocity control; arm X is the surface-position feedback control. Both
run the *identical* APPROACH/READY/PUSH-clock/BRAKE/SETTLE protocol, on the same roots, the same
8 perturbation variants and the same initial states, so the only difference is the PUSH force.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.roots import (  # noqa: E402
    DEV_PER_CLASS,
    TEST_PER_CLASS,
    build_roots,
)
from coimagine.local_contact_option.roots import manifest  # noqa: E402
from coimagine.local_contact_option.runner import (  # noqa: E402
    root_metadata,
    run_split,
)

ARMS = ("V", "X")
VARIANTS = ("baseline", "pos_2mm", "pos_10mm", "vel_0.02", "vel_0.05", "noise", "delay",
            "param")

#: the frozen v3 baseline: Kv=25 N s/m, reference acceleration cap 3.0 m/s^2, and the reference
#: scale parameters chosen on the dev split of that round
BASE = dict(kv=25.0, a_ref_max=3.0, approach_max_speed=0.30, approach_contact_speed=0.06,
            approach_gain=4.0, ready_precontact_speed=0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--split", default="test", choices=("dev", "test"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--log-every", type=int, default=1)
    args = parser.parse_args()

    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    option = OptionConfig(**BASE)
    per_class = TEST_PER_CLASS if args.split == "test" else DEV_PER_CLASS
    roots = build_roots(args.split, per_class)
    target = out_dir / (f"roots_{args.split}.json" if args.split == "test" else "roots_dev.json")
    target.write_text(json.dumps({
        "frozen_config": {
            "baseline": BASE, "kp": option.kp, "kd": option.kd, "e_max": option.e_max,
            "beta_ramp": option.beta_ramp, "force_limit": option.force_limit,
            "patch_support_margin": option.patch_support_margin,
            "delta_per_speed": option.kd / option.kp,
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
