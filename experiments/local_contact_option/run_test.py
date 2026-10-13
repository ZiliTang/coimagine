"""Test phase: the frozen 120-root, 8-variant, 4-arm paired matrix.

    python experiments/local_contact_option/run_test.py --workers 12

The gain comes from ``dev_kv_selection.json``; the variant and arm lists are the frozen ones.
No knob is read from the test results, and the test set is never used to re-tune anything.
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
    VARIANT_SPECS,
    build_roots,
    manifest,
)
from coimagine.local_contact_option.runner import root_metadata, run_split  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-v1")
    parser.add_argument("--out-root", default="outputs/local_contact_option_v1")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--split", default="test", choices=("dev", "test"))
    args = parser.parse_args()

    out_dir = ROOT / args.out_root / args.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    selection = json.loads((out_dir / "dev_kv_selection.json").read_text())
    # the selection file may also carry the reference scale parameters chosen on the dev split;
    # applying them here (instead of editing defaults) keeps the change auditable
    scale = {key: float(value) for key, value in (selection.get("scales") or {}).items()}
    option = OptionConfig(kv=float(selection["selected_kv"]), **scale)

    per_class = TEST_PER_CLASS if args.split == "test" else DEV_PER_CLASS
    roots = build_roots(args.split, per_class)
    dev_path = out_dir / "roots_dev.json"
    dev_metadata = json.loads(dev_path.read_text())["roots"] if dev_path.exists() else []
    target = out_dir / (f"roots_{args.split}.json" if args.split == "test" else "roots_dev_matrix.json")
    target.write_text(json.dumps({
        "manifest": manifest(dev_metadata if args.split == "test" else [], roots),
        "frozen_kv": option.kv,
        "frozen_scales": scale,
        "roots": [root_metadata(root) for root in roots],
    }, indent=1, sort_keys=True))
    (out_dir / "roots_manifest.json").write_text(json.dumps(
        manifest([], roots), indent=1, sort_keys=True))

    summary = run_split(
        roots, out_dir=out_dir, option_config=option,
        variants=[spec.name for spec in VARIANT_SPECS], workers=args.workers,
        tag=args.split, log_every=args.log_every,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "option_config"}, indent=1))
    print("TEST_OK" if args.split == "test" else "DEV_MATRIX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
