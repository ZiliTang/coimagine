"""Phase C3: the frozen go / no-go rule for the next data tier.

    python experiments/coverage_scale/run_decision.py --run-id coverage-scale-20261013 --tiers 8000,16000

Reads only the new validation set, at the main selection point, and applies the thresholds
frozen in the task book (5% relative P decrease with <= 2 pp illegal-rate worsening, or >= 3 pp
absolute illegal-rate decrease with <= 5% P worsening; at least 2/3 seeds; no family worse by
more than 10% P or 5 pp illegal rate). Thresholds are never adapted to the observed trend.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.coverage_scale.evaluation import family_metrics  # noqa: E402

P_RELATIVE_IMPROVEMENT = 0.05
P_RELATIVE_WORSENING = 0.05
INVALID_ABSOLUTE_IMPROVEMENT = 0.03
INVALID_WORSENING_LIMIT = 0.02
SEED_AGREEMENT_REQUIRED = 2
FAMILY_P_WORSENING_LIMIT = 0.10
FAMILY_INVALID_WORSENING_LIMIT = 0.05


def load_cells(run_dir: Path, tier: str, widths, seeds, kind: str = "best_new_val_total",
               subdir: str = "runs", prefix: str = "") -> dict:
    cells = {}
    for width in widths:
        for seed in seeds:
            path = run_dir / subdir / f"{prefix}{tier}_{width}_seed{seed}" / "selection_eval.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            if kind not in payload:
                continue
            summary = payload[kind]["new_val"]
            cells[(width, seed)] = {
                "update": payload[kind]["update"],
                "P": summary["weighted"]["position_mean"],
                "invalid": summary["legality"]["invalid_rate_weighted"],
                "object": summary["weighted"]["object_position"],
                "robot0": summary["weighted"]["robot0_position"],
                "robot1": summary["weighted"]["robot1_position"],
                "angle": summary["weighted"]["object_angle"],
                "total_loss": summary["weighted"]["total_loss"],
                "success_recall": summary["success"]["recall"],
                "true_success_windows": summary["success"]["true_success_windows"],
                "families": family_metrics(summary),
                "unweighted_P": summary["unweighted"]["position_mean"],
                "unweighted_invalid": summary["unweighted"]["invalid"],
                "persistence": summary["weighted"]["persistence_object"],
            }
    return cells


def compare_window(previous: dict, current: dict, widths, seeds) -> dict:
    """Pair the same width and the same seed across two sizes."""

    result = {"per_width": {}, "any_benefit": False, "widths_with_benefit": []}
    for width in widths:
        rows = []
        for seed in seeds:
            before = previous.get((width, seed))
            after = current.get((width, seed))
            if before is None or after is None:
                continue
            p_change = (after["P"] - before["P"]) / before["P"] if before["P"] else None
            invalid_change = after["invalid"] - before["invalid"]
            rows.append({
                "seed": seed, "P_before": before["P"], "P_after": after["P"],
                "P_relative_change": p_change, "invalid_before": before["invalid"],
                "invalid_after": after["invalid"], "invalid_pp_change": 100.0 * invalid_change,
                "success_recall": after["success_recall"],
                "true_success_windows": after["true_success_windows"],
            })
        if not rows:
            continue
        criterion_p = [
            row for row in rows
            if row["P_relative_change"] is not None
            and row["P_relative_change"] <= -P_RELATIVE_IMPROVEMENT
            and row["invalid_pp_change"] <= 100.0 * INVALID_WORSENING_LIMIT
        ]
        criterion_invalid = [
            row for row in rows
            if row["invalid_pp_change"] <= -100.0 * INVALID_ABSOLUTE_IMPROVEMENT
            and row["P_relative_change"] is not None
            and row["P_relative_change"] <= P_RELATIVE_WORSENING
        ]
        family_flags = {}
        for family in ("S0", "S1"):
            p_before = [previous[(width, seed)]["families"].get(family, {}).get("position_mean")
                        for seed in seeds if (width, seed) in previous]
            p_after = [current[(width, seed)]["families"].get(family, {}).get("position_mean")
                       for seed in seeds if (width, seed) in current]
            i_before = [previous[(width, seed)]["families"].get(family, {}).get("invalid_rate")
                        for seed in seeds if (width, seed) in previous]
            i_after = [current[(width, seed)]["families"].get(family, {}).get("invalid_rate")
                       for seed in seeds if (width, seed) in current]
            pairs = [(b, a) for b, a in zip(p_before, p_after) if b and a]
            pair_i = [(b, a) for b, a in zip(i_before, i_after) if b is not None and a is not None]
            p_change = (sum(a for _, a in pairs) / len(pairs)) / (sum(b for b, _ in pairs) / len(pairs)) - 1.0 if pairs else None
            i_change = (sum(a for _, a in pair_i) / len(pair_i)) - (sum(b for b, _ in pair_i) / len(pair_i)) if pair_i else None
            family_flags[family] = {
                "P_relative_change": p_change, "invalid_pp_change": None if i_change is None else 100.0 * i_change,
                "P_worse_beyond_limit": bool(p_change is not None and p_change > FAMILY_P_WORSENING_LIMIT),
                "invalid_worse_beyond_limit": bool(i_change is not None and i_change > FAMILY_INVALID_WORSENING_LIMIT),
            }
        mixed = any(flag["P_worse_beyond_limit"] or flag["invalid_worse_beyond_limit"] for flag in family_flags.values())
        benefit = (len(criterion_p) >= SEED_AGREEMENT_REQUIRED or len(criterion_invalid) >= SEED_AGREEMENT_REQUIRED) and not mixed
        result["per_width"][str(width)] = {
            "rows": rows,
            "seeds_meeting_P_criterion": len(criterion_p),
            "seeds_meeting_invalid_criterion": len(criterion_invalid),
            "families": family_flags,
            "mixed_family_trend": mixed,
            "benefit": benefit,
        }
        if benefit:
            result["any_benefit"] = True
            result["widths_with_benefit"].append(width)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    parser.add_argument("--tiers", default="8000")
    parser.add_argument("--widths", default="256,512")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--narrow-tier", default="")
    parser.add_argument("--narrow-wide-tier", default="16000",
                        help="the wide tier the narrow control is equal in window count to")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    widths = [int(value) for value in args.widths.split(",") if value]
    seeds = [int(value) for value in args.seeds.split(",") if value]
    tiers = [value for value in args.tiers.split(",") if value]
    by_kind = {}
    for kind in ("best_new_val_total", "final"):
        cells = {tier: load_cells(run_dir, tier, widths, seeds, kind) for tier in tiers}
        windows = {}
        for index in range(1, len(tiers)):
            windows[f"{tiers[index-1]}->{tiers[index]}"] = compare_window(
                cells[tiers[index - 1]], cells[tiers[index]], widths, seeds
            )
        by_kind[kind] = {"tiers": tiers, "cells": {
            tier: {f"{width}_{seed}": row for (width, seed), row in cells[tier].items()} for tier in tiers
        }, "transitions": windows}

    sensitivity = {}
    for index in range(1, len(tiers)):
        main = by_kind["best_new_val_total"]["transitions"][f"{tiers[index-1]}->{tiers[index]}"]
        fixed = by_kind["final"]["transitions"][f"{tiers[index-1]}->{tiers[index]}"]
        sensitivity[f"{tiers[index-1]}->{tiers[index]}"] = {
            "main_any_benefit": main["any_benefit"],
            "final_any_benefit": fixed["any_benefit"],
            "only_final_improves": (not main["any_benefit"]) and fixed["any_benefit"],
        }

    narrow = None
    if args.narrow_tier:
        narrow_cells = load_cells(run_dir, args.narrow_tier, widths, seeds, subdir="narrow", prefix="")
        wide_cells = by_kind["best_new_val_total"]["cells"].get(args.narrow_wide_tier, {})
        comparison = {}
        for width in widths:
            for seed in seeds:
                key = f"{width}_{seed}"
                if (width, seed) in narrow_cells and key in wide_cells:
                    narrow_row = narrow_cells[(width, seed)]
                    wide_row = wide_cells[key]
                    comparison[key] = {
                        "narrow": narrow_row, "wide": wide_row,
                        "P_relative_change_wide_vs_narrow":
                            (wide_row["P"] - narrow_row["P"]) / narrow_row["P"],
                        "invalid_pp_change_wide_vs_narrow":
                            100.0 * (wide_row["invalid"] - narrow_row["invalid"]),
                    }
        narrow = {"narrow_tier": args.narrow_tier, "wide_tier": args.narrow_wide_tier,
                  "independent_initial_states": 6,
                  "cells": {f"{w}_{s}": row for (w, s), row in narrow_cells.items()},
                  "wide_vs_narrow": comparison}

    payload = {
        "thresholds": {
            "P_relative_improvement": P_RELATIVE_IMPROVEMENT,
            "P_relative_worsening": P_RELATIVE_WORSENING,
            "invalid_absolute_improvement_pp": 100.0 * INVALID_ABSOLUTE_IMPROVEMENT,
            "invalid_worsening_limit_pp": 100.0 * INVALID_WORSENING_LIMIT,
            "seed_agreement_required": SEED_AGREEMENT_REQUIRED,
            "family_P_worsening_limit": FAMILY_P_WORSENING_LIMIT,
            "family_invalid_worsening_limit_pp": 100.0 * FAMILY_INVALID_WORSENING_LIMIT,
        },
        "main_selection_point": "best_new_val_total",
        "by_selection_point": by_kind,
        "sensitivity": sensitivity,
        "narrow": narrow,
        "note": "engineering stop rule, not a statistical significance claim",
    }
    (run_dir / "decision.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({
        "tiers": tiers,
        "transitions": {
            key: {"any_benefit": value["any_benefit"],
                  "widths_with_benefit": value["widths_with_benefit"]}
            for key, value in by_kind["best_new_val_total"]["transitions"].items()
        },
        "sensitivity": sensitivity,
    }, indent=1))
    print("DECISION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
