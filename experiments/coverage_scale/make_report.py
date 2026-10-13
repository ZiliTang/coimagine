"""Phase E: automatic tables and figures for the coverage-scale report.

    python experiments/coverage_scale/make_report.py --run-id coverage-scale-20261013

Everything it prints is read from files written by the collection and training runs; it never
recomputes a model result.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIER_ORDER = ["8000", "16000", "32000", "64000", "128000"]
WIDTHS = [256, 512]


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def collect_cells(run_dir: Path):
    rows = []
    for tier_dir in sorted(run_dir.glob("runs/*")):
        if not tier_dir.is_dir():
            continue
        summary = read_json(tier_dir / "run_summary.json")
        selection = read_json(tier_dir / "selection_eval.json")
        if summary is None or selection is None:
            continue
        kind = "best_new_val_total"
        entry = selection.get(kind)
        if entry is None:
            continue
        cell_meta = read_json(tier_dir / "cell_meta.json", {})
        rows.append({
            "tier": str(summary["tier"]), "width": int(summary["width"]), "seed": int(summary["seed"]),
            "windows": int(summary["windows"]), "updates": int(summary["updates"]),
            "epochs_seen": float(summary["epochs_seen"]), "wall_clock": float(summary["wall_clock"]),
            "selected_update": entry["update"],
            "P": entry["new_val"]["weighted"]["position_mean"],
            "invalid": entry["new_val"]["legality"]["invalid_rate_weighted"],
            "object": entry["new_val"]["weighted"]["object_position"],
            "robot0": entry["new_val"]["weighted"]["robot0_position"],
            "robot1": entry["new_val"]["weighted"]["robot1_position"],
            "angle": entry["new_val"]["weighted"]["object_angle"],
            "total_loss": entry["new_val"]["weighted"]["total_loss"],
            "unweighted_P": entry["new_val"]["unweighted"]["position_mean"],
            "unweighted_invalid": entry["new_val"]["legality"]["invalid_rate_unweighted"],
            "persistence": entry["new_val"]["weighted"]["persistence_object"],
            "success_recall": entry["new_val"]["success"]["recall"],
            "true_success_windows": entry["new_val"]["success"]["true_success_windows"],
            "train_full_loss": entry["train_full"]["weighted_total_loss"],
            "train_full_P": entry["train_full"]["weighted_position_mean"],
            "train_full_invalid": entry["train_full"]["weighted_invalid_rate"],
            "train_windows": entry["train_full"]["n"],
            "independent_roots": cell_meta.get("train", {}).get("independent_roots"),
            "monitor_windows": cell_meta.get("monitor_windows"),
        })
    return rows


def narrow_cells(run_dir: Path, tier: str = ""):
    rows = []
    for cell_dir in sorted((run_dir / "narrow").glob("*_seed*")):
        if tier and not cell_dir.name.startswith(f"{tier}_"):
            continue
        summary = read_json(cell_dir / "run_summary.json")
        selection = read_json(cell_dir / "selection_eval.json")
        if summary is None or selection is None:
            continue
        entry = selection["best_new_val_total"]
        rows.append({
            "tier": str(summary["tier"]), "width": int(summary["width"]), "seed": int(summary["seed"]),
            "windows": int(summary["windows"]),
            "P": entry["new_val"]["weighted"]["position_mean"],
            "invalid": entry["new_val"]["legality"]["invalid_rate_weighted"],
            "total_loss": entry["new_val"]["weighted"]["total_loss"],
            "success_recall": entry["new_val"]["success"]["recall"],
            "true_success_windows": entry["new_val"]["success"]["true_success_windows"],
        })
    return rows


def mean_std(rows, key):
    values = [row[key] for row in rows if row.get(key) is not None]
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, 0.0
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, variance ** 0.5


def learning_curve(rows, tiers, output: Path, title: str):
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    for width in WIDTHS:
        xs, ys, errs = [], [], []
        for tier in tiers:
            group = [row for row in rows if row["tier"] == tier and row["width"] == width]
            mean, std = mean_std(group, "P")
            if mean is None:
                continue
            xs.append(int(tier))
            ys.append(mean)
            errs.append(std)
        if xs:
            axes[0][0].errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=f"width {width}")
    axes[0][0].set_xscale("log")
    axes[0][0].set_xlabel("training windows (deduplicated)")
    axes[0][0].set_ylabel("P: mean of the three bodies' position error (m)")
    axes[0][0].set_title("new validation set: the frozen physical metric P")
    axes[0][0].legend(fontsize=8)
    axes[0][0].grid(alpha=0.3)

    for width in WIDTHS:
        xs, ys, errs = [], [], []
        for tier in tiers:
            group = [row for row in rows if row["tier"] == tier and row["width"] == width]
            mean, std = mean_std(group, "object")
            if mean is None:
                continue
            xs.append(int(tier))
            ys.append(mean)
            errs.append(std)
        if xs:
            axes[0][1].errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=f"model object, width {width}")
    persistence = [row["persistence"] for row in rows if row.get("persistence") is not None]
    if persistence:
        axes[0][1].axhline(sum(persistence) / len(persistence), color="grey", ls="--", lw=1.2,
                           label="persistence (predict no motion), object only")
    axes[0][1].set_xscale("log")
    axes[0][1].set_xlabel("training windows")
    axes[0][1].set_ylabel("object position error (m)")
    axes[0][1].set_title("object only, against the no-motion baseline")
    axes[0][1].legend(fontsize=8)
    axes[0][1].grid(alpha=0.3)

    for width in WIDTHS:
        xs, train_ys, val_ys = [], [], []
        for tier in tiers:
            group = [row for row in rows if row["tier"] == tier and row["width"] == width]
            if not group:
                continue
            xs.append(int(tier))
            train_ys.append(sum(row["train_full_loss"] for row in group) / len(group))
            val_ys.append(sum(row["total_loss"] for row in group) / len(group))
        if xs:
            axes[1][0].plot(xs, train_ys, marker="o", ls="--", label=f"train full, width {width}")
            axes[1][0].plot(xs, val_ys, marker="s", label=f"new val, width {width}")
    axes[1][0].set_xscale("log")
    axes[1][0].set_xlabel("training windows")
    axes[1][0].set_ylabel("weighted four-term total loss")
    axes[1][0].set_title("train / new-val gap")
    axes[1][0].legend(fontsize=8)
    axes[1][0].grid(alpha=0.3)

    for width in WIDTHS:
        xs, ys, errs = [], [], []
        for tier in tiers:
            group = [row for row in rows if row["tier"] == tier and row["width"] == width]
            mean, std = mean_std(group, "invalid")
            if mean is None:
                continue
            xs.append(int(tier))
            ys.append(100.0 * mean)
            errs.append(100.0 * std)
        if xs:
            axes[1][1].errorbar(xs, ys, yerr=errs, marker="o", capsize=3, label=f"width {width}")
    axes[1][1].set_xscale("log")
    axes[1][1].set_xlabel("training windows")
    axes[1][1].set_ylabel("predicted-state invalid rate (%)")
    axes[1][1].set_title("geometric legality of the predicted state")
    axes[1][1].legend(fontsize=8)
    axes[1][1].grid(alpha=0.3)
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)


def candidate_figure(run_dir: Path, output: Path):
    path = run_dir / "candidates" / "aggregate.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    groups: dict = defaultdict(list)
    for row in payload["aggregate"]:
        groups[str(row["label"]).split("/")[0]].append(row)
    order = [name for name in ("8000", "16000", "32000", "64000", "128000", "narrow_16017") if name in groups]
    metrics = [
        ("chosen_true_return", "chosen true J (unfiltered)"),
        ("feasible_chosen_true_return", "chosen true J (execution filtered)"),
        ("misrejection_loss", "misrejection loss"),
        ("ranking_loss", "ranking loss"),
        ("execution_regret", "execution regret"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    width = 0.8 / len(metrics)
    for index, (key, label) in enumerate(metrics):
        values, errors = [], []
        for group in order:
            mean, std = mean_std(groups[group], key)
            values.append(mean)
            errors.append(std)
        axes[0].bar([position + index * width for position in range(len(order))], values,
                    width=width, yerr=errors, capsize=2, label=label)
    axes[0].set_xticks([position + 0.4 - width / 2 for position in range(len(order))])
    axes[0].set_xticklabels(order, rotation=20)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_ylabel("true J")
    axes[0].set_title("fixed-candidate decision on 12 new test roots (H in {1,3}, K=64)")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3, axis="y")

    for index, (key, label) in enumerate((("rejection_rate", "rejection rate"),
                                          ("any_true_success", "starts with a truly successful candidate"))):
        values, errors = [], []
        for group in order:
            mean, std = mean_std(groups[group], key)
            values.append(None if mean is None else 100.0 * mean)
            errors.append(None if std is None else 100.0 * std)
        axes[1].bar([position + index * 0.4 for position in range(len(order))], values,
                    width=0.4, yerr=errors, capsize=2, label=label)
    axes[1].set_xticks([position + 0.2 for position in range(len(order))])
    axes[1].set_xticklabels(order, rotation=20)
    axes[1].set_ylabel("%")
    axes[1].set_title("feasibility filtering and candidate coverage")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)


def coverage_figure(run_dir: Path, output: Path):
    path = run_dir / "coverage_table.json"
    if not path.exists():
        return
    payload = json.loads(path.read_text())
    cells = []
    for split in ("train", "val", "test"):
        stats = payload["split_stats"].get(split)
        if not stats:
            continue
        cells.append((split, stats))
    if not cells:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    keys = ["S0/broad", "S0/wall", "S0/goal", "S1/broad", "S1/wall", "S1/goal"]
    colours = {"train": "tab:blue", "val": "tab:orange", "test": "tab:green"}
    order = [name for name, _ in cells]
    for split, stats in cells:
        roots = [stats["cells"].get(key, {}).get("roots", 0) for key in keys]
        windows = [stats["cells"].get(key, {}).get("windows", 0) for key in keys]
        positions = list(range(len(keys)))
        offset = 0.8 / len(cells) * order.index(split)
        axes[0].bar([p + offset for p in positions], roots, width=0.8 / len(cells),
                    color=colours.get(split), label=split)
        axes[1].bar([p + offset for p in positions], windows, width=0.8 / len(cells),
                    color=colours.get(split), label=split)
        success = [stats["cells"].get(key, {}).get("roots_with_success", 0)
                   / max(1, stats["cells"].get(key, {}).get("roots", 1)) for key in keys]
        axes[2].bar([p + offset for p in positions], [100.0 * value for value in success],
                    width=0.8 / len(cells), color=colours.get(split), label=split)
        for axis in axes:
            axis.set_xticks([p + 0.4 - 0.4 / len(cells) for p in positions])
            axis.set_xticklabels(keys, rotation=25, fontsize=7)
    axes[0].set_ylabel("collection roots")
    axes[0].set_title("independent roots per family x mode cell")
    axes[1].set_ylabel("real windows")
    axes[1].set_title("deduplicated windows per cell")
    axes[2].set_ylabel("% of roots with a success")
    axes[2].set_title("root success rate per cell")
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(output, dpi=130)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="coverage-scale-20261013")
    parser.add_argument("--out-root", default="outputs/coverage_scale_v1")
    args = parser.parse_args()
    run_dir = ROOT / args.out_root / args.run_id
    figures = run_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    rows = collect_cells(run_dir)
    tiers = [tier for tier in TIER_ORDER if any(row["tier"] == tier for row in rows)]
    (run_dir / "learning_curve_rows.json").write_text(json.dumps(rows, indent=1, sort_keys=True))

    # -- per-tier / per-width table -----------------------------------------
    table = []
    for tier in tiers:
        for width in WIDTHS:
            group = [row for row in rows if row["tier"] == tier and row["width"] == width]
            if not group:
                continue
            entry = {"tier": tier, "width": width, "seeds": len(group)}
            for key in ("windows", "updates", "epochs_seen", "P", "invalid", "object", "robot0",
                        "robot1", "angle", "total_loss", "train_full_loss", "train_full_P",
                        "train_full_invalid", "success_recall", "true_success_windows",
                        "unweighted_P", "unweighted_invalid", "selected_update"):
                mean, std = mean_std(group, key)
                entry[key] = mean
                entry[key + "_sd"] = std
            entry["train_windows"] = group[0]["train_windows"]
            entry["independent_roots"] = group[0]["independent_roots"]
            entry["success_recall_note"] = (
                "null: no true success window in the validation set"
                if entry["true_success_windows"] in (None, 0) else ""
            )
            table.append(entry)
    (run_dir / "tier_table.json").write_text(json.dumps(table, indent=1, sort_keys=True))

    if rows:
        learning_curve(rows, tiers, figures / "C_learning_curve.png",
                       "coverage_scale_v1: data scale x width on the frozen new validation set")
    candidate_figure(run_dir, figures / "D_candidates.png")
    coverage_figure(run_dir, figures / "B_coverage.png")

    # -- narrow control table -------------------------------------------------
    narrow = []
    narrow.extend(narrow_cells(run_dir))
    if narrow:
        (run_dir / "narrow_table.json").write_text(json.dumps(narrow, indent=1, sort_keys=True))

    # -- markdown appendix ----------------------------------------------------
    lines = ["# coverage_scale_v1 - automatic tables", ""]
    lines.append("## data scale x width (main selection point `best_new_val_total`, new val)")
    lines.append("")
    lines.append("| tier windows | width | seeds | actual windows | independent roots | updates | epochs | P (m) | invalid (%) | object (m) | robot0 (m) | robot1 (m) | angle (rad) |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in table:
        lines.append(
            f"| {row['tier']} | {row['width']} | {row['seeds']} | {row['windows']:.0f} | "
            f"{row['independent_roots']} | {row['updates']:.0f} | {row['epochs_seen']:.1f} | "
            f"{row['P']:.4f} | {100*row['invalid']:.1f} | {row['object']:.4f} | "
            f"{row['robot0']:.4f} | {row['robot1']:.4f} | {row['angle']:.4f} |"
        )
    lines.append("")
    lines.append("## train / validation gap and success identification")
    lines.append("")
    lines.append("| tier | width | train full loss | new val loss | train full P | train full invalid (%) | true success windows | recall |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in table:
        recall = "null" if row["success_recall"] is None else f"{row['success_recall']:.3f}"
        lines.append(
            f"| {row['tier']} | {row['width']} | {row['train_full_loss']:.5f} | {row['total_loss']:.5f} | "
            f"{row['train_full_P']:.4f} | {100*row['train_full_invalid']:.1f} | "
            f"{row['true_success_windows']:.0f} | {recall} |"
        )
    lines.append("")
    if narrow:
        lines.append("## equal-size narrow-coverage control (six fixed initial states, same recipe)")
        lines.append("")
        lines.append("| tier | width | seeds | windows | P (m) | invalid (%) | loss |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|")
        by = defaultdict(list)
        for row in narrow:
            by[(row["tier"], row["width"])].append(row)
        for (tier, width), group in sorted(by.items()):
            lines.append(
                f"| {tier} | {width} | {len(group)} | {group[0]['windows']} | "
                f"{sum(r['P'] for r in group)/len(group):.4f} | "
                f"{100*sum(r['invalid'] for r in group)/len(group):.1f} | "
                f"{sum(r['total_loss'] for r in group)/len(group):.5f} |"
            )
        lines.append("")
    (run_dir / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("REPORT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
