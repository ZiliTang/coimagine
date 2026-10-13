"""Build the 128k / 256k comparison table, the figure and the short conclusion.

    python experiments/coverage_scale/make_report_256k.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA_RUN = "outputs/coverage_scale_v1/coverage-scale-20261013"
OUT_RUN = "outputs/coverage_scale_v1/coverage-scale-256k"
BODIES = ("object_position", "robot0_position", "robot1_position")
PERSISTENCE = ("persistence_object", "persistence_robot0", "persistence_robot1")
GROUPS = ("all", "translation", "rotation", "stationary", "contact", "no_contact")


def read(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def fmt(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    args = parser.parse_args()
    data_dir = ROOT / DATA_RUN
    out_dir = ROOT / OUT_RUN
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    groups = read(out_dir / "groups_table.json", {})
    holdout = read(out_dir / "holdout_256k.json", {"rows": []})
    cand = read(out_dir / "candidates" / "aggregate_by_horizon.json", {})
    manifest = read(out_dir / "model_list.json", {"models": []})
    comparison = {}

    # -- per-tier training identity ----------------------------------------
    for tier in ("128000", "256000"):
        base = data_dir / "runs" if tier == "128000" else out_dir / "runs"
        metas = []
        for cell in sorted(base.glob(f"{tier}_*_seed*")):
            meta = read(cell / "cell_meta.json")
            if meta:
                metas.append(meta)
        if metas:
            comparison.setdefault(tier, {})["train"] = {
                "windows": metas[0]["train"]["actual_windows"],
                "independent_roots": metas[0]["train"]["independent_roots"],
                "updates": mean([meta.get("updates") for meta in metas]),
                "epochs_seen": mean([meta.get("epochs_seen") or
                                     (meta.get("updates", 0) * 128 / max(1, meta["train"]["actual_windows"]))
                                     for meta in metas]),
                "wall_clock": mean([meta.get("wall_clock") for meta in metas]),
                "sample_order_hash_per_seed": sorted({(meta["seed"], meta["sample_order_hash"][:12])
                                                      for meta in metas}),
            }

    # -- new validation (group "all") and test ------------------------------
    for tier in ("128000", "256000"):
        for width in (256, 512):
            key = f"{tier}/{width}/best_new_val_total"
            if key in groups:
                entry = groups[key]["groups"]["all"]

                def pick(name, weighted_name):
                    value = entry.get(weighted_name)
                    return entry[name] if value is None else value

                comparison.setdefault(tier, {}).setdefault("val", {})[str(width)] = {
                    "object": pick("object_position", "weighted_object_position"),
                    "robot0": pick("robot0_position", "weighted_robot0_position"),
                    "robot1": pick("robot1_position", "weighted_robot1_position"),
                    "P": pick("position_mean", "weighted_position_mean"),
                    "unweighted_object": entry["object_position"],
                    "unweighted_robot0": entry["robot0_position"],
                    "unweighted_robot1": entry["robot1_position"],
                    "unweighted_P": entry["position_mean"],
                    "persistence_object": entry["persistence_object"],
                    "persistence_robot0": entry["persistence_robot0"],
                    "persistence_robot1": entry["persistence_robot1"],
                    "invalid_rate": pick("invalid_rate", "weighted_invalid_rate"),
                    "unweighted_invalid_rate": entry["invalid_rate"],
                    "success_windows": entry["success_windows"],
                }
    collected = defaultdict(list)
    for row in holdout["rows"]:
        if row["kind"] != "best_new_val_total":
            continue
        collected[(row["tier"], str(row["width"]))].append(row["new_test"])
    for (tier, width), entries in collected.items():
        comparison.setdefault(tier, {}).setdefault("test", {})[width] = {
            "seeds": len(entries),
            "P": mean([e["weighted"]["position_mean"] for e in entries]),
            "object": mean([e["weighted"]["object_position"] for e in entries]),
            "robot0": mean([e["weighted"]["robot0_position"] for e in entries]),
            "robot1": mean([e["weighted"]["robot1_position"] for e in entries]),
            "invalid_rate": mean([e["legality"]["invalid_rate_weighted"] for e in entries]),
            "recall": mean([e["success"]["recall"] for e in entries]),
            "persistence_object": mean([e["persistence_weighted_object_position"] for e in entries]),
        }

    # -- candidate selection, H=1 and H=3 apart ------------------------------
    for horizon, block in cand.get("by_horizon", {}).items():
        for row in block["aggregate"]:
            if not row["label"].endswith("best_new_val_total"):
                continue
            tier, width, seed, _ = row["label"].split("/")
            comparison.setdefault(tier, {}).setdefault(f"candidates_H{horizon}", {}).setdefault(width, []).append({
                "seed": int(seed), "chosen_true_return": row["chosen_true_return"],
                "feasible_chosen_true_return": row["feasible_chosen_true_return"],
                "misrejection_loss": row["misrejection_loss"], "ranking_loss": row["ranking_loss"],
                "execution_regret": row["execution_regret"],
                "rejection_rate": row["rejection_rate"], "no_feasible_ratio": row["no_feasible_ratio"],
                "no_true_success_ratio": row["no_true_success_ratio"],
            })

    (out_dir / "comparison.json").write_text(json.dumps(comparison, indent=1, sort_keys=True))

    # -- the comparison table -------------------------------------------------
    lines = ["# coverage_scale_v1 — 128k / 256k 对照表", "",
             "同一新验证集（3584 窗口 / 128 独立根）与同一新测试集（3497 窗口 / 128 独立根）；"
             "模型清单先冻结后评测试集。", ""]
    lines.append("## 训练身份与规模")
    lines.append("")
    lines.append("| 档 | 去重窗口 | 独立根 | 更新数 | 遍数 | 单 run 墙钟(s) |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for tier in ("128000", "256000"):
        train = comparison.get(tier, {}).get("train")
        if train:
            lines.append(f"| {tier} | {train['windows']} | {train['independent_roots']} | "
                         f"{train['updates']:.0f} | {train['epochs_seen']:.1f} | {train['wall_clock']:.0f} |")
    lines.append("")
    lines.append("## 新验证集（主选点；冻结的根→格→权重聚合，括号内为窗口均值）")
    lines.append("")
    lines.append("| 档 | 宽度 | 物体(m) | 物体 persistence | 机器人0(m) | 机器人0 persistence | 机器人1(m) | 机器人1 persistence | P(m) | 非法率(%) | 成功窗 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for tier in ("128000", "256000"):
        for width in ("256", "512"):
            val = comparison.get(tier, {}).get("val", {}).get(width)
            if not val:
                continue
            lines.append(
                f"| {tier} | {width} | {fmt(val['object'])} ({fmt(val['unweighted_object'])}) | "
                f"{fmt(val['persistence_object'])} | "
                f"{fmt(val['robot0'])} ({fmt(val['unweighted_robot0'])}) | {fmt(val['persistence_robot0'])} | "
                f"{fmt(val['robot1'])} ({fmt(val['unweighted_robot1'])}) | {fmt(val['persistence_robot1'])} | "
                f"{fmt(val['P'])} ({fmt(val['unweighted_P'])}) | "
                f"{100*val['invalid_rate']:.1f} ({100*val['unweighted_invalid_rate']:.1f}) | "
                f"{val['success_windows']} |")
    lines.append("")
    lines.append("每列的 persistence 是该刚体「预测不动」的位移，与被评估模型无关。")
    lines.append("")
    lines.append("## 同预算对照（更新数一致时的更多数据作用）")
    lines.append("")
    lines.append("| 档 | 宽度 | seed | 选点 | 更新数 | 加权 P(m) | 非法率(%) |")
    lines.append("|---|---:|---:|---|---:|---:|---:|")
    for tier, base in (("128000", ROOT / DATA_RUN / "runs"), ("256000", ROOT / OUT_RUN / "runs")):
        for width in (256, 512):
            for seed in (0, 1, 2):
                cell = base / f"{tier}_{width}_seed{seed}"
                selection = read(cell / "selection_eval.json")
                summary = read(cell / "run_summary.json", {})
                if not selection:
                    continue
                for kind in ("update60000", "best_new_val_total", "final"):
                    entry = selection.get(kind)
                    if not entry:
                        continue
                    update = entry.get("update")
                    if update is None:
                        update = (60000 if kind == "update60000" else summary.get("updates"))
                    lines.append(
                        f"| {tier} | {width} | {seed} | {kind} | {update} | "
                        f"{fmt(entry['new_val']['weighted']['position_mean'])} | "
                        f"{100*entry['new_val']['legality']['invalid_rate_weighted']:.1f} |")
    lines.append("")
    lines.append("注：128k 档的 `final` 就是 update60000，因此 128k 的 `final` 与 256k 的 `update60000` "
                 "更新数相同，可直接比较「同样更新、两倍数据」。")
    lines.append("")
    lines.append("## 新测试集（主选点，一次性评估；三个种子均值）")
    lines.append("")
    lines.append("| 档 | 宽度 | P(m) | 物体(m) | 机器人0(m) | 机器人1(m) | 非法率(%)) | 成功召回 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for tier in ("128000", "256000"):
        for width in ("256", "512"):
            test = comparison.get(tier, {}).get("test", {}).get(width)
            if not test:
                continue
            recall = "null" if test["recall"] is None else f"{test['recall']:.3f}"
            lines.append(f"| {tier} | {width} | {fmt(test['P'])} | {fmt(test['object'])} | "
                         f"{fmt(test['robot0'])} | {fmt(test['robot1'])} | "
                         f"{100*test['invalid_rate']:.1f} | {recall} |")
    lines.append("")
    lines.append("## 分组（新验证集，主选点，两宽度三种子均值；分组阈值 0.01 m / 0.02 rad）")
    lines.append("")
    lines.append("| 分组 | n 窗口 | n 根 | 档 | 物体(m) | 机器人0(m) | 机器人1(m) | 非法率(%) |")
    lines.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for name in GROUPS:
        for tier in ("128000", "256000"):
            cells = [groups.get(f"{tier}/{width}/best_new_val_total", {}).get("groups", {}).get(name)
                     for width in (256, 512)]
            cells = [cell for cell in cells if cell]
            if not cells:
                continue
            lines.append(f"| {name} | {cells[0]['n_windows']} | {cells[0]['n_roots']} | {tier} | "
                         f"{fmt(mean([c['object_position'] for c in cells]))} | "
                         f"{fmt(mean([c['robot0_position'] for c in cells]))} | "
                         f"{fmt(mean([c['robot1_position'] for c in cells]))} | "
                         f"{100*mean([c['invalid_rate'] for c in cells]):.1f} |")
    lines.append("")
    overlap = None
    for key, value in groups.items():
        if key.startswith("128000/") and "overlap" in value:
            overlap = value["overlap"]
            break
    if overlap:
        lines.append("分组是否重叠：运动分组与接触分组是**两个独立的划分**，互相重叠；运动分组内部 "
                     "`translation` 与 `rotation` 也可以同时成立，只有 `stationary` 是二者并集的补集。"
                     f"实测：平移且转动 {overlap['translation_and_rotation']}、仅平移 {overlap['translation_only']}、"
                     f"仅转动 {overlap['rotation_only']}、静止 {overlap['stationary']}、"
                     f"接触且平移 {overlap['contact_and_translation']}、无接触且静止 {overlap['no_contact_and_stationary']}。")
        lines.append("")
    lines.append("## 固定候选（主选点；H=1 与 H=3 分表；共同 (start_id, candidate_seed) 配对）")
    lines.append("")
    for horizon in ("1", "3"):
        block = comparison.get("128000", {}).get(f"candidates_H{horizon}") or \
            comparison.get("256000", {}).get(f"candidates_H{horizon}")
        if not block:
            continue
        lines.append(f"### H = {horizon}")
        lines.append("")
        lines.append("| 档 | 宽度 | 选中真实 J（未筛选） | 选中真实 J（执行筛选） | 误拒损失 | 排序损失 | 执行 regret | 拒绝率 | 无可行方案比例 | 无好候选比例 |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for tier in ("128000", "256000"):
            for width in ("256", "512"):
                rows = comparison.get(tier, {}).get(f"candidates_H{horizon}", {}).get(width)
                if not rows:
                    continue
                lines.append(
                    f"| {tier} | {width} | {fmt(mean([r['chosen_true_return'] for r in rows]))} | "
                    f"{fmt(mean([r['feasible_chosen_true_return'] for r in rows]))} | "
                    f"{fmt(mean([r['misrejection_loss'] for r in rows]))} | "
                    f"{fmt(mean([r['ranking_loss'] for r in rows]))} | "
                    f"{fmt(mean([r['execution_regret'] for r in rows]))} | "
                    f"{fmt(mean([r['rejection_rate'] for r in rows]),3)} | "
                    f"{fmt(mean([r['no_feasible_ratio'] for r in rows]),3)} | "
                    f"{fmt(mean([r['no_true_success_ratio'] for r in rows]),3)} |")
        lines.append("")
    lines.append("配对的 128k→256k 差（仅两档都可行的共同行）：")
    lines.append("")
    lines.append("| H | 宽度 | seed | 共同行 | 两侧都可行 | Δ选中真实 J(筛选) | Δ误拒损失 | Δ排序损失 | Δ执行 regret | 有解→无解 | 无解→有解 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for horizon, block in cand.get("by_horizon", {}).items():
        for entry in block["paired"]:
            left = entry["pair"].split("->")[0]
            right = entry["pair"].split("->")[1]
            if not (left.startswith("128000") and right.startswith("256000")):
                continue
            if not left.endswith("best_new_val_total"):
                continue
            lines.append(
                f"| {horizon} | {left.split('/')[1]} | {left.split('/')[2]} | {entry['common_rows']} | "
                f"{entry['feasible_both']} | "
                f"{fmt(entry['delta_feasible_chosen_true_return'])} | {fmt(entry['delta_misrejection_loss'])} | "
                f"{fmt(entry['delta_ranking_loss'])} | {fmt(entry['delta_execution_regret'])} | "
                f"{entry['solved_to_unsolved']} | {entry['unsolved_to_solved']} |")
    lines.append("")
    (out_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    # -- figure: extended learning curve ------------------------------------
    rows = read(data_dir / "learning_curve_rows.json", [])
    extended = []
    for row in rows:
        extended.append((int(row["tier"]), row["width"], row["P"], row["object"], row["invalid"]))
    for tier in ("256000",):
        for width in (256, 512):
            key = f"{tier}/{width}/best_new_val_total"
            if key not in groups:
                continue
            entry = groups[key]["groups"]["all"]

            def weighted(name, fallback):
                value = entry.get("weighted_" + name)
                return entry[fallback] if value is None else value

            extended.append((
                int(tier), width,
                weighted("position_mean", "position_mean"),
                weighted("object_position", "object_position"),
                weighted("invalid_rate", "invalid_rate"),
            ))
    if extended:
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.6))
        for width, colour in ((256, "tab:blue"), (512, "tab:orange")):
            series = sorted([row for row in extended if row[1] == width])
            xs = [row[0] for row in series]
            axes[0].plot(xs, [row[2] for row in series], marker="o", color=colour, label=f"width {width}")
            axes[1].plot(xs, [row[3] for row in series], marker="o", color=colour, label=f"width {width}")
            axes[2].plot(xs, [100 * row[4] for row in series], marker="o", color=colour, label=f"width {width}")
        persistence = mean([row["persistence"] for row in rows]) if rows else None
        if persistence:
            axes[1].axhline(persistence, color="grey", ls="--", lw=1.2, label="persistence, object only")
        axes[0].set_title("new validation P (three-body mean)")
        axes[0].set_ylabel("P (m)")
        axes[1].set_title("object position error vs persistence")
        axes[1].set_ylabel("object error (m)")
        axes[2].set_title("predicted-state invalid rate")
        axes[2].set_ylabel("%")
        for axis in axes:
            axis.set_xscale("log")
            axis.set_xlabel("training windows")
            axis.legend(fontsize=8)
            axis.grid(alpha=0.3)
        figure.suptitle("coverage_scale_v1: the 256k continuation on the frozen new validation set")
        figure.tight_layout()
        figure.savefig(figures / "E_256k_learning_curve.png", dpi=130)
        plt.close(figure)

    # -- figure: grouped body errors ----------------------------------------
    if groups:
        names = [name for name in GROUPS if name != "all"]
        positions = list(range(len(names)))
        figure, axes = plt.subplots(1, 4, figsize=(19, 4.6))
        for axis, body in zip(axes[:3], BODIES):
            for index, tier in enumerate(("128000", "256000")):
                values = []
                for name in names:
                    cells = [groups.get(f"{tier}/{width}/best_new_val_total", {}).get("groups", {}).get(name)
                             for width in (256, 512)]
                    cells = [cell for cell in cells if cell]
                    values.append(mean([cell[body] for cell in cells]) if cells else 0.0)
                axis.bar([p + index * 0.4 for p in positions], values, width=0.4, label=tier)
            persistence_key = dict(zip(BODIES, PERSISTENCE))[body]
            baseline = [mean([groups.get(f"128000/{width}/best_new_val_total", {}).get("groups", {}).get(name, {}).get(persistence_key)
                              for width in (256, 512)]) for name in names]
            axis.plot(positions, baseline, "k_", markersize=22, label="persistence")
            axis.set_xticks([p + 0.2 for p in positions])
            axis.set_xticklabels(names, rotation=25, fontsize=8)
            axis.set_ylabel("position error (m)")
            axis.set_title(body)
            axis.legend(fontsize=8)
            axis.grid(alpha=0.3, axis="y")
        for index, tier in enumerate(("128000", "256000")):
            values = []
            for name in names:
                cells = [groups.get(f"{tier}/{width}/best_new_val_total", {}).get("groups", {}).get(name)
                         for width in (256, 512)]
                cells = [cell for cell in cells if cell]
                values.append(100.0 * mean([cell["invalid_rate"] for cell in cells]) if cells else 0.0)
            axes[3].bar([p + index * 0.4 for p in positions], values, width=0.4, label=tier)
        axes[3].set_xticks([p + 0.2 for p in positions])
        axes[3].set_xticklabels(names, rotation=25, fontsize=8)
        axes[3].set_ylabel("invalid rate (%)")
        axes[3].set_title("predicted-state invalid rate")
        axes[3].legend(fontsize=8)
        axes[3].grid(alpha=0.3, axis="y")
        figure.suptitle("coverage_scale_v1 256k: per-body error by window group (256 and 512 averaged), "
                        "against each body's own no-motion baseline")
        figure.tight_layout()
        figure.savefig(figures / "F_groups.png", dpi=130)
        plt.close(figure)
    print("REPORT_256K_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
