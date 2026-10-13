"""Phase C6: coverage report for Old / RandomAdd / BranchAdd.

    python experiments/data_upgrade/run_coverage.py --run-id <id>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from coimagine.data_upgrade.coverage import anchor_progress_differences, dataset_coverage

ROOT = Path(__file__).resolve().parents[2]


def _load(path: Path):
    return [json.loads(line) for line in path.open()] if path.exists() else []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="data-upgrade-20261012")
    parser.add_argument("--out-root", default="outputs/data_upgrade_v1")
    parser.add_argument("--source-run", default="outputs/forecast_baseline_v1/forecast-20261012")
    args = parser.parse_args()

    run_dir = ROOT / args.out_root / args.run_id
    branch_dir = run_dir / "branch_data"
    out_dir = run_dir / "coverage"
    out_dir.mkdir(parents=True, exist_ok=True)
    source = ROOT / args.source_run

    original = _load(source / "dataset" / "windows.jsonl")
    original_train = [r for r in original if r["split"] == "train"]
    branch_all = _load(branch_dir / "branch_windows.jsonl")
    random_matched = _load(branch_dir / "random_train_matched.jsonl")
    branch_matched = _load(branch_dir / "branch_train_matched.jsonl")

    report = {
        "Old(train)": dataset_coverage(original_train, "Old(train)"),
        "RandomAdd(matched)": dataset_coverage(random_matched, "RandomAdd(matched)"),
        "BranchAdd(matched)": dataset_coverage(branch_matched, "BranchAdd(matched)"),
        "BranchAdd(all splits)": dataset_coverage(branch_all, "BranchAdd(all splits)"),
        "anchor_progress": anchor_progress_differences(branch_all),
    }
    (out_dir / "coverage.json").write_text(json.dumps(report, indent=2, sort_keys=True, default=str))
    lines = ["# 数据覆盖（Old / RandomAdd / BranchAdd）", "",
             "| 数据集 | 场景 | 回合 | 锚点 | 不同状态 | 不同 state-action | 窗口 | 平移 | 转动 | 静止 | 接触 | 成功窗口 | 近成功窗口 | 物理失败 | 成功覆盖 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for label, row in report.items():
        if label == "anchor_progress":
            continue
        lines.append(
            f"| {label} | {row['n_scenes']} | {row['episodes']} | {row['anchors']} | {row['distinct_states']} | "
            f"{row['distinct_state_actions']} | {row['windows']} | {row['object_translation']} | {row['object_rotation']} | "
            f"{row['stationary']} | {sum(row['contact_types'].values())} | {row['success_windows']} | "
            f"{row['near_success_windows']} | {row['failure_windows']} | {row['success_coverage']} |"
        )
    lines += ["", "技能组合：", "",
              "| 数据集 | " + " | ".join(sorted({k for row in report.values() if isinstance(row, dict) for k in (row.get("combinations") or {})})) + " |",
              "|---|" + "---:|" * len({k for row in report.values() if isinstance(row, dict) for k in (row.get("combinations") or {})})]
    keys = sorted({k for row in report.values() if isinstance(row, dict) for k in (row.get("combinations") or {})})
    for label, row in report.items():
        if label == "anchor_progress":
            continue
        lines.append("| " + label + " | " + " | ".join(str((row["combinations"] or {}).get(k, 0)) for k in keys) + " |")
    lines += ["", f"同锚点不同动作的真实进展 spread：锚点数 {report['anchor_progress']['anchors']}，"
                  f"平均 spread {report['anchor_progress']['mean_progress_spread']}，"
                  f"最大 spread {report['anchor_progress']['max_progress_spread']}（passage_cost_to_go 的下降量，米）。"]
    (out_dir / "coverage_table.md").write_text("\n".join(lines) + "\n")
    for label, row in report.items():
        if label == "anchor_progress":
            continue
        print(f"{label}: windows={row['windows']} anchors={row['anchors']} success={row['success_windows']} "
              f"near_success={row['near_success_windows']} failure={row['failure_windows']} {row['success_coverage']}")
    print("COVERAGE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
