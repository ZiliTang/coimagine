"""B: the before/after and between-arm comparison on the same roots.

    python experiments/local_contact_option/make_comparison.py

Reads the frozen v1 executions (the old run) and the repaired v2 executions (the same 24 dev
and 120 test roots, same 8 variants, same 4 arms) and writes ``comparison.{md,json}``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.local_contact_option.metrics import quantile, summarise  # noqa: E402

V1 = ROOT / "outputs/local_contact_option_v1/local-contact-option-v1"
ARMS = ("O", "P", "F", "N")
PRIMARY = ("solo_central", "dual_same_side")
SECONDARY = ("solo_central", "solo_offset", "dual_same_side", "dual_rotate")


def load(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.open()]


def key(row) -> tuple:
    return (row["root_id"], row["variant"], row["arm"])


def _fmt(value, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def arm_summary(rows, classes=PRIMARY) -> Dict[str, object]:
    out = {}
    for arm in ARMS:
        selected = [row for row in rows if row["arm"] == arm and row["variant"] == "baseline"
                    and row["interaction_class"] in classes]
        if not selected:
            continue
        out[arm] = {
            "n": len(selected),
            "exit_reasons": _counts(row["metrics"]["exit_reason"] for row in selected),
            "effective_push_ratio": _ratio(selected, lambda row: row["metrics"]["reached_effective_push"]),
            "settled_observed_ratio": _ratio(selected, lambda row: row["metrics"]["settled_observed"]),
            "effective_push_seconds": summarise([row["metrics"].get("active_contact_time") for row in selected]),
            "wall_clock_seconds": summarise([row["metrics"]["wall_clock_seconds"] for row in selected]),
            "displacement": summarise([row["metrics"]["displacement"] for row in selected]),
            "rotation_rad": summarise([row["metrics"]["rotation"] for row in selected]),
            "final_speed": summarise([row["metrics"]["final_speed"] for row in selected]),
            "force_saturation_ratio": summarise([row["metrics"]["force_saturation_ratio"] for row in selected]),
            "stage_motion": _stage_motion(selected),
        }
    return out


def _stage_motion(rows) -> Dict[str, object]:
    buckets: Dict[str, List[float]] = defaultdict(list)
    for row in rows:
        for stage, entry in (row["metrics"].get("stage_motion") or {}).items():
            buckets[stage].append(entry["linear"])
    return {stage: summarise(values) for stage, values in sorted(buckets.items())}


def _counts(values) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for value in values:
        out[str(value)] = out.get(str(value), 0) + 1
    return out


def _ratio(rows, predicate) -> Optional[float]:
    rows = list(rows)
    return (sum(1 for row in rows if predicate(row)) / len(rows)) if rows else None


def before_after(v1, v2) -> Dict[str, object]:
    """Paired by (root, variant, arm): the same executions before and after the repair."""

    index_v1 = {key(row): row for row in v1}
    index_v2 = {key(row): row for row in v2}
    shared = sorted(set(index_v1) & set(index_v2))
    out: Dict[str, object] = {"paired_rows": len(shared), "per_arm": {}}
    for arm in ARMS:
        rows = [(index_v1[k], index_v2[k]) for k in shared if k[2] == arm]
        if not rows:
            continue
        entry = {"n": len(rows)}
        for name, getter in (
            ("effective_push_ratio", lambda m: m.get("reached_effective_push")),
            ("settled_observed_ratio", lambda m: m["settled_observed"]),
            ("false_stable_ratio", lambda m: m["false_stable"]),
            ("active_contact_seconds", lambda m: m.get("active_contact_time")),
            ("wall_clock_seconds", lambda m: m["wall_clock_seconds"]),
            ("final_speed", lambda m: m["final_speed"]),
            ("displacement", lambda m: m["displacement"]),
        ):
            before = [getter(a["metrics"]) for a, _ in rows]
            after = [getter(b["metrics"]) for _, b in rows]
            if name.endswith("ratio"):
                entry[name] = {"before": _ratio(before, bool), "after": _ratio(after, bool)}
            else:
                entry[name] = {"before": summarise(before), "after": summarise(after)}
        entry["exit_changes"] = _counts(
            f"{a['metrics']['exit_reason']}->{b['metrics']['exit_reason']}" for a, b in rows
        )
        out["per_arm"][arm] = entry
    return out


def speed_subset(rows, speed: float = 0.26) -> Dict[str, object]:
    out = {}
    for arm in ("P", "N"):
        selected = [row for row in rows if row["arm"] == arm and row["variant"] == "baseline"
                    and row["interaction_class"] in PRIMARY and row["speed"][0] == speed]
        out[arm] = {
            "n": len(selected),
            "displacement": summarise([row["metrics"]["displacement"] for row in selected]),
            "achieved_speed": summarise([row["metrics"]["displacement"] / max(1e-9, row["metrics"].get("active_contact_time") or 1e-9)
                                         for row in selected]),
            "force_saturation_ratio": summarise([row["metrics"]["force_saturation_ratio"] for row in selected]),
            "pos_10mm_p90": _variant_p90(selected, rows, arm, "pos_10mm"),
            "angle_10mm_p90": _variant_angle_p90(selected, rows, arm, "pos_10mm"),
            "subset_roots": len(selected),
        }
    return out


def _variant_p90(selected, all_rows, arm, variant) -> Optional[float]:
    roots = {row["root_id"] for row in selected}
    base = {row["root_id"]: row["metrics"]["relative_object"] for row in all_rows
            if row["arm"] == arm and row["variant"] == "baseline"}
    values = []
    for row in all_rows:
        if row["arm"] != arm or row["variant"] != variant or row["root_id"] not in roots:
            continue
        origin = base.get(row["root_id"])
        if origin is None:
            continue
        now = row["metrics"]["relative_object"]
        values.append(math.hypot(now[0] - origin[0], now[1] - origin[1]))
    return quantile(values, 0.9)


def _variant_angle_p90(selected, all_rows, arm, variant) -> Optional[float]:
    roots = {row["root_id"] for row in selected}
    base = {row["root_id"]: row["metrics"]["relative_object"] for row in all_rows
            if row["arm"] == arm and row["variant"] == "baseline"}
    values = []
    for row in all_rows:
        if row["arm"] != arm or row["variant"] != variant or row["root_id"] not in roots:
            continue
        origin = base.get(row["root_id"])
        if origin is None:
            continue
        now = row["metrics"]["relative_object"]
        values.append(abs((now[2] - origin[2] + math.pi) % (2 * math.pi) - math.pi))
    return quantile(values, 0.9)


def bins(rows) -> Dict[str, object]:
    out = {}
    edges = ((0.0, 0.02), (0.02, 0.10), (0.10, float("inf")))
    for arm in ("P", "N"):
        arm_rows = [row for row in rows if row["arm"] == arm]
        base = {row["root_id"]: row["metrics"]["relative_object"] for row in arm_rows
                if row["variant"] == "baseline"}
        table = {}
        for low, high in edges:
            values = []
            roots = set()
            for row in arm_rows:
                if row["variant"] != "pos_10mm":
                    continue
                origin = base.get(row["root_id"])
                if origin is None:
                    continue
                baseline_row = next((r for r in arm_rows
                                     if r["root_id"] == row["root_id"] and r["variant"] == "baseline"), None)
                if baseline_row is None:
                    continue
                if not (low <= baseline_row["metrics"]["displacement"] < high):
                    continue
                now = row["metrics"]["relative_object"]
                values.append(math.hypot(now[0] - origin[0], now[1] - origin[1]))
                roots.add(row["root_id"])
            table[f"{low:.2f}-{high:.2f}" if high != float("inf") else f"{low:.2f}+"] = {
                "roots": len(roots), "summary": summarise(values),
                "root_ids": sorted(roots)[:6],
            }
        out[arm] = table
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-contact-option-repair-v2")
    parser.add_argument("--out-root", default="outputs/local_contact_option_repair_v2")
    parser.add_argument("--v1", default=str(V1.relative_to(ROOT)))
    args = parser.parse_args()
    v1_dir = ROOT / args.v1
    v2_dir = ROOT / args.out_root / args.run_id
    v1 = load(v1_dir / "executions_test.jsonl")
    v2 = load(v2_dir / "executions_test.jsonl")

    payload = {
        "v1_executions": len(v1), "v2_executions": len(v2),
        "before_after": before_after(v1, v2),
        "v1_arms_primary": arm_summary(v1, PRIMARY),
        "v2_arms_primary": arm_summary(v2, PRIMARY),
        "v1_arms_secondary": arm_summary(v1, SECONDARY),
        "v2_arms_secondary": arm_summary(v2, SECONDARY),
        "v2_speed_0.26": speed_subset(v2),
        "v1_speed_0.26": speed_subset(v1),
        "v2_bins": bins(v2),
        "stage_motion": {arm: arm_summary(v2, PRIMARY).get(arm, {}).get("stage_motion")
                         for arm in ARMS},
        "one_shot_totals": {
            "v1": {"O": _counts(r["metrics"]["exit_reason"] for r in v1 if r["arm"] == "O"),
                   "P": _counts(r["metrics"]["exit_reason"] for r in v1 if r["arm"] == "P"),
                   "F": _counts(r["metrics"]["exit_reason"] for r in v1 if r["arm"] == "F"),
                   "N": _counts(r["metrics"]["exit_reason"] for r in v1 if r["arm"] == "N")},
            "v2": {"O": _counts(r["metrics"]["exit_reason"] for r in v2 if r["arm"] == "O"),
                   "P": _counts(r["metrics"]["exit_reason"] for r in v2 if r["arm"] == "P"),
                   "F": _counts(r["metrics"]["exit_reason"] for r in v2 if r["arm"] == "F"),
                   "N": _counts(r["metrics"]["exit_reason"] for r in v2 if r["arm"] == "N")},
        },
    }
    (v2_dir / "comparison.json").write_text(json.dumps(payload, indent=1, sort_keys=True))

    lines = ["# local_contact_option_repair_v2 — 同批复核对照", "",
             f"同一批 root/扰动/动作：v1 {len(v1)} 条、v2 {len(v2)} 条测试执行。", "",
             "## 1. 修复前后（按 (root, variant, arm) 配对）", "",
             "| 臂 | 配对行 | 有效推动2s 前→后 | 观测判稳 前→后 | 假稳定 前→后 | 有效接触秒中位 前→后 | 墙钟中位 前→后 | 终态速度中位 前→后 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, entry in sorted(payload["before_after"]["per_arm"].items()):
        lines.append(
            f"| {arm} | {entry['n']} | "
            f"{_fmt(entry['effective_push_ratio']['before'])}→{_fmt(entry['effective_push_ratio']['after'])} | "
            f"{_fmt(entry['settled_observed_ratio']['before'])}→{_fmt(entry['settled_observed_ratio']['after'])} | "
            f"{_fmt(entry['false_stable_ratio']['before'])}→{_fmt(entry['false_stable_ratio']['after'])} | "
            f"{_fmt(entry['active_contact_seconds']['before']['median'],2)}→{_fmt(entry['active_contact_seconds']['after']['median'],2)} | "
            f"{_fmt(entry['wall_clock_seconds']['before']['median'],2)}→{_fmt(entry['wall_clock_seconds']['after']['median'],2)} | "
            f"{_fmt(entry['final_speed']['before']['median'],4)}→{_fmt(entry['final_speed']['after']['median'],4)} |")
    lines += ["", "退出原因迁移（前→后，仅列出现过的）：", ""]
    for arm, entry in sorted(payload["before_after"]["per_arm"].items()):
        top = sorted(entry["exit_changes"].items(), key=lambda item: -item[1])[:5]
        lines.append(f"- **{arm}**: " + ", ".join(f"{k}×{v}" for k, v in top))
    lines += ["", "## 2. 修复后四臂（主要分母：solo_central + dual_same_side，基准变体 40 root）", "",
              "| 臂 | 有效推动2s | 观测判稳 | 有效接触秒中位 | 墙钟中位 | 位移中位 | 转角中位(rad) | 终态速度中位 | 力饱和均值 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for arm, entry in sorted(payload["v2_arms_primary"].items()):
        lines.append(
            f"| {arm} | {_fmt(entry['effective_push_ratio'])} | {_fmt(entry['settled_observed_ratio'])} | "
            f"{_fmt(entry['effective_push_seconds']['median'],2)} | {_fmt(entry['wall_clock_seconds']['median'],2)} | "
            f"{_fmt(entry['displacement']['median'],3)} | {_fmt(entry['rotation_rad']['median'],3)} | "
            f"{_fmt(entry['final_speed']['median'],4)} | {_fmt(entry['force_saturation_ratio']['mean'],4)} |")
    lines += ["", "同样四臂，次要分母（四个可推动类，80 root）：", "",
              "| 臂 | 有效推动2s | 观测判稳 | 位移中位 |", "|---|---:|---:|---:|"]
    for arm, entry in sorted(payload["v2_arms_secondary"].items()):
        lines.append(f"| {arm} | {_fmt(entry['effective_push_ratio'])} | "
                     f"{_fmt(entry['settled_observed_ratio'])} | {_fmt(entry['displacement']['median'],3)} |")
    lines += ["", "## 3. 相同 0.26 m/s 命令子集（主要分母）", "",
              "| 版本 | 臂 | n | 位移中位 | 平均实际速度 | 力饱和均值 | pos10mm P90 | angle10mm P90(rad) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for version in ("v1_speed_0.26", "v2_speed_0.26"):
        for arm, entry in sorted(payload[version].items()):
            lines.append(f"| {version[:2]} | {arm} | {entry['n']} | {_fmt(entry['displacement']['median'],3)} | "
                         f"{_fmt(entry['achieved_speed']['median'],3)} | {_fmt(entry['force_saturation_ratio']['mean'],4)} | "
                         f"{_fmt(entry['pos_10mm_p90'],4)} | {_fmt(entry['angle_10mm_p90'],4)} |")
    lines += ["", "旧控制没有 0.08 / 0.16 档，所以混合速度的总体不是公平的速度比较；此表只用 0.26 m/s。", "",
              "## 4. 分阶段位移（主要分母，基准变体，米）", "",
              "| 臂 | APPROACH | READY | PUSH | BRAKE | SETTLE |", "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        motion = (payload["v2_arms_primary"].get(arm) or {}).get("stage_motion") or {}
        row = [ _fmt((motion.get(stage) or {}).get("median"), 4)
                for stage in ("approach", "ready", "push", "brake", "settle", "old_window") ]
        lines.append(f"| {arm} | " + " | ".join(row[:5]) + " |")
    lines += ["", "## 5. 位移分箱（pos_10mm 相对基准的终态位置差 P90）", "",
              "| 臂 | 0-2cm | 2-10cm | >10cm |", "|---|---:|---:|---:|"]
    for arm, table in sorted(payload["v2_bins"].items()):
        lines.append(f"| {arm} | " + " | ".join(
            f"{_fmt(table[key]['summary']['p90'],4)} (n={table[key]['roots']})" for key in
            ("0.00-0.02", "0.02-0.10", "0.10+")) + " |")
    lines += ["", "分箱后各箱的 root 构成不同，属于分层描述，不是严格匹配的比较。", ""]
    (v2_dir / "comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("COMPARISON_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
