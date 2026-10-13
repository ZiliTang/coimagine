"""Before/after comparison for the protocol fix (S4), read-only over both runs.

    python experiments/continuous2d/compare_baselines.py \
        --before outputs/migration_v1/phaseD-final/continuous2d \
        --after  outputs/protocol_fix_v1/<run_id>/continuous2d \
        --out    outputs/protocol_fix_v1/<run_id>/comparison

This is a **protocol-fix before/after**, not a new method's performance gain. It joins the
two ``results.csv`` files on (condition, scene, seed), reports both tables with complete
denominators, and for every success flip records the first diverging window (chosen skill
and termination) plus the after-run's decision record for that window.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
KEY = ("condition", "scene", "seed")


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def _index(rows: List[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, str]]:
    out = {}
    for row in rows:
        out[(row["condition"], row["scene"], row["seed"])] = row
    return out


def _read_windows(path: Path) -> Dict[Tuple[str, int], Dict[str, object]]:
    out: Dict[Tuple[str, int], Dict[str, object]] = {}
    if not path.exists():
        return out
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            key = (
                f"{record.get('condition', '')}".strip()
                or str(record.get("episode_id", "")),
                int(record["window_index"]),
            )
            out[key] = record
    return out


def _windows_by_episode(path: Path) -> Dict[str, List[Dict[str, object]]]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    if not path.exists():
        return grouped
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            grouped.setdefault(str(record["episode_id"]), []).append(record)
    for records in grouped.values():
        records.sort(key=lambda record: int(record["window_index"]))
    return grouped


def _labels(record: Dict[str, object]) -> List[str]:
    if "joint_skill_labels" in record:
        return [str(value) for value in record["joint_skill_labels"]]  # type: ignore[union-attr]
    return [str(value) for value in record.get("joint_skill_specs", [])]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", default="outputs/migration_v1/phaseD-final/continuous2d")
    parser.add_argument("--after", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scenes", default="S0_dev,S0_v1,S0_v2,S0_v3,S0_v4,S1_dev,S1_v1,S1_v2,S1_v3,S1_v4")
    args = parser.parse_args()

    before_dir, after_dir = ROOT / args.before, ROOT / args.after
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    scenes = [value for value in args.scenes.split(",") if value]

    before_rows = _read_csv(before_dir / "results.csv")
    after_rows = _read_csv(after_dir / "results.csv")
    before = _index(before_rows)
    after = _index(after_rows)
    before_windows = _windows_by_episode(before_dir / "windows.jsonl")
    after_windows = _windows_by_episode(after_dir / "windows.jsonl")
    after_episodes = {
        str(record["episode_id"]): record
        for record in (json.loads(line) for line in (after_dir / "episodes.jsonl").open())
    }

    conditions = ["random", "mpc_depth1", "mpc_depth3"]
    table_path = out_dir / "before_after.csv"
    fieldnames = [
        "condition", "scene", "seed", "before_success", "after_success",
        "before_termination", "after_termination", "before_windows", "after_windows",
        "before_time", "after_time", "before_control_usage", "after_control_usage",
        "before_terminal_distance", "after_terminal_distance",
        "before_plan_nodes", "after_plan_nodes",
        "before_candidate_simulated_steps", "after_candidate_simulated_steps",
        "before_prefix_replay_steps", "after_prefix_replay_steps",
        "before_plan_wall_clock", "after_plan_wall_clock",
        "before_live_physics_steps", "after_live_physics_steps",
        "flip",
    ]
    with table_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        flips: List[Tuple[str, str, str]] = []
        for scene in scenes:
            for condition in conditions:
                seeds = sorted({seed for (cond, sc, seed) in after if cond == condition and sc == scene})
                for seed in seeds:
                    key = (condition, scene, seed)
                    b, a = before.get(key), after.get(key)
                    if a is None:
                        continue
                    flip = ""
                    if b is not None and b["success"] != a["success"]:
                        flip = f"{b['success']}->{a['success']}"
                        flips.append(key)
                    writer.writerow({
                        "condition": condition, "scene": scene, "seed": seed,
                        "before_success": b["success"] if b else "NA",
                        "after_success": a["success"],
                        "before_termination": b["termination"] if b else "NA",
                        "after_termination": a["termination"],
                        "before_windows": b["windows"] if b else "NA",
                        "after_windows": a["windows"],
                        "before_time": b["success_time"] if b else "NA",
                        "after_time": a["success_time"],
                        "before_control_usage": b["control_usage"] if b else "NA",
                        "after_control_usage": a["control_usage"],
                        "before_terminal_distance": b["terminal_distance"] if b else "NA",
                        "after_terminal_distance": a["terminal_distance"],
                        "before_plan_nodes": b["plan_nodes"] if b else "NA",
                        "after_plan_nodes": a["plan_nodes"],
                        "before_candidate_simulated_steps": "NA",
                        "after_candidate_simulated_steps": a["candidate_simulated_steps"],
                        "before_prefix_replay_steps": "NA",
                        "after_prefix_replay_steps": a["prefix_replay_steps"],
                        "before_plan_wall_clock": "NA",
                        "after_plan_wall_clock": a["plan_wall_clock"],
                        "before_live_physics_steps": b.get("physics_steps", "NA") if b else "NA",
                        "after_live_physics_steps": a["live_physics_steps"],
                        "flip": flip,
                    })

    def _before_episode_id(condition: str, scene: str, seed: str) -> str:
        # the old runner used mpc1-/mpc3- prefixes for the depth conditions
        if condition.startswith("mpc_depth"):
            return f"mpc{condition[len('mpc_depth'):]}-{scene}-{seed}"
        return f"{condition}-{scene}-{seed}"

    divergences: List[Dict[str, object]] = []
    for condition, scene, seed in flips:
        episode_id = f"{condition}-{scene}-{seed}"
        before_records = before_windows.get(_before_episode_id(condition, scene, seed), [])
        after_records = after_windows.get(episode_id, [])
        first = None
        for index in range(max(len(before_records), len(after_records))):
            b = before_records[index] if index < len(before_records) else None
            a = after_records[index] if index < len(after_records) else None
            b_labels = _labels(b) if b else []
            a_labels = _labels(a) if a else []
            if b is None or a is None or b_labels != a_labels or b.get("termination") != a.get("termination"):
                first = index
                break
        decision = None
        if first is not None:
            episode = after_episodes.get(episode_id, {})
            decisions = episode.get("decisions", []) if isinstance(episode, dict) else []
            decision = decisions[first] if first < len(decisions) else None
        divergences.append({
            "condition": condition,
            "scene": scene,
            "seed": seed,
            "before_success": before[(condition, scene, seed)]["success"],
            "after_success": after[(condition, scene, seed)]["success"],
            "first_diverging_window": first,
            "before_skills": _labels(before_records[first]) if first is not None and first < len(before_records) else None,
            "before_windows": len(before_records),
            "after_windows": len(after_records),
            "after_skills": _labels(after_records[first]) if first is not None and first < len(after_records) else None,
            "before_window_termination": before_records[first].get("termination") if first is not None and first < len(before_records) else None,
            "after_window_termination": after_records[first].get("termination") if first is not None and first < len(after_records) else None,
            "before_episode_termination": before[(condition, scene, seed)]["termination"] if (condition, scene, seed) in before else None,
            "after_episode_termination": after[(condition, scene, seed)]["termination"],
            "after_decision": decision,
        })
    (out_dir / "first_divergence.jsonl").write_text(
        "\n".join(json.dumps(item) for item in divergences) + ("\n" if divergences else "")
    )

    def _totals(rows: List[Dict[str, str]], condition: str) -> Tuple[int, int]:
        subset = [row for row in rows if row["condition"] == condition]
        return sum(int(row["success"]) for row in subset), len(subset)

    lines = [
        "# 协议修复前后主表（before = 旧协议，after = 修复后）",
        "",
        "这是**协议修复前后对比**，不是新方法的性能增益。分母为完整场景集合。",
        "",
        "| 条件 | before 成功 | after 成功 |",
        "|---|---:|---:|",
    ]
    for condition in conditions:
        b_ok, b_n = _totals(before_rows, condition)
        a_ok, a_n = _totals(after_rows, condition)
        lines.append(f"| {condition} | {b_ok}/{b_n} | {a_ok}/{a_n} |")
    lines += [
        "",
        f"成功翻转 {len(divergences)} 条，逐条首处分歧见 `first_divergence.jsonl`；逐场景逐种子明细见 `before_after.csv`。",
        "",
        "注：旧运行未记录候选模拟步数/精确重放步数（记为 NA）；旧 live physics steps 列为 `physics_steps`。",
    ]
    (out_dir / "before_after.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
