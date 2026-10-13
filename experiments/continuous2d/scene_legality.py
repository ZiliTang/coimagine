"""Closeout A1: per-scene initial legality and the minimal geometric fix record.

    python experiments/continuous2d/scene_legality.py --out outputs/closeout_v1/<run_id>

Writes ``scene_legality.json`` (per-scene clearances, versions, hashes, before/after) and
``scene_legality.md``. Read-only over the physics: it only measures the frozen scenes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from coimagine.continuous2d.legality import DEFAULT_MARGIN, legality_report
from coimagine.continuous2d.scenes import default_scenes

ROOT = Path(__file__).resolve().parents[2]

#: Recorded pre-fix robot poses for the three scenes that were modified in A1. Taken from
#: the v1 definition (base x = 0.22 + object_dx, no y override).
BEFORE = {
    "S0_dev": [(0.22, 0.90), (0.22, 1.30)],
    "S0_v2": [(0.17, 0.83), (0.17, 1.23)],
    "S0_v4": [(0.19, 0.79), (0.19, 1.19)],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, object]] = []
    for scene in default_scenes():
        report = legality_report(scene, DEFAULT_MARGIN)
        record = {
            "scene": scene.name,
            "category": scene.category,
            "scene_version": scene.scene_version,
            "content_hash": report["content_hash"],
            "legal_after": report["legal"],
            "min_clearance_after": report["min_clearance"],
            "robots_after": [[pose.x, pose.y] for pose in scene.robot_poses],
            "violations_after": report["violations"],
        }
        if scene.name in BEFORE:
            record["robots_before"] = BEFORE[scene.name]
            record["displacement"] = [
                round(((after[0] - before[0]) ** 2 + (after[1] - before[1]) ** 2) ** 0.5, 6)
                for before, after in zip(BEFORE[scene.name], record["robots_after"])
            ]
            record["modified"] = True
        else:
            record["modified"] = False
        records.append(record)

    payload = {"margin": DEFAULT_MARGIN, "scene_version_library": "closeout_v2", "scenes": records}
    (out_dir / "scene_legality.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# 场景初态合法性（closeout A1）",
        "",
        f"统一非接触裕量 {DEFAULT_MARGIN} m；边界为半径 0.05 的 capsule，因此机器人中心"
        "到场地边的几何距离必须满足 x ≥ 0.235。逐对间距由几何精确计算，不使用引擎 contacts。",
        "",
        "| 场景 | scene_version | 内容哈希 | 合法 | 最小间距 | 机器人 before → after | 位移 |",
        "|---|---|---|---:|---:|---|---:|",
    ]
    for record in records:
        before = record.get("robots_before", record["robots_after"])
        cells = " / ".join(
            f"({b[0]:.2f},{b[1]:.2f})→({a[0]:.5f},{a[1]:.5f})" for b, a in zip(before, record["robots_after"])
        )
        displacement = ", ".join(f"{value:.5f}" for value in record.get("displacement", [])) or "-"
        lines.append(
            f"| {record['scene']} | {record['scene_version']} | `{record['content_hash']}` | "
            f"{record['legal_after']} | {record['min_clearance_after']:+.5f} | {cells} | {displacement} |"
        )
    lines += [
        "",
        "## 修复规则与范围",
        "",
        "- 仅移动违规机器人的位姿：沿违规约束内法向迭代投影，直到全部非接触对满足裕量；"
        "步长与迭代在 `continuous2d/legality.py:minimal_robot_fix` 中固定，与成功率无关。",
        "- S0_dev / S0_v4 只需 x 方向；S0_v2 的第二台机器人仅靠 x 会触犯物体裕量，"
        "因此同时做了最小 y 位移（记录在上表位移列）。",
        "- 物体、目标、障碍、质量、摩擦、阻尼、时间步、力限幅、成功阈值、控制器与候选模板均未改动。",
        "- 三个被修改场景标记 `scene_version=closeout_v2`；其余七个几何未变，保留 `v1` 与旧结果引用。",
        "",
    ]
    (out_dir / "scene_legality.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
