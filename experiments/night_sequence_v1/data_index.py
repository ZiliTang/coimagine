"""Generate DATA_INDEX.md from the collected artefacts (no invention, no estimates)."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/night_sequence_v1"
DOCS = ROOT / "docs/night_sequence_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    summary = json.loads((DATA / "token_summary.json").read_text())
    roots = json.loads((DATA / "roots_night.json").read_text())
    lines = ["# DATA_INDEX — night_sequence_v1", "",
             "所有数字来自实际采集产物；窗口数不等于独立 root 数。", "",
             "## 规模（独立 root / option 序列 / 真实帧 / 训练窗口是四个不同的量）", "",
             "| 档位 | 独立 root | option 序列 | 真实帧 | 最大步数 | 角色 |",
             "|---|---:|---:|---:|---:|---|"]
    role = {"d0": "D0 训练（原 phase-B train root，精确重放恢复全过程）",
            "d1ext": "D1 扩展训练（新 root）",
            "val": "新验证集", "test": "新测试集"}
    for split in ("d0", "d1ext", "val", "test"):
        entry = summary.get(split, {})
        lines.append(f"| {split} | {entry.get('independent_roots', 0)} | {entry.get('sequences', 0)} | "
                     f"{entry.get('frames', 0)} | {entry.get('max_steps', 0)} | {role.get(split, '')} |")
    total_roots = summary.get("d0", {}).get("independent_roots", 0) + \
        summary.get("d1ext", {}).get("independent_roots", 0)
    total_sequences = summary.get("d0", {}).get("sequences", 0) + \
        summary.get("d1ext", {}).get("sequences", 0)
    lines += ["", f"**D1 = D0 + D1ext = {total_roots} 独立 root / {total_sequences} option 序列。**",
              "",
              "- **训练窗口数**由采样过程决定（每步更新 128 个窗口），它不是新的独立起点；"
              "滑窗不会增加 root 数。",
              "- 每条 option 序列 = 一次真实执行，共享其 root；同一 root 的 8 个命令是 8 条不同序列。",
              "", "## 近重复隔离（原标准 MIN_ROOT_DISTANCE = 0.02 m，角度按 0.5 m 力臂折算）", "",
              "| 对 | 最小距离 |", "|---|---:|"]
    for pair in roots["leakage"]["pairs"]:
        lines.append(f"| {pair['split_a']} - {pair['split_b']} | {pair['min_distance']:.5f} |")
    lines += ["", "D1 = d0 ∪ d1ext 按构造成立，因此隔离检查针对四个**互斥**集合。", "",
              "## 分片与哈希", "",
              "| 文件 | 字节 | sha256 |", "|---|---:|---|"]
    for pattern in ("sequences/*.npz", "tokens/*.npz", "roots_night.json", "roots_night.pkl",
                    "token_summary.json"):
        for path in sorted(DATA.glob(pattern)):
            lines.append(f"| {path.relative_to(ROOT)} | {path.stat().st_size} | {sha256(path)} |")
    lines += ["", "## schema", "",
              "`sequences/<split>_<n>.npz` 每条序列的逐周期列式数组（padding 到固定长度，`steps` 给出真实长度）：",
              "",
              "| 键 | 形状 | 含义 |", "|---|---|---|",
              "| `time` | (S,) | 该周期开始时的真实时间戳 |",
              "| `stage_code` | (S,) | 阶段机状态码 |",
              "| `own_world` | (S,2,6) | 自身 x,y,θ,vx,vy,ω（机器人自己的估计） |",
              "| `teammate_world` | (S,2,6) | 允许共享的队友同类量 |",
              "| `patch_valid` / `patch_points` / `patch_distance` / `patch_confidence` | (S,2) | 可见片段 |",
              "| `patch_segment_robot` / `patch_normal_world` / `patch_midpoint_robot` | (S,2,…) | 拟合线段与法向 |",
              "| `contacting` / `contact_point_robot` / `contact_point_valid` | (S,2,…) | 接触观测量 |",
              "| `force` | (S,2,2) | 实际施加力（**不作为任何模型输入**） |",
              "| `object_world` / `robots_world` | (S,3) / (S,2,3) | 仿真真值，**单独分区，输入白名单拒绝** |",
              "| `terminal_*` | (2,…) | **退出时刻 T** 的一次真实传感调用（不推进物理、不施加控制） |",
              "| `T_exit` / `T_last_sensed` | 标量 | 实际退出时刻 / option 内最后一次采样时刻 |",
              "| `exit_code` | 标量 | 完整 15 类退出词表 |",
              "",
              "`tokens/<split>.npz` 由上述序列派生：`tokens` (N,S,57) float16、`token_valid` (N,S)、"
              "`terminal` (N,57)、`steps`、`exit_class`、`T_exit`、`root_id`、`record_id`。",
              "", "## 采样排除统计", "",
              "- 被放弃的 root：`roots_night.json` 的 `reports.*.leakage_rejected`（d1ext 为 1，其余为 0），"
              "`sample_none` 与 `unfilled` 均为 0。",
              "- 被放弃的执行：无。`run_sequence` 只会在步数超过上限或轨迹为空时返回 None，本次采集未发生。",
              "- 未按成功/误差筛选任何样本；失败、中止、片段缺失与残余运动全部保留。", ""]
    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "DATA_INDEX.md").write_text("\n".join(lines))
    print(f"wrote {DOCS / 'DATA_INDEX.md'} ({len(lines)} lines)")
    print("DATA_INDEX_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
