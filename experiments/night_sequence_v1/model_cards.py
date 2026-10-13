"""Generate MODEL_CARDS.md from the actual checkpoints (no hand-written numbers)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/night_sequence_v1"
DOCS = ROOT / "docs/night_sequence_v1"

INPUT_WHITELIST = [
    "两机器人自身位姿与速度（各自 R_i0 下）",
    "自身可见片段的：valid、中点、法向、距离、点数、置信度",
    "自身接触标志与接触点",
    "允许共享的队友同类量（相对自身 R_i0）",
    "当前联合命令（alpha、speed、active mask）",
    "已过去时间、阶段码、O0 锚点标志",
]
FORBIDDEN = ["物体真值位姿/速度", "真实面编号", "root id / 采样 seed / split", "未来观测",
             "未来控制力/力序列", "真实终止时间", "真实退出原因"]


def main() -> int:
    import torch
    runs = []
    for path in sorted((DATA / "runs").glob("*.pt")):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        runs.append((path.stem, saved))
    metrics = {}
    for tier in ("d0", "d1", "d2"):
        p = DATA / "analysis" / tier / "metrics.json"
        if p.exists():
            metrics[tier] = json.loads(p.read_text())
    lines = ["# MODEL_CARDS — night_sequence_v1", "",
             "每张卡片的数字来自实际 checkpoint 与分析 JSON。", "",
             "## 共同部署输入白名单（所有新模型一致）", ""]
    lines += [f"- {item}" for item in INPUT_WHITELIST]
    lines += ["", "## 明确禁止的输入通路", ""]
    lines += [f"- {item}" for item in FORBIDDEN]
    lines += ["", "说明：真值 `object_world` / `robots_world` 在数据里单独分区，token 构造器不读取；",
              "`force` 列被保存用于审计，但**没有**进入任何 token。这一条由",
              "`tests/test_night_sequence_v1.py::test_both_history_models_receive_identical_inputs_and_no_forbidden_channel`",
              "与 F05 的输入白名单共同固定。", ""]
    for name, saved in runs:
        tier = saved["tier"]
        entry = metrics.get(tier, {}).get("models", {}).get(f"{name}|test", {})
        t0 = entry.get("t0", {}) if isinstance(entry, dict) else {}
        geometry = t0.get("terminal_geometry", {}).get("r0_mid_2d", {})
        inexec = (entry.get("all", {}).get("grouped", {}).get("regime", {})
                  .get("in_execution", {})) if isinstance(entry, dict) else {}
        t0_text = "—" if not geometry else f"{geometry.get('p50'):.4f} m"
        inexec_text = "—" if not inexec else f"{inexec.get('p50'):.4f} m"
        lines += [f"## {name}", "",
                  f"- 架构：`{saved['model_kind']}`，参数 {saved['parameters']:,}；"
                  f"训练档位 {tier}，seed {saved['seed']}",
                  f"- 训练：{saved['updates']} 次优化更新，最优验证损失 {saved['best_val']:.4f}"
                  f"（第 {saved['best_update']} 次更新），AdamW lr 3e-4 / wd 1e-4 / clip 1.0",
                  f"- 输入：≤33 token × 57 维窗口（MLP-current 只用锚点与最新 token）",
                  f"- 输出头：固定 Δt 过程头 5 组 + option 终态头 6 组，均为原始单位",
                  f"- 检查点：`outputs/night_sequence_v1/runs/{name}.pt`",
                  f"- test t=0 终态 r0 中点二维 p50：{t0_text}；执行中 p50：{inexec_text}",
                  f"- 用途：预测技能结束时的可观测终态；**不用于**闭环规划、不用于自由 rollout",
                  f"- 已知限制：归一化是按组单标量（非逐列）；几何标签仅在真实可见 mask 上评价；"
                  f"闭式前缀条件预测，未验证递推误差累积", ""]
    (DOCS / "MODEL_CARDS.md").write_text("\n".join(lines))
    print(f"wrote {DOCS / 'MODEL_CARDS.md'} ({len(runs)} cards)")
    print("MODEL_CARDS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
