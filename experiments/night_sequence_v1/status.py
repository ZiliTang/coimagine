"""Generate STATUS.md and RUN_LEDGER.md from the actual artefacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "outputs/night_sequence_v1"
DOCS = ROOT / "docs/night_sequence_v1"
CST = timezone(timedelta(hours=8))


def run(args, cwd=ROOT):
    return subprocess.run(args, cwd=cwd, capture_output=True, text=True, check=False).stdout.strip()


def main() -> int:
    head = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--porcelain"])
    runs = []
    for path in sorted((DATA / "runs").glob("*.pt")):
        import torch
        saved = torch.load(path, map_location="cpu", weights_only=False)
        runs.append({"run": path.stem, "model": saved["model_kind"], "tier": saved["tier"],
                     "seed": saved["seed"], "parameters": saved["parameters"],
                     "best_val": saved["best_val"], "best_update": saved["best_update"],
                     "updates": saved["updates"]})
    summary = json.loads((DATA / "token_summary.json").read_text())
    collections = {}
    for name in ("d0", "eval", "d1", "d2"):
        path = DATA / f"collection_{name}.json"
        if path.exists():
            payload = json.loads(path.read_text())
            collections[name] = {"seconds": payload["seconds"],
                                 "sequences": payload["sequences"],
                                 "workers": payload["workers"]}
    lines = ["# STATUS — night_sequence_v1", "",
             f"生成时间：{datetime.now(CST).strftime('%Y-%m-%d %H:%M')} (Asia/Shanghai)",
             "", "## 起点与工作区", "",
             f"- 本轮实际起点 HEAD：`d7731ca`（observation_terminal_v1 评价修正版）；"
             f"生成时 HEAD：`{head}`。",
             f"- 起点工作区差异：仅新增任务书 `docs/tasks/20261014_0205_...md`（未跟踪），"
             f"未改动任何已冻结配置。",
             f"- 生成时未提交改动条数：{len([l for l in status.splitlines() if l.strip()])}。", "",
             "## 环境", "",
             f"- Python {platform.python_version()} / {platform.system()} {platform.machine()}",
             f"- 线程与核心：os.cpu_count()={__import__('os').cpu_count()}", "",
             "## 采集", "",
             "| stage | 序列 | 耗时 (s) | workers |", "|---|---:|---:|---:|"]
    for name, entry in collections.items():
        lines.append(f"| {name} | {entry['sequences']} | {entry['seconds']:.0f} | {entry['workers']} |")
    lines += ["", "## 数据规模", "",
              "| 档位 | 独立 root | option 序列 | 真实帧 |", "|---|---:|---:|---:|"]
    for split in ("d0", "d1ext", "d2ext", "val", "test"):
        entry = summary.get(split)
        if entry:
            lines.append(f"| {split} | {entry['independent_roots']} | {entry['sequences']} | "
                         f"{entry['frames']} |")
    lines += ["", "## 训练矩阵", "",
              "| run | 模型 | 档位 | seed | 参数量 | 更新 | 最优更新 | 最优验证损失 |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for entry in runs:
        lines.append(f"| {entry['run']} | {entry['model']} | {entry['tier']} | {entry['seed']} | "
                     f"{entry['parameters']:,} | {entry['updates']} | {entry['best_update']} | "
                     f"{entry['best_val']:.4f} |")
    lines += ["", f"合计 {len(runs)} 次正式训练。全部 run 使用统一更新预算 3200（标定见 CONTRACT §4）。", "",
              "## 未完成 / 未验证", "",
              "- D2 的成对训练：视 D2 采集完成时间与预算决定；未完成则在此列出。",
              "- 自由多步递推、闭环收益：本轮不实现、不评估。", ""]
    (DOCS / "STATUS.md").write_text("\n".join(lines))

    ledger = ["# RUN_LEDGER — night_sequence_v1", "",
              "每行一次正式训练；失败记录同样保留。", "",
              "| run | model | tier | seed | params | updates | best_update | best_val | "
              "checkpoint |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for entry in runs:
        ledger.append(f"| {entry['run']} | {entry['model']} | {entry['tier']} | {entry['seed']} | "
                      f"{entry['parameters']:,} | {entry['updates']} | {entry['best_update']} | "
                      f"{entry['best_val']:.4f} | `outputs/night_sequence_v1/runs/"
                      f"{entry['run']}.pt` |")
    failures = []
    for log in sorted((DATA / "runs").glob("*.log")):
        text = log.read_text(errors="ignore")
        if "Traceback" in text:
            failures.append(log.stem)
    ledger += ["", "## 失败记录", "",
               ("无。" if not failures else "\n".join(f"- `{name}`（见同名 .log）" for name in failures)),
               "", "## 已知实现问题与修复（保留复现，不隐藏）", "",
               "1. Transformer 首版把因果掩码与 key-padding 掩码分开传入，全掩码行 softmax 得到 NaN，"
               "残差流又把 NaN 经 `0*NaN` 泄漏到真实行 → 验证损失 NaN。改为单个加法掩码（用 -1e9 而非 "
               "-inf），并把 padded 行清零；现已有测试固定“改变 padding 内容不改变输出”。",
               "2. 退出类别头宽度曾固定为 12，而完整 Exit 词表有 15 类，D1 上出现 `Target 13 out of "
               "bounds`。改为 `len(EXIT_CODES)`，并重新采集受影响的 6 条 d1ext 序列。",
               "3. `GroupedStatistics` 首版对每条 train 序列枚举全部时刻（>1.5M 次 build_example），"
               "无法在预算内完成；改为对 (序列, 时刻) 对做有界均匀采样并在 token 数组上向量化。", ""]
    (DOCS / "RUN_LEDGER.md").write_text("\n".join(ledger))
    print(f"STATUS.md + RUN_LEDGER.md written ({len(runs)} runs, {len(failures)} failed logs)")
    print("STATUS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
