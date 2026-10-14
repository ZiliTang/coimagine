"""Data card and schema for the collected episodes (task book section 9.7)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

COLLECT = ROOT / "outputs/night_repair_optionrl_v1/collection"

#: every field an episode carries, with its role. ``legal`` fields may feed a policy or a world
#: model; ``privileged`` fields are labels and must never be concatenated into an observation.
SCHEMA = {
    "episode_id": ("identifier", "unique per episode"),
    "root_id": ("legal", "which generator root; also the split unit"),
    "split": ("identifier", "dev / test family, assigned per root and never per frame"),
    "scene": ("legal", "the generator's interaction class"),
    "template": ("legal", "the generator's scene template"),
    "active_robots": ("legal", "which robots the scene places at the object"),
    "behaviour_source": ("audit", "policy | mixed | branch"),
    "epsilon": ("audit", "probability of a random legal skill at a boundary"),
    "policy_version": ("audit", "raw or trained"),
    "goal": ("legal", "the public goal region"),
    "goal_distance_m": ("legal", "task definition"),
    "seed": ("audit", "generator seed"),
    "frames": ("legal", "one legal joint observation per 0.05 s control period"),
    "frame_time": ("legal", "the clock on the same grid"),
    "boundaries": ("legal", "skill boundaries: commands, duration, exit reason, controller stage"),
    "exit_reasons": ("legal", "observed skill outcomes"),
    "snapshots": ("audit", "state snapshots for the branch source"),
    "object_pose": ("privileged", "true object position and heading"),
    "object_velocity": ("privileged", "true object velocity"),
    "net_progress_m": ("privileged", "reward-side quantity"),
    "total_reward": ("privileged", "reward"),
    "success": ("privileged", "task outcome"),
    "terminated": ("privileged", "real terminal"),
    "truncated": ("privileged", "buffer or budget cut"),
    "trace": ("privileged", "true state per period, for rendering only"),
}


def scan(directory: Path) -> Dict[str, object]:
    shards = sorted(directory.rglob("episode_*.npz"))
    episodes: List[Dict[str, object]] = []
    frames_total = 0
    for shard in shards:
        data = dict(np.load(shard, allow_pickle=True))
        meta = json.loads(str(data["meta"]))
        episodes.append(meta)
        frames_total += int(data["frames"].shape[0])
    roots = {e["root_id"] for e in episodes}
    reasons: Dict[str, int] = {}
    for episode in episodes:
        for reason in episode["exit_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "shards": len(shards), "episodes": len(episodes), "independent_roots": len(roots),
        "raw_frames": frames_total, "skills": int(sum(e["skills"] for e in episodes)),
        "episodes_with_frames": int(sum(1 for e in episodes if e["raw_frames"] > 0)),
        "success_rate": float(np.mean([e["success"] for e in episodes])) if episodes else None,
        "truncated_rate": float(np.mean([e["truncated"] for e in episodes])) if episodes else None,
        "scene_counts": {scene: sum(1 for e in episodes if e["scene"] == scene)
                         for scene in sorted({e["scene"] for e in episodes})},
        "source_counts": {source: sum(1 for e in episodes if e["behaviour_source"] == source)
                          for source in sorted({e["behaviour_source"] for e in episodes})},
        "exit_reason_histogram": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "mean_skills": float(np.mean([e["skills"] for e in episodes])) if episodes else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(COLLECT))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    summary = scan(Path(args.root))
    card = {"schema_version": "night-repair-optionrl-v1/episode-1",
            "fields": {name: {"role": role, "meaning": meaning}
                       for name, (role, meaning) in SCHEMA.items()},
            "observation_layout": "the `frames` array follows OptionRLEnv's ObservationSpec",
            "privileged_isolation": "privileged fields live in a separate block and in the trace; "
                                    "they are never concatenated with `frames`",
            "summary": summary,
            "loading_example": (
                "import numpy as np, json\n"
                "d = dict(np.load('episode_000000.npz', allow_pickle=True))\n"
                "meta = json.loads(str(d['meta']))\n"
                "legal_frames = d['frames']          # (T, 32) legal observations\n"
                "trace = json.loads(str(d['trace'])) # privileged, rendering only\n")}
    out = Path(args.out) if args.out else Path(args.root) / "DATACARD.json"
    out.write_text(json.dumps(card, indent=1, default=float))
    markdown = ["# DATA_CARD — 试采数据卡", "",
                f"schema：`{card['schema_version']}`", "",
                "## 规模", "",
                f"- shard 数：{summary['shards']}",
                f"- episode 数：{summary['episodes']}",
                f"- 独立 root 数：{summary['independent_roots']}",
                f"- 原始帧数：{summary['raw_frames']}",
                f"- 技能数：{summary['skills']}",
                f"- 平均每 episode 技能数：{summary['mean_skills']:.2f}"
                if summary["mean_skills"] is not None else "- 平均每 episode 技能数：—",
                f"- 成功率：{summary['success_rate']:.4f}"
                if summary["success_rate"] is not None else "- 成功率：—",
                f"- 截断率：{summary['truncated_rate']:.4f}"
                if summary["truncated_rate"] is not None else "- 截断率：—",
                "", "## 来源与场景", "",
                f"- 来源：{summary['source_counts']}",
                f"- 场景：{summary['scene_counts']}",
                "", "## 退出原因", "",
                "| 原因 | 次数 |", "|---|---:|"]
    for reason, count in summary["exit_reason_histogram"].items():
        markdown.append(f"| {reason} | {count} |")
    markdown += ["", "## 字段分区", "", "| 字段 | 角色 | 含义 |", "|---|---|---|"]
    for name, (role, meaning) in SCHEMA.items():
        markdown.append(f"| `{name}` | {role} | {meaning} |")
    markdown += ["", "## 加载示例", "", "```python", card["loading_example"].strip(), "```"]
    (Path(args.root) / "DATA_CARD.md").write_text("\n".join(markdown) + "\n")
    print(json.dumps(summary, indent=1, default=float))
    print("DATACARD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
