"""Data card generator for the corrected episode schema."""

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

SCHEMA_VERSION = "night-repair2-v1/episode-2"

#: every field an episode carries. ``legal`` fields may feed a policy or a world model;
#: ``privileged`` fields are labels; ``audit`` fields describe how the data was produced.
FIELDS: Dict[str, tuple] = {
    "schema": ("audit", "schema version"),
    "root_id": ("legal", "generator root; also the split unit"),
    "split": ("identifier", "dev / test family, assigned per root"),
    "scene": ("legal", "generator interaction class"),
    "template": ("legal", "generator scene template"),
    "active_robots": ("legal", "robots the scene places at the object"),
    "behaviour_source": ("audit", "policy | mixed | branch"),
    "epsilon": ("audit", "exploration rate of the mixture actually sampled from"),
    "policy_version": ("audit", "trained | untrained"),
    "goal": ("legal", "public goal region"),
    "control_dt": ("legal", "0.05 s"),
    "frames": ("legal", "one legal joint observation per control period, plus the initial frame; "
                        "a branch also carries the parent prefix"),
    "prefix_frames": ("legal", "leading frames inherited from the parent (branch only)"),
    "episode_frames": ("legal", "frames produced by this episode's own skills"),
    "physics_steps": ("legal", "control periods actually advanced"),
    "time_axis_consistent": ("audit", "frames == prefix_frames + physics_steps + 1"),
    "outcome": ("privileged", "success | task_budget | sampling_truncation"),
    "success": ("privileged", "task outcome"),
    "bootstraps": ("audit", "true only for a sampling cut"),
    "skills": ("legal", "skills executed in this episode"),
    "start_distance_m": ("privileged", "object-to-goal distance at the start"),
    "end_distance_m": ("privileged", "object-to-goal distance at the end"),
    "progress_towards_goal_m": ("privileged", "start - end; positive means closer"),
    "distance_change_m": ("privileged", "end - start; positive means farther"),
    "total_reward": ("privileged", "sum of the per-skill training rewards"),
    "boundaries": ("legal", "per-skill record: action, allowed set, probabilities, reward "
                            "components, exit reason"),
    "parent_snapshot_id": ("audit", "branch family identifier"),
    "parent_episode_id": ("audit", "identifier of the prefix trajectory"),
    "parent_prefix_length": ("audit", "frames inherited from the parent"),
    "parent_counters": ("audit", "skills_used / hold / physics_steps carried into the branch"),
    "candidate_action_index": ("audit", "the action the caller forced"),
    "candidate_matches_executed": ("audit", "must be true"),
    "branch_intervention": ("audit", "the first action was forced, not sampled"),
    "parent_prefix_saved": ("audit", "the parent's observation frames are stored in this file"),
}

#: the fields of one ``boundaries`` record, with their role
BOUNDARY_FIELDS = {
    "action_index": "legal", "action": "legal", "active_robots": "legal",
    "allowed_actions": "legal", "policy_logprob": "audit", "behaviour_logprob": "audit",
    "forced_intervention": "audit", "intervention_note": "audit", "value_before": "audit",
    "reward_components": "audit", "skill_reward": "privileged", "physics_steps": "legal",
    "duration_seconds": "legal", "exit_reason": "legal",
}


def scan(directory: Path) -> Dict[str, object]:
    shards = sorted(directory.rglob("episode_*.npz"))
    episodes: List[Dict[str, object]] = []
    for shard in shards:
        data = dict(np.load(shard, allow_pickle=True))
        episodes.append(json.loads(str(data["meta"])))
    reasons: Dict[str, int] = {}
    for episode in episodes:
        for boundary in episode.get("boundaries", []):
            reason = boundary.get("exit_reason", "")
            reasons[reason] = reasons.get(reason, 0) + 1
    outcomes: Dict[str, int] = {}
    for episode in episodes:
        outcomes[episode["outcome"]] = outcomes.get(episode["outcome"], 0) + 1
    branch = [e for e in episodes if e["behaviour_source"] == "branch"]
    return {
        "shards": len(shards), "episodes": len(episodes),
        "independent_roots": len({e["root_id"] for e in episodes}),
        "raw_frames": int(sum(e["frames"] for e in episodes)),
        "physics_steps": int(sum(e["physics_steps"] for e in episodes)),
        "skills": int(sum(e["skills"] for e in episodes)),
        "time_axis_all_consistent": bool(all(e["time_axis_consistent"] for e in episodes)),
        "success_rate": float(np.mean([e["success"] for e in episodes])) if episodes else None,
        "outcomes": outcomes,
        "scene_counts": {s: sum(1 for e in episodes if e["scene"] == s)
                         for s in sorted({e["scene"] for e in episodes})},
        "source_counts": {s: sum(1 for e in episodes if e["behaviour_source"] == s)
                          for s in sorted({e["behaviour_source"] for e in episodes})},
        "exit_reason_histogram": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
        "mean_skills": float(np.mean([e["skills"] for e in episodes])) if episodes else None,
        "branch_prefix_consistent": (bool(all(e["parent_prefix_saved"] for e in branch))
                                     if branch else None),
        "branch_candidate_matches": (float(np.mean([e["candidate_matches_executed"]
                                                    for e in branch])) if branch else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    root = Path(args.root)
    summary = scan(root)
    card = {"schema_version": SCHEMA_VERSION,
            "episode_fields": {name: {"role": role, "meaning": meaning}
                               for name, (role, meaning) in FIELDS.items()},
            "boundary_fields": BOUNDARY_FIELDS,
            "observation_layout": "OptionRLEnv.ObservationSpec; column semantics are identical in "
                                  "the initial frame, the inner frames and the boundary frame",
            "privileged_isolation": "privileged quantities are episode-level scalars only; the "
                                    "frame array contains legal observations exclusively",
            "summary": summary}
    out = Path(args.out) if args.out else root / "DATACARD.json"
    out.write_text(json.dumps(card, indent=1, default=float))
    success = ("%.4f" % summary["success_rate"]) if summary["success_rate"] is not None else "—"
    lines = [
        "# DATA_CARD — 试采数据卡", "", "schema：`%s`" % SCHEMA_VERSION, "", "## 规模", "",
        "- shard 数：%d" % summary["shards"],
        "- episode 数：%d" % summary["episodes"],
        "- 独立 root 数：%d" % summary["independent_roots"],
        "- 原始帧数：%d" % summary["raw_frames"],
        "- 物理控制周期数：%d" % summary["physics_steps"],
        "- 技能数：%d" % summary["skills"],
        "- 时间轴全部一致：**%s**" % summary["time_axis_all_consistent"],
        "- 成功率：%s" % success,
        "- 结局分布：%s" % summary["outcomes"],
        "- 分支候选一致率：%s" % summary["branch_candidate_matches"],
        "- 分支父前缀已保存：%s" % summary["branch_prefix_consistent"],
        "", "## 来源与场景", "",
        "- 来源：%s" % summary["source_counts"],
        "- 场景：%s" % summary["scene_counts"],
        "", "## 退出原因", "", "| 原因 | 次数 |", "|---|---:|",
    ]
    for reason, count in summary["exit_reason_histogram"].items():
        lines.append("| %s | %d |" % (reason or "(空)", count))
    lines += ["", "## episode 字段", "", "| 字段 | 角色 | 含义 |", "|---|---|---|"]
    for name, (role, meaning) in FIELDS.items():
        lines.append("| `%s` | %s | %s |" % (name, role, meaning))
    lines += ["", "## boundary 字段", "", "| 字段 | 角色 |", "|---|---|"]
    for name, role in BOUNDARY_FIELDS.items():
        lines.append("| `%s` | %s |" % (name, role))
    lines += ["", "## 加载示例", "", "```python",
              "import numpy as np, json",
              "d = dict(np.load('episode_000000.npz', allow_pickle=True))",
              "meta = json.loads(str(d['meta']))",
              "frames = d['frames']       # (T, 32) legal observations",
              "times = d['frame_time']    # T * 0.05, the physical grid",
              "assert meta['time_axis_consistent']",
              "```", ""]
    (root / "DATA_CARD.md").write_text("\n".join(lines))
    print(json.dumps(summary, indent=1, default=float))
    print("DATACARD_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
