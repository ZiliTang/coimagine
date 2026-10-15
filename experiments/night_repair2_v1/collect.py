"""Corrected episode collection.

Fixes against the previous collector:

* the **same** ``rollout`` that training uses, so the stored observation sequence is the one the
  policy actually consumed;
* the time axis is the physical grid ``i * control_dt`` and the number of frames is
  ``physics_steps + 1``; the previous version appended the boundary observation as an extra frame,
  which added one control period per skill (a 34.25 s episode was stored as 34.85 s);
* the boundary observation is the last inner frame, so an inner frame and a boundary frame now have
  the same field meanings -- previously column 28 was the sensed distance at a boundary and the
  skill-local clock inside a skill;
* branch episodes restore a real **mid-trajectory** snapshot after the reset and execute the
  **specified** candidate action; the previous version reset over the snapshot and re-sampled a
  different action (12 of 12 sampled shards disagreed with their own label).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.night_repair_optionrl_v1.option_env import (  # noqa: E402
    MENU,
    OptionRLConfig,
    OptionRLEnv,
    ObservationSpec,
    default_goal,
)
from experiments.night_repair2_v1.rl import (  # noqa: E402
    Episode,
    RecurrentPolicy,
    legal_actions,
    rollout,
)
from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.roots import RootSpec, build_roots  # noqa: E402

OUT = ROOT / "outputs/night_repair2_v1/collection"
SCHEMA_VERSION = "night-repair2-v1/episode-2"
CONTROL_DT = 0.05


def load_policy(path: Path) -> RecurrentPolicy:
    saved = torch.load(path, map_location="cpu", weights_only=False)
    policy = RecurrentPolicy(saved["observation_width"], saved["action_count"],
                             hidden=saved["hidden"])
    policy.load_state_dict(saved["state_dict"])
    policy.eval()
    return policy


def episode_record(episode: Episode, env: OptionRLEnv, *, source: str, epsilon: float,
                   policy_version: str, root: RootSpec, parent_snapshot_id: Optional[str] = None,
                   candidate: Optional[int] = None) -> Dict[str, object]:
    """The stored payload. Privileged quantities live in their own block, never in ``frames``."""

    boundaries = []
    for index, decision in enumerate(episode.decisions):
        boundaries.append({
            "option_id": index,
            "buffer_index": decision.buffer_index,
            "action_index": int(decision.action),
            "action": MENU[decision.action].label(),
            "active_robots": list(MENU[decision.action].active),
            "alpha_deg": [round(np.degrees(a), 4) for a in MENU[decision.action].alpha],
            "speed": [float(v) for v in MENU[decision.action].speed],
            "allowed_actions": list(decision.allowed),
            "policy_logprob": float(decision.policy_logprob),
            "behaviour_logprob": float(decision.behaviour_logprob),
            "value_before": float(decision.value),
            "skill_reward": float(decision.reward),
            "cross_skill_discount": float(decision.discount),
            "physics_steps": int(decision.skill_steps),
            "duration_seconds": float(decision.skill_steps) * CONTROL_DT,
            "exit_reason": episode.exit_reasons[index] if index < len(episode.exit_reasons) else "",
        })
    return {
        "schema": SCHEMA_VERSION,
        "root_id": str(root.root_id), "split": str(root.split),
        "scene": str(root.interaction_class), "template": str(root.template),
        "active_robots": list(root.active_robots),
        "behaviour_source": source, "epsilon": float(epsilon),
        "policy_version": policy_version,
        "goal": [float(episode.goal[0]), float(episode.goal[1])],
        "control_dt": CONTROL_DT,
        "frames": int(len(episode.observations)),
        "physics_steps": int(episode.physics_steps),
        "time_axis_consistent": bool(len(episode.observations) == episode.physics_steps + 1),
        "outcome": episode.outcome, "success": bool(episode.success),
        "bootstraps": bool(episode.outcome in ("sampling_truncation",)),
        "skills": len(episode.decisions),
        "start_distance_m": float(episode.start_distance),
        "end_distance_m": float(episode.end_distance),
        "progress_towards_goal_m": float(episode.net_progress_towards_goal),
        "distance_change_m": float(episode.net_distance_change),
        "total_reward": float(episode.total_reward),
        "boundaries": boundaries,
        "parent_snapshot_id": parent_snapshot_id,
        "candidate_action_index": candidate,
        "candidate_matches_executed": (None if candidate is None
                                       else int(candidate) == int(episode.decisions[0].action)),
    }


def write_shard(directory: Path, payload: Dict[str, object], frames: np.ndarray,
                index: int) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{index:06d}.npz"
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, frames=frames,
                        frame_time=(np.arange(frames.shape[0]) * CONTROL_DT),
                        meta=np.asarray(json.dumps(payload)))
    temporary.replace(path)
    return path


def trial(*, policy_path: Optional[Path] = None, episodes_per_setting: int = 20,
          epsilons: Sequence[float] = (0.0, 0.1, 0.3), branch_snapshots: int = 20,
          branch_candidates: int = 4, seed: int = 0, goal_distance: float = 0.45,
          max_skills: int = 12, out_dir: Path = OUT,
          roots: Optional[Sequence[RootSpec]] = None) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    policy = load_policy(policy_path) if policy_path and Path(policy_path).exists() else None
    pool = list(roots) if roots is not None else build_roots("dev", 6)
    summary: Dict[str, object] = {"schema": SCHEMA_VERSION, "policy": str(policy_path),
                                  "policy_loaded": policy is not None, "settings": {}}
    index = 0
    all_records: List[Dict[str, object]] = []
    for epsilon in epsilons:
        source = "policy" if epsilon == 0.0 else "mixed"
        directory = Path(out_dir) / f"{source}_eps{epsilon:g}"
        records = []
        for _ in range(episodes_per_setting):
            root = pool[int(rng.integers(0, len(pool)))]
            env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30)),
                                             max_skills=max_skills))
            episode = rollout(env, policy, root, default_goal(root, goal_distance), rng=rng,
                              epsilon=epsilon, max_skills=max_skills)
            payload = episode_record(episode, env, source=source, epsilon=epsilon,
                                     policy_version="trained" if policy else "untrained",
                                     root=root)
            write_shard(directory, payload, episode.observations, index)
            index += 1
            records.append(payload)
        summary["settings"][f"{source}_eps{epsilon:g}"] = _summarise(records)
        all_records.extend(records)
    # ---- branch: restore a real mid-trajectory snapshot and execute the chosen candidate ----
    directory = Path(out_dir) / "branch"
    branch_records = []
    for family in range(branch_snapshots):
        root = pool[int(rng.integers(0, len(pool)))]
        goal = default_goal(root, goal_distance)
        env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30)), max_skills=max_skills))
        warm = rollout(env, policy, root, goal, rng=rng,
                       epsilon=0.0 if policy else 1.0, max_skills=max_skills)
        depth = int(rng.integers(1, max(2, len(warm.decisions))))
        # rebuild the world up to ``depth`` skills and snapshot there
        probe = OptionRLEnv(OptionRLConfig(seed=env.config.seed, max_skills=max_skills))
        prefix = rollout(probe, policy, root, goal, rng=np.random.default_rng(seed + family),
                         epsilon=0.0 if policy else 1.0, max_skills=depth)
        del prefix
        snapshot = probe.env.export_state()
        parent_id = f"branch_family_{family:03d}"
        allowed = legal_actions(root)
        count = min(branch_candidates, len(allowed))
        picks = [int(allowed[i]) for i in rng.choice(len(allowed), size=count, replace=False)]
        for candidate in picks:
            branch_env = OptionRLEnv(OptionRLConfig(seed=env.config.seed, max_skills=max_skills))
            episode = rollout(branch_env, policy, root, goal, rng=rng, snapshot=snapshot,
                              forced_first_action=candidate, max_skills=1)
            payload = episode_record(episode, branch_env, source="branch", epsilon=1.0,
                                     policy_version="trained" if policy else "untrained",
                                     root=root, parent_snapshot_id=parent_id, candidate=candidate)
            payload["branch_depth"] = depth
            write_shard(directory, payload, episode.observations, index)
            index += 1
            branch_records.append(payload)
        all_records.extend(branch_records[-count:])
    summary["settings"]["branch"] = _summarise(branch_records)
    summary["branch_families"] = branch_snapshots
    summary["episodes_total"] = len(all_records)
    summary["raw_frames_total"] = int(sum(r["frames"] for r in all_records))
    summary["physics_steps_total"] = int(sum(r["physics_steps"] for r in all_records))
    summary["time_axis_all_consistent"] = bool(all(r["time_axis_consistent"] for r in all_records))
    summary["independent_roots"] = len({r["root_id"] for r in all_records})
    summary["skills_total"] = int(sum(r["skills"] for r in all_records))
    summary["success_rate"] = float(np.mean([r["success"] for r in all_records]))
    summary["branch_candidate_matches"] = float(np.mean(
        [r["candidate_matches_executed"] for r in branch_records])) if branch_records else None
    summary["goal_distance_m"] = goal_distance
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "TRIAL_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _summarise(records: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {"episodes": 0}
    outcomes: Dict[str, int] = {}
    for record in records:
        outcomes[record["outcome"]] = outcomes.get(record["outcome"], 0) + 1
    return {
        "episodes": len(records),
        "roots": len({r["root_id"] for r in records}),
        "scenes": len({r["scene"] for r in records}),
        "skills": int(sum(r["skills"] for r in records)),
        "raw_frames": int(sum(r["frames"] for r in records)),
        "physics_steps": int(sum(r["physics_steps"] for r in records)),
        "success_rate": float(np.mean([r["success"] for r in records])),
        "mean_end_distance_m": float(np.mean([r["end_distance_m"] for r in records])),
        "mean_progress_towards_goal_m": float(np.mean(
            [r["progress_towards_goal_m"] for r in records])),
        "mean_skills": float(np.mean([r["skills"] for r in records])),
        "outcomes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    parser.add_argument("--branch-snapshots", type=int, default=20)
    parser.add_argument("--branch-candidates", type=int, default=4)
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--max-skills", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    summary = trial(policy_path=Path(args.policy) if args.policy else None,
                    episodes_per_setting=args.episodes, epsilons=args.epsilons,
                    branch_snapshots=args.branch_snapshots,
                    branch_candidates=args.branch_candidates, seed=args.seed,
                    goal_distance=args.goal_distance, max_skills=args.max_skills,
                    out_dir=Path(args.out))
    print(json.dumps(summary, indent=1, default=float))
    print("COLLECT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
