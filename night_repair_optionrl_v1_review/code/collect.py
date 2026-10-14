"""Cross-skill long-trajectory collection: policy, mixed and branch sources.

Every episode is one world carried across many skills. What is saved is designed so the three
consumers the task book names can be built from it later:

* a single-skill real prefix -> next step / terminal;
* a multi-skill history -> the consequence of a given candidate skill;
* a fixed sequence prediction when the future command is given explicitly.

Privileged quantities (true object pose and velocity, reward components, success) are written into a
**separate label block**, never into the observation tensor. The coordinate frames are the world
frame plus the per-option anchor recorded at each boundary, so a loader can rebuild any single-skill
window without silently mixing frames.

``epsilon`` is the probability that a boundary uses a random **legal skill** instead of the policy's
choice. It is not action noise inside a skill.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, field
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
from experiments.night_repair_optionrl_v1.policy import RecurrentActorCritic  # noqa: E402
from experiments.night_repair_optionrl_v1.ppo import legal_actions  # noqa: E402
from coimagine.local_contact_option.roots import RootSpec, VariantSpec, build_roots  # noqa: E402

OUT = ROOT / "outputs/night_repair_optionrl_v1/collection"
SCHEMA_VERSION = "night-repair-optionrl-v1/episode-1"
#: reward component names, always reported separately
REWARD_COMPONENTS = ("progress", "success", "failure", "time")


def load_policy(path: Path, device: str = "cpu") -> RecurrentActorCritic:
    saved = torch.load(path, map_location=device, weights_only=False)
    policy = RecurrentActorCritic(saved["observation_width"], saved["action_count"],
                                  hidden=saved["hidden"])
    policy.load_state_dict(saved["state_dict"])
    policy.eval()
    return policy


def _snapshot_to_dict(snapshot) -> Optional[Dict[str, object]]:
    """Serialise a snapshot for the branch-and-resume path, or ``None`` if it is not serialisable."""

    if hasattr(snapshot, "as_dict"):
        return snapshot.as_dict()
    try:
        import dataclasses
        return dataclasses.asdict(snapshot)
    except Exception:
        return None


def _observation_block(env: OptionRLEnv) -> Dict[str, np.ndarray]:
    """The legal observation, split into its named parts so a loader does not guess the layout."""

    return {"legal": env._observation()}


def _privileged_block(env: OptionRLEnv) -> Dict[str, object]:
    pose = env.privileged_object()
    return {"object_pose": [float(v) for v in pose[:3]],
            "object_velocity": [float(v) for v in pose[3:6]],
            "goal": [float(env.goal[0]), float(env.goal[1])],
            "note": "privileged; must never be concatenated into an observation tensor"}


def collect_episode(env: OptionRLEnv, policy: Optional[RecurrentActorCritic], root: RootSpec,
                    *, source: str, epsilon: float, rng: np.random.Generator,
                    goal_distance: float = 0.45, max_skills: int = 12,
                    snapshot_every: int = 1) -> Dict[str, object]:
    """One episode with the fields section 6.3 requires."""

    goal = default_goal(root, goal_distance)
    observation = env.reset(root, goal)
    allowed = legal_actions(root)
    hidden = policy.initial_hidden(1) if policy is not None else None
    episode_id = f"{source}_{root.root_id.replace('/', '_')}_{int(rng.integers(0, 1 << 31)):08x}"
    frames: List[np.ndarray] = []
    frame_time: List[float] = []
    trace: List[Dict[str, object]] = []
    boundaries: List[Dict[str, object]] = []
    snapshots: List[Dict[str, object]] = []
    exit_reasons: List[str] = []
    total_reward = 0.0
    start = env.privileged_object().copy()
    success = terminated = truncated = False
    for index in range(max_skills):
        if policy is not None:
            with torch.no_grad():
                logits, _value, hidden = policy.step(torch.tensor(observation[None, :]), hidden)
                mask = torch.full((len(MENU),), float("-inf"))
                mask[allowed] = 0.0
                probabilities = torch.softmax(logits[0] + mask, dim=-1).numpy()
            if epsilon > 0.0 and rng.random() < epsilon:
                action_index = int(allowed[rng.integers(0, len(allowed))])
                chosen_by = "random_legal"
            else:
                action_index = int(rng.choice(len(MENU), p=probabilities / probabilities.sum()))
                chosen_by = "policy"
        else:
            action_index = int(allowed[rng.integers(0, len(allowed))])
            probabilities = np.full(len(MENU), 1.0 / len(allowed))
            chosen_by = "random_legal"
        if index % snapshot_every == 0 and env.env is not None:
            snapshot = env.env.export_state()
            snapshots.append({"snapshot_id": f"{episode_id}#{index}", "skill_index": index,
                              "state": _snapshot_to_dict(snapshot),
                              "anchor_goal": [float(goal[0]), float(goal[1])]})
        before = env.privileged_object().copy()
        result = env.step(action_index)
        after = env.privileged_object().copy()
        for row_index, row in enumerate(result.inner_observations):
            frames.append(row)
            frame_time.append(float(len(frames)) * 0.05)
        trace.extend(result.trace)
        frames.append(result.observation)
        frame_time.append(float(len(frames)) * 0.05)
        boundaries.append({
            "option_id": index,
            "chosen_by": chosen_by,
            "action_index": int(action_index),
            "action": MENU[action_index].label(),
            "active_robots": list(MENU[action_index].active),
            "alpha_deg": [round(np.degrees(a), 4) for a in MENU[action_index].alpha],
            "speed": [float(v) for v in MENU[action_index].speed],
            "action_probability": float(probabilities[action_index]),
            "duration_seconds": float(result.skill_steps) * 0.05,
            "physics_steps": int(result.skill_steps),
            "exit_reason": str(env.boundaries[-1].get("exit_reason", "")),
            "settled": bool(env.boundaries[-1].get("settled", False)),
            "push_completed": bool(env.boundaries[-1].get("push_completed", False)),
            "object_before": [float(v) for v in before[:2]],
            "object_after": [float(v) for v in after[:2]],
            "reward": float(result.reward),
        })
        exit_reasons.append(str(env.boundaries[-1].get("exit_reason", "")))
        total_reward += float(result.reward)
        if result.terminated:
            terminated, success = True, True
            break
        if result.truncated:
            truncated = True
            break
        observation = result.observation
    final = env.privileged_object()
    payload = {
        "schema": SCHEMA_VERSION,
        "episode_id": episode_id,
        "root_id": str(root.root_id),
        "split": str(root.split),
        "scene": str(root.interaction_class),
        "template": str(root.template),
        "active_robots": list(root.active_robots),
        "behaviour_source": source,
        "epsilon": float(epsilon),
        "goal_distance_m": float(goal_distance),
        "goal": [float(goal[0]), float(goal[1])],
        "policy_version": "raw" if policy is None else "trained",
        "seed": int(root.seed),
        "goal_distance_start_m": float(np.hypot(start[0] - goal[0], start[1] - goal[1])),
        "goal_distance_end_m": float(np.hypot(final[0] - goal[0], final[1] - goal[1])),
        "net_progress_m": float(np.hypot(final[0] - goal[0], final[1] - goal[1])
                                - np.hypot(start[0] - goal[0], start[1] - goal[1])),
        "total_reward": float(total_reward),
        "success": bool(success),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "skills": len(boundaries),
        "raw_frames": int(len(frames)),
        "exit_reasons": exit_reasons,
        "boundaries": boundaries,
        "snapshots": [s for s in snapshots if s.get("state") is not None],
    }
    return {"payload": payload,
            "frames": np.asarray(frames, dtype=np.float32) if frames else np.zeros((0, 0)),
            "frame_time": np.asarray(frame_time, dtype=np.float64),
            "privileged": _privileged_block(env),
            "trace": trace}


def write_shard(directory: Path, payload: Dict[str, object], frames: np.ndarray,
                frame_time: np.ndarray, index: int,
                trace: Optional[Sequence[Dict[str, object]]] = None) -> Path:
    """One episode per shard, written atomically so an interruption cannot leave a half file."""

    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"episode_{index:06d}.npz"
    # numpy appends its own extension unless the name already ends in .npz, so the temporary file
    # keeps the suffix and the atomic replace still targets the final name
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, frames=frames, frame_time=frame_time,
                        trace=np.asarray(json.dumps(trace)), meta=np.asarray(json.dumps(payload)))
    temporary.replace(path)
    return path


def trial_collection(*, policy_path: Optional[Path], episodes_per_setting: int = 20,
                     epsilons: Sequence[float] = (0.0, 0.1, 0.3), branch_snapshots: int = 20,
                     branch_candidates: int = 4, seed: int = 0, goal_distance: float = 0.8,
                     out_dir: Path = OUT, roots: Optional[Sequence[RootSpec]] = None) -> Dict[str, object]:
    """The bounded trial the task book asks for before any expansion."""

    rng = np.random.default_rng(seed)
    policy = load_policy(policy_path) if policy_path and Path(policy_path).exists() else None
    pool = list(roots) if roots is not None else build_roots("dev", 6)
    summary: Dict[str, object] = {"schema": SCHEMA_VERSION, "policy": str(policy_path),
                                  "policy_loaded": policy is not None, "settings": {}}
    index = 0
    all_frames = 0
    all_episodes = []
    for epsilon in epsilons:
        source = "policy" if epsilon == 0.0 else "mixed"
        directory = Path(out_dir) / f"{source}_eps{epsilon:g}"
        episodes = []
        for _ in range(episodes_per_setting):
            root = pool[int(rng.integers(0, len(pool)))]
            env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30))))
            record = collect_episode(env, policy, root, source=source, epsilon=epsilon, rng=rng,
                                     goal_distance=goal_distance)
            write_shard(directory, record["payload"], record["frames"], record["frame_time"],
                        index, record["trace"])
            index += 1
            all_frames += int(record["frames"].shape[0])
            episodes.append(record["payload"])
            all_episodes.append(record["payload"])
        summary["settings"][f"{source}_eps{epsilon:g}"] = _summarise(episodes)
    # branch: from saved snapshots, execute several legal candidates from the same state
    directory = Path(out_dir) / "branch"
    branch_episodes = []
    for family in range(branch_snapshots):
        root = pool[int(rng.integers(0, len(pool)))]
        env = OptionRLEnv(OptionRLConfig(seed=int(rng.integers(0, 1 << 30))))
        goal = default_goal(root, goal_distance)
        env.reset(root, goal)
        snapshot = env.env.export_state()
        allowed = legal_actions(root)
        picks = [int(allowed[i]) for i in
                 rng.choice(len(allowed), size=min(branch_candidates, len(allowed)), replace=False)]
        for candidate in picks:
            probe = OptionRLEnv(OptionRLConfig(seed=env.config.seed))
            probe.reset(root, goal)
            probe.env.restore(snapshot, exact=True)
            record = collect_episode(probe, policy, root, source="branch", epsilon=1.0, rng=rng,
                                     goal_distance=goal_distance, max_skills=1)
            record["payload"]["parent_snapshot_id"] = f"branch_family_{family:03d}"
            record["payload"]["candidate_action_index"] = int(candidate)
            record["payload"]["candidate_action"] = MENU[candidate].label()
            write_shard(directory, record["payload"], record["frames"], record["frame_time"],
                        index, record["trace"])
            index += 1
            all_frames += int(record["frames"].shape[0])
            branch_episodes.append(record["payload"])
            all_episodes.append(record["payload"])
    summary["settings"]["branch"] = _summarise(branch_episodes)
    summary["branch_families"] = branch_snapshots
    summary["episodes_total"] = len(all_episodes)
    summary["raw_frames_total"] = int(all_frames)
    summary["independent_roots"] = len({e["root_id"] for e in all_episodes})
    summary["skills_total"] = int(sum(e["skills"] for e in all_episodes))
    summary["success_rate"] = float(np.mean([e["success"] for e in all_episodes]))
    summary["truncated_rate"] = float(np.mean([e["truncated"] for e in all_episodes]))
    summary["goal_distance_m"] = goal_distance
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "TRIAL_SUMMARY.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary


def _summarise(episodes: Sequence[Dict[str, object]]) -> Dict[str, object]:
    if not episodes:
        return {"episodes": 0}
    reasons: Dict[str, int] = {}
    for episode in episodes:
        for reason in episode["exit_reasons"]:
            reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "episodes": len(episodes),
        "roots": len({e["root_id"] for e in episodes}),
        "scenes": len({e["scene"] for e in episodes}),
        "skills": int(sum(e["skills"] for e in episodes)),
        "raw_frames": int(sum(e["raw_frames"] for e in episodes)),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "truncated_rate": float(np.mean([e["truncated"] for e in episodes])),
        "mean_net_progress_m": float(np.mean([e["net_progress_m"] for e in episodes])),
        "mean_skills": float(np.mean([e["skills"] for e in episodes])),
        "exit_reason_histogram": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default=None)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--epsilons", type=float, nargs="+", default=[0.0, 0.1, 0.3])
    parser.add_argument("--branch-snapshots", type=int, default=20)
    parser.add_argument("--branch-candidates", type=int, default=4)
    parser.add_argument("--goal-distance", type=float, default=0.45)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    summary = trial_collection(policy_path=Path(args.policy) if args.policy else None,
                               episodes_per_setting=args.episodes, epsilons=args.epsilons,
                               branch_snapshots=args.branch_snapshots,
                               branch_candidates=args.branch_candidates, seed=args.seed,
                               goal_distance=args.goal_distance, out_dir=Path(args.out))
    print(json.dumps(summary, indent=1, default=float))
    print("COLLECT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
