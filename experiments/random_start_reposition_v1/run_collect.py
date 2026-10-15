"""S5 trial collection, plus a small schema-validation run.

The task book gates this stage on S4 showing stable learning, so the default mode is
``schema``: a few complete trajectories and one-step branches whose only purpose is to prove
that the data path, the branch restore and the generated data card agree with the real schema.
``--mode trial`` runs the numbers section 9 authorises (20 + 20 + 20 complete trajectories and
20 states x 4 forced joint actions) and must only be used once the S4 gate has passed.

Everything that a controller may read is stored in ``observation`` (the 324-dim legal joint
vector of that period). Privileged quantities (true object pose/velocity, reward components,
task events, physics state for animation) are stored under ``privileged`` and are loaded only by
an explicit whitelist, never concatenated into an input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from coimagine.random_start import env as E  # noqa: E402
from coimagine.random_start import layout  # noqa: E402
from coimagine.random_start.spec import ROBOT_ACTION_COUNT, ROBOT_ACTIONS  # noqa: E402

OUT = ROOT / "outputs/random_start_reposition_v1"
SCHEMA_VERSION = "random-start-collection-v1"


def behaviour_action(policy, environment, epsilon: float, rng: np.random.Generator,
                     stochastic: bool = True) -> Tuple[int, int, float, str]:
    """One joint action with the behaviour probability actually used to draw it."""

    if epsilon > 0.0 and float(rng.random()) < epsilon:
        first = int(rng.integers(0, ROBOT_ACTION_COUNT))
        second = int(rng.integers(0, ROBOT_ACTION_COUNT))
        return first, second, float(epsilon) / ROBOT_ACTION_COUNT ** 2, "epsilon_mix"
    if policy is None:
        first = int(rng.integers(0, ROBOT_ACTION_COUNT))
        second = int(rng.integers(0, ROBOT_ACTION_COUNT))
        return first, second, 1.0 / ROBOT_ACTION_COUNT ** 2, "uniform"
    hidden = policy.initial_hidden()
    with torch.no_grad():
        frame = torch.as_tensor(environment.boundary_vector(), dtype=torch.float32).unsqueeze(0)
        hidden = policy.advance(frame, hidden)
        first, second, logp, _entropy = policy.sample(hidden, deterministic=not stochastic)
    return int(first), int(second), float(np.exp(logp)), "policy"


def collect_episode(instance, policy, epsilon: float, seed: int, rng: np.random.Generator,
                    keep_rows: bool = False) -> Dict[str, object]:
    environment = E.RandomStartEnv(instance)
    decisions: List[Dict[str, object]] = []
    vectors: List[np.ndarray] = []
    while environment.termination is None:
        first, second, probability, source = behaviour_action(policy, environment, epsilon, rng)
        index_start = len(environment.rows)
        environment.joint_step(first, second)
        boundary = environment.boundaries[-1]
        rows = environment.rows[index_start:index_start + int(boundary["periods"])]
        decisions.append({
            "skill_index": int(boundary["skill_index"]),
            "actions": [int(first), int(second)],
            "action_labels": [ROBOT_ACTIONS[first].label, ROBOT_ACTIONS[second].label],
            "behaviour_probability": float(probability), "behaviour_source": source,
            "periods": int(boundary["periods"]),
            "start_period": int(boundary["start_period"]),
            "start_time": float(boundary["start_time"]), "end_time": float(boundary["end_time"]),
            "skill_ends": boundary["skill_ends"],
            "reward": boundary["reward"],
            "success": bool(boundary["success"]), "truncated": bool(boundary["truncated"]),
            "anomaly": boundary["anomaly"], "termination": boundary["termination"],
            "phi_before": float(boundary["phi_before"]),
            "phi_next_used": float(boundary["phi_next_used"]),
        })
        for row in rows:
            vectors.append(np.asarray(environment.history[row["period"] - 1]["observation"],
                                      dtype=np.float16))
    return {
        "schema_version": SCHEMA_VERSION,
        "episode_id": f"{instance.name}_eps{epsilon:.1f}_seed{seed}",
        "instance": instance.as_dict(),
        "behaviour": {"policy": "untrained_or_trained_joint_gru" if policy else "uniform_random",
                      "epsilon": float(epsilon), "seed": int(seed)},
        "checkpoint_hash": (hashlib.sha256(
            json.dumps(sorted((name, list(parameter.shape))
                              for name, parameter in (policy.state_dict().items()
                                                      if policy is not None else []))
                       ).encode()).hexdigest()[:16] if policy is not None else None),
        "goal": {"centre": list(instance.goal_centre), "radius": instance.goal_radius},
        "periods": int(environment.total_periods),
        "skills_used": int(environment.skills_used),
        "time": float(environment.total_periods) * E.CONTROL_DT,
        "success": bool(environment.success), "termination": environment.termination,
        "return": float(environment.reward_total),
        "reward_components": dict(environment.reward_components),
        "decisions": decisions,
        "observation": np.stack(vectors) if vectors else np.zeros((0, 0), dtype=np.float16),
        "privileged": {
            "object_final": list(environment.object_state().as_tuple()),
            "final_goal_distance": float(environment.object_distance()),
            "rows": environment.rows if keep_rows else None,
            "note": "privileged block: true object pose/velocity, reward components and task "
                    "events; never concatenated into a controller input",
        },
    }


def collect_branches(instances: Sequence[layout.InstanceSpec], states_per_instance: int,
                     actions_per_state: int, rng: np.random.Generator,
                     parent_skills: int = 2) -> Tuple[List[Dict[str, object]], int]:
    """One-step branches from non-terminal intermediate states, with a restorable parent.

    The parent is cut by the **sampler** (a fixed number of skills), never by the task budget, so
    a truncated parent can never be confused with a task-budget termination.
    """

    branches: List[Dict[str, object]] = []
    simulated = 0
    for instance in instances:
        environment = E.RandomStartEnv(instance)
        while environment.skills_used < parent_skills and environment.termination is None:
            before = environment.world_advances
            environment.joint_step(int(rng.integers(0, ROBOT_ACTION_COUNT)),
                                   int(rng.integers(0, ROBOT_ACTION_COUNT)))
            simulated += environment.world_advances - before
        for state_index in range(states_per_instance):
            if environment.termination is not None:
                break
            snapshot = environment.export_branch_state()
            parent_periods = environment.total_periods
            for action_index in range(actions_per_state):
                first = int(rng.integers(0, ROBOT_ACTION_COUNT))
                second = int(rng.integers(0, ROBOT_ACTION_COUNT))
                environment.restore_branch_state(snapshot)
                before_branch = environment.world_advances
                boundary = environment.joint_step(first, second)
                simulated += environment.world_advances - before_branch
                parent_snapshot_hash = hashlib.sha256(
                    json.dumps({"periods": parent_periods,
                                "skills": int(snapshot["skills_used"]),
                                "counters": [snapshot["total_periods"],
                                             snapshot["world_advances"]]},
                               sort_keys=True).encode()).hexdigest()[:12]
                branch = {
                    "branch_id": f"{instance.name}_s{state_index}_a{action_index}",
                    "parent": {"episode_instance": instance.name,
                               "parent_periods": parent_periods,
                               "parent_skills_used": int(snapshot["skills_used"]),
                               "parent_snapshot_hash": parent_snapshot_hash,
                               "parent_termination": snapshot["termination"],
                               "parent_cut_by": "sampler"},
                    "forced_action": [first, second],
                    "forced_action_labels": [ROBOT_ACTIONS[first].label,
                                             ROBOT_ACTIONS[second].label],
                    "intervention": True,
                    "behaviour_probability": None,
                    "periods": int(boundary["periods"]),
                    "reward": boundary["reward"], "success": bool(boundary["success"]),
                    "truncated": bool(boundary["truncated"]),
                    "termination": boundary["termination"],
                    "privileged": {"object_final": list(environment.object_state().as_tuple()),
                                   "final_goal_distance": float(environment.object_distance())},
                }
                branches.append(branch)
            # advance the parent so the next state is a genuinely different intermediate state
            environment.restore_branch_state(snapshot)
            before = environment.world_advances
            environment.joint_step(int(rng.integers(0, ROBOT_ACTION_COUNT)), 13)
            simulated += environment.world_advances - before
    return branches, simulated


def data_card(episodes: Sequence[Dict[str, object]], branches: Sequence[Dict[str, object]],
              stored_vectors: int, simulated_periods: int) -> Dict[str, object]:
    first = episodes[0] if episodes else {}
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_from": "the actual arrays written by run_collect.py; no field is described "
                          "here that the loader does not produce",
        "episodes": len(episodes),
        "branches": len(branches),
        "episode_fields": sorted(list(first.keys())) if first else [],
        "decision_fields": sorted(list(first["decisions"][0].keys())) if first.get("decisions")
        else [],
        "observation_field_count": int(first["observation"].shape[1])
        if first.get("observation") is not None and first["observation"].size else 0,
        "observation_dtype": "float16 on disk (float32 in memory)",
        "stored_observation_vectors": stored_vectors,
        "privileged_fields": ["object_final", "final_goal_distance", "rows"],
        "privileged_note": "reward components, true object state and task events live only in "
                           "the privileged block; the input loader copies the observation key",
        "successes": int(sum(1 for episode in episodes if episode["success"])),
        "failures": int(sum(1 for episode in episodes if not episode["success"])),
        "truncations": int(sum(1 for episode in episodes
                               if episode["termination"] == "task_budget_exhausted")),
        "anomalies": int(sum(1 for episode in episodes
                             if episode["termination"] in E.ANOMALY_TERMINATIONS)),
        "behaviour_mix": sorted({float(episode["behaviour"]["epsilon"]) for episode in episodes}),
        "unique_initial_states": len({episode["episode_id"].rsplit("_eps", 1)[0]
                                      for episode in episodes}),
        "frames_stored": int(sum(int(episode["periods"]) for episode in episodes)),
        "unique_physical_frames": int(sum(int(episode["periods"]) for episode in episodes)),
        "simulated_control_periods": int(simulated_periods),
        "simulation_note": "stored frames count each episode once; the parent prefixes and branch "
                           "replays are extra simulated periods and are not counted as new "
                           "independent initial states",
        "branch_interventions": int(sum(1 for branch in branches if branch["intervention"])),
        "compatibility": "the observation layout is the new 324-dim joint vector of this round; "
                         "the previous round's world model was trained on the old observation "
                         "and skill protocol and must not be fed this data",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--mode", default="schema", choices=("schema", "trial"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = Path(args.out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)

    policy = None
    if args.checkpoint and Path(args.checkpoint).exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "rl", ROOT / "experiments/random_start_reposition_v1/rl.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        probe, _ = layout.build_split("L0", "train", count=1)
        policy = module.JointPolicy(E.RandomStartEnv(probe[0]).observation_fields().__len__())
        policy.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
        policy.eval()

    if args.mode == "schema":
        counts = {"eps0": 2, "eps01": 2, "eps03": 2}
        states_per_instance, actions_per_state = 5, 4
    else:
        counts = {"eps0": 20, "eps01": 20, "eps03": 20}
        states_per_instance, actions_per_state = 20, 4
    rng = np.random.default_rng(args.seed)
    instances, _ = layout.build_split("L0", "train", count=8)

    episodes: List[Dict[str, object]] = []
    for condition, epsilon in (("eps0", 0.0), ("eps01", 0.1), ("eps03", 0.3)):
        for index in range(counts[condition]):
            instance = instances[index % len(instances)]
            episode = collect_episode(instance, policy, epsilon, args.seed + index, rng,
                                      keep_rows=(index < 2))
            episodes.append(episode)
    branches, branch_periods = collect_branches(instances[:4], states_per_instance,
                                                actions_per_state, rng)

    stored_vectors = int(sum(int(episode["periods"]) for episode in episodes))
    # full per-period logs for the first two episodes of each condition
    sampled = [episode for index, episode in enumerate(episodes) if index % counts["eps0"] < 2]
    for episode in sampled:
        payload = {key: value for key, value in episode.items()
                   if key not in ("observation", "privileged")}
        (out / "logs" / f"{episode['episode_id']}_trajectory.json").write_text(
            json.dumps(payload, indent=1, default=str))
        np.savez_compressed(out / "data" / f"{episode['episode_id']}_observations.npz",
                            observation=episode["observation"])
    np.savez_compressed(
        out / "data" / "episodes.npz",
        **{f"obs_{index}": episode["observation"] for index, episode in enumerate(episodes)})
    (out / "data" / "branches.json").write_text(json.dumps(branches, indent=1, default=str))
    episode_periods = int(sum(int(episode["periods"]) for episode in episodes))
    card = data_card(episodes, branches, stored_vectors, episode_periods + branch_periods)
    (out / "data" / "DATACARD.json").write_text(json.dumps(card, indent=1))
    (out / "results" / f"collection_{args.mode}.json").write_text(json.dumps(
        {"mode": args.mode, "card": card,
         "episodes": [{key: value for key, value in episode.items()
                       if key not in ("observation", "privileged")} for episode in episodes],
         "branches": branches}, indent=1, default=str))
    print(json.dumps(card, indent=1))
    print("COLLECT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
