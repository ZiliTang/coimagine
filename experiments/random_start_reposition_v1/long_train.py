"""Bounded 24h dense-reward training: 8 isolated rollout processes, one synchronous learner.

Monitoring is driven by completed updates, not a separate policy or training mechanism.
The supervisor enforces the original deadline, including on resume, and kills the process
group at that deadline so rollout workers cannot become orphan simulations.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import multiprocessing as mp
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

import rl


def atomic_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_save(path, data):
    temporary = path.with_suffix(".tmp")
    torch.save(data, temporary)
    temporary.replace(path)


def snapshot(policy):
    return {key: value.detach().cpu().numpy().copy() for key, value in policy.state_dict().items()}


def snapshot_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def balanced_indices(seed, update, count=256, batch=8):
    if count % batch:
        raise ValueError("Panel size must be divisible by batch size")
    epoch, slot = divmod(update - 1, count // batch)
    order = np.random.default_rng(np.random.SeedSequence([seed, epoch, 256])).permutation(count)
    return order[slot * batch:(slot + 1) * batch].tolist()


def worker_init():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def rollout_job(job):
    """Only this process owns the world's state/history; transferred arrays own their memory."""
    instance, state, hidden, seed, deterministic, config, uniform = job
    environment = rl.E.RandomStartEnv(instance, config)
    if uniform:
        rng = np.random.default_rng(seed)
        while environment.termination is None:
            environment.joint_step(int(rng.integers(15)), int(rng.integers(15)))
        episode = dict(decisions=[], success=environment.success, termination=environment.termination,
                       final_goal_distance=environment.object_distance(),
                       skills_used=environment.skills_used, periods=environment.total_periods,
                       return_=environment.reward_total)
        episode["return"] = episode.pop("return_")
    else:
        policy = rl.JointPolicy(len(environment.observation_fields()), hidden=hidden)
        policy.load_state_dict({key: torch.from_numpy(value.copy()) for key, value in state.items()})
        policy.eval()
        episode = rl.rollout(environment, policy, stochastic=not deterministic,
                             generator=torch.Generator().manual_seed(seed))
        for decision in episode["decisions"]:
            for key in ("observations", "boundary_frame"):
                decision[key] = decision[key].numpy().copy()
    episode.update(instance=instance.name, behaviour_seed=seed, policy_hash=snapshot_hash(state))
    return episode


def collect(executor, jobs, timeout=900):
    futures = [executor.submit(rollout_job, job) for job in jobs]
    # Original job order, NOT worker completion order, defines the update batch.
    episodes = [future.result(timeout=timeout) for future in futures]
    for episode in episodes:
        if episode["termination"] in rl.E.ANOMALY_TERMINATIONS:
            raise RuntimeError(f"Simulation anomaly: {episode['instance']} {episode['termination']}")
        for decision in episode["decisions"]:
            for key in ("observations", "boundary_frame"):
                decision[key] = torch.from_numpy(decision[key])
    return episodes


def evaluate(executor, policy, instances, config, stochastic, uniform=False):
    state = snapshot(policy)
    jobs = [(instance, state, policy.hidden, seed, not stochastic, config, uniform)
            for instance in instances for seed in range(3 if stochastic else 1)]
    episodes = collect(executor, jobs)
    rows = [{key: episode[key] for key in ("instance", "behaviour_seed", "success", "return",
                                          "periods", "final_goal_distance", "skills_used")}
            for episode in episodes]
    return dict(success=float(np.mean([row["success"] for row in rows])),
                return_mean=float(np.mean([row["return"] for row in rows])),
                median_final_distance=float(np.median([row["final_goal_distance"] for row in rows])),
                rows=rows, instances=len(instances), seeds_per_instance=3 if stochastic else 1)


def save_checkpoint(path, policy, optimizer, history, initial, best, manifest):
    atomic_save(path, dict(env_version=rl.ENV_VERSION, config=manifest["config"],
                          update=history[-1]["update"] if history else 0,
                          model=policy.state_dict(), optimizer=optimizer.state_dict(),
                          torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                          history=history, initial=initial, best=best,
                          deadline_unix=manifest["deadline_unix"]))


def load_checkpoint(path, policy, optimizer, manifest):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state["env_version"] != rl.ENV_VERSION or state["config"] != manifest["config"]
            or state["deadline_unix"] != manifest["deadline_unix"]):
        raise ValueError("Resume contract/config/deadline mismatch")
    policy.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    return state


def monitor(out, manifest, history, best, status, plot=False):
    latest = history[-1] if history else {}
    points = [row for row in history if "evaluation" in row]
    recent = history[-50:]
    initial_path = out / "initial.json"
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else None
    gate = rl.sustained_pass(
        [dict(dev_success_deterministic=row["evaluation"]["deterministic"]["success"]) for row in points],
        initial["deterministic"]["success"]) if initial else False
    summary = dict(status=status, update=latest.get("update", 0),
                   training_episodes=len(history) * manifest["config"]["episodes_per_update"],
                   elapsed_hours=(time.time() - manifest["started_unix"]) / 3600,
                   deadline_unix=manifest["deadline_unix"], best=best,
                   recent_return_mean=float(np.mean([r["return_mean"] for r in recent])) if recent else None,
                   recent_success=float(np.mean([r["success"] for r in recent])) if recent else None,
                   latest_evaluation=points[-1]["evaluation"] if points else None,
                   latest_critic=latest.get("critic_after"), sustained_s4_gate=gate)
    atomic_json(out / "monitor.json", summary)
    (out / "README.md").write_text(
        "# 24小时稠密奖励训练\n\n"
        f"状态：**{status}**；更新 **{summary['update']}/5000**；已完成训练轨迹 **{summary['training_episodes']}**。\n\n"
        f"耗时 {summary['elapsed_hours']:.2f} 小时；最近最多50更新平均回报 {summary['recent_return_mean']}，"
        f"采样成功率 {summary['recent_success']}。\n\n"
        "实时详情：`monitor.json`；逐更新：`updates.jsonl`；定期监测：`monitor_history.jsonl`；"
        "曲线：`curves.png`；恢复：`checkpoints/resume.pt`；评价最佳：`checkpoints/best.pt`。\n\n"
        f"{len(manifest['config']['instances'])}个训练实例；不使用测试集；不自动进入S5。奖励、网络和技能冻结。\n")
    if plot and history:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 1, figsize=(10, 10))
        x = [row["update"] for row in history]
        for axis, key in zip(axes[:2], ("return_mean", "success")):
            y = np.asarray([row[key] for row in history])
            axis.plot(x, y, alpha=.25, label="training batch")
            n = min(50, len(y))
            axis.plot(x[n - 1:], np.convolve(y, np.ones(n) / n, mode="valid"), label=f"mean {n}")
            axis.set_ylabel(key)
        for key in ("deterministic", "stochastic"):
            axes[1].plot([row["update"] for row in points],
                         [row["evaluation"][key]["success"] for row in points], label=key)
        for key in ("train_panel", "validation"):
            available = [row for row in points if key in row["evaluation"]]
            if available:
                axes[1].plot([row["update"] for row in available],
                             [row["evaluation"][key]["success"] for row in available], label=key)
        axes[2].plot(x, [row["critic_after"]["explained_variance"] for row in history], label="critic EV, same batch")
        for axis in axes:
            axis.legend(); axis.grid(alpha=.2)
        axes[2].set_xlabel("update")
        fig.tight_layout()
        temporary = out / "curves.tmp.png"
        fig.savefig(temporary, dpi=130); plt.close(fig)
        temporary.replace(out / "curves.png")
    return summary


# ===========================================================================
# Bounded two-arm stability comparison (task book 2026-10-19)
#
# Arm A ("shared") keeps the existing critic gradient into the encoder/GRU. Arm B
# ("stopped") detaches the hidden state at the value read-out only. Every other setting is
# shared, including the warm start, the optimiser, the batch order, the seeds and the global
# gradient clip. The two arms are run sequentially, each in its own directory, with its own
# deadline and its own resumable checkpoint.
# ===========================================================================

STABILITY_ROOT = rl.ROOT / "outputs/random_start_rl_stability_long_v1"
DIAGNOSIS_ROOT = rl.ROOT / "outputs/random_start_rl_diagnosis_v4"
WARM_START = rl.ROOT / "outputs/random_start_time_budget_v3_coverage256/checkpoints/initial.pt"
WARM_START_SHA256 = "1e61a1e46ee30a2bcc60cad707a10dba5b66df5f016c0e11bf9bd09f38e6a32f"
REPLAY_INITIAL = DIAGNOSIS_ROOT / "replay_initial.pt"
TRACKING_NAMES = ["L0_train_0005", "L0_train_0010", "L0_train_0013", "L0_train_0017",
                  "L0_train_0020", "L0_train_0038", "L0_train_0056", "L0_train_0063"]
STABILITY_ARMS = {"shared": True, "stopped": False}
STABILITY_RUN_DIRS = {("pilot", "shared"): "pilot_A", ("pilot", "stopped"): "pilot_B",
                      ("long", "shared"): "A_shared", ("long", "stopped"): "B_stopped"}
PANEL_NAMES = ("train", "tracking", "train_panel", "validation", "test")


def instance_from_dict(payload):
    from coimagine.continuous2d.scenes import Box, Pose

    return rl.layout.InstanceSpec(
        name=payload["name"], map=payload["map"], split=payload["split"],
        index=int(payload["index"]), seed=int(payload["seed"]),
        arena=tuple(payload["arena"]),
        obstacles=tuple(Box(**box) for box in payload["obstacles"]),
        object_pose=Pose(**payload["object"]),
        robot_poses=tuple(Pose(**pose) for pose in payload["robots"]),
        goal_centre=tuple(payload["goal_centre"]),
        goal_radius=float(payload["goal_radius"]),
        env_version=payload.get("env_version", rl.ENV_VERSION))


def build_stability_panels(verify=True):
    """Rebuild the independent-v2 L0 panels and verify their physical content.

    The panels are rebuilt deterministically rather than trusted from a name list. The
    verification compares full physical content hashes against the frozen diagnosis panels,
    so a changed sampler cannot silently swap the holdout.
    """

    panels = {}
    for split, count in (("train", 256), ("validation", 64), ("test", 64)):
        panels[split], _metadata = rl.layout.build_split("L0", split, count=count)
    audit = rl.layout.audit_split_content(
        {split: panels[split] for split in ("train", "validation", "test")})
    if not audit["ok"]:
        raise ValueError(f"Train/validation/test physical content overlaps: {audit}")
    by_name = {instance.name: instance for instance in panels["train"]}
    missing = [name for name in TRACKING_NAMES if name not in by_name]
    if missing:
        raise ValueError(f"Tracking instances missing from train256: {missing}")
    panels["tracking"] = [by_name[name] for name in TRACKING_NAMES]
    coverage_manifest = json.loads(
        (rl.ROOT / "outputs/random_start_time_budget_v3_coverage256/manifest.json").read_text())
    eval_names = list(coverage_manifest["config"]["evaluation_train_instances"])
    if len(eval_names) != 32 or any(name not in by_name for name in eval_names):
        raise ValueError("The fixed 32-instance training panel is unavailable")
    panels["train_panel"] = [by_name[name] for name in eval_names]
    if verify:
        reference = json.loads((DIAGNOSIS_ROOT / "panels.json").read_text())
        if reference.get("version") != rl.layout.SPLIT_VERSION:
            raise ValueError("Diagnosis panels are not the independent-v2 split")
        if reference.get("test_policy_evaluation") is not False:
            raise ValueError("Diagnosis panels were not frozen before test evaluation")
        for split in ("train", "validation", "test"):
            frozen = {rl.layout.content_hash(instance_from_dict(payload))
                      for payload in reference["instances"][f"L0/{split}"]}
            rebuilt = {rl.layout.content_hash(instance) for instance in panels[split]}
            if frozen != rebuilt:
                raise ValueError(f"Rebuilt L0/{split} physical content differs from the "
                                 "frozen diagnosis panels")
            if len(frozen) != len(panels[split]):
                raise ValueError(f"L0/{split} panel has physically duplicate instances")
    return panels


def panel_contract(panels):
    return {name: rl.layout.split_manifest(panels[name]) for name in PANEL_NAMES}


def write_stability_panels(root, panels):
    directory = root / "panels"
    directory.mkdir(parents=True, exist_ok=True)
    for name in PANEL_NAMES:
        atomic_json(directory / f"{name}.json", [instance.as_dict() for instance in panels[name]])
    atomic_json(directory / "panels.json",
                dict(split_version=rl.layout.SPLIT_VERSION, contract=panel_contract(panels),
                     content_audit=rl.layout.audit_split_content(
                         {split: panels[split] for split in ("train", "validation", "test")}),
                     test_policy_evaluation=False))


def stability_run_config(arm, stage, workers, panels):
    observation_dim = len(rl.E.RandomStartEnv(panels["train"][0]).observation_fields())
    return dict(
        arm=arm, stage=stage, critic_shared_grad=STABILITY_ARMS[arm],
        critic_mode=("shared" if STABILITY_ARMS[arm] else "stopped"),
        updates=64 if stage == "pilot" else 5000,
        hours_budget=2.0 if stage == "pilot" else 24.0,
        episodes_per_update=8, workers=workers, seed=0,
        split_version=rl.layout.SPLIT_VERSION, environment=rl.E.EnvConfig().as_dict(),
        panel_contract=panel_contract(panels), observation_dim=observation_dim,
        hidden=rl.HIDDEN, learning_rate=rl.LEARNING_RATE, clip=rl.CLIP_RANGE,
        value_weight=rl.VALUE_WEIGHT, entropy_weight=rl.ENTROPY_WEIGHT, weight_decay=0,
        update_epochs=rl.UPDATE_EPOCHS, eval_every=100, audit_every=100, full_eval_every=500,
        warm_start=str(WARM_START.relative_to(rl.ROOT)), warm_start_sha256=WARM_START_SHA256,
        optimizer_initialization="fresh AdamW; the warm start contains weights only",
        sampling="shuffled_epochs: permute train256 each update, 8 distinct instances per batch",
        reward_config=rl.E.EnvConfig().as_dict(),
        advantage_estimator="Monte-Carlo semi-Markov return minus value; no GAE",
        gradient_truncation="detach after each pre-action actor/critic head",
        training_semantics_version="pre_action_value_tbptt_v2",
        entropy_statistic="exact joint entropy H(a0)+H(a1|a0)=H(a0,a1); no estimator",
        best_selection="validation64 deterministic success desc, then median final distance "
                       "asc, then keep the earlier checkpoint; update0 is a candidate")


def evaluate_panel(executor, policy, instances, config, *, stochastic, seeds=3):
    """Evaluate a fixed panel; stochastic seeds are clustered by instance."""

    state = snapshot(policy)
    jobs = [(instance, state, policy.hidden, seed, not stochastic, config, False)
            for instance in instances for seed in range(seeds if stochastic else 1)]
    episodes = collect(executor, jobs)
    rows = []
    for episode in episodes:
        rows.append(dict(instance=episode["instance"], behaviour_seed=episode["behaviour_seed"],
                         success=bool(episode["success"]), return_=float(episode["return"]),
                         periods=int(episode["periods"]), skills_used=int(episode["skills_used"]),
                         decisions=len(episode["decisions"]),
                         final_goal_distance=float(episode["final_goal_distance"]),
                         short_decisions=int(episode.get("short_decisions", 0)),
                         exits=dict(episode.get("exits", {})),
                         reward_components=dict(episode.get("reward_components", {}))))
    by_instance = defaultdict(list)
    for row in rows:
        by_instance[row["instance"]].append(1.0 if row["success"] else 0.0)
    per_instance = {name: float(np.mean(values)) for name, values in by_instance.items()}
    return dict(success=float(np.mean(list(per_instance.values()))) if per_instance else 0.0,
                return_mean=float(np.mean([row["return_"] for row in rows])) if rows else None,
                median_final_distance=float(np.median([row["final_goal_distance"]
                                                       for row in rows])) if rows else None,
                median_skills=float(np.median([row["skills_used"] for row in rows]))
                if rows else None,
                episodes=len(rows), instances=len(per_instance),
                seeds_per_instance=seeds if stochastic else 1,
                stochastic=bool(stochastic), rows=rows, per_instance=per_instance)


def _reward_component_totals(episodes):
    totals = defaultdict(float)
    for episode in episodes:
        for key, value in episode.get("reward_components", {}).items():
            if isinstance(value, (int, float)):
                totals[key] += float(value)
    return {key: float(value) for key, value in sorted(totals.items())}


def _episode_group_diagnostics(episodes):
    def summarise(group):
        if not group:
            return None
        periods = sum(int(episode["periods"]) for episode in group)
        decisions = sum(len(episode["decisions"]) for episode in group)
        short_decisions = sum(int(episode.get("short_decisions", 0)) for episode in group)
        short_periods = sum(float(episode.get("short_time_fraction", 0.0)) * int(episode["periods"])
                            for episode in group)
        repeated = sum(int(episode.get("repeated_action_pairs", 0)) for episode in group)
        exits = Counter()
        for episode in group:
            exits.update(episode.get("exits", {}))
        durations = np.asarray([int(decision["periods"]) for episode in group
                                for decision in episode["decisions"]], dtype=np.int64)
        skills = np.asarray([int(episode["skills_used"]) for episode in group], dtype=np.int64)
        return dict(episodes=len(group), decisions=decisions,
                    short_decisions=int(short_decisions),
                    short_decision_fraction=(float(short_decisions / decisions)
                                             if decisions else 0.0),
                    short_time_fraction=float(short_periods / periods) if periods else 0.0,
                    repeated_action_pairs=int(repeated),
                    longest_repeat_run=int(max(int(episode.get("longest_repeat_run", 0))
                                               for episode in group)),
                    mean_repeat_run=float(np.mean([float(episode.get("mean_repeat_run", 0.0))
                                                   for episode in group])),
                    skill_periods_p50=float(np.quantile(durations, 0.5)) if durations.size else None,
                    skill_periods_p90=float(np.quantile(durations, 0.9)) if durations.size else None,
                    skill_periods_max=int(durations.max()) if durations.size else None,
                    episode_skills_p50=float(np.quantile(skills, 0.5)) if skills.size else None,
                    episode_skills_p90=float(np.quantile(skills, 0.9)) if skills.size else None,
                    episode_skills_max=int(skills.max()) if skills.size else None,
                    exits=dict(sorted(exits.items())))

    success = [episode for episode in episodes if episode["success"]]
    failure = [episode for episode in episodes if not episode["success"]]
    total_decisions = sum(len(episode["decisions"]) for episode in episodes)
    return {"success": summarise(success), "failure": summarise(failure),
            "batch_episodes": len(episodes), "batch_decisions": total_decisions,
            "success_episode_share": float(len(success) / len(episodes)) if episodes else 0.0,
            "failure_episode_share": float(len(failure) / len(episodes)) if episodes else 0.0,
            "success_decision_share": float(sum(len(episode["decisions"]) for episode in success)
                                            / total_decisions) if total_decisions else 0.0,
            "failure_decision_share": float(sum(len(episode["decisions"]) for episode in failure)
                                            / total_decisions) if total_decisions else 0.0}


def sustained_gate(history, initial_success):
    """The old S4 learnability gate, on the original 8 instances (deterministic).

    This is deliberately NOT the validation-based best-model rule: the S4 gate is a fixed
    reporting item about the eight original instances, while checkpoint selection uses the
    independent validation64 panel.
    """
    points = [row["evaluation"]["deterministic"]["success"] for row in history
              if "evaluation" in row and "deterministic" in row["evaluation"]]
    return dict(points=points,
                passed=bool(len(points) >= 3
                            and all(success >= 0.75 and success - initial_success >= 0.20
                                    for success in points[-3:])),
                rule="original-8 deterministic success >= 0.75 and +20pp over update0 at "
                     "three consecutive evaluation points")


def _compact_best(best):
    if not best:
        return None
    return {key: best[key] for key in ("update", "score", "selection") if key in best}


def stability_monitor(out, manifest, history, best, status, plot=False):
    latest = history[-1] if history else {}
    points = [row for row in history if "evaluation" in row]
    recent = history[-50:]
    initial_path = out / "initial.json"
    initial = json.loads(initial_path.read_text()) if initial_path.exists() else None
    gate = sustained_gate(history, initial["deterministic"]["success"]) if initial else None
    summary = dict(
        status=status, arm=manifest["config"]["arm"],
        critic_mode=manifest["config"]["critic_mode"], update=latest.get("update", 0),
        updates_budget=manifest["config"]["updates"],
        training_episodes=len(history) * manifest["config"]["episodes_per_update"],
        elapsed_hours=(time.time() - manifest["started_unix"]) / 3600,
        deadline_unix=manifest["deadline_unix"], best=_compact_best(best),
        recent_return_mean=float(np.mean([row["return_mean"] for row in recent]))
        if recent else None,
        recent_success=float(np.mean([row["success"] for row in recent])) if recent else None,
        latest_evaluation={key: {"success": value["success"],
                                 "median_final_distance": value["median_final_distance"]}
                           for key, value in points[-1]["evaluation"].items()} if points else None,
        latest_critic=latest.get("critic_after"), sustained_gate=gate)
    atomic_json(out / "monitor.json", summary)
    (out / "README.md").write_text(
        f"# {manifest['config']['arm']} arm ({manifest['config']['critic_mode']})\n\n"
        f"状态：**{status}**；更新 **{summary['update']}/{manifest['config']['updates']}**；"
        f"训练轨迹 **{summary['training_episodes']}**；耗时 {summary['elapsed_hours']:.2f} 小时。\n\n"
        f"critic 对共享表示：**{manifest['config']['critic_mode']}**；"
        "其余网络、优势、优化器、数据顺序与另一臂相同。\n\n"
        "实时：`monitor.json`；逐更新：`updates.jsonl`；评价：`evaluations.jsonl`；"
        "曲线：`curves.png`；恢复：`checkpoints/resume.pt`；最佳：`checkpoints/best.pt`。\n\n"
        "本臂不读取test；最佳模型只按预先固定的validation64规则选择。\n")
    if plot and history:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(4, 1, figsize=(11, 13))
        x = [row["update"] for row in history]
        y = np.asarray([row["return_mean"] for row in history])
        axes[0].plot(x, y, alpha=.25, label="training batch")
        window = min(50, len(y))
        axes[0].plot(x[window - 1:], np.convolve(y, np.ones(window) / window, mode="valid"),
                     label=f"mean {window}")
        axes[0].set_ylabel("return")
        axes[1].plot(x, [row["success"] for row in history], alpha=.25, label="training batch")
        for key in ("deterministic", "stochastic", "train_panel", "validation"):
            available = [row for row in points if key in row["evaluation"]]
            if available:
                axes[1].plot([row["update"] for row in available],
                             [row["evaluation"][key]["success"] for row in available], label=key)
        axes[1].set_ylabel("success")
        axes[2].plot(x, [row["critic_after"]["explained_variance"]
                         if row.get("critic_after", {}).get("explained_variance") is not None
                         else np.nan for row in history], label="critic EV, same batch")
        axes[3].plot(x, [row.get("grad_norm_last", np.nan) for row in history],
                     label="pre-clip grad norm")
        axes[3].plot(x, [row.get("entropy_mean", np.nan) for row in history],
                     label="exact joint entropy")
        for axis in axes:
            axis.legend(); axis.grid(alpha=.2)
        axes[3].set_xlabel("update")
        figure.tight_layout()
        temporary = out / "curves.tmp.png"
        figure.savefig(temporary, dpi=120)
        plt.close(figure)
        temporary.replace(out / "curves.png")
    return summary


def save_stability_checkpoint(path, policy, optimizer, history, initial, best, manifest):
    atomic_save(path, dict(env_version=rl.ENV_VERSION,
                           split_version=rl.layout.SPLIT_VERSION,
                           critic_contract=policy.contract(),
                           panel_contract=manifest["config"]["panel_contract"],
                           config=manifest["config"],
                           update=history[-1]["update"] if history else 0,
                           model=policy.state_dict(), optimizer=optimizer.state_dict(),
                           torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(),
                           history=history, initial=initial, best=best,
                           deadline_unix=manifest["deadline_unix"]))


def load_stability_checkpoint(path, policy, optimizer, manifest):
    state = torch.load(path, map_location="cpu", weights_only=False)
    checks = [
        (state.get("env_version") == rl.ENV_VERSION, "environment version"),
        (state.get("split_version") == rl.layout.SPLIT_VERSION, "split version"),
        (state.get("critic_contract") == policy.contract(), "critic mode / input schema"),
        (state.get("panel_contract") == manifest["config"]["panel_contract"], "panel content"),
        (state.get("config") == manifest["config"], "training configuration"),
        (state.get("deadline_unix") == manifest["deadline_unix"], "original deadline"),
    ]
    for ok, label in checks:
        if not ok:
            raise ValueError(f"Resume rejected: {label} mismatch")
    policy.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["torch_rng"])
    np.random.set_state(state["numpy_rng"])
    return state


def save_weight_checkpoint(path, policy, sidecar, update, selection=None):
    atomic_save(path, policy.state_dict())
    payload = dict(update=update, contract=policy.contract(), selection=selection,
                   **sidecar)
    atomic_json(path.with_suffix(".json"), payload)


def load_fixed_histories():
    if not REPLAY_INITIAL.exists():
        return []
    episodes = torch.load(REPLAY_INITIAL, map_location="cpu", weights_only=False)
    fixed = []
    for episode in episodes:
        if not episode.get("success"):
            continue
        decisions = []
        for decision in episode["decisions"]:
            decision = dict(decision)
            decision["boundary_frame"] = torch.as_tensor(decision["boundary_frame"],
                                                         dtype=torch.float32)
            decision["observations"] = torch.as_tensor(decision["observations"],
                                                       dtype=torch.float32)
            decisions.append(decision)
        fixed.append(dict(episode, decisions=decisions))
    return fixed


def _evaluate_bundle(executor, policy, panels, config, *, validation_stochastic):
    result = {
        "deterministic": evaluate_panel(executor, policy, panels["tracking"], config,
                                        stochastic=False),
        "stochastic": evaluate_panel(executor, policy, panels["tracking"], config,
                                     stochastic=True, seeds=3),
        "train_panel": evaluate_panel(executor, policy, panels["train_panel"], config,
                                      stochastic=False),
        "validation": evaluate_panel(executor, policy, panels["validation"], config,
                                     stochastic=False),
    }
    if validation_stochastic:
        result["validation_stochastic"] = evaluate_panel(
            executor, policy, panels["validation"], config, stochastic=True, seeds=3)
    return result


def _selection_score(validation):
    return [validation["success"], -validation["median_final_distance"]]


def _record_evaluation(out, policy, manifest, index, evaluation, best):
    sidecar = dict(env_version=rl.ENV_VERSION, split_version=rl.layout.SPLIT_VERSION,
                   critic_mode=policy.contract()["critic_mode"],
                   panel_sha256=manifest["config"]["panel_contract"],
                   environment=manifest["config"]["environment"])
    save_weight_checkpoint(out / "checkpoints" / f"update_{index:05d}.pt", policy, sidecar, index)
    score = _selection_score(evaluation["validation"])
    improved = best is None or score > best["score"]
    if improved:
        best = dict(update=index, score=score, selection=(
            "validation64 deterministic success desc, then median final distance asc, "
            "then keep the earlier checkpoint"), evaluation=evaluation)
        save_weight_checkpoint(out / "checkpoints" / "best.pt", policy, sidecar, index, best)
        atomic_json(out / "checkpoints" / "best_selection.json", best)
    atomic_json(out / "evaluations" / f"update_{index:05d}.json",
                dict(update=index, score=score, improved=bool(improved), evaluation=evaluation))
    return best


def stability_train(out, resume=False):
    torch.set_num_threads(1)
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["config"]
    if config.get("split_version") != rl.layout.SPLIT_VERSION:
        raise ValueError("Missing/legacy split version; refusing to train")
    panels = build_stability_panels(verify=True)
    if panel_contract(panels) != config["panel_contract"]:
        raise ValueError("Frozen panel content changed")
    environment = rl.E.EnvConfig()
    if environment.as_dict() != config["environment"]:
        raise ValueError("Frozen environment changed")
    arm = config["arm"]
    if STABILITY_ARMS.get(arm) != config["critic_shared_grad"]:
        raise ValueError("Arm and critic-gradient flag disagree")
    warm_path = out / "checkpoints" / "warm_start.pt"
    if hashlib.sha256(warm_path.read_bytes()).hexdigest() != WARM_START_SHA256:
        raise ValueError("Warm-start weights hash mismatch")
    observation_dim = len(rl.E.RandomStartEnv(panels["train"][0]).observation_fields())
    if observation_dim != config["observation_dim"]:
        raise ValueError("Observation schema changed")
    policy = rl.JointPolicy(observation_dim, hidden=config["hidden"],
                            critic_shared_grad=config["critic_shared_grad"])
    policy.load_state_dict(torch.load(warm_path, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    history, initial, best, start_update = [], None, None, 1
    checkpoint = out / "checkpoints" / "resume.pt"
    fixed = load_fixed_histories()
    baseline_path = out / "checkpoints" / "fixed_history_baseline.pt"
    fixed_baseline = rl.path_scores(policy, fixed) if fixed else []
    atomic_json(out / "pid.json",
                dict(supervisor_pid=os.getppid(), training_pid=os.getpid(),
                     process_group=os.getpgid(0), started=time.time(),
                     status="running"))

    with concurrent.futures.ProcessPoolExecutor(
            max_workers=config["workers"], mp_context=mp.get_context("spawn"),
            initializer=worker_init) as executor:
        if resume:
            state = load_stability_checkpoint(checkpoint, policy, optimizer, manifest)
            history, initial, best = state["history"], state["initial"], state["best"]
            start_update = int(state["update"]) + 1
            log = out / "updates.jsonl"
            if log.exists():
                log.rename(out / f"updates_before_resume_{time.time_ns()}.jsonl")
            log.write_text("".join(json.dumps(row) + "\n" for row in history))
            if baseline_path.exists():
                fixed_baseline = torch.load(baseline_path, map_location="cpu", weights_only=False)
        else:
            save_weight_checkpoint(
                out / "checkpoints" / "initial.pt", policy,
                dict(env_version=rl.ENV_VERSION, split_version=rl.layout.SPLIT_VERSION,
                     critic_mode=policy.contract()["critic_mode"],
                     panel_sha256=config["panel_contract"],
                     environment=config["environment"]), 0)
            initial = _evaluate_bundle(executor, policy, panels, environment,
                                       validation_stochastic=True)
            atomic_json(out / "initial.json", initial)
            torch.save(fixed_baseline, baseline_path)
            best = dict(update=0, score=_selection_score(initial["validation"]),
                        selection="update0 is a best-model candidate: validation64 "
                                  "deterministic success desc, then median distance asc",
                        evaluation=initial)
            save_weight_checkpoint(out / "checkpoints" / "best.pt", policy,
                                   dict(env_version=rl.ENV_VERSION,
                                        split_version=rl.layout.SPLIT_VERSION,
                                        critic_mode=policy.contract()["critic_mode"],
                                        panel_sha256=config["panel_contract"],
                                        environment=config["environment"]), 0, best)
            atomic_json(out / "checkpoints" / "best_selection.json", best)
            save_stability_checkpoint(checkpoint, policy, optimizer, history, initial, best,
                                      manifest)
        (out / "evaluations").mkdir(exist_ok=True)
        stability_monitor(out, manifest, history, best, "running")
        last_monitor = time.monotonic()
        reserve = 120.0
        for index in range(start_update, config["updates"] + 1):
            if time.time() >= manifest["deadline_unix"] - reserve:
                break
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], index]))
            state = snapshot(policy)
            digest = snapshot_hash(state)
            chosen = balanced_indices(config["seed"], index, len(panels["train"]),
                                      config["episodes_per_update"])
            jobs = [(panels["train"][instance_index], state, policy.hidden,
                     int(rng.integers(2 ** 31)), False, environment, False)
                    for instance_index in chosen]
            episodes = collect(executor, jobs)
            if snapshot_hash(snapshot(policy)) != digest:
                raise RuntimeError("Learner changed between snapshot and rollout")
            if any(episode["policy_hash"] != digest for episode in episodes):
                raise RuntimeError("Rollout policy snapshot mismatch")
            for episode in episodes:
                if episode["termination"] in rl.E.ANOMALY_TERMINATIONS:
                    raise RuntimeError(f"Simulation anomaly: {episode['instance']} "
                                       f"{episode['termination']}")
                recomputed = float(sum(decision["reward"] for decision in episode["decisions"]))
                if abs(recomputed - float(episode["return"])) > 1e-6:
                    raise ValueError("Reward recomputation mismatch; refusing to train")
            audit = index % config["audit_every"] == 0
            if audit:
                gradient_report = rl.grouped_gradient_audit(policy, episodes)
                distributions_before = rl.joint_distribution_panel(policy, episodes)
            before = rl.value_fit_metrics(policy, episodes)
            metrics = rl.update(policy, optimizer, episodes)
            if not all(bool(torch.isfinite(parameter).all()) for parameter in policy.parameters()):
                raise FloatingPointError("Non-finite parameters; arm stopped")
            after = rl.value_fit_metrics(policy, episodes)
            metrics.update(
                update=index, policy_hash=digest, critic_before=before, critic_after=after,
                return_mean=float(np.mean([episode["return"] for episode in episodes])),
                success=float(np.mean([episode["success"] for episode in episodes])),
                periods_mean=float(np.mean([episode["periods"] for episode in episodes])),
                physical_seconds_mean=float(np.mean([episode["periods"] for episode in episodes])
                                            * rl.CONTROL_DT),
                decisions=sum(len(episode["decisions"]) for episode in episodes),
                final_distance_mean=float(np.mean([episode["final_goal_distance"]
                                                   for episode in episodes])),
                reward_components=_reward_component_totals(episodes),
                advantage_groups=rl.advantage_group_statistics(episodes),
                diagnostics=_episode_group_diagnostics(episodes),
                simulated_anomalies=0,
                elapsed_seconds=time.time() - manifest["started_unix"])
            if audit:
                metrics["gradient_audit"] = gradient_report
                distributions_after = rl.joint_distribution_panel(policy, episodes)
                metrics["joint_kl"] = rl.joint_kl(distributions_before, distributions_after)
                if fixed:
                    scores = rl.path_scores(policy, fixed)
                    metrics["fixed_history"] = rl.fixed_history_change(fixed_baseline, scores)
                else:
                    metrics["fixed_history"] = []
                metrics["fixed_history_note"] = (
                    "fixed old successful trajectories from diagnosis v4 (initial "
                    "0017/0056/0063); legal recorded observations and actions only; they do "
                    "not enter training or model selection and may not reproduce under v3.1")
            do_eval = (index % config["eval_every"] == 0) or (index == config["updates"])
            if do_eval:
                full = (index % config["full_eval_every"] == 0) or (index == config["updates"])
                metrics["evaluation"] = _evaluate_bundle(
                    executor, policy, panels, environment, validation_stochastic=full)
                best = _record_evaluation(out, policy, manifest, index, metrics["evaluation"],
                                          best)
            history.append(metrics)
            with (out / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            if do_eval:
                with (out / "evaluations.jsonl").open("a") as stream:
                    stream.write(json.dumps(dict(update=index, evaluation=metrics["evaluation"],
                                                 best=_compact_best(best))) + "\n")
            periodic = time.monotonic() - last_monitor >= 1800
            if index % 5 == 0 or periodic:
                save_stability_checkpoint(checkpoint, policy, optimizer, history, initial, best,
                                          manifest)
            summary = stability_monitor(out, manifest, history, best, "running",
                                        plot=periodic or do_eval or index == start_update)
            if periodic or do_eval or index == start_update:
                with (out / "monitor_history.jsonl").open("a") as stream:
                    stream.write(json.dumps(summary) + "\n")
                print(json.dumps({key: summary[key] for key in
                                  ("status", "arm", "update", "best", "latest_evaluation",
                                   "elapsed_hours")}), flush=True)
                last_monitor = time.monotonic()
        # a time-budget stop may fall between evaluation points: evaluate the actual final model
        if history and "evaluation" not in history[-1]:
            index = history[-1]["update"]
            history[-1]["evaluation"] = _evaluate_bundle(
                executor, policy, panels, environment, validation_stochastic=True)
            best = _record_evaluation(out, policy, manifest, index, history[-1]["evaluation"], best)
            with (out / "evaluations.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(update=index,
                                             evaluation=history[-1]["evaluation"],
                                             best=_compact_best(best))) + "\n")
            with (out / "updates.jsonl").open("a") as stream:
                # the last row was already written without the evaluation; write a correction row
                stream.write(json.dumps(dict(update=index, late_evaluation=True,
                                             evaluation=history[-1]["evaluation"])) + "\n")
        save_stability_checkpoint(checkpoint, policy, optimizer, history, initial, best, manifest)
        update_budget_reached = bool(history and history[-1]["update"] == config["updates"])
        status = "update_budget_reached" if update_budget_reached else "time_budget_reached"
        summary = stability_monitor(out, manifest, history, best, status, plot=True)


def setup_stability_root(root):
    root.mkdir(parents=True, exist_ok=True)
    panels = build_stability_panels(verify=True)
    write_stability_panels(root, panels)
    warm_hash = hashlib.sha256(WARM_START.read_bytes()).hexdigest()
    if warm_hash != WARM_START_SHA256:
        raise ValueError("Warm-start checkpoint hash does not match the diagnosis manifest")
    observation_dim = len(rl.E.RandomStartEnv(panels["train"][0]).observation_fields())
    torch.manual_seed(20261019)
    shared = rl.JointPolicy(observation_dim, hidden=rl.HIDDEN, critic_shared_grad=True)
    stopped = rl.JointPolicy(observation_dim, hidden=rl.HIDDEN, critic_shared_grad=False)
    weights = torch.load(WARM_START, map_location="cpu", weights_only=True)
    shared.load_state_dict(weights)
    stopped.load_state_dict(weights)
    probe = torch.randn(1, observation_dim,
                        generator=torch.Generator().manual_seed(20261019))
    with torch.no_grad():
        hidden = shared.encoder(probe)
        table_shared = rl.joint_log_distribution(shared, hidden)
        table_stopped = rl.joint_log_distribution(stopped, hidden)
        value_shared = float(shared.value(hidden))
        value_stopped = float(stopped.value(hidden))
    forward_check = dict(
        observation_dim=observation_dim, parameter_keys_equal=(
            list(shared.state_dict()) == list(stopped.state_dict())),
        value_abs_difference=abs(value_shared - value_stopped),
        joint_table_max_abs_difference=float((table_shared - table_stopped).abs().max()),
        joint_table_sums=[float(table_shared.exp().sum()), float(table_stopped.exp().sum())],
        warm_start_sha256=warm_hash, warm_start=str(WARM_START.relative_to(rl.ROOT)),
        equal=bool(abs(value_shared - value_stopped) < 1e-7
                   and float((table_shared - table_stopped).abs().max()) < 1e-7))
    if not forward_check["equal"]:
        raise ValueError("Arm A/B forward functions differ before training")
    (root / "setup").mkdir(exist_ok=True)
    atomic_json(root / "setup" / "arm_forward_check.json", forward_check)
    sources = [Path(__file__), Path(rl.__file__)]
    sources += sorted((rl.ROOT / "src/coimagine/random_start").glob("*.py"))
    sources += sorted((rl.ROOT / "src/coimagine/continuous2d").glob("*.py"))
    hashes = {}
    for source in sources:
        hashes[str(source.relative_to(rl.ROOT))] = hashlib.sha256(
            source.read_bytes()).hexdigest()
    atomic_json(root / "manifest.json", dict(
        experiment="random_start_rl_stability_long_v1",
        task="docs/tasks/20261019_CoImagine_RL_Stability_LongRun_v1.md",
        split_version=rl.layout.SPLIT_VERSION, env_version=rl.ENV_VERSION,
        panels=panel_contract(panels), panel_content_audit=rl.layout.audit_split_content(
            {split: panels[split] for split in ("train", "validation", "test")}),
        static_evaluation_panels=["train_panel", "validation", "test"],
        arms={name: dict(critic_shared_grad=flag, run_dir=STABILITY_RUN_DIRS[("long", name)],
                         pilot_dir=STABILITY_RUN_DIRS[("pilot", name)])
              for name, flag in STABILITY_ARMS.items()},
        warm_start=dict(path=str(WARM_START.relative_to(rl.ROOT)), sha256=warm_hash,
                        note="weights of the original-8 best update 1550; new AdamW"),
        source_sha256=hashes, torch_version=torch.__version__,
        numpy_version=np.__version__, python=sys.version, platform=platform.platform(),
        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=rl.ROOT,
                                         text=True).strip(),
        note="test64 is built and hash-frozen but is not read until S3 final evaluation"))
    (root / "requirements-freeze.txt").write_text(
        subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
    return panels, forward_check


def stability_entry(args):
    if args.setup_root is not None:
        panels, forward_check = setup_stability_root(args.setup_root)
        print(json.dumps(dict(setup=str(args.setup_root),
                              panels={name: len(panels[name]) for name in PANEL_NAMES},
                              forward_check=forward_check), indent=1))
        return
    if args.stability_child:
        if args.run_dir is None:
            raise SystemExit("--stability-child requires --run-dir")
        stability_train(args.run_dir, args.resume)
        return
    root = args.out_root
    if args.stage is None or args.arm is None:
        raise SystemExit("choose --stage and --arm, or pass --setup-root")
    setup_stability_root(root)
    panels = build_stability_panels(verify=True)
    run_dir = root / STABILITY_RUN_DIRS[(args.stage, args.arm)]
    if not args.resume:
        if run_dir.exists():
            raise FileExistsError(f"Refusing to overwrite {run_dir}; use --resume")
        run_dir.mkdir(parents=True)
        (run_dir / "checkpoints").mkdir()
        (run_dir / "evaluations").mkdir()
        config = stability_run_config(args.arm, args.stage, args.workers, panels)
        if args.updates is not None:
            config["updates"] = int(args.updates)
        if args.hours is not None:
            config["hours_budget"] = float(args.hours)
        started = time.time()
        shutil.copyfile(WARM_START, run_dir / "checkpoints/warm_start.pt")
        sources = [Path(__file__), Path(rl.__file__)]
        sources += sorted((rl.ROOT / "src/coimagine/random_start").glob("*.py"))
        sources += sorted((rl.ROOT / "src/coimagine/continuous2d").glob("*.py"))
        hashes = {}
        for source in sources:
            relative = source.relative_to(rl.ROOT)
            target = run_dir / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = dict(config=config, started_unix=started,
                        deadline_unix=started + config["hours_budget"] * 3600,
                        status="starting", source_sha256=hashes,
                        torch_version=torch.__version__, numpy_version=np.__version__,
                        python=sys.version, platform=platform.platform(),
                        git_head=subprocess.check_output(
                            ["git", "rev-parse", "HEAD"], cwd=rl.ROOT, text=True).strip())
        atomic_json(run_dir / "manifest.json", manifest)
        (run_dir / "requirements-freeze.txt").write_text(
            subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))
        stability_monitor(run_dir, manifest, [], None, "starting")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((rl.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen training source changed: {relative}")
    remaining = manifest["deadline_unix"] - time.time()
    if remaining <= 0:
        raise RuntimeError("Original wall-clock budget exhausted; no automatic extension")
    command = [sys.executable, str(Path(__file__).resolve()), "--stability-child",
               "--run-dir", str(run_dir)]
    if args.resume:
        command.append("--resume")
    with (run_dir / "train.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        process_group = os.getpgid(process.pid)
        manifest.update(supervisor_pid=os.getpid(), training_pid=process.pid)
        atomic_json(run_dir / "manifest.json", manifest)
        atomic_json(run_dir / "pid.json",
                    dict(supervisor_pid=os.getpid(), training_pid=process.pid,
                         process_group=process_group, started=time.time(),
                         status="running"))
        try:
            code = process.wait(timeout=remaining)
            if code == 0:
                manifest["status"] = json.loads(
                    (run_dir / "monitor.json").read_text())["status"]
            elif code < 0:
                manifest["status"] = f"terminated_by_signal_{-code}"
            else:
                manifest["status"] = "failed"
            manifest["exit_code"] = code
            if code != 0:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            manifest["status"] = "time_budget_reached"
            manifest["termination"] = "deadline"
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            manifest["status"] = "supervisor_exception"
            raise
        finally:
            manifest["finished_unix"] = time.time()
            atomic_json(run_dir / "manifest.json", manifest)
            atomic_json(run_dir / "pid.json",
                        dict(supervisor_pid=os.getpid(), training_pid=process.pid,
                             process_group=process_group,
                             started=manifest.get("started_unix"),
                             finished=time.time(),
                             exit_code=manifest.get("exit_code"),
                             status=manifest["status"]))
            if (run_dir / "updates.jsonl").exists():
                history = []
                for line in (run_dir / "updates.jsonl").read_text().splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        break
                    if row.get("late_evaluation"):
                        for previous in reversed(history):
                            if previous["update"] == row["update"]:
                                previous["evaluation"] = row["evaluation"]
                                break
                    else:
                        history.append(row)
                monitor_path = run_dir / "monitor.json"
                best = json.loads(monitor_path.read_text()).get("best") if monitor_path.exists() \
                    else None
                stability_monitor(run_dir, manifest, history, best, manifest["status"], plot=True)
    print(json.dumps(dict(status=manifest["status"], output=str(run_dir))), flush=True)


def train(out, resume=False):
    torch.set_num_threads(1)
    manifest = json.loads((out / "manifest.json").read_text())
    config = manifest["config"]
    if config.get("split_version") != rl.layout.SPLIT_VERSION:
        raise ValueError("Missing/legacy split version: use the historical source snapshot for "
                         "replay; do not silently replace evaluation panels on resume")
    torch.manual_seed(config["seed"])
    expanded = config.get("sampling") == "shuffled_epochs"
    panel, _ = rl.layout.build_split("L0", "train", count=256 if expanded else 64)
    by_name = {instance.name: instance for instance in panel}
    instances = [by_name[name] for name in config["instances"]]
    tracking = [by_name[name] for name in config.get("tracking_instances", config["instances"])]
    train_panel = [by_name[name] for name in config.get("evaluation_train_instances", config["instances"])]
    validation = rl.layout.build_split("L0", "validation", count=64)[0] if expanded else []
    split_audit = rl.layout.audit_split_content({"train": panel, "validation": validation})
    if not split_audit["ok"]:
        raise ValueError(f"Evaluation content overlaps training: {split_audit}")
    panel_contract = dict(split_version=rl.layout.SPLIT_VERSION,
                          train=[x.as_dict() for x in panel],
                          validation=[x.as_dict() for x in validation])
    panel_path = out / "evaluation_panels.json"
    if resume:
        if not panel_path.exists() or json.loads(panel_path.read_text()) != panel_contract:
            raise ValueError("Frozen evaluation panel content missing or changed")
    else:
        atomic_json(panel_path, panel_contract)
    env_config = rl.E.EnvConfig()
    if env_config.as_dict() != config["environment"]:
        raise ValueError("Frozen environment changed")
    policy = rl.JointPolicy(len(rl.E.RandomStartEnv(instances[0]).observation_fields()))
    if config.get("warm_start_sha256"):
        weights = out / "checkpoints/warm_start.pt"
        if hashlib.sha256(weights.read_bytes()).hexdigest() != config["warm_start_sha256"]:
            raise ValueError("Warm-start weights hash mismatch")
        policy.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True))
    optimizer = torch.optim.AdamW(policy.parameters(), lr=rl.LEARNING_RATE, weight_decay=0)
    history, best = [], None
    checkpoint = out / "checkpoints/resume.pt"
    with concurrent.futures.ProcessPoolExecutor(max_workers=config["workers"],
            mp_context=mp.get_context("spawn"), initializer=worker_init) as executor:
        if resume:
            state = load_checkpoint(checkpoint, policy, optimizer, manifest)
            history, initial, best = state["history"], state["initial"], state["best"]
            # JSONL may contain completed-but-uncheckpointed updates. Preserve it as evidence
            # and rebuild the canonical log from the checkpoint before deterministic replay.
            log = out / "updates.jsonl"
            if log.exists():
                log.rename(out / f"updates_before_resume_{time.time_ns()}.jsonl")
            log.write_text("".join(json.dumps(row) + "\n" for row in history))
        else:
            atomic_save(out / "checkpoints/initial.pt", policy.state_dict())
            initial = dict(deterministic=evaluate(executor, policy, tracking, env_config, False),
                           stochastic=evaluate(executor, policy, tracking, env_config, True),
                           uniform=evaluate(executor, policy, tracking, env_config, True, uniform=True))
            if expanded:
                initial["train_panel"] = evaluate(executor, policy, train_panel, env_config, False)
                initial["validation"] = evaluate(executor, policy, validation, env_config, False)
                best = dict(update=0, score=[initial["validation"]["success"],
                                           -initial["validation"]["median_final_distance"]],
                            selection="fixed validation64, deterministic", evaluation=initial)
                atomic_save(out / "checkpoints/best.pt", policy.state_dict())
            atomic_json(out / "initial.json", initial)
            save_checkpoint(checkpoint, policy, optimizer, history, initial, best, manifest)
        monitor(out, manifest, history, best, "running")
        last_monitor = time.monotonic()
        start_update = history[-1]["update"] + 1 if history else 1
        for index in range(start_update, config["updates"] + 1):
            if time.time() >= manifest["deadline_unix"] - 60:
                break  # reserve time for atomic save and pool shutdown; supervisor still has hard deadline
            rng = np.random.default_rng(np.random.SeedSequence([config["seed"], index]))
            state = snapshot(policy)
            digest = snapshot_hash(state)
            jobs = [(instances[int(rng.integers(len(instances)))], state, policy.hidden,
                     int(rng.integers(2 ** 31)), False, env_config, False)
                    for _ in range(config["episodes_per_update"])]
            if expanded:
                jobs = [(instances[i], state, policy.hidden, int(rng.integers(2 ** 31)),
                         False, env_config, False)
                        for i in balanced_indices(config["seed"], index, len(instances),
                                                  config["episodes_per_update"])]
            episodes = collect(executor, jobs)
            if snapshot_hash(snapshot(policy)) != digest or any(e["policy_hash"] != digest for e in episodes):
                raise RuntimeError("Rollout policy snapshot mismatch")
            if index == 1:
                logp_error, value_error = 0.0, 0.0
                with torch.no_grad():
                    for episode in episodes:
                        for decision, hidden in rl.decision_states(policy, episode):
                            logp, _ = policy.evaluate(hidden, *decision["action"])
                            logp_error = max(logp_error, abs(logp.item() - decision["logp"]))
                            value_error = max(value_error, abs(policy.value(hidden).item() - decision["value"]))
                atomic_json(out / "parallel_replay_check.json",
                            dict(logp_max_error=logp_error, value_max_error=value_error))
                if max(logp_error, value_error) > 1e-5:
                    raise RuntimeError("First parallel batch does not replay on the learner")
            before = rl.value_fit_metrics(policy, episodes)
            metrics = rl.update(policy, optimizer, episodes)
            if not all(bool(torch.isfinite(p).all()) for p in policy.parameters()):
                raise FloatingPointError("Non-finite parameters")
            metrics.update(update=index, policy_hash=digest, critic_before=before,
                           critic_after=rl.value_fit_metrics(policy, episodes),
                           return_mean=float(np.mean([e["return"] for e in episodes])),
                           success=float(np.mean([e["success"] for e in episodes])),
                           periods_mean=float(np.mean([e["periods"] for e in episodes])),
                           final_distance_mean=float(np.mean([e["final_goal_distance"] for e in episodes])),
                           elapsed_seconds=time.time() - manifest["started_unix"])
            metrics["episodes"] = [{key: e[key] for key in
                                    ("instance", "behaviour_seed", "success", "return", "periods",
                                     "skills_used", "final_goal_distance", "termination")}
                                   for e in episodes]
            for summary, episode in zip(metrics["episodes"], episodes):
                summary["decision_fraction"] = len(episode["decisions"]) / metrics["decisions"]
            if index % config["eval_every"] == 0:
                metrics["evaluation"] = dict(
                    deterministic=evaluate(executor, policy, tracking, env_config, False),
                    stochastic=evaluate(executor, policy, tracking, env_config, True))
                if expanded:
                    metrics["evaluation"]["train_panel"] = evaluate(executor, policy, train_panel, env_config, False)
                    metrics["evaluation"]["validation"] = evaluate(executor, policy, validation, env_config, False)
                selection = metrics["evaluation"]["validation" if expanded else "deterministic"]
                score = [selection["success"], -selection["median_final_distance"]]
                if best is None or score > best["score"]:
                    best = dict(update=index, score=score, evaluation=metrics["evaluation"])
                    atomic_save(out / "checkpoints/best.pt", policy.state_dict())
                atomic_save(out / "checkpoints" / f"update_{index:05d}.pt", policy.state_dict())
            history.append(metrics)
            with (out / "updates.jsonl").open("a") as stream:
                stream.write(json.dumps(metrics, allow_nan=False) + "\n")
            periodic = time.monotonic() - last_monitor >= 1800
            if index % 5 == 0 or periodic:
                save_checkpoint(checkpoint, policy, optimizer, history, initial, best, manifest)
            summary = monitor(out, manifest, history, best, "running",
                              plot=periodic or "evaluation" in metrics or index == 1)
            if periodic or index == 1:
                with (out / "monitor_history.jsonl").open("a") as stream:
                    stream.write(json.dumps(summary) + "\n")
                print(json.dumps(summary), flush=True)
                last_monitor = time.monotonic()
        save_checkpoint(checkpoint, policy, optimizer, history, initial, best, manifest)
        monitor(out, manifest, history, best,
                "update_budget_reached" if history and history[-1]["update"] == config["updates"]
                else "time_budget_reached", plot=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=rl.ROOT / "outputs/random_start_time_budget_v3_long")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--warm-start-run", type=Path,
                        help="Start a new shuffled 256-instance run from this run's best.pt")
    parser.add_argument("--train", action="store_true", help=argparse.SUPPRESS)
    # -- bounded two-arm stability comparison (task book 2026-10-19) ----------
    parser.add_argument("--stability", action="store_true",
                        help="run the bounded critic-shared-gradient comparison")
    parser.add_argument("--out-root", type=Path, default=STABILITY_ROOT,
                        help="root for the two-arm stability experiment")
    parser.add_argument("--setup-root", type=Path, default=None,
                        help="write panels/contracts under this root and stop")
    parser.add_argument("--arm", choices=sorted(STABILITY_ARMS), default=None)
    parser.add_argument("--stage", choices=("pilot", "long"), default=None)
    parser.add_argument("--hours", type=float, default=None,
                        help="wall-clock budget for this arm/stage (default 2h pilot, 24h long)")
    parser.add_argument("--updates", type=int, default=None,
                        help="update budget for this arm/stage (default 64 pilot, 5000 long)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--stability-child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.stability or args.setup_root is not None or args.stability_child:
        stability_entry(args)
        return
    if args.train:
        train(args.out, args.resume)
        return
    out = args.out
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        (out / "checkpoints").mkdir()
        subset_path = rl.ROOT / "outputs/random_start_reposition_v1/results/diagnostic_subset.json"
        subset = json.loads(subset_path.read_text())
        if len(subset["selected"]) != 8 or not set(subset["selected"]).issubset(subset["successes"]):
            raise ValueError("Expected eight frozen train instances with success evidence")
        started = time.time()
        config = dict(updates=5000, episodes_per_update=8, workers=8, seed=0,
                      split_version=rl.layout.SPLIT_VERSION,
                      eval_every=50, environment=rl.E.EnvConfig().as_dict(),
                      instances=subset["selected"], hidden=rl.HIDDEN, learning_rate=rl.LEARNING_RATE,
                      clip=rl.CLIP_RANGE, value_weight=rl.VALUE_WEIGHT, entropy_weight=rl.ENTROPY_WEIGHT,
                       weight_decay=0, update_epochs=rl.UPDATE_EPOCHS)
        if args.warm_start_run:
            source_manifest = json.loads((args.warm_start_run / "manifest.json").read_text())
            if source_manifest["config"]["environment"] != config["environment"]:
                raise ValueError("Cannot warm-start across environment contracts")
            weights = (args.warm_start_run / "checkpoints/best.pt").read_bytes()
            (out / "checkpoints/warm_start.pt").write_bytes(weights)
            names = [instance.name for instance in rl.layout.build_split("L0", "train", count=256)[0]]
            panel_indices = np.random.default_rng(20261017).choice(256, size=32, replace=False)
            config.update(sampling="shuffled_epochs", instances=names,
                          tracking_instances=subset["selected"],
                          evaluation_train_instances=[names[int(i)] for i in sorted(panel_indices)],
                          validation_instances=64, warm_start_sha256=hashlib.sha256(weights).hexdigest(),
                          warm_start_run=str(args.warm_start_run.resolve()),
                          warm_start_best=json.loads((args.warm_start_run / "monitor.json").read_text())["best"],
                          optimizer_initialization="fresh AdamW; best checkpoint contains weights only")
        sources = [Path(__file__), Path(rl.__file__)]
        sources += list((rl.ROOT / "src/coimagine/random_start").glob("*.py"))
        sources += list((rl.ROOT / "src/coimagine/continuous2d").glob("*.py"))
        hashes = {}
        for source in sources:
            relative = source.relative_to(rl.ROOT)
            target = out / "code" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read_bytes())
            hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
        manifest = dict(config=config, started_unix=started, deadline_unix=started + 86400,
                        status="running", source_sha256=hashes,
                        torch_version=torch.__version__, numpy_version=np.__version__,
                        python=sys.version, platform=platform.platform(), subset=subset,
                        git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=rl.ROOT, text=True).strip())
        atomic_json(out / "manifest.json", manifest)
        (out / "requirements-freeze.txt").write_text(subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True))
        monitor(out, manifest, [], None, "starting_workers_and_initial_evaluation")
    manifest = json.loads((out / "manifest.json").read_text())
    for relative, expected in manifest["source_sha256"].items():
        if hashlib.sha256((rl.ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen training source changed: {relative}")
    remaining = manifest["deadline_unix"] - time.time()
    if remaining <= 0:
        raise RuntimeError("Original 24h deadline exhausted; no automatic extension")
    command = [sys.executable, str(Path(__file__).resolve()), "--train", "--out", str(out)]
    if args.resume:
        command.append("--resume")
    with (out / "train.log").open("a") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        manifest.update(supervisor_pid=os.getpid(), training_pid=process.pid)
        atomic_json(out / "manifest.json", manifest)
        try:
            code = process.wait(timeout=remaining)
            manifest["status"] = (json.loads((out / "monitor.json").read_text())["status"]
                                  if code == 0 else "failed")
            manifest["exit_code"] = code
            if code != 0:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            manifest["status"] = "time_budget_reached"
        finally:
            manifest["finished_unix"] = time.time()
            atomic_json(out / "manifest.json", manifest)
            if (out / "updates.jsonl").exists():
                history = []
                for line in (out / "updates.jsonl").read_text().splitlines():
                    try:
                        history.append(json.loads(line))
                    except json.JSONDecodeError:
                        break  # a hard deadline may interrupt the last log write
                best = json.loads((out / "monitor.json").read_text()).get("best")
                monitor(out, manifest, history, best, manifest["status"], plot=True)
    print(json.dumps(dict(status=manifest["status"], output=str(out))), flush=True)


if __name__ == "__main__":
    main()
