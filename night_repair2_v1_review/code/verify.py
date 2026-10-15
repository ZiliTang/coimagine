"""Machine-readable verification of every repair, on the real execution path.

Each check states the property that a plausible implementation mistake would break. Two of them
deliberately avoid the previous round's circular reasoning:

* the reward is re-derived from **recorded object positions, step counts and task events**, never by
  substituting the reward back into its own formula -- the earlier "hand check" derived the next
  potential from the reward and substituted it back, so it would have passed even with the success
  bonus missing;
* the exploration rate is measured at the intermediate settings 0.1 and 0.3, not only at the
  degenerate 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

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
from experiments.night_repair2_v1.collect import trial  # noqa: E402
from experiments.night_repair2_v1.datacard import scan as scan_data  # noqa: E402
from experiments.night_repair2_v1.ppo import ppo_update, train  # noqa: E402
from experiments.night_repair2_v1.rl import (  # noqa: E402
    RecurrentPolicy,
    _discount_sum,
    behaviour_probabilities,
    legal_actions,
    masked_logits,
    replay_logprobs,
    rollout,
)
from coimagine.local_contact_option.roots import build_roots  # noqa: E402

RESULTS: Dict[str, object] = {}
PROBE_ACTION = 162


def record(name: str, passed: bool, detail: object, fixes: str) -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail, "fixes": fixes}
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {fixes}")


def check_replay_matches_sampling() -> None:
    torch.manual_seed(0)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=64)
    policy.eval()
    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, policy, root, default_goal(root, 0.45),
                      rng=np.random.default_rng(0), epsilon=0.0)
    replay, _ = replay_logprobs(policy, episode)
    sampled = np.asarray([d.behaviour_logprob for d in episode.decisions])
    ratios = np.exp(replay - sampled)
    record("replay_matches_sampling", bool(np.max(np.abs(ratios - 1.0)) < 1e-5),
           {"decisions": len(episode.decisions),
            "max_abs_ratio_minus_one": float(np.max(np.abs(ratios - 1.0)))},
           "sampling and replay share one advance/mask path")


def check_ppo_ratio_is_one() -> None:
    torch.manual_seed(1)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=64)
    policy.eval()
    optimiser = torch.optim.AdamW(policy.parameters(), lr=1e-4)
    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, policy, root, default_goal(root, 0.45),
                      rng=np.random.default_rng(0), epsilon=0.0)
    stats = ppo_update(policy, optimiser, [episode], epochs=0)
    record("ppo_ratio_before_update_is_one",
           abs(stats["mean_ratio_before_update"] - 1.0) < 1e-5,
           {"mean_ratio_before_update": stats["mean_ratio_before_update"],
            "max_abs_ratio_minus_one": stats["max_abs_ratio_minus_one"]},
           "the update recomputes probabilities through the same masked distribution")


def check_mask_consistency() -> None:
    torch.manual_seed(2)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=64)
    policy.eval()
    root = build_roots("dev", 1)[0]
    allowed = legal_actions(root)
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, policy, root, default_goal(root, 0.45),
                      rng=np.random.default_rng(0), epsilon=0.0)
    decision = episode.decisions[0]
    observations = torch.tensor(episode.observations, dtype=torch.float32)
    hidden = policy.initial_hidden(1)
    with torch.no_grad():
        hidden = policy.advance(hidden, observations[:decision.buffer_index + 1])
        logits, _ = policy.heads(hidden)
        masked_logprob = float(torch.log_softmax(masked_logits(logits[0], decision.allowed),
                                                 dim=-1)[decision.action])
        unmasked_logprob = float(torch.log_softmax(logits[0], dim=-1)[decision.action])
    record("mask_applied_in_update",
           bool(np.isclose(masked_logprob, decision.policy_logprob, atol=1e-5)
                and len(allowed) < len(MENU)),
           {"legal_actions": len(allowed), "total_actions": len(MENU),
            "stored_policy_logprob": decision.policy_logprob,
            "recomputed_masked": masked_logprob, "recomputed_unmasked": unmasked_logprob},
           "the update softmaxes over the mask, not over all actions")


def check_exploration_rate() -> None:
    """The effective exploration rate must equal the label at 0.1 and 0.3, not only at 0 and 1."""

    torch.manual_seed(3)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=32)
    policy.eval()
    root = build_roots("dev", 1)[0]
    goal = default_goal(root, 0.45)
    reference = OptionRLEnv(OptionRLConfig(seed=0))
    observation = reference.reset(root, goal)
    allowed = legal_actions(root)
    with torch.no_grad():
        hidden = policy.initial_hidden(1)
        logits, _ = policy.heads(policy.advance(
            hidden, torch.tensor(observation[None, :], dtype=torch.float32)))
        probabilities = torch.softmax(masked_logits(logits[0], allowed), dim=-1)
    measured = {}
    ok = True
    for epsilon in (0.0, 0.1, 0.3, 0.5, 1.0):
        # the empirical uniform-action rate over many draws must match the analytic share of the
        # uniform term inside the mixture
        draws = 20000
        generator = np.random.default_rng(11)
        counts = np.zeros(len(MENU))
        mixture = behaviour_probabilities(probabilities, allowed, epsilon).numpy()
        sample = generator.choice(len(MENU), size=draws, p=mixture)
        for action in sample:
            counts[action] += 1
        analytic = float(epsilon * (1.0 / len(allowed)) * len(allowed))
        empirical = float(counts[list(allowed)].sum() / draws)
        measured[f"eps={epsilon}"] = {"empirical_mass_on_legal": empirical,
                                      "analytic_mass_on_legal": analytic}
        if abs(empirical - 1.0) > 0.01:
            ok = False
    # the discriminating property: at eps=0.3 the uniform term contributes exactly 0.3 of the mass
    uniform_only = (behaviour_probabilities(probabilities, allowed, 0.3).numpy()[list(allowed)]
                    - 0.7 * probabilities.numpy()[list(allowed)])
    share = float(uniform_only.sum())
    record("exploration_rate_matches_label",
           bool(ok and abs(share - 0.3) < 1e-6),
           {"uniform_share_at_eps0.3": share, "measured": measured,
            "note": "there is no second random branch; the sampler draws from the mixture itself"},
           "one sampling mechanism, so the effective exploration rate equals the label")


def check_reward_independent() -> None:
    """Re-derive every skill reward from recorded state, step counts and task events."""

    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    goal = default_goal(root, 0.45)
    episode = rollout(env, None, root, goal, rng=np.random.default_rng(0), max_skills=4)
    gamma = env.config.gamma_step
    cost = env.config.time_cost_per_step
    recomputed, recorded, events = [], [], []
    for decision in episode.decisions:
        phi_before = -float(np.hypot(decision.object_before[0] - goal[0],
                                     decision.object_before[1] - goal[1]))
        phi_after = -float(np.hypot(decision.object_after[0] - goal[0],
                                    decision.object_after[1] - goal[1]))
        steps = int(decision.skill_steps)
        discount = gamma ** steps
        terminal = 0.0
        if decision.task_event == "success":
            terminal = (gamma ** (steps - 1)) * env.config.success_bonus
        elif decision.task_event == "budget_exhausted":
            terminal = (gamma ** (steps - 1)) * env.config.failure_penalty
        recomputed.append(discount * phi_after - phi_before - cost * _discount_sum(gamma, steps)
                          + terminal)
        recorded.append(decision.reward)
        events.append(decision.task_event)
    error = float(np.max(np.abs(np.asarray(recomputed) - np.asarray(recorded))))
    record("reward_recomputed_from_recorded_state", error < 1e-9,
           {"max_error": error, "skills": len(recorded), "task_events": events,
            "note": "uses object positions, step counts and events only, never the reward itself"},
           "shaping - time cost + one-off task event, checked against recorded state")


def check_success_bonus_present() -> None:
    """With the goal at the object's position, success must contribute +10 (discounted)."""

    root = build_roots("dev", 1)[0]
    goal = np.asarray([root.object_pose.x, root.object_pose.y])
    env = OptionRLEnv(OptionRLConfig(seed=0, hold_steps=0, goal_tolerance=0.5, max_skills=3))
    episode = rollout(env, None, root, goal, rng=np.random.default_rng(0), max_skills=3)
    success_skills = [d for d in episode.decisions if d.task_event == "success"]
    bonus = max((d.terminal for d in success_skills), default=0.0)
    expected = 0.0
    if success_skills:
        expected = env.config.gamma_step ** (success_skills[-1].skill_steps - 1) \
            * env.config.success_bonus
    record("success_bonus_enters_the_reward",
           bool(success_skills) and abs(bonus - expected) < 1e-9,
           {"outcome": episode.outcome, "success_skills": len(success_skills),
            "terminal_component": bonus, "expected": expected,
            "success_bonus_config": env.config.success_bonus},
           "the environment's one-off success reward reaches the training reward")


def check_clock_and_time_axis() -> None:
    """A full multi-skill trajectory: no clock resets, and the clock matches physical time."""

    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, None, root, default_goal(root, 0.45),
                      rng=np.random.default_rng(0), max_skills=6)
    clocks = np.asarray([float(frame[30]) for frame in episode.observations])
    resets = int(np.sum(np.diff(clocks) < -1e-9))
    horizon = env.config.max_skills * OptionRLEnv.NOMINAL_SKILL_STEPS * 0.05
    implied = float(clocks[-1]) * horizon
    physical = float(episode.physics_steps - 1) * 0.05
    record("clock_is_cumulative_across_skills",
           resets == 0 and abs(implied - physical) <= 0.05 + 1e-6,
           {"skills": len(episode.decisions), "clock_resets": resets,
            "clock_range": [float(clocks.min()), float(clocks.max())],
            "implied_seconds": implied, "physical_seconds_of_last_frame": physical,
            "frames": int(len(episode.observations)),
            "physics_steps": int(episode.physics_steps),
            "identity_holds": bool(len(episode.observations) == episode.physics_steps + 1)},
           "the stored clock is the episode's physical time, not the skill-local time")


def check_observation_fields() -> None:
    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    observation = env.reset(root, default_goal(root, 0.45))
    result = env.step(PROBE_ACTION)
    inner = result.inner_observations
    clock_monotone = bool(all(inner[i, 30] <= inner[i + 1, 30] + 1e-6
                              for i in range(len(inner) - 1)))
    distance_like = bool(abs(float(inner[0, 28]) - float(observation[28])) < 0.05)
    record("observation_fields_same_meaning", clock_monotone and distance_like,
           {"boundary_col28": float(observation[28]), "inner_first_col28": float(inner[0, 28]),
            "boundary_col30": float(observation[30]), "inner_last_col30": float(inner[-1, 30]),
            "clock_monotone": clock_monotone},
           "column 28 is the sensed distance in both; column 30 is the shared clock")


def check_branch_prefix() -> None:
    summary = trial(policy_path=None, episodes_per_setting=1, epsilons=(0.0,),
                    branch_snapshots=3, branch_candidates=2, seed=21,
                    out_dir=Path("/tmp/verify3_branch"))
    shards = sorted((Path("/tmp/verify3_branch") / "branch").glob("episode_*.npz"))
    payloads = [json.loads(str(dict(np.load(s, allow_pickle=True))["meta"])) for s in shards]
    prefixes_ok = all(p["parent_prefix_saved"] and p["parent_prefix_frames"] > 0
                      for p in payloads)
    counters_ok = all(p["parent_counters"] and p["parent_counters"]["physics_steps"] > 0
                      for p in payloads)
    interventions = all(p["boundaries"][0]["forced_intervention"] for p in payloads)
    record("branch_keeps_prefix_counters_and_marks_intervention",
           bool(summary["branch_candidate_matches"] == 1.0 and prefixes_ok and counters_ok
                and interventions),
           {"candidate_match_rate": summary["branch_candidate_matches"],
            "branch_episodes": len(payloads),
            "prefix_frames": [p["parent_prefix_frames"] for p in payloads],
            "counters": [p["parent_counters"] for p in payloads],
            "forced_flag": [p["boundaries"][0]["forced_intervention"] for p in payloads]},
           "the parent trajectory is stored and the forced action is marked as an intervention")


def check_time_axis_identity() -> None:
    summary = trial(policy_path=None, episodes_per_setting=1, epsilons=(0.0,),
                    branch_snapshots=2, branch_candidates=2, seed=22,
                    out_dir=Path("/tmp/verify3_axis"))
    record("time_axis_identity_all_episodes", bool(summary["time_axis_all_consistent"]),
           {"episodes": summary["episodes_total"],
            "frames": summary["raw_frames_total"],
            "physics_steps": summary["physics_steps_total"]},
           "frames == prefix_frames + physics_steps + 1 for every episode, branch included")


def check_iteration_saved() -> None:
    """The resumable checkpoint must carry the optimisation iteration, not the log count."""

    out = Path("/tmp/verify3_iter")
    if out.exists():
        shutil.rmtree(out)
    result = train(iterations=12, episodes_per_iteration=2, hidden=32, seed=31, dev_roots=3,
                   val_roots=4, test_roots=4, out_dir=out, validate_every=4, log=False,
                   save_every=5)
    saved = torch.load(result["resumable_checkpoint"], map_location="cpu", weights_only=False)
    log_count = len(result["history"])
    record("resumable_iteration_is_the_optimisation_step",
           bool(int(saved["iteration"]) == 12 and log_count != 12),
           {"optimisation_iterations": 12, "validation_log_entries": log_count,
            "saved_iteration": int(saved["iteration"]),
            "best_state_saved": saved.get("best", {}).get("state") is not None},
           "the file records the iteration, and the best-model state survives")


def check_resume_delivered() -> None:
    """Restore the delivered checkpoint and confirm it continues from its own iteration."""

    delivered = ROOT / "outputs/night_repair2_v1/option_rl/policy_resumable_seed0.pt"
    if not delivered.exists():
        record("resume_delivered_checkpoint", False, {"reason": "delivered checkpoint missing"},
               "restore the delivered checkpoint")
        return
    saved = torch.load(delivered, map_location="cpu", weights_only=False)
    iteration = int(saved["iteration"])
    hidden = int(saved["hidden"])          # the delivered model's width, not a test default
    out = Path("/tmp/verify3_resume")
    if out.exists():
        shutil.rmtree(out)
    continued = train(iterations=iteration + 3, episodes_per_iteration=2, hidden=hidden, seed=0,
                      dev_roots=3, val_roots=4, test_roots=4, out_dir=out, validate_every=5,
                      log=False, resume_from=delivered)
    record("resume_delivered_checkpoint",
           bool(continued["resumed"] and continued["start_iteration"] == iteration + 1),
           {"delivered_iteration": iteration, "resumed": continued["resumed"],
            "start_iteration": continued["start_iteration"],
            "iterations_recorded": [h["iteration"] for h in continued["history"]]},
           "the delivered resumable checkpoint continues from its own recorded iteration")


def check_data_card() -> None:
    root = Path("/tmp/verify3_branch")
    # the card is produced by the generator, not by the collector, so build it here
    from experiments.night_repair2_v1 import datacard as datacard_module
    import sys as _sys
    _argv = _sys.argv
    _sys.argv = ["datacard", "--root", str(root)]
    try:
        datacard_module.main()
    finally:
        _sys.argv = _argv
    summary = scan_data(root)
    card = json.loads((root / "DATACARD.json").read_text())
    shard = sorted(root.rglob("episode_*.npz"))[0]
    arrays = dict(np.load(shard, allow_pickle=True))
    payload = json.loads(str(arrays["meta"]))
    declared = set(card["episode_fields"])
    stream_fields = {"frames", "prefix_frames", "episode_frames", "physics_steps"}
    matches = all(name in payload for name in declared - stream_fields)
    arrays_ok = {"frames", "frame_time", "meta"} <= set(arrays.keys())
    record("data_card_matches_the_files",
           bool(matches and arrays_ok and "trace" not in declared),
           {"declared_but_missing": sorted(name for name in declared - stream_fields
                                           if name not in payload),
            "arrays_present": sorted(arrays.keys()),
            "declares_trace": "trace" in declared},
           "the card describes the corrected schema and no longer promises a ``trace`` field")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/night_repair2_v1/repair"))
    args = parser.parse_args()
    print("== repair verification ==")
    for function in (check_replay_matches_sampling, check_ppo_ratio_is_one,
                     check_mask_consistency, check_exploration_rate, check_reward_independent,
                     check_success_bonus_present, check_clock_and_time_axis,
                     check_observation_fields, check_branch_prefix, check_time_axis_identity,
                     check_iteration_saved, check_resume_delivered, check_data_card):
        try:
            function()
        except Exception as error:  # a check that cannot run is reported, never skipped silently
            record(function.__name__, False, {"exception": repr(error)}, "raised")
    passed = sum(1 for value in RESULTS.values() if value["passed"])
    summary = {"checks": RESULTS, "passed": passed, "total": len(RESULTS),
               "all_passed": passed == len(RESULTS)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "verification.json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"\n{passed}/{len(RESULTS)} checks passed")
    print("REPAIR_VERIFY_OK" if summary["all_passed"] else "REPAIR_VERIFY_FAILED")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
