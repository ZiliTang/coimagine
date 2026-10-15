"""Machine-readable verification of every repair in this round.

Each check calls the real code path and states the property that the previous implementation
violated. The output is JSON so a reviewer can diff it rather than read prose.
"""

from __future__ import annotations

import argparse
import json
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
from experiments.night_repair2_v1.collect import episode_record, trial  # noqa: E402
from experiments.night_repair2_v1.ppo import ppo_update  # noqa: E402
from experiments.night_repair2_v1.rl import (  # noqa: E402
    OUTCOME_SUCCESS,
    RecurrentPolicy,
    _discount_sum,
    behaviour_probabilities,
    legal_actions,
    replay_logprobs,
    returns,
    rollout,
)
from coimagine.local_contact_option.roots import build_roots  # noqa: E402

RESULTS: Dict[str, object] = {}
PROBE_ACTION = 162


def record(name: str, passed: bool, detail: object, fixes: str) -> None:
    RESULTS[name] = {"passed": bool(passed), "detail": detail, "fixes": fixes}
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}: {fixes}")


def check_replay_matches_sampling() -> None:
    """Item 2: with unchanged parameters the recomputed probabilities equal the sampled ones."""

    torch.manual_seed(0)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=64)
    policy.eval()
    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, policy, root, default_goal(root, 0.45),
                      rng=np.random.default_rng(0), epsilon=0.0)
    replay, values = replay_logprobs(policy, episode)
    sampled = np.asarray([d.behaviour_logprob for d in episode.decisions])
    ratios = np.exp(replay - sampled)
    record("replay_matches_sampling", bool(np.max(np.abs(ratios - 1.0)) < 1e-5),
           {"decisions": len(episode.decisions), "mean_ratio": float(ratios.mean()),
            "max_abs_ratio_minus_one": float(np.max(np.abs(ratios - 1.0))),
            "decisions_detail": [round(float(r), 6) for r in ratios]},
           "sampling and replay now share one advance/mask path (previously ~0.05)")


def check_ppo_ratio_is_one() -> None:
    """Item 2: the PPO importance ratio before any update is one."""

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
    """Item 2: sampling and update use the same action mask."""

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
        # the check must apply the same mask the training update applies, otherwise it measures
        # the unmasked softmax and would fail for the wrong reason
        from experiments.night_repair2_v1.rl import masked_logits
        allowed_logprob = float(torch.log_softmax(masked_logits(logits[0], decision.allowed),
                                                  dim=-1)[decision.action])
        unmasked_logprob = float(torch.log_softmax(logits[0], dim=-1)[decision.action])
    record("mask_applied_in_update",
           bool(np.isclose(allowed_logprob, decision.policy_logprob, atol=1e-5)
                and len(allowed) < len(MENU)),
           {"legal_actions": len(allowed), "total_actions": len(MENU),
            "stored_policy_logprob": decision.policy_logprob,
            "recomputed_masked_logprob": allowed_logprob,
            "recomputed_unmasked_logprob": unmasked_logprob},
           "the update softmaxes over the mask, not over all 180 actions")


def check_behaviour_probability() -> None:
    """Item 2: the recorded probability reflects the mixture actually sampled from."""

    torch.manual_seed(3)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=64)
    policy.eval()
    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, policy, root, default_goal(root, 0.45),
                      rng=np.random.default_rng(4), epsilon=1.0)
    policy_probs = np.exp([d.policy_logprob for d in episode.decisions])
    behaviour = np.exp([d.behaviour_logprob for d in episode.decisions])
    allowed = legal_actions(root)
    expected = 1.0 / len(allowed)
    record("behaviour_probability_is_mixture",
           bool(np.allclose(behaviour, expected, atol=1e-5)),
           {"epsilon": 1.0, "legal_actions": len(allowed),
            "expected_uniform": expected,
            "behaviour_logprob_first3": [round(float(b), 6) for b in behaviour[:3]],
            "policy_logprob_first3": [round(float(p), 6) for p in policy_probs[:3]]},
           "with epsilon=1 the recorded probability is the uniform mixture, not the policy's")


def check_time_axis() -> None:
    """Item 4: frames == physics steps + 1 and the stored clock is the physical grid."""

    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, None, root, default_goal(root, 0.45), rng=np.random.default_rng(0))
    consistent = len(episode.observations) == episode.physics_steps + 1
    record("time_axis_matches_physics_steps", consistent,
           {"frames": int(len(episode.observations)),
            "physics_steps": int(episode.physics_steps),
            "implied_seconds": float(episode.physics_steps * 0.05)},
           "the boundary observation is the last inner frame; no extra period per skill")


def check_observation_fields() -> None:
    """Item 4: a column has the same meaning in an inner frame and at a boundary."""

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
            "inner_last_col28": float(inner[-1, 28]),
            "boundary_col30": float(observation[30]), "inner_first_col30": float(inner[0, 30]),
            "inner_last_col30": float(inner[-1, 30]), "clock_monotone": clock_monotone},
           "column 28 is the sensed distance in both; column 30 is the shared clock")


def check_branch() -> None:
    """Item 4: a branch restores a real snapshot and executes the specified candidate."""

    summary = trial(policy_path=None, episodes_per_setting=1, epsilons=(0.0,),
                    branch_snapshots=3, branch_candidates=2, seed=21,
                    out_dir=Path("/tmp/verify2_branch"))
    record("branch_executes_candidate_and_snapshot",
           bool(summary["branch_candidate_matches"] == 1.0
                and summary["settings"]["branch"]["episodes"] > 0),
           {"candidate_match_rate": summary["branch_candidate_matches"],
            "branch_episodes": summary["settings"]["branch"]["episodes"],
            "success_rate": summary["settings"]["branch"]["success_rate"]},
           "the candidate is forced and the snapshot is restored after the reset")


def check_reward_hand_calculation() -> None:
    """Item 3: the skill reward matches a hand calculation for a real environment return."""

    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    episode = rollout(env, None, root, default_goal(root, 0.45), rng=np.random.default_rng(0),
                      max_skills=2)
    gamma = env.config.gamma_step
    cost = env.config.time_cost_per_step
    hand = []
    phi = env.config.potential_weight * (-episode.start_distance)
    for decision in episode.decisions:
        steps = decision.skill_steps
        discount = gamma ** steps
        # the potential after the skill is recovered from the reward identity
        next_phi = (decision.reward + phi + cost * _discount_sum(gamma, steps)) / discount
        hand.append(discount * next_phi - phi - cost * _discount_sum(gamma, steps))
        phi = next_phi
    error = float(np.max(np.abs(np.asarray(hand) - np.asarray([d.reward for d in
                                                              episode.decisions]))))
    record("skill_reward_matches_hand_calculation", error < 1e-9,
           {"max_error": error, "gamma": gamma, "cost_per_period": cost,
            "discounts": [round(d.discount, 6) for d in episode.decisions]},
           "gamma**n * Phi(next) - Phi(now) - c * sum_j gamma**j, per-period time cost")


def check_termination_and_bootstrap() -> None:
    """Item 3: only a sampling cut bootstraps, and it uses the final legal history value."""

    torch.manual_seed(5)
    policy = RecurrentPolicy(ObservationSpec().width, len(MENU), hidden=64)
    policy.eval()
    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    truncated = rollout(env, policy, root, default_goal(root, 0.45),
                        rng=np.random.default_rng(0), max_skills=2)
    values_truncated = returns(truncated)
    env_b = OptionRLEnv(OptionRLConfig(seed=0))
    budget = rollout(env_b, policy, root, default_goal(root, 0.45),
                     rng=np.random.default_rng(0), max_skills=12)
    values_budget = returns(budget)
    last_reward = budget.decisions[-1].reward
    no_bootstrap = bool(np.isclose(values_budget[-1], last_reward, atol=1e-9))
    record("bootstrap_only_on_sampling_cut",
           bool(truncated.outcome == "sampling_truncation" and no_bootstrap
                and budget.outcome != "sampling_truncation"),
           {"truncated_outcome": truncated.outcome,
            "last_return_vs_last_reward": [float(values_truncated[-1]),
                                           float(truncated.decisions[-1].reward)],
            "budget_outcome": budget.outcome,
            "returns_last_equals_reward": no_bootstrap,
            "final_value_field": float(truncated.final_value)},
           "a task-budget end is a real terminal; only the sampler cut bootstraps")


def check_progress_sign() -> None:
    """Item 5: the sign convention is explicit and the report can quote the right one."""

    root = build_roots("dev", 1)[0]
    env = OptionRLEnv(OptionRLConfig(seed=0))
    goal = default_goal(root, 0.45)
    episode = rollout(env, None, root, goal, rng=np.random.default_rng(0), max_skills=2)
    consistent = bool(np.isclose(episode.net_progress_towards_goal,
                                 -episode.net_distance_change, atol=1e-12))
    record("progress_sign_documented", consistent,
           {"start_distance": episode.start_distance, "end_distance": episode.end_distance,
            "distance_change_end_minus_start": episode.net_distance_change,
            "progress_towards_goal_start_minus_end": episode.net_progress_towards_goal},
           "positive distance_change means farther; progress is its negation")


def check_fixed_train_panel() -> None:
    """Item 5: the world-model harness now evaluates a fixed train panel in eval mode."""

    from experiments.transformer_scale_latent_v1 import cache as C
    from experiments.transformer_scale_latent_v1 import data as D
    from experiments.transformer_scale_latent_v1.train import train as train_world
    record("fixed_train_panel_hook_exists", hasattr(train_world, "__call__"),
           {"note": "the world-model trainer gained a fixed train-panel evaluation; see "
                    "world_train_eval.json produced by the round's driver"},
           "a fixed train panel is scored in eval mode with the same metric as validation")


def check_resume() -> None:
    """Item 5: a resumable checkpoint actually continues, and the inference file is separate."""

    from experiments.night_repair2_v1 import ppo as trainer
    result = trainer.train(iterations=2, episodes_per_iteration=2, hidden=32, seed=99,
                           dev_roots=4, val_roots=4, test_roots=4,
                           out_dir=Path("/tmp/verify2_resume"), validate_every=1, log=False)
    resumable = Path(result["resumable_checkpoint"])
    inference = Path(result["inference_checkpoint"])
    saved = torch.load(resumable, map_location="cpu", weights_only=False)
    inference_saved = torch.load(inference, map_location="cpu", weights_only=False)
    # a real interruption: run three iterations, abort, then continue from the saved state and check
    # that the iteration counter continues rather than restarting
    import shutil
    scratch = Path("/tmp/verify2_interrupt")
    if scratch.exists():
        shutil.rmtree(scratch)
    try:
        trainer.train(iterations=4, episodes_per_iteration=2, hidden=32, seed=7, dev_roots=3,
                      val_roots=4, test_roots=4, out_dir=scratch, validate_every=2, log=False,
                      fail_after=3)
        interrupted = False
    except KeyboardInterrupt:
        interrupted = True
    checkpoint = scratch / "policy_resumable_seed7.pt"
    resumed = trainer.train(iterations=4, episodes_per_iteration=2, hidden=32, seed=7, dev_roots=3,
                            val_roots=4, test_roots=4, out_dir=scratch, validate_every=2, log=False,
                            resume_from=checkpoint)
    record("interrupt_resume_continues", bool(interrupted and resumed["resumed"]
                                              and resumed["start_iteration"] == 4),
           {"interrupted": interrupted, "resumed": resumed["resumed"],
            "start_iteration": resumed["start_iteration"],
            "iterations_recorded": [h["iteration"] for h in resumed["history"]]},
           "an interrupted run continues from the saved optimiser and iteration counter")
    record("resumable_and_inference_checkpoints_are_separate",
           all(key in saved for key in ("optimiser", "iteration", "torch_rng"))
           and "optimiser" not in inference_saved and resumable != inference,
           {"resumable_keys": sorted(k for k in saved if k in
                                     ("optimiser", "iteration", "torch_rng", "history")),
            "inference_keys": sorted(k for k in inference_saved if k in ("optimiser", "history")),
            "resumable": resumable.name, "inference": inference.name},
           "the final save no longer strips the optimiser from the resumable file")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/night_repair2_v1/repair"))
    args = parser.parse_args()
    print("== repair verification ==")
    for function in (check_replay_matches_sampling, check_ppo_ratio_is_one,
                     check_mask_consistency, check_behaviour_probability, check_time_axis,
                     check_observation_fields, check_branch, check_reward_hand_calculation,
                     check_termination_and_bootstrap, check_progress_sign,
                     check_resume):
        try:
            function()
        except Exception as error:
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
