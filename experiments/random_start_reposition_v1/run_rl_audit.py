"""Bounded training-semantics audit; no reward changes or policy training campaign.

Two frozen diagnostic instances, one policy initialisation, two complete rollouts, then
20 critic-only passes on that SAME fixed data. This tests fitting, not learnability or
generalisation. Historical outputs and checkpoints are never loaded or overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

import rl


def values(policy, episodes, legacy=False):
    predictions = []
    with torch.no_grad():
        for episode in episodes:
            if legacy:
                hidden = policy.initial_hidden()
                for decision in episode["decisions"]:
                    hidden = policy.advance(decision["boundary_frame"], hidden)
                    hidden = policy.advance(decision["observations"], hidden)
                    predictions.append(policy.value(hidden).item())
            else:
                for _decision, hidden in rl.decision_states(policy, episode):
                    predictions.append(policy.value(hidden).item())
    return np.asarray(predictions)


def legacy_decision_states(policy, episode):
    """Reproduce old critic timing and its untruncated graph, for this audit only."""
    hidden = policy.initial_hidden()
    for decision in episode["decisions"]:
        hidden = policy.advance(decision["boundary_frame"], hidden)
        hidden = policy.advance(decision["observations"], hidden)
        yield decision, hidden


def fit_metrics(predictions, targets):
    residual = targets - predictions
    variance = float(np.var(targets))
    return {"mse": float(np.mean(residual ** 2)),
            "mae": float(np.mean(np.abs(residual))),
            "explained_variance": float(1.0 - np.var(residual) / variance)
            if variance > 1e-12 else None,
            "target_variance": variance}


def main():
    if rl.E.EnvConfig().gamma != 0.997:
        raise RuntimeError("Historical v1 discount/critic audit requires its recorded v1 code snapshot")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path,
                        default=rl.ROOT / "outputs/random_start_rl_audit_v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / "audit.json"
    if report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {report_path}; choose a new --out")
    torch.set_num_threads(1)
    torch.manual_seed(0)
    start = time.time()
    panel, _ = rl.layout.build_split("L0", "train", count=32)
    selected = [panel[5], panel[10]]
    policy = rl.JointPolicy(len(rl.E.RandomStartEnv(selected[0]).observation_fields()))
    episodes = [rl.rollout(rl.E.RandomStartEnv(instance), policy,
                           generator=torch.Generator().manual_seed(100 + index))
                for index, instance in enumerate(selected)]
    entries = rl.flatten(episodes)["entries"]
    targets = np.asarray([entry["target"] for entry in entries])
    recorded = np.asarray([entry["value"] for entry in entries])
    corrected = values(policy, episodes)
    legacy = values(policy, episodes, legacy=True)
    smoke_policy = copy.deepcopy(policy)
    frozen_heads = {name: parameter.detach().clone()
                    for name, parameter in smoke_policy.named_parameters()}
    zero_lr_metrics = rl.update(smoke_policy,
                                torch.optim.AdamW(smoke_policy.parameters(), lr=0.0), episodes)
    unchanged = all(torch.equal(parameter, frozen_heads[name])
                    for name, parameter in smoke_policy.named_parameters())
    curves = {}
    for mode in ("corrected", "legacy"):
        model = copy.deepcopy(policy)
        optimizer = torch.optim.AdamW(model.parameters(), lr=rl.LEARNING_RATE)
        curve = [{"epoch": 0, **fit_metrics(values(model, episodes), targets)}]
        replay = rl.decision_states if mode == "corrected" else legacy_decision_states
        # Same data, initial weights, targets and optimiser. Always evaluate at deployment's
        # PRE-action time, even when the legacy training loss uses the wrong post-action time.
        for epoch in range(1, 21):
            optimizer.zero_grad()
            losses = []
            offset = 0
            for episode in episodes:
                for _decision, hidden in replay(model, episode):
                    losses.append((model.value(hidden) - float(targets[offset])) ** 2)
                    offset += 1
            torch.stack(losses).mean().backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            curve.append({"epoch": epoch, **fit_metrics(values(model, episodes), targets)})
        curves[mode] = curve
    generator = torch.Generator().manual_seed(55)
    rng_before = torch.get_rng_state().clone()
    policy.sample(policy.initial_hidden(), generator=generator)
    fields = rl.E.RandomStartEnv(selected[0]).observation_fields()
    report = {
        "scope": "two fixed rollouts, 20 critic-only fitting passes; NOT an S4 rerun",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                              cwd=rl.ROOT, text=True).strip(),
        "working_tree_changes": subprocess.check_output(["git", "status", "--short"],
                                                          cwd=rl.ROOT, text=True),
        "torch_version": torch.__version__, "seed": 0, "threads": 1,
        "source_sha256": {str(path.relative_to(rl.ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in (Path(rl.__file__), Path(__file__),
                                       rl.ROOT / "tests/test_random_start_rl.py")},
        "instances": [instance.name for instance in selected],
        "rollout_seeds": [100, 101], "lr": rl.LEARNING_RATE,
        "reward_contract_changed": False, "observation_schema_changed": False,
        "episodes": [{key: episode[key] for key in
                      ("success", "termination", "skills_used", "periods", "return")}
                     for episode in episodes],
        "decisions": len(entries),
        "corrected_value_replay_max_abs": float(np.max(np.abs(corrected - recorded))),
        "legacy_value_replay_mean_abs": float(np.mean(np.abs(legacy - recorded))),
        "legacy_value_replay_max_abs": float(np.max(np.abs(legacy - recorded))),
        "sampling_preserves_global_rng": bool(torch.equal(rng_before, torch.get_rng_state())),
        "real_ppo_zero_lr_check": {"parameters_unchanged": unchanged,
                                   "ratio_mean": zero_lr_metrics["ratio_mean"],
                                   "ratio_max": zero_lr_metrics["ratio_max"]},
        "critic_fit_curves": curves,
        "input_audit": {"dimensions": len(fields), "fields": fields,
                        "past_action_present": False, "skill_stage_present": False,
                        "note": "missing legal history is a schema gap, not proof of impossibility"},
        "discount_audit": {"gamma_per_period": rl.GAMMA,
                           "half_life_seconds": float(np.log(0.5) / np.log(rl.GAMMA)
                                                      * rl.CONTROL_DT),
                           "terminal_multiplier_at_1700_periods": rl.GAMMA ** 1699,
                           "causal_verdict": "discount alone not isolated"},
        "wall_seconds": time.time() - start,
        "limits": "in-sample critic fitting only; no reward ablation, no actor improvement claim",
    }
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("decisions", "corrected_value_replay_max_abs",
                       "legacy_value_replay_max_abs", "sampling_preserves_global_rng",
                       "wall_seconds")}, indent=2))
    print(json.dumps({mode: {"before": curve[0], "after": curve[-1]}
                      for mode, curve in curves.items()}, indent=2))


if __name__ == "__main__":
    main()
