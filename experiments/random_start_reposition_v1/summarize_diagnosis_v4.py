"""Offline paired uncertainty, parameter changes and recovery-action probabilities."""
import json

import numpy as np
import torch

import diagnose_v4 as d


def main():
    torch.set_num_threads(1)
    out = d.OUT
    replay = json.loads((out / "replay.json").read_text())
    initial = replay["validation_v2_initial"]["rows"]
    late = replay["validation_v2_late_4600"]["rows"]
    assert [x["instance"] for x in initial] == [x["instance"] for x in late]
    delta = np.array([int(b["success"])-int(a["success"]) for a, b in zip(initial, late)])
    rng = np.random.default_rng(20261018)
    boot = delta[rng.integers(0, len(delta), (20000, len(delta)))].mean(axis=1)
    report = dict(n=len(delta), initial_success=sum(x["success"] for x in initial),
                  late_success=sum(x["success"] for x in late),
                  paired_improved=int(sum(delta == 1)), paired_worsened=int(sum(delta == -1)),
                  difference=float(delta.mean()), bootstrap_95_percentile=np.quantile(boot, [.025, .975]).tolist(),
                  seed=20261018, replicates=20000, parameter_changes={})
    weights = torch.load(d.SOURCE / "checkpoints/initial.pt", weights_only=True)
    for mode in ("joint", "actor_only", "critic_only"):
        changed = torch.load(out / f"one_update_{mode}.pt", weights_only=True)
        report["parameter_changes"][mode] = {
            group: max(float((changed[k]-v).abs().max()) for k, v in weights.items() if k.startswith(prefix))
            for group, prefix in (("shared", ("encoder.", "gru.")),
                                  ("actor_heads", ("head_first.", "head_second.", "embed.")),
                                  ("value_head", ("value_head.",)))}
    d.save(out / "paired_and_parameter_checks.json", report)
    rows = []
    for model, name, indices in (("initial", "L0_train_0010", (7, 662)),
                                  ("late_4600", "L0_train_0063", (5, 1002))):
        policy = d.load_policy(d.SOURCE / "checkpoints" / d.CHECKPOINTS[model])
        ep = next(e for e in torch.load(out / f"replay_{model}.pt", weights_only=False)
                  if e["instance"] == name)
        probabilities = d.path_scores(policy, [ep])[0]["joint_logs"].exp()
        for index in indices:
            probs = probabilities[index].reshape(15, 15)
            action = ep["decisions"][index]["action"]
            rows.append(dict(model=model, instance=name, boundary=index, chosen_action=action,
                chosen_probability=float(probs[action[0], action[1]]),
                tested_reposition_pairs_probability=float(probs[[9,11,9,11], [9,11,11,9]].sum()),
                all_reposition_pairs_probability=float(probs[9:13, 9:13].sum()),
                all_push_pairs_probability=float(probs[:9, :9].sum())))
    d.save(out / "actor_recovery_probabilities.json", rows)
    batch = torch.load(out / "stochastic24.pt", weights_only=False)[:8]
    original = d.load_policy(d.SOURCE / "checkpoints/initial.pt")
    before = d.path_scores(original, batch)
    kl_report = {}
    for mode in ("joint", "actor_only", "critic_only"):
        changed = d.load_policy(out / f"one_update_{mode}.pt")
        changes = d.path_change(before, d.path_scores(changed, batch))
        weights = np.array([len(e["decisions"]) for e in batch])
        kl_report[mode] = dict(
            decision_mean_KL=float(np.average([r["mean_exact_joint_KL"] for r in changes], weights=weights)),
            episode_mean_KL=float(np.mean([r["mean_exact_joint_KL"] for r in changes])), rows=changes)
    d.save(out / "training_batch_KL.json", kl_report)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
