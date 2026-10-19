"""S3 analysis for the bounded two-arm stability comparison.

Reads only saved artefacts (``updates.jsonl``, ``initial.json``, ``best_selection.json`` and
weight sidecars). It never retrains and never changes a checkpoint. ``final-test`` is the one
place that touches the independent test64 panel; it runs once per arm after the best model has
been frozen by the pre-registered validation rule.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import torch

EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import long_train as lt  # noqa: E402
import rl  # noqa: E402


def read_updates(run_dir):
    path = run_dir / "updates.jsonl"
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            break
        if row.get("late_evaluation"):
            for previous in reversed(rows):
                if previous["update"] == row["update"]:
                    previous["evaluation"] = row["evaluation"]
                    break
        else:
            rows.append(row)
    return rows


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def evaluation_points(rows):
    return [(row["update"], row["evaluation"]) for row in rows if "evaluation" in row]


def transition_rows(points, key):
    if not points:
        return []
    def success_set(evaluation):
        return {name for name, value in evaluation[key]["per_instance"].items() if value >= 0.5}
    initial = success_set(points[0][1])
    previous = initial
    history_success = set(initial)
    rows = []
    for update, evaluation in points[1:]:
        current = success_set(evaluation)
        history_success |= current
        rows.append(dict(
            update=update, instances=len(current),
            gained=sorted(current - previous), lost=sorted(previous - current),
            retained=sorted(current & previous),
            initial_retention=(len(current & initial) / len(initial) if initial else None),
            history_success_rate=(len(current & history_success) / len(history_success)
                                  if history_success else None)))
        previous = current
    return rows


def average(rows, path, default=None):
    values = []
    for row in rows:
        node = row
        for key in path:
            if not isinstance(node, dict) or key not in node:
                node = None
                break
            node = node[key]
        if isinstance(node, (int, float)) and node is not None:
            values.append(float(node))
    return float(np.mean(values)) if values else default


def summarise_run(run_dir):
    manifest = read_json(run_dir / "manifest.json")
    if manifest is None:
        return None
    rows = read_updates(run_dir)
    points = evaluation_points(rows)
    initial = read_json(run_dir / "initial.json")
    best = read_json(run_dir / "checkpoints" / "best_selection.json",
                     read_json(run_dir / "checkpoints" / "best.json"))
    config = manifest["config"]
    def point_summary(evaluation):
        if evaluation is None:
            return None
        summary = {}
        for key, value in evaluation.items():
            summary[key] = {"success": value["success"],
                            "median_final_distance": value["median_final_distance"],
                            "return_mean": value.get("return_mean")}
        return summary
    window = max(5, min(100, len(rows) // 4))
    early = rows[:window]
    late = rows[-window:]
    return dict(
        run_dir=str(run_dir.relative_to(rl.ROOT)) if run_dir.is_relative_to(rl.ROOT)
        else str(run_dir),
        arm=config["arm"], critic_mode=config["critic_mode"], stage=config["stage"],
        updates_budget=config["updates"], status=manifest.get("status"),
        completed_update=(rows[-1]["update"] if rows else 0),
        training_episodes=len(rows) * config["episodes_per_update"],
        elapsed_hours=((manifest.get("finished_unix") or manifest["started_unix"])
                       - manifest["started_unix"]) / 3600,
        best=({key: best[key] for key in ("update", "score", "selection") if key in best}
              if best else None),
        initial=point_summary(initial), final=point_summary(points[-1][1]) if points else None,
        evaluation_points=[update for update, _ in points],
        tracking_transitions=transition_rows(points, "deterministic"),
        validation_transitions=transition_rows(points, "validation"),
        critic_same_batch_ev=average(rows, ("critic_after", "explained_variance")),
        critic_value_error=average(rows, ("value_error_mean",)),
        grad_norm=average(rows, ("grad_norm_last",)),
        grad_clip_scale=average(rows, ("grad_clip_scale_last",)),
        entropy=average(rows, ("entropy_mean",)),
        early=dict(
            success=average(early, ("success",)),
            failure_short_decision_fraction=average(
                early, ("diagnostics", "failure", "short_decision_fraction")),
            failure_short_time_fraction=average(
                early, ("diagnostics", "failure", "short_time_fraction")),
            success_short_decision_fraction=average(
                early, ("diagnostics", "success", "short_decision_fraction")),
            success_short_time_fraction=average(
                early, ("diagnostics", "success", "short_time_fraction"))),
        late=dict(
            success=average(late, ("success",)),
            failure_short_decision_fraction=average(
                late, ("diagnostics", "failure", "short_decision_fraction")),
            failure_short_time_fraction=average(
                late, ("diagnostics", "failure", "short_time_fraction")),
            success_short_decision_fraction=average(
                late, ("diagnostics", "success", "short_decision_fraction")),
            success_short_time_fraction=average(
                late, ("diagnostics", "success", "short_time_fraction"))),
        s4_gate=lt.sustained_gate(
            [dict(evaluation=evaluation) for _, evaluation in points],
            initial["deterministic"]["success"]) if initial and points else None)


def load_policy(run_dir, name="best.pt"):
    weights = torch.load(run_dir / "checkpoints" / name, map_location="cpu", weights_only=True)
    sidecar = read_json(run_dir / "checkpoints" / name.replace(".pt", ".json"))
    contract = (sidecar or {}).get("contract", {})
    policy = rl.JointPolicy(contract.get("observation_dim", 324),
                            hidden=contract.get("hidden", rl.HIDDEN),
                            critic_shared_grad=contract.get("critic_shared_grad", True))
    policy.load_state_dict(weights)
    policy.eval()
    return policy, sidecar


def critic_on_rollouts(executor, policy, instances, config, seeds=3):
    """Explained variance on newly generated stochastic evaluation trajectories.

    This is separate from the same-mini-batch EV logged during training: it uses the current
    policy's own fresh rollouts, not the terminal returns of an old behaviour.
    """
    state = lt.snapshot(policy)
    jobs = [(instance, state, policy.hidden, seed, False, config, False)
            for instance in instances for seed in range(seeds)]
    episodes = lt.collect(executor, jobs)
    targets, predictions = [], []
    for episode in episodes:
        for decision in episode["decisions"]:
            targets.append(float(decision["return_to_go"]) / max(1e-12, float(decision["prefix_before"])))
            predictions.append(float(decision["value"]))
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    residual = targets - predictions
    variance = float(np.var(targets))
    return dict(episodes=len(episodes), decisions=int(targets.size),
                mse=float(np.mean(residual ** 2)),
                explained_variance=(float(1.0 - np.var(residual) / variance)
                                    if variance > 1e-12 else None),
                target_variance=variance)


def final_test(root, workers=8):
    panels = lt.build_stability_panels(verify=True)
    environment = rl.E.EnvConfig()
    runs = [(stage, arm, root / lt.STABILITY_RUN_DIRS[(stage, arm)])
            for stage in ("long",) for arm in ("shared", "stopped")]
    report = {}
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=workers, mp_context=mp.get_context("spawn"),
            initializer=lt.worker_init) as executor:
        for stage, arm, run_dir in runs:
            if not (run_dir / "checkpoints" / "best.pt").exists():
                report[f"{stage}/{arm}"] = {"status": "missing_model"}
                continue
            entry = {"status": "evaluated"}
            for label, name in (("update0", "initial.pt"), ("best", "best.pt"),
                                ("final_at_budget", "update_00500.pt")):
                if not (run_dir / "checkpoints" / name).exists():
                    entry[label] = {"status": "missing"}
                    continue
                policy, sidecar = load_policy(run_dir, name)
                entry[label] = dict(
                    sidecar={key: sidecar.get(key) for key in
                             ("update", "critic_mode", "env_version", "split_version")}
                    if sidecar else None,
                    test_deterministic=lt.evaluate_panel(
                        executor, policy, panels["test"], environment, stochastic=False),
                    test_stochastic=lt.evaluate_panel(
                        executor, policy, panels["test"], environment, stochastic=True, seeds=3),
                    critic_fresh_rollouts=critic_on_rollouts(
                        executor, policy, panels["tracking"], environment, seeds=3))
            report[f"{stage}/{arm}"] = entry
            lt.atomic_json(run_dir / "final_test.json", entry)
    lt.atomic_json(root / "comparison" / "final_test.json", report)
    return report


def bootstrap_mean_interval(values, resamples=20000, seed=20261019):
    """Instance-level paired bootstrap. This is NOT cross-training-seed evidence."""
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, values.size, (resamples, values.size))
    means = values[draws].mean(axis=1)
    return dict(mean=float(values.mean()), lo=float(np.quantile(means, 0.025)),
                hi=float(np.quantile(means, 0.975)), n=int(values.size),
                resamples=int(resamples))


def paired_difference(left, right):
    names = sorted(set(left) & set(right))
    differences = [float(left[name]) - float(right[name]) for name in names]
    interval = bootstrap_mean_interval(differences)
    if interval is not None:
        interval["positive"] = int(sum(value > 0 for value in differences))
        interval["negative"] = int(sum(value < 0 for value in differences))
        interval["zero"] = int(sum(value == 0 for value in differences))
    return interval


def test_comparison(root):
    final = read_json(root / "comparison" / "final_test.json")
    if not final:
        return None
    report = {"note": "paired by instance; random execution is first averaged over the three "
                      "behaviour seeds inside each instance; one training seed only",
              "runs": {}}
    for run, entry in final.items():
        if entry.get("status") != "evaluated":
            report["runs"][run] = {"status": entry.get("status")}
            continue
        update0 = entry.get("update0", {})
        best = entry.get("best", {})
        if not update0.get("test_deterministic") or not best.get("test_deterministic"):
            report["runs"][run] = {"status": "missing_evaluation"}
            continue
        det0 = update0["test_deterministic"]["per_instance"]
        detb = best["test_deterministic"]["per_instance"]
        stob = best["test_stochastic"]["per_instance"]
        st0 = update0.get("test_stochastic", {}).get("per_instance", {})
        report["runs"][run] = dict(
            best_update=best.get("sidecar", {}).get("update")
            if isinstance(best.get("sidecar"), dict) else None,
            test_deterministic_update0=float(np.mean(list(det0.values()))),
            test_deterministic_best=float(np.mean(list(detb.values()))),
            test_stochastic_best=float(np.mean(list(stob.values()))) if stob else None,
            best_minus_update0_deterministic=paired_difference(detb, det0),
            best_minus_update0_stochastic=paired_difference(stob, st0) if st0 else None,
            critic_fresh_rollouts_best=best.get("critic_fresh_rollouts"),
            critic_fresh_rollouts_update0=update0.get("critic_fresh_rollouts"))
        final_at_budget = entry.get("final_at_budget", {})
        if final_at_budget.get("test_deterministic"):
            detf = final_at_budget["test_deterministic"]["per_instance"]
            stof = final_at_budget["test_stochastic"]["per_instance"]
            report["runs"][run].update(
                test_deterministic_final_at_budget=float(np.mean(list(detf.values()))),
                test_stochastic_final_at_budget=float(np.mean(list(stof.values()))),
                final_at_budget_minus_update0_deterministic=paired_difference(detf, det0),
                final_at_budget_minus_update0_stochastic=paired_difference(stof, st0)
                if st0 else None,
                critic_fresh_rollouts_final_at_budget=final_at_budget.get(
                    "critic_fresh_rollouts"))
    if "long/shared" in report["runs"] and "long/stopped" in report["runs"]:
        a = report["runs"]["long/shared"]
        b = report["runs"]["long/stopped"]
        if "test_deterministic_best" in a and "test_deterministic_best" in b:
            # recover per-instance values from the raw final_test payload
            det_a = final["long/shared"]["best"]["test_deterministic"]["per_instance"]
            det_b = final["long/stopped"]["best"]["test_deterministic"]["per_instance"]
            sto_a = final["long/shared"]["best"]["test_stochastic"]["per_instance"]
            sto_b = final["long/stopped"]["best"]["test_stochastic"]["per_instance"]
            report["A_minus_B"] = dict(deterministic=paired_difference(det_a, det_b),
                                       stochastic=paired_difference(sto_a, sto_b))
            fa = final["long/shared"].get("final_at_budget", {})
            fb = final["long/stopped"].get("final_at_budget", {})
            if fa.get("test_deterministic") and fb.get("test_deterministic"):
                report["A_minus_B_final_at_budget"] = dict(
                    deterministic=paired_difference(
                        fa["test_deterministic"]["per_instance"],
                        fb["test_deterministic"]["per_instance"]),
                    stochastic=paired_difference(
                        fa["test_stochastic"]["per_instance"],
                        fb["test_stochastic"]["per_instance"]))
    lt.atomic_json(root / "comparison" / "test_paired.json", report)
    return report


def plot_curves(root, summaries, stage):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))
    keys = (("deterministic", "original-8 deterministic"),
            ("stochastic", "original-8 stochastic"),
            ("train_panel", "fixed 32 training panel deterministic"),
            ("validation", "independent validation64 deterministic"))
    colors = {"shared": "#1f77b4", "stopped": "#d62728"}
    for axis, (key, title) in zip(axes.flatten(), keys):
        for summary in summaries:
            if summary["stage"] != stage:
                continue
            run_dir = rl.ROOT / summary["run_dir"]
            points = evaluation_points(read_updates(run_dir))
            initial = read_json(run_dir / "initial.json")
            if initial is not None:
                points = [(0, initial)] + points
            xs = [update for update, evaluation in points if key in evaluation]
            ys = [evaluation[key]["success"] for update, evaluation in points if key in evaluation]
            if xs:
                axis.plot(xs, ys, marker=".", color=colors[summary["arm"]],
                          label=f"{summary['arm']}")
        axis.set_title(title); axis.set_xlabel("update"); axis.set_ylabel("success")
        axis.set_ylim(-0.02, 1.02); axis.grid(alpha=.2); axis.legend()
    figure.suptitle(f"bounded stability comparison — {stage}")
    figure.tight_layout()
    path = root / "comparison" / f"curves_{stage}.png"
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=130); plt.close(figure); temporary.replace(path)


def plot_heatmap(root, summaries, stage, key, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    entries = [summary for summary in summaries if summary["stage"] == stage]
    if not entries:
        return
    figure, axes = plt.subplots(len(entries), 1, figsize=(14, 4 * len(entries)), squeeze=False)
    for axis, summary in zip(axes[:, 0], entries):
        run_dir = rl.ROOT / summary["run_dir"]
        points = evaluation_points(read_updates(run_dir))
        initial = read_json(run_dir / "initial.json")
        if initial is not None:
            points = [(0, initial)] + points
        if not points:
            axis.set_visible(False)
            continue
        instances = sorted(points[0][1][key]["per_instance"])
        matrix = np.asarray([[evaluation[key]["per_instance"].get(name, np.nan)
                              for name in instances] for _, evaluation in points])
        image = axis.imshow(matrix.T, aspect="auto", interpolation="nearest",
                            vmin=0.0, vmax=1.0, cmap="viridis")
        axis.set_yticks(range(len(instances)))
        axis.set_yticklabels(instances, fontsize=7)
        step = max(1, len(points) // 20)
        axis.set_xticks(range(0, len(points), step))
        axis.set_xticklabels([points[i][0] for i in range(0, len(points), step)],
                             rotation=45, fontsize=7)
        axis.set_title(f"{summary['arm']} ({summary['critic_mode']}) — {key}")
        axis.set_xlabel("update"); axis.set_ylabel("instance")
        figure.colorbar(image, ax=axis, fraction=0.02)
    figure.tight_layout()
    path = root / "comparison" / filename
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=130); plt.close(figure); temporary.replace(path)


def compare(root):
    (root / "comparison").mkdir(parents=True, exist_ok=True)
    summaries = []
    for stage in ("pilot", "long"):
        for arm in ("shared", "stopped"):
            summary = summarise_run(root / lt.STABILITY_RUN_DIRS[(stage, arm)])
            if summary is not None:
                summaries.append(summary)
    report = {"generated_unix": __import__("time").time(), "runs": summaries}
    lt.atomic_json(root / "comparison" / "comparison.json", report)
    for stage in ("pilot", "long"):
        if any(summary["stage"] == stage for summary in summaries):
            plot_curves(root, summaries, stage)
            plot_heatmap(root, summaries, stage, "deterministic",
                         f"heatmap_tracking_{stage}.png")
            plot_heatmap(root, summaries, stage, "validation",
                         f"heatmap_validation_{stage}.png")
    lines = ["# 两臂对照摘要（自动生成）", "",
             "所有数字来自 `updates.jsonl` / `initial.json` / `best_selection.json`。", ""]
    for summary in summaries:
        lines += [f"## {summary['stage']} / {summary['arm']} ({summary['critic_mode']})", "",
                  f"- 状态：{summary['status']}；完成更新 {summary['completed_update']}/"
                  f"{summary['updates_budget']}；训练轨迹 {summary['training_episodes']}；"
                  f"墙钟 {summary['elapsed_hours']:.2f} h",
                  f"- best：{summary['best']}",
                  f"- 同批 critic EV 均值：{summary['critic_same_batch_ev']}",
                  f"- 早期/晚期采样成功率：{summary['early']['success']} / "
                  f"{summary['late']['success']}",
                  f"- 早期/晚期失败短技能决策占比："
                  f"{summary['early']['failure_short_decision_fraction']} / "
                  f"{summary['late']['failure_short_decision_fraction']}",
                  f"- 早期/晚期失败短技能物理时间占比："
                  f"{summary['early']['failure_short_time_fraction']} / "
                  f"{summary['late']['failure_short_time_fraction']}",
                  f"- S4 连续门槛：{summary['s4_gate']}", ""]
        if summary["initial"] and summary["final"]:
            lines += ["| 评价 | update0 | 最终 |", "|---|---:|---:|"]
            for key in summary["final"]:
                lines.append(f"| {key} | {summary['initial'][key]['success']} | "
                             f"{summary['final'][key]['success']} |")
            lines.append("")
    (root / "comparison" / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    test_report = test_comparison(root)
    if test_report is not None:
        report["test_paired"] = test_report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("compare", "final-test"))
    parser.add_argument("--root", type=Path, default=lt.STABILITY_ROOT)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.stage == "compare":
        print(json.dumps(compare(args.root)["runs"], indent=1, default=str)[:4000])
    else:
        print(json.dumps(final_test(args.root, args.workers), indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
