"""S3 representative replays: A / B / update0 from the same physical start.

Selection is fixed before looking at the replays:
1. an instance the common initial model already solved;
2. an instance newly solved during training;
3. the failing instance with the most short skills (<= 2 control periods).

True state in these logs is diagnostic truth only; it is never a policy input.
"""
from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path

import torch

EXPERIMENT = Path(__file__).resolve().parent
if str(EXPERIMENT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT))
import long_train as lt  # noqa: E402
import rl  # noqa: E402
from coimagine.random_start import render  # noqa: E402


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def find_run(root, arm):
    for stage in ("long", "pilot"):
        directory = root / lt.STABILITY_RUN_DIRS[(stage, arm)]
        if (directory / "manifest.json").exists():
            return stage, directory
    raise FileNotFoundError(f"no run for arm {arm}")


def evaluation_rows(run_dir, which="best", panel="deterministic"):
    if which == "initial":
        payload = read_json(run_dir / "initial.json")
    else:
        payload = read_json(run_dir / "checkpoints" / "best_selection.json",
                            read_json(run_dir / "checkpoints" / "best.json"))
        payload = payload.get("evaluation") if payload else None
    if not payload:
        raise FileNotFoundError(f"no {which} evaluation in {run_dir}")
    return {row["instance"]: row for row in payload[panel]["rows"]}


def choose_cases(root):
    stage_a, run_a = find_run(root, "shared")
    stage_b, run_b = find_run(root, "stopped")
    initial = evaluation_rows(run_a, "initial")
    best_a = evaluation_rows(run_a, "best")
    best_b = evaluation_rows(run_b, "best")
    initial_val = evaluation_rows(run_a, "initial", "validation")
    best_a_val = evaluation_rows(run_a, "best", "validation")
    best_b_val = evaluation_rows(run_b, "best", "validation")

    def merged(name):
        """Prefer a successful read-out; otherwise keep whichever arm has a row."""
        a, b = best_a.get(name), best_b.get(name)
        for row in (a, b):
            if row is not None and row["success"]:
                return row
        return a if a is not None else b

    names = sorted(set(best_a) | set(best_b))
    initial_success = {name for name, row in initial.items() if row["success"]}
    gained = {name for name in names if merged(name)["success"] and name not in initial_success}
    failures = [merged(name) for name in names
                if not merged(name)["success"] and name not in initial_success]
    cases = []
    if initial_success:
        pick = min((initial[name] for name in initial_success), key=lambda row: row["periods"])
        cases.append(dict(label="initial_success", instance=pick["instance"],
                          panel="tracking",
                          rule="common initial model success; fewest periods",
                          facts=dict(initial=initial[pick["instance"]],
                                     best_A=best_a.get(pick["instance"]),
                                     best_B=best_b.get(pick["instance"]))))
    panel = "tracking"
    if gained:
        pick = max((merged(name) for name in sorted(gained)),
                   key=lambda row: (row["success"], -row["final_goal_distance"]))
    else:
        def merged_val(name):
            a, b = best_a_val.get(name), best_b_val.get(name)
            for row in (a, b):
                if row is not None and row["success"]:
                    return row
            return a if a is not None else b

        gained_val = {name for name in set(best_a_val) | set(best_b_val)
                      if merged_val(name)["success"]
                      and not initial_val.get(name, {"success": False})["success"]}
        if gained_val:
            panel = "validation"
            pick = max((merged_val(name) for name in sorted(gained_val)),
                       key=lambda row: (row["success"], -row["final_goal_distance"]))
        else:
            pick = None
    if pick is not None:
        cases.append(dict(label="newly_gained_success", instance=pick["instance"],
                          panel=panel,
                          rule="successful in the selected best model and not in update0 "
                               f"({panel} panel; tracking first, validation fallback)",
                          facts=dict(initial=(initial if panel == "tracking" else initial_val)
                                     .get(pick["instance"]),
                                     best_A=(best_a if panel == "tracking" else best_a_val)
                                     .get(pick["instance"]),
                                     best_B=(best_b if panel == "tracking" else best_b_val)
                                     .get(pick["instance"]))))
    if failures:
        pick = max(failures, key=lambda row: (row["short_decisions"], row["periods"]))
        cases.append(dict(label="most_short_skills_failure", instance=pick["instance"],
                          panel="tracking",
                          rule="best-model failure with the most <=2-period skills",
                          facts=dict(initial=initial[pick["instance"]],
                                     best_A=best_a.get(pick["instance"]),
                                     best_B=best_b.get(pick["instance"]))))
    return dict(stage_A=stage_a, stage_B=stage_b, run_A=str(run_a), run_B=str(run_b),
                cases=cases)


def load_policy(run_dir, which):
    name = "initial.pt" if which == "initial" else "best.pt"
    weights = torch.load(run_dir / "checkpoints" / name, map_location="cpu", weights_only=True)
    sidecar = read_json(run_dir / "checkpoints" / name.replace(".pt", ".json"), {})
    contract = (sidecar or {}).get("contract", {})
    policy = rl.JointPolicy(contract.get("observation_dim", 324),
                            hidden=contract.get("hidden", rl.HIDDEN),
                            critic_shared_grad=contract.get("critic_shared_grad", True))
    policy.load_state_dict(weights)
    policy.eval()
    return policy


def run_case(root, case, panels, environment, out):
    instance = next(item for item in panels[case.get("panel", "tracking")]
                    if item.name == case["instance"])
    stage_a, run_a = find_run(root, "shared")
    stage_b, run_b = find_run(root, "stopped")
    results = {}
    labels = [("initial", run_a, "initial"), ("A_best", run_a, "best"),
              ("B_best", run_b, "best")]
    for label, run_dir, which in labels:
        policy = load_policy(run_dir, which)
        environment_run = rl.E.RandomStartEnv(instance, environment)
        episode = rl.rollout(environment_run, policy, stochastic=False)
        exits = {end["exit_reason"]
                 for boundary in environment_run.boundaries for end in boundary["skill_ends"].values()}
        directory = out / "keyframes"
        meta = {"name": f"{case['instance']}__{label}", "map": instance.map,
                "arena": list(instance.arena),
                "obstacles": [{"cx": box.cx, "cy": box.cy, "half_w": box.half_w,
                               "half_h": box.half_h} for box in instance.obstacles],
                "title": (f"{case['label']}: {case['instance']} / {label}, "
                          f"success={episode['success']}, {episode['periods']} periods "
                          f"({episode['periods'] * rl.CONTROL_DT:.2f} s), "
                          f"{episode['skills_used']} joint skills, final distance "
                          f"{episode['final_goal_distance']:.2f} m")}
        stem = f"{case['label']}_{case['instance']}_{label}"
        picks = render.write_keyframes(environment_run.rows, meta, directory, stem)
        step = max(1, len(environment_run.rows) // 100)
        # Only the initial-success case needs full clips for all three policies: it is the
        # clean same-start A/B divergence. Other cases keep keyframes and the full row log.
        clip = (render.render_episode(environment_run.rows, meta, out / "clips", stem,
                                      fps=20, step=step)
                if case["label"] == "initial_success" else None)
        with gzip.open(out / "logs" / f"{stem}.json.gz", "wt") as handle:
            json.dump({"instance": instance.as_dict(), "policy": label,
                       "episode": {"success": episode["success"],
                                   "termination": episode["termination"],
                                   "periods": episode["periods"],
                                   "physical_seconds": episode["periods"] * rl.CONTROL_DT,
                                   "skills_used": episode["skills_used"],
                                   "final_goal_distance": episode["final_goal_distance"],
                                   "return": episode["return"],
                                   "reward_components": episode["reward_components"],
                                   "exits": episode["exits"],
                                   "short_decisions": episode["short_decisions"],
                                   "short_time_fraction": episode["short_time_fraction"],
                                   "longest_repeat_run": episode["longest_repeat_run"]},
                       "decisions": [{"skill_index": decision["skill_index"],
                                      "action": decision["action"],
                                      "logp": decision["logp"], "value": decision["value"],
                                      "periods": decision["periods"],
                                      "reward": decision["reward"]}
                                     for decision in episode["decisions"]],
                       "boundaries": environment_run.boundaries,
                       "rows": environment_run.rows,
                       "note": "true state / goal distance / hold timer are diagnostic truth; "
                               "they are never policy inputs"},
                      handle, default=str)
        results[label] = dict(success=bool(episode["success"]),
                              termination=episode["termination"],
                              periods=int(episode["periods"]),
                              physical_seconds=float(episode["periods"] * rl.CONTROL_DT),
                              skills_used=int(episode["skills_used"]),
                              final_goal_distance=float(episode["final_goal_distance"]),
                              return_=float(episode["return"]),
                              short_decisions=int(episode["short_decisions"]),
                              short_time_fraction=float(episode["short_time_fraction"]),
                              exits=dict(episode["exits"]),
                              exit_reasons=sorted(exits),
                              decisions=[decision["action"]
                                         for decision in episode["decisions"]],
                              keyframes=[pick["path"] for pick in picks],
                              clip=clip)
    case["replays"] = results
    return case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=lt.STABILITY_ROOT)
    args = parser.parse_args()
    root = args.root
    panels = lt.build_stability_panels(verify=True)
    environment = rl.E.EnvConfig()
    out = root / "cases"
    for directory in ("logs", "keyframes", "clips"):
        (out / directory).mkdir(parents=True, exist_ok=True)
    selection = choose_cases(root)
    report = dict(selection_rule=selection, cases=[])
    for case in selection["cases"]:
        report["cases"].append(run_case(root, case, panels, environment, out))
        lt.atomic_json(out / "selection.json", report)
    lines = ["# 代表回放（同一起点：update0 / A_shared / B_stopped）", "",
             "选择规则预先固定；真值仅用于诊断绘图，不进入策略输入。", ""]
    for case in report["cases"]:
        lines += [f"## {case['label']} — {case['instance']}", "", f"规则：{case['rule']}", "",
                  "| 策略 | 成功 | 周期 | 物理时间(s) | 技能数 | 终点距离(m) | 短技能决策 | 短技能时间占比 | 退出 |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---|"]
        for label, row in case["replays"].items():
            lines.append(f"| {label} | {row['success']} | {row['periods']} | "
                         f"{row['physical_seconds']:.2f} | {row['skills_used']} | "
                         f"{row['final_goal_distance']:.3f} | {row['short_decisions']} | "
                         f"{row['short_time_fraction']:.3f} | {row['exit_reasons']} |")
        lines.append("")
    (out / "README.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({case["label"]: {label: row["success"]
                                      for label, row in case["replays"].items()}
                      for case in report["cases"]}, indent=1))


if __name__ == "__main__":
    main()
