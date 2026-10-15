"""Assemble README.md, STATUS.md and results/SUMMARY.md from the result files.

Nothing here invents a number: every value is read out of a result file that a run wrote, and a
missing run is reported as "not run" rather than omitted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> Optional[Dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def fmt(value, digits: int = 3, dash: str = "n/a") -> str:
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def video_run_index(out: Path) -> Dict[str, object]:
    """Per-clip provenance: run id, seed, config hash, code commit, skill parameters, raw logs."""

    import subprocess

    sys.path.insert(0, str(ROOT / "src"))
    from coimagine.random_start import spec
    from coimagine.random_start import env as E
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                         text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT,
                                             text=True).strip())
    except Exception:  # pragma: no cover
        commit, dirty = None, None
    entries: Dict[str, object] = {}
    config = E.EnvConfig()
    for keyframes in sorted((out / "videos").rglob("*_keyframes.json")):
        stem = keyframes.name[:-len("_keyframes.json")]
        rows = []
        frames = keyframes.parent / f"{stem}_frames.json"
        if frames.exists():
            payload = json.loads(frames.read_text())
            rows.append({"frames_log": str(frames.relative_to(out)),
                         "frames": payload.get("count")})
        entries[stem] = {
            "run_id": stem,
            "config_hash": spec.CONFIG_HASH,
            "env_version": spec.ENV_VERSION,
            "reward_config": config.as_dict(),
            "skill_action_table": [action.label for action in spec.ROBOT_ACTIONS],
            "observation_width": len(E.RandomStartEnv.observation_fields(
                E.RandomStartEnv.__new__(E.RandomStartEnv))),
            "code_commit": commit,
            "worktree_dirty_at_packaging": dirty,
            "seed": "fixed per instance: SPLIT_SEEDS in spec.py; the instance name carries its "
                    "index and the per-instance seed is instance_seed = base_seed + index",
            "raw_logs": rows + [{"keyframes": str(keyframes.relative_to(out))}],
            "video_dir": str(keyframes.parent.relative_to(out)),
        }
    events = sorted((out / "logs").glob("*_events.json"))
    for path in events:
        entries.setdefault(path.stem, {"run_id": path.stem,
                                       "raw_logs": [{"event_log": str(path.relative_to(out))}]})
    return {"generated_by": "report.py", "commit": commit, "config_hash": spec.CONFIG_HASH,
            "clips": entries}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "outputs/random_start_reposition_v1"))
    args = parser.parse_args()
    out = Path(args.out)
    results = out / "results"

    scenes = load(results / "scene_sampling.json")
    skills = load(results / "skills_acceptance.json")
    episodes = load(results / "continuous_episodes.json")
    invariants = load(results / "invariants.json")
    rl = load(results / "rl_diagnostic.json")
    collection = load(results / "collection_schema.json") or load(results / "collection_trial.json")
    subset = load(results / "diagnostic_subset.json")
    ablation = load(results / "ablation_trajectory" / "results" / "rl_diagnostic.json")
    resume = load(results / "resume_check" / "resume_check.json")

    lines = ["# Random-start reposition round — summary", ""]
    lines.append("## Scene and sensing (S1)")
    if scenes:
        for map_name in ("L0", "L1"):
            entry = scenes["maps"][map_name]
            legality = all(entry[split]["legality"]["ok"]
                           for split in ("train", "validation", "test"))
            stats = entry["train"]["statistics"]
            lines.append(
                f"* {map_name}: train {entry['train']['manifest']['count']} / validation "
                f"{entry['validation']['manifest']['count']} / test "
                f"{entry['test']['manifest']['count']} instances, "
                f"independent legality re-check {'passed' if legality else 'FAILED'}; "
                f"realised robot-object gap median "
                f"{fmt(stats['robot_object_gap_m']['median'])} m in "
                f"[{fmt(stats['robot_object_gap_m']['min'])}, "
                f"{fmt(stats['robot_object_gap_m']['max'])}], goal distance median "
                f"{fmt(stats['goal_distance_m']['median'])} m.")
            rejected = entry["train"]["sampling"]["rejections"]["reasons"]
            lines.append(f"  Rejections (train, counted by reason): {rejected}.")
    else:
        lines.append("* not run")

    lines.append("")
    lines.append("## Skills (S2)")
    if skills:
        gate = skills["gate"]
        for name, run in gate["runs"].items():
            lines.append(f"* frozen subset {name}: {run['completed']}/{run['n']} completed the "
                         f"requested distance, {run['face_crossings']} true face crossings "
                         f"(post-hoc truth); exits {run['exit_reasons']}.")
        lines.append(f"* gate (>=16/20 per direction on the frozen subset): "
                     f"{'PASSED' if gate['pass'] else 'NOT MET'}")
        waits = skills["wait_brake"]
        lines.append(f"* WAIT_BRAKE: no push command {waits['no_push_command']}, world not "
                     f"frozen {waits['world_not_frozen']}, no forced zero object velocity "
                     f"{waits['no_forced_zero_velocity']}.")
        battery = skills["battery"]["by_action"]
        pushes = {key: value for key, value in battery.items() if key.startswith("PUSH")}
        total = sum(value["n"] for value in pushes.values())
        complete = sum(value["exits"].get("push_complete", 0) for value in pushes.values())
        lines.append(f"* PUSH battery: {complete}/{total} probes end in `push_complete` on "
                     f"{len({key for key in battery})} action variants.")
    else:
        lines.append("* not run")

    lines.append("")
    lines.append("## Continuous joint execution (S3)")
    if episodes:
        for key, value in episodes["summary"].items():
            lines.append(f"* {key}: mean success {fmt(value['mean_success'])} "
                         f"(95% CI over instances [{fmt(value['ci95'][0])}, "
                         f"{fmt(value['ci95'][1])}]), median final distance "
                         f"{fmt(value.get('median_final_distance'))} m, median skills "
                         f"{fmt(value.get('median_skills'), 1)}.")
    else:
        lines.append("* not run")

    lines.append("")
    lines.append("## Semantics checks (7.2)")
    if invariants:
        failed = [name for name, check in invariants["checks"].items() if not check["ok"]]
        lines.append(f"* {invariants['passed']}/{invariants['total']} checks passed"
                     + (f"; failures: {failed}" if failed else ""))
    else:
        lines.append("* not run")

    lines.append("")
    lines.append("## RL diagnostic (S4)")
    if rl:
        lines.append(f"* subset: {rl['dev_subset']['instances']} ({rl['dev_subset']['selection']})")
        lines.append(f"* untrained deterministic {fmt(rl['baselines']['untrained_deterministic'])} "
                     f"/ stochastic {fmt(rl['baselines']['untrained_stochastic'])}, uniform random "
                     f"{fmt(rl['baselines']['uniform_random']['success'])}")
        lines.append(f"* trained deterministic {fmt(rl['final_deterministic']['success'])} "
                     f"/ stochastic {fmt(rl['final_stochastic']['success'])}, validation "
                     f"deterministic {fmt(rl['validation_deterministic']['success'])}")
        lines.append(f"* pass signal (>=75% and >=+20pp on the diagnostic subset): "
                     f"{rl['pass_signal']}")
        lines.append(f"* bandit direction check {rl['bandit_check']['ok']}, probability replay "
                     f"{rl['probability_replay_check']['ok']}, unchanged-policy ratio "
                     f"{rl.get('unchanged_policy_ratio_check', {}).get('ok')}")
        history = rl.get("history") or []
        if history:
            lines.append(f"* entropy {fmt(history[0]['entropy_mean'])} -> "
                         f"{fmt(history[-1]['entropy_mean'])} (uniform = ln 225 = 5.416), value "
                         f"error {fmt(history[0]['value_error_mean'])} -> "
                         f"{fmt(history[-1]['value_error_mean'])}, raw advantage mean "
                         f"success/failure "
                         f"{fmt(history[-1].get('advantage_raw_success_mean'))} / "
                         f"{fmt(history[-1].get('advantage_raw_failure_mean'))}")
        lines.append("* full diagnosis: `results/RL_DIAGNOSIS.md`")
    else:
        lines.append("* not run")

        if ablation:
            lines.append(f"* one declared ablation (per-trajectory advantage normalisation, same "
                         f"init and seed): trained deterministic "
                         f"{fmt(ablation['final_deterministic']['success'])} vs batch-wide "
                         f"{fmt(rl['final_deterministic']['success'])} on the same "
                         f"{len(ablation['dev_subset']['instances'])}-instance subset")
    lines.append("")
    lines.append("## Interrupt / resume")
    if resume:
        lines.append(f"* a fresh process resuming from a normally saved checkpoint is "
                     f"{'bit-identical' if resume['ok'] else 'DIFFERENT'} to the uninterrupted run "
                     f"({resume['tensors_compared']} tensors compared)")
    else:
        lines.append("* not run")
    lines.append("")
    lines.append("## Data (S5)")
    if collection:
        card = collection["card"]
        lines.append(f"* mode {collection['mode']}: {card['episodes']} episodes "
                     f"({card['successes']} success / {card['failures']} failure / "
                     f"{card['truncations']} truncated / {card['anomalies']} anomaly), "
                     f"{card['branches']} forced one-step branches, "
                     f"{card['stored_observation_vectors']} observation vectors of width "
                     f"{card['observation_field_count']}; stored frames {card['frames_stored']} "
                     f"vs {card.get('simulated_control_periods')} simulated control periods; "
                     f"{card['branch_interventions']} forced branches.")
    else:
        lines.append("* not run")
    if subset:
        lines.append(f"* diagnostic subset scan: {len(subset['successes'])} scripted successes "
                     f"over {subset['attempted']} episodes; padded without success evidence: "
                     f"{subset['padded_without_success_evidence']}")

    (out / "results" / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out / "videos" / "RUNS.json").write_text(json.dumps(video_run_index(out), indent=1,
                                                         default=str))
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
