from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from coimagine.envs.grid.generation.procedural import (
    ScenarioInstance,
    generate_diagnostic,
    generate_local_imagination_showcase,
    natural_instances,
)
from coimagine.evaluation.episodes import rollout_receding_horizon
from coimagine.evaluation.local_imagination import (
    run_horizon_study,
    summarize_dependency_study,
    summarize_horizon_study,
    summarize_support_study,
)
from coimagine.evaluation.gate_a import evaluate_baselines, evaluate_gate_a, evaluate_instance
from coimagine.evaluation.records import BaselineResult, EpisodeResultRecord, ScalingResult, SceneResult
from coimagine.evaluation.scaling import entity_scaling, robot_factor_scaling
from coimagine.core.types import PlannerMode
from coimagine.reporting.figures import (
    ExampleBundle,
    plot_event_summary,
    plot_factor_diagnostics,
    plot_qualitative_examples,
    plot_scaling,
    plot_topology_examples,
    plot_value_summary,
)
from coimagine.reporting.io import write_csv, write_json
from coimagine.reporting.local_gate_a import (
    plot_dependency_summary,
    plot_horizon_summary,
    plot_local_imagination_story,
    plot_structural_scaling,
    plot_support_summary,
)
from coimagine.reporting.tables import (
    summarize_baselines,
    summarize_events,
    summarize_scenes,
    write_latex_table,
    write_markdown_table,
    write_semantics_glossary,
)


@dataclass(frozen=True)
class RunArtifacts:
    output_dir: Path
    scene_records: tuple[SceneResult, ...]
    baseline_records: tuple[BaselineResult, ...]
    scaling_records: tuple[ScalingResult, ...]
    episode_records: tuple[EpisodeResultRecord, ...]
    failures: tuple[dict[str, str], ...]


def load_config(profile: str, repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[3]
    path = Path(profile)
    if not path.exists():
        path = root / "configs" / f"gate_a_{profile}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Gate-A profile not found: {profile}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data["_config_path"] = str(path)
    return data


def _build_instances(config: dict[str, Any]) -> tuple[list[ScenarioInstance], list[ScenarioInstance]]:
    natural_cfg = config["natural"]
    seeds = range(int(natural_cfg.get("seed_start", 0)), int(natural_cfg.get("seed_stop", 1)))
    natural = natural_instances(
        seeds,
        blocker_counts=tuple(int(value) for value in natural_cfg.get("blocker_counts", [1, 2, 3])),
        robot_counts=tuple(int(value) for value in natural_cfg.get("robot_counts", [2, 3])),
    )
    diagnostic_cfg = config["diagnostic"]
    diagnostic: list[ScenarioInstance] = []
    for task in diagnostic_cfg.get("tasks", []):
        for seed in range(int(diagnostic_cfg.get("seeds_per_task", 1))):
            diagnostic.append(generate_diagnostic(task, seed))
    return natural, diagnostic


def _natural_worker(payload):
    instance, horizon, observation_radius, candidate_budget, max_arity = payload
    return evaluate_instance(
        instance,
        horizon=horizon,
        information_mode="exact_state_oracle",
        observation_radius=observation_radius,
        candidate_budget=candidate_budget,
        max_arity=max_arity,
    )


def _load_scene_records(path: Path) -> list[SceneResult]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    records: list[SceneResult] = []
    for row in rows:
        events = row.get("topology_events", ())
        row["topology_events"] = tuple(filter(None, events.split(";"))) if isinstance(events, str) else tuple(events)
        records.append(SceneResult(**row))
    return records


def _load_dataclass_records(path: Path, cls):
    return [cls(**row) for row in json.loads(path.read_text(encoding="utf-8"))]


def _evaluate_records(
    *,
    natural: list[ScenarioInstance],
    diagnostic: list[ScenarioInstance],
    config: dict[str, Any],
    horizon: int,
    candidate_budget: int,
    max_arity: int,
    observation_radius: int,
    log,
) -> tuple[list[SceneResult], list[BaselineResult], list[ScalingResult], list[dict[str, str]]]:
    scene_records: list[SceneResult] = []
    baseline_records: list[BaselineResult] = []
    failures: list[dict[str, str]] = []

    workers = max(1, int(config.get("execution", {}).get("workers", 1)))
    natural_payloads = [
        (instance, horizon, observation_radius, candidate_budget, max_arity)
        for instance in natural
    ]
    if workers == 1:
        for payload in natural_payloads:
            instance = payload[0]
            try:
                scene_records.append(_natural_worker(payload))
            except Exception as exc:
                failures.append({"instance_id": instance.instance_id, "stage": "natural", "error": repr(exc)})
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_instance = {executor.submit(_natural_worker, payload): payload[0] for payload in natural_payloads}
            for future in as_completed(future_to_instance):
                instance = future_to_instance[future]
                try:
                    scene_records.append(future.result())
                except Exception as exc:
                    failures.append({"instance_id": instance.instance_id, "stage": "natural", "error": repr(exc)})
        scene_records.sort(key=lambda item: (item.task_name, item.seed, item.information_mode))
    log(f"natural evaluation complete: {len(scene_records)} records")

    local_cfg = config.get("local_belief", {})
    local_tasks = set(local_cfg.get("tasks", []))
    local_modes = tuple(local_cfg.get("modes", ["symbolic_local_belief"]))
    for instance in diagnostic:
        try:
            exact_record = evaluate_instance(
                instance,
                horizon=horizon,
                information_mode="exact_state_oracle",
                observation_radius=observation_radius,
                candidate_budget=candidate_budget,
                max_arity=max_arity,
            )
            scene_records.append(exact_record)
            if not exact_record.mechanism_valid:
                failures.append({
                    "instance_id": instance.instance_id,
                    "stage": "diagnostic_mechanism",
                    "error": exact_record.mechanism_notes,
                })
            if instance.task_name in local_tasks:
                for information_mode in local_modes:
                    scene_records.append(
                        evaluate_instance(
                            instance,
                            horizon=horizon,
                            information_mode=information_mode,
                            observation_radius=observation_radius,
                            candidate_budget=candidate_budget,
                            max_arity=max_arity,
                        )
                    )
            baseline_records.extend(
                evaluate_baselines(
                    instance,
                    horizon=horizon,
                    candidate_budget=candidate_budget,
                    max_arity=max_arity,
                )
            )
        except Exception as exc:
            failures.append({"instance_id": instance.instance_id, "stage": "diagnostic", "error": repr(exc)})
    log(f"diagnostic evaluation complete: {len(scene_records)} scene records, {len(baseline_records)} baseline records")

    scaling_cfg = config.get("scaling", {})
    scaling_records: list[ScalingResult] = []
    scaling_records.extend(
        entity_scaling(
            tuple(int(value) for value in scaling_cfg.get("entity_counts", [1, 2, 3, 5, 8, 12])),
            robots=2,
            options_per_entity=int(scaling_cfg.get("options_per_entity", 4)),
        )
    )
    scaling_records.extend(
        robot_factor_scaling(
            tuple(int(value) for value in scaling_cfg.get("robot_counts", [2, 3, 4, 5, 6, 8])),
            tuple(int(value) for value in scaling_cfg.get("factor_arities", [1, 2, 3, 4])),
            candidates_per_robot=int(scaling_cfg.get("candidates_per_robot", 3)),
        )
    )
    log(f"scaling evaluation complete: {len(scaling_records)} records")
    return scene_records, baseline_records, scaling_records, failures



def _evaluate_main_experiment(
    config: dict[str, Any],
    *,
    max_arity: int,
    observation_radius: int,
    log,
):
    """Run the v0.5 CST/IET horizon and successor-support study."""

    cfg = config.get("main_experiment", {})
    if not cfg.get("enabled", True):
        return [], {}, [], [], [], [], []

    horizons = tuple(int(value) for value in cfg.get("horizons", [2, 3]))
    seed_start = int(cfg.get("seed_start", 0))
    seed_stop = int(cfg.get("seed_stop", 24))
    budget = int(cfg.get("candidate_budget", 8))
    max_steps = int(cfg.get("max_steps", 5))
    local_radius = int(cfg.get("observation_radius", observation_radius))
    failures: list[dict[str, str]] = []

    try:
        records, support = run_horizon_study(
            range(seed_start, seed_stop),
            horizons=horizons,
            observation_radius=local_radius,
            candidate_budget=budget,
            max_arity=max_arity,
            max_steps=max_steps,
        )
    except Exception as exc:
        failures.append({
            "instance_id": "local-imagination-study",
            "stage": "main_horizon_study",
            "error": repr(exc),
        })
        return [], {}, [], [], [], [], failures

    horizon_summary = summarize_horizon_study(records)
    dependency_summary = summarize_dependency_study(records)
    support_summary = summarize_support_study(support)

    showcase = generate_local_imagination_showcase(
        int(cfg.get("showcase_seed", 0))
    )
    traces = {}
    for mode, horizon, name in (
        (PlannerMode.COST, 3, "CST H=3"),
        (PlannerMode.IET, 2, "CoImagine-IET H=2"),
        (PlannerMode.IET, 3, "CoImagine-IET H=3"),
        (
            PlannerMode.PRIVILEGED_IET_ORACLE,
            3,
            "Privileged IET Oracle H=3",
        ),
    ):
        try:
            traces[name] = rollout_receding_horizon(
                showcase,
                mode=mode,
                horizon=horizon,
                candidate_budget=budget,
                max_arity=max_arity,
                max_steps=max_steps,
                method_name=name,
                observation_radius=local_radius,
            )
        except Exception as exc:
            failures.append({
                "instance_id": showcase.instance_id,
                "stage": f"showcase_{name}",
                "error": repr(exc),
            })

    log(
        "main Gate-A local-imagination study complete: "
        f"{len(records)} episode records, {len(support)} support records"
    )
    return records, traces, support, horizon_summary, dependency_summary, support_summary, failures


def run_gate_a(
    profile: str = "smoke",
    *,
    output_dir: str | Path | None = None,
    repo_root: Path | None = None,
) -> RunArtifacts:
    config = load_config(profile, repo_root=repo_root)
    root = repo_root or Path(__file__).resolve().parents[3]
    output = Path(output_dir) if output_dir is not None else root / config.get("output_dir", f"outputs/gate_a/{profile}")
    output.mkdir(parents=True, exist_ok=True)

    natural, diagnostic = _build_instances(config)
    planning_cfg = config.get("planning", {})
    horizon = int(planning_cfg.get("horizon", 2))
    candidate_budget = int(planning_cfg.get("candidate_budget", 48))
    max_arity = int(planning_cfg.get("max_arity", 4))
    observation_radius = int(config.get("local_belief", {}).get("observation_radius", 4))
    progress = bool(config.get("execution", {}).get("progress", False))

    def log(message: str) -> None:
        if progress:
            print(f"[CoImagine Gate A] {message}", flush=True)

    log(f"starting {len(natural)} natural and {len(diagnostic)} diagnostic instances")
    scene_path = output / "data" / "scene_results.json"
    baseline_path = output / "data" / "baseline_results.json"
    scaling_path = output / "data" / "scaling_results.json"
    resume = bool(config.get("execution", {}).get("resume", False))

    if resume and scene_path.exists() and baseline_path.exists() and scaling_path.exists():
        scene_records = _load_scene_records(scene_path)
        baseline_records = _load_dataclass_records(baseline_path, BaselineResult)
        scaling_records = _load_dataclass_records(scaling_path, ScalingResult)
        try:
            failures = json.loads((output / "data" / "failures.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            failures = []
        log(f"resumed {len(scene_records)} scene, {len(baseline_records)} baseline, and {len(scaling_records)} scaling records")
    else:
        scene_records, baseline_records, scaling_records, failures = _evaluate_records(
            natural=natural,
            diagnostic=diagnostic,
            config=config,
            horizon=horizon,
            candidate_budget=candidate_budget,
            max_arity=max_arity,
            observation_radius=observation_radius,
            log=log,
        )
        write_csv(output / "data" / "scene_results.csv", scene_records)
        write_json(scene_path, scene_records)
        write_csv(output / "data" / "baseline_results.csv", baseline_records)
        write_json(baseline_path, baseline_records)
        write_csv(output / "data" / "scaling_results.csv", scaling_records)
        write_json(scaling_path, scaling_records)
        (output / "data" / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        log("raw data written")


    (
        episode_records,
        showcase_traces,
        support_records,
        horizon_summary,
        dependency_summary,
        support_summary,
        main_failures,
    ) = _evaluate_main_experiment(
        config,
        max_arity=max_arity,
        observation_radius=observation_radius,
        log=log,
    )
    failures.extend(main_failures)
    write_csv(output / "data" / "episode_results.csv", episode_records)
    write_json(output / "data" / "episode_results.json", episode_records)
    write_csv(output / "data" / "successor_support.csv", support_records)
    write_json(output / "data" / "successor_support.json", support_records)
    write_csv(output / "tables" / "main" / "horizon_summary.csv", horizon_summary)
    write_markdown_table(
        output / "tables" / "main" / "horizon_summary.md", horizon_summary
    )
    write_csv(output / "tables" / "main" / "dependency_summary.csv", dependency_summary)
    write_markdown_table(
        output / "tables" / "main" / "dependency_summary.md", dependency_summary
    )
    write_csv(output / "tables" / "main" / "support_summary.csv", support_summary)
    write_markdown_table(
        output / "tables" / "main" / "support_summary.md", support_summary
    )

    scene_summary = summarize_scenes(scene_records)
    event_summary = summarize_events(scene_records)
    baseline_summary = summarize_baselines(baseline_records)
    write_csv(output / "tables" / "gate_a_summary.csv", scene_summary)
    write_markdown_table(output / "tables" / "gate_a_summary.md", scene_summary)
    write_latex_table(output / "tables" / "gate_a_summary.tex", scene_summary, "Gate-A exact structural value and local-belief diagnostics.", "tab:gate-a-summary")
    write_csv(output / "tables" / "event_summary.csv", event_summary)
    write_markdown_table(output / "tables" / "event_summary.md", event_summary)
    write_latex_table(output / "tables" / "event_summary.tex", event_summary, "Gate-A results stratified by exclusive realized topology event.", "tab:gate-a-events")
    write_csv(output / "tables" / "baseline_summary.csv", baseline_summary)
    write_markdown_table(output / "tables" / "baseline_summary.md", baseline_summary)
    write_latex_table(output / "tables" / "baseline_summary.tex", baseline_summary, "Controlled topology and factorization diagnostics.", "tab:gate-a-baselines")
    write_semantics_glossary(output / "tables")
    log("tables and glossary written")

    main_figure_dir = output / "figures" / "main"
    supplementary_figure_dir = output / "figures" / "supplementary"
    iet_h3 = showcase_traces.get("CoImagine-IET H=3")
    cost_h3 = showcase_traces.get("CST H=3")
    if iet_h3 is not None and cost_h3 is not None:
        plot_local_imagination_story(
            iet_h3,
            cost_h3,
            main_figure_dir,
            observation_radius=int(
                config.get("main_experiment", {}).get(
                    "observation_radius", observation_radius
                )
            ),
        )
    if horizon_summary:
        plot_horizon_summary(horizon_summary, main_figure_dir)
    if dependency_summary:
        plot_dependency_summary(dependency_summary, main_figure_dir)
    if support_summary:
        plot_support_summary(support_summary, main_figure_dir)
    if scaling_records:
        plot_structural_scaling(scaling_records, main_figure_dir)
    render_supplementary = bool(config.get("reporting", {}).get("render_supplementary", True))
    if render_supplementary:
        plot_value_summary(scene_records, supplementary_figure_dir)
        plot_event_summary(scene_records, supplementary_figure_dir)
        plot_scaling(scaling_records, supplementary_figure_dir)
    log("main-paper figures written" + ("; supplementary quantitative figures written" if render_supplementary else ""))

    exact_lookup = {
        (record.task_name, record.seed): record
        for record in scene_records
        if record.suite == "diagnostic" and record.information_mode == "exact_state_oracle"
    }
    examples: dict[str, ExampleBundle] = {}
    for instance in diagnostic:
        if instance.task_name in examples:
            continue
        record = exact_lookup.get((instance.task_name, instance.seed))
        if record is None:
            continue
        try:
            result = evaluate_gate_a(
                instance.state,
                horizon=horizon,
                candidate_budget=candidate_budget,
                max_arity=max_arity,
            )
            examples[instance.task_name] = ExampleBundle(instance, result, record)
        except Exception as exc:
            failures.append({"instance_id": instance.instance_id, "stage": "figure_trace", "error": repr(exc)})
    log(f"qualitative traces prepared: {len(examples)}")

    if render_supplementary:
        plot_qualitative_examples(examples, supplementary_figure_dir)
        plot_topology_examples(examples, supplementary_figure_dir, candidate_budget=candidate_budget, max_arity=max_arity)
        plot_factor_diagnostics(examples, baseline_records, supplementary_figure_dir)
        log("supplementary qualitative and topology figures written")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile": profile,
        "config": config,
        "natural_instances": len(natural),
        "diagnostic_instances": len(diagnostic),
        "scene_records": len(scene_records),
        "baseline_records": len(baseline_records),
        "scaling_records": len(scaling_records),
        "episode_records": len(episode_records),
        "successor_support_records": len(support_records),
        "main_horizons": config.get("main_experiment", {}).get("horizons", [2, 3]),
        "mechanism_invalid_count": sum(
            not record.mechanism_valid
            for record in scene_records
            if record.suite == "diagnostic" and record.information_mode == "exact_state_oracle"
        ),
        "failure_count": len(failures),
        "main_story_figure": "figures/main/fig_gate_a_local_imagination_story.pdf",
        "main_horizon_figure": "figures/main/fig_gate_a_horizon_comparison.pdf",
        "main_tables": "tables/main",
        "supplementary_figures": "figures/supplementary",
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output / "data" / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    return RunArtifacts(
        output_dir=output,
        scene_records=tuple(scene_records),
        baseline_records=tuple(baseline_records),
        scaling_records=tuple(scaling_records),
        episode_records=tuple(episode_records),
        failures=tuple(failures),
    )
