#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

from coimagine.core.types import PlannerMode
from coimagine.envs.grid.generation.procedural import (
    generate_local_imagination_channel_showcase,
    generate_local_imagination_showcase,
    generate_randomized_channel_benchmark_case,
)
from coimagine.evaluation.episodes import rollout_receding_horizon
from coimagine.evaluation.records import ScalingResult
from coimagine.evaluation.local_imagination import (
    run_stratified_horizon_study,
    run_randomized_channel_study,
    summarize_dependency_study,
    summarize_horizon_study,
    summarize_randomized_channel_study,
    summarize_randomized_channel_failures,
    summarize_paired_study,
    summarize_support_study,
)
from coimagine.evaluation.scaling import entity_scaling, robot_factor_scaling
from coimagine.reporting.io import write_csv, write_json
from coimagine.reporting.progress import progress_message
from coimagine.reporting.local_gate_a import (
    plot_channel_imagination_story,
    plot_dependency_summary,
    plot_episode_distributions,
    plot_horizon_summary,
    plot_local_imagination_story,
    plot_paired_summary,
    plot_randomized_channel_debug_gallery,
    plot_randomized_channel_summary,
    plot_structural_scaling,
    plot_support_summary,
)


def _write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No records.\n", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append("—" if value != value else f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_failure_log(path: Path, records) -> None:
    """Write one compact, grep-friendly line for every failed episode."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for row in records:
        if row.completed:
            continue
        lines.append(
            " | ".join(
                (
                    f"seed={row.seed}",
                    f"difficulty={row.difficulty_variant}",
                    f"profile={row.scene_subprofile or '-'}",
                    f"method={row.method}",
                    f"reason={row.termination_reason}",
                    f"detail={row.failure_detail or '-'}",
                    f"progress={row.final_potential:.4f}",
                    f"cleared={row.cleared_entity_count}/{row.entity_count}",
                    f"steps={row.option_count}/{row.episode_budget}",
                    f"K={row.candidate_budget}",
                    f"candidate_saturated_steps={row.candidate_saturated_steps}",
                    f"root_admitted_min={row.root_admitted_min}",
                    f"root_support_mean={row.root_support_count_mean:.2f}",
                    f"root_selected_mean={row.root_candidate_count_mean:.2f}",
                    f"root_truncated_mean={row.root_truncated_template_count_mean:.2f}",
                    f"recovered_root_templates={row.predicted_recovered_root_template_count}",
                    f"imagined_new_templates={row.predicted_new_template_count}",
                    f"structurally_validated={row.structurally_validated}",
                    f"causal_valid={row.future_support_structure_valid}",
                    f"causal_reason={row.future_support_validation_reason or '-'}",
                    f"critical_slots={row.critical_root_slot_count}->{row.critical_successor_slot_count}",
                    f"idle_steps={row.idle_joint_steps}",
                    f"no_progress_steps={row.no_progress_steps}",
                )
            )
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _read_json_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON row list in {path}")
    return [dict(row) for row in payload]


def _render_existing_results(
    output: Path,
    *,
    seed_start: int,
    progress: bool,
    skip_debug_gallery: bool = False,
) -> None:
    """Regenerate figures from tables already written by a completed simulation.

    This recovery path is intentionally independent of episode execution, so a
    plotting failure never forces the user to rerun hundreds of exact-search
    episodes.
    """

    progress_message("[render-only] Loading existing JSON summaries.", enabled=progress)
    random_summary = _read_json_rows(output / "random_channel_summary.json")
    if not random_summary:
        raise FileNotFoundError(
            f"Missing or empty {output / 'random_channel_summary.json'}"
        )
    plot_randomized_channel_summary(random_summary, output / "figures")

    scaling_rows = _read_json_rows(output / "structural_scaling.json")
    if scaling_rows:
        scaling = [ScalingResult(**row) for row in scaling_rows]
        plot_structural_scaling(scaling, output / "figures")
    horizon = _read_json_rows(output / "horizon_summary.json")
    dependency = _read_json_rows(output / "dependency_summary.json")
    paired = _read_json_rows(output / "paired_summary.json")
    support = _read_json_rows(output / "support_summary.json")
    if horizon:
        plot_horizon_summary(horizon, output / "figures")
    if dependency:
        plot_dependency_summary(dependency, output / "figures")
    if paired:
        plot_paired_summary(paired, output / "figures")
    if support:
        plot_support_summary(support, output / "figures")

    if not skip_debug_gallery:
        debug_difficulties = ("easy", "moderate", "hard")
        debug_instances = [
            generate_randomized_channel_benchmark_case(
                seed_start + 30_000_000 + index,
                difficulty=debug_difficulties[index % len(debug_difficulties)],
            )
            for index in range(12)
        ]
        plot_randomized_channel_debug_gallery(
            debug_instances, output / "figures", columns=4
        )
    progress_message(f"[done] Figures regenerated in {output / 'figures'}", enabled=progress)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v0.13 controlled and heterogeneous Gate-A CST/IET benchmarks."
    )
    parser.add_argument("--open-per-family", type=int, default=100)
    parser.add_argument("--surprise-cases", type=int, default=100)
    parser.add_argument("--random-channel-per-difficulty", type=int, default=50)
    parser.add_argument(
        "--random-channel-candidate-budget",
        type=int,
        default=5,
        help="Fixed local-option budget (default: 5); use 0 to use each generated scene budget.",
    )
    parser.add_argument(
        "--random-channel-profile",
        choices=("main", "full", "template-diagnostic"),
        default="main",
        help=(
            "main: CST-Full/IET H=3; full: H=2/H=3 plus oracle; "
            "template-diagnostic: CST-Frozen-K/CST-Full/IET H=3."
        ),
    )
    parser.add_argument(
        "--random-channel-max-steps-cap",
        type=int,
        default=None,
        help="Optional step cap. The benchmark always preserves the 3M feasibility floor.",
    )
    parser.add_argument(
        "--random-channel-steps-per-object",
        type=int,
        default=3,
        help=(
            "Episode-budget floor per blocker (default: 3). Values below 3 are "
            "rejected because one blocker may need Reposition-Approach-Push."
        ),
    )
    parser.add_argument(
        "--random-channel-extra-steps",
        type=int,
        default=None,
        help=(
            "Exact additive slack beyond steps-per-object x M. When omitted, use "
            "the scene-generated recovery slack."
        ),
    )
    parser.add_argument(
        "--random-channel-episode-budget",
        type=int,
        default=None,
        help=(
            "Optional absolute episode budget. It overrides generated/additive "
            "slack but cannot violate the per-object feasibility floor."
        ),
    )
    parser.add_argument("--skip-controlled", action="store_true")
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=4)
    progress_group = parser.add_mutually_exclusive_group()
    progress_group.add_argument(
        "--progress",
        dest="progress",
        action="store_true",
        default=True,
        help="Show live case/episode progress, rates, elapsed time, and ETA (default).",
    )
    progress_group.add_argument(
        "--no-progress",
        dest="progress",
        action="store_false",
        help="Disable live progress output.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress records when stderr is redirected (default: 5).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 8 open instances per family and 8 surprise cases for a smoke run.",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Regenerate figures from existing JSON summaries without rerunning episodes.",
    )
    parser.add_argument(
        "--skip-story-figures",
        action="store_true",
        help="Skip the expensive qualitative CST/IET story rollouts and figures.",
    )
    parser.add_argument(
        "--skip-debug-gallery",
        action="store_true",
        help="Skip the 12-scene randomized debug gallery (useful for fast smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/gate_a/v013_randomized"),
    )
    args = parser.parse_args()

    if args.random_channel_candidate_budget < 0:
        parser.error("--random-channel-candidate-budget must be nonnegative")
    if args.random_channel_max_steps_cap is not None and args.random_channel_max_steps_cap < 1:
        parser.error("--random-channel-max-steps-cap must be positive")
    if args.random_channel_steps_per_object < 3:
        parser.error("--random-channel-steps-per-object must be at least 3")
    if args.random_channel_extra_steps is not None and args.random_channel_extra_steps < 0:
        parser.error("--random-channel-extra-steps must be nonnegative")
    if args.random_channel_episode_budget is not None and args.random_channel_episode_budget < 1:
        parser.error("--random-channel-episode-budget must be positive")
    if args.render_only:
        _render_existing_results(
            args.output,
            seed_start=args.seed_start,
            progress=args.progress,
            skip_debug_gallery=args.skip_debug_gallery,
        )
        return

    open_per_family = 8 if args.quick else args.open_per_family
    surprise_cases = 8 if args.quick else args.surprise_cases
    random_channel_per_difficulty = 3 if args.quick else args.random_channel_per_difficulty
    controlled_cases = 0 if args.skip_controlled else open_per_family * 4 + surprise_cases
    random_cases = random_channel_per_difficulty * 3
    random_episodes_per_case = {
        "main": 2,
        "full": 5,
        "template-diagnostic": 3,
    }[args.random_channel_profile]
    progress_message(
        (
            f"[setup] jobs={args.jobs} | controlled={controlled_cases} cases | "
            f"random-channel={random_cases} cases x {random_episodes_per_case} episodes | "
            f"K={'scene' if args.random_channel_candidate_budget == 0 else args.random_channel_candidate_budget} | "
            f"episode-budget="
            f"{args.random_channel_episode_budget or f'{args.random_channel_steps_per_object}M+' + (str(args.random_channel_extra_steps) if args.random_channel_extra_steps is not None else 'scene')} | "
            f"step-cap={args.random_channel_max_steps_cap or 'none'} | "
            f"output={args.output}"
        ),
        enabled=args.progress,
    )
    started = perf_counter()
    if args.skip_controlled:
        records, support = [], []
    else:
        records, support = run_stratified_horizon_study(
            open_per_family=open_per_family,
            surprise_cases=surprise_cases,
            seed_start=args.seed_start,
            jobs=args.jobs,
            progress=args.progress,
            progress_interval_seconds=args.progress_interval,
        )
    elapsed = perf_counter() - started

    progress_message("[phase] Starting heterogeneous randomized-channel suite.", enabled=args.progress)
    random_started = perf_counter()
    random_channel_records = run_randomized_channel_study(
        cases_per_difficulty=random_channel_per_difficulty,
        seed_start=args.seed_start + 20_000_000,
        candidate_budget=args.random_channel_candidate_budget,
        jobs=args.jobs,
        profile=args.random_channel_profile,
        max_steps_cap=args.random_channel_max_steps_cap,
        steps_per_object=args.random_channel_steps_per_object,
        extra_steps=args.random_channel_extra_steps,
        episode_budget=args.random_channel_episode_budget,
        progress=args.progress,
        progress_interval_seconds=args.progress_interval,
    )
    random_elapsed = perf_counter() - random_started

    progress_message("[phase] Aggregating statistics and structural scaling.", enabled=args.progress)
    horizon_summary = summarize_horizon_study(records) if records else []
    dependency_summary = summarize_dependency_study(records) if records else []
    paired_summary = summarize_paired_study(records) if records else []
    support_summary = summarize_support_study(support) if support else []
    random_channel_summary = summarize_randomized_channel_study(random_channel_records)
    random_channel_failure_summary = summarize_randomized_channel_failures(
        random_channel_records
    )
    scaling_records = entity_scaling(
        (1, 2, 3, 5, 8, 12, 16, 24, 32),
        robots=2,
        options_per_entity=4,
    )
    scaling_records.extend(
        robot_factor_scaling(
            (2, 3, 4, 5, 6, 8, 10, 12),
            (2, 3, 4),
            candidates_per_robot=3,
        )
    )

    progress_message("[phase] Writing CSV, JSON, and Markdown tables.", enabled=args.progress)
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "episode_results.csv", records)
    write_csv(args.output / "random_channel_episode_results.csv", random_channel_records)
    write_json(args.output / "random_channel_episode_results.json", random_channel_records)
    write_csv(args.output / "random_channel_summary.csv", random_channel_summary)
    write_json(args.output / "random_channel_summary.json", random_channel_summary)
    failed_random_records = [row for row in random_channel_records if not row.completed]
    write_csv(
        args.output / "random_channel_failure_diagnostics.csv", failed_random_records
    )
    write_json(
        args.output / "random_channel_failure_diagnostics.json", failed_random_records
    )
    write_csv(
        args.output / "random_channel_failure_summary.csv",
        random_channel_failure_summary,
    )
    write_json(
        args.output / "random_channel_failure_summary.json",
        random_channel_failure_summary,
    )
    _write_failure_log(
        args.output / "random_channel_failures.log", random_channel_records
    )
    write_json(args.output / "episode_results.json", records)
    write_csv(args.output / "successor_support.csv", support)
    write_json(args.output / "successor_support.json", support)
    write_csv(args.output / "horizon_summary.csv", horizon_summary)
    write_json(args.output / "horizon_summary.json", horizon_summary)
    write_csv(args.output / "dependency_summary.csv", dependency_summary)
    write_json(args.output / "dependency_summary.json", dependency_summary)
    write_csv(args.output / "paired_summary.csv", paired_summary)
    write_json(args.output / "paired_summary.json", paired_summary)
    write_csv(args.output / "support_summary.csv", support_summary)
    write_json(args.output / "support_summary.json", support_summary)
    write_csv(args.output / "structural_scaling.csv", scaling_records)
    write_json(args.output / "structural_scaling.json", scaling_records)
    _write_markdown(args.output / "horizon_summary.md", horizon_summary)
    _write_markdown(args.output / "dependency_summary.md", dependency_summary)
    _write_markdown(args.output / "paired_summary.md", paired_summary)
    _write_markdown(args.output / "support_summary.md", support_summary)
    _write_markdown(args.output / "random_channel_summary.md", random_channel_summary)
    _write_markdown(
        args.output / "random_channel_failure_summary.md",
        random_channel_failure_summary,
    )
    progress_message(
        (
            f"[diagnostics] failed random-channel episodes={len(failed_random_records)}; "
            f"details={args.output / 'random_channel_failures.log'}"
        ),
        enabled=args.progress,
    )

    progress_message("[phase] Rendering publication and debug figures.", enabled=args.progress)
    if records:
        plot_horizon_summary(horizon_summary, args.output / "figures")
        plot_dependency_summary(dependency_summary, args.output / "figures")
        plot_paired_summary(paired_summary, args.output / "figures")
        plot_episode_distributions(records, args.output / "figures")
    if support_summary:
        plot_support_summary(support_summary, args.output / "figures")
    plot_structural_scaling(scaling_records, args.output / "figures")
    plot_randomized_channel_summary(random_channel_summary, args.output / "figures")
    if not args.skip_debug_gallery:
        debug_difficulties = ("easy", "moderate", "hard")
        debug_instances = [
            generate_randomized_channel_benchmark_case(
                args.seed_start + 30_000_000 + index,
                difficulty=debug_difficulties[index % len(debug_difficulties)],
            )
            for index in range(12)
        ]
        plot_randomized_channel_debug_gallery(
            debug_instances, args.output / "figures", columns=4
        )

    iet = cst = None
    if not args.skip_story_figures:
        progress_message("[phase] Rendering CST/IET qualitative story figures.", enabled=args.progress)
        showcase = generate_local_imagination_showcase(0)
        iet = rollout_receding_horizon(
            showcase,
            mode=PlannerMode.IET,
            horizon=3,
            candidate_budget=8,
            max_arity=3,
            max_steps=5,
            method_name="CoImagine-IET H=3",
            observation_radius=6,
        )
        cst = rollout_receding_horizon(
            showcase,
            mode=PlannerMode.COST,
            horizon=3,
            candidate_budget=8,
            max_arity=3,
            max_steps=5,
            method_name="CST H=3",
            observation_radius=6,
        )
        plot_local_imagination_story(
            iet,
            cst,
            args.output / "figures",
            observation_radius=6,
        )
        plot_channel_imagination_story(
            generate_local_imagination_channel_showcase(0),
            args.output / "figures",
            observation_radius=10,
        )

    manifest = {
        "benchmark_version": "v0.13",
        "open_per_family": open_per_family,
        "open_families": 4,
        "open_instances": 0 if args.skip_controlled else open_per_family * 4,
        "surprise_instances": 0 if args.skip_controlled else surprise_cases,
        "controlled_instances": controlled_cases,
        "random_channel_instances": random_channel_per_difficulty * 3,
        "total_instances": controlled_cases + random_channel_per_difficulty * 3,
        "controlled_episode_records": len(records),
        "random_channel_episode_records": len(random_channel_records),
        "episode_records": len(records) + len(random_channel_records),
        "support_records": len(support),
        "horizons": [2, 3],
        "methods": {
            "main": ["CST", "CoImagine-IET"],
            "full": ["CST", "CoImagine-IET", "Privileged IET Oracle"],
            "template-diagnostic": ["CST-Frozen-K", "CST", "CoImagine-IET"],
        }[args.random_channel_profile],
        "jobs": args.jobs,
        "seed_start": args.seed_start,
        "controlled_elapsed_seconds": elapsed,
        "random_channel_elapsed_seconds": random_elapsed,
        "random_channel_per_difficulty": random_channel_per_difficulty,
        "random_channel_profile": args.random_channel_profile,
        "random_channel_candidate_budget": args.random_channel_candidate_budget,
        "random_channel_max_steps_cap": args.random_channel_max_steps_cap,
        "random_channel_steps_per_object": args.random_channel_steps_per_object,
        "random_channel_extra_steps": args.random_channel_extra_steps,
        "random_channel_episode_budget": args.random_channel_episode_budget,
        "random_channel_step_floor": f"{args.random_channel_steps_per_object}M (minimum 3M)",
        "skip_controlled": args.skip_controlled,
        "skip_story_figures": args.skip_story_figures,
        "skip_debug_gallery": args.skip_debug_gallery,
        "outcome_screening": False,
        "headline_split": "open",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    report = {
        "manifest": manifest,
        "horizon_summary": horizon_summary,
        "support_summary": support_summary,
        "paired_summary": paired_summary,
        "structural_scaling_records": len(scaling_records),
        "showcase": (
            {
                "iet_completed": iet.completed,
                "iet_option_count": iet.option_count,
                "cst_completed": cst.completed,
                "cst_option_count": cst.option_count,
                "channel_story_figure": "figures/fig_gate_a_channel_imagination_story.pdf",
            }
            if iet is not None and cst is not None
            else {"skipped": True}
        ),
    }
    (args.output / "study_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    progress_message(f"[done] Results written to {args.output}", enabled=args.progress)
    print(json.dumps(report, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
