from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from math import comb, sqrt
from multiprocessing import get_context
from statistics import mean, median
from typing import Iterable, Mapping, Sequence

import numpy as np

from coimagine.core.types import JointOption, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import (
    LOCAL_IMAGINATION_FAMILIES,
    ScenarioInstance,
    generate_local_imagination_benchmark_case,
    generate_local_imagination_benchmark_variant,
    generate_local_imagination_variant,
    generate_randomized_channel_benchmark_case,
)
from coimagine.evaluation.episodes import rollout_receding_horizon, to_episode_record
from coimagine.evaluation.records import EpisodeResultRecord
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch
from coimagine.reporting.progress import ProgressReporter


@dataclass(frozen=True)
class SuccessorSupportRecord:
    instance_id: str
    seed: int
    horizon: int
    surprise_obstacle: bool
    scene_type: str
    scene_family: str
    dependency_depth: int
    difficulty_variant: str
    selected_root_action: str
    root_support_template_count: int
    root_selected_template_count: int
    root_truncated_template_count: int
    optimistic_new_option_count: int
    oracle_new_option_count: int
    cost_new_option_count: int
    cost_recovered_root_option_count: int
    frozen_recovered_root_option_count: int
    option_false_positive_count: int
    new_option_precision: float
    new_option_recall: float | None
    optimistic_new_edge_count: int
    oracle_new_edge_count: int
    cost_new_edge_count: int
    edge_false_positive_count: int
    new_edge_precision: float
    new_edge_recall: float | None
    iet_matches_oracle_root: bool
    cost_matches_oracle_root: bool
    frozen_matches_oracle_root: bool
    cost_root_value: float
    frozen_root_value: float
    iet_root_value: float
    oracle_root_value: float
    retained_oracle_gain: float | None

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


def _precision_recall(
    predicted: set[tuple], reference: set[tuple]
) -> tuple[float, float | None]:
    overlap = len(predicted & reference)
    precision = overlap / len(predicted) if predicted else (1.0 if not reference else 0.0)
    recall = overlap / len(reference) if reference else None
    return precision, recall


def _candidate_template_set(candidates) -> set[tuple]:
    return {
        option.template_key
        for options in candidates.values()
        for option in options
    }


def _edge_template_set(admission) -> set[tuple]:
    return {joint.template_key for joint in admission.joint_options}


def successor_support_diagnostics(
    instance: ScenarioInstance,
    *,
    horizon: int = 3,
    observation_radius: int = 6,
    candidate_budget: int = 8,
    max_arity: int = 3,
) -> SuccessorSupportRecord:
    """Compare optimistic and privileged successor support on IET's root branch."""

    state = instance.state.clone()
    env = GridEnvironment(observation_radius=observation_radius)
    generator = GridOptionGenerator(candidate_budget=candidate_budget)
    planner = ExactTopologySearch(
        env,
        generator,
        horizon=horizon,
        max_arity=max_arity,
    )
    beliefs = planner.initialize_beliefs(state)
    root_support = planner._candidate_support_sets(beliefs)
    root_candidates = planner._select_candidate_sets(beliefs, root_support)
    root_structure = planner.admission.build_root_structure(
        state,
        root_candidates,
        support_candidates=root_support,
    )

    iet = planner.plan(state, beliefs, mode=PlannerMode.IET)
    cost = planner.plan(state, beliefs, mode=PlannerMode.COST)
    frozen = planner.plan(state, beliefs, mode=PlannerMode.COST_FROZEN_K)
    oracle = planner.plan(state, beliefs, mode=PlannerMode.PRIVILEGED_IET_ORACLE)
    root_joint: JointOption = iet.joint_option
    transition = env.execute(state, root_joint)
    if not transition.result.success:
        raise RuntimeError("IET selected a root action that failed exact execution")

    optimistic_beliefs = {
        robot_id: planner.belief_imaginator.predict(belief, root_joint)
        for robot_id, belief in beliefs.items()
    }
    oracle_beliefs = {
        robot_id: planner.belief_updater.update(
            belief,
            env.observe(transition.state, robot_id),
            source="privileged_successor_observation",
        )
        for robot_id, belief in beliefs.items()
    }
    optimistic_candidates = planner._candidate_sets_for_depth(
        optimistic_beliefs,
        PlannerMode.IET,
        root_structure,
        depth=1,
    )
    oracle_candidates = planner._candidate_sets_for_depth(
        oracle_beliefs,
        PlannerMode.PRIVILEGED_IET_ORACLE,
        root_structure,
        depth=1,
    )
    cost_candidates = planner._candidate_sets_for_depth(
        optimistic_beliefs,
        PlannerMode.COST,
        root_structure,
        depth=1,
    )
    frozen_candidates = planner._candidate_sets_for_depth(
        optimistic_beliefs,
        PlannerMode.COST_FROZEN_K,
        root_structure,
        depth=1,
    )

    root_templates = _candidate_template_set(root_support)
    selected_root_templates = _candidate_template_set(root_candidates)
    optimistic_new = _candidate_template_set(optimistic_candidates) - root_templates
    oracle_new = _candidate_template_set(oracle_candidates) - root_templates
    cost_template_set = _candidate_template_set(cost_candidates)
    frozen_template_set = _candidate_template_set(frozen_candidates)
    cost_new = cost_template_set - root_templates
    cost_recovered = (cost_template_set & root_templates) - selected_root_templates
    frozen_recovered = (frozen_template_set & root_templates) - selected_root_templates
    option_precision, option_recall = _precision_recall(optimistic_new, oracle_new)

    root_admission = planner._admit_at_depth(
        state,
        beliefs,
        root_candidates,
        PlannerMode.IET,
        root_structure,
        depth=0,
    )
    optimistic_admission = planner._admit_at_depth(
        transition.state,
        optimistic_beliefs,
        optimistic_candidates,
        PlannerMode.IET,
        root_structure,
        depth=1,
    )
    oracle_admission = planner._admit_at_depth(
        transition.state,
        oracle_beliefs,
        oracle_candidates,
        PlannerMode.PRIVILEGED_IET_ORACLE,
        root_structure,
        depth=1,
    )
    cost_admission = planner._admit_at_depth(
        transition.state,
        optimistic_beliefs,
        optimistic_candidates,
        PlannerMode.COST,
        root_structure,
        depth=1,
    )
    root_edges = _edge_template_set(root_admission)
    optimistic_edges = _edge_template_set(optimistic_admission) - root_edges
    oracle_edges = _edge_template_set(oracle_admission) - root_edges
    cost_edges = _edge_template_set(cost_admission) - root_edges
    edge_precision, edge_recall = _precision_recall(optimistic_edges, oracle_edges)

    denominator = oracle.value - cost.value
    retained = (
        (iet.value - cost.value) / denominator
        if abs(denominator) > 1e-12
        else None
    )
    return SuccessorSupportRecord(
        instance_id=instance.instance_id,
        seed=instance.seed,
        horizon=horizon,
        surprise_obstacle=bool(state.task.metadata.get("surprise_obstacle", False)),
        scene_type=(
            "surprise"
            if bool(state.task.metadata.get("surprise_obstacle", False))
            else "open"
        ),
        scene_family=str(state.task.metadata.get("scene_family", instance.task_name)),
        dependency_depth=int(state.task.metadata.get("dependency_depth", 0)),
        difficulty_variant=str(state.task.metadata.get("difficulty_variant", "")),
        selected_root_action=str(root_joint.template_key),
        root_support_template_count=len(root_templates),
        root_selected_template_count=len(selected_root_templates),
        root_truncated_template_count=max(0, len(root_templates) - len(selected_root_templates)),
        optimistic_new_option_count=len(optimistic_new),
        oracle_new_option_count=len(oracle_new),
        cost_new_option_count=len(cost_new),
        cost_recovered_root_option_count=len(cost_recovered),
        frozen_recovered_root_option_count=len(frozen_recovered),
        option_false_positive_count=len(optimistic_new - oracle_new),
        new_option_precision=option_precision,
        new_option_recall=option_recall,
        optimistic_new_edge_count=len(optimistic_edges),
        oracle_new_edge_count=len(oracle_edges),
        cost_new_edge_count=len(cost_edges),
        edge_false_positive_count=len(optimistic_edges - oracle_edges),
        new_edge_precision=edge_precision,
        new_edge_recall=edge_recall,
        iet_matches_oracle_root=iet.joint_option.stable_key == oracle.joint_option.stable_key,
        cost_matches_oracle_root=cost.joint_option.stable_key == oracle.joint_option.stable_key,
        frozen_matches_oracle_root=frozen.joint_option.stable_key == oracle.joint_option.stable_key,
        cost_root_value=cost.value,
        frozen_root_value=frozen.value,
        iet_root_value=iet.value,
        oracle_root_value=oracle.value,
        retained_oracle_gain=retained,
    )


def _run_instance_methods(
    instance: ScenarioInstance,
    *,
    horizons: Sequence[int],
    candidate_budget: int,
    max_arity: int,
    max_steps: int,
) -> tuple[list[EpisodeResultRecord], SuccessorSupportRecord | None]:
    observation_radius = int(instance.state.task.metadata.get("observation_radius", 6))
    methods = (
        (PlannerMode.COST, "CST"),
        (PlannerMode.IET, "CoImagine-IET"),
        (PlannerMode.PRIVILEGED_IET_ORACLE, "Privileged IET Oracle"),
    )
    records: list[EpisodeResultRecord] = []
    for horizon in horizons:
        for mode, name in methods:
            trace = rollout_receding_horizon(
                instance,
                mode=mode,
                horizon=int(horizon),
                candidate_budget=candidate_budget,
                max_arity=max_arity,
                max_steps=max_steps,
                method_name=f"{name} H={horizon}",
                observation_radius=observation_radius,
            )
            records.append(
                to_episode_record(
                    instance,
                    trace,
                    observation_radius=observation_radius,
                )
            )
    support = None
    if int(instance.state.task.metadata.get("dependency_depth", 0)) >= 3:
        support = successor_support_diagnostics(
            instance,
            horizon=max(horizons),
            observation_radius=observation_radius,
            candidate_budget=candidate_budget,
            max_arity=max_arity,
        )
    return records, support


def _stratified_worker(
    spec: tuple[str, int, str, tuple[int, ...], int, int, int]
) -> tuple[list[EpisodeResultRecord], SuccessorSupportRecord | None]:
    family, seed, scene_type, horizons, candidate_budget, max_arity, max_steps = spec
    instance = generate_local_imagination_benchmark_case(
        seed,
        family=family,
        scene_type=scene_type,
    )
    return _run_instance_methods(
        instance,
        horizons=horizons,
        candidate_budget=candidate_budget,
        max_arity=max_arity,
        max_steps=max_steps,
    )


def run_horizon_study(
    seeds: Iterable[int],
    *,
    horizons: Sequence[int] = (2, 3),
    observation_radius: int = 6,
    candidate_budget: int = 8,
    max_arity: int = 3,
    max_steps: int = 5,
) -> tuple[list[EpisodeResultRecord], list[SuccessorSupportRecord]]:
    """Backward-compatible v0.7 mixed-cycle study."""

    episode_records: list[EpisodeResultRecord] = []
    support_records: list[SuccessorSupportRecord] = []
    for seed in seeds:
        instance = generate_local_imagination_benchmark_variant(int(seed))
        # Preserve the public observation-radius argument for legacy calls.
        instance.state.task.metadata["observation_radius"] = observation_radius
        rows, support = _run_instance_methods(
            instance,
            horizons=horizons,
            candidate_budget=candidate_budget,
            max_arity=max_arity,
            max_steps=max_steps,
        )
        episode_records.extend(rows)
        if support is not None:
            support_records.append(support)
    return episode_records, support_records


def run_stratified_horizon_study(
    *,
    open_per_family: int = 100,
    surprise_cases: int = 100,
    seed_start: int = 0,
    horizons: Sequence[int] = (2, 3),
    candidate_budget: int = 10,
    max_arity: int = 3,
    max_steps: int = 6,
    jobs: int = 1,
    progress: bool = False,
    progress_interval_seconds: float = 5.0,
) -> tuple[list[EpisodeResultRecord], list[SuccessorSupportRecord]]:
    """Run the v0.8 family-stratified benchmark without outcome screening.

    Every open causal family receives exactly ``open_per_family`` independent
    seeds.  Surprise cases are sampled separately so their proportion does not
    alter the headline open-task completion rate.
    """

    if open_per_family < 1:
        raise ValueError("open_per_family must be positive")
    if surprise_cases < 0:
        raise ValueError("surprise_cases cannot be negative")
    if jobs < 1:
        raise ValueError("jobs must be positive")

    horizon_tuple = tuple(int(value) for value in horizons)
    specs: list[tuple[str, int, str, tuple[int, ...], int, int, int]] = []
    for family_index, family in enumerate(LOCAL_IMAGINATION_FAMILIES):
        offset = seed_start + family_index * 1_000_000
        for local_index in range(open_per_family):
            specs.append(
                (
                    family,
                    offset + local_index,
                    "open",
                    horizon_tuple,
                    candidate_budget,
                    max_arity,
                    max_steps,
                )
            )
    surprise_offset = seed_start + 10_000_000
    for local_index in range(surprise_cases):
        specs.append(
            (
                "reposition_approach_push",
                surprise_offset + local_index,
                "surprise",
                horizon_tuple,
                candidate_budget,
                max_arity,
                max_steps,
            )
        )

    reporter = ProgressReporter(
        len(specs),
        label="controlled",
        unit="case",
        enabled=progress,
        line_interval_seconds=progress_interval_seconds,
    )
    episode_records: list[EpisodeResultRecord] = []
    support_records: list[SuccessorSupportRecord] = []
    executor = None
    try:
        if jobs == 1:
            for spec in specs:
                rows, support = _stratified_worker(spec)
                episode_records.extend(rows)
                if support is not None:
                    support_records.append(support)
                reporter.update(detail=f"{len(episode_records)} episodes")
        else:
            executor = ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn"))
            futures = [executor.submit(_stratified_worker, spec) for spec in specs]
            for future in as_completed(futures):
                rows, support = future.result()
                episode_records.extend(rows)
                if support is not None:
                    support_records.append(support)
                reporter.update(detail=f"{len(episode_records)} episodes")
        reporter.close(detail=f"{len(episode_records)} episodes")
    except BaseException:
        reporter.close(status="failed", detail=f"{len(episode_records)} episodes")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    episode_records.sort(
        key=lambda row: (row.scene_type, row.scene_family, row.seed, row.method, row.horizon)
    )
    support_records.sort(key=lambda row: (row.scene_type, row.seed))
    return episode_records, support_records



def resolve_randomized_channel_budgets(
    instance: ScenarioInstance,
    *,
    candidate_budget: int,
    max_steps_cap: int | None,
    steps_per_object: int = 3,
    extra_steps: int | None = None,
    episode_budget: int | None = None,
) -> tuple[int, int, int]:
    """Resolve candidate, effective episode, and minimum feasibility budgets.

    ``steps_per_object`` controls the explicit feasibility floor.  The benchmark
    never allows fewer than three boundaries per object because a blocker may
    require Reposition -> Approach -> Push.  ``extra_steps`` replaces the
    generated slack, while ``episode_budget`` provides an absolute override.
    The legacy ``max_steps_cap`` is applied last and cannot violate the floor.
    """

    metadata = instance.state.task.metadata
    effective_candidate_budget = (
        int(candidate_budget)
        if int(candidate_budget) > 0
        else int(metadata.get("candidate_budget", 8))
    )
    object_count = len(instance.state.blockers)
    generated_steps = int(metadata.get("episode_max_steps", 8))
    generated_minimum = max(
        3 * len(instance.state.blockers),
        int(metadata.get("minimum_episode_steps", 0)),
    )
    minimum_steps = max(generated_minimum, int(steps_per_object) * object_count)
    generated_slack = max(0, generated_steps - generated_minimum)
    if episode_budget is not None:
        max_steps = max(minimum_steps, int(episode_budget))
    elif extra_steps is not None:
        max_steps = minimum_steps + int(extra_steps)
    else:
        max_steps = minimum_steps + generated_slack
    if max_steps_cap is not None:
        # The CLI cap is allowed to tighten surplus slack but never invalidate the
        # 3M feasibility floor promised by the benchmark definition.
        max_steps = max(minimum_steps, min(max_steps, int(max_steps_cap)))
    return effective_candidate_budget, max_steps, minimum_steps


def _randomized_channel_worker(
    spec: tuple[
        int,
        str,
        int,
        int,
        str,
        int | None,
        int,
        int | None,
        int | None,
    ]
) -> list[EpisodeResultRecord]:
    (
        seed,
        difficulty,
        candidate_budget,
        max_arity,
        profile,
        max_steps_cap,
        steps_per_object,
        extra_steps,
        episode_budget,
    ) = spec
    instance = generate_randomized_channel_benchmark_case(seed, difficulty=difficulty)
    metadata = dict(instance.state.task.metadata)
    observation_radius = int(metadata.get("observation_radius", 6))
    effective_candidate_budget, max_steps, minimum_steps = resolve_randomized_channel_budgets(
        instance,
        candidate_budget=candidate_budget,
        max_steps_cap=max_steps_cap,
        steps_per_object=steps_per_object,
        extra_steps=extra_steps,
        episode_budget=episode_budget,
    )
    metadata.update(
        {
            "candidate_budget": effective_candidate_budget,
            "episode_max_steps": max_steps,
            "minimum_episode_steps": minimum_steps,
            "requested_max_steps_cap": max_steps_cap,
            "requested_steps_per_object": steps_per_object,
            "requested_extra_steps": -1 if extra_steps is None else extra_steps,
            "requested_episode_budget": 0 if episode_budget is None else episode_budget,
        }
    )
    task = replace(instance.state.task, metadata=metadata)
    state = replace(instance.state, task=task)
    instance = replace(instance, state=state)
    # The general suite focuses on the main H=3 configuration while retaining
    # matched H=2 CST/IET controls.  The privileged oracle is evaluated at H=3.
    if profile == "main":
        settings = (
            (PlannerMode.COST, "CST", 3),
            (PlannerMode.IET, "CoImagine-IET", 3),
        )
    elif profile == "full":
        settings = (
            (PlannerMode.COST, "CST", 2),
            (PlannerMode.IET, "CoImagine-IET", 2),
            (PlannerMode.COST, "CST", 3),
            (PlannerMode.IET, "CoImagine-IET", 3),
            (PlannerMode.PRIVILEGED_IET_ORACLE, "Privileged IET Oracle", 3),
        )
    elif profile == "template-diagnostic":
        settings = (
            (PlannerMode.COST_FROZEN_K, "CST-Frozen-K", 3),
            (PlannerMode.COST, "CST", 3),
            (PlannerMode.IET, "CoImagine-IET", 3),
        )
    else:
        raise ValueError(f"Unsupported randomized-channel profile: {profile}")
    records: list[EpisodeResultRecord] = []
    for mode, name, horizon in settings:
        trace = rollout_receding_horizon(
            instance,
            mode=mode,
            horizon=horizon,
            candidate_budget=effective_candidate_budget,
            max_arity=max_arity,
            max_steps=max_steps,
            method_name=f"{name} H={horizon}",
            observation_radius=observation_radius,
        )
        records.append(
            to_episode_record(instance, trace, observation_radius=observation_radius)
        )
    return records


def run_randomized_channel_study(
    *,
    cases_per_difficulty: int = 50,
    seed_start: int = 20_000_000,
    candidate_budget: int = 5,
    max_arity: int = 3,
    jobs: int = 1,
    profile: str = "main",
    max_steps_cap: int | None = None,
    steps_per_object: int = 3,
    extra_steps: int | None = None,
    episode_budget: int | None = None,
    progress: bool = False,
    progress_interval_seconds: float = 5.0,
) -> list[EpisodeResultRecord]:
    """Run the heterogeneous v0.13 multi-blocker channel benchmark.

    The suite is sampled independently from the controlled causal families and
    never screened by planner outcome. Each difficulty stratum receives an
    identical number of seeds. ``candidate_budget <= 0`` uses each scene's
    generated budget. The explicit episode budget is resolved from
    ``steps_per_object`` and ``extra_steps`` (or ``episode_budget``), and a supplied
    cap cannot reduce the 3M feasibility floor.
    """

    if cases_per_difficulty < 1:
        raise ValueError("cases_per_difficulty must be positive")
    if jobs < 1:
        raise ValueError("jobs must be positive")
    if profile not in {"main", "full", "template-diagnostic"}:
        raise ValueError("profile must be main, full, or template-diagnostic")
    if steps_per_object < 3:
        raise ValueError("steps_per_object must be at least 3")
    if extra_steps is not None and extra_steps < 0:
        raise ValueError("extra_steps must be nonnegative")
    if episode_budget is not None and episode_budget < 1:
        raise ValueError("episode_budget must be positive")
    difficulties = ("easy", "moderate", "hard")
    specs: list[
        tuple[int, str, int, int, str, int | None, int, int | None, int | None]
    ] = []
    for index, difficulty in enumerate(difficulties):
        offset = seed_start + index * 1_000_000
        for local_index in range(cases_per_difficulty):
            specs.append(
                (
                    offset + local_index,
                    difficulty,
                    candidate_budget,
                    max_arity,
                    profile,
                    max_steps_cap,
                    steps_per_object,
                    extra_steps,
                    episode_budget,
                )
            )

    reporter = ProgressReporter(
        len(specs),
        label="random-channel",
        unit="case",
        enabled=progress,
        line_interval_seconds=progress_interval_seconds,
    )
    records: list[EpisodeResultRecord] = []
    executor = None
    try:
        if jobs == 1:
            for spec in specs:
                rows = _randomized_channel_worker(spec)
                records.extend(rows)
                reporter.update(detail=f"{len(records)} episodes")
        else:
            executor = ProcessPoolExecutor(max_workers=jobs, mp_context=get_context("spawn"))
            futures = [executor.submit(_randomized_channel_worker, spec) for spec in specs]
            for future in as_completed(futures):
                rows = future.result()
                records.extend(rows)
                reporter.update(detail=f"{len(records)} episodes")
        reporter.close(detail=f"{len(records)} episodes")
    except BaseException:
        reporter.close(status="failed", detail=f"{len(records)} episodes")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    records.sort(key=lambda row: (row.difficulty_variant, row.seed, row.method, row.horizon))
    return records


def summarize_randomized_channel_study(
    records: Sequence[EpisodeResultRecord],
) -> list[dict[str, object]]:
    """Aggregate completion and partial-progress statistics by difficulty."""

    rows: list[dict[str, object]] = []
    for difficulty in ("all", "easy", "moderate", "hard"):
        groups: dict[tuple[str, int], list[EpisodeResultRecord]] = {}
        for record in records:
            if difficulty != "all" and record.difficulty_variant != difficulty:
                continue
            key = (record.method.rsplit(" H=", 1)[0], record.horizon)
            groups.setdefault(key, []).append(record)
        for (method, horizon), group in sorted(groups.items()):
            runs = len(group)
            successes = sum(int(row.completed) for row in group)
            ci_low, ci_high = _wilson_interval(successes, runs)
            potential = _distribution([row.final_potential for row in group])
            options = _distribution([float(row.option_count) for row in group])
            timing = _distribution([1000.0 * row.mean_planning_seconds for row in group])
            rows.append(
                {
                    "difficulty": difficulty,
                    "method": method,
                    "horizon": horizon,
                    "runs": runs,
                    "completion_rate": successes / runs if runs else 0.0,
                    "completion_ci_low": ci_low,
                    "completion_ci_high": ci_high,
                    "final_potential_median": potential["median"],
                    "final_potential_q25": potential["q25"],
                    "final_potential_q75": potential["q75"],
                    "option_count_median": options["median"],
                    "planning_ms_median": timing["median"],
                    "planning_ms_q25": timing["q25"],
                    "planning_ms_q75": timing["q75"],
                }
            )
    return rows


def summarize_randomized_channel_failures(
    records: Sequence[EpisodeResultRecord],
) -> list[dict[str, object]]:
    """Summarize why nonterminal randomized-channel episodes stopped."""

    rows: list[dict[str, object]] = []
    for difficulty in ("all", "easy", "moderate", "hard"):
        method_groups: dict[tuple[str, int], list[EpisodeResultRecord]] = {}
        for record in records:
            if difficulty != "all" and record.difficulty_variant != difficulty:
                continue
            key = (record.method.rsplit(" H=", 1)[0], record.horizon)
            method_groups.setdefault(key, []).append(record)
        for (method, horizon), group in sorted(method_groups.items()):
            failures = [row for row in group if not row.completed]
            reason_counts: dict[tuple[str, str], int] = {}
            for record in failures:
                key = (record.termination_reason or "unknown", record.failure_detail or "")
                reason_counts[key] = reason_counts.get(key, 0) + 1
            for (reason, detail), count in sorted(reason_counts.items()):
                rows.append(
                    {
                        "difficulty": difficulty,
                        "method": method,
                        "horizon": horizon,
                        "runs": len(group),
                        "failures": len(failures),
                        "termination_reason": reason,
                        "failure_detail": detail,
                        "count": count,
                        "fraction_of_failures": count / len(failures) if failures else 0.0,
                        "fraction_of_runs": count / len(group) if group else 0.0,
                    }
                )
    return rows

def _wilson_interval(successes: int, runs: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if runs <= 0:
        return 0.0, 0.0
    rate = successes / runs
    denominator = 1.0 + z * z / runs
    centre = (rate + z * z / (2.0 * runs)) / denominator
    half = z * sqrt(rate * (1.0 - rate) / runs + z * z / (4.0 * runs * runs)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q25": float(np.percentile(array, 25)),
        "q75": float(np.percentile(array, 75)),
    }


def summarize_horizon_study(
    records: Sequence[EpisodeResultRecord],
) -> list[dict[str, object]]:
    """Aggregate results with Wilson intervals and robust distribution summaries."""

    rows: list[dict[str, object]] = []
    for scene_type in ("all", "open", "surprise"):
        groups: dict[tuple[str, int], list[EpisodeResultRecord]] = {}
        for record in records:
            if scene_type != "all" and record.scene_type != scene_type:
                continue
            key = (record.method.rsplit(" H=", 1)[0], record.horizon)
            groups.setdefault(key, []).append(record)
        for (method, horizon), group in sorted(groups.items()):
            if not group:
                continue
            runs = len(group)
            successes = sum(int(row.completed) for row in group)
            ci_low, ci_high = _wilson_interval(successes, runs)
            final_potential = _distribution([row.final_potential for row in group])
            realized_return = _distribution([row.realized_return for row in group])
            option_count = _distribution([float(row.option_count) for row in group])
            duration = _distribution([row.total_duration for row in group])
            planning = _distribution([row.mean_planning_seconds for row in group])
            rows.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "scene_type": scene_type,
                    "runs": runs,
                    "successes": successes,
                    "completion_rate": successes / runs,
                    "completion_ci95_low": ci_low,
                    "completion_ci95_high": ci_high,
                    "mean_final_potential": final_potential["mean"],
                    "median_final_potential": final_potential["median"],
                    "mean_realized_return": realized_return["mean"],
                    "median_realized_return": realized_return["median"],
                    "return_q25": realized_return["q25"],
                    "return_q75": realized_return["q75"],
                    "mean_option_count": option_count["mean"],
                    "median_option_count": option_count["median"],
                    "option_count_q25": option_count["q25"],
                    "option_count_q75": option_count["q75"],
                    "mean_total_duration": duration["mean"],
                    "median_total_duration": duration["median"],
                    "mean_planning_seconds": planning["mean"],
                    "median_planning_seconds": planning["median"],
                    "planning_q25_seconds": planning["q25"],
                    "planning_q75_seconds": planning["q75"],
                }
            )
    return rows


def summarize_dependency_study(
    records: Sequence[EpisodeResultRecord],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, int, str, int, str], list[EpisodeResultRecord]] = {}
    for record in records:
        key = (
            record.method.rsplit(" H=", 1)[0],
            record.horizon,
            record.scene_family,
            record.dependency_depth,
            record.scene_type,
        )
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, object]] = []
    for (method, horizon, family, depth, scene_type), group in sorted(groups.items()):
        runs = len(group)
        successes = sum(int(row.completed) for row in group)
        ci_low, ci_high = _wilson_interval(successes, runs)
        return_stats = _distribution([row.realized_return for row in group])
        option_stats = _distribution([float(row.option_count) for row in group])
        planning_stats = _distribution([row.mean_planning_seconds for row in group])
        rows.append(
            {
                "method": method,
                "horizon": horizon,
                "scene_family": family,
                "dependency_depth": depth,
                "scene_type": scene_type,
                "runs": runs,
                "successes": successes,
                "completion_rate": successes / runs,
                "completion_ci95_low": ci_low,
                "completion_ci95_high": ci_high,
                "mean_realized_return": return_stats["mean"],
                "median_realized_return": return_stats["median"],
                "mean_option_count": option_stats["mean"],
                "median_option_count": option_stats["median"],
                "mean_planning_seconds": planning_stats["mean"],
                "median_planning_seconds": planning_stats["median"],
            }
        )
    return rows


def _mcnemar_exact(b: int, c: int) -> float:
    discordant = b + c
    if discordant == 0:
        return 1.0
    tail = sum(comb(discordant, index) for index in range(0, min(b, c) + 1)) / (2.0 ** discordant)
    return min(1.0, 2.0 * tail)


def _paired_bootstrap_ci(values: Sequence[float], *, seed: int = 7, repeats: int = 5000) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(repeats, len(array)))
    samples = np.mean(array[indices], axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def summarize_paired_study(
    records: Sequence[EpisodeResultRecord],
) -> list[dict[str, object]]:
    """Paired comparisons on identical randomized instances."""

    comparisons = (
        ("CoImagine-IET", 3, "CST", 3, "IET H=3 vs CST H=3"),
        ("CoImagine-IET", 3, "CoImagine-IET", 2, "IET H=3 vs IET H=2"),
        ("Privileged IET Oracle", 3, "CoImagine-IET", 3, "Oracle H=3 vs IET H=3"),
    )
    by_key = {
        (row.instance_id, row.method.rsplit(" H=", 1)[0], row.horizon): row
        for row in records
    }
    instance_scene = {row.instance_id: row.scene_type for row in records}
    rows: list[dict[str, object]] = []
    for scene_type in ("all", "open", "surprise"):
        instance_ids = sorted(
            instance_id
            for instance_id, value in instance_scene.items()
            if scene_type == "all" or value == scene_type
        )
        for a_method, a_h, b_method, b_h, label in comparisons:
            pairs = []
            for instance_id in instance_ids:
                a = by_key.get((instance_id, a_method, a_h))
                b = by_key.get((instance_id, b_method, b_h))
                if a is not None and b is not None:
                    pairs.append((a, b))
            if not pairs:
                continue
            a_only = sum(int(a.completed and not b.completed) for a, b in pairs)
            b_only = sum(int(b.completed and not a.completed) for a, b in pairs)
            completion_deltas = [float(a.completed) - float(b.completed) for a, b in pairs]
            ci_low, ci_high = _paired_bootstrap_ci(completion_deltas)
            return_deltas = [a.realized_return - b.realized_return for a, b in pairs]
            option_deltas = [float(a.option_count - b.option_count) for a, b in pairs]
            planning_ratios = [
                a.mean_planning_seconds / max(1e-12, b.mean_planning_seconds)
                for a, b in pairs
            ]
            rows.append(
                {
                    "comparison": label,
                    "scene_type": scene_type,
                    "pairs": len(pairs),
                    "a_only_completed": a_only,
                    "b_only_completed": b_only,
                    "completion_delta": mean(completion_deltas),
                    "completion_delta_ci95_low": ci_low,
                    "completion_delta_ci95_high": ci_high,
                    "mcnemar_exact_p": _mcnemar_exact(a_only, b_only),
                    "mean_return_delta": mean(return_deltas),
                    "median_return_delta": median(return_deltas),
                    "mean_option_count_delta": mean(option_deltas),
                    "median_option_count_delta": median(option_deltas),
                    "median_planning_time_ratio": median(planning_ratios),
                }
            )
    return rows


def summarize_support_study(
    records: Sequence[SuccessorSupportRecord],
) -> list[dict[str, object]]:
    groups: dict[str, list[SuccessorSupportRecord]] = {}
    for record in records:
        groups.setdefault(record.scene_type, []).append(record)

    def optional_mean(values):
        valid = [value for value in values if value is not None]
        return mean(valid) if valid else None

    rows: list[dict[str, object]] = []
    for scene_type, group in sorted(groups.items()):
        finite_retained = [
            row.retained_oracle_gain
            for row in group
            if row.retained_oracle_gain is not None
        ]
        rows.append(
            {
                "scene_type": scene_type,
                "runs": len(group),
                "root_action_agreement": mean(
                    float(row.iet_matches_oracle_root) for row in group
                ),
                "cost_root_action_agreement": mean(
                    float(row.cost_matches_oracle_root) for row in group
                ),
                "future_option_precision": mean(
                    row.new_option_precision for row in group
                ),
                "future_option_recall": optional_mean(
                    row.new_option_recall for row in group
                ),
                "future_edge_precision": mean(
                    row.new_edge_precision for row in group
                ),
                "future_edge_recall": optional_mean(
                    row.new_edge_recall for row in group
                ),
                "mean_optimistic_new_options": mean(
                    row.optimistic_new_option_count for row in group
                ),
                "mean_oracle_new_options": mean(
                    row.oracle_new_option_count for row in group
                ),
                "mean_cost_new_options": mean(
                    row.cost_new_option_count for row in group
                ),
                "mean_option_false_positives": mean(
                    row.option_false_positive_count for row in group
                ),
                "mean_edge_false_positives": mean(
                    row.edge_false_positive_count for row in group
                ),
                "retained_oracle_gain": (
                    mean(finite_retained) if finite_retained else None
                ),
            }
        )
    return rows
