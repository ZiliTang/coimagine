#!/usr/bin/env python3
"""Regression check for semantics-preserving Gate-A search optimizations."""

from __future__ import annotations

import math
import time

from coimagine.core.types import PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import generate_randomized_channel_benchmark_case
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch


def _planner(instance, *, optimized: bool) -> ExactTopologySearch:
    radius = int(instance.state.task.metadata.get("observation_radius", 8))
    env = GridEnvironment(observation_radius=radius)
    generator = GridOptionGenerator(
        candidate_budget=4,
        enable_runtime_cache=optimized,
    )
    return ExactTopologySearch(
        env,
        generator,
        horizon=2,
        max_arity=3,
        enable_runtime_cache=optimized,
        enable_successor_dominance=optimized,
    )


def _assert_same(reference, optimized, label: str) -> None:
    if reference.joint_option.stable_key != optimized.joint_option.stable_key:
        raise AssertionError(
            f"{label}: root action mismatch\n"
            f"reference={reference.joint_option.stable_key}\n"
            f"optimized={optimized.joint_option.stable_key}"
        )
    if not math.isclose(reference.value, optimized.value, rel_tol=0.0, abs_tol=1e-12):
        raise AssertionError(
            f"{label}: value mismatch {reference.value} != {optimized.value}"
        )
    if set(reference.action_values) != set(optimized.action_values):
        raise AssertionError(f"{label}: admitted root action set changed")
    for key, value in reference.action_values.items():
        if not math.isclose(value, optimized.action_values[key], rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                f"{label}: root action value changed for {key}: "
                f"{value} != {optimized.action_values[key]}"
            )


def main() -> None:
    seeds = (31_000_000, 31_000_001, 31_000_002)
    modes = (PlannerMode.COST_FROZEN_K, PlannerMode.COST, PlannerMode.IET)
    reference_seconds = 0.0
    optimized_seconds = 0.0
    aggregate_stats: dict[str, int] = {}

    for index, seed in enumerate(seeds):
        difficulty = ("easy", "moderate", "hard")[index]
        instance = generate_randomized_channel_benchmark_case(
            seed,
            difficulty=difficulty,
        )
        for mode in modes:
            reference_planner = _planner(instance, optimized=False)
            start = time.perf_counter()
            reference = reference_planner.plan(instance.state.clone(), mode=mode)
            reference_seconds += time.perf_counter() - start

            optimized_planner = _planner(instance, optimized=True)
            start = time.perf_counter()
            result = optimized_planner.plan(instance.state.clone(), mode=mode)
            optimized_seconds += time.perf_counter() - start
            _assert_same(reference, result, f"seed={seed}, mode={mode.value}")
            for key, value in result.runtime_stats.items():
                aggregate_stats[key] = aggregate_stats.get(key, 0) + int(value)

    speedup = reference_seconds / max(optimized_seconds, 1e-12)
    print(
        "[safe-search-check] exact agreement passed for "
        f"{len(seeds) * len(modes)} plans"
    )
    print(
        f"[safe-search-check] reference={reference_seconds:.3f}s | "
        f"optimized={optimized_seconds:.3f}s | speedup={speedup:.2f}x"
    )
    selected = {
        key: value
        for key, value in sorted(aggregate_stats.items())
        if value > 0
        and (
            "hits" in key
            or "pruned" in key
            or "reuses" in key
        )
    }
    print(f"[safe-search-check] cache/pruning stats: {selected}")


if __name__ == "__main__":
    main()
