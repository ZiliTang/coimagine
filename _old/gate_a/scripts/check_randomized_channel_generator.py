#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from statistics import mean, median

from coimagine.envs.grid.generation.procedural import (
    RANDOMIZED_CHANNEL_DIFFICULTIES,
    generate_randomized_channel_benchmark_case,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate randomized-channel causal structure and distributions."
    )
    parser.add_argument("--per-difficulty", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=30_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_difficulty < 1:
        raise ValueError("--per-difficulty must be positive")

    total = 0
    for difficulty in RANDOMIZED_CHANNEL_DIFFICULTIES:
        rows = []
        for offset in range(args.per_difficulty):
            seed = args.seed_start + offset
            instance = generate_randomized_channel_benchmark_case(
                seed, difficulty=difficulty
            )
            metadata = instance.state.task.metadata
            future = bool(metadata.get("future_support_required", False))
            valid = bool(
                metadata.get(
                    "future_support_structure_valid" if future else "root_supported_structure_valid",
                    False,
                )
            )
            if not valid:
                raise AssertionError(
                    f"Invalid structural label: {difficulty=} {seed=} {metadata=}"
                )
            if future:
                if metadata.get("future_support_validation_reason") != "strict_h3_chain":
                    raise AssertionError(
                        f"Future-support scene lacks strict H=3 chain: {difficulty=} {seed=}"
                    )
                if int(metadata.get("critical_root_slot_count", 0)) >= 2:
                    raise AssertionError(
                        f"Critical coalition is already root-supported: {difficulty=} {seed=}"
                    )
                if int(metadata.get("critical_new_slot_count", 0)) < 1:
                    raise AssertionError(
                        f"No newly grounded critical slot: {difficulty=} {seed=}"
                    )
            rows.append(metadata)
        total += len(rows)
        profile_counts = Counter(str(row.get("scene_subprofile", "")) for row in rows)
        position_counts = Counter(str(row.get("initial_position_mode", "")) for row in rows)
        blockers = [int(row.get("blocker_count", 0)) for row in rows]
        heavy = [int(row.get("heavy_blocker_count", 0)) for row in rows]
        occluders = [int(row.get("occluder_count", 0)) for row in rows]
        attempts = [int(row.get("generation_attempt", 0)) for row in rows]
        future_count = sum(bool(row.get("future_support_required", False)) for row in rows)
        activation_count = sum(
            int(row.get("activation_dependency_count", 0))
            if "activation_dependency_count" in row
            else 0
            for row in rows
        )
        print(
            f"{difficulty}: n={len(rows)} | future={future_count / len(rows):.1%} | "
            f"M={min(blockers)}-{max(blockers)} (mean {mean(blockers):.2f}) | "
            f"heavy mean={mean(heavy):.2f} | occluders mean={mean(occluders):.2f} | "
            f"attempt median/max={median(attempts):.0f}/{max(attempts)}"
        )
        print(f"  profiles={dict(profile_counts)}")
        print(f"  positions={dict(position_counts)}")
        if activation_count:
            print(f"  activation dependencies={activation_count}")

    print(f"randomized-channel structural validation passed for {total} scenes")


if __name__ == "__main__":
    main()
