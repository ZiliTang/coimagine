from __future__ import annotations

from itertools import combinations

from coimagine.evaluation.records import ScalingResult


def entity_scaling(
    blocker_counts: tuple[int, ...],
    *,
    robots: int = 2,
    options_per_entity: int = 4,
) -> list[ScalingResult]:
    results = []
    for entities in blocker_counts:
        local_candidates = entities * options_per_entity
        naive = local_candidates**robots
        factor_checks = entities * (options_per_entity**robots)
        results.append(
            ScalingResult(
                study="entity_scaling",
                x_label="entities",
                x_value=entities,
                robot_count=robots,
                factor_arity=robots,
                candidate_count=local_candidates,
                naive_count=naive,
                factor_count=factor_checks,
                recursive_calls=factor_checks,
                retained_tuples=factor_checks,
                oracle_recall=1.0,
            )
        )
    return results


def _same_role_executable(tuple_items: tuple[tuple[int, int], ...], arity: int) -> bool:
    return len(tuple_items) == arity and len({role for _, role in tuple_items}) == 1


def _extendable(
    selected: tuple[tuple[int, int], ...],
    *,
    robot_count: int,
    candidates_per_robot: int,
    arity: int,
) -> bool:
    """Analytic extendability for the synthetic same-role factor benchmark.

    Every remaining robot has one candidate for every role. A selected tuple is
    therefore extendable iff all selected candidates share one role and enough
    unused robots remain to reach the target arity. This is equivalent to the
    previous exhaustive completion search but avoids an exponential inner loop.
    """

    del candidates_per_robot  # encoded by the guaranteed same role in every pool
    selected_robots = {robot for robot, _ in selected}
    roles = {role for _, role in selected}
    needed = arity - len(selected)
    return (
        needed > 0
        and len(roles) == 1
        and robot_count - len(selected_robots) >= needed
    )


def robot_factor_scaling(
    robot_counts: tuple[int, ...],
    arities: tuple[int, ...],
    *,
    candidates_per_robot: int = 3,
) -> list[ScalingResult]:
    results: list[ScalingResult] = []
    for robot_count in robot_counts:
        for arity in arities:
            if arity > robot_count:
                continue
            current = [((robot, role),) for robot in range(robot_count) for role in range(candidates_per_robot)]
            calls = 0
            retained = 0
            executable = set()
            seen = set(current)
            for order in range(1, arity + 1):
                next_level = []
                for selected in current:
                    calls += 1
                    canonical = tuple(sorted(selected))
                    if _same_role_executable(canonical, arity):
                        executable.add(canonical)
                    can_extend = _extendable(
                        canonical,
                        robot_count=robot_count,
                        candidates_per_robot=candidates_per_robot,
                        arity=arity,
                    )
                    if can_extend:
                        retained += 1
                    if order < arity and can_extend:
                        used = {robot for robot, _ in canonical}
                        for robot in range(robot_count):
                            if robot in used:
                                continue
                            for role in range(candidates_per_robot):
                                expanded = tuple(sorted(canonical + ((robot, role),)))
                                if expanded not in seen:
                                    seen.add(expanded)
                                    next_level.append(expanded)
                current = next_level

            full_tuples = candidates_per_robot**robot_count
            oracle_completions = candidates_per_robot * max(1, len(tuple(combinations(range(robot_count), arity))))
            recall = 1.0 if executable else 0.0
            results.append(
                ScalingResult(
                    study="robot_factor_scaling",
                    x_label="robots",
                    x_value=robot_count,
                    robot_count=robot_count,
                    factor_arity=arity,
                    candidate_count=robot_count * candidates_per_robot,
                    naive_count=full_tuples,
                    factor_count=1,
                    recursive_calls=calls,
                    retained_tuples=retained,
                    oracle_recall=recall,
                )
            )
    return results
