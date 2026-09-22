#!/usr/bin/env python3
"""Deterministic semantic regression checks for Gate-A CST/IET support.

The checks isolate two requirements:
1. CST-Full may recover a root-observed template that was omitted by root top-K.
2. CST-Full never introduces a template absent from the complete root observation,
   while IET may introduce one after imagined successor observation.
"""

from __future__ import annotations

from coimagine.core.types import OptionKind, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import (
    generate_local_imagination_showcase,
    generate_randomized_channel_benchmark_case,
)
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch, SearchResult


def _template_counts(result: SearchResult) -> tuple[int, int]:
    recovered = 0
    imagined_new = 0
    for step in result.trace[1:]:
        for option in step.joint_option.options:
            support = result.root_support_templates.get(option.robot_id, frozenset())
            selected = result.root_selected_templates.get(option.robot_id, frozenset())
            if option.template_key in support and option.template_key not in selected:
                recovered += 1
            elif option.template_key not in support:
                imagined_new += 1
    return recovered, imagined_new


def _planner(instance, *, candidate_budget: int) -> tuple[ExactTopologySearch, object]:
    radius = int(instance.state.task.metadata.get("observation_radius", 6))
    planner = ExactTopologySearch(
        GridEnvironment(observation_radius=radius),
        GridOptionGenerator(candidate_budget=candidate_budget),
        horizon=3,
    )
    return planner, planner.initialize_beliefs(instance.state)


def check_side_aware_reposition_selection() -> None:
    instance = generate_randomized_channel_benchmark_case(30_000_000, difficulty="easy")
    planner, beliefs = _planner(instance, candidate_budget=8)
    boundary_rows = tuple(
        int(value)
        for value in instance.state.task.metadata["channel_boundary_rows"]
    )
    lower_boundary = min(boundary_rows)
    upper_boundary = max(boundary_rows)

    found_two_sided_pool = False
    for belief in beliefs.values():
        pool = planner.option_generator.generate_support_pool(belief)
        all_anchors = [
            option.target
            for option in pool
            if option.kind == OptionKind.REPOSITION and option.target is not None
        ]
        has_lower = any(anchor[1] < lower_boundary for anchor in all_anchors)
        has_upper = any(anchor[1] > upper_boundary for anchor in all_anchors)
        if not (has_lower and has_upper):
            continue
        found_two_sided_pool = True
        selected = planner.option_generator.select_candidates(belief, pool)
        selected_anchors = [
            option.target
            for option in selected
            if option.kind == OptionKind.REPOSITION and option.target is not None
        ]
        assert any(anchor[1] < lower_boundary for anchor in selected_anchors)
        assert any(anchor[1] > upper_boundary for anchor in selected_anchors)
        print(
            "[template-check] side-aware reposition selection passed: "
            f"robot={belief.robot_id}, selected={selected_anchors}"
        )
        break
    assert found_two_sided_pool, "deterministic test scene lacks a two-sided anchor pool"


def check_root_top_k_separation() -> None:
    instance = generate_randomized_channel_benchmark_case(30_000_000, difficulty="easy")
    planner, beliefs = _planner(instance, candidate_budget=4)
    support = planner._candidate_support_sets(beliefs)
    selected = planner._select_candidate_sets(beliefs, support)
    support_count = sum(len(options) for options in support.values())
    selected_count = sum(len(options) for options in selected.values())
    assert support_count > selected_count, (support_count, selected_count)

    frozen = planner.plan(instance.state, beliefs, mode=PlannerMode.COST_FROZEN_K)
    full = planner.plan(instance.state, beliefs, mode=PlannerMode.COST)
    frozen_recovered, frozen_new = _template_counts(frozen)
    full_recovered, full_new = _template_counts(full)

    assert frozen_recovered == 0
    assert frozen_new == 0
    assert full_recovered > 0, "CST-Full failed to recover a root-supported template"
    assert full_new == 0, "CST-Full introduced a non-root template"
    assert full.value > frozen.value + 1e-9, (frozen.value, full.value)
    print(
        "[template-check] root top-K separation passed: "
        f"support={support_count}, selected={selected_count}, "
        f"recovered={full_recovered}, values={frozen.value:.3f}->{full.value:.3f}"
    )


def check_future_observation_boundary() -> None:
    instance = generate_local_imagination_showcase(0)
    planner, beliefs = _planner(instance, candidate_budget=8)
    cst = planner.plan(instance.state, beliefs, mode=PlannerMode.COST)
    iet = planner.plan(instance.state, beliefs, mode=PlannerMode.IET)
    cst_recovered, cst_new = _template_counts(cst)
    iet_recovered, iet_new = _template_counts(iet)

    root_support = planner._candidate_support_sets(beliefs)
    root_selected = planner._select_candidate_sets(beliefs, root_support)
    root_structure = planner.admission.build_root_structure(
        instance.state,
        root_selected,
        support_candidates=root_support,
    )
    transition = planner.env.execute(instance.state, iet.joint_option)
    assert transition.result.success
    successor_beliefs = {
        robot_id: planner.belief_imaginator.predict(belief, iet.joint_option)
        for robot_id, belief in beliefs.items()
    }
    cst_candidates = planner._candidate_sets_for_depth(
        successor_beliefs,
        PlannerMode.COST,
        root_structure,
        depth=1,
    )
    iet_candidates = planner._candidate_sets_for_depth(
        successor_beliefs,
        PlannerMode.IET,
        root_structure,
        depth=1,
    )
    root_keys = {
        option.template_key
        for options in root_support.values()
        for option in options
    }
    cst_keys = {
        option.template_key
        for options in cst_candidates.values()
        for option in options
    }
    iet_keys = {
        option.template_key
        for options in iet_candidates.values()
        for option in options
    }

    assert cst_keys <= root_keys, "CST candidate pool escaped root support"
    assert iet_keys - root_keys, "IET did not add successor-observation support"
    assert cst_new == 0, "CST-Full introduced an unseen contact template"
    assert iet_new > 0, "IET failed to select future-observation-supported templates"
    assert iet.value > cst.value + 1e-9, (cst.value, iet.value)
    print(
        "[template-check] future-observation boundary passed: "
        f"CST(recovered={cst_recovered}, new={cst_new}, value={cst.value:.3f}); "
        f"IET(recovered={iet_recovered}, new={iet_new}, value={iet.value:.3f})"
    )


def main() -> None:
    check_side_aware_reposition_selection()
    check_root_top_k_separation()
    check_future_observation_boundary()
    print("Gate-A CST template-support regression checks passed.")


if __name__ == "__main__":
    main()
