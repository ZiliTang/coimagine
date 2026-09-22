from __future__ import annotations

from coimagine.core.types import OptionKind, PlannerMode
from coimagine.envs.grid.generation.procedural import generate_diagnostic
from coimagine.evaluation.gate_a import evaluate_baselines, evaluate_instance


def test_all_diagnostic_generators_meet_named_mechanism() -> None:
    tasks = (
        "edge_creation",
        "edge_destruction",
        "edge_replacement",
        "higher_order_conflict",
        "positive_higher_order_synergy",
        "hold_pass",
        "reveal_act",
    )
    for task in tasks:
        result = evaluate_instance(
            generate_diagnostic(task, 0),
            horizon=2,
            candidate_budget=32,
            max_arity=4,
        )
        assert result.mechanism_valid, (task, result.mechanism_notes)


def test_reveal_act_has_competing_first_action_and_two_step_continuation() -> None:
    result = evaluate_instance(
        generate_diagnostic("reveal_act", 0),
        horizon=2,
        candidate_budget=32,
        max_arity=4,
    )
    assert result.first_action_reversal
    assert result.first_action_regret > 0.0
    assert "Observe" in result.evolving_root_action
    assert "visible-decoy" in result.fixed_root_action


def test_higher_order_conflict_is_pairwise_false_positive() -> None:
    rows = {
        row.mode: row
        for row in evaluate_baselines(
            generate_diagnostic("higher_order_conflict", 0),
            horizon=2,
            candidate_budget=32,
            max_arity=4,
        )
    }
    pairwise = rows[PlannerMode.PAIRWISE_ONLY.value]
    recursive = rows[PlannerMode.RECURSIVE_FACTORS.value]
    assert pairwise.exact_root_recall == 1.0
    assert pairwise.exact_root_precision < 1.0
    assert recursive.exact_root_recall == 1.0
    assert recursive.exact_root_precision == 1.0


def test_positive_synergy_is_pairwise_false_negative() -> None:
    rows = {
        row.mode: row
        for row in evaluate_baselines(
            generate_diagnostic("positive_higher_order_synergy", 0),
            horizon=2,
            candidate_budget=32,
            max_arity=4,
        )
    }
    assert rows[PlannerMode.PAIRWISE_ONLY.value].exact_root_recall < 1.0
    assert rows[PlannerMode.RECURSIVE_FACTORS.value].exact_root_recall == 1.0


def test_temporal_event_classes_are_exclusive() -> None:
    destruction = evaluate_instance(generate_diagnostic("edge_destruction", 0))
    replacement = evaluate_instance(generate_diagnostic("edge_replacement", 0))
    assert destruction.dominant_event == "pure_edge_destruction"
    assert replacement.dominant_event == "mixed_edge_turnover"
    assert destruction.created_edge_count == 0
    assert replacement.created_edge_count > 0 and replacement.destroyed_edge_count > 0
