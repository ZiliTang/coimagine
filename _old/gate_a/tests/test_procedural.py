from coimagine.envs.grid.generation.procedural import (
    generate_diagnostic,
    generate_multi_blocker,
    generate_stable_control,
)
from coimagine.evaluation.gate_a import evaluate_baselines, evaluate_instance


def test_procedural_generation_is_deterministic() -> None:
    first = generate_multi_blocker(17, blockers=3, robots=2)
    second = generate_multi_blocker(17, blockers=3, robots=2)
    assert first.state == second.state
    assert first.instance_id == second.instance_id


def test_topology_stable_negative_control_has_no_oracle_gain() -> None:
    result = evaluate_instance(generate_stable_control(0), candidate_budget=24)
    assert abs(result.oracle_gain) < 1e-9


def test_pairwise_only_misses_positive_higher_order_synergy() -> None:
    rows = evaluate_baselines(generate_diagnostic("positive_higher_order_synergy", 0))
    by_mode = {row.mode: row for row in rows}
    assert by_mode["pairwise_only"].exact_root_recall < 1.0
    assert by_mode["recursive_factors"].exact_root_recall == 1.0
