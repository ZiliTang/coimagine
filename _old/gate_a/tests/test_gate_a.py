from coimagine.evaluation.gate_a import evaluate_gate_a
from coimagine.envs.grid.scenarios import two_robot_approach_then_push


def test_evolving_topology_has_oracle_advantage() -> None:
    result = evaluate_gate_a(two_robot_approach_then_push(), horizon=2)
    assert result.oracle_gain > 0.0
