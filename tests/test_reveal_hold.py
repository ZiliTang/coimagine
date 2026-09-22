from coimagine.envs.grid.generation.procedural import generate_hold_pass, generate_reveal_act
from coimagine.evaluation.gate_a import evaluate_instance


def test_reveal_act_has_horizon_two_structural_value() -> None:
    result = evaluate_instance(generate_reveal_act(0), information_mode="exact_state_oracle")
    assert result.oracle_gain > 0.0
    assert "information_reveal" in result.topology_events


def test_hold_pass_survives_symbolic_local_observation() -> None:
    result = evaluate_instance(
        generate_hold_pass(0),
        information_mode="symbolic_local_belief",
        observation_radius=5,
    )
    assert result.oracle_gain > 0.0
