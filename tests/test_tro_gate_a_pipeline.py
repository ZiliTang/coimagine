from pathlib import Path

from coimagine.core.types import OptionKind, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import (
    generate_channel_clearing_showcase,
)
from coimagine.evaluation.episodes import rollout_receding_horizon
from coimagine.options.generator import GridOptionGenerator
from coimagine.options.library import UNIVERSAL_OPTION_LIBRARY


def test_universal_option_library_is_shared_and_complete() -> None:
    generator = GridOptionGenerator(candidate_budget=8)
    assert generator.option_library is UNIVERSAL_OPTION_LIBRARY
    assert {item.kind for item in UNIVERSAL_OPTION_LIBRARY} == {
        OptionKind.REPOSITION,
        OptionKind.APPROACH,
        OptionKind.PUSH,
        OptionKind.HOLD,
        OptionKind.OBSERVE,
        OptionKind.IDLE,
    }


def test_showcase_evolving_changes_first_action_and_completes() -> None:
    instance = generate_channel_clearing_showcase(0)
    evolving = rollout_receding_horizon(
        instance,
        mode=PlannerMode.EVOLVING,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
    )
    root_reuse = rollout_receding_horizon(
        instance,
        mode=PlannerMode.ROOT_STRUCTURE_REUSE,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
    )
    assert evolving.completed
    assert evolving.final_potential == 1.0
    assert evolving.option_count < root_reuse.option_count
    assert not root_reuse.completed
    assert root_reuse.final_potential < 0.25
    assert evolving.first_action is not None and root_reuse.first_action is not None
    assert evolving.first_action.template_key != root_reuse.first_action.template_key
    assert evolving.steps[0].topology_delta.created_edge_count > 0


def test_myopic_control_stalls_while_h2_succeeds() -> None:
    instance = generate_channel_clearing_showcase(0)
    h1 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.EVOLVING,
        horizon=1,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
    )
    h2 = rollout_receding_horizon(
        instance,
        mode=PlannerMode.EVOLVING,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
    )
    assert not h1.completed
    assert h2.completed
    assert h2.final_potential > h1.final_potential


def test_symbolic_local_belief_preserves_showcase_decision() -> None:
    instance = generate_channel_clearing_showcase(0)
    exact = rollout_receding_horizon(
        instance,
        mode=PlannerMode.EVOLVING,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
    )
    local = rollout_receding_horizon(
        instance,
        mode=PlannerMode.EVOLVING,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
        observation_radius=5,
    )
    assert local.completed
    assert local.first_action is not None and exact.first_action is not None
    assert local.first_action.template_key == exact.first_action.template_key


def test_rear_contact_options_activate_only_after_heavy_gate_clears() -> None:
    instance = generate_channel_clearing_showcase(0)
    env = GridEnvironment(observation_radius=30)
    generator = GridOptionGenerator(candidate_budget=12)
    state = instance.state
    root_belief = GridEnvironment(observation_radius=30).observe(state, 2)
    from coimagine.belief.symbolic import RuleBasedBeliefUpdater
    root_options = generator.generate(RuleBasedBeliefUpdater().initialize(root_belief, state.width, state.height))
    assert not any(option.entity_id == 'light-rear' for option in root_options)

    trace = rollout_receding_horizon(
        instance,
        mode=PlannerMode.EVOLVING,
        horizon=2,
        candidate_budget=8,
        max_arity=3,
        max_steps=8,
    )
    # After the two-robot heavy push, the rear blocker is no longer dependency-locked.
    state_after_heavy = trace.steps[1].state_after
    rear_observation = GridEnvironment(observation_radius=30).observe(state_after_heavy, 2)
    rear_options = generator.generate(RuleBasedBeliefUpdater().initialize(rear_observation, state_after_heavy.width, state_after_heavy.height))
    assert any(option.entity_id == 'light-rear' for option in rear_options)
    assert env.task_potential(state_after_heavy) > env.task_potential(state)
