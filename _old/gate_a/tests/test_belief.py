from coimagine.belief.symbolic import CellKnowledge, RuleBasedBeliefUpdater
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.scenarios import two_robot_approach_then_push


def test_private_belief_is_built_from_observation() -> None:
    state = two_robot_approach_then_push()
    env = GridEnvironment(observation_radius=2)
    updater = RuleBasedBeliefUpdater()
    observation = env.observe(state, 0)
    belief = updater.initialize(observation, state.width, state.height)
    assert belief.own_position == state.robots[0].position
    assert belief.cell_knowledge((9, 8)) == CellKnowledge.UNKNOWN
    assert "blocker-0" not in belief.blockers
