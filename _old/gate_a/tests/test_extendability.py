from coimagine.belief.symbolic import RuleBasedBeliefUpdater
from coimagine.core.types import OptionKind
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.scenarios import two_robot_approach_then_push
from coimagine.options.generator import GridOptionGenerator
from coimagine.topology.exact_verifier import ExactFactorVerifier
from coimagine.topology.factors import FactorBuilder


def test_single_push_is_not_executable_but_extendable() -> None:
    state = two_robot_approach_then_push()
    # Move robots directly to the two south support cells.
    robots = dict(state.robots)
    robots[0] = type(robots[0])(0, (4, 3))
    robots[1] = type(robots[1])(1, (5, 3))
    state = state.with_updates(robots=robots)

    env = GridEnvironment(observation_radius=20)
    updater = RuleBasedBeliefUpdater()
    generator = GridOptionGenerator(candidate_budget=64)
    candidates = {
        rid: generator.generate(updater.initialize(env.observe(state, rid), state.width, state.height))
        for rid in state.robots
    }
    factors = FactorBuilder().build(candidates)
    factor = factors["entity:blocker-0"]
    pushes = [
        option
        for options in candidates.values()
        for option in options
        if option.kind == OptionKind.PUSH and option.direction.value == "north"
    ]
    assert len(pushes) >= 2
    verifier = ExactFactorVerifier(state, max_arity=2)
    label = verifier.label(factor, [pushes[0]])
    assert not label.executable
    assert label.extendable
