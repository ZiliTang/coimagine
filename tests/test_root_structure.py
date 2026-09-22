from coimagine.core.types import OptionKind, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import generate_local_imagination_showcase
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch
from coimagine.topology.admission import TopologyAdmission


def test_new_push_templates_do_not_enter_current_supported_topology() -> None:
    instance = generate_local_imagination_showcase(0)
    state = instance.state
    env = GridEnvironment(observation_radius=6)
    generator = GridOptionGenerator(candidate_budget=8)
    planner = ExactTopologySearch(env, generator, horizon=3)
    beliefs = planner.initialize_beliefs(state)
    root_candidates = {rid: generator.generate(belief) for rid, belief in beliefs.items()}
    admission = TopologyAdmission(env, max_arity=3)
    root = admission.build_root_structure(state, root_candidates)

    iet = planner.plan(state, beliefs, mode=PlannerMode.IET)
    successor = env.execute(state, iet.joint_option).state
    successor_beliefs = {
        rid: planner.belief_imaginator.predict(beliefs[rid], iet.joint_option)
        for rid in beliefs
    }
    successor_candidates = {
        rid: generator.generate(belief) for rid, belief in successor_beliefs.items()
    }
    assert any(
        option.kind == OptionKind.PUSH and option.slot_token == "north:1"
        for options in successor_candidates.values()
        for option in options
    )
    projected = admission.project_to_root(successor_candidates, root, strict=False)
    assert all(
        not (option.kind == OptionKind.PUSH and option.slot_token == "north:1")
        for options in projected.values()
        for option in options
    )
    assert successor.step_index == 1
