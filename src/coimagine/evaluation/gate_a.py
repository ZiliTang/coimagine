from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from coimagine.core.types import BeliefTransitionMode, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import ScenarioInstance
from coimagine.envs.grid.state import GridState
from coimagine.evaluation.diagnostics import validate_diagnostic_mechanism
from coimagine.evaluation.events import classify_transition
from coimagine.evaluation.metrics import rank_agreement
from coimagine.evaluation.records import SceneResult
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch, SearchResult
from coimagine.reporting.semantics import baseline_display_name, format_joint_option


@dataclass(frozen=True)
class GateAResult:
    evolving: SearchResult
    fixed: SearchResult

    @property
    def oracle_gain(self) -> float:
        return self.evolving.value - self.fixed.value

    @property
    def ranking_reversal(self) -> bool:
        return self.evolving.joint_option.stable_key != self.fixed.joint_option.stable_key


def _effective_horizon(state: GridState, horizon: int) -> int:
    return int(state.task.metadata.get("horizon", horizon))


def _effective_candidate_budget(state: GridState, candidate_budget: int) -> int:
    return int(state.task.metadata.get("candidate_budget", candidate_budget))


def evaluate_gate_a(
    state: GridState,
    horizon: int = 2,
    *,
    observation_radius: int | None = None,
    belief_transition_mode: BeliefTransitionMode = BeliefTransitionMode.EXACT_LOCAL_OBSERVATION,
    candidate_budget: int = 64,
    max_arity: int = 3,
) -> GateAResult:
    horizon = _effective_horizon(state, horizon)
    candidate_budget = _effective_candidate_budget(state, candidate_budget)
    radius = observation_radius if observation_radius is not None else max(state.width, state.height)
    env = GridEnvironment(observation_radius=radius)
    generator = GridOptionGenerator(candidate_budget=candidate_budget)
    planner = ExactTopologySearch(
        env,
        generator,
        horizon=horizon,
        max_arity=max_arity,
        belief_transition_mode=belief_transition_mode,
    )
    beliefs = planner.initialize_beliefs(state)
    evolving = planner.plan(state, beliefs, mode=PlannerMode.EVOLVING)
    fixed = planner.plan(state, beliefs, mode=PlannerMode.ROOT_STRUCTURE_REUSE)
    return GateAResult(evolving=evolving, fixed=fixed)


def evaluate_instance(
    instance: ScenarioInstance,
    *,
    horizon: int = 2,
    information_mode: str = "exact_state_oracle",
    observation_radius: int = 4,
    candidate_budget: int = 64,
    max_arity: int = 3,
) -> SceneResult:
    state = instance.state
    horizon = _effective_horizon(state, horizon)
    candidate_budget = _effective_candidate_budget(state, candidate_budget)
    if information_mode == "exact_state_oracle":
        radius = max(state.width, state.height)
        belief_mode = BeliefTransitionMode.EXACT_LOCAL_OBSERVATION
    elif information_mode == "symbolic_local_belief":
        radius = observation_radius
        belief_mode = BeliefTransitionMode.EXACT_LOCAL_OBSERVATION
    elif information_mode == "symbolic_imagination":
        radius = observation_radius
        belief_mode = BeliefTransitionMode.SYMBOLIC_IMAGINATION
    else:
        raise ValueError(f"Unknown information mode: {information_mode}")

    env = GridEnvironment(observation_radius=radius)
    generator = GridOptionGenerator(candidate_budget=candidate_budget)
    planner = ExactTopologySearch(
        env,
        generator,
        horizon=horizon,
        max_arity=max_arity,
        belief_transition_mode=belief_mode,
    )
    beliefs = planner.initialize_beliefs(state)

    start = perf_counter()
    evolving = planner.plan(state, beliefs, mode=PlannerMode.EVOLVING)
    evolving_seconds = perf_counter() - start
    start = perf_counter()
    fixed = planner.plan(state, beliefs, mode=PlannerMode.ROOT_STRUCTURE_REUSE)
    fixed_seconds = perf_counter() - start

    fixed_key = fixed.joint_option.stable_key
    evolving_value_of_fixed = evolving.action_values.get(fixed_key, float("-inf"))
    first_regret = evolving.value - evolving_value_of_fixed

    if evolving.trace:
        delta = classify_transition(
            evolving.trace[0].state_before,
            evolving.trace[0].state_after,
            env,
            generator,
        )
    else:
        delta = classify_transition(state, state, env, generator)

    mechanism_valid, mechanism_notes = validate_diagnostic_mechanism(
        instance,
        evolving,
        fixed,
        delta,
        planner,
    )

    return SceneResult(
        instance_id=instance.instance_id,
        suite=instance.suite,
        task_name=instance.task_name,
        seed=instance.seed,
        information_mode=information_mode,
        belief_transition_mode=belief_mode.value,
        horizon=horizon,
        candidate_budget=candidate_budget,
        robot_count=len(state.robots),
        entity_count=len(state.blockers),
        evolving_value=evolving.value,
        fixed_value=fixed.value,
        oracle_gain=evolving.value - fixed.value,
        first_action_regret=first_regret,
        first_action_reversal=evolving.joint_option.stable_key != fixed.joint_option.stable_key,
        rank_agreement=rank_agreement(evolving.action_values, fixed.action_values),
        positive_gain=evolving.value - fixed.value > 1e-9,
        dominant_event=delta.dominant_event,
        topology_events=delta.labels,
        root_node_count=len(delta.before_nodes),
        successor_node_count=len(delta.after_nodes),
        root_edge_count=len(delta.before_edges),
        successor_edge_count=len(delta.after_edges),
        created_node_count=len(delta.created_nodes),
        destroyed_node_count=len(delta.destroyed_nodes),
        created_edge_count=delta.created_edge_count,
        destroyed_edge_count=delta.destroyed_edge_count,
        persistent_edge_count=delta.persistent_edge_count,
        revealed_entity_count=len(delta.revealed_entities),
        higher_order_context=delta.higher_order_context,
        mechanism_valid=mechanism_valid,
        mechanism_notes=mechanism_notes,
        evolving_root_action=format_joint_option(evolving.joint_option),
        fixed_root_action=format_joint_option(fixed.joint_option),
        evolving_cartesian=evolving.root_admission.total_cartesian,
        evolving_admitted=evolving.root_admission.admitted_count,
        evolving_factors=evolving.root_admission.factor_count,
        evolving_verifier_calls=evolving.root_admission.verifier_calls,
        fixed_cartesian=fixed.root_admission.total_cartesian,
        fixed_admitted=fixed.root_admission.admitted_count,
        evolving_planning_seconds=evolving_seconds,
        fixed_planning_seconds=fixed_seconds,
        total_planning_seconds=evolving_seconds + fixed_seconds,
    )


def evaluate_baselines(
    instance: ScenarioInstance,
    *,
    horizon: int = 2,
    candidate_budget: int = 64,
    max_arity: int = 3,
) -> list["BaselineResult"]:
    from coimagine.evaluation.records import BaselineResult

    state = instance.state
    horizon = _effective_horizon(state, horizon)
    candidate_budget = _effective_candidate_budget(state, candidate_budget)
    env = GridEnvironment(observation_radius=max(state.width, state.height))
    generator = GridOptionGenerator(candidate_budget=candidate_budget)
    planner = ExactTopologySearch(env, generator, horizon=horizon, max_arity=max_arity)
    beliefs = planner.initialize_beliefs(state)
    exact = planner.plan(state, beliefs, mode=PlannerMode.FULL_TUPLE_ORACLE)
    exact_edges = set(exact.action_values)
    modes = (
        PlannerMode.LOCAL_AVAILABILITY,
        PlannerMode.PAIRWISE_ONLY,
        PlannerMode.RECURSIVE_FACTORS,
        PlannerMode.FULL_TUPLE_ORACLE,
        PlannerMode.ROOT_STRUCTURE_REUSE,
        PlannerMode.STRICT_ROOT_REUSE,
    )
    rows: list[BaselineResult] = []
    for mode in modes:
        result = planner.plan(state, beliefs, mode=mode)
        admitted = set(result.action_values)
        overlap = len(admitted & exact_edges)
        recall = overlap / max(1, len(exact_edges))
        precision = overlap / max(1, len(admitted))
        rows.append(
            BaselineResult(
                instance_id=instance.instance_id,
                task_name=instance.task_name,
                mode=mode.value,
                method_name=baseline_display_name(mode.value),
                value=result.value,
                root_action=format_joint_option(result.joint_option),
                total_cartesian=result.root_admission.total_cartesian,
                admitted_count=result.root_admission.admitted_count,
                factor_count=result.root_admission.factor_count,
                verifier_calls=result.root_admission.verifier_calls,
                retained_tuples=result.root_admission.retained_tuples,
                exact_root_recall=recall,
                exact_root_precision=precision,
            )
        )
    return rows
