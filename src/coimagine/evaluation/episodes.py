from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Optional

from coimagine.core.types import JointOption, OptionKind, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import ScenarioInstance
from coimagine.envs.grid.state import GridState
from coimagine.evaluation.events import TopologyDelta, classify_transition
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch, SearchResult


@dataclass(frozen=True)
class EpisodeStep:
    index: int
    state_before: GridState
    state_after: GridState
    selected_action: JointOption
    predicted_continuation: Optional[JointOption]
    plan_value: float
    realized_stage_utility: float
    duration: float
    potential_before: float
    potential_after: float
    topology_delta: TopologyDelta
    search_result: SearchResult
    planning_seconds: float


@dataclass(frozen=True)
class EpisodeTrace:
    instance_id: str
    task_name: str
    method: str
    planner_mode: str
    horizon: int
    steps: tuple[EpisodeStep, ...]
    completed: bool
    final_state: GridState
    final_potential: float
    total_duration: float
    realized_return: float
    total_planning_seconds: float
    termination_reason: str
    failure_detail: str
    execution_failure_type: str | None
    max_steps: int

    @property
    def option_count(self) -> int:
        return len(self.steps)

    @property
    def first_action(self) -> JointOption | None:
        return self.steps[0].selected_action if self.steps else None


def rollout_receding_horizon(
    instance: ScenarioInstance,
    *,
    mode: PlannerMode,
    horizon: int = 2,
    candidate_budget: int = 8,
    max_arity: int = 3,
    max_steps: int | None = None,
    duration_weight: float = 0.02,
    failure_cost: float = 1.0,
    method_name: str | None = None,
    observation_radius: int | None = None,
) -> EpisodeTrace:
    """Execute a matched receding-horizon policy from one exact grid snapshot.

    Root-Structure Reuse freezes the decision structure only inside each imagined
    horizon. At the next real option boundary it receives the exact new state and
    builds a new root, matching the paper's receding-horizon baseline definition.
    """

    state = instance.state.clone()
    effective_horizon = int(horizon)
    effective_budget = int(state.task.metadata.get("candidate_budget", candidate_budget))
    max_steps = max_steps or int(state.task.metadata.get("episode_max_steps", 8))
    env = GridEnvironment(observation_radius=(observation_radius if observation_radius is not None else max(state.width, state.height)))
    generator = GridOptionGenerator(candidate_budget=effective_budget)
    planner = ExactTopologySearch(
        env,
        generator,
        horizon=effective_horizon,
        duration_weight=duration_weight,
        failure_cost=failure_cost,
        max_arity=max_arity,
    )

    beliefs = planner.initialize_beliefs(state)
    steps: list[EpisodeStep] = []
    total_duration = 0.0
    realized_return = 0.0
    total_planning_seconds = 0.0
    termination_reason = ""
    failure_detail = ""
    execution_failure_type: str | None = None
    initial_potential = env.task_potential(state)
    for index in range(max_steps):
        if env.is_terminal(state):
            termination_reason = "completed"
            break
        planning_start = perf_counter()
        try:
            result = planner.plan(state, beliefs, mode=mode)
        except RuntimeError as exc:
            if "No admitted root joint option" not in str(exc):
                raise
            termination_reason = "planner_no_admitted_root_option"
            failure_detail = str(exc)
            break
        planning_seconds = perf_counter() - planning_start
        transition = env.execute(state, result.joint_option)
        before_potential = env.task_potential(state)
        after_potential = env.task_potential(transition.state)
        if transition.result.success:
            stage = (
                after_potential
                - before_potential
                - duration_weight * transition.result.duration
            )
        else:
            stage = -failure_cost - duration_weight * transition.result.duration
        delta = classify_transition(state, transition.state, env, generator)
        continuation = result.trace[1].joint_option if len(result.trace) > 1 else None
        steps.append(
            EpisodeStep(
                index=index,
                state_before=state,
                state_after=transition.state,
                selected_action=result.joint_option,
                predicted_continuation=continuation,
                plan_value=result.value,
                realized_stage_utility=stage,
                duration=transition.result.duration,
                potential_before=before_potential,
                potential_after=after_potential,
                topology_delta=delta,
                search_result=result,
                planning_seconds=planning_seconds,
            )
        )
        total_duration += transition.result.duration
        realized_return += stage
        total_planning_seconds += planning_seconds
        if not transition.result.success:
            termination_reason = "execution_failure"
            execution_failure_type = transition.result.failure_type
            failure_detail = transition.result.failure_type or "unknown_execution_failure"
            break
        state = transition.state
        beliefs = {
            robot_id: planner.belief_updater.update(
                belief, env.observe(state, robot_id)
            )
            for robot_id, belief in beliefs.items()
        }

    completed = env.is_terminal(state)
    if completed:
        termination_reason = "completed"
        failure_detail = ""
    elif not termination_reason:
        termination_reason = "budget_exhausted"
        recent_steps = steps[-2:]
        recent_all_idle = bool(recent_steps) and all(
            all(option.kind == OptionKind.IDLE for option in step.selected_action.options)
            for step in recent_steps
        )
        final_potential = env.task_potential(state)
        if recent_all_idle:
            failure_detail = "stalled_idle"
        elif final_potential <= initial_potential + 1e-12:
            failure_detail = "no_task_progress"
        else:
            failure_detail = "partial_progress"

    return EpisodeTrace(
        instance_id=instance.instance_id,
        task_name=instance.task_name,
        method=method_name or mode.value,
        planner_mode=mode.value,
        horizon=effective_horizon,
        steps=tuple(steps),
        completed=completed,
        final_state=state,
        final_potential=env.task_potential(state),
        total_duration=total_duration,
        realized_return=realized_return,
        total_planning_seconds=total_planning_seconds,
        termination_reason=termination_reason,
        failure_detail=failure_detail,
        execution_failure_type=execution_failure_type,
        max_steps=max_steps,
    )


def to_episode_record(
    instance: ScenarioInstance,
    trace: EpisodeTrace,
    *,
    observation_radius: int,
):
    from coimagine.evaluation.records import EpisodeResultRecord
    from coimagine.reporting.semantics import format_joint_option_compact

    first = trace.steps[0] if trace.steps else None
    initial_corridor = set(instance.state.task.corridor)
    cleared_entities = sum(
        not bool(blocker.cells & initial_corridor)
        for blocker in trace.final_state.blockers.values()
    )
    remaining_entities = sum(
        bool(blocker.cells & initial_corridor)
        for blocker in trace.final_state.blockers.values()
    )
    no_progress_steps = sum(
        abs(step.potential_after - step.potential_before) <= 1e-12
        for step in trace.steps
    )
    idle_joint_steps = sum(
        all(option.kind == OptionKind.IDLE for option in step.selected_action.options)
        for step in trace.steps
    )
    repeated_action_steps = sum(
        trace.steps[index].selected_action.stable_key
        == trace.steps[index - 1].selected_action.stable_key
        for index in range(1, len(trace.steps))
    )
    root_candidate_counts = [
        sum(len(options) for options in step.search_result.root_admission.candidate_options.values())
        for step in trace.steps
    ]
    root_support_counts = [
        sum(len(templates) for templates in step.search_result.root_support_templates.values())
        for step in trace.steps
    ]
    root_truncated_template_counts = [
        max(0, support - selected)
        for support, selected in zip(root_support_counts, root_candidate_counts)
    ]
    predicted_recovered_root_template_count = 0
    predicted_new_template_count = 0
    for episode_step in trace.steps:
        support = episode_step.search_result.root_support_templates
        selected = episode_step.search_result.root_selected_templates
        for predicted_step in episode_step.search_result.trace[1:]:
            for option in predicted_step.joint_option.options:
                template = option.template_key
                robot_support = support.get(option.robot_id, frozenset())
                robot_selected = selected.get(option.robot_id, frozenset())
                if template in robot_support and template not in robot_selected:
                    predicted_recovered_root_template_count += 1
                elif template not in robot_support:
                    predicted_new_template_count += 1
    root_cartesian = [
        step.search_result.root_admission.total_cartesian for step in trace.steps
    ]
    root_admitted = [
        step.search_result.root_admission.admitted_count for step in trace.steps
    ]
    root_factors = [
        step.search_result.root_admission.factor_count for step in trace.steps
    ]
    root_verifier_calls = sum(
        step.search_result.root_admission.verifier_calls for step in trace.steps
    )
    effective_candidate_budget = int(
        instance.state.task.metadata.get("candidate_budget", 0)
    )
    candidate_saturated_steps = sum(
        any(
            len(options) >= effective_candidate_budget
            for options in step.search_result.root_admission.candidate_options.values()
        )
        for step in trace.steps
        if effective_candidate_budget > 0
    )

    def _mean(values: list[int]) -> float:
        return sum(values) / len(values) if values else 0.0

    return EpisodeResultRecord(
        instance_id=instance.instance_id,
        suite=instance.suite,
        task_name=instance.task_name,
        seed=instance.seed,
        method=trace.method,
        planner_mode=trace.planner_mode,
        horizon=trace.horizon,
        observation_radius=observation_radius,
        robot_count=len(instance.state.robots),
        entity_count=len(instance.state.blockers),
        completed=trace.completed,
        option_count=trace.option_count,
        total_duration=trace.total_duration,
        final_potential=trace.final_potential,
        realized_return=trace.realized_return,
        total_planning_seconds=trace.total_planning_seconds,
        mean_planning_seconds=(
            trace.total_planning_seconds / max(1, trace.option_count)
        ),
        first_action=format_joint_option_compact(first.selected_action) if first else "",
        first_predicted_continuation=(
            format_joint_option_compact(first.predicted_continuation)
            if first and first.predicted_continuation is not None
            else ""
        ),
        first_created_edges=(first.topology_delta.created_edge_count if first else 0),
        first_destroyed_edges=(first.topology_delta.destroyed_edge_count if first else 0),
        first_event=(first.topology_delta.dominant_event if first else "no_action"),
        scene_family=str(instance.state.task.metadata.get("scene_family", instance.task_name)),
        scene_subprofile=str(
            instance.state.task.metadata.get("scene_subprofile", "")
        ),
        dependency_depth=int(instance.state.task.metadata.get("dependency_depth", 0)),
        surprise_obstacle=bool(instance.state.task.metadata.get("surprise_obstacle", False)),
        scene_type=str(
            instance.state.task.metadata.get(
                "scene_type",
                "surprise" if instance.state.task.metadata.get("surprise_obstacle", False) else "open",
            )
        ),
        difficulty_variant=str(instance.state.task.metadata.get("difficulty_variant", "")),
        candidate_budget=int(instance.state.task.metadata.get("candidate_budget", 0)),
        episode_budget=int(instance.state.task.metadata.get("episode_max_steps", 0)),
        heavy_entity_count=int(
            instance.state.task.metadata.get(
                "heavy_blocker_count",
                sum(blocker.required_pushers > 1 for blocker in instance.state.blockers.values()),
            )
        ),
        occluder_count=int(instance.state.task.metadata.get("occluder_count", 0)),
        future_support_required=bool(
            instance.state.task.metadata.get("future_support_required", False)
        ),
        structurally_validated=bool(
            instance.state.task.metadata.get("structurally_validated", False)
        ),
        root_supported_structure_valid=bool(
            instance.state.task.metadata.get(
                "root_supported_structure_valid", False
            )
        ),
        future_support_structure_valid=bool(
            instance.state.task.metadata.get(
                "future_support_structure_valid", False
            )
        ),
        future_support_validation_reason=str(
            instance.state.task.metadata.get(
                "future_support_validation_reason", ""
            )
        ),
        critical_entity=str(instance.state.task.metadata.get("critical_entity", "")),
        critical_root_slot_count=int(
            instance.state.task.metadata.get("critical_root_slot_count", 0)
        ),
        critical_successor_slot_count=int(
            instance.state.task.metadata.get("critical_successor_slot_count", 0)
        ),
        critical_new_slot_count=int(
            instance.state.task.metadata.get("critical_new_slot_count", 0)
        ),
        generation_attempt=int(
            instance.state.task.metadata.get("generation_attempt", 0)
        ),
        activation_dependency_count=len(
            instance.state.task.activation_dependencies
        ),
        termination_reason=trace.termination_reason,
        failure_detail=trace.failure_detail,
        execution_failure_type=trace.execution_failure_type or "",
        budget_exhausted=trace.termination_reason == "budget_exhausted",
        steps_remaining=max(0, trace.max_steps - trace.option_count),
        no_progress_steps=no_progress_steps,
        idle_joint_steps=idle_joint_steps,
        repeated_action_steps=repeated_action_steps,
        cleared_entity_count=cleared_entities,
        remaining_blocked_entity_count=remaining_entities,
        root_candidate_count_mean=_mean(root_candidate_counts),
        root_candidate_count_max=max(root_candidate_counts, default=0),
        root_support_count_mean=_mean(root_support_counts),
        root_support_count_max=max(root_support_counts, default=0),
        root_truncated_template_count_mean=_mean(root_truncated_template_counts),
        root_truncated_template_count_max=max(root_truncated_template_counts, default=0),
        predicted_recovered_root_template_count=predicted_recovered_root_template_count,
        predicted_new_template_count=predicted_new_template_count,
        root_cartesian_mean=_mean(root_cartesian),
        root_cartesian_max=max(root_cartesian, default=0),
        root_admitted_mean=_mean(root_admitted),
        root_admitted_min=min(root_admitted, default=0),
        root_factor_count_mean=_mean(root_factors),
        root_verifier_calls_total=root_verifier_calls,
        candidate_saturated_steps=candidate_saturated_steps,
        initial_position_mode=str(
            instance.state.task.metadata.get("initial_position_mode", "")
        ),
        requested_steps_per_object=int(
            instance.state.task.metadata.get("requested_steps_per_object", 0)
        ),
        requested_extra_steps=int(
            instance.state.task.metadata.get("requested_extra_steps", -1)
        ),
        requested_episode_budget=int(
            instance.state.task.metadata.get("requested_episode_budget", 0)
        ),
        requested_max_steps_cap=int(
            instance.state.task.metadata.get("requested_max_steps_cap") or 0
        ),
    )
