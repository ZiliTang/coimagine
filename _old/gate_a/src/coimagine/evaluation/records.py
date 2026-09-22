from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping


@dataclass(frozen=True)
class SceneResult:
    instance_id: str
    suite: str
    task_name: str
    seed: int
    information_mode: str
    belief_transition_mode: str
    horizon: int
    candidate_budget: int
    robot_count: int
    entity_count: int
    evolving_value: float
    fixed_value: float
    oracle_gain: float
    first_action_regret: float
    first_action_reversal: bool
    rank_agreement: float
    positive_gain: bool
    dominant_event: str
    topology_events: tuple[str, ...]
    root_node_count: int
    successor_node_count: int
    root_edge_count: int
    successor_edge_count: int
    created_node_count: int
    destroyed_node_count: int
    created_edge_count: int
    destroyed_edge_count: int
    persistent_edge_count: int
    revealed_entity_count: int
    higher_order_context: bool
    mechanism_valid: bool
    mechanism_notes: str
    evolving_root_action: str
    fixed_root_action: str
    evolving_cartesian: int
    evolving_admitted: int
    evolving_factors: int
    evolving_verifier_calls: int
    fixed_cartesian: int
    fixed_admitted: int
    evolving_planning_seconds: float
    fixed_planning_seconds: float
    total_planning_seconds: float

    def to_dict(self) -> dict:
        data = asdict(self)
        data["topology_events"] = ";".join(self.topology_events)
        return data


@dataclass(frozen=True)
class ScalingResult:
    study: str
    x_label: str
    x_value: int
    robot_count: int
    factor_arity: int
    candidate_count: int
    naive_count: int
    factor_count: int
    recursive_calls: int
    retained_tuples: int
    oracle_recall: float

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeResultRecord:
    instance_id: str
    suite: str
    task_name: str
    seed: int
    method: str
    planner_mode: str
    horizon: int
    observation_radius: int
    robot_count: int
    entity_count: int
    completed: bool
    option_count: int
    total_duration: float
    final_potential: float
    realized_return: float
    total_planning_seconds: float
    mean_planning_seconds: float
    first_action: str
    first_predicted_continuation: str
    first_created_edges: int
    first_destroyed_edges: int
    first_event: str
    scene_family: str = ""
    scene_subprofile: str = ""
    dependency_depth: int = 0
    surprise_obstacle: bool = False
    scene_type: str = "open"
    difficulty_variant: str = ""
    candidate_budget: int = 0
    episode_budget: int = 0
    heavy_entity_count: int = 0
    occluder_count: int = 0
    future_support_required: bool = False
    structurally_validated: bool = False
    root_supported_structure_valid: bool = False
    future_support_structure_valid: bool = False
    future_support_validation_reason: str = ""
    critical_entity: str = ""
    critical_root_slot_count: int = 0
    critical_successor_slot_count: int = 0
    critical_new_slot_count: int = 0
    generation_attempt: int = 0
    activation_dependency_count: int = 0
    termination_reason: str = ""
    failure_detail: str = ""
    execution_failure_type: str = ""
    budget_exhausted: bool = False
    steps_remaining: int = 0
    no_progress_steps: int = 0
    idle_joint_steps: int = 0
    repeated_action_steps: int = 0
    cleared_entity_count: int = 0
    remaining_blocked_entity_count: int = 0
    root_candidate_count_mean: float = 0.0
    root_candidate_count_max: int = 0
    root_support_count_mean: float = 0.0
    root_support_count_max: int = 0
    root_truncated_template_count_mean: float = 0.0
    root_truncated_template_count_max: int = 0
    predicted_recovered_root_template_count: int = 0
    predicted_new_template_count: int = 0
    root_cartesian_mean: float = 0.0
    root_cartesian_max: int = 0
    root_admitted_mean: float = 0.0
    root_admitted_min: int = 0
    root_factor_count_mean: float = 0.0
    root_verifier_calls_total: int = 0
    candidate_saturated_steps: int = 0
    initial_position_mode: str = ""
    requested_steps_per_object: int = 0
    requested_extra_steps: int = -1
    requested_episode_budget: int = 0
    requested_max_steps_cap: int = 0

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BaselineResult:
    instance_id: str
    task_name: str
    mode: str
    method_name: str
    value: float
    root_action: str
    total_cartesian: int
    admitted_count: int
    factor_count: int
    verifier_calls: int
    retained_tuples: int
    exact_root_recall: float
    exact_root_precision: float

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)
