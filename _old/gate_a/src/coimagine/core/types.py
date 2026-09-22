from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import FrozenSet, Mapping, Optional, Tuple

Cell = Tuple[int, int]
RobotId = int
EntityId = str
FactorId = str
OptionTemplateKey = tuple


class Direction(str, Enum):
    NORTH = "north"
    EAST = "east"
    SOUTH = "south"
    WEST = "west"

    @property
    def delta(self) -> Cell:
        return {
            Direction.NORTH: (0, 1),
            Direction.EAST: (1, 0),
            Direction.SOUTH: (0, -1),
            Direction.WEST: (-1, 0),
        }[self]


class OptionKind(str, Enum):
    REPOSITION = "reposition"
    APPROACH = "approach"
    PUSH = "push"
    HOLD = "hold"
    OBSERVE = "observe"
    IDLE = "idle"


class FactorType(str, Enum):
    ENTITY = "entity"
    RESOURCE = "resource"
    SYNCHRONIZATION = "synchronization"


class TaskKind(str, Enum):
    CLEARING = "clearing"
    TRANSPORT = "transport"
    HOLD_PASS = "hold_pass"
    REVEAL_ACT = "reveal_act"
    STABLE_CONTROL = "stable_control"


class PlannerMode(str, Enum):
    # Gate-A v0.5 main comparison.
    COST = "current_observation_supported_topology"
    COST_FROZEN_K = "current_observation_top_k_frozen"
    IET = "imagined_evolving_topology"
    PRIVILEGED_IET_ORACLE = "privileged_successor_observation_oracle"

    # Structural baselines retained for controlled diagnostics and scaling.
    EVOLVING = "evolving"
    ROOT_STRUCTURE_REUSE = "root_structure_reuse"
    STRICT_ROOT_REUSE = "strict_root_reuse"
    LOCAL_AVAILABILITY = "local_availability"
    PAIRWISE_ONLY = "pairwise_only"
    RECURSIVE_FACTORS = "recursive_factors"
    FULL_TUPLE_ORACLE = "full_tuple_oracle"


class BeliefTransitionMode(str, Enum):
    # The simulator renders the true successor local observation. This is an
    # evaluation oracle, not an information source available to the main method.
    PRIVILEGED_SUCCESSOR_OBSERVATION = "privileged_successor_observation"
    # Known local entities and teammate option effects are propagated; unobstructed
    # unknown space is assigned an optimistic free-space prior.
    OPTIMISTIC_LOCAL_IMAGINATION = "optimistic_local_imagination"
    # Legacy names kept for old profiles and external scripts.
    EXACT_LOCAL_OBSERVATION = "exact_local_observation"
    SYMBOLIC_IMAGINATION = "symbolic_imagination"


@dataclass(frozen=True)
class FactorSignature:
    """Communicable structural signature; no private metric map is included."""

    entity_ids: FrozenSet[EntityId] = frozenset()
    resource_tokens: FrozenSet[str] = frozenset()
    temporal_tokens: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class LocalOption:
    option_id: str
    robot_id: RobotId
    kind: OptionKind
    entity_id: Optional[EntityId] = None
    direction: Optional[Direction] = None
    target: Optional[Cell] = None
    contact_cell: Optional[Cell] = None
    duration_hint: float = 1.0
    role_token: Optional[str] = None
    slot_token: Optional[str] = None
    signature: FactorSignature = field(default_factory=FactorSignature)

    @property
    def stable_key(self) -> tuple:
        return (
            self.robot_id,
            self.kind.value,
            self.entity_id,
            self.direction.value if self.direction else None,
            self.role_token,
            self.slot_token,
            self.target,
            self.contact_cell,
        )

    @property
    def template_key(self) -> OptionTemplateKey:
        """Observation-grounded identity used by COST.

        Metric execution parameters are excluded for object-relative templates.
        ``slot_token`` preserves semantic contact slots and exact reposition anchors,
        so a future observation may update an existing template but cannot silently
        introduce a new option vertex under COST.
        """

        return (
            self.robot_id,
            self.kind.value,
            self.entity_id,
            self.direction.value if self.direction else None,
            self.role_token,
            self.slot_token,
        )


@dataclass(frozen=True)
class JointOption:
    options: Tuple[LocalOption, ...]

    def __post_init__(self) -> None:
        robot_ids = [option.robot_id for option in self.options]
        if len(robot_ids) != len(set(robot_ids)):
            raise ValueError("A joint option must contain at most one option per robot.")

    @property
    def stable_key(self) -> tuple:
        return tuple(sorted((option.stable_key for option in self.options)))

    @property
    def template_key(self) -> tuple:
        return tuple(sorted((option.template_key for option in self.options)))

    def by_robot(self) -> Mapping[RobotId, LocalOption]:
        return {option.robot_id: option for option in self.options}


@dataclass(frozen=True)
class ExecutionResult:
    success: bool
    duration: float
    failure_type: Optional[str] = None
    details: Mapping[str, object] = field(default_factory=dict)
