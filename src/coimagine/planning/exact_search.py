from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

from coimagine.belief.symbolic import (
    RuleBasedBeliefUpdater,
    SymbolicBelief,
    SymbolicBeliefImaginator,
)
from coimagine.core.types import (
    BeliefTransitionMode,
    JointOption,
    LocalOption,
    OptionKind,
    PlannerMode,
    RobotId,
)
from coimagine.envs.grid.environment import GridEnvironment, Transition
from coimagine.envs.grid.state import GridState
from coimagine.options.generator import GridOptionGenerator
from coimagine.topology.admission import AdmissionResult, RootStructure, TopologyAdmission


@dataclass(frozen=True)
class SearchStep:
    depth: int
    state_before: GridState
    state_after: GridState
    joint_option: JointOption
    success: bool
    stage_utility: float
    duration: float
    failure_type: str | None
    potential_before: float
    potential_after: float
    admission: AdmissionResult
    belief_source: str = ""


@dataclass(frozen=True)
class SearchResult:
    joint_option: JointOption
    value: float
    trace: tuple[SearchStep, ...]
    action_values: Mapping[tuple, float]
    root_admission: AdmissionResult
    root_support_templates: Mapping[RobotId, frozenset[tuple]]
    root_selected_templates: Mapping[RobotId, frozenset[tuple]]
    runtime_stats: Mapping[str, int]


@dataclass(frozen=True)
class _PreparedAction:
    joint_option: JointOption
    transition: Transition
    next_beliefs: Mapping[RobotId, SymbolicBelief] | None
    stage_utility: float
    step: SearchStep
    successor_key: tuple | None


class ExactTopologySearch:
    """Finite-horizon Gate-A planner under matched topology policies.

    CST and IET share the same root observations, option-level dynamics, horizon,
    task potential, top-K budget, and recursive verifier. CST re-instantiates the
    complete root-observation-supported template bank at future depths; IET augments
    that bank with templates grounded by imagined successor observations. The
    CST-Frozen-K diagnostic instead freezes only the templates selected into the root
    top-K bank. The privileged oracle uses simulator-rendered successor observations.
    """

    def __init__(
        self,
        env: GridEnvironment,
        option_generator: GridOptionGenerator,
        horizon: int = 2,
        duration_weight: float = 0.02,
        failure_cost: float = 1.0,
        max_arity: int = 3,
        belief_transition_mode: BeliefTransitionMode = BeliefTransitionMode.OPTIMISTIC_LOCAL_IMAGINATION,
        enable_runtime_cache: bool = True,
        enable_successor_dominance: bool = True,
        cache_capacity: int = 8192,
    ) -> None:
        if horizon < 1:
            raise ValueError("horizon must be at least one")
        self.env = env
        self.option_generator = option_generator
        self.horizon = horizon
        self.duration_weight = duration_weight
        self.failure_cost = failure_cost
        self.enable_runtime_cache = bool(enable_runtime_cache)
        self.enable_successor_dominance = bool(enable_successor_dominance)
        self.cache_capacity = max(64, int(cache_capacity))
        self.admission = TopologyAdmission(
            env,
            max_arity=max_arity,
            enable_runtime_cache=enable_runtime_cache,
            cache_capacity=cache_capacity,
        )
        self.belief_updater = RuleBasedBeliefUpdater()
        self.belief_imaginator = SymbolicBeliefImaginator(env, self.belief_updater)
        self.belief_transition_mode = belief_transition_mode
        self._value_cache: dict[tuple, tuple[float, tuple[SearchStep, ...]]] = {}
        self._transition_cache: dict[tuple, Transition] = {}
        self._belief_transition_cache: dict[
            tuple, Mapping[RobotId, SymbolicBelief]
        ] = {}
        self._state_key_cache: dict[int, tuple[GridState, tuple]] = {}
        self._belief_key_cache: dict[int, tuple[SymbolicBelief, tuple]] = {}
        self._belief_identity_cache: dict[int, SymbolicBelief] = {}
        self._runtime_stats: dict[str, int] = {}

    def _state_key(self, state: GridState) -> tuple:
        object_id = id(state)
        cached = self._state_key_cache.get(object_id)
        if cached is not None and cached[0] is state:
            return cached[1]
        key = (
            state.width,
            state.height,
            tuple(sorted(state.walls)),
            tuple(sorted((rid, robot.position) for rid, robot in state.robots.items())),
            tuple(
                sorted(
                    (
                        entity_id,
                        tuple(sorted(blocker.cells)),
                        blocker.required_pushers,
                        blocker.mass_class,
                        blocker.task_weight,
                        tuple(
                            sorted(
                                direction.value
                                for direction in blocker.allowed_push_directions
                            )
                        ),
                    )
                    for entity_id, blocker in state.blockers.items()
                )
            ),
            tuple(sorted(state.revealed_entities)),
        )
        self._state_key_cache[object_id] = (state, key)
        return key

    def _belief_key(self, belief: SymbolicBelief) -> tuple:
        object_id = id(belief)
        cached = self._belief_key_cache.get(object_id)
        if cached is not None and cached[0] is belief:
            return cached[1]
        # The task specification is constant within one plan and therefore does
        # not need to be copied into every successor-belief key.  Likewise,
        # ``last_seen_step``, ``step_index``, display labels, and the source tag do
        # not affect option generation, transition prediction, admission, or value.
        key = (
            belief.robot_id,
            belief.width,
            belief.height,
            belief.own_position,
            tuple(sorted((cell, value.value) for cell, value in belief.cells.items())),
            tuple(
                sorted(
                    (
                        entity_id,
                        tuple(sorted(blocker.cells)),
                        tuple(sorted(blocker.visible_cells)),
                        blocker.required_pushers,
                        blocker.mass_class,
                        blocker.task_weight,
                        tuple(
                            sorted(
                                direction.value
                                for direction in blocker.allowed_push_directions
                            )
                        ),
                        blocker.certain,
                    )
                    for entity_id, blocker in belief.blockers.items()
                )
            ),
            tuple(sorted(belief.teammate_positions.items())),
            tuple(sorted(belief.revealed_entities)),
            tuple(sorted(belief.current_visible_cells)),
            tuple(sorted(belief.current_free_cells)),
            tuple(sorted(belief.current_walls)),
        )
        self._belief_key_cache[object_id] = (belief, key)
        return key

    def _beliefs_identity_key(
        self,
        beliefs: Mapping[RobotId, SymbolicBelief],
    ) -> tuple:
        rows = []
        for robot_id, belief in sorted(beliefs.items()):
            object_id = id(belief)
            cached = self._belief_identity_cache.get(object_id)
            if cached is not None and cached is not belief:
                raise RuntimeError("Belief object id was unexpectedly reused inside one plan")
            self._belief_identity_cache[object_id] = belief
            rows.append((robot_id, object_id))
        return tuple(rows)

    def _beliefs_key(self, beliefs: Mapping[RobotId, SymbolicBelief]) -> tuple:
        return tuple(
            (robot_id, self._belief_key(belief))
            for robot_id, belief in sorted(beliefs.items())
        )

    def _increment(self, key: str, amount: int = 1) -> None:
        self._runtime_stats[key] = self._runtime_stats.get(key, 0) + amount

    def _cache_store(self, cache: dict, key: tuple, value: object) -> None:
        if not self.enable_runtime_cache:
            return
        if len(cache) >= self.cache_capacity:
            cache.pop(next(iter(cache)))
        cache[key] = value

    def _reset_runtime_state(self) -> None:
        self._value_cache = {}
        self._transition_cache = {}
        self._belief_transition_cache = {}
        self._state_key_cache = {}
        self._belief_key_cache = {}
        self._belief_identity_cache = {}
        self._runtime_stats = {}
        self.option_generator.clear_runtime_caches()
        self.admission.clear_runtime_caches()

    def _combined_runtime_stats(self) -> Mapping[str, int]:
        stats = dict(self._runtime_stats)
        for prefix, source in (
            ("option", self.option_generator.runtime_stats),
            ("admission", self.admission.runtime_stats),
        ):
            for key, value in source.items():
                stats[f"{prefix}_{key}"] = int(value)
        return stats

    def initialize_beliefs(self, state: GridState) -> Mapping[RobotId, SymbolicBelief]:
        return {
            robot_id: self.belief_updater.initialize(
                self.env.observe(state, robot_id), state.width, state.height
            )
            for robot_id in state.robots
        }

    def _candidate_support_sets(
        self, beliefs: Mapping[RobotId, SymbolicBelief]
    ) -> Mapping[RobotId, tuple[LocalOption, ...]]:
        return {
            robot_id: self.option_generator.generate_support_pool(belief)
            for robot_id, belief in beliefs.items()
        }

    def _select_candidate_sets(
        self,
        beliefs: Mapping[RobotId, SymbolicBelief],
        support_sets: Mapping[RobotId, Sequence[LocalOption]],
    ) -> Mapping[RobotId, tuple[LocalOption, ...]]:
        return {
            robot_id: self.option_generator.select_candidates(
                beliefs[robot_id], support_sets.get(robot_id, ())
            )
            for robot_id in beliefs
        }

    def _candidate_sets(
        self, beliefs: Mapping[RobotId, SymbolicBelief]
    ) -> Mapping[RobotId, tuple[LocalOption, ...]]:
        support_sets = self._candidate_support_sets(beliefs)
        return self._select_candidate_sets(beliefs, support_sets)

    @staticmethod
    def _merge_template_pools(
        primary: Sequence[LocalOption],
        fallback: Sequence[LocalOption],
    ) -> tuple[LocalOption, ...]:
        """Union by template identity, preferring current-observation instances."""

        merged = {option.template_key: option for option in fallback}
        merged.update({option.template_key: option for option in primary})
        return tuple(merged.values())

    def _candidate_sets_for_depth(
        self,
        beliefs: Mapping[RobotId, SymbolicBelief],
        mode: PlannerMode,
        root_structure: RootStructure,
        *,
        depth: int,
    ) -> Mapping[RobotId, tuple[LocalOption, ...]]:
        if depth == 0:
            support = self._candidate_support_sets(beliefs)
            return self._select_candidate_sets(beliefs, support)

        support_sets: dict[RobotId, tuple[LocalOption, ...]] = {}
        for robot_id, belief in beliefs.items():
            if mode == PlannerMode.COST:
                pool = self.option_generator.instantiate_templates(
                    belief, root_structure.support_root_options.get(robot_id, ())
                )
            elif mode == PlannerMode.COST_FROZEN_K:
                pool = self.option_generator.instantiate_templates(
                    belief, root_structure.concrete_root_options.get(robot_id, ())
                )
            elif mode in {PlannerMode.IET, PlannerMode.PRIVILEGED_IET_ORACLE}:
                current = self.option_generator.generate_support_pool(belief)
                remembered = self.option_generator.instantiate_templates(
                    belief, root_structure.support_root_options.get(robot_id, ())
                )
                pool = self._merge_template_pools(current, remembered)
            else:
                pool = self.option_generator.generate_support_pool(belief)
            support_sets[robot_id] = tuple(pool)
        return self._select_candidate_sets(beliefs, support_sets)

    @staticmethod
    def _uses_optimistic_model(mode: PlannerMode) -> bool:
        return mode in {
            PlannerMode.COST,
            PlannerMode.COST_FROZEN_K,
            PlannerMode.IET,
        }

    @staticmethod
    def _joint_tie_key(joint: JointOption) -> tuple:
        """Prefer active progress over horizon-shifting Idle ties.

        Undiscounted finite-horizon plans can assign the same value to acting now
        and delaying an otherwise identical sequence by one boundary.  A receding-
        horizon controller would then repeatedly choose Idle under a lexical tie.
        The secondary ordering is method-independent and only applies at equal
        value: fewer Idle options, then more task-directed option kinds.
        """

        idle_count = sum(option.kind == OptionKind.IDLE for option in joint.options)
        kind_score = {
            OptionKind.PUSH: 4,
            OptionKind.HOLD: 4,
            OptionKind.OBSERVE: 3,
            OptionKind.APPROACH: 2,
            OptionKind.REPOSITION: 1,
            OptionKind.IDLE: 0,
        }
        active_score = sum(kind_score[option.kind] for option in joint.options)
        return (-idle_count, active_score, tuple(map(str, joint.stable_key)))

    @staticmethod
    def _belief_effect_key(joint: JointOption) -> tuple:
        """Action abstraction sufficient for the symbolic belief transition.

        Reposition and Approach to the same endpoint have the same sensing effect;
        Hold and Idle both preserve pose and reveal no entity.  Push and Observe
        retain their entity-level effects.  This key enables exact reuse of belief
        transitions for action aliases that reach the same successor state.
        """

        rows = []
        for option in sorted(joint.options, key=lambda item: item.robot_id):
            if option.kind in {OptionKind.REPOSITION, OptionKind.APPROACH}:
                effect = ("move", option.target)
            elif option.kind == OptionKind.PUSH:
                effect = (
                    "push",
                    option.entity_id,
                    option.direction.value if option.direction else None,
                    option.contact_cell,
                )
            elif option.kind == OptionKind.OBSERVE:
                effect = ("observe", option.entity_id)
            else:
                effect = ("stay",)
            rows.append((option.robot_id, effect))
        return tuple(rows)

    def _simulate_transition(
        self,
        state: GridState,
        beliefs: Mapping[RobotId, SymbolicBelief],
        joint: JointOption,
        *,
        depth: int,
        mode: PlannerMode,
    ) -> Transition:
        # Root actions are grounded by the current real observations and are checked
        # against the exact simulator before execution. Optimism only affects future
        # branch actions that are never directly executed without replanning.
        optimistic = depth > 0 and self._uses_optimistic_model(mode)
        cache_key = (
            "optimistic" if optimistic else "exact",
            self._state_key(state),
            self._beliefs_identity_key(beliefs) if optimistic else (),
            joint.stable_key,
        )
        if self.enable_runtime_cache:
            cached = self._transition_cache.get(cache_key)
            if cached is not None:
                self._increment("transition_cache_hits")
                return cached
        self._increment("transition_cache_misses")
        transition = (
            self.env.execute_optimistic(state, beliefs, joint)
            if optimistic
            else self.env.execute(state, joint)
        )
        self._cache_store(self._transition_cache, cache_key, transition)
        return transition

    def _next_beliefs(
        self,
        beliefs: Mapping[RobotId, SymbolicBelief],
        next_state: GridState,
        joint: JointOption,
        *,
        mode: PlannerMode,
    ) -> Mapping[RobotId, SymbolicBelief]:
        cache_key = (
            mode.value,
            self._beliefs_identity_key(beliefs),
            self._state_key(next_state),
            self.belief_transition_mode.value,
        )
        if self.enable_runtime_cache:
            cached = self._belief_transition_cache.get(cache_key)
            if cached is not None:
                self._increment("belief_transition_cache_hits")
                return cached
        self._increment("belief_transition_cache_misses")

        if mode == PlannerMode.PRIVILEGED_IET_ORACLE:
            result = {
                robot_id: self.belief_updater.update(
                    belief,
                    self.env.observe(next_state, robot_id),
                    source="privileged_successor_observation",
                )
                for robot_id, belief in beliefs.items()
            }
        elif mode in {PlannerMode.COST, PlannerMode.COST_FROZEN_K, PlannerMode.IET}:
            result = {
                robot_id: self.belief_imaginator.predict(belief, joint)
                for robot_id, belief in beliefs.items()
            }
        elif self.belief_transition_mode in {
            BeliefTransitionMode.OPTIMISTIC_LOCAL_IMAGINATION,
            BeliefTransitionMode.SYMBOLIC_IMAGINATION,
        }:
            result = {
                robot_id: self.belief_imaginator.predict(belief, joint)
                for robot_id, belief in beliefs.items()
            }
        else:
            result = {
                robot_id: self.belief_updater.update(
                    belief,
                    self.env.observe(next_state, robot_id),
                    source="privileged_successor_observation",
                )
                for robot_id, belief in beliefs.items()
            }
        self._cache_store(self._belief_transition_cache, cache_key, result)
        return result

    def _admit_at_depth(
        self,
        state: GridState,
        beliefs: Mapping[RobotId, SymbolicBelief],
        current_candidates: Mapping[RobotId, Sequence[LocalOption]],
        mode: PlannerMode,
        root_structure: RootStructure,
        *,
        depth: int,
    ) -> AdmissionResult:
        def execution_check(joint: JointOption) -> bool:
            return self._simulate_transition(
                state, beliefs, joint, depth=depth, mode=mode
            ).result.success

        if mode in {PlannerMode.COST, PlannerMode.COST_FROZEN_K}:
            return self.admission.admit(
                state,
                current_candidates,
                mode,
                execution_check=execution_check,
            )

        if mode in {PlannerMode.IET, PlannerMode.PRIVILEGED_IET_ORACLE}:
            return self.admission.admit(
                state,
                current_candidates,
                mode,
                execution_check=execution_check,
            )

        # Legacy root-structure baselines share the exact root decision structure
        # and only freeze it on imagined successor branches.
        if depth == 0 and mode in {
            PlannerMode.ROOT_STRUCTURE_REUSE,
            PlannerMode.STRICT_ROOT_REUSE,
        }:
            return self.admission.admit(
                state,
                current_candidates,
                PlannerMode.FULL_TUPLE_ORACLE,
                execution_check=execution_check,
            )
        if mode == PlannerMode.ROOT_STRUCTURE_REUSE:
            projected = self.admission.project_to_root(
                current_candidates,
                root_structure,
                strict=False,
                selected_only=True,
            )
            return self.admission.admit(
                state,
                projected,
                PlannerMode.ROOT_STRUCTURE_REUSE,
                root_structure,
                execution_check=execution_check,
            )
        if mode == PlannerMode.STRICT_ROOT_REUSE:
            projected = self.admission.project_to_root(
                current_candidates, root_structure, strict=True
            )
            return self.admission.admit(
                state,
                projected,
                PlannerMode.STRICT_ROOT_REUSE,
                root_structure,
                execution_check=execution_check,
            )
        return self.admission.admit(
            state,
            current_candidates,
            mode,
            execution_check=execution_check,
        )

    def plan(
        self,
        state: GridState,
        beliefs: Optional[Mapping[RobotId, SymbolicBelief]] = None,
        *,
        mode: PlannerMode = PlannerMode.IET,
    ) -> SearchResult:
        beliefs = beliefs or self.initialize_beliefs(state)
        self._reset_runtime_state()
        root_support = self._candidate_support_sets(beliefs)
        root_candidates = self._select_candidate_sets(beliefs, root_support)
        root_structure = self.admission.build_root_structure(
            state,
            root_candidates,
            support_candidates=root_support,
        )
        root_admission = self._admit_at_depth(
            state,
            beliefs,
            root_candidates,
            mode,
            root_structure,
            depth=0,
        )
        if not root_admission.joint_options:
            raise RuntimeError("No admitted root joint option was generated.")

        prepared_actions = [
            self._prepare_action(
                state,
                beliefs,
                joint,
                depth=0,
                mode=mode,
                admission_at_state=root_admission,
            )
            for joint in root_admission.joint_options
        ]
        future_by_successor: dict[tuple, tuple[float, tuple[SearchStep, ...]]] = {}
        action_values: dict[tuple, float] = {}
        action_traces: dict[tuple, tuple[SearchStep, ...]] = {}
        action_objects: dict[tuple, JointOption] = {}
        for prepared in prepared_actions:
            if prepared.transition.result.success:
                assert prepared.successor_key is not None
                cached_future = future_by_successor.get(prepared.successor_key)
                if cached_future is None:
                    cached_future = self._future_for_prepared(
                        prepared,
                        depth=0,
                        mode=mode,
                        root_structure=root_structure,
                    )
                    future_by_successor[prepared.successor_key] = cached_future
                else:
                    self._increment("root_duplicate_successor_reuses")
                future, future_trace = cached_future
            else:
                future, future_trace = 0.0, ()
            value = prepared.stage_utility + future
            trace = (prepared.step,) + future_trace
            key = prepared.joint_option.stable_key
            action_values[key] = value
            action_traces[key] = trace
            action_objects[key] = prepared.joint_option

        best_key = max(
            action_values,
            key=lambda key: (
                action_values[key],
                self._joint_tie_key(action_objects[key]),
            ),
        )
        return SearchResult(
            joint_option=action_objects[best_key],
            value=action_values[best_key],
            trace=action_traces[best_key],
            action_values=action_values,
            root_admission=root_admission,
            root_support_templates=root_structure.candidate_templates,
            root_selected_templates=root_structure.selected_candidate_templates,
            runtime_stats=self._combined_runtime_stats(),
        )

    def _prepare_action(
        self,
        state: GridState,
        beliefs: Mapping[RobotId, SymbolicBelief],
        joint: JointOption,
        *,
        depth: int,
        mode: PlannerMode,
        admission_at_state: AdmissionResult,
    ) -> _PreparedAction:
        transition = self._simulate_transition(
            state,
            beliefs,
            joint,
            depth=depth,
            mode=mode,
        )
        potential_before = self.env.task_potential(state)
        potential_after = self.env.task_potential(transition.state)
        next_beliefs: Mapping[RobotId, SymbolicBelief] | None = None
        successor_key: tuple | None = None
        if transition.result.success:
            stage = (
                potential_after
                - potential_before
                - self.duration_weight * transition.result.duration
            )
            next_beliefs = self._next_beliefs(
                beliefs,
                transition.state,
                joint,
                mode=mode,
            )
            successor_key = (
                self._state_key(transition.state),
                self._beliefs_key(next_beliefs),
            )
        else:
            stage = (
                -self.failure_cost
                - self.duration_weight * transition.result.duration
            )

        source = ",".join(sorted({belief.source for belief in beliefs.values()}))
        step = SearchStep(
            depth=depth,
            state_before=state,
            state_after=transition.state,
            joint_option=joint,
            success=transition.result.success,
            stage_utility=stage,
            duration=transition.result.duration,
            failure_type=transition.result.failure_type,
            potential_before=potential_before,
            potential_after=potential_after,
            admission=admission_at_state,
            belief_source=source,
        )
        return _PreparedAction(
            joint_option=joint,
            transition=transition,
            next_beliefs=next_beliefs,
            stage_utility=stage,
            step=step,
            successor_key=successor_key,
        )

    def _future_for_prepared(
        self,
        prepared: _PreparedAction,
        *,
        depth: int,
        mode: PlannerMode,
        root_structure: RootStructure,
    ) -> tuple[float, tuple[SearchStep, ...]]:
        if not prepared.transition.result.success:
            return 0.0, ()
        assert prepared.next_beliefs is not None
        return self._value(
            prepared.transition.state,
            prepared.next_beliefs,
            depth=depth + 1,
            mode=mode,
            root_structure=root_structure,
        )

    def _evaluate_action(
        self,
        state: GridState,
        beliefs: Mapping[RobotId, SymbolicBelief],
        joint: JointOption,
        *,
        depth: int,
        mode: PlannerMode,
        root_structure: RootStructure,
        admission_at_state: AdmissionResult,
    ) -> tuple[float, tuple[SearchStep, ...]]:
        prepared = self._prepare_action(
            state,
            beliefs,
            joint,
            depth=depth,
            mode=mode,
            admission_at_state=admission_at_state,
        )
        future, future_trace = self._future_for_prepared(
            prepared,
            depth=depth,
            mode=mode,
            root_structure=root_structure,
        )
        return prepared.stage_utility + future, (prepared.step,) + future_trace

    def _dominance_representatives(
        self,
        prepared_actions: Sequence[_PreparedAction],
    ) -> tuple[_PreparedAction, ...]:
        """Keep only the best prefix for each identical successor information state.

        Two branches with the same successor simulator state and the same successor
        local beliefs have identical continuation values at a fixed depth.  The one
        with lower immediate utility is therefore strictly dominated.  This pruning
        is exact and method-independent; it does not rank different successors.
        """

        if not self.enable_successor_dominance:
            return tuple(prepared_actions)

        best_by_successor: dict[tuple, _PreparedAction] = {}
        best_failure: _PreparedAction | None = None
        for prepared in prepared_actions:
            if prepared.successor_key is None:
                if best_failure is None or (
                    prepared.stage_utility,
                    self._joint_tie_key(prepared.joint_option),
                ) > (
                    best_failure.stage_utility,
                    self._joint_tie_key(best_failure.joint_option),
                ):
                    best_failure = prepared
                continue
            incumbent = best_by_successor.get(prepared.successor_key)
            if incumbent is None or (
                prepared.stage_utility,
                self._joint_tie_key(prepared.joint_option),
            ) > (
                incumbent.stage_utility,
                self._joint_tie_key(incumbent.joint_option),
            ):
                best_by_successor[prepared.successor_key] = prepared

        representatives = list(best_by_successor.values())
        if best_failure is not None:
            representatives.append(best_failure)
        pruned = len(prepared_actions) - len(representatives)
        if pruned > 0:
            self._increment("successor_dominance_pruned", pruned)
        return tuple(
            sorted(
                representatives,
                key=lambda item: item.joint_option.stable_key,
            )
        )

    def _value(
        self,
        state: GridState,
        beliefs: Mapping[RobotId, SymbolicBelief],
        *,
        depth: int,
        mode: PlannerMode,
        root_structure: RootStructure,
    ) -> tuple[float, tuple[SearchStep, ...]]:
        if depth >= self.horizon or self.env.is_terminal(state):
            return 0.0, ()

        cache_key = (
            depth,
            mode.value,
            self._state_key(state),
            self._beliefs_key(beliefs),
        )
        cached = self._value_cache.get(cache_key)
        if cached is not None:
            self._increment("value_cache_hits")
            return cached
        self._increment("value_cache_misses")

        current_candidates = self._candidate_sets_for_depth(
            beliefs,
            mode,
            root_structure,
            depth=depth,
        )
        admission = self._admit_at_depth(
            state,
            beliefs,
            current_candidates,
            mode,
            root_structure,
            depth=depth,
        )
        if not admission.joint_options:
            result = (0.0, ())
            self._value_cache[cache_key] = result
            return result

        prepared_actions = [
            self._prepare_action(
                state,
                beliefs,
                joint,
                depth=depth,
                mode=mode,
                admission_at_state=admission,
            )
            for joint in admission.joint_options
        ]
        representatives = self._dominance_representatives(prepared_actions)

        best_value = float("-inf")
        best_trace: tuple[SearchStep, ...] = ()
        for prepared in representatives:
            future, future_trace = self._future_for_prepared(
                prepared,
                depth=depth,
                mode=mode,
                root_structure=root_structure,
            )
            value = prepared.stage_utility + future
            trace = (prepared.step,) + future_trace
            candidate_key = (value, self._joint_tie_key(prepared.joint_option))
            incumbent_key = (
                best_value,
                self._joint_tie_key(best_trace[0].joint_option)
                if best_trace
                else (),
            )
            if candidate_key > incumbent_key:
                best_value = value
                best_trace = trace
        result = (best_value, best_trace)
        self._value_cache[cache_key] = result
        return result
