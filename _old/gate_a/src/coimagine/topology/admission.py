from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Mapping, MutableMapping, Sequence

from coimagine.core.types import JointOption, LocalOption, PlannerMode, RobotId
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.state import GridState
from coimagine.topology.exact_verifier import ExactFactorVerifier
from coimagine.topology.factors import Factor, FactorBuilder


@dataclass(frozen=True)
class AdmissionResult:
    candidate_options: Mapping[RobotId, tuple[LocalOption, ...]]
    joint_options: tuple[JointOption, ...]
    total_cartesian: int
    admitted_count: int
    factor_count: int
    verifier_calls: int
    retained_tuples: int


@dataclass(frozen=True)
class RootStructure:
    # Complete templates supported by the current real observations, before the
    # common top-K computational truncation.
    candidate_templates: Mapping[RobotId, frozenset[tuple]]
    # Templates actually selected into the root top-K candidate bank.  This is
    # retained only for the CST-Frozen-K diagnostic baseline.
    selected_candidate_templates: Mapping[RobotId, frozenset[tuple]]
    support_root_options: Mapping[RobotId, tuple[LocalOption, ...]]
    joint_template_edges: frozenset[tuple]
    concrete_root_options: Mapping[RobotId, tuple[LocalOption, ...]]


class TopologyAdmission:
    def __init__(
        self,
        env: GridEnvironment,
        max_arity: int = 3,
        *,
        enable_runtime_cache: bool = True,
        cache_capacity: int = 4096,
    ) -> None:
        self.env = env
        self.max_arity = max_arity
        self.factor_builder = FactorBuilder()
        self.enable_runtime_cache = bool(enable_runtime_cache)
        self.cache_capacity = max(32, int(cache_capacity))
        self._factor_cache: dict[tuple, Mapping[str, Factor]] = {}
        self._cartesian_cache: dict[tuple, tuple[JointOption, ...]] = {}
        self._factor_stats_cache: dict[tuple, object] = {}
        self._runtime_stats: dict[str, int] = {}

    def clear_runtime_caches(self) -> None:
        self._factor_cache.clear()
        self._cartesian_cache.clear()
        self._factor_stats_cache.clear()
        self._runtime_stats = {}

    @property
    def runtime_stats(self) -> Mapping[str, int]:
        return dict(self._runtime_stats)

    def _increment(self, key: str, amount: int = 1) -> None:
        self._runtime_stats[key] = self._runtime_stats.get(key, 0) + amount

    def _cache_lookup(self, cache: MutableMapping[tuple, object], key: tuple, name: str):
        if not self.enable_runtime_cache:
            self._increment(f"{name}_misses")
            return None
        value = cache.get(key)
        if value is None:
            self._increment(f"{name}_misses")
        else:
            self._increment(f"{name}_hits")
        return value

    def _cache_store(self, cache: MutableMapping[tuple, object], key: tuple, value: object) -> None:
        if not self.enable_runtime_cache:
            return
        if len(cache) >= self.cache_capacity:
            cache.pop(next(iter(cache)))
        cache[key] = value

    @staticmethod
    def _candidate_key(candidates: Mapping[RobotId, Sequence[LocalOption]]) -> tuple:
        return tuple(
            (
                robot_id,
                tuple(option.stable_key for option in candidates[robot_id]),
            )
            for robot_id in sorted(candidates)
        )

    @staticmethod
    def _verification_state_key(state: GridState) -> tuple:
        return (
            tuple(
                sorted(
                    (
                        entity_id,
                        blocker.required_pushers,
                        tuple(sorted(direction.value for direction in blocker.allowed_push_directions)),
                    )
                    for entity_id, blocker in state.blockers.items()
                )
            ),
            tuple(sorted((str(token), int(value)) for token, value in state.task.resource_capacities.items())),
        )

    @staticmethod
    def _build_cartesian(candidates: Mapping[RobotId, Sequence[LocalOption]]) -> tuple[JointOption, ...]:
        robot_ids = tuple(sorted(candidates))
        if any(not candidates[robot_id] for robot_id in robot_ids):
            return ()
        return tuple(
            JointOption(tuple(items))
            for items in product(*(candidates[robot_id] for robot_id in robot_ids))
        )

    def _factors(
        self,
        candidates: Mapping[RobotId, Sequence[LocalOption]],
    ) -> Mapping[str, Factor]:
        key = self._candidate_key(candidates)
        cached = self._cache_lookup(self._factor_cache, key, "factor_build")
        if cached is not None:
            return cached  # type: ignore[return-value]
        factors = self.factor_builder.build(candidates)
        self._cache_store(self._factor_cache, key, factors)
        return factors

    def _cartesian(
        self,
        candidates: Mapping[RobotId, Sequence[LocalOption]],
    ) -> tuple[JointOption, ...]:
        key = self._candidate_key(candidates)
        cached = self._cache_lookup(self._cartesian_cache, key, "cartesian")
        if cached is not None:
            return cached  # type: ignore[return-value]
        cartesian = self._build_cartesian(candidates)
        self._cache_store(self._cartesian_cache, key, cartesian)
        return cartesian

    @staticmethod
    def _selected_for_factor(joint: JointOption, factor: Factor) -> tuple[LocalOption, ...]:
        selected = []
        for option in joint.options:
            if any(
                option.option_id == candidate.option_id
                for candidate in factor.options_by_robot.get(option.robot_id, ())
            ):
                selected.append(option)
        return tuple(selected)

    def _factor_exact(
        self,
        verifier: ExactFactorVerifier,
        factors: Mapping[str, Factor],
        joint: JointOption,
        executable_cache: MutableMapping[tuple, bool],
    ) -> bool:
        for factor in factors.values():
            selected = self._selected_for_factor(joint, factor)
            if not selected:
                continue
            key = (
                factor.factor_id,
                tuple(option.stable_key for option in selected),
            )
            executable = executable_cache.get(key)
            if executable is None:
                executable = verifier.executable(factor, selected)
                executable_cache[key] = executable
                self._increment("factor_assignment_misses")
            else:
                self._increment("factor_assignment_hits")
            if not executable:
                return False
        return True

    def _factor_pairwise(self, state: GridState, factors: Mapping[str, Factor], joint: JointOption) -> bool:
        verifier = ExactFactorVerifier(state, self.max_arity)
        for factor in factors.values():
            selected = self._selected_for_factor(joint, factor)
            if not selected:
                continue
            if len(selected) <= 2:
                if not verifier.executable(factor, selected):
                    return False
            else:
                # Pairwise-clique approximation: every singleton and pair must be
                # executable. This intentionally misses positive higher-order synergy.
                for order in (1, 2):
                    for subset in combinations(selected, order):
                        if not verifier.executable(factor, subset):
                            return False
        return True

    def build_root_structure(
        self,
        state: GridState,
        candidates: Mapping[RobotId, Sequence[LocalOption]],
        *,
        support_candidates: Mapping[RobotId, Sequence[LocalOption]] | None = None,
    ) -> RootStructure:
        """Build root execution structure and observation-support template bank.

        ``candidates`` is the concrete top-K bank used for the root decision.
        ``support_candidates`` is the complete current-observation-supported pool
        before top-K truncation.  Keeping them separate prevents CST from being
        artificially weakened by whichever templates happened to survive root
        candidate ranking.
        """

        exact = self.admit(state, candidates, PlannerMode.FULL_TUPLE_ORACLE)
        support = support_candidates or candidates
        return RootStructure(
            candidate_templates={
                robot_id: frozenset(option.template_key for option in options)
                for robot_id, options in support.items()
            },
            selected_candidate_templates={
                robot_id: frozenset(option.template_key for option in options)
                for robot_id, options in candidates.items()
            },
            support_root_options={
                robot_id: tuple(options) for robot_id, options in support.items()
            },
            joint_template_edges=frozenset(joint.template_key for joint in exact.joint_options),
            concrete_root_options={
                robot_id: tuple(options) for robot_id, options in candidates.items()
            },
        )

    def project_to_root(
        self,
        candidates: Mapping[RobotId, Sequence[LocalOption]],
        root: RootStructure,
        *,
        strict: bool,
        selected_only: bool = False,
    ) -> Mapping[RobotId, tuple[LocalOption, ...]]:
        if strict:
            return root.concrete_root_options
        projected = {}
        template_bank = (
            root.selected_candidate_templates
            if selected_only
            else root.candidate_templates
        )
        for robot_id, options in candidates.items():
            allowed = template_bank.get(robot_id, frozenset())
            projected[robot_id] = tuple(
                option for option in options if option.template_key in allowed
            )
        return projected

    def admit(
        self,
        state: GridState,
        candidates: Mapping[RobotId, Sequence[LocalOption]],
        mode: PlannerMode,
        root_structure: RootStructure | None = None,
        *,
        execution_check: Callable[[JointOption], bool] | None = None,
    ) -> AdmissionResult:
        factors = self._factors(candidates)
        cartesian = self._cartesian(candidates)
        verifier_calls = 0
        retained = 0

        if mode in {PlannerMode.ROOT_STRUCTURE_REUSE, PlannerMode.STRICT_ROOT_REUSE}:
            if root_structure is None:
                raise ValueError("root_structure is required for root reuse modes")
            admitted = tuple(
                joint for joint in cartesian if joint.template_key in root_structure.joint_template_edges
            )
        elif mode == PlannerMode.LOCAL_AVAILABILITY:
            admitted = cartesian
        elif mode == PlannerMode.PAIRWISE_ONLY:
            admitted = tuple(
                joint for joint in cartesian if self._factor_pairwise(state, factors, joint)
            )
            for factor in factors.values():
                n = sum(len(options) for options in factor.options_by_robot.values())
                verifier_calls += n + n * max(0, n - 1) // 2
        elif mode in {
            PlannerMode.RECURSIVE_FACTORS,
            PlannerMode.COST,
            PlannerMode.COST_FROZEN_K,
            PlannerMode.IET,
            PlannerMode.PRIVILEGED_IET_ORACLE,
        }:
            verifier = ExactFactorVerifier(state, self.max_arity)
            verification_state_key = self._verification_state_key(state)
            for factor in factors.values():
                stats_key = (
                    verification_state_key,
                    factor.factor_id,
                    tuple(
                        (
                            robot_id,
                            tuple(option.stable_key for option in options),
                        )
                        for robot_id, options in sorted(factor.options_by_robot.items())
                    ),
                    self.max_arity,
                )
                stats = self._cache_lookup(
                    self._factor_stats_cache,
                    stats_key,
                    "factor_stats",
                )
                if stats is None:
                    stats = verifier.recursive_stats(factor)
                    self._cache_store(self._factor_stats_cache, stats_key, stats)
                verifier_calls += stats.verifier_calls
                retained += stats.retained_tuples
            executable_cache: dict[tuple, bool] = {}
            admitted = tuple(
                joint
                for joint in cartesian
                if self._factor_exact(
                    verifier,
                    factors,
                    joint,
                    executable_cache,
                )
                and (
                    execution_check(joint)
                    if execution_check is not None
                    else self.env.execute(state, joint).result.success
                )
            )
        elif mode in {PlannerMode.EVOLVING, PlannerMode.FULL_TUPLE_ORACLE}:
            admitted_list = []
            for joint in cartesian:
                verifier_calls += 1
                if (
                    execution_check(joint)
                    if execution_check is not None
                    else self.env.execute(state, joint).result.success
                ):
                    admitted_list.append(joint)
            admitted = tuple(admitted_list)
        else:
            raise ValueError(f"Unsupported planner mode: {mode}")

        return AdmissionResult(
            candidate_options={
                robot_id: tuple(options) for robot_id, options in candidates.items()
            },
            joint_options=tuple(sorted(admitted, key=lambda item: item.stable_key)),
            total_cartesian=len(cartesian),
            admitted_count=len(admitted),
            factor_count=len(factors),
            verifier_calls=verifier_calls,
            retained_tuples=retained,
        )
