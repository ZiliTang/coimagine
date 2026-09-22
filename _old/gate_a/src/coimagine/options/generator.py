from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Iterable, List, Mapping, MutableMapping

from coimagine.belief.symbolic import CellKnowledge, SymbolicBelief
from coimagine.core.grid import add_cell, manhattan
from coimagine.core.types import Direction, FactorSignature, LocalOption, OptionKind, TaskKind
from coimagine.options.library import UNIVERSAL_OPTION_LIBRARY


def _option_id(*parts: object) -> str:
    return ":".join(str(part) for part in parts)


def _current_free(belief: SymbolicBelief, cell: tuple[int, int]) -> bool:
    x, y = cell
    if not (0 <= x < belief.width and 0 <= y < belief.height):
        return False
    if cell == belief.own_position:
        return True
    return cell in belief.current_free_cells


def _reachable_current(belief: SymbolicBelief, target: tuple[int, int]) -> bool:
    if belief.own_position == target:
        return True
    if not _current_free(belief, target):
        return False
    queue = deque([belief.own_position])
    visited = {belief.own_position}
    while queue:
        cell = queue.popleft()
        for direction in Direction:
            nxt = add_cell(cell, direction.delta)
            if nxt in visited or not _current_free(belief, nxt):
                continue
            if nxt == target:
                return True
            visited.add(nxt)
            queue.append(nxt)
    return False


def _known_free(belief: SymbolicBelief, cell: tuple[int, int]) -> bool:
    x, y = cell
    if not (0 <= x < belief.width and 0 <= y < belief.height):
        return False
    if cell == belief.own_position:
        return True
    if cell in belief.teammate_positions.values():
        return False
    if any(cell in blocker.cells for blocker in belief.blockers.values()):
        return False
    return (
        cell in belief.current_free_cells
        or belief.cell_knowledge(cell) == CellKnowledge.FREE
    )


def _reachable_known(belief: SymbolicBelief, target: tuple[int, int]) -> bool:
    """Reachability over the robot's persistent, observation-grounded free-space memory."""

    if belief.own_position == target:
        return True
    if not _known_free(belief, target):
        return False
    queue = deque([belief.own_position])
    visited = {belief.own_position}
    while queue:
        cell = queue.popleft()
        for direction in Direction:
            nxt = add_cell(cell, direction.delta)
            if nxt in visited or not _known_free(belief, nxt):
                continue
            if nxt == target:
                return True
            visited.add(nxt)
            queue.append(nxt)
    return False


def _flood_reachable(
    belief: SymbolicBelief,
    *,
    use_known_memory: bool,
) -> frozenset[tuple[int, int]]:
    """Compute one connected reachable set for all option checks in a belief.

    The previous implementation launched a fresh BFS for every anchor and contact
    support.  This flood fill is semantically identical, but its result can be
    shared by all candidate-generation and template-instantiation queries.
    """

    traversable = _known_free if use_known_memory else _current_free
    queue = deque([belief.own_position])
    visited = {belief.own_position}
    while queue:
        cell = queue.popleft()
        for direction in Direction:
            nxt = add_cell(cell, direction.delta)
            if nxt in visited or not traversable(belief, nxt):
                continue
            visited.add(nxt)
            queue.append(nxt)
    return frozenset(visited)


def _shared_resource_tokens(
    belief: SymbolicBelief, cell: tuple[int, int] | None
) -> frozenset[str]:
    if cell is None:
        return frozenset()
    return frozenset(
        token for token, region in belief.task.resource_regions.items() if cell in region
    )


def _nearest_reachable(
    belief: SymbolicBelief,
    cells: frozenset[tuple[int, int]],
    reachable: frozenset[tuple[int, int]],
) -> tuple[int, int] | None:
    candidates = [cell for cell in cells if cell in reachable]
    if not candidates:
        return None
    return min(candidates, key=lambda cell: (manhattan(cell, belief.own_position), cell))


def _metadata_cells(raw: object) -> tuple[tuple[int, int], ...]:
    if raw is None:
        return ()
    cells: list[tuple[int, int]] = []
    for item in raw:  # type: ignore[union-attr]
        if isinstance(item, (list, tuple)) and len(item) == 2:
            cells.append((int(item[0]), int(item[1])))
    return tuple(cells)


class GridOptionGenerator:
    """Generate observation-grounded local options from a robot-owned belief."""

    option_library = UNIVERSAL_OPTION_LIBRARY

    def __init__(
        self,
        candidate_budget: int = 32,
        *,
        anchor_stride: int = 2,
        max_reposition_candidates: int = 4,
        enable_runtime_cache: bool = True,
        cache_capacity: int = 4096,
    ) -> None:
        if candidate_budget < 1:
            raise ValueError("candidate_budget must be positive")
        if anchor_stride < 1:
            raise ValueError("anchor_stride must be positive")
        self.candidate_budget = candidate_budget
        self.anchor_stride = anchor_stride
        self.max_reposition_candidates = max_reposition_candidates
        self.enable_runtime_cache = bool(enable_runtime_cache)
        self.cache_capacity = max(32, int(cache_capacity))
        self._reachable_cache: dict[tuple, frozenset[tuple[int, int]]] = {}
        self._support_pool_cache: dict[tuple, tuple[LocalOption, ...]] = {}
        self._instantiation_cache: dict[tuple, tuple[LocalOption, ...]] = {}
        self._selection_cache: dict[tuple, tuple[LocalOption, ...]] = {}
        self._belief_key_cache: dict[int, SymbolicBelief] = {}
        self._runtime_stats: dict[str, int] = {}

    def clear_runtime_caches(self) -> None:
        self._reachable_cache.clear()
        self._support_pool_cache.clear()
        self._instantiation_cache.clear()
        self._selection_cache.clear()
        self._belief_key_cache.clear()
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

    def _reachable_cells(
        self,
        belief: SymbolicBelief,
        *,
        use_known_memory: bool,
    ) -> frozenset[tuple[int, int]]:
        key = (self._belief_key(belief), use_known_memory)
        cached = self._cache_lookup(
            self._reachable_cache,
            key,
            "reachable_known" if use_known_memory else "reachable_current",
        )
        if cached is not None:
            return cached  # type: ignore[return-value]
        reachable = _flood_reachable(
            belief,
            use_known_memory=use_known_memory,
        )
        self._cache_store(self._reachable_cache, key, reachable)
        return reachable

    def _belief_key(self, belief: SymbolicBelief) -> int:
        object_id = id(belief)
        cached = self._belief_key_cache.get(object_id)
        if cached is not None and cached is not belief:
            raise RuntimeError("Belief object id was unexpectedly reused inside one plan")
        self._belief_key_cache[object_id] = belief
        return object_id

    def _reposition_score(
        self,
        belief: SymbolicBelief,
        cell: tuple[int, int],
    ) -> tuple[int, int, tuple[int, int]]:
        unknown_neighbors = sum(
            belief.cell_knowledge(add_cell(cell, direction.delta))
            == CellKnowledge.UNKNOWN
            for direction in Direction
        )
        return (-unknown_neighbors, manhattan(cell, belief.own_position), cell)

    def _reachable_reposition_anchors(
        self,
        belief: SymbolicBelief,
        reachable_current: frozenset[tuple[int, int]] | None = None,
    ) -> tuple[tuple[int, int], ...]:
        """Return the complete root-observation-supported anchor pool.

        This method deliberately does not apply the reposition quota.  The full
        reachable pool defines observation support for CST; computational quotas
        are applied later when concrete candidates are selected at each depth.
        """

        explicit = _metadata_cells(belief.task.metadata.get("reposition_anchors"))
        enabled = bool(
            belief.task.metadata.get(
                "enable_reposition",
                bool(explicit),
            )
        )
        if not enabled:
            return ()
        if explicit:
            pool = explicit
        else:
            stride = int(
                belief.task.metadata.get("anchor_stride", self.anchor_stride)
            )
            pool = tuple(
                (x, y)
                for x in range(0, belief.width, stride)
                for y in range(0, belief.height, stride)
            )

        reachable = (
            reachable_current
            if reachable_current is not None
            else self._reachable_cells(belief, use_known_memory=False)
        )
        candidates = {
            cell
            for cell in pool
            if cell != belief.own_position and cell in reachable
        }
        return tuple(
            sorted(candidates, key=lambda cell: self._reposition_score(belief, cell))
        )

    def _select_reposition_anchors(
        self,
        belief: SymbolicBelief,
        pool: Iterable[tuple[int, int]],
    ) -> tuple[tuple[int, int], ...]:
        """Select a small, side-aware and spatially diverse anchor subset."""

        ordered = tuple(
            sorted(
                set(pool),
                key=lambda cell: self._reposition_score(belief, cell),
            )
        )
        limit = max(
            0,
            int(
                belief.task.metadata.get(
                    "max_reposition_candidates", self.max_reposition_candidates
                )
            ),
        )
        if limit == 0 or not ordered:
            return ()
        if len(ordered) <= limit:
            return ordered

        selected: list[tuple[int, int]] = []
        raw_rows = belief.task.metadata.get("channel_boundary_rows", ())
        parsed_rows = tuple(int(value) for value in raw_rows) if raw_rows else ()
        channel_y_raw = belief.task.metadata.get("channel_y")
        if channel_y_raw is not None:
            lower_boundary = min(parsed_rows) if parsed_rows else int(channel_y_raw)
            upper_boundary = max(parsed_rows) if parsed_rows else int(channel_y_raw)
            lower = [cell for cell in ordered if cell[1] < lower_boundary]
            upper = [cell for cell in ordered if cell[1] > upper_boundary]
            # Preserve one representative on each side whenever both are currently
            # reachable.  This prevents a near-side cluster from consuming the full
            # reposition quota before cross-side preparation is considered.
            if lower and upper and limit >= 2:
                selected.extend((lower[0], upper[0]))
            elif lower:
                selected.append(lower[0])
            elif upper:
                selected.append(upper[0])

        if not selected:
            selected.append(ordered[0])

        while len(selected) < limit:
            remaining = [cell for cell in ordered if cell not in selected]
            if not remaining:
                break
            # Farthest-point completion preserves spatial coverage; frontier score
            # and travel distance remain deterministic tie-breakers.
            next_cell = min(
                remaining,
                key=lambda cell: (
                    -min(manhattan(cell, chosen) for chosen in selected),
                    self._reposition_score(belief, cell),
                ),
            )
            selected.append(next_cell)
        return tuple(selected)

    @staticmethod
    def _priority(belief: SymbolicBelief, option: LocalOption) -> tuple:
        ready_push = (
            option.kind == OptionKind.PUSH
            and option.target == belief.own_position
        )
        group = {
            True: 0,
            False: {
                OptionKind.HOLD: 1,
                OptionKind.OBSERVE: 2,
                OptionKind.APPROACH: 3,
                OptionKind.PUSH: 4,
                OptionKind.REPOSITION: 5,
                OptionKind.IDLE: 6,
            }[option.kind],
        }
        rank = group[ready_push] if ready_push else group[False]
        return (rank, option.duration_hint, option.stable_key)

    def select_candidates(
        self,
        belief: SymbolicBelief,
        support_pool: Iterable[LocalOption],
    ) -> tuple[LocalOption, ...]:
        """Apply per-kind diversification and the common top-K compute budget."""

        pool = tuple(support_pool)
        cache_key = (
            self._belief_key(belief),
            tuple(option.stable_key for option in pool),
            self.candidate_budget,
            self.max_reposition_candidates,
        )
        cached = self._cache_lookup(self._selection_cache, cache_key, "candidate_selection")
        if cached is not None:
            return cached  # type: ignore[return-value]
        reposition_by_target = {
            option.target: option
            for option in pool
            if option.kind == OptionKind.REPOSITION and option.target is not None
        }
        selected_anchor_targets = self._select_reposition_anchors(
            belief, reposition_by_target
        )
        selected_reposition = [
            reposition_by_target[target]
            for target in selected_anchor_targets
            if target in reposition_by_target
        ]
        eligible = [
            option for option in pool if option.kind != OptionKind.REPOSITION
        ] + selected_reposition
        ordered = sorted(eligible, key=lambda option: self._priority(belief, option))

        selected = list(ordered[: self.candidate_budget])
        idle = next(
            (option for option in ordered if option.kind == OptionKind.IDLE),
            None,
        )
        if idle is not None and idle not in selected:
            if selected:
                selected[-1] = idle
            else:
                selected.append(idle)

        # Reserve a small side-diverse reposition footprint when reposition is
        # enabled.  This quota is shared by CST and IET and therefore does not
        # alter their information advantage; it only avoids a kind-collapse of
        # the common top-K candidate bank.
        reserve_count = min(
            len(selected_reposition),
            2 if self.candidate_budget >= 6 else (1 if self.candidate_budget >= 3 else 0),
        )
        reserved = selected_reposition[:reserve_count]
        for option in reserved:
            if option in selected:
                continue
            replaceable = [
                index
                for index, incumbent in enumerate(selected)
                if incumbent.kind not in {OptionKind.IDLE, OptionKind.REPOSITION}
            ]
            if replaceable:
                selected[replaceable[-1]] = option
            elif len(selected) < self.candidate_budget:
                selected.append(option)

        # Remove possible duplicates introduced by quota replacement, then refill.
        deduplicated: list[LocalOption] = []
        seen = set()
        for option in selected:
            if option.stable_key in seen:
                continue
            seen.add(option.stable_key)
            deduplicated.append(option)
        for option in ordered:
            if len(deduplicated) >= self.candidate_budget:
                break
            if option.stable_key not in seen:
                seen.add(option.stable_key)
                deduplicated.append(option)
        result = tuple(
            sorted(
                deduplicated[: self.candidate_budget],
                key=lambda option: self._priority(belief, option),
            )
        )
        self._cache_store(self._selection_cache, cache_key, result)
        return result

    def generate_support_pool(self, belief: SymbolicBelief) -> tuple[LocalOption, ...]:
        """Generate every option template supported by the current observation."""
        belief_key = self._belief_key(belief)
        cached = self._cache_lookup(self._support_pool_cache, belief_key, "support_pool")
        if cached is not None:
            return cached  # type: ignore[return-value]

        reachable_current = self._reachable_cells(
            belief,
            use_known_memory=False,
        )
        candidates: List[LocalOption] = [
            LocalOption(
                option_id=_option_id(belief.robot_id, "idle"),
                robot_id=belief.robot_id,
                kind=OptionKind.IDLE,
                role_token="idle",
                slot_token="idle",
            )
        ]

        for anchor in self._reachable_reposition_anchors(
            belief,
            reachable_current,
        ):
            candidates.append(
                LocalOption(
                    option_id=_option_id(belief.robot_id, "reposition", anchor),
                    robot_id=belief.robot_id,
                    kind=OptionKind.REPOSITION,
                    target=anchor,
                    duration_hint=float(max(1, manhattan(anchor, belief.own_position))),
                    role_token="reposition",
                    slot_token=f"anchor:{anchor[0]}:{anchor[1]}",
                    signature=FactorSignature(
                        resource_tokens=frozenset({f"cell:{anchor[0]}:{anchor[1]}"})
                        | _shared_resource_tokens(belief, anchor)
                    ),
                )
            )

        for entity_id in sorted(belief.blockers):
            blocker = belief.blockers[entity_id]
            # Current options require direct observation support, not only memory.
            if not blocker.visible_cells:
                continue
            if belief.task.kind in {
                TaskKind.CLEARING,
                TaskKind.REVEAL_ACT,
                TaskKind.STABLE_CONTROL,
            }:
                deactivate = bool(
                    belief.task.metadata.get("deactivate_cleared_entities", True)
                )
                if deactivate and belief.task.corridor and not any(
                    cell in belief.task.corridor for cell in blocker.cells
                ):
                    continue
            dependency = belief.task.activation_dependencies.get(entity_id)
            if dependency is not None:
                prerequisite = belief.blockers.get(dependency)
                if prerequisite is not None and any(
                    cell in belief.task.corridor for cell in prerequisite.cells
                ):
                    continue

            for direction in blocker.allowed_push_directions:
                dx, dy = direction.delta
                boundary_contacts = [
                    contact
                    for contact in sorted(blocker.cells)
                    if contact in blocker.visible_cells
                    and (contact[0] - dx, contact[1] - dy) not in blocker.cells
                ]
                # Slot ranks are computed on the full recognized footprint, so the
                # same physical slot has a stable template key across viewpoints.
                full_boundary = [
                    contact
                    for contact in sorted(blocker.cells)
                    if (contact[0] - dx, contact[1] - dy) not in blocker.cells
                ]
                rank_by_cell = {cell: rank for rank, cell in enumerate(full_boundary)}
                for contact in boundary_contacts:
                    support = (contact[0] - dx, contact[1] - dy)
                    if support not in reachable_current:
                        continue
                    resource = f"cell:{support[0]}:{support[1]}"
                    slot = f"{direction.value}:{rank_by_cell[contact]}"

                    # Push templates are created before contact execution. Admission
                    # decides whether they are executable at the current depth.
                    candidates.append(
                        LocalOption(
                            option_id=_option_id(
                                belief.robot_id,
                                "push",
                                entity_id,
                                direction.value,
                                contact,
                            ),
                            robot_id=belief.robot_id,
                            kind=OptionKind.PUSH,
                            entity_id=entity_id,
                            direction=direction,
                            target=support,
                            contact_cell=contact,
                            role_token="push",
                            slot_token=slot,
                            signature=FactorSignature(
                                entity_ids=frozenset({entity_id}),
                                resource_tokens=frozenset({resource})
                                | _shared_resource_tokens(belief, support),
                                temporal_tokens=frozenset({f"sync:{entity_id}"}),
                            ),
                        )
                    )
                    if belief.own_position != support:
                        candidates.append(
                            LocalOption(
                                option_id=_option_id(
                                    belief.robot_id,
                                    "approach",
                                    entity_id,
                                    direction.value,
                                    support,
                                ),
                                robot_id=belief.robot_id,
                                kind=OptionKind.APPROACH,
                                entity_id=entity_id,
                                direction=direction,
                                target=support,
                                contact_cell=contact,
                                duration_hint=float(
                                    max(1, manhattan(support, belief.own_position))
                                ),
                                role_token="approach_contact",
                                slot_token=slot,
                                signature=FactorSignature(
                                    resource_tokens=frozenset({resource})
                                    | _shared_resource_tokens(belief, support)
                                ),
                            )
                        )

        task = belief.task
        if task.kind == TaskKind.HOLD_PASS:
            # The payload role is task-assigned; only support robots may prepare
            # and execute the temporary gate hold.
            if belief.robot_id != task.payload_robot_id:
                hold_target = _nearest_reachable(
                    belief,
                    task.hold_cells,
                    reachable_current,
                )
                if belief.own_position in task.hold_cells:
                    candidates.append(
                        LocalOption(
                            option_id=_option_id(belief.robot_id, "hold", "gate"),
                            robot_id=belief.robot_id,
                            kind=OptionKind.HOLD,
                            role_token="hold_gate",
                            slot_token="gate",
                            signature=FactorSignature(
                                resource_tokens=frozenset({"gate:hold"}),
                                temporal_tokens=frozenset({"sync:hold_pass"}),
                            ),
                        )
                    )
                elif hold_target is not None:
                    candidates.append(
                        LocalOption(
                            option_id=_option_id(
                                belief.robot_id, "approach_hold", hold_target
                            ),
                            robot_id=belief.robot_id,
                            kind=OptionKind.APPROACH,
                            target=hold_target,
                            role_token="approach_hold",
                            slot_token="gate",
                            signature=FactorSignature(
                                resource_tokens=frozenset(
                                    {f"cell:{hold_target[0]}:{hold_target[1]}"}
                                )
                                | _shared_resource_tokens(belief, hold_target)
                            ),
                        )
                    )

            if belief.robot_id == task.payload_robot_id:
                goal = _nearest_reachable(
                    belief,
                    task.goal_region,
                    reachable_current,
                )
                if goal is not None and belief.own_position not in task.goal_region:
                    candidates.append(
                        LocalOption(
                            option_id=_option_id(belief.robot_id, "pass_goal", goal),
                            robot_id=belief.robot_id,
                            kind=OptionKind.APPROACH,
                            target=goal,
                            role_token="pass_goal",
                            slot_token="payload_goal",
                            signature=FactorSignature(
                                resource_tokens=frozenset({"passage:guarded"})
                                | _shared_resource_tokens(belief, goal),
                                temporal_tokens=frozenset({"sync:hold_pass"}),
                            ),
                        )
                    )

        if task.kind == TaskKind.REVEAL_ACT:
            hidden = sorted(set(task.hidden_entities) - set(belief.revealed_entities))
            if hidden:
                entity_id = hidden[0]
                if belief.own_position in task.observation_cells:
                    candidates.append(
                        LocalOption(
                            option_id=_option_id(
                                belief.robot_id, "observe", entity_id
                            ),
                            robot_id=belief.robot_id,
                            kind=OptionKind.OBSERVE,
                            entity_id=entity_id,
                            role_token="observe_hidden",
                            slot_token=entity_id,
                            signature=FactorSignature(
                                temporal_tokens=frozenset({f"observe:{entity_id}"})
                            ),
                        )
                    )
                else:
                    target = _nearest_reachable(
                        belief,
                        task.observation_cells,
                        reachable_current,
                    )
                    if target is not None:
                        candidates.append(
                            LocalOption(
                                option_id=_option_id(
                                    belief.robot_id, "approach_observe", target
                                ),
                                robot_id=belief.robot_id,
                                kind=OptionKind.APPROACH,
                                target=target,
                                entity_id=entity_id,
                                role_token="approach_observe",
                                slot_token=entity_id,
                                signature=FactorSignature(
                                    resource_tokens=frozenset(
                                        {f"cell:{target[0]}:{target[1]}"}
                                    )
                                    | _shared_resource_tokens(belief, target)
                                ),
                            )
                        )

        result = tuple(
            sorted(candidates, key=lambda option: self._priority(belief, option))
        )
        self._cache_store(self._support_pool_cache, belief_key, result)
        return result

    def instantiate_templates(
        self,
        belief: SymbolicBelief,
        templates: Iterable[LocalOption],
    ) -> tuple[LocalOption, ...]:
        """Re-instantiate previously observed templates in a successor belief.

        Object-relative contact slots are remapped to the predicted object pose.
        No unseen contact slot, direction variation, or anchor is introduced.
        """

        template_tuple = tuple(templates)
        cache_key = (
            self._belief_key(belief),
            tuple(option.stable_key for option in template_tuple),
        )
        cached = self._cache_lookup(
            self._instantiation_cache,
            cache_key,
            "template_instantiation",
        )
        if cached is not None:
            return cached  # type: ignore[return-value]

        reachable_known = self._reachable_cells(
            belief,
            use_known_memory=True,
        )
        instantiated: list[LocalOption] = []
        for template in template_tuple:
            if template.robot_id != belief.robot_id:
                continue
            if template.kind == OptionKind.IDLE:
                instantiated.append(template)
                continue
            if template.kind == OptionKind.REPOSITION:
                target = template.target
                if target is None or target not in reachable_known:
                    continue
                instantiated.append(
                    replace(
                        template,
                        option_id=_option_id(belief.robot_id, "reposition", target),
                        target=target,
                        duration_hint=float(
                            max(1, manhattan(target, belief.own_position))
                        ),
                        signature=FactorSignature(
                            resource_tokens=frozenset(
                                {f"cell:{target[0]}:{target[1]}"}
                            )
                            | _shared_resource_tokens(belief, target)
                        ),
                    )
                )
                continue
            if (
                template.kind in {OptionKind.PUSH, OptionKind.APPROACH}
                and template.entity_id is not None
                and template.direction is not None
                and template.slot_token is not None
                and template.role_token in {"push", "approach_contact"}
            ):
                blocker = belief.blockers.get(template.entity_id)
                if blocker is None:
                    continue
                direction = template.direction
                dx, dy = direction.delta
                full_boundary = [
                    contact
                    for contact in sorted(blocker.cells)
                    if (contact[0] - dx, contact[1] - dy) not in blocker.cells
                ]
                try:
                    slot_direction, slot_rank_raw = template.slot_token.split(":", 1)
                    slot_rank = int(slot_rank_raw)
                except (ValueError, TypeError):
                    continue
                if slot_direction != direction.value or not (0 <= slot_rank < len(full_boundary)):
                    continue
                contact = full_boundary[slot_rank]
                support = (contact[0] - dx, contact[1] - dy)
                if support not in reachable_known:
                    continue
                resource = f"cell:{support[0]}:{support[1]}"
                signature = FactorSignature(
                    entity_ids=(
                        frozenset({template.entity_id})
                        if template.kind == OptionKind.PUSH
                        else frozenset()
                    ),
                    resource_tokens=frozenset({resource})
                    | _shared_resource_tokens(belief, support),
                    temporal_tokens=(
                        frozenset({f"sync:{template.entity_id}"})
                        if template.kind == OptionKind.PUSH
                        else frozenset()
                    ),
                )
                instantiated.append(
                    replace(
                        template,
                        option_id=_option_id(
                            belief.robot_id,
                            template.kind.value,
                            template.entity_id,
                            direction.value,
                            contact if template.kind == OptionKind.PUSH else support,
                        ),
                        target=support,
                        contact_cell=contact,
                        duration_hint=(
                            1.0
                            if template.kind == OptionKind.PUSH
                            else float(
                                max(1, manhattan(support, belief.own_position))
                            )
                        ),
                        signature=signature,
                    )
                )
                continue

            target = template.target
            if target is not None and target not in reachable_known:
                continue
            instantiated.append(
                replace(
                    template,
                    duration_hint=(
                        float(max(1, manhattan(target, belief.own_position)))
                        if target is not None
                        else template.duration_hint
                    ),
                )
            )

        by_template: dict[tuple, LocalOption] = {}
        for option in instantiated:
            by_template[option.template_key] = option
        result = tuple(
            sorted(
                by_template.values(),
                key=lambda option: self._priority(belief, option),
            )
        )
        self._cache_store(self._instantiation_cache, cache_key, result)
        return result

    def generate(self, belief: SymbolicBelief) -> tuple[LocalOption, ...]:
        return self.select_candidates(belief, self.generate_support_pool(belief))
