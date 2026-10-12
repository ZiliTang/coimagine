"""Conditional per-parameter state propagation, through the whole planning and update loop.

The problem this fixes
----------------------
The shipped factored pipeline carries **one** dynamic vector plus ``b(theta)``::

    s_t   +   b_t(theta)

Every transition is evaluated by copying ``s_t`` sixteen times and mixing the results.  A
world where the release succeeded and a world where it failed are represented by one
released probability, and the next step is evaluated from combinations that never occurred.
Measured on two short chains with **exact** dynamics, that factorisation alone (no learning
error) moves the position distribution by TV 0.375-0.609 and the conditional release
probability by up to 0.75.

The representation here is::

    { b_t(theta), s_t^theta } for theta = 1..16

``s_t^theta`` is "what the state is now, if the parameters are this one".  Aggregating to a
position distribution, an expected distance or a success probability for **scoring** is
allowed and done; the aggregate is never fed back as the state of the next step.

The four links the association has to survive, and where each one lives here
----------------------------------------------------------------------------
1. **Action expansion** -- :func:`expand` advances every parameter row with ``model.step``
   under its own one-hot, so each hypothesis predicts from its own state.
2. **Feedback prediction and update** -- :func:`per_theta_outcome` reads each row's own
   distribution over feedback symbols (the diagonal of ``model.observe``), and
   :func:`condition` reweights ``b(theta)`` by that row's likelihood.  No mix-then-copy.
3. **Search branching** -- :func:`plan_with_conditional_states` and
   :func:`beam_value_conditional` carry the full ``(rows, belief)`` pair through every
   branch and every beam level; deeper levels do not re-compress the state.
4. **Post-execution update** -- :func:`correct_observables` replaces the *observable* part
   of every row (object cell and robot poses) with the new scan and keeps each row's own
   hidden ``released`` value.  The belief is reweighted before that, from the rows that
   actually produced the feedback.

Residual approximation, stated rather than hidden: a row is still a within-parameter
distribution over object cells plus a released probability, so the ``(cell, released)``
correlation *inside* one parameter is not represented.  This is not an exact joint belief;
it is strictly more than the shipped scheme, which also collapses across parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..grid.dynamics import FEEDBACK_SYMBOLS, execute
from ..grid.planner import PHYSICS
from ..grid.state import ObjectState, Scene
from ..learning.belief_model import DYNAMIC_DIM, N_PHYSICS
from ..learning.data import _joint_for
from ..learning.features import ACTION_IDS

_EPS = 1e-12
SYMBOL_THRESHOLD = 0.02
PROGRESS_WEIGHT = 3.0
FAIL_COST = 20.0


@dataclass
class ConditionalState:
    """``rows[theta]`` is parameter ``theta``'s own dynamic vector; ``belief`` weights them."""

    rows: np.ndarray  # (N_PHYSICS, DYNAMIC_DIM)
    belief: np.ndarray  # (N_PHYSICS,)

    def copy(self) -> "ConditionalState":
        return ConditionalState(self.rows.copy(), self.belief.copy())

    def aggregate(self) -> np.ndarray:
        """The belief-mixed dynamic vector, for scoring only."""

        return np.einsum("t,td->d", self.belief, self.rows)

    def position(self) -> np.ndarray:
        density = np.clip(self.rows[:, 1 : 1 + 36], 0.0, None)
        density = density / np.maximum(density.sum(axis=1, keepdims=True), _EPS)
        return self.belief @ density

    def goal_probability(self, target: frozenset) -> float:
        indices = [index for index, origin in enumerate(_origins()) if ObjectState(origin).within(target)]
        return float(self.position()[indices].sum()) if indices else 0.0

    def released_probability(self) -> float:
        return float(self.belief @ np.clip(self.rows[:, 0], 0.0, 1.0))

    def released_by_theta(self) -> np.ndarray:
        return np.clip(self.rows[:, 0], 0.0, 1.0).copy()


def _origins() -> Tuple[Tuple[int, int], ...]:
    return tuple((x, y) for y in range(6) for x in range(6))


def _onehot(index: int) -> np.ndarray:
    vector = np.zeros((1, N_PHYSICS))
    vector[0, index] = 1.0
    return vector


def expand(model, state: ConditionalState, action: int) -> np.ndarray:
    """Link 1: every parameter row advances from *its own* state.

    Returns the next rows ``(N_PHYSICS, DYNAMIC_DIM)``.  The cost is **not** returned here:
    ``BeliefWorldModel.cost_actions`` pairs row ``i`` with action ``i``, so a single row
    would always be scored as action 0.  Costs are read once for all six actions in
    :func:`expand_actions`.
    """

    next_rows = np.zeros_like(state.rows)
    for index in range(N_PHYSICS):
        next_rows[index] = model.step(
            state.rows[index : index + 1], action, _onehot(index)
        )[0]
    return next_rows


def expand_actions(model, state: ConditionalState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All six actions at once: ``(N_ACTIONS, N_PHYSICS, DYNAMIC_DIM)``, costs, aggregates.

    The cost head is queried once with the six belief-mixed rows, which is the layout it
    expects (row ``a`` pairs with action ``a``).
    """

    rows = np.zeros((len(ACTION_IDS), N_PHYSICS, DYNAMIC_DIM))
    for action in range(len(ACTION_IDS)):
        rows[action] = expand(model, state, action)
    aggregate = np.einsum("t,atd->ad", state.belief, rows)
    costs = np.asarray(model.cost_actions(aggregate, state.belief[None, :]))
    return rows, costs, aggregate


def per_theta_outcome(model, rows: np.ndarray, action: int) -> np.ndarray:
    """Link 2a: each row's own feedback distribution.

    ``model.observe`` evaluates every parameter against every input row; the entry we want is
    the diagonal ``P(y | a, s_theta, theta)``.
    """

    table = model.observe(rows, action, np.full((1, N_PHYSICS), 1.0 / N_PHYSICS))
    return np.array([table[index, index] for index in range(N_PHYSICS)])


def condition(belief: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """Link 2b: ``b'(theta) ∝ P(y | a, s_theta, theta) b(theta)``."""

    posterior = belief * likelihood
    total = posterior.sum()
    if total <= _EPS:
        return belief.copy()
    return posterior / total


def correct_observables(rows: np.ndarray, observed: np.ndarray) -> np.ndarray:
    """Link 4: correct the observable part of every row, keep each row's hidden part.

    ``observed`` is the encoder's dynamic vector for the new scan.  The object cell and the
    robot poses are observable, so all rows take them; ``released`` is hidden, so each row
    keeps its own.  This is the per-row form of the shipped ``fuse_hidden_state``.
    """

    corrected = np.repeat(np.atleast_2d(observed), rows.shape[0], axis=0)
    corrected[:, 0] = rows[:, 0]
    return corrected


# ---------------------------------------------------------------------------
# Scoring (aggregates only; never fed back as a state)
# ---------------------------------------------------------------------------


def _distance_table(goal: frozenset) -> np.ndarray:
    table = np.zeros(36)
    for index, origin in enumerate(_origins()):
        members = [(origin[0] + dx, origin[1] + dy) for dx in (0, 1) for dy in (0, 1)]
        # same formula as belief_probe._cell_distance_distribution: the worst cell's distance
        # to its *nearest* goal cell.  The other inversion is the defect fixed in round 20.
        table[index] = float(
            max(min(abs(c[0] - g[0]) + abs(c[1] - g[1]) for g in goal) for c in members)
        )
    return table


def cut_value(state: ConditionalState, distance_table: np.ndarray) -> float:
    """The same cut the shipped planner uses, evaluated on the aggregated state."""

    density = state.position()
    distance = float(density @ distance_table)
    at_goal = float(density[distance_table < 0.5].sum())
    return PROGRESS_WEIGHT * distance + FAIL_COST * (1.0 - at_goal)


def beam_value_conditional(
    model,
    state: ConditionalState,
    distance_table: np.ndarray,
    *,
    steps: int,
    beam: int,
) -> float:
    """Link 3: the beam keeps whole conditional states, not a re-compressed summary."""

    states = [state]
    costs = [0.0]
    for _ in range(steps):
        candidates: List[ConditionalState] = []
        candidate_costs: List[float] = []
        for partial, spent in zip(states, costs):
            rows, action_costs, _aggregate = expand_actions(model, partial)
            for action in range(len(ACTION_IDS)):
                candidates.append(ConditionalState(rows[action], partial.belief.copy()))
                candidate_costs.append(spent + float(action_costs[action]))
        scored = [
            (candidate_costs[i] + cut_value(candidates[i], distance_table), i)
            for i in range(len(candidates))
        ]
        scored.sort(key=lambda pair: (pair[0], pair[1]))
        keep = [index for _value, index in scored[:beam]]
        states = [candidates[index] for index in keep]
        costs = [candidate_costs[index] for index in keep]
    return float(min(costs[i] + cut_value(states[i], distance_table) for i in range(len(states))))


def plan_with_conditional_states(
    model,
    goal: frozenset,
    state: ConditionalState,
    *,
    depth: int = 2,
    beam: int = 1,
) -> Tuple[int, List[Dict[str, object]]]:
    """The shipped ``plan_with_search`` with the conditional state in place of one vector.

    Same candidate set, same branch threshold, same terminal cut, same tie-break on the
    action name.  The only difference is what is carried through a branch.
    """

    distance_table = _distance_table(goal)
    rows, action_costs, _aggregate = expand_actions(model, state)
    branches: List[Dict[str, object]] = []
    for action in range(len(ACTION_IDS)):
        likelihood = per_theta_outcome(model, rows[action], action)
        outcome = state.belief @ likelihood  # (N_OBSERVATION_SYMBOLS,)
        expected = 0.0
        total_probability = 0.0
        for symbol in range(len(outcome)):
            probability = float(outcome[symbol])
            if probability < SYMBOL_THRESHOLD:
                continue
            posterior = condition(state.belief, likelihood[:, symbol])
            child = ConditionalState(rows[action], posterior)
            value = beam_value_conditional(
                model, child, distance_table, steps=depth - 1, beam=beam
            )
            expected += probability * value
            total_probability += probability
        if total_probability < 1.0:
            expected += (1.0 - total_probability) * cut_value(
                ConditionalState(rows[action], state.belief), distance_table
            )
        branches.append(
            {
                "action": ACTION_IDS[action],
                "action_index": action,
                "score": float(action_costs[action]) + expected,
                "first_cost": float(action_costs[action]),
                "expected_distance": float(
                    ConditionalState(rows[action], state.belief).position() @ distance_table
                ),
                "released_probability": float(
                    state.belief @ np.clip(rows[action][:, 0], 0.0, 1.0)
                ),
                "symbol_probabilities": [float(p) for p in outcome],
            }
        )
    chosen = min(branches, key=lambda branch: (branch["score"], branch["action"]))
    return int(chosen["action_index"]), branches


# ---------------------------------------------------------------------------
# Episodes: the shipped driver and the conditional driver, side by side
# ---------------------------------------------------------------------------


def encode_state(
    model,
    scene: Scene,
    history_a: Sequence[str] = (),
    public_history: Sequence[str] = (),
) -> np.ndarray:
    """The encoder's dynamic vector for the current scan.

    The histories matter: ``dyn_head`` sees the whole 188-dim observation, including the
    feedback and public-event histograms, so passing empty histories would be a different
    input from the shipped driver.
    """

    from ..grid.sensing import ray_cast
    from ..learning.features import observation_vector

    observation = np.asarray(
        observation_vector(
            ray_cast(scene, 0, radius=5), scene.robot_pose(0),
            tuple(history_a), tuple(public_history),
        )
    )
    return np.atleast_2d(model.dynamic_state(observation))


def initial_conditional_state(model, scene: Scene) -> ConditionalState:
    encoded = encode_state(model, scene)[0]
    rows = np.repeat(encoded[None, :], N_PHYSICS, axis=0)
    return ConditionalState(rows, np.full(N_PHYSICS, 1.0 / N_PHYSICS))


def _record(scene_id, succeeded, cost, actions, reachable):
    return {
        "scene_id": scene_id,
        "success": int(succeeded),
        "action_cost": float(cost),
        "steps": len(actions),
        "actions": list(actions),
        "first_action": actions[0] if actions else None,
        "wait_actions": sum(1 for action in actions if action == "wait"),
        "reachable_steps": reachable,
        "exact_reachable": int(reachable is not None),
    }


def run_shipped_episode(
    model,
    scene: Scene,
    *,
    scene_id: int,
    horizon: int = 8,
    depth: int = 2,
    beam: int = 1,
    warmup: Optional[Tuple[int, int]] = None,
    reachable: Optional[int] = None,
) -> Dict[str, object]:
    """The repo's ``run_search_episode``, with the action sequence recorded for the flips."""

    from .belief_probe import plan_with_search

    goal = frozenset(scene.static.target)
    history_a: List[str] = []
    public_history: List[str] = []
    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    carried: Optional[np.ndarray] = None
    total = 0.0
    actions: List[str] = []

    def observe(current: Scene):
        from ..grid.sensing import ray_cast
        from ..learning.features import observation_vector

        observation = np.asarray(
            observation_vector(
                ray_cast(current, 0, radius=5), current.robot_pose(0),
                tuple(history_a), tuple(public_history),
            )
        )
        observed = model.dynamic_state(observation)
        if carried is None:
            return observation, observed
        from .belief_probe import fuse_hidden_state

        return observation, fuse_hidden_state(observed, carried)

    if warmup is not None:
        action, symbol = warmup
        _obs, dynamic = observe(scene)
        belief = model.update_belief(belief, dynamic, action, symbol)
        carried = model.step(dynamic, action, belief)
        history_a.append(FEEDBACK_SYMBOLS[symbol])

    current = scene
    for _ in range(horizon):
        if current.is_goal():
            break
        observation, dynamic = observe(current)
        chosen, _branches = plan_with_search(
            model, observation, goal, belief=belief, dynamic=dynamic,
            depth=depth, beam=beam,
        )
        action_id = ACTION_IDS[chosen]
        actions.append(action_id)
        successor, result = execute(current, _joint_for(None, action_id))
        total += float(result.cost)
        symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
        belief = model.update_belief(belief, dynamic, chosen, symbol)
        carried = model.step(dynamic, chosen, belief)
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        current = successor
    return _record(scene_id, current.is_goal(), total, actions, reachable)


def run_conditional_episode(
    model,
    scene: Scene,
    *,
    scene_id: int,
    horizon: int = 8,
    depth: int = 2,
    beam: int = 1,
    warmup: Optional[Tuple[int, int]] = None,
    reachable: Optional[int] = None,
) -> Dict[str, object]:
    """The same loop with ``{b(theta), s^theta}`` carried through all four links."""

    goal = frozenset(scene.static.target)
    state = initial_conditional_state(model, scene)
    total = 0.0
    actions: List[str] = []
    history_a: List[str] = []
    public_history: List[str] = []

    if warmup is not None:
        action, symbol = warmup
        rows, _cost, _aggregate = expand_actions(model, state)
        likelihood = per_theta_outcome(model, rows[action], action)[:, symbol]
        state = ConditionalState(rows[action], condition(state.belief, likelihood))
        # the warm-up's own feedback and public event are part of the observation history
        from ..grid.dynamics import execute as _execute
        from ..learning.data import _joint_for as _joint

        warm_scene, warm_result = _execute(scene, _joint(None, ACTION_IDS[action]))
        history_a.append(warm_result.feedback[0])
        public_history.append(warm_result.y_public)
        current = warm_scene
    else:
        current = scene

    for _ in range(horizon):
        if current.is_goal():
            break
        # link 4: the new scan corrects only the observable part of every row
        state = ConditionalState(
            correct_observables(
                state.rows, encode_state(model, current, history_a, public_history)
            ),
            state.belief,
        )
        chosen, _branches = plan_with_conditional_states(
            model, goal, state, depth=depth, beam=beam
        )
        action_id = ACTION_IDS[chosen]
        actions.append(action_id)
        successor, result = execute(current, _joint_for(None, action_id))
        total += float(result.cost)
        symbol = FEEDBACK_SYMBOLS.index(result.feedback[0])
        # links 1 and 2: advance each row under its own parameter, then reweight
        next_rows = expand(model, state, chosen)
        likelihood = per_theta_outcome(model, next_rows, chosen)[:, symbol]
        state = ConditionalState(next_rows, condition(state.belief, likelihood))
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        current = successor
    return _record(scene_id, current.is_goal(), total, actions, reachable)


def summarise_records(records: Sequence[Dict[str, object]], fail_cost: float = FAIL_COST) -> Dict[str, object]:
    n = len(records)
    if n == 0:
        return {"n": 0}
    successes = sum(record["success"] for record in records)
    total_costs = [
        record["action_cost"] + (0.0 if record["success"] else fail_cost)
        for record in records
    ]
    successful = [
        record["action_cost"] + (0.0 if record["success"] else fail_cost)
        for record in records
        if record["success"]
    ]
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n,
        "mean_total_cost_all": float(np.mean(total_costs)),
        "mean_total_cost_success": (float(np.mean(successful)) if successes else None),
        "mean_steps": float(np.mean([record["steps"] for record in records])),
        "episodes_with_wait_first": sum(
            1 for record in records if record["first_action"] == "wait"
        ),
        "episodes_with_any_wait": sum(
            1 for record in records if record["wait_actions"] > 0
        ),
        "total_wait_actions": sum(record["wait_actions"] for record in records),
    }


def compare_records(
    old: Sequence[Dict[str, object]], new: Sequence[Dict[str, object]]
) -> Dict[str, object]:
    """Per-episode flips on the same scene ids, plus the wait change."""

    old_by_id = {record["scene_id"]: record for record in old}
    new_by_id = {record["scene_id"]: record for record in new}
    shared = sorted(set(old_by_id) & set(new_by_id))
    improved = [
        {
            "scene_id": scene_id,
            "old_actions": old_by_id[scene_id]["actions"],
            "new_actions": new_by_id[scene_id]["actions"],
            "old_cost": old_by_id[scene_id]["action_cost"],
            "new_cost": new_by_id[scene_id]["action_cost"],
        }
        for scene_id in shared
        if not old_by_id[scene_id]["success"] and new_by_id[scene_id]["success"]
    ]
    regressed = [
        {
            "scene_id": scene_id,
            "old_actions": old_by_id[scene_id]["actions"],
            "new_actions": new_by_id[scene_id]["actions"],
            "old_cost": old_by_id[scene_id]["action_cost"],
            "new_cost": new_by_id[scene_id]["action_cost"],
        }
        for scene_id in shared
        if old_by_id[scene_id]["success"] and not new_by_id[scene_id]["success"]
    ]
    return {
        "shared_scenes": len(shared),
        "improved_failure_to_success": improved,
        "regressed_success_to_failure": regressed,
        "wait_first_old": sum(1 for i in shared if old_by_id[i]["first_action"] == "wait"),
        "wait_first_new": sum(1 for i in shared if new_by_id[i]["first_action"] == "wait"),
        "total_wait_actions_old": sum(old_by_id[i]["wait_actions"] for i in shared),
        "total_wait_actions_new": sum(new_by_id[i]["wait_actions"] for i in shared),
    }


# ---------------------------------------------------------------------------
# Short-chain check and the closed-loop comparison
# ---------------------------------------------------------------------------


def conditional_chain_check(model, chain, *, tolerance: float = 0.02) -> Dict[str, object]:
    """Confirm the association improvement survives in the new pipeline (links 1 and 3).

    Fixed actions, no feedback conditioning, uniform prior: the same setting as the exact
    per-world reference, so the comparison isolates what is carried through the expansion.
    The four-link update is exercised by the closed loop below.
    """

    from ..learning.data import probe_scene
    from .state_association import (
        ExactWorldChain,
        LearnedChain,
        _encode_true_state,
        _initial_worlds,
        _summarise_exact,
        _summarise_learned,
        _tv,
    )

    scene = probe_scene(chain.origin, chain.physics)
    target = frozenset(scene.static.target)
    reference = ExactWorldChain(scene.static, _initial_worlds(scene))
    uniform = np.full(N_PHYSICS, 1.0 / N_PHYSICS)
    shipped = LearnedChain(model, _encode_true_state(model, scene), uniform)
    state = initial_conditional_state(model, scene)

    def released_gap(by_theta) -> float:
        reference_conditional = np.asarray(
            _summarise_exact(reference.worlds, target, 0, "").released_by_theta
        )
        return float(np.max(np.abs(np.asarray(by_theta) - reference_conditional)))

    steps: List[Dict[str, object]] = [
        {
            "step": 0,
            "action": None,
            "reference_position": _summarise_exact(reference.worlds, target, 0, "").position,
            "shipped_tv": 0.0,
            "conditional_tv": 0.0,
            "shipped_released_gap": released_gap(
                _summarise_learned(
                    shipped.current, shipped.belief, target, 0, "", per_theta=False
                ).released_by_theta
            ),
            "conditional_released_gap": released_gap(state.released_by_theta()),
        }
    ]
    for step, action_id in enumerate(chain.actions, start=1):
        action = ACTION_IDS.index(action_id)
        reference.step_exact(action_id)
        shipped.step_current(action)
        rows = expand(model, state, action)
        state = ConditionalState(rows, state.belief)

        reference_position = np.asarray(
            _summarise_exact(reference.worlds, target, step, action_id).position
        )
        reference_conditional = np.asarray(
            _summarise_exact(reference.worlds, target, step, action_id).released_by_theta
        )
        shipped_conditional = np.asarray(
            _summarise_learned(
                shipped.current, shipped.belief, target, step, action_id, per_theta=False
            ).released_by_theta
        )
        conditional_conditional = np.asarray(state.released_by_theta())
        steps.append(
            {
                "step": step,
                "action": action_id,
                "reference_position": [float(v) for v in reference_position],
                "shipped_tv": _tv(shipped.current[0, 1 : 1 + 36], reference_position),
                "conditional_tv": _tv(state.position(), reference_position),
                "shipped_released_gap": float(
                    np.max(np.abs(shipped_conditional - reference_conditional))
                ),
                "conditional_released_gap": float(
                    np.max(np.abs(conditional_conditional - reference_conditional))
                ),
            }
        )
    return {"chain": chain.name, "actions": list(chain.actions), "steps": steps}


def run_closed_loop_comparison(
    model,
    *,
    n_scenes: int = 100,
    horizon: int = 8,
    depths: Sequence[int] = (1, 2, 3),
    beam: int = 1,
    seed: int = 55_000,
    warmup: str = "probe:0:E",
) -> Dict[str, object]:
    """Same checkpoint, same scenes, same depth and beam; only the propagation differs."""

    from .diagnosis import _measurement_starts, exact_reachability

    starts, pre_solved, warm_solved = _measurement_starts(
        n_scenes=n_scenes, seed=seed, warmup=warmup
    )
    prepared = []
    for entry in starts:
        scene = entry["scene"]
        prepared.append(
            {
                "scene_id": entry["scene_id"],
                "scene": scene,
                "warmup": entry["warmup"],
                "reachable": exact_reachability(scene, horizon=horizon),
            }
        )
    results: List[Dict[str, object]] = []
    for depth in depths:
        old: List[Dict[str, object]] = []
        new: List[Dict[str, object]] = []
        for entry in prepared:
            old.append(
                run_shipped_episode(
                    model, entry["scene"], scene_id=entry["scene_id"],
                    horizon=horizon, depth=depth, beam=beam,
                    warmup=entry["warmup"], reachable=entry["reachable"],
                )
            )
            new.append(
                run_conditional_episode(
                    model, entry["scene"], scene_id=entry["scene_id"],
                    horizon=horizon, depth=depth, beam=beam,
                    warmup=entry["warmup"], reachable=entry["reachable"],
                )
            )
        reachable_ids = {entry["scene_id"] for entry in prepared if entry["reachable"] is not None}
        old_reachable = [record for record in old if record["scene_id"] in reachable_ids]
        new_reachable = [record for record in new if record["scene_id"] in reachable_ids]
        results.append(
            {
                "depth": depth,
                "beam": beam,
                "old_all": summarise_records(old),
                "new_all": summarise_records(new),
                "old_exact_reachable": summarise_records(old_reachable),
                "new_exact_reachable": summarise_records(new_reachable),
                "flips": compare_records(old, new),
                "old_records": old,
                "new_records": new,
            }
        )
    return {
        "n_scenes": n_scenes,
        "horizon": horizon,
        "beam": beam,
        "seed": seed,
        "warmup": warmup,
        "excluded_already_solved": pre_solved,
        "excluded_warmup_solved": warm_solved,
        "n_measured": len(prepared),
        "n_exact_reachable": sum(1 for entry in prepared if entry["reachable"] is not None),
        "depths": results,
    }


def boundary_case_study(
    model,
    scene_id: int,
    *,
    depth: int,
    seed: int = 55_000,
    warmup: str = "probe:0:E",
    top: int = 4,
) -> Dict[str, object]:
    """The first decision of one scene under both propagation schemes.

    Shows the whole ranking, not just the chosen action, so a flip can be attributed to a
    score that moved rather than to a tie-break.
    """

    from .belief_probe import fuse_hidden_state, plan_with_search
    from .diagnosis import _measurement_starts

    starts, _pre, _warm = _measurement_starts(n_scenes=100, seed=seed, warmup=warmup)
    entry = next(item for item in starts if item["scene_id"] == scene_id)
    scene = entry["scene"]
    goal = frozenset(scene.static.target)

    # shipped: advance the single state and belief through the warm-up
    warm_action, warm_symbol = entry["warmup"]
    observation = encode_state(model, scene)
    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    belief = model.update_belief(belief, observation, warm_action, warm_symbol)
    carried = model.step(observation, warm_action, belief)
    warm_scene, warm_result = execute(scene, _joint_for(None, ACTION_IDS[warm_action]))
    history_a = [warm_result.feedback[0]]
    public_history = [warm_result.y_public]
    observation = encode_state(model, warm_scene, history_a, public_history)
    dynamic = fuse_hidden_state(observation, carried)
    old_chosen, old_branches = plan_with_search(
        model, observation, goal, belief=belief, dynamic=dynamic, depth=depth, beam=1
    )

    # conditional: the same warm-up through the four links
    state = initial_conditional_state(model, scene)
    rows, _cost, _aggregate = expand_actions(model, state)
    likelihood = per_theta_outcome(model, rows[warm_action], warm_action)[:, warm_symbol]
    state = ConditionalState(rows[warm_action], condition(state.belief, likelihood))
    state = ConditionalState(
        correct_observables(
            state.rows, encode_state(model, warm_scene, history_a, public_history)
        ),
        state.belief,
    )
    new_chosen, new_branches = plan_with_conditional_states(
        model, goal, state, depth=depth, beam=1
    )

    def rank(branches):
        ordered = sorted(branches, key=lambda branch: (branch["score"], branch["action"]))
        return [
            {
                "action": branch["action"],
                "score": round(float(branch["score"]), 4),
                "first_cost": round(float(branch["first_cost"]), 4),
                "expected_distance": round(float(branch["expected_distance"]), 4),
                "released_probability": round(float(branch["released_probability"]), 4),
            }
            for branch in ordered[:top]
        ]

    return {
        "scene_id": scene_id,
        "depth": depth,
        "physics": entry["physics"].label,
        "origin": list(scene.obj.origin),
        "old_chosen": ACTION_IDS[old_chosen],
        "old_ranking": rank(old_branches),
        "new_chosen": ACTION_IDS[new_chosen],
        "new_ranking": rank(new_branches),
    }


def trace_boundaries(
    model,
    scene_id: int,
    *,
    depth: int,
    boundaries: int = 2,
    seed: int = 55_000,
    warmup: str = "probe:0:E",
) -> Dict[str, object]:
    """Per-boundary branch rankings under both schemes, on one scene.

    Replays the same decisions both schemes actually make, so a divergence can be read off at
    the boundary where it happens (for example the *continuation* after a successful push)
    instead of only at the first decision.
    """

    from .belief_probe import fuse_hidden_state, plan_with_search
    from .diagnosis import _measurement_starts

    starts, _pre, _warm = _measurement_starts(n_scenes=100, seed=seed, warmup=warmup)
    entry = next(item for item in starts if item["scene_id"] == scene_id)
    scene = entry["scene"]
    goal = frozenset(scene.static.target)
    warm_action, warm_symbol = entry["warmup"]

    def rank(branches, top=3):
        ordered = sorted(branches, key=lambda b: (b["score"], b["action"]))
        return [
            {
                "action": branch["action"],
                "score": round(float(branch["score"]), 4),
                "first_cost": round(float(branch["first_cost"]), 4),
                "expected_distance": round(float(branch["expected_distance"]), 4),
            }
            for branch in ordered[:top]
        ]

    # shipped
    observation = encode_state(model, scene)
    belief = np.full((1, N_PHYSICS), 1.0 / N_PHYSICS)
    belief = model.update_belief(belief, observation, warm_action, warm_symbol)
    carried = model.step(observation, warm_action, belief)
    current, result = execute(scene, _joint_for(None, ACTION_IDS[warm_action]))
    history_a = [result.feedback[0]]
    public_history = [result.y_public]
    old_trace = []
    for boundary in range(boundaries):
        observation = encode_state(model, current, history_a, public_history)
        dynamic = fuse_hidden_state(observation, carried)
        chosen, branches = plan_with_search(
            model, observation, goal, belief=belief, dynamic=dynamic, depth=depth, beam=1
        )
        old_trace.append({"boundary": boundary + 1, "chosen": ACTION_IDS[chosen], "ranking": rank(branches)})
        successor, result = execute(current, _joint_for(None, ACTION_IDS[chosen]))
        belief = model.update_belief(belief, dynamic, chosen, FEEDBACK_SYMBOLS.index(result.feedback[0]))
        carried = model.step(dynamic, chosen, belief)
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        current = successor
        if current.is_goal():
            break

    # conditional
    state = initial_conditional_state(model, scene)
    rows, _cost, _aggregate = expand_actions(model, state)
    likelihood = per_theta_outcome(model, rows[warm_action], warm_action)[:, warm_symbol]
    state = ConditionalState(rows[warm_action], condition(state.belief, likelihood))
    current = scene
    current, result = execute(scene, _joint_for(None, ACTION_IDS[warm_action]))
    history_a = [result.feedback[0]]
    public_history = [result.y_public]
    new_trace = []
    for boundary in range(boundaries):
        state = ConditionalState(
            correct_observables(
                state.rows, encode_state(model, current, history_a, public_history)
            ),
            state.belief,
        )
        chosen, branches = plan_with_conditional_states(model, goal, state, depth=depth, beam=1)
        new_trace.append({"boundary": boundary + 1, "chosen": ACTION_IDS[chosen], "ranking": rank(branches)})
        successor, result = execute(current, _joint_for(None, ACTION_IDS[chosen]))
        next_rows = expand(model, state, chosen)
        likelihood = per_theta_outcome(model, next_rows, chosen)[:, FEEDBACK_SYMBOLS.index(result.feedback[0])]
        state = ConditionalState(next_rows, condition(state.belief, likelihood))
        history_a.append(result.feedback[0])
        public_history.append(result.y_public)
        current = successor
        if current.is_goal():
            break

    return {
        "scene_id": scene_id,
        "depth": depth,
        "physics": entry["physics"].label,
        "old": old_trace,
        "new": new_trace,
    }
