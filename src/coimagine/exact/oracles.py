"""Exact oracles and one deliberately non-exact limited search.

Implements the hierarchy of ``docs/2d_protocol_spec.md`` section 7:

* :func:`o_full`            -- true state, exact backward induction;
* :func:`o_central`         -- both private observation histories, exact POMDP;
* :func:`o_codec`           -- fixed codec, exact planning over future message
  branches that pass through the same codec;
* :func:`o_b_exact`         -- exact over all message rules and public action rules
  at a **single** decision boundary (``H = 1``);
* :func:`true_model_limited_search` -- true model, depth/node limited, explicitly
  **not** optimal.

Every result reports the horizon, strategy class, whether the codec is fixed, the
node count, the wall-clock time and an ``exact`` flag. A budget that is exhausted
returns ``exact=False`` with an ``incomplete`` note rather than a pseudo-exact value.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..toy.model import MicroModel
from .codec import Codec, NullCodec, local_alphabet
from .filter import (
    PublicBelief,
    StateBelief,
    goal_probability,
    initial_central_beliefs,
    initial_public_beliefs,
    public_expected_cost,
    public_goal_probability,
    signature,
    successor_central_beliefs,
    successor_public_beliefs,
)


@dataclass(frozen=True)
class OracleResult:
    name: str
    instance: str
    horizon: int
    expected_cost: float
    success_prob: float
    strategy_class: str
    codec: str
    fixed_codec: bool
    bits: int
    nodes: int
    seconds: float
    exact: bool
    note: str = ""

    @property
    def incomplete(self) -> bool:
        return not self.exact


def _initial_y(model: MicroModel) -> str:
    values = {y for row in model.init_observations.values() for (_o, y) in row}
    if len(values) != 1:
        raise ValueError(
            "The micro instances are specified with a single initial public "
            f"observation; got {sorted(values)}."
        )
    return next(iter(values))


def _immediate_failure_value(model: MicroModel, x: str, theta: str, action: str) -> float:
    """Immediate expected cost of an action plus the failure tail."""

    cost = model.cost(x, action)
    tail = 0.0
    for x_next, prob in model.transition(x, theta, action).items():
        if not model.is_goal(x_next):
            tail += prob * model.fail_cost
    return cost + tail


# --------------------------------------------------------------------------
# O_full
# --------------------------------------------------------------------------


def o_full(model: MicroModel) -> OracleResult:
    """Full-information optimum: the true ``(x, theta)`` is known."""

    start = time.perf_counter()
    nodes = 0
    theta_values = model.theta_values

    # value[step][(x, theta)] and the argmin action for success roll-out
    value: Dict[Tuple[str, str], float] = {
        (x, theta): 0.0 if model.is_goal(x) else model.fail_cost
        for x in model.x_values
        for theta in theta_values
    }
    policy: Dict[Tuple[int, str, str], str] = {}
    for t in range(model.horizon - 1, -1, -1):
        current: Dict[Tuple[str, str], float] = {}
        for x in model.x_values:
            for theta in theta_values:
                if model.is_goal(x):
                    current[(x, theta)] = 0.0
                    continue
                best_value = float("inf")
                best_action = model.actions[0]
                for action in model.actions:
                    nodes += 1
                    total = model.cost(x, action)
                    for x_next, prob in model.transition(x, theta, action).items():
                        total += prob * value[(x_next, theta)]
                    if total < best_value - 1e-12:
                        best_value = total
                        best_action = action
                current[(x, theta)] = best_value
                policy[(t, x, theta)] = best_action
        value = current

    expected_cost = sum(
        model.theta_prior[theta] * value[(model.initial_x, theta)]
        for theta in theta_values
    )

    # success probability under the optimal policy, per hidden cause
    success = 0.0
    for theta in theta_values:
        dist: Dict[str, float] = {model.initial_x: 1.0}
        reached = 0.0
        for t in range(model.horizon):
            nxt: Dict[str, float] = {}
            for x, w in dist.items():
                if model.is_goal(x):
                    reached += w
                    continue
                action = policy[(t, x, theta)]
                for x_next, prob in model.transition(x, theta, action).items():
                    nxt[x_next] = nxt.get(x_next, 0.0) + w * prob
            dist = nxt
        reached += sum(w for x, w in dist.items() if model.is_goal(x))
        success += model.theta_prior[theta] * min(reached, 1.0)

    return OracleResult(
        name="O_full",
        instance=model.name,
        horizon=model.horizon,
        expected_cost=expected_cost,
        success_prob=success,
        strategy_class="full-information finite-horizon optimum",
        codec="n/a",
        fixed_codec=False,
        bits=-1,
        nodes=nodes,
        seconds=time.perf_counter() - start,
        exact=True,
    )


# --------------------------------------------------------------------------
# O_central
# --------------------------------------------------------------------------


def _central_dp(model: MicroModel):
    """Exact POMDP dynamic program over beliefs about ``(x, theta)``.

    Returns ``(value_fn, policy, nodes)`` where ``policy[(belief_key, t)]`` is the
    chosen joint action. Shared by :func:`o_central` and
    :func:`o_central_root_actions` so the reported value and the inspected policy
    can never disagree.
    """

    nodes = [0]
    cost_memo: Dict[Tuple[object, ...], float] = {}
    success_memo: Dict[Tuple[object, ...], float] = {}
    policy: Dict[Tuple[object, ...], str] = {}

    def key(belief: StateBelief, t: int) -> Tuple[object, ...]:
        return (tuple(sorted(belief.items())), t)

    def value(belief: StateBelief, t: int) -> float:
        k = key(belief, t)
        if k in cost_memo:
            return cost_memo[k]
        nodes[0] += 1
        if t == model.horizon:
            result = model.fail_cost * (1.0 - goal_probability(model, belief))
            cost_memo[k] = result
            return result
        best_value = float("inf")
        best_action = model.actions[0]
        for action in model.actions:
            total = sum(w * model.cost(x, action) for (x, _th), w in belief.items())
            for _obs, (prob, nxt) in successor_central_beliefs(
                model, belief, action
            ).items():
                total += prob * value(nxt, t + 1)
            if total < best_value - 1e-12:
                best_value = total
                best_action = action
        cost_memo[k] = best_value
        policy[k] = best_action
        return best_value

    def success(belief: StateBelief, t: int) -> float:
        k = key(belief, t)
        if k in success_memo:
            return success_memo[k]
        if t == model.horizon:
            result = goal_probability(model, belief)
            success_memo[k] = result
            return result
        action = policy.get(k, model.actions[0])
        result = 0.0
        for _obs, (prob, nxt) in successor_central_beliefs(model, belief, action).items():
            result += prob * success(nxt, t + 1)
        success_memo[k] = result
        return result

    return value, success, policy, nodes


def o_central(model: MicroModel) -> OracleResult:
    """Exact POMDP optimum for a planner that sees both robots' observations."""

    start = time.perf_counter()
    y0 = _initial_y(model)
    value, success, _policy, nodes = _central_dp(model)

    expected_cost = 0.0
    success_prob = 0.0
    for _obs, (prob, belief) in initial_central_beliefs(model, y0).items():
        expected_cost += prob * value(belief, 0)
        success_prob += prob * success(belief, 0)

    return OracleResult(
        name="O_central",
        instance=model.name,
        horizon=model.horizon,
        expected_cost=expected_cost,
        success_prob=success_prob,
        strategy_class="exact POMDP over both private observation histories",
        codec="n/a (full observation sharing)",
        fixed_codec=False,
        bits=-1,
        nodes=nodes[0],
        seconds=time.perf_counter() - start,
        exact=True,
    )


# --------------------------------------------------------------------------
# Inspectable policies (for protocol / probe-choice tests)
# --------------------------------------------------------------------------


def best_immediate_action(model: MicroModel, belief: StateBelief) -> str:
    """Argmin of the expected immediate cost plus the failure tail.

    A one-step decision rule used by tests to show that identical public input with
    different private evidence can select different joint actions.
    """

    best_action = model.actions[0]
    best_value = float("inf")
    for action in model.actions:
        value = 0.0
        for (x, theta), w in belief.items():
            value += w * _immediate_failure_value(model, x, theta, action)
        if value < best_value - 1e-12:
            best_value = value
            best_action = action
    return best_action


def o_central_root_actions(model: MicroModel) -> Dict[Tuple[str, str, str], str]:
    """The exact O_central policy's action at the first boundary, per initial obs.

    Used to check whether an informative probe is chosen by the solver (rather than
    a hard-coded rule) and to build public-same / private-different pairs. The
    values come from the same dynamic program that produces :func:`o_central`.
    """

    y0 = _initial_y(model)
    value, _success, policy, _nodes = _central_dp(model)
    out: Dict[Tuple[str, str, str], str] = {}
    for obs, (_prob, belief) in initial_central_beliefs(model, y0).items():
        key = (tuple(sorted(belief.items())), 0)
        if key not in policy:
            value(belief, 0)
        out[obs] = policy.get(key, model.actions[0])
    return out


# --------------------------------------------------------------------------
# O_codec
# --------------------------------------------------------------------------


def o_codec(model: MicroModel, codec: Codec) -> OracleResult:
    """Exact optimum for a **fixed** codec, including future message branches."""

    start = time.perf_counter()
    nodes = [0]
    y0 = _initial_y(model)
    candidates = model.actions
    cost_memo: Dict[Tuple[object, ...], float] = {}
    success_memo: Dict[Tuple[object, ...], float] = {}
    policy: Dict[Tuple[object, ...], str] = {}

    def key(
        belief: PublicBelief, y_public: Tuple[str, ...], actions: Tuple[str, ...], t: int
    ) -> Tuple[object, ...]:
        return (signature(belief), y_public, actions, t)

    def value(
        belief: PublicBelief,
        y_public: Tuple[str, ...],
        actions: Tuple[str, ...],
        messages: Tuple[Tuple[int, int], ...],
        t: int,
    ) -> float:
        k = key(belief, y_public, actions, t)
        if k in cost_memo:
            return cost_memo[k]
        nodes[0] += 1
        if t == model.horizon:
            result = model.fail_cost * (1.0 - public_goal_probability(model, belief))
            cost_memo[k] = result
            return result
        best_value = float("inf")
        best_action = model.actions[0]
        for action in model.actions:
            total = public_expected_cost(model, belief, action)
            for _obs, (prob, nxt, _msgs) in successor_public_beliefs(
                model, belief, action, y_public, actions, messages, codec, candidates
            ).items():
                total += prob * value(
                    nxt,
                    y_public + (_obs[0],),
                    actions + (action,),
                    messages + ((_obs[1], _obs[2]),),
                    t + 1,
                )
            if total < best_value - 1e-12:
                best_value = total
                best_action = action
        cost_memo[k] = best_value
        policy[k] = best_action
        return best_value

    def success(
        belief: PublicBelief,
        y_public: Tuple[str, ...],
        actions: Tuple[str, ...],
        messages: Tuple[Tuple[int, int], ...],
        t: int,
    ) -> float:
        k = key(belief, y_public, actions, t)
        if k in success_memo:
            return success_memo[k]
        if t == model.horizon:
            result = public_goal_probability(model, belief)
            success_memo[k] = result
            return result
        action = policy.get(k, model.actions[0])
        result = 0.0
        for _obs, (prob, nxt, _msgs) in successor_public_beliefs(
            model, belief, action, y_public, actions, messages, codec, candidates
        ).items():
            result += prob * success(
                nxt,
                y_public + (_obs[0],),
                actions + (action,),
                messages + ((_obs[1], _obs[2]),),
                t + 1,
            )
        success_memo[k] = result
        return result

    expected_cost = 0.0
    success_prob = 0.0
    for (_m_a, _m_b), (belief, prob) in initial_public_beliefs(
        model, y0, codec, candidates
    ).items():
        expected_cost += prob * value(belief, (y0,), (), (), 0)
        success_prob += prob * success(belief, (y0,), (), (), 0)

    return OracleResult(
        name="O_codec",
        instance=model.name,
        horizon=model.horizon,
        expected_cost=expected_cost,
        success_prob=success_prob,
        strategy_class="exact public policy for a fixed codec, future messages modelled",
        codec=codec.name,
        fixed_codec=True,
        bits=codec.bits,
        nodes=nodes[0],
        seconds=time.perf_counter() - start,
        exact=True,
    )


# --------------------------------------------------------------------------
# O_B -- exact only at a single decision boundary
# --------------------------------------------------------------------------


def _message_rules(bits: int, domain: Tuple[Tuple[str, str], ...]) -> List[Dict[Tuple[str, str], int]]:
    """All deterministic rules ``domain -> {0..2**bits-1}``."""

    symbols = range(1 << bits)
    return [dict(zip(domain, combo)) for combo in itertools.product(symbols, repeat=len(domain))]


def o_b_exact(model: MicroModel, bits: int, *, node_budget: int = 5_000_000) -> OracleResult:
    """Exact ``O_B`` over the single-boundary protocol class.

    The strategy class is: each robot maps its boundary-0 information
    ``(o_i, y_0)`` to one of ``2**B`` symbols; the public rule maps
    ``(y_0, m_A, m_B)`` to a joint action. Both robots send exactly one message and
    the commit rule is public. This is **not** the unrestricted Dec-POMDP optimum.

    Only defined for ``H = 1``. Multi-step exact ``O_B`` is a Dec-POMDP and is
    reported as incomplete rather than approximated silently.
    """

    start = time.perf_counter()
    if model.horizon != 1:
        return OracleResult(
            name="O_B",
            instance=model.name,
            horizon=model.horizon,
            expected_cost=float("nan"),
            success_prob=float("nan"),
            strategy_class="single-boundary exact O_B (H=1 only)",
            codec="enumerated",
            fixed_codec=False,
            bits=bits,
            nodes=0,
            seconds=0.0,
            exact=False,
            note=(
                "multi-step exact O_B requires solving a finite-horizon Dec-POMDP "
                "over message rules and public action rules; outside the first-round "
                "budget, reported incomplete"
            ),
        )

    y0 = _initial_y(model)
    alphabet = local_alphabet(model)
    domain = tuple((o, y0) for o in alphabet)
    rules = _message_rules(bits, domain)

    nodes = 0
    best_value = float("inf")
    best_success = 0.0
    for rule_a in rules:
        for rule_b in rules:
            nodes += 1
            if nodes > node_budget:
                return OracleResult(
                    name="O_B",
                    instance=model.name,
                    horizon=1,
                    expected_cost=float("nan"),
                    success_prob=float("nan"),
                    strategy_class="single-boundary exact O_B (H=1 only)",
                    codec="enumerated",
                    fixed_codec=False,
                    bits=bits,
                    nodes=nodes,
                    seconds=time.perf_counter() - start,
                    exact=False,
                    note=f"node budget {node_budget} exhausted during enumeration",
                )
            # public info (y_0, m_A, m_B) -> posterior weights over theta
            info: Dict[Tuple[str, int, int], Dict[str, float]] = {}
            for theta in model.theta_values:
                prior = model.theta_prior[theta]
                for ((o_a, o_b), y), prob in model.initial_observation(theta).items():
                    if y != y0 or prob <= 0.0:
                        continue
                    key = (y0, rule_a[(o_a, y0)], rule_b[(o_b, y0)])
                    bucket = info.setdefault(key, {})
                    bucket[theta] = bucket.get(theta, 0.0) + prior * prob

            total_value = 0.0
            total_success = 0.0
            for key, theta_weights in info.items():
                total = sum(theta_weights.values())
                if total <= 0.0:
                    continue
                best_action_value = float("inf")
                best_action_success = 0.0
                for action in model.actions:
                    value = 0.0
                    success = 0.0
                    for theta, w in theta_weights.items():
                        row = model.transition(model.initial_x, theta, action)
                        value += w * model.cost(model.initial_x, action)
                        for x_next, prob in row.items():
                            if model.is_goal(x_next):
                                success += w * prob
                            else:
                                value += w * prob * model.fail_cost
                    if value < best_action_value - 1e-12:
                        best_action_value = value
                        best_action_success = success
                total_value += best_action_value
                total_success += best_action_success
            if total_value < best_value - 1e-12:
                best_value = total_value
                best_success = total_success

    return OracleResult(
        name="O_B",
        instance=model.name,
        horizon=1,
        expected_cost=best_value,
        success_prob=best_success,
        strategy_class=(
            "single-boundary exact: enumerate (mu_A, mu_B) and the optimal public rule"
        ),
        codec="enumerated message rules",
        fixed_codec=False,
        bits=bits,
        nodes=nodes,
        seconds=time.perf_counter() - start,
        exact=True,
        note=f"{len(rules)} message rules per robot; class is one message round then public commit",
    )


# --------------------------------------------------------------------------
# True-model limited search (NOT an oracle)
# --------------------------------------------------------------------------


def true_model_limited_search(
    model: MicroModel,
    codec: Optional[Codec] = None,
    *,
    depth_limit: int = 1,
    node_budget: int = 200_000,
) -> OracleResult:
    """Depth-limited expectimax under the true model.

    This is the "no future-information-branch" baseline: within ``depth_limit``
    steps it branches over observations, but beyond that it falls back to the
    failure cost. It is explicitly **not** optimal and is never called an oracle.
    """

    start = time.perf_counter()
    codec = codec or NullCodec()
    y0 = _initial_y(model)
    candidates = model.actions
    nodes = [0]
    incomplete = [False]
    memo: Dict[Tuple[object, ...], float] = {}

    def value(
        belief: PublicBelief,
        y_public: Tuple[str, ...],
        actions: Tuple[str, ...],
        messages: Tuple[Tuple[int, int], ...],
        remaining: int,
    ) -> float:
        k = (signature(belief), y_public, actions, remaining)
        if k in memo:
            return memo[k]
        nodes[0] += 1
        if nodes[0] > node_budget:
            incomplete[0] = True
            return model.fail_cost
        if remaining == 0:
            result = model.fail_cost * (1.0 - public_goal_probability(model, belief))
            memo[k] = result
            return result
        best_value = float("inf")
        for action in model.actions:
            total = public_expected_cost(model, belief, action)
            for _obs, (prob, nxt, _msgs) in successor_public_beliefs(
                model, belief, action, y_public, actions, messages, codec, candidates
            ).items():
                total += prob * value(
                    nxt,
                    y_public + (_obs[0],),
                    actions + (action,),
                    messages + ((_obs[1], _obs[2]),),
                    remaining - 1,
                )
            if total < best_value - 1e-12:
                best_value = total
        memo[k] = best_value
        return best_value

    expected_cost = 0.0
    for (_m_a, _m_b), (belief, prob) in initial_public_beliefs(
        model, y0, codec, candidates
    ).items():
        expected_cost += prob * value(belief, (y0,), (), (), depth_limit)

    return OracleResult(
        name="TrueModelLimitedSearch",
        instance=model.name,
        horizon=model.horizon,
        expected_cost=expected_cost,
        success_prob=float("nan"),
        strategy_class=f"depth-limited expectimax (depth {depth_limit}), receding horizon",
        codec=codec.name,
        fixed_codec=True,
        bits=codec.bits,
        nodes=nodes[0],
        seconds=time.perf_counter() - start,
        exact=False,
        note=(
            "budget exhausted; value is a lower-bound-ish estimate and is NOT exact"
            if incomplete[0]
            else "not optimal by construction; reported as a computation-limit diagnostic"
        ),
    )
