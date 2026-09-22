"""Exact Bayes filters for the micro instances.

Two objects are provided, matching ``docs/2d_protocol_spec.md`` section 5:

* :func:`robot_posterior` -- ``P(x, theta | h_i, C)``, what one robot can compute
  from its own history and the public record alone;
* the *public coordination belief* over :data:`PublicEntry`
  ``(x, theta, h_A, h_B)`` -- ``P(x, theta, h_A, h_B | C)``. This is the state a
  public commit rule needs, because a future message depends on the private history
  that produced it. State ``(x, theta)`` alone is *not* sufficient.

Two robots' posteriors are never multiplied. Where a joint quantity is required it
is read off the public coordination belief, which is built from the published joint
observation table.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

from ..toy.model import MicroModel

#: ``(x, theta, h_A, h_B)``; histories are tuples of local observations.
PublicEntry = Tuple[str, str, Tuple[str, ...], Tuple[str, ...]]
#: A normalised distribution over :data:`PublicEntry`.
PublicBelief = Dict[PublicEntry, float]
#: A normalised distribution over ``(x, theta)``.
StateBelief = Dict[Tuple[str, str], float]


@dataclass(frozen=True)
class RobotView:
    """Everything a message codec may legally read at one boundary.

    This is the deployment-visible input: own private observations, the public
    observation and action histories, and the public candidate list. It contains no
    ``TrueState`` and no teammate history.
    """

    robot_id: int
    observations: Tuple[str, ...]
    y_public: Tuple[str, ...]
    actions: Tuple[str, ...]
    candidates: Tuple[str, ...]

    def key(self) -> Tuple[object, ...]:
        return (
            self.robot_id,
            self.observations,
            self.y_public,
            self.actions,
            self.candidates,
        )


def normalize(weights: Mapping[object, float]) -> Dict[object, float]:
    """Normalise non-negative weights, dropping zeros. Empty stays empty."""

    total = sum(weights.values())
    if total <= 0.0:
        return {}
    return {k: v / total for k, v in weights.items() if v > 0.0}


# --------------------------------------------------------------------------
# Single-robot belief
# --------------------------------------------------------------------------


def robot_posterior(
    model: MicroModel,
    robot_id: int,
    observations: Tuple[str, ...],
    y_public: Tuple[str, ...],
    actions: Tuple[str, ...],
) -> StateBelief:
    """``P(x, theta | h_i, C)`` for one robot, exactly.

    ``actions`` holds the committed joint actions before this boundary, so
    ``len(actions) == len(observations) - 1 == len(y_public) - 1``.
    """

    if not observations:
        raise ValueError("A boundary always has at least the initial local observation.")
    if len(observations) != len(y_public):
        raise ValueError("Each boundary has exactly one local and one public observation.")
    if len(actions) != len(observations) - 1:
        raise ValueError("Committed-action history length does not match the boundary.")

    belief: StateBelief = {}
    o_0, y_0 = observations[0], y_public[0]
    for theta in model.theta_values:
        prob = model.initial_local_public_marginal(robot_id, theta).get((o_0, y_0), 0.0)
        if prob > 0.0:
            belief[(model.initial_x, theta)] = model.theta_prior[theta] * prob
    belief = normalize(belief)  # type: ignore[assignment]

    for t in range(1, len(observations)):
        action = actions[t - 1]
        o_t, y_t = observations[t], y_public[t]
        nxt: StateBelief = {}
        for (x, theta), w in belief.items():
            for x_next, tx in model.transition(x, theta, action).items():
                if tx <= 0.0:
                    continue
                prob = model.local_public_marginal(
                    robot_id, x, theta, action, x_next
                ).get((o_t, y_t), 0.0)
                if prob > 0.0:
                    key = (x_next, theta)
                    nxt[key] = nxt.get(key, 0.0) + w * tx * prob
        belief = normalize(nxt)  # type: ignore[assignment]
        if not belief:
            # An observation with zero likelihood under every surviving hypothesis.
            return {}
    return belief


def theta_posterior(belief: StateBelief) -> Dict[str, float]:
    """Marginal over the hidden cause."""

    out: Dict[str, float] = {}
    for (_x, theta), w in belief.items():
        out[theta] = out.get(theta, 0.0) + w
    return out


def goal_probability(model: MicroModel, belief: StateBelief) -> float:
    return sum(w for (x, _theta), w in belief.items() if model.is_goal(x))


# --------------------------------------------------------------------------
# Public coordination belief over (x, theta, h_A, h_B)
# --------------------------------------------------------------------------


def marginal_state(belief: PublicBelief) -> StateBelief:
    out: StateBelief = {}
    for (x, theta, _h_a, _h_b), w in belief.items():
        key = (x, theta)
        out[key] = out.get(key, 0.0) + w
    return out


def marginal_private(
    belief: PublicBelief, robot_id: int, history: Tuple[str, ...]
) -> StateBelief:
    """Robot ``robot_id``'s posterior conditional on its actual private history."""

    out: StateBelief = {}
    for (x, theta, h_a, h_b), w in belief.items():
        h = h_a if robot_id == 0 else h_b
        if h == history:
            key = (x, theta)
            out[key] = out.get(key, 0.0) + w
    return normalize(out)  # type: ignore[return-value]


def public_goal_probability(model: MicroModel, belief: PublicBelief) -> float:
    return sum(w for (x, _t, _a, _b), w in belief.items() if model.is_goal(x))


def public_expected_cost(model: MicroModel, belief: PublicBelief, action: str) -> float:
    return sum(w * model.cost(x, action) for (x, _t, _a, _b), w in belief.items())


def signature(belief: PublicBelief) -> Tuple[Tuple[PublicEntry, float], ...]:
    """A hashable canonical signature used for planner memoisation."""

    return tuple(sorted(belief.items()))


# --------------------------------------------------------------------------
# Initial beliefs
# --------------------------------------------------------------------------


def initial_public_beliefs(
    model: MicroModel,
    y_0: str,
    codec,
    candidates: Tuple[str, ...],
) -> Dict[Tuple[int, int], Tuple[PublicBelief, float]]:
    """Public beliefs at boundary 0, split by the message pair the codec produces.

    Returns ``{(m_A, m_B): (belief, probability)}``.
    """

    buckets: Dict[Tuple[int, int], Dict[PublicEntry, float]] = {}
    for theta in model.theta_values:
        prior = model.theta_prior[theta]
        for ((o_a, o_b), y), prob in model.initial_observation(theta).items():
            if y != y_0 or prob <= 0.0:
                continue
            weight = prior * prob
            view_a = RobotView(0, (o_a,), (y_0,), (), candidates)
            view_b = RobotView(1, (o_b,), (y_0,), (), candidates)
            key = (codec.encode(model, view_a), codec.encode(model, view_b))
            bucket = buckets.setdefault(key, {})
            entry: PublicEntry = (model.initial_x, theta, (o_a,), (o_b,))
            bucket[entry] = bucket.get(entry, 0.0) + weight

    out: Dict[Tuple[int, int], Tuple[PublicBelief, float]] = {}
    for key, weights in buckets.items():
        total = sum(weights.values())
        if total > 0.0:
            out[key] = (normalize(weights), total)  # type: ignore[arg-type]
    return out


def successor_public_beliefs(
    model: MicroModel,
    belief: PublicBelief,
    action: str,
    y_public: Tuple[str, ...],
    actions: Tuple[str, ...],
    messages: Tuple[Tuple[int, int], ...],
    codec,
    candidates: Tuple[str, ...],
) -> Dict[Tuple[str, int, int], Tuple[float, PublicBelief, Tuple[Tuple[int, int], ...]]]:
    """One-step lookahead of the public coordination belief.

    Returns ``{(y_next, m_A, m_B): (probability, next_belief, next_messages)}``.
    The next messages are produced by the same codec from the successor private
    histories, so a fixed codec is respected in planning.
    """

    buckets: Dict[Tuple[str, int, int], Dict[PublicEntry, float]] = {}
    for (x, theta, h_a, h_b), w in belief.items():
        for x_next, tx in model.transition(x, theta, action).items():
            if tx <= 0.0:
                continue
            for ((o_a2, o_b2), y2), po in model.observation(
                x, theta, action, x_next
            ).items():
                if po <= 0.0:
                    continue
                new_h_a = h_a + (o_a2,)
                new_h_b = h_b + (o_b2,)
                new_y = y_public + (y2,)
                new_actions = actions + (action,)
                view_a = RobotView(0, new_h_a, new_y, new_actions, candidates)
                view_b = RobotView(1, new_h_b, new_y, new_actions, candidates)
                key = (
                    y2,
                    codec.encode(model, view_a),
                    codec.encode(model, view_b),
                )
                entry: PublicEntry = (x_next, theta, new_h_a, new_h_b)
                bucket = buckets.setdefault(key, {})
                bucket[entry] = bucket.get(entry, 0.0) + w * tx * po

    out: Dict[
        Tuple[str, int, int], Tuple[float, PublicBelief, Tuple[Tuple[int, int], ...]]
    ] = {}
    for (y2, m_a, m_b), weights in buckets.items():
        total = sum(weights.values())
        if total > 0.0:
            out[(y2, m_a, m_b)] = (
                total,
                normalize(weights),  # type: ignore[arg-type]
                messages + ((m_a, m_b),),
            )
    return out


def initial_central_beliefs(
    model: MicroModel, y_0: str
) -> Dict[Tuple[str, str, str], Tuple[float, StateBelief]]:
    """Initial belief over ``(x, theta)`` for a planner that sees both local obs."""

    buckets: Dict[Tuple[str, str, str], Dict[Tuple[str, str], float]] = {}
    for theta in model.theta_values:
        prior = model.theta_prior[theta]
        for ((o_a, o_b), y), prob in model.initial_observation(theta).items():
            if y != y_0 or prob <= 0.0:
                continue
            key = (o_a, o_b, y)
            bucket = buckets.setdefault(key, {})
            entry = (model.initial_x, theta)
            bucket[entry] = bucket.get(entry, 0.0) + prior * prob

    out: Dict[Tuple[str, str, str], Tuple[float, StateBelief]] = {}
    for key, weights in buckets.items():
        total = sum(weights.values())
        if total > 0.0:
            out[key] = (total, normalize(weights))  # type: ignore[arg-type]
    return out


def successor_central_beliefs(
    model: MicroModel, belief: StateBelief, action: str
) -> Dict[Tuple[str, str, str], Tuple[float, StateBelief]]:
    """Joint-observation branching for a planner that sees both local observations."""

    buckets: Dict[Tuple[str, str, str], Dict[Tuple[str, str], float]] = {}
    for (x, theta), w in belief.items():
        for x_next, tx in model.transition(x, theta, action).items():
            if tx <= 0.0:
                continue
            for ((o_a2, o_b2), y2), po in model.observation(
                x, theta, action, x_next
            ).items():
                if po <= 0.0:
                    continue
                key = (o_a2, o_b2, y2)
                bucket = buckets.setdefault(key, {})
                entry = (x_next, theta)
                bucket[entry] = bucket.get(entry, 0.0) + w * tx * po

    out: Dict[Tuple[str, str, str], Tuple[float, StateBelief]] = {}
    for key, weights in buckets.items():
        total = sum(weights.values())
        if total > 0.0:
            out[key] = (total, normalize(weights))  # type: ignore[arg-type]
    return out
