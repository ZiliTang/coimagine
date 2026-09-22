"""Finite tabular generative model for the micro diagnostic instances.

The model is the single source of truth for dynamics and feedback. It publishes

* the transition table ``T(x' | x, theta, a)`` with ``theta`` stationary;
* the **complete joint observation table**
  ``O((o_A, o_B, y_pub) | x, theta, a, x')``;
* a cost table ``cost(x, a)``;
* the initial observation table ``O_init((o_A, o_B, y_pub) | theta)``.

No component may add an information-gain or enabling term on top of this. The
model is queryable by every planner.

Observation keys are ``((o_A, o_B), y_pub)``. ``y_pub`` is stored inside the joint
table on purpose so that all methods consume one object and correlated local
observations are handled without an independence assumption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import FrozenSet, Mapping, Tuple

JointObservation = Tuple[Tuple[str, str], str]


@dataclass(frozen=True)
class MicroModel:
    """An explicit, finite, tabular two-robot POMDP."""

    name: str
    x_values: Tuple[str, ...]
    theta_values: Tuple[str, ...]
    actions: Tuple[str, ...]
    horizon: int
    initial_x: str
    goal_x: str
    theta_prior: Mapping[str, float]
    transitions: Mapping[Tuple[str, str, str], Mapping[str, float]]
    observations: Mapping[Tuple[str, str, str, str], Mapping[JointObservation, float]]
    init_observations: Mapping[str, Mapping[JointObservation, float]]
    costs: Mapping[Tuple[str, str], float]
    fail_cost: float
    probe_actions: FrozenSet[str] = frozenset()
    description: str = ""
    metadata: Mapping[str, object] = field(default_factory=dict)

    # -- queries -------------------------------------------------------------

    def transition(self, x: str, theta: str, action: str) -> Mapping[str, float]:
        return self.transitions[(x, theta, action)]

    def observation(
        self, x: str, theta: str, action: str, x_next: str
    ) -> Mapping[JointObservation, float]:
        return self.observations[(x, theta, action, x_next)]

    def initial_observation(self, theta: str) -> Mapping[JointObservation, float]:
        return self.init_observations[theta]

    def cost(self, x: str, action: str) -> float:
        return self.costs[(x, action)]

    def is_goal(self, x: str) -> bool:
        return x == self.goal_x

    def is_probe(self, action: str) -> bool:
        return action in self.probe_actions

    # -- marginals -----------------------------------------------------------

    def local_marginal(
        self, robot_id: int, x: str, theta: str, action: str, x_next: str
    ) -> Mapping[str, float]:
        """Marginal distribution of robot ``robot_id``'s own observation only."""

        out: dict[str, float] = {}
        for ((o_a, o_b), _y), prob in self.observation(
            x, theta, action, x_next
        ).items():
            value = o_a if robot_id == 0 else o_b
            out[value] = out.get(value, 0.0) + prob
        return out

    def local_public_marginal(
        self, robot_id: int, x: str, theta: str, action: str, x_next: str
    ) -> Mapping[Tuple[str, str], float]:
        """Marginal of ``(o_i, y_pub)`` — what robot ``i`` actually observes."""

        out: dict[Tuple[str, str], float] = {}
        for ((o_a, o_b), y), prob in self.observation(
            x, theta, action, x_next
        ).items():
            value = o_a if robot_id == 0 else o_b
            key = (value, y)
            out[key] = out.get(key, 0.0) + prob
        return out

    def public_marginal(
        self, x: str, theta: str, action: str, x_next: str
    ) -> Mapping[str, float]:
        out: dict[str, float] = {}
        for ((_o_a, _o_b), y), prob in self.observation(
            x, theta, action, x_next
        ).items():
            out[y] = out.get(y, 0.0) + prob
        return out

    def joint_local_marginal(
        self, x: str, theta: str, action: str, x_next: str
    ) -> Mapping[Tuple[str, str], float]:
        out: dict[Tuple[str, str], float] = {}
        for ((o_a, o_b), _y), prob in self.observation(
            x, theta, action, x_next
        ).items():
            key = (o_a, o_b)
            out[key] = out.get(key, 0.0) + prob
        return out

    def initial_local_public_marginal(
        self, robot_id: int, theta: str
    ) -> Mapping[Tuple[str, str], float]:
        out: dict[Tuple[str, str], float] = {}
        for ((o_a, o_b), y), prob in self.initial_observation(theta).items():
            value = o_a if robot_id == 0 else o_b
            key = (value, y)
            out[key] = out.get(key, 0.0) + prob
        return out

    def initial_joint_local_marginal(self, theta: str) -> Mapping[Tuple[str, str], float]:
        out: dict[Tuple[str, str], float] = {}
        for ((o_a, o_b), _y), prob in self.initial_observation(theta).items():
            key = (o_a, o_b)
            out[key] = out.get(key, 0.0) + prob
        return out


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


class ModelValidationError(ValueError):
    """Raised when a micro model table is malformed."""


def validate_model(model: MicroModel, *, tol: float = 1e-9) -> None:
    """Check every table is finite, non-negative and normalised.

    Also checks the structural completeness a planner relies on: for every
    ``(x, theta, action)`` there is a transition row, and for every reachable
    successor there is an observation row.
    """

    errors: list[str] = []

    prior_sum = sum(model.theta_prior.values())
    if abs(prior_sum - 1.0) > tol:
        errors.append(f"theta_prior sums to {prior_sum!r}, expected 1.0")
    for theta, p in model.theta_prior.items():
        if theta not in model.theta_values:
            errors.append(f"theta_prior has unknown theta {theta!r}")
        if p < 0:
            errors.append(f"theta_prior[{theta}] is negative")

    if model.initial_x not in model.x_values:
        errors.append("initial_x is not in x_values")
    if model.goal_x not in model.x_values:
        errors.append("goal_x is not in x_values")
    if model.horizon < 1:
        errors.append("horizon must be >= 1")
    if model.fail_cost <= 0:
        errors.append("fail_cost must be positive so failure is never cheap")
    if len(set(model.actions)) != len(model.actions):
        errors.append("actions must be unique and stably ordered")

    for x in model.x_values:
        for theta in model.theta_values:
            for action in model.actions:
                key = (x, theta, action)
                if key not in model.transitions:
                    errors.append(f"missing transition row {key}")
                    continue
                row = model.transitions[key]
                if not row:
                    errors.append(f"empty transition row {key}")
                total = 0.0
                for x_next, p in row.items():
                    if x_next not in model.x_values:
                        errors.append(f"{key}: unknown successor {x_next!r}")
                    if p < 0:
                        errors.append(f"{key}: negative probability for {x_next}")
                    total += p
                    obs_key = (x, theta, action, x_next)
                    if p > 0 and obs_key not in model.observations:
                        errors.append(f"missing observation row {obs_key}")
                if abs(total - 1.0) > tol:
                    errors.append(f"transition row {key} sums to {total!r}")

    for key, row in model.observations.items():
        total = sum(row.values())
        if abs(total - 1.0) > tol:
            errors.append(f"observation row {key} sums to {total!r}")
        for obs, p in row.items():
            if p < 0:
                errors.append(f"observation row {key} has negative probability")
            if len(obs) != 2 or len(obs[0]) != 2:
                errors.append(f"observation row {key} has malformed key {obs!r}")

    for theta in model.theta_values:
        if theta not in model.init_observations:
            errors.append(f"missing initial observation row for theta {theta!r}")
            continue
        total = sum(model.init_observations[theta].values())
        if abs(total - 1.0) > tol:
            errors.append(f"initial observation row for {theta} sums to {total!r}")

    for x in model.x_values:
        for action in model.actions:
            if (x, action) not in model.costs:
                errors.append(f"missing cost for ({x}, {action})")
    for (x, action), c in model.costs.items():
        if c < 0:
            errors.append(f"cost({x}, {action}) is negative")

    if errors:
        raise ModelValidationError(
            f"model {model.name!r} failed validation:\n  " + "\n  ".join(errors)
        )
