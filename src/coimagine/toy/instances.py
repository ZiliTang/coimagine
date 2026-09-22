"""Micro diagnostic instances M0-M3.

All instances come from the same engine in :mod:`coimagine.toy.model`. No
instance carries an information-gain, enabling or probe bonus: the only cost is
``cost(x, a)`` plus the explicit failure cost, and the only information channel is
the published joint observation table.

State space (all instances)::

    x     in {CONTACT, PREPARED, GOAL}
    theta in {HEAVY, SLIP}

Local observable ``o_i in {loaded, slipping}``; public observable
``y_pub in {moved, stalled}``. ``y_pub = moved`` exactly when the configuration
changed, otherwise ``stalled``.
"""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from .model import JointObservation, MicroModel, validate_model

# --------------------------------------------------------------------------
# Labels
# --------------------------------------------------------------------------

X_CONTACT = "CONTACT"
X_PREPARED = "PREPARED"
X_GOAL = "GOAL"

TH_HEAVY = "HEAVY"
TH_SLIP = "SLIP"

A_CO_PUSH = "CO_PUSH"
A_RELEASE_PUSH_B = "RELEASE_PUSH_B"
A_REPOSITION_B = "REPOSITION_B"
A_PROBE = "PROBE"
A_HOLD = "HOLD"

O_LOADED = "loaded"
O_SLIPPING = "slipping"

Y_MOVED = "moved"
Y_STALLED = "stalled"

#: Correlated joint local tables. ``both`` = both robots see the cause-consistent
#: value, ``only_a`` / ``only_b`` = exactly one does, ``neither`` = neither does.
_LOCAL_STRENGTHS: Mapping[str, Mapping[str, float]] = {
    "weak": {"both": 0.45, "only_a": 0.20, "only_b": 0.20, "neither": 0.15},
    "strong": {"both": 0.90, "only_a": 0.04, "only_b": 0.04, "neither": 0.02},
}

#: An uninformative joint table: a fair independent coin in every cell. It is
#: identical for both hidden causes, so it carries no evidence.
_UNINFORMATIVE = {
    (O_LOADED, O_LOADED): 0.25,
    (O_LOADED, O_SLIPPING): 0.25,
    (O_SLIPPING, O_LOADED): 0.25,
    (O_SLIPPING, O_SLIPPING): 0.25,
}


def _joint_local(theta: str, strength: str) -> Mapping[Tuple[str, str], float]:
    """Cause-consistent correlated joint local observation table.

    The two robots' observations are **not** independent for ``weak``/``strong``;
    the table is published as-is and never factored.
    """

    if strength == "none":
        return dict(_UNINFORMATIVE)
    cause = O_LOADED if theta == TH_HEAVY else O_SLIPPING
    other = O_SLIPPING if cause == O_LOADED else O_LOADED
    spec = _LOCAL_STRENGTHS[strength]
    return {
        (cause, cause): spec["both"],
        (cause, other): spec["only_a"],
        (other, cause): spec["only_b"],
        (other, other): spec["neither"],
    }


def _build_observation_tables(
    transitions: Mapping[Tuple[str, str, str], Mapping[str, float]],
    *,
    actions: Tuple[str, ...],
    x_values: Tuple[str, ...],
    theta_values: Tuple[str, ...],
    goal_x: str,
    strength_for_action: Mapping[str, str],
) -> Dict[Tuple[str, str, str, str], Mapping[JointObservation, float]]:
    tables: Dict[Tuple[str, str, str, str], Mapping[JointObservation, float]] = {}
    for x in x_values:
        for theta in theta_values:
            for action in actions:
                for x_next, prob in transitions[(x, theta, action)].items():
                    if prob <= 0.0:
                        continue
                    if x == goal_x:
                        local = _UNINFORMATIVE
                    else:
                        local = _joint_local(
                            theta, strength_for_action.get(action, "weak")
                        )
                    y = Y_MOVED if x_next != x else Y_STALLED
                    tables[(x, theta, action, x_next)] = {
                        ((o_a, o_b), y): p for (o_a, o_b), p in local.items()
                    }
    return tables


def _build_model(
    name: str,
    *,
    transitions: Mapping[Tuple[str, str, str], Mapping[str, float]],
    actions: Tuple[str, ...],
    theta_prior: Mapping[str, float],
    horizon: int = 3,
    fail_cost: float = 10.0,
    strength_for_action: Mapping[str, str],
    default_action_strength: str = "weak",
    initial_strength: str,
    action_costs: Mapping[str, float],
    probe_actions: Tuple[str, ...] = (),
    description: str = "",
    metadata: Mapping[str, object] | None = None,
) -> MicroModel:
    x_values = (X_CONTACT, X_PREPARED, X_GOAL)
    theta_values = (TH_HEAVY, TH_SLIP)
    observations = _build_observation_tables(
        transitions,
        actions=actions,
        x_values=x_values,
        theta_values=theta_values,
        goal_x=X_GOAL,
        strength_for_action={
            action: strength_for_action.get(action, default_action_strength)
            for action in actions
        },
    )
    init_observations = {
        theta: {
            ((o_a, o_b), Y_STALLED): p
            for (o_a, o_b), p in _joint_local(theta, initial_strength).items()
        }
        for theta in theta_values
    }
    costs: Dict[Tuple[str, str], float] = {}
    for x in x_values:
        for action in actions:
            costs[(x, action)] = 0.0 if x == X_GOAL else float(action_costs[action])

    model = MicroModel(
        name=name,
        x_values=x_values,
        theta_values=theta_values,
        actions=actions,
        horizon=horizon,
        initial_x=X_CONTACT,
        goal_x=X_GOAL,
        theta_prior=dict(theta_prior),
        transitions=dict(transitions),
        observations=observations,
        init_observations=init_observations,
        costs=costs,
        fail_cost=fail_cost,
        probe_actions=frozenset(probe_actions),
        description=description,
        metadata=dict(metadata or {}),
    )
    validate_model(model)
    return model


def _absorbing_goal(
    transitions: Dict[Tuple[str, str, str], Mapping[str, float]], actions: Tuple[str, ...]
) -> None:
    for theta in (TH_HEAVY, TH_SLIP):
        for action in actions:
            transitions[(X_GOAL, theta, action)] = {X_GOAL: 1.0}


def _tables(
    rows: Mapping[Tuple[str, str], Mapping[str, float]],
    *,
    actions: Tuple[str, ...],
) -> Dict[Tuple[str, str, str], Mapping[str, float]]:
    """Expand ``{(x, action): {x_next: p}}`` over both hidden causes."""

    out: Dict[Tuple[str, str, str], Mapping[str, float]] = {}
    for theta in (TH_HEAVY, TH_SLIP):
        for (x, action), row in rows.items():
            out[(x, theta, action)] = dict(row)
    return out


_ALWAYS_GOAL = {X_GOAL: 1.0}
_STAY = {X_CONTACT: 1.0}
_PREPARED_ROW = {X_PREPARED: 1.0}


# --------------------------------------------------------------------------
# M0 - public information is sufficient
# --------------------------------------------------------------------------


def build_m0(*, horizon: int = 3, fail_cost: float = 10.0) -> MicroModel:
    """Plain cooperative push: the cause does not change the correct action.

    ``CO_PUSH`` succeeds from ``CONTACT`` for both hidden causes, and the local
    feedback is uninformative. Private communication can therefore add nothing.
    """

    rows: Dict[Tuple[str, str], Mapping[str, float]] = {
        (X_CONTACT, A_CO_PUSH): _ALWAYS_GOAL,
        (X_CONTACT, A_REPOSITION_B): _PREPARED_ROW,
        (X_CONTACT, A_HOLD): _STAY,
        (X_PREPARED, A_CO_PUSH): _ALWAYS_GOAL,
        (X_PREPARED, A_REPOSITION_B): _PREPARED_ROW,
        (X_PREPARED, A_HOLD): _PREPARED_ROW,
    }
    actions = (A_CO_PUSH, A_REPOSITION_B, A_HOLD)
    transitions = _tables(rows, actions=actions)
    _absorbing_goal(transitions, actions)
    return _build_model(
        "M0",
        transitions=transitions,
        actions=actions,
        theta_prior={TH_HEAVY: 0.5, TH_SLIP: 0.5},
        horizon=horizon,
        fail_cost=fail_cost,
        strength_for_action={},
        default_action_strength="none",
        initial_strength="none",
        action_costs={A_CO_PUSH: 1.0, A_REPOSITION_B: 1.0, A_HOLD: 0.0},
        description=(
            "Public information sufficient: the cooperative push succeeds for both "
            "causes and private local feedback is uninformative."
        ),
        metadata={"expectation": "communication gives no gain"},
    )


# --------------------------------------------------------------------------
# M1 - stall-cause ambiguity, each cause has its own one-step recovery
# --------------------------------------------------------------------------


def build_m1(
    *, horizon: int = 3, fail_cost: float = 10.0, initial_strength: str = "weak"
) -> MicroModel:
    """Same public stall, different local evidence, different recovery.

    ``CO_PUSH`` only succeeds for ``HEAVY``; ``RELEASE_PUSH_B`` (B repositions and
    re-pushes as one coordinated skill) only succeeds for ``SLIP``.
    """

    rows: Dict[Tuple[str, str], Mapping[str, float]] = {
        (X_CONTACT, A_CO_PUSH): {X_GOAL: 1.0},  # HEAVY
        (X_CONTACT, A_RELEASE_PUSH_B): {X_CONTACT: 1.0},  # HEAVY: not the fix
        (X_CONTACT, A_HOLD): _STAY,
        (X_PREPARED, A_CO_PUSH): _ALWAYS_GOAL,
        (X_PREPARED, A_RELEASE_PUSH_B): _PREPARED_ROW,
        (X_PREPARED, A_HOLD): _PREPARED_ROW,
    }
    actions = (A_CO_PUSH, A_RELEASE_PUSH_B, A_HOLD)
    # SLIP mirrors: the coordinated push stalls, the reposition-push succeeds.
    transitions: Dict[Tuple[str, str, str], Mapping[str, float]] = {}
    for theta in (TH_HEAVY, TH_SLIP):
        for action in actions:
            row = rows[(X_CONTACT, action)]
            if theta == TH_HEAVY:
                transitions[(X_CONTACT, theta, action)] = dict(row)
            else:
                if action == A_CO_PUSH:
                    transitions[(X_CONTACT, theta, action)] = {X_CONTACT: 1.0}
                elif action == A_RELEASE_PUSH_B:
                    transitions[(X_CONTACT, theta, action)] = {X_GOAL: 1.0}
                else:
                    transitions[(X_CONTACT, theta, action)] = {X_CONTACT: 1.0}
        for action in actions:
            transitions[(X_PREPARED, theta, action)] = (
                {X_GOAL: 1.0} if action == A_CO_PUSH else {X_PREPARED: 1.0}
            )
    _absorbing_goal(transitions, actions)
    return _build_model(
        "M1",
        transitions=transitions,
        actions=actions,
        theta_prior={TH_HEAVY: 0.5, TH_SLIP: 0.5},
        horizon=horizon,
        fail_cost=fail_cost,
        strength_for_action={},  # default "weak" for every action
        default_action_strength="weak",
        initial_strength=initial_strength,
        action_costs={A_CO_PUSH: 1.0, A_RELEASE_PUSH_B: 1.0, A_HOLD: 0.0},
        description=(
            "Stall-cause ambiguity: public stalled is identical, the correlated "
            "local tables carry complementary evidence, and each cause has a "
            "different one-step recovery."
        ),
        metadata={"expectation": "a short message changes the committed action"},
    )


# --------------------------------------------------------------------------
# M2 - costly identification, with two negative controls
# --------------------------------------------------------------------------


def build_m2(
    *,
    horizon: int = 3,
    fail_cost: float = 10.0,
    probe_cost: float = 0.2,
    probe_strength: str = "strong",
    initial_strength: str = "none",
    name: str = "M2",
) -> MicroModel:
    """Costly probe: the initial local evidence is uninformative, ``PROBE`` is not.

    ``probe_strength="none"`` gives the uninformative-probe negative control;
    a large ``probe_cost`` gives the too-expensive-probe negative control.
    """

    rows: Dict[Tuple[str, str], Mapping[str, float]] = {
        (X_CONTACT, A_CO_PUSH): {X_GOAL: 1.0},  # HEAVY
        (X_CONTACT, A_RELEASE_PUSH_B): {X_CONTACT: 1.0},  # HEAVY
        (X_CONTACT, A_PROBE): _STAY,
        (X_CONTACT, A_HOLD): _STAY,
        (X_PREPARED, A_CO_PUSH): _ALWAYS_GOAL,
        (X_PREPARED, A_RELEASE_PUSH_B): _PREPARED_ROW,
        (X_PREPARED, A_PROBE): _PREPARED_ROW,
        (X_PREPARED, A_HOLD): _PREPARED_ROW,
    }
    actions = (A_CO_PUSH, A_RELEASE_PUSH_B, A_PROBE, A_HOLD)
    transitions: Dict[Tuple[str, str, str], Mapping[str, float]] = {}
    for theta in (TH_HEAVY, TH_SLIP):
        for action in actions:
            if theta == TH_HEAVY:
                row = rows[(X_CONTACT, action)]
            else:
                if action == A_CO_PUSH:
                    row = {X_CONTACT: 1.0}
                elif action == A_RELEASE_PUSH_B:
                    row = {X_GOAL: 1.0}
                else:
                    row = {X_CONTACT: 1.0}
            transitions[(X_CONTACT, theta, action)] = dict(row)
        for action in actions:
            transitions[(X_PREPARED, theta, action)] = (
                {X_GOAL: 1.0} if action == A_CO_PUSH else {X_PREPARED: 1.0}
            )
    _absorbing_goal(transitions, actions)
    return _build_model(
        name,
        transitions=transitions,
        actions=actions,
        theta_prior={TH_HEAVY: 0.5, TH_SLIP: 0.5},
        horizon=horizon,
        fail_cost=fail_cost,
        strength_for_action={A_PROBE: probe_strength},
        default_action_strength="none",
        initial_strength=initial_strength,
        action_costs={
            A_CO_PUSH: 1.0,
            A_RELEASE_PUSH_B: 1.0,
            A_PROBE: probe_cost,
            A_HOLD: 0.0,
        },
        probe_actions=(A_PROBE,),
        description=(
            "Costly identification: the initial local feedback is uninformative "
            "and a short diagnostic push is informative at a cost."
        ),
        metadata={"expectation": "probe iff its information is worth the cost"},
    )


# --------------------------------------------------------------------------
# M3 - preparation changes the contact mode (known cause, no information value)
# --------------------------------------------------------------------------


def build_m3(*, horizon: int = 3, fail_cost: float = 10.0) -> MicroModel:
    """Physical preparation only: ``REPOSITION_B`` enables the coordinated push.

    The hidden cause does not change the dynamics and the local feedback is
    uninformative, so the value of preparation is physical, not informational.
    """

    rows: Dict[Tuple[str, str], Mapping[str, float]] = {
        (X_CONTACT, A_CO_PUSH): _STAY,
        (X_CONTACT, A_REPOSITION_B): _PREPARED_ROW,
        (X_CONTACT, A_HOLD): _STAY,
        (X_PREPARED, A_CO_PUSH): _ALWAYS_GOAL,
        (X_PREPARED, A_REPOSITION_B): _PREPARED_ROW,
        (X_PREPARED, A_HOLD): _PREPARED_ROW,
    }
    actions = (A_CO_PUSH, A_REPOSITION_B, A_HOLD)
    transitions = _tables(rows, actions=actions)
    _absorbing_goal(transitions, actions)
    return _build_model(
        "M3",
        transitions=transitions,
        actions=actions,
        theta_prior={TH_HEAVY: 0.5, TH_SLIP: 0.5},
        horizon=horizon,
        fail_cost=fail_cost,
        strength_for_action={},
        default_action_strength="none",
        initial_strength="none",
        action_costs={A_CO_PUSH: 1.0, A_REPOSITION_B: 1.0, A_HOLD: 0.0},
        description=(
            "Preparation changes the contact mode: the push is impossible until B "
            "repositions, independently of any information."
        ),
        metadata={"expectation": "preparation has physical, not informational value"},
    )


def all_instances(*, horizon: int = 3) -> Tuple[MicroModel, ...]:
    """The default M0-M3 suite, including the two M2 negative controls."""

    return (
        build_m0(horizon=horizon),
        build_m1(horizon=horizon),
        build_m2(
            horizon=horizon,
            probe_cost=0.2,
            probe_strength="strong",
            name="M2",
        ),
        build_m2(
            horizon=horizon,
            probe_cost=0.2,
            probe_strength="none",
            name="M2-uninformative",
        ),
        build_m2(
            horizon=horizon,
            probe_cost=100.0,
            probe_strength="strong",
            name="M2-expensive",
        ),
        build_m3(horizon=horizon),
    )
