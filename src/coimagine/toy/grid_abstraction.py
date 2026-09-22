"""A finite abstraction of the Stage 2 grid interaction, used as a design gate.

Before building the full grid benchmark we must know whether its hidden structure
makes the research questions *decidable*. The micro instances could not separate
message representations because their posterior over two hidden causes is a scalar
and every message was a monotone function of it.

This module encodes the grid's structure into the existing, verified finite engine:

* **six** hidden types, not two -- resistance x constraint mode x which robot has
  grip, including the case where two robots cannot satisfy the resistance;
* **four** local feedback symbols -- ``drove`` / ``strained`` / ``slipped`` /
  ``idle``, so one bit is genuinely lossy;
* the observation pair is **deterministically correlated** through the shared hidden
  type and the shared physical outcome, so the two robots' feedback is complementary
  rather than independent noise;
* actions ``CO_PUSH`` (both robots, capacity 2), ``PUSH_ONE`` (robot A alone,
  capacity 1), ``PROBE`` (short push, no displacement, cheap), ``RELEASE`` (the
  explicit constrained->free transition) and ``HOLD``;
* a release moves the configuration to ``RELEASED`` without moving the object, so
  the public event stays ``stalled`` exactly as in the grid dynamics.

This is a *gate*, not a benchmark: if the structure cannot even separate message
representations here, the grid would repeat the Stage 1 null result.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Tuple

from .model import JointObservation, MicroModel, validate_model

X_CONTACT = "CONTACT"
X_RELEASED = "RELEASED"
X_GOAL = "GOAL"

A_CO_PUSH = "CO_PUSH"
A_PUSH_ONE = "PUSH_ONE"
A_PROBE = "PROBE"
A_RELEASE = "RELEASE"
A_HOLD = "HOLD"

O_DROVE = "drove"
O_STRAINED = "strained"
O_SLIPPED = "slipped"
O_IDLE = "idle"

Y_MOVED = "moved"
Y_STALLED = "stalled"

X_VALUES: Tuple[str, ...] = (X_CONTACT, X_RELEASED, X_GOAL)
ACTIONS: Tuple[str, ...] = (A_CO_PUSH, A_PUSH_ONE, A_PROBE, A_RELEASE, A_HOLD)

#: ``name -> (resistance, constraint, traction_A, traction_B)``
THETA_SPECS: Mapping[str, Tuple[int, str, str, str]] = {
    "r1_free_gg": (1, "free", "grip", "grip"),
    "r2_free_gg": (2, "free", "grip", "grip"),
    "r2_free_gs": (2, "free", "grip", "slip"),
    "r1_cstr_gg": (1, "constrained", "grip", "grip"),
    "r2_cstr_gg": (2, "constrained", "grip", "grip"),
    "r1_free_sg": (1, "free", "slip", "grip"),
}

THETA_VALUES: Tuple[str, ...] = tuple(THETA_SPECS)

ACTION_COSTS: Mapping[str, float] = {
    A_CO_PUSH: 1.0,
    A_PUSH_ONE: 1.0,
    A_PROBE: 0.3,
    A_RELEASE: 1.0,
    A_HOLD: 0.0,
}


def _capacity(traction: Tuple[str, str], pushers: Tuple[int, ...]) -> int:
    return sum(1 for i in pushers if traction[i] == "grip")


def _push_succeeds(
    spec: Tuple[int, str, str, str], x: str, pushers: Tuple[int, ...]
) -> bool:
    resistance, constraint, traction_a, traction_b = spec
    if constraint == "constrained" and x == X_CONTACT:
        return False
    return _capacity((traction_a, traction_b), pushers) >= resistance


def _release_succeeds(spec: Tuple[int, str, str, str], x: str) -> bool:
    """A release needs a robot at the object, in a not-yet-released configuration.

    The gate models use different names for "robots at the object" (``CONTACT`` in
    the first, ``SAME`` in the split model), so both are accepted.
    """

    _resistance, constraint, traction_a, _traction_b = spec
    at_object = x in (X_CONTACT, X_SAME)
    return constraint == "constrained" and traction_a == "grip" and at_object


def _transition(spec: Tuple[int, str, str, str], x: str, action: str) -> str:
    if x == X_GOAL:
        return X_GOAL
    if action == A_CO_PUSH:
        return X_GOAL if _push_succeeds(spec, x, (0, 1)) else x
    if action == A_PUSH_ONE:
        return X_GOAL if _push_succeeds(spec, x, (0,)) else x
    if action == A_PROBE:
        return x
    if action == A_RELEASE:
        if _release_succeeds(spec, x):
            return X_RELEASED
        return x
    return x


def _nominals(
    spec: Tuple[int, str, str, str],
    x: str,
    action: str,
    x_next: str,
    *,
    coarse_loaded_feedback: bool = False,
) -> Tuple[str, str]:
    _resistance, _constraint, traction_a, traction_b = spec
    traction = (traction_a, traction_b)
    moved = x_next == X_GOAL and x != X_GOAL

    def robot_symbol(robot_id: int, would_succeed: bool, moved_flag: bool) -> str:
        # Under a loaded push the load is shared and a slipping robot cannot
        # separate its own slip from the object's resistance; a dedicated short
        # probe is instrumented and can. This is a modelling assumption, exposed as
        # a flag because C2 depends on it.
        if coarse_loaded_feedback and action in (A_CO_PUSH, A_PUSH_ONE) and not moved_flag:
            return O_STRAINED
        if traction[robot_id] == "slip":
            return O_SLIPPED
        if action == A_PROBE:
            return O_DROVE if would_succeed else O_STRAINED
        if action == A_RELEASE:
            return O_DROVE if would_succeed else O_STRAINED
        return O_DROVE if moved_flag else O_STRAINED

    if action == A_CO_PUSH:
        would = _push_succeeds(spec, x, (0, 1))
        return (
            robot_symbol(0, would, moved),
            robot_symbol(1, would, moved),
        )
    if action == A_PUSH_ONE:
        would = _push_succeeds(spec, x, (0,))
        return (robot_symbol(0, would, moved), O_IDLE)
    if action == A_PROBE:
        would = _push_succeeds(spec, x, (0,))
        return (robot_symbol(0, would, moved), O_IDLE)
    if action == A_RELEASE:
        would = _release_succeeds(spec, x)
        return (robot_symbol(0, would, moved), O_IDLE)
    return (O_IDLE, O_IDLE)


def _y_public(x: str, x_next: str) -> str:
    return Y_MOVED if (x_next == X_GOAL and x != X_GOAL) else Y_STALLED


def build_grid_abstraction(
    *,
    horizon: int = 3,
    probe_cost: float = 0.3,
    initial_informative: bool = False,
    coarse_loaded_feedback: bool = False,
    fail_cost: float = 10.0,
    name: Optional[str] = None,
) -> MicroModel:
    """Build the finite gate model.

    ``initial_informative=True`` gives both robots the feedback of an already
    attempted cooperative push that stalled, so evidence exists at ``t = 0`` (the
    M1-like case). Otherwise the initial feedback is uninformative and the team must
    act to learn (the M2-like case).
    """

    costs = {A_CO_PUSH: 1.0, A_PUSH_ONE: 1.0, A_PROBE: probe_cost,
             A_RELEASE: 1.0, A_HOLD: 0.0}

    transitions: Dict[Tuple[str, str, str], Mapping[str, float]] = {}
    observations: Dict[Tuple[str, str, str, str], Mapping[JointObservation, float]] = {}
    for theta in THETA_VALUES:
        spec = THETA_SPECS[theta]
        for x in X_VALUES:
            for action in ACTIONS:
                x_next = _transition(spec, x, action)
                transitions[(x, theta, action)] = {x_next: 1.0}
                nominals = _nominals(
                    spec, x, action, x_next,
                    coarse_loaded_feedback=coarse_loaded_feedback,
                )
                y = _y_public(x, x_next)
                observations[(x, theta, action, x_next)] = {
                    (nominals, y): 1.0
                }

    init_observations: Dict[str, Mapping[JointObservation, float]] = {}
    for theta in THETA_VALUES:
        spec = THETA_SPECS[theta]
        if initial_informative:
            # A diagnostic push already happened: the public event stays 'stalled'
            # (a short push does not displace the object) while the private feedback
            # is informative. This keeps a single initial public observation, which
            # the oracle suite requires.
            nominals = _nominals(
                spec, X_CONTACT, A_PROBE, X_CONTACT,
                coarse_loaded_feedback=coarse_loaded_feedback,
            )
        else:
            nominals = (O_IDLE, O_IDLE)
        init_observations[theta] = {(nominals, Y_STALLED): 1.0}

    model = MicroModel(
        name=name
        or ("grid_abstraction_informed" if initial_informative else "grid_abstraction_blind"),
        x_values=X_VALUES,
        theta_values=THETA_VALUES,
        actions=ACTIONS,
        horizon=horizon,
        initial_x=X_CONTACT,
        goal_x=X_GOAL,
        theta_prior={theta: 1.0 / len(THETA_VALUES) for theta in THETA_VALUES},
        transitions=transitions,
        observations=observations,
        init_observations=init_observations,
        costs={(x, action): (0.0 if x == X_GOAL else costs[action])
               for x in X_VALUES for action in ACTIONS},
        fail_cost=fail_cost,
        probe_actions=frozenset({A_PROBE}),
        description=(
            "Finite abstraction of the Stage 2 grid interaction: six hidden contact "
            "types, four correlated feedback symbols, probe / release / cooperative "
            "push. Used only as a design gate for C1 and C2."
        ),
        metadata={"role": "stage2 design gate", "theta_count": len(THETA_VALUES)},
    )
    validate_model(model)
    return model


# --------------------------------------------------------------------------
# Second gate: split contact, where the information changes the *sequence*
# --------------------------------------------------------------------------

X_SPLIT = "SPLIT"
X_SAME = "SAME"
X_RELEASED_SAME = "RELEASED"
X_VALUES_SPLIT: Tuple[str, ...] = (X_SPLIT, X_SAME, X_RELEASED_SAME, X_GOAL)

A_REPOSITION = "REPOSITION"
ACTIONS_SPLIT: Tuple[str, ...] = (
    A_CO_PUSH,
    A_PUSH_ONE,
    A_PROBE,
    A_REPOSITION,
    A_RELEASE,
    A_HOLD,
)

ACTION_COSTS_SPLIT: Mapping[str, float] = {
    A_CO_PUSH: 1.0,
    A_PUSH_ONE: 1.0,
    A_PROBE: 0.3,
    A_REPOSITION: 1.0,
    A_RELEASE: 1.0,
    A_HOLD: 0.0,
}


def _single_push_succeeds(spec: Tuple[int, str, str, str], x: str) -> bool:
    resistance, constraint, traction_a, _tb = spec
    if constraint == "constrained" and x != X_RELEASED_SAME:
        return False
    return (1 if traction_a == "grip" else 0) >= resistance


def _co_push_succeeds(spec: Tuple[int, str, str, str], x: str) -> bool:
    resistance, constraint, traction_a, traction_b = spec
    if constraint == "constrained" and x != X_RELEASED_SAME:
        return False
    return _capacity((traction_a, traction_b), (0, 1)) >= resistance


def _transition_split(spec: Tuple[int, str, str, str], x: str, action: str) -> str:
    if x == X_GOAL:
        return X_GOAL
    if action == A_REPOSITION:
        return X_SAME if x == X_SPLIT else x
    if action == A_PUSH_ONE:
        return X_GOAL if _single_push_succeeds(spec, x) else x
    if action == A_CO_PUSH:
        # Opposite faces cannot cooperate: the push is wasted.
        if x == X_SPLIT:
            return X_SPLIT
        return X_GOAL if _co_push_succeeds(spec, x) else x
    if action == A_PROBE:
        return x
    if action == A_RELEASE:
        if x == X_SAME and _release_succeeds(spec, x):
            return X_RELEASED_SAME
        return x
    return x


def _nominals_split(
    spec: Tuple[int, str, str, str], x: str, action: str, x_next: str
) -> Tuple[str, str]:
    _resistance, _constraint, traction_a, traction_b = spec
    traction = (traction_a, traction_b)
    moved = x_next == X_GOAL and x != X_GOAL

    def symbol(robot_id: int, would: bool) -> str:
        if traction[robot_id] == "slip":
            return O_SLIPPED
        if action in (A_PROBE, A_RELEASE):
            return O_DROVE if would else O_STRAINED
        return O_DROVE if moved else O_STRAINED

    if action == A_CO_PUSH and x == X_SPLIT:
        return (O_STRAINED, O_STRAINED)
    if action in (A_CO_PUSH, A_PUSH_ONE):
        would = (
            _co_push_succeeds(spec, x)
            if action == A_CO_PUSH
            else _single_push_succeeds(spec, x)
        )
        first = symbol(0, would)
        second = symbol(1, would) if action == A_CO_PUSH else O_IDLE
        return (first, second)
    if action == A_PROBE:
        return (symbol(0, _single_push_succeeds(spec, x)), O_IDLE)
    if action == A_RELEASE:
        return (symbol(0, _release_succeeds(spec, x)), O_IDLE)
    return (O_IDLE, O_IDLE)


def build_grid_split_abstraction(
    *,
    horizon: int = 3,
    probe_cost: float = 0.3,
    initial_informative: bool = False,
    public_event_informative: bool = True,
    fail_cost: float = 10.0,
    name: str = "grid_split_gate",
) -> MicroModel:
    """Gate model where the robots start on **opposite faces** of the object.

    From ``SPLIT`` a single push works only for resistance 1, while resistance 2
    requires repositioning to ``SAME`` and a cooperative push. The probe therefore
    changes the *sequence*, not just the current action, which is the structure C2
    needs.
    """

    costs = dict(ACTION_COSTS_SPLIT)
    costs[A_PROBE] = probe_cost

    transitions: Dict[Tuple[str, str, str], Mapping[str, float]] = {}
    observations: Dict[Tuple[str, str, str, str], Mapping[JointObservation, float]] = {}
    for theta in THETA_VALUES:
        spec = THETA_SPECS[theta]
        for x in X_VALUES_SPLIT:
            for action in ACTIONS_SPLIT:
                x_next = _transition_split(spec, x, action)
                transitions[(x, theta, action)] = {x_next: 1.0}
                nominals = _nominals_split(spec, x, action, x_next)
                y = _y_public(x, x_next) if public_event_informative else Y_STALLED
                observations[(x, theta, action, x_next)] = {(nominals, y): 1.0}

    init_observations: Dict[str, Mapping[JointObservation, float]] = {}
    for theta in THETA_VALUES:
        spec = THETA_SPECS[theta]
        if initial_informative:
            nominals = _nominals_split(spec, X_SPLIT, A_PROBE, X_SPLIT)
        else:
            nominals = (O_IDLE, O_IDLE)
        init_observations[theta] = {(nominals, Y_STALLED): 1.0}

    model = MicroModel(
        name=name,
        x_values=X_VALUES_SPLIT,
        theta_values=THETA_VALUES,
        actions=ACTIONS_SPLIT,
        horizon=horizon,
        initial_x=X_SPLIT,
        goal_x=X_GOAL,
        theta_prior={theta: 1.0 / len(THETA_VALUES) for theta in THETA_VALUES},
        transitions=transitions,
        observations=observations,
        init_observations=init_observations,
        costs={(x, action): (0.0 if x == X_GOAL else costs[action])
               for x in X_VALUES_SPLIT for action in ACTIONS_SPLIT},
        fail_cost=fail_cost,
        probe_actions=frozenset({A_PROBE}),
        description=(
            "Split-contact grid gate: single push only works for resistance 1, so "
            "resistance 2 forces reposition + cooperative push."
        ),
        metadata={"role": "stage2 design gate (C2)", "theta_count": len(THETA_VALUES)},
    )
    validate_model(model)
    return model
