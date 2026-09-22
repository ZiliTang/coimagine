"""A limited message channel: what should the teammate actually say?

Round 20 established that the teammate-slipping mode is *information-theoretically*
unavailable to a silent robot: it observes only the public ``moved``/``stalled`` event,
which does not say who failed to contribute. That makes messaging an evidence-driven
question rather than an assumed one, and this module poses it in its narrowest useful form:

    with one message per decision and at most ``B`` bits, what should it contain?

The schemes below are declared, inspectable codebooks. Each is a function of the sender's
**own private state or history**, so the message is something the sender can actually
compute; it is one-directional (teammate -> self) and the receiver simply appends the
decoded message to its own observation. That is deliberately less than the full protocol
(no joint commit rule, no message-dependent reference belief), and it measures exactly the
question C1 asks: the value of the *content*, at a fixed bit budget.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

from ..grid.dynamics import FEEDBACK_SYMBOLS
from ..grid.state import HiddenPhysics

#: ``scheme -> (bits, codebook)``. ``bits`` is the maximum payload per decision; the
#: codebook is the inspectable discrete alphabet, never a float array.
MESSAGE_SCHEMES: Dict[str, Tuple[int, Tuple[str, ...]]] = {
    # A zero-bit scheme carries no payload at all: an empty codebook, so the receiver's
    # observation gains nothing. Giving silence a one-symbol codebook would add a constant
    # column to every feature vector for no information and shift the action slice.
    "none": (0, ()),
    "traction": (1, ("grip", "slip")),
    "feedback": (2, tuple(FEEDBACK_SYMBOLS)),
}

#: Preferred order for reporting.
SCHEME_ORDER: Tuple[str, ...] = ("none", "traction", "feedback")


def message_dim(scheme: str) -> int:
    return len(MESSAGE_SCHEMES[scheme][1])


def message_symbol(
    scheme: str, physics: HiddenPhysics, teammate_history: Sequence[str]
) -> str:
    """The symbol the teammate would send, from its own private information only."""

    if scheme == "none":
        return "silent"
    if scheme == "traction":
        return physics.traction[1]
    if scheme == "feedback":
        return teammate_history[-1] if teammate_history else FEEDBACK_SYMBOLS[0]
    raise ValueError(scheme)


def message_vector(
    scheme: str, physics: HiddenPhysics, teammate_history: Sequence[str]
) -> Tuple[float, ...]:
    """One-hot encoding of the message, of length :func:`message_dim`."""

    codebook = MESSAGE_SCHEMES[scheme][1]
    symbol = message_symbol(scheme, physics, teammate_history)
    return tuple(1.0 if symbol == other else 0.0 for other in codebook)


def payload_bits(scheme: str) -> int:
    return MESSAGE_SCHEMES[scheme][0]
