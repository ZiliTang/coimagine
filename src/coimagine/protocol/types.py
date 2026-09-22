"""Strictly isolated information objects for the 2D protocol.

The classes here exist so that the information boundary of
``docs/2d_protocol_spec.md`` is enforced by *types*, not by convention:

* :class:`TrueState` and :class:`OracleContext` are simulator / offline-design
  objects. They are privileged and must never be handed to a deployment policy.
* :class:`PublicContext` and :class:`PrivateHistory` are the *only* objects a
  deployment policy or message codec may read.

:func:`reject_privileged` is the runtime guard used by the deployment-facing
interfaces. A test changes the hidden truth while holding the visible inputs fixed
and asserts that the guard still passes for the unchanged visible objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


# --------------------------------------------------------------------------
# Privileged objects (simulator / offline oracle design only)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrueState:
    """Ground-truth simulator state ``s = (x, theta)``.

    ``theta`` is the hidden physical type and is never public.
    """

    x: str
    theta: str

    def __post_init__(self) -> None:
        if not isinstance(self.x, str) or not isinstance(self.theta, str):
            raise TypeError("TrueState coordinates must be strings in the micro family.")


@dataclass(frozen=True)
class OracleContext:
    """Privileged bundle for offline oracle design and evaluation.

    Carries the true state and the generative model. It is deliberately a
    different type from :class:`PublicContext` so that a deployment interface can
    reject it.
    """

    true_state: TrueState
    model: Any  # coimagine.toy.model.MicroModel, kept untyped to avoid a cycle
    note: str = "privileged; offline oracle design and evaluation only"


# --------------------------------------------------------------------------
# Deployment-visible objects
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PublicRecord:
    """The common record ``C`` shared by both robots.

    Invariant at a boundary *before* this round's messages are sent:

    * ``y_public`` holds ``t + 1`` entries (``y_0 .. y_t``);
    * ``actions`` holds ``t`` entries (``a_0 .. a_{t-1}``);
    * ``messages`` holds ``t`` entries (``m_0 .. m_{t-1}``).

    ``messages[k]`` is the pair ``(m_A, m_B)`` sent at boundary ``k``.
    """

    y_public: Tuple[str, ...] = ()
    actions: Tuple[str, ...] = ()
    messages: Tuple[Tuple[int, int], ...] = ()

    # -- construction helpers ------------------------------------------------

    def advance(self, *, action: str, messages: Tuple[int, int], y_next: str) -> "PublicRecord":
        """Advance one full boundary: commit the action, send, then observe.

        The action, this round's message pair and the next common observation are
        appended together so that the invariants
        ``len(y_public) == len(actions) + 1`` and ``len(messages) == len(actions)``
        hold at every decision boundary.
        """

        return PublicRecord(
            y_public=self.y_public + (y_next,),
            actions=self.actions + (action,),
            messages=self.messages + (messages,),
        )

    @property
    def step(self) -> int:
        """Index of the current decision boundary."""
        return len(self.actions)

    def commit_key(self, m_a: int, m_b: int) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[Tuple[int, int], ...], int, int]:
        """The public information a commit rule is allowed to read.

        Used by tests to show that two robots with the same public record and the
        same message pair produce the same committed action.
        """

        return (self.y_public, self.actions, self.messages, m_a, m_b)

    def __post_init__(self) -> None:
        for m in self.messages:
            if len(m) != 2:
                raise ValueError("Each message entry must be a (m_A, m_B) pair.")
        if len(self.y_public) != len(self.actions) + 1:
            raise ValueError(
                "PublicRecord invariant violated: len(y_public) must equal "
                "len(actions) + 1."
            )
        if len(self.messages) != len(self.actions):
            raise ValueError(
                "PublicRecord invariant violated: len(messages) must equal "
                "len(actions)."
            )


@dataclass(frozen=True)
class PrivateHistory:
    """Robot ``i``'s private observation history ``h_i = (o_i,0 .. o_i,t)``."""

    robot_id: int
    observations: Tuple[str, ...] = ()

    def extended(self, observation: str) -> "PrivateHistory":
        return PrivateHistory(
            robot_id=self.robot_id,
            observations=self.observations + (observation,),
        )


@dataclass(frozen=True)
class PublicContext:
    """The deployment-visible context for one robot at a decision boundary.

    It bundles the public record with the robot's own private history and the
    publicly derived candidate set. It contains no :class:`TrueState`.
    """

    record: PublicRecord
    history: PrivateHistory
    candidates: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("A decision boundary must expose at least one candidate.")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError("Candidate IDs must be unique and stably ordered.")


@dataclass(frozen=True)
class Message:
    """A serialised, inspectable message."""

    robot_id: int
    symbol: int
    bits: int
    codebook: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.symbol not in self.codebook:
            raise ValueError("Message symbol is not part of its codebook.")
        if self.bits < 0:
            raise ValueError("Message bits must be non-negative.")
        if len(self.codebook) != (1 << self.bits):
            raise ValueError(
                "The codebook must contain exactly 2**bits symbols; "
                "a float array is not a 1-bit message."
            )


# --------------------------------------------------------------------------
# Runtime information-boundary guard
# --------------------------------------------------------------------------


class InformationBoundaryError(RuntimeError):
    """Raised when a privileged object reaches a deployment-facing interface."""


_DEPLOYMENT_TYPES = (PublicContext, PrivateHistory, PublicRecord)


def reject_privileged(*objects: Any, interface: str = "deployment interface") -> None:
    """Raise if any object is privileged.

    Deployment-facing functions call this so that passing a :class:`TrueState`,
    an :class:`OracleContext`, or a raw environment object is an error rather than
    a silent leak.
    """

    for obj in objects:
        if isinstance(obj, (TrueState, OracleContext)):
            raise InformationBoundaryError(
                f"{interface} received a privileged object of type "
                f"{type(obj).__name__}; only PublicContext / PrivateHistory / "
                f"PublicRecord are allowed."
            )


def visible_signature(context: PublicContext) -> Tuple[Any, ...]:
    """A hashable summary of everything the deployment side may read.

    Used by tests: changing hidden truth must not change this signature for a
    fixed visible input.
    """

    reject_privileged(context, interface="visible_signature")
    return (
        context.record.y_public,
        context.record.actions,
        context.record.messages,
        context.history.robot_id,
        context.history.observations,
        context.candidates,
    )
