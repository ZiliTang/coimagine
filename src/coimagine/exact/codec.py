"""Inspectable discrete message codecs with real bit accounting.

Every codec is a deterministic function of a :class:`~coimagine.exact.filter.RobotView`
(the deployment-visible information) to a symbol in ``{0, ..., 2**B - 1}``. The
codebook is exactly ``2**B`` symbols, so a reported ``B`` is a real payload size and
never a relabelled float array.

The codec family covers the baselines required by the execution prompt:

* :class:`NullCodec` -- public information only (``B = 0``);
* :class:`ObservationCodec` -- raw local observation summary;
* :class:`PosteriorModeCodec` -- compact hidden-cause posterior;
* :class:`PosteriorThresholdCodec` -- a deliberately coarse posterior (lossy at
  the same ``B``), used to expose message-representation loss;
* :class:`ConsequenceCodec` -- action-conditioned consequence summary: what the
  robot predicts will happen if the public candidates are executed.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from ..toy.model import MicroModel
from .filter import RobotView, robot_posterior, theta_posterior


def codebook(bits: int) -> Tuple[int, ...]:
    if bits < 0:
        raise ValueError("bits must be non-negative")
    return tuple(range(1 << bits))


def local_alphabet(model: MicroModel) -> Tuple[str, ...]:
    values = set()
    for ((o_a, o_b), _y) in model.init_observations[model.theta_values[0]]:
        values.add(o_a)
        values.add(o_b)
    for row in model.observations.values():
        for (o_a, o_b), _y in row:
            values.add(o_a)
            values.add(o_b)
    return tuple(sorted(values))


class Codec:
    """Base class: deterministic, cached, 2**bits symbols."""

    bits: int = 0
    name: str = "codec"

    def __init__(self) -> None:
        self._cache: Dict[Tuple[object, ...], int] = {}
        self.calls = 0

    # -- interface -----------------------------------------------------------

    def encode(self, model: MicroModel, view: RobotView) -> int:
        key = (model.name, view.key())
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.calls += 1
        symbol = self._encode(model, view)
        if not 0 <= symbol < (1 << self.bits):
            raise ValueError(
                f"{self.name} produced symbol {symbol} outside its "
                f"{1 << self.bits}-symbol codebook."
            )
        self._cache[key] = symbol
        return symbol

    def _encode(self, model: MicroModel, view: RobotView) -> int:  # pragma: no cover
        raise NotImplementedError

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}(bits={self.bits}, name={self.name!r})"


def _quantize(value: float, bits: int) -> int:
    """Quantise ``value in [0, 1]`` into ``2**bits`` equal bins."""

    if bits == 0:
        return 0
    levels = 1 << bits
    clipped = min(max(value, 0.0), 1.0)
    index = int(clipped * levels)
    if index >= levels:
        index = levels - 1
    return index


class NullCodec(Codec):
    """``B = 0``: no task message. The commit rule sees public information only."""

    bits = 0
    name = "null"

    def _encode(self, model: MicroModel, view: RobotView) -> int:
        return 0


class ObservationCodec(Codec):
    """Transmit the robot's current local observation.

    ``bits`` must be large enough to index the local alphabet; otherwise the
    observation is truncated deterministically.
    """

    name = "observation"

    def __init__(self, bits: int = 1) -> None:
        super().__init__()
        self.bits = bits

    def _encode(self, model: MicroModel, view: RobotView) -> int:
        alphabet = local_alphabet(model)
        current = view.observations[-1]
        index = alphabet.index(current)
        levels = 1 << self.bits
        return index % levels


class PosteriorModeCodec(Codec):
    """Compact posterior over the hidden cause.

    Quantises ``P(theta == theta_values[0])`` into ``2**B`` equal bins. At ``B = 1``
    this is the argmax-cause message.
    """

    name = "posterior-mode"

    def __init__(self, bits: int = 1) -> None:
        super().__init__()
        self.bits = bits

    def _encode(self, model: MicroModel, view: RobotView) -> int:
        posterior = robot_posterior(
            model, view.robot_id, view.observations, view.y_public, view.actions
        )
        if not posterior:
            return 0
        theta = theta_posterior(posterior)
        # A uniform posterior must have a deterministic, information-free output.
        p0 = theta.get(model.theta_values[0], 0.0)
        return _quantize(p0, self.bits)


class PosteriorThresholdCodec(Codec):
    """A coarse confidence message at ``B = 1``.

    Sends 1 only when ``P(theta == theta_values[0]) > threshold``. With a threshold
    away from ``0.5`` this discards usable evidence and therefore exposes the cost
    of a poorly chosen compact message.
    """

    bits = 1
    name = "posterior-threshold"

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__()
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must lie strictly inside (0, 1)")
        self.threshold = threshold

    def _encode(self, model: MicroModel, view: RobotView) -> int:
        posterior = robot_posterior(
            model, view.robot_id, view.observations, view.y_public, view.actions
        )
        if not posterior:
            return 0
        theta = theta_posterior(posterior)
        p0 = theta.get(model.theta_values[0], 0.0)
        return 1 if p0 > self.threshold else 0


class ConsequenceCodec(Codec):
    """Action-conditioned consequence summary.

    The robot predicts, for every public candidate, the probability of reaching the
    goal on the next transition. If the budget can index the candidate set it sends
    the identity of the best predicted candidate; otherwise it quantises the
    predicted consequence of the publicly first candidate. The message is about
    *what will happen if we act*, not about the hidden cause directly.
    """

    name = "consequence"

    def __init__(self, bits: int = 1) -> None:
        super().__init__()
        self.bits = bits

    def _encode(self, model: MicroModel, view: RobotView) -> int:
        posterior = robot_posterior(
            model, view.robot_id, view.observations, view.y_public, view.actions
        )
        if not posterior:
            return 0
        predicted: List[Tuple[str, float]] = []
        for action in view.candidates:
            success = 0.0
            for (x, theta), w in posterior.items():
                row = model.transition(x, theta, action)
                success += w * row.get(model.goal_x, 0.0)
            predicted.append((action, success))
        levels = 1 << self.bits
        if levels >= len(predicted):
            best_index = 0
            best_value = -1.0
            for index, (_action, success) in enumerate(predicted):
                if success > best_value + 1e-12:
                    best_value = success
                    best_index = index
            return best_index
        # Budget too small to name a candidate: summarise the public reference
        # candidate's predicted consequence.
        return _quantize(predicted[0][1], self.bits)


def default_codecs() -> List[Codec]:
    """The codec panel compared at matched ``B`` in the diagnostics."""

    return [
        NullCodec(),
        ObservationCodec(bits=1),
        PosteriorModeCodec(bits=1),
        ConsequenceCodec(bits=1),
        PosteriorThresholdCodec(threshold=0.8),
    ]
