"""A fixed-delay observation queue with arrival timestamps.

The previous round's queue kept the frames that had *not* arrived yet and dropped the ones that
had, so it replayed the oldest observation forever. This queue does the opposite: it delivers
the newest frame whose arrival time is at least the configured delay in the past, keeps the
frames still in flight, and discards only history that has been superseded. Before anything has
arrived it reports **unavailable**; it never falls back to a zero-delay fresh observation, and it
never silently keeps one field fresh while delaying another.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class DelayFrame:
    """One observation plus both of its timestamps.

    ``sample_time`` is when the sensor sampled the world; ``arrival_time`` is when the frame
    became available to this robot. They are different quantities and the settle estimator uses
    the former.
    """

    sample_time: float
    arrival_time: float
    observation: Any
    message: Any = None


@dataclass
class FixedDelayQueue:
    delay: float
    frames: List[DelayFrame] = field(default_factory=list)
    delivered: Optional[DelayFrame] = None
    max_queue: int = 0
    unavailable_periods: int = 0
    delivered_periods: int = 0
    selected_sample_times: List[float] = field(default_factory=list)
    ages: List[float] = field(default_factory=list)
    last_sample_time: Optional[float] = None

    def push(self, frame: DelayFrame) -> None:
        self.frames.append(frame)
        self.max_queue = max(self.max_queue, len(self.frames))

    def deliver(self, now: float) -> Optional[DelayFrame]:
        """Return the newest frame with ``now - arrival_time >= delay``, or ``None``."""

        if self.delay <= 0.0:
            if not self.frames:
                self.unavailable_periods += 1
                return None
            chosen = self.frames[-1]
            self.frames = [chosen]
        else:
            eligible = [frame for frame in self.frames if now - frame.arrival_time >= self.delay]
            if not eligible:
                self.unavailable_periods += 1
                return None
            chosen = eligible[-1]
            self.frames = [frame for frame in self.frames if frame.arrival_time >= chosen.arrival_time]
        self.delivered = chosen
        self.delivered_periods += 1
        self.selected_sample_times.append(chosen.sample_time)
        self.ages.append(now - chosen.arrival_time)
        self.last_sample_time = chosen.sample_time
        return chosen

    def stats(self) -> Dict[str, object]:
        return {
            "delay": self.delay,
            "delivered_periods": self.delivered_periods,
            "unavailable_periods": self.unavailable_periods,
            "max_queue": self.max_queue,
            "max_age": max(self.ages) if self.ages else None,
            "min_age": min(self.ages) if self.ages else None,
            "selection_advances": _monotone_steps(self.selected_sample_times),
            "selected_head": self.selected_sample_times[:8],
            "last_sample_time": self.last_sample_time,
        }


def _monotone_steps(values: List[float]) -> bool:
    return all(b > a for a, b in zip(values, values[1:]) if b != a) and all(
        b >= a for a, b in zip(values, values[1:])
    )
