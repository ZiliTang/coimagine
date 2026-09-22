from __future__ import annotations

import sys
from dataclasses import dataclass
from time import perf_counter
from typing import TextIO


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "--:--"
    seconds = int(round(seconds))
    if seconds < 3600:
        minutes, secs = divmod(seconds, 60)
        return f"{minutes:02d}:{secs:02d}"
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}"


@dataclass
class ProgressSnapshot:
    completed: int
    total: int
    elapsed_seconds: float
    rate_per_second: float
    eta_seconds: float | None


class ProgressReporter:
    """Small dependency-free terminal progress reporter.

    Interactive terminals receive a single updating progress bar. Redirected
    output receives periodic newline-delimited records so long experiments are
    still observable in logs and CI jobs.
    """

    def __init__(
        self,
        total: int,
        *,
        label: str,
        unit: str = "case",
        enabled: bool = True,
        stream: TextIO | None = None,
        line_interval_seconds: float = 5.0,
        tty: bool | None = None,
        bar_width: int = 26,
    ) -> None:
        if total < 0:
            raise ValueError("total cannot be negative")
        if line_interval_seconds <= 0:
            raise ValueError("line_interval_seconds must be positive")
        self.total = int(total)
        self.label = label
        self.unit = unit
        self.enabled = bool(enabled)
        self.stream = stream if stream is not None else sys.stderr
        self.line_interval_seconds = float(line_interval_seconds)
        self.tty = bool(self.stream.isatty()) if tty is None else bool(tty)
        self.bar_width = max(10, int(bar_width))
        self.started = perf_counter()
        self.completed = 0
        self._last_emit_time = self.started - self.line_interval_seconds
        self._last_percent_bucket = -1
        self._closed = False
        self._last_emitted_completed = -1
        if self.enabled:
            self._emit(force=True)

    def snapshot(self) -> ProgressSnapshot:
        now = perf_counter()
        elapsed = max(0.0, now - self.started)
        rate = self.completed / elapsed if elapsed > 1e-9 else 0.0
        remaining = max(0, self.total - self.completed)
        eta = remaining / rate if rate > 1e-12 else None
        return ProgressSnapshot(
            completed=self.completed,
            total=self.total,
            elapsed_seconds=elapsed,
            rate_per_second=rate,
            eta_seconds=eta,
        )

    def update(self, advance: int = 1, *, detail: str | None = None) -> None:
        if self._closed:
            return
        self.completed = min(self.total, self.completed + max(0, int(advance)))
        self._emit(detail=detail)

    def set(self, completed: int, *, detail: str | None = None) -> None:
        if self._closed:
            return
        self.completed = min(self.total, max(0, int(completed)))
        self._emit(detail=detail)

    def close(self, *, status: str = "done", detail: str | None = None) -> None:
        if self._closed:
            return
        if status == "done":
            self.completed = self.total
        already_emitted_final = (
            status == "done"
            and self.completed == self.total
            and self._last_emitted_completed == self.total
        )
        if not already_emitted_final:
            self._emit(force=True, detail=detail, status=status)
        if self.enabled and self.tty:
            self.stream.write("\n")
            self.stream.flush()
        self._closed = True

    def _render(self, *, detail: str | None, status: str | None) -> str:
        snap = self.snapshot()
        if snap.total:
            fraction = snap.completed / snap.total
            percent = 100.0 * fraction
        else:
            fraction = 1.0
            percent = 100.0
        rate = f"{snap.rate_per_second:5.2f} {self.unit}/s" if snap.rate_per_second else f"  -- {self.unit}/s"
        elapsed = _format_duration(snap.elapsed_seconds)
        eta = _format_duration(snap.eta_seconds)
        suffix = f" | elapsed {elapsed} | ETA {eta} | {rate}"
        if detail:
            suffix += f" | {detail}"
        if status and status != "done":
            suffix += f" | {status.upper()}"
        if self.tty:
            filled = min(self.bar_width, int(round(self.bar_width * fraction)))
            bar = "█" * filled + "░" * (self.bar_width - filled)
            return (
                f"[{self.label}] [{bar}] {snap.completed}/{snap.total} "
                f"{self.unit}s ({percent:5.1f}%){suffix}"
            )
        return (
            f"[{self.label}] {snap.completed}/{snap.total} {self.unit}s "
            f"({percent:5.1f}%){suffix}"
        )

    def _emit(
        self,
        *,
        force: bool = False,
        detail: str | None = None,
        status: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        now = perf_counter()
        percent_bucket = (
            int(100 * self.completed / self.total) // 5 if self.total else 20
        )
        should_emit = force or self.tty
        if not self.tty:
            should_emit = should_emit or (
                now - self._last_emit_time >= self.line_interval_seconds
                or percent_bucket > self._last_percent_bucket
                or self.completed >= self.total
            )
        if not should_emit:
            return
        text = self._render(detail=detail, status=status)
        if self.tty:
            self.stream.write("\r\x1b[2K" + text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
        self._last_emit_time = now
        self._last_percent_bucket = percent_bucket
        self._last_emitted_completed = self.completed


def progress_message(message: str, *, enabled: bool = True, stream: TextIO | None = None) -> None:
    if not enabled:
        return
    target = stream if stream is not None else sys.stderr
    target.write(message.rstrip() + "\n")
    target.flush()
