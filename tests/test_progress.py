from __future__ import annotations

from io import StringIO

from coimagine.reporting.progress import ProgressReporter, progress_message


def test_progress_reporter_emits_log_records() -> None:
    stream = StringIO()
    reporter = ProgressReporter(
        2,
        label="unit",
        unit="case",
        stream=stream,
        tty=False,
        line_interval_seconds=100.0,
    )
    reporter.update(detail="3 episodes")
    reporter.update(detail="6 episodes")
    reporter.close(detail="6 episodes")
    output = stream.getvalue()
    assert "[unit]" in output
    assert "0/2 cases" in output
    assert "2/2 cases" in output
    assert "6 episodes" in output
    assert "ETA" in output


def test_progress_reporter_can_be_disabled() -> None:
    stream = StringIO()
    reporter = ProgressReporter(1, label="quiet", stream=stream, enabled=False)
    reporter.update()
    reporter.close()
    progress_message("hidden", enabled=False, stream=stream)
    assert stream.getvalue() == ""
