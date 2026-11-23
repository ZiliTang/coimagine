"""Bounded closed-loop policy switching on held-out development roots."""
import gzip
import time
from pathlib import Path

import numpy as np
import torch

from ..environment import Environment, EventConfig
from ..stream_history import StreamHistory
from ..memory_flow import current_event
from ..trusted_native import owned_copy
from .dataset import flatten, read, save
from .model import load_model
from .runtime import CAPS, TraceLibrary, entries

_LIBRARY = None
_MODELS = None


def initialize(bundle, out):
    global _LIBRARY, _MODELS
    torch.set_num_threads(1)
    _LIBRARY = TraceLibrary(entries(bundle))
    _MODELS = {arm: load_model(Path(out)/'fit'/arm/'model.pt') for arm in ('GA', 'LOCAL', 'GLOBAL')}


def evaluate(job):
    out, arm, case = job
    out = Path(out)
    directory = out/'evaluation'/arm/case['case_id']
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'summary.json').exists():
        return read(directory/'summary.json')
    start = time.monotonic()
    model = _MODELS[arm]
    environment = Environment(case['fixture'], EventConfig(mode='duration'),
                              fast_feedback=True, history_factory=StreamHistory, record_details=True)
    history = environment.history
    rows = [dict(current_event(history), history_hash=history.sha256)]
    decisions = []
    stop = 'decision_budget'
    error = None
    trace_seconds = network_seconds = 0.
    invalid = fallback = switches = 0
    previous_action = None
    try:
        for index in range(300):
            if (out/'STOP').exists() or time.monotonic()-start > 1800:
                stop = 'infrastructure_budget'
                break
            packet = _LIBRARY.advance(history)
            trace_seconds += packet['seconds']
            invalid += sum(not v for v in packet['legal'])
            fallback += int(packet['fallback'])
            network_start = time.monotonic()
            with torch.no_grad():
                values = model.all_values(flatten(packet)).numpy()
            network_seconds += time.monotonic()-network_start
            available = np.asarray(packet['legal'])
            if available.any():
                expert = int(np.argmax(np.where(available, values, -np.inf)))
                action = packet['proposals'][expert]
            else:
                expert = None
                action = packet['pool'][0]
            if previous_action is not None and previous_action != action:
                switches += 1
            previous_action = owned_copy(action)
            event = environment.step(action, remaining_ticks=1800-environment.env.total_periods)
            decisions.append(dict(index=index, cap=None if expert is None else CAPS[expert],
                                  values=values.tolist(), proposals=packet['proposals'],
                                  legal=packet['legal'], candidates=packet['pool'], sources=packet['sources'],
                                  action=action, event=event))
            rows.append(dict(current_event(history), history_hash=history.sha256))
            if environment.env.termination is not None:
                stop = 'terminal'
                break
            if environment.env.total_periods >= 1800:
                stop = 'tick_budget'
                break
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        stop = 'tool_error'
    from coimagine.random_start.lidar_native import runtime_info
    summary = dict(arm=arm, case_id=case['case_id'], success=bool(environment.env.success),
                   complete=stop in ('terminal', 'tick_budget', 'decision_budget'),
                   stop=stop, termination=environment.env.termination, error=error,
                   ticks=environment.env.total_periods, decisions=environment.decision_count,
                   seconds=time.monotonic()-start, trace_seconds=trace_seconds, network_seconds=network_seconds,
                   invalid_proposals=invalid, fallback_boundaries=fallback, action_switches=switches,
                   analytic_lidar_backend=runtime_info())
    result = dict(summary=summary, public_rows=rows, decisions=decisions,
                  events=environment.events, replay=environment.replay())
    import json
    with gzip.open(directory/'result.json.gz', 'wt', compresslevel=1) as f:
        json.dump(result, f, allow_nan=False)
    save(directory/'summary.json', summary)
    return summary
