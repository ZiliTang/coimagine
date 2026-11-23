"""Real online transitions and frozen-model/CAP evaluation, with durable replays."""
import gzip
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..environment import Environment, EventConfig
from ..stream_history import StreamHistory
from ..memory_flow import current_event
from ..features import Features
from ..runner import rollout
from ..scale import protocol
from experiments.planar_tasks.relational_search.contract import digest
from .dataset import read, save, flatten
from .runtime import TraceLibrary, entries
from .full_data import candidate_arrays
from .q_model import load_checkpoint

LIBRARY = None
MODELS = {}
CAP_FEATURES = {}
CAP_LIBRARY = {}


def initialize(bundle, priority_bundle):
    global LIBRARY, CAP_LIBRARY
    torch.set_num_threads(1)
    LIBRARY = TraceLibrary(entries(bundle))
    old = Path(priority_bundle)
    CAP_LIBRARY = {r['name']: (old/r['policy']).read_text() for r in read(old/'library.json')}


def model_at(path):
    if path not in MODELS:
        MODELS.clear()
        MODELS[path] = load_checkpoint(path)[0]
    return MODELS[path]


def run(job):
    directory = Path(job['directory'])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'summary.json').exists():
        return read(directory/'summary.json')
    case = job['case']
    start = time.monotonic()
    if job.get('cap'):
        name = job['cap']
        if name not in CAP_FEATURES:
            source = CAP_LIBRARY[name]
            CAP_FEATURES[name] = Features(source, fast_embedding=True, action_protocol=protocol(EventConfig(mode='duration')),
                                         execution_backend='trusted_inprocess', trusted_source_sha256=digest(source))
        result, _ = rollout(case['fixture'], CAP_LIBRARY[name], features=CAP_FEATURES[name],
                            config=EventConfig(mode='duration'), max_ticks=1800, max_decisions=300,
                            wall_seconds=1800, deterministic=True, fast_feedback=True, record_details=True)
        summary = result['summary']
        summary.update(condition=job['condition'], case_id=case['case_id'],
                       complete=summary['stop'] in ('terminal', 'decision_budget', 'execution_budget') and not summary['error'])
        with gzip.open(directory/'result.json.gz', 'wt', compresslevel=1) as f:
            json.dump(result, f, allow_nan=False)
        save(directory/'summary.json', summary)
        return summary
    model = model_at(job['checkpoint'])
    rng = np.random.default_rng(job.get('action_seed', 17))
    epsilon = job.get('epsilon', 0.)
    record_transitions = job.get('online', False)
    environment = Environment(case['fixture'], EventConfig(mode='duration'), fast_feedback=True,
                              history_factory=StreamHistory, record_details=True)
    h = environment.history
    public = [dict(current_event(h), history_hash=h.sha256)]
    decisions = []
    values = {k: [] for k in ('x', 'candidates', 'sources', 'valid', 'chosen', 'duration')}
    stop, error = 'decision_budget', None
    trace_seconds = network_seconds = 0.
    invalid = fallback = 0
    try:
        for index in range(300):
            if time.monotonic()-start > 1800:
                stop = 'infrastructure_budget'
                break
            packet = LIBRARY.advance(h)
            trace_seconds += packet['seconds']
            invalid += sum(not v for v in packet['legal'])
            fallback += int(packet['fallback'])
            raw = flatten(packet)
            candidates, sources, valid = candidate_arrays(packet, LIBRARY)
            began = time.monotonic()
            with torch.no_grad():
                scores = model.values(torch.tensor(raw[None]), torch.tensor(candidates[None]), torch.tensor(sources[None]))[0].numpy()
            network_seconds += time.monotonic()-began
            greedy = int(np.argmax(np.where(valid, scores, -np.inf)))
            chosen = int(rng.choice(np.flatnonzero(valid))) if rng.random() < epsilon else greedy
            action = packet['pool'][chosen]
            event = environment.step(action, remaining_ticks=1800-environment.env.total_periods)
            if hasattr(model, 'executed'):
                model.executed(sources[chosen])
            if record_transitions:
                for key, value in [('x', raw), ('candidates', candidates), ('sources', sources), ('valid', valid),
                                   ('chosen', chosen), ('duration', event['periods'])]:
                    values[key].append(value)
            decisions.append(dict(index=index, candidates=packet['pool'], sources=packet['sources'],
                                  scores=scores[:len(packet['pool'])].tolist(), chosen=chosen, greedy=greedy,
                                  chosen_value=float(scores[chosen]), epsilon=epsilon, action=action, event=event))
            public.append(dict(current_event(h), history_hash=h.sha256))
            if environment.env.termination is not None:
                stop = 'terminal'
                break
            if environment.env.total_periods >= 1800:
                stop = 'tick_budget'
                break
    except Exception as exc:
        error = dict(type=type(exc).__name__, message=str(exc))
        stop = 'tool_error'
    complete = stop in ('terminal', 'tick_budget', 'decision_budget') and error is None
    from coimagine.random_start.lidar_native import runtime_info
    summary = dict(condition=job['condition'], case_id=case['case_id'], complete=complete,
                   success=bool(environment.env.success), error=error, stop=stop, termination=environment.env.termination,
                   ticks=environment.env.total_periods, decisions=environment.decision_count,
                   seconds=time.monotonic()-start, trace_seconds=trace_seconds, network_seconds=network_seconds,
                   invalid_proposals=invalid, fallback_boundaries=fallback,
                   initial_value=None if not decisions else decisions[0]['chosen_value'],
                   epsilon=epsilon, action_seed=job.get('action_seed'), analytic_lidar_backend=runtime_info())
    result = dict(summary=summary, public_rows=public, decisions=decisions, events=environment.events,
                  replay=environment.replay(), checkpoint=job['checkpoint'])
    with gzip.open(directory/'result.json.gz', 'wt', compresslevel=1) as f:
        json.dump(result, f, allow_nan=False)
    if record_transitions and complete:
        n = len(values['x'])
        arrays = {k: np.asarray(v) for k, v in values.items()}
        arrays.update(y=np.full(n, float(summary['success']), dtype=np.float32),
                      cap=np.full(n, -1, dtype=np.int64), root=np.full(n, job['root_number'], dtype=np.int64),
                      boundary=np.arange(n, dtype=np.int32), reward=np.zeros(n, dtype=np.float32),
                      done=np.zeros(n, dtype=np.bool_))
        arrays['reward'][-1] = float(summary['success'])
        arrays['done'][-1] = True
        np.savez_compressed(directory/'transitions.npz', **arrays)
        summary['transition_shard'] = dict(path=str(directory/'transitions.npz'), boundaries=n,
                                           case_id=case['case_id'], success=summary['success'])
    save(directory/'summary.json', summary)
    return summary
