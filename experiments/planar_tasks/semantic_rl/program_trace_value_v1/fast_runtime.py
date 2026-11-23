"""Exact public-history caching and episode-local sensor read sharing.

Physics invalidates the sensor/contact caches before and after every advance.
CAP modules still receive separate owned inputs, and public history hashes retain
the original canonical JSON protocol. This module is opt-in for the new run.
"""
from contextlib import ExitStack
import gc
import json
from types import SimpleNamespace

from ..environment import Environment
from ..runner import rollout
from experiments.planar_tasks.relational_search.contract import public_observation
from ..memory_flow import MemoryHistory
from ..stream_history import StreamHistory
from . import full_worker as W
from .long_online import rollout_metrics
from .dataset import read, save
from pathlib import Path

ACTIVE = None


class AuditedCachedHistory(MemoryHistory):
    recording_identity = 'canonical_hash'

    def __init__(self):
        StreamHistory.__init__(self)
        self._event = None
        self.action_index = None

    def _append_projected(self, observation, action=None):
        # Environment.public() already projected this newly owned packet.
        self._event = self._commit_projected(observation, action, self._event)

    def append(self, observation, action=None):
        self._append_projected(public_observation(observation), action)

    def snapshot(self):
        return StreamHistory.snapshot(self)

    @classmethod
    def restore(cls, value):
        original = StreamHistory.restore(value)
        result = cls()
        result.__dict__.update(original.__dict__)
        result._event = json.loads(result._encoded)
        return result


class ScopedEnvironment(Environment):
    def __init__(self, *args, **kwargs):
        global ACTIVE
        super().__init__(*args, **kwargs)
        self._read_transaction = ExitStack()
        self._read_transaction.enter_context(self.env.env.backend.contact_snapshot(True))
        self._read_transaction.enter_context(self.env.lidar.read_scope(True))
        ACTIVE = self

    def close_reads(self):
        self._read_transaction.close()


def single_write_dump(value, stream, **kwargs):
    # Same JSON document; eliminate hundreds of thousands of tiny gzip writes.
    stream.write(json.dumps(value, **kwargs))


def scoped_cap_rollout(fixture, source, *, features, **kwargs):
    """Use the same synchronous read scopes for single-CAP worker episodes."""
    global ACTIVE
    previous = ACTIVE
    with ExitStack() as stack:
        def make_environment(*args, **options):
            if features.runtime == 'typed_stream':
                options['history_factory'] = AuditedCachedHistory
            env = ScopedEnvironment(*args, **options)
            stack.callback(env.close_reads)
            return env

        try:
            return rollout(fixture, source, features=features,
                           environment_factory=make_environment, **kwargs)
        finally:
            ACTIVE = previous


def initialize(bundle, priority):
    W.initialize(bundle, priority)
    # Dedicated rollout processes churn through short-lived observation trees.
    # Keep cyclic GC enabled, but amortize its scans over larger batches.
    young, middle, old = gc.get_threshold()
    if young:
        gc.set_threshold(max(young, 50000), middle, old)
    W.Environment = ScopedEnvironment
    W.StreamHistory = AuditedCachedHistory
    W.rollout = scoped_cap_rollout
    W.json = SimpleNamespace(dump=single_write_dump)


def run(job):
    global ACTIVE
    try:
        result = W.run(job)
        if not job.get('cap') and result['complete']:
            path = Path(job['directory'])/'behavior_metrics.json'
            if not path.exists():
                save(path, rollout_metrics(Path(job['directory'])/'result.json.gz'))
            result = dict(result, behavior=read(path))
        return result
    finally:
        if ACTIVE is not None:
            ACTIVE.close_reads()
            ACTIVE = None
