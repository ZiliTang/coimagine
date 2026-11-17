"""Sandboxed CAP features on prefix choices, with explicit legacy adapter."""
from copy import deepcopy
from contextlib import contextmanager
import math
import time
import numpy as np
from experiments.planar_tasks.action_semantics.program import Program, action_encoding
from experiments.planar_tasks.relational_search.contract import canonical, digest
from experiments.planar_tasks.relational_search.pipeline_semantics import embedding
from .actions import FIELDS


class Features:
    def __init__(self, source, *, schema=None, fast_embedding=False, runtime="legacy", action_protocol=None,
                 execution_backend="isolated", trusted_source_sha256=None):
        from .typed_compiler import source_api, STREAM_API
        api=source_api(source)
        if api:runtime="typed_stream" if api==STREAM_API else "typed_batch"
        if runtime=="typed_stream" and api!=STREAM_API:raise ValueError("SEMANTIC_API must be typed-stream-v1")
        if execution_backend not in ('isolated','trusted_inprocess'):raise ValueError('unknown semantic execution backend')
        if execution_backend=='trusted_inprocess' and runtime!='typed_stream':raise ValueError('trusted backend requires typed_stream')
        self.execution_backend=execution_backend
        self.action_protocol = deepcopy(action_protocol)
        self.source = source
        self.sha256 = digest(source)
        self.calls = 0
        self.names = None
        self.schema = schema
        self.fast_embedding = fast_embedding
        if runtime not in ('legacy', 'cached_fork', 'typed_batch', 'typed_stream'):
            raise ValueError('unknown feature runtime')
        self.runtime = runtime
        self.worker = None
        if runtime in ('cached_fork','typed_batch'):
            from .fast_features import FastProgram
            self.worker = FastProgram(source, max_names=len(schema.slots) if schema else 24,typed=runtime=='typed_batch')
        self.semantic_protocol=None
        if runtime=='typed_stream':
            from .stream_features import StreamProgram
            from .schema import Schema
            if execution_backend=='trusted_inprocess':
                from .trusted_native import TrustedStreamProgram
                self.worker=TrustedStreamProgram(source,trusted_source_sha256=trusted_source_sha256)
            else:self.worker=StreamProgram(source)
            self.semantic_protocol=self.worker.protocol
            if self.schema is None:self.schema=Schema.create(self.worker.names)
        self.timing = dict(sandbox_seconds=0., embedding_seconds=0., panels=0)

    @contextmanager
    def decision(self, history):
        # Host history cannot change during synchronous policy selection. No
        # authored function has a reference into it. Outside this scope caches
        # remain content-checked, including in-place observation mutations.
        if self.worker:
            started=time.monotonic()
            before=self.worker.calls
            try:self.worker.prepare(history)
            finally:
                if self.runtime=="typed_stream":self.calls+=self.worker.calls-before
            self.timing['sandbox_seconds'] += time.monotonic()-started
            self.worker.prepared_for = history
        try:
            yield
        finally:
            if self.worker:self.worker.prepared_for = None

    def new_history(self):
        if self.runtime=='typed_stream':
            from .stream_history import StreamHistory
            return StreamHistory()
        from experiments.planar_tasks.relational_search.contract import History
        return History()

    def snapshot(self):
        if self.runtime!='typed_stream':raise ValueError('stream runtime required')
        return self.worker.snapshot()

    def restore(self,packet,history):
        if self.runtime!='typed_stream':raise ValueError('stream runtime required')
        self.worker.restore(packet,history)

    def close(self):
        if self.worker is not None:
            self.worker.close()

    def recommendation(self, history):
        if self.runtime not in ('typed_batch','typed_stream'):raise ValueError('typed recommendation interface required')
        with self.decision(history):
            if self.runtime=='typed_stream':return self.worker.recommendation()
            before=self.worker.calls
            try:return self.worker.evaluate(history,[],{})
            finally:self.calls+=self.worker.calls-before

    def panel(self, history, tree, prefix):
        choices = tree.choices(prefix)
        representatives = [tree.complete(p) for p in choices]
        # Optional author API; complete-action semantics remain a labelled proxy.
        mapping = {canonical(a): p for a, p in zip(representatives, choices)}
        if self.worker is None:
            adapter = '\nimport json\n_PREFIX_MAP = ' + repr(mapping) + '''
_COMPLETE_FEATURES = semantic_features
def semantic_features(m, o, a, proposals):
    if 'prefix_semantic_features' in globals():
        prefix = _PREFIX_MAP.get(json.dumps(a,sort_keys=True,separators=(',',':')), a)
        return prefix_semantic_features(m, o, prefix, proposals)
    return _COMPLETE_FEATURES(m, o, a, proposals)
'''
            program = Program(self.source + adapter, max_names=len(self.schema.slots) if self.schema else 24)
        else:
            program = self.worker
        started=time.monotonic()
        # Stream replies have passed exact action-schema validation. Use the
        # immutable fields for host lookup; retain canonical JSON at the
        # worker/protocol boundary and for all legacy runtimes.
        key = (lambda a: (a['target_id'], a['a0'], a['a1'], a['ticks'])) if self.runtime=='typed_stream' else canonical
        values = {}; proposals = None; invalid = None
        for start in range(0, len(representatives), 128):
            before=program.calls
            try:
                result = (program.evaluate(history, representatives[start:start+128], mapping) if self.worker
                          else program.evaluate(history, representatives[start:start+128], generic=False))
            finally:
                # Failed sandbox work is still an attempted feature batch.
                self.calls += program.calls-before
            if self.names is None:
                self.names = result['names']
            if self.names != result['names']:
                raise ValueError('semantic schema changed')
            if proposals is not None and proposals != result['proposals']:
                raise ValueError('non-deterministic proposals')
            proposals, invalid = result['proposals'], result['invalid_proposals']
            values.update({key(a): v for a, v in zip(result['actions'], result['values'])})
        self.timing['sandbox_seconds'] += time.monotonic()-started
        started=time.monotonic()
        semantic, raw = [], []
        o = tree.observation
        embeddings=([self.worker.geometry(a) for a in representatives] if self.runtime=='typed_stream' else
                    embedding_batch(history,representatives) if self.fast_embedding else [embedding(history,a) for a in representatives])
        window=self.worker.vector_window() if self.runtime=='typed_stream' else []
        for p, a, raw_embedding in zip(choices, representatives, embeddings):
            # Unchosen coordinates are masked in the explicit encoding.
            enc = action_encoding(o, a)
            if 'ticks' not in p:
                enc[0] = 0.
            if 'a0' not in p:
                enc[3:82] = [0.] * 79
            if 'a1' not in p:
                enc[82:] = [0.] * 79
            match = any(all(q[k] == v for k, v in p.items()) for q in proposals)
            flags = [float(k in p) for k in FIELDS]
            row=values[key(a)]
            if self.schema:row=self.schema.encode(self.names,row)
            semantic.append(row + window + [float(match)] + flags + enc)
            raw.append(raw_embedding + flags + enc)
        self.timing['embedding_seconds'] += time.monotonic()-started
        self.timing['panels'] += 1
        panel = dict(prefix=deepcopy(prefix), choices=choices, representatives=representatives,
                    semantic=semantic, raw=raw, proposals=proposals,
                    invalid_proposals=invalid, semantic_sha256=self.sha256,
                    schema=self.schema.to_dict() if self.schema else None,
                    history_hash=self.worker.history_hash if self.worker else history.sha256)
        if self.semantic_protocol is not None:
            panel['semantic_protocol']=deepcopy(self.semantic_protocol)
            panel['semantic_window']=deepcopy(self.worker.frames)
        if self.action_protocol is not None:panel["action_protocol"]=deepcopy(self.action_protocol)
        return panel


def embedding_batch(history,actions):
    """Reuse target-dependent history geometry, preserving original arithmetic."""
    cache={};rows=[]
    for action in actions:
        target=action['target_id']
        if target not in cache:cache[target]=embedding(history,action)
        row=list(cache[target]);vectors=[]
        for start in (0,47):
            v=np.zeros(2)
            for a in (action['a0'],action['a1']):
                if start<=a<start+32:
                    direction,speed=divmod(a-start,4);angle=direction*math.pi/4
                    v+=.1*(speed+1)*np.array([math.cos(angle),math.sin(angle)])
            vectors.extend(v.tolist())
        row[0]=action['ticks']*.05;row[2:6]=vectors;rows.append(row)
    return rows
