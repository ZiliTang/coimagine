"""Same actual public prefix for every expert; bound advice and trace channels."""
import json
import time
from pathlib import Path

import numpy as np

from ..memory_flow import current_event
from ..relational_transfer_features import RelativeGeometry, action_encoding
from ..trusted_native import owned_copy
from experiments.planar_tasks.relational_search.contract import validate_action
from .instrumentation import instrument, load_native
from .pose_kernel import configure as configure_pose

CAPS = ['CAP-RR-' + n for n in ('009', '003', '001', '010', '006', '007', '002')]


def entries(bundle):
    bundle = Path(bundle)
    rows = {r['cap_id']: r for r in json.loads((bundle / 'library.json').read_text())}
    return [dict(name=n, source=(bundle / rows[n]['path']).read_text()) for n in CAPS]


class TraceLibrary:
    def __init__(self, library, parity=False):
        self.library = library
        self.modules = []
        self.schemas = []
        self.originals = []
        for entry in library:
            source, schema = instrument(entry['source'])
            module = load_native(source)[0]
            configure_pose(module, entry['source'], generated=source, schema=schema)
            self.modules.append(module)
            self.schemas.append(schema)
            if parity:
                self.originals.append(load_native(entry['source'])[0])
        self.reset()

    def reset(self):
        self.memory = [None] * len(self.modules)
        self.original_memory = [None] * len(self.modules)
        self.geometry = RelativeGeometry(8)
        self.episode = None
        self.index = -1
        self.parity_boundaries = 0

    def advance(self, history):
        if self.episode != history.episode:
            if history.index != 0:
                raise ValueError('full actual prefix required')
            self.reset()
            self.episode = history.episode
        if history.index == self.index:
            return self.packet
        if history.index != self.index + 1:
            raise ValueError('skipped public boundary')
        start = time.perf_counter()
        event = current_event(history)
        observation = event['observation']
        self.geometry.advance(history)
        traces = []
        proposals = []
        legal = []
        all_advice = []
        for i, module in enumerate(self.modules):
            trace = [float('nan')] * self.schemas[i]['width']
            actual = dict(event, executed_ticks=None if history.previous_tick is None
                          else observation['tick'] - history.previous_tick)
            memory = self.memory[i]
            if memory is None:
                memory = module.initial_memory()
            memory = module.update_memory(memory, owned_copy(actual), _tb=trace)
            context = module.prepare_context(owned_copy(memory), owned_copy(observation), _tb=trace)
            self.memory[i] = memory
            if self.originals:
                original = self.originals[i]
                old = self.original_memory[i]
                if old is None:
                    old = original.initial_memory()
                old = original.update_memory(old, owned_copy(actual))
                original_context = original.prepare_context(owned_copy(old), owned_copy(observation))
                self.original_memory[i] = old
                if json.dumps([old, original_context], sort_keys=True) != json.dumps([memory, context], sort_keys=True):
                    raise ValueError('instrumentation changed memory/context: ' + CAPS[i])
            proposal = context['proposals'][0] if context['proposals'] else None
            valid = proposal is not None
            if valid:
                try:
                    validate_action(proposal, observation)
                except (ValueError, TypeError, KeyError):
                    valid = False
            proposals.append(proposal)
            legal.append(valid)
            traces.append(np.array(trace, dtype=np.float32))
            # Invalid advice is explicit; never replaced with outcome information.
            if valid:
                a = np.concatenate([action_encoding(proposal, observation['goal']['target_id'], self.geometry.theta),
                                    self.geometry.candidate_geometry(proposal), [1.]])
            else:
                a = np.full(86, np.nan, dtype=np.float32)
                a[-1] = 0.
            all_advice.append(a)
        pool = []
        sources = []
        for i, (action, valid) in enumerate(zip(proposals, legal)):
            if valid:
                if action not in pool:
                    pool.append(action)
                    sources.append([])
                sources[pool.index(action)].append(i)
        fallback = not pool
        if fallback:
            action = dict(target_id=None, a0=44, a1=44, ticks=1)
            validate_action(action, observation)
            pool = [action]
            sources = [[]]
        self.index = history.index
        self.parity_boundaries += int(bool(self.originals))
        self.packet = dict(g=np.asarray(self.geometry.vector, dtype=np.float32),
                           advice=np.asarray(all_advice, dtype=np.float32),
                           trace=np.concatenate(traces),
                           budget=np.asarray([(1800-observation['tick'])/1800,
                                              (300-observation['decision'])/300], dtype=np.float32),
                           proposals=proposals, legal=legal, pool=pool, sources=sources,
                           fallback=fallback, seconds=time.perf_counter()-start)
        return self.packet
