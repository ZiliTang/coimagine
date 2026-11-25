"""Execute composed CAP code only in the existing read-only bubblewrap stream worker.

The simulator remains in the parent. Workers receive the current public event and
bounded explicit author memory. Real updates commit the resulting memory; event
previews discard it, as in the frozen single-CAP progress executor.
"""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source
from tools import cap_fault_curriculum_prefix as prefix


class IsolatedCAP:
    def __init__(self, broker):
        self.broker = broker
        self.memory = None
        self.proposal = None
        self.index = -1

    def advance(self, event):
        return self.broker.request(dict(op='advance', memory=self.memory, event=event))

    def update(self, history):
        if history.index != self.index + 1:
            raise ValueError('missed a committed history boundary')
        event = prefix.s.phase.cap.current_event(history)
        elapsed = None if history.previous_tick is None else event['observation']['tick']-history.previous_tick
        result = self.advance(dict(event, executed_ticks=elapsed))
        self.memory = result['memory']
        self.proposal = result['proposals'][0]
        self.index = history.index

    def propose(self, obs):
        return validate_action(self.proposal, obs)


def preview(member, obs, action, elapsed):
    result = member.advance(dict(observation=obs, action=action, executed_ticks=elapsed))
    try:
        return validate_action(result['proposals'][0], obs)
    except ValueError:
        return None


def initialize(source):
    global BROKER
    BROKER = StreamProgram(source)


def episode(job):
    return prefix.run_member(job, IsolatedCAP(BROKER), preview)


def scan_panel(source, fixtures, *, workers=12, trace=True):
    compile_source(source)
    jobs = [(key, fixture, trace) for key, fixture in fixtures.items()]
    if workers == 1:
        initialize(source)
        try:
            return dict(map(episode, jobs))
        finally:
            BROKER.close()
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork'),
                             initializer=initialize, initargs=(source,)) as pool:
        return dict(pool.map(episode, jobs))
