"""Bounded executable feature adapter. Functions run in the caller's process.

This is a trusted-function mechanism interface; untrusted authored functions
must later run through the authoring worker sandbox, not on the host.
"""
from copy import deepcopy
from .contract import canonical, finite


class FeatureProgram:
    def __init__(self, initial_memory, update_memory, semantic_features, names):
        if not 1 <= len(names) <= 16 or len(set(names)) != len(names) or not all(isinstance(k, str) for k in names):
            raise ValueError('one to sixteen fixed semantic names')
        self.initial = initial_memory
        self.update = update_memory
        self.query = semantic_features
        self.names = tuple(names)

    @staticmethod
    def bounded(memory):
        if not isinstance(memory, dict) or len(canonical(memory).encode()) > 8192:
            raise ValueError('memory must be a JSON object of at most 8192 bytes')
        return deepcopy(memory)

    def evaluate(self, history, action):
        if not history.events:
            raise ValueError('nonempty legal history required')
        memory = self.bounded(self.initial())
        previous = None
        for event in history.events:
            observation = deepcopy(event['observation'])
            # The first retained sample initializes the shared finite window.
            update_event = dict(observation=observation,
                action=None if previous is None else deepcopy(event['action']),
                executed_ticks=None if previous is None else observation['tick']-previous)
            memory = self.bounded(self.update(memory, update_event))
            previous = observation['tick']
        values = self.query(deepcopy(memory), deepcopy(history.events[-1]['observation']), deepcopy(action))
        if not isinstance(values, dict) or set(values) != set(self.names):
            raise ValueError('semantic feature schema changed')
        if any(v is not None and type(v) is not bool and not finite(v) for v in values.values()):
            raise ValueError('scalar, boolean or unknown semantic values required')
        return [None if values[k] is None else float(values[k]) for k in self.names]
