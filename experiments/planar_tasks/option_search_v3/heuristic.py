"""Weak neutral reference, not an optimized transport policy or task-stage script."""
import math


def evaluate_state(node,memory):
    state=node['state']
    if state is None:return 0.
    o=state['object'];g=node['goal']
    return math.hypot(o['x']-g[0],o['y']-g[1])/.3


def rank_actions(parent,candidates,memory):
    # Explicitly uninformative prior: do not invent equal action effects then
    # penalize durations. Stable candidate order breaks ties; diagnostics flag it.
    return dict(costs=[evaluate_state(parent,memory)]*len(candidates),proposals=[])
