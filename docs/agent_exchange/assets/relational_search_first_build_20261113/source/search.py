"""Separate expansion priorities from task seconds; all returned edges predicted."""
from copy import deepcopy
from dataclasses import dataclass
import time
from .contract import action_space, finite, validate_action


@dataclass(frozen=True)
class Budget:
    prediction_ticks: int = 1200
    calls: int = 40
    lookahead_ticks: int = 240
    seconds: float = 5.
    batch: int = 8
    fair_every: int = 4

    def __post_init__(self):
        for k in ('prediction_ticks', 'calls', 'lookahead_ticks', 'batch', 'fair_every'):
            if type(getattr(self, k)) is not int or getattr(self, k) < 1:
                raise ValueError(k)
        if not finite(self.seconds) or self.seconds <= 0:
            raise ValueError('seconds')


def choose(history, root_handle, model, score, leaf_seconds, budget=Budget()):
    """model.predict(handle, action) returns public observation, ticks, cost_seconds,
    terminal, success, handle. Opaque handles belong to an independent model.
    Exceptions charge the entire requested interval; partial reports charge actual
    ticks. Synthetic history is branch-local. No root-prior cost backup is used.
    """
    started = time.monotonic()
    nodes, pending, trace, results = [], [], [], []
    ticks = calls = 0

    def add(h, handle, path, cost, elapsed, terminal=False):
        node = dict(history=h, handle=handle, path=path, cost=cost, elapsed=elapsed,
                    actions=iter(action_space(h.events[-1]['observation'])), terminal=terminal,
                    exhausted=terminal, pending=0)
        nodes.append(node)
        return node

    root = add(history.fork(), root_handle, [], 0., 0)

    def refill(node):
        if node['exhausted'] or node['pending']:
            return
        for _ in range(budget.batch):
            try:
                a = next(node['actions'])
            except StopIteration:
                node['exhausted'] = True
                break
            if node['elapsed'] + a['ticks'] > budget.lookahead_ticks:
                continue
            priority = score(node['history'].fork(), deepcopy(a))
            if not finite(priority):
                raise ValueError('finite expansion priority required')
            pending.append((priority, node, a))
            node['pending'] += 1

    while calls < budget.calls and time.monotonic() - started < budget.seconds:
        for node in list(nodes):
            refill(node)
        affordable = [i for i, (_, _, a) in enumerate(pending) if ticks + a['ticks'] <= budget.prediction_ticks]
        if not affordable:
            # Oversized windows in one batch must not hide later affordable ones.
            if pending:
                for _, node, _ in pending:
                    node['pending'] -= 1
                pending.clear()
            if any(not n['exhausted'] and not n['pending'] for n in nodes):
                continue
            break
        fair = (calls + 1) % budget.fair_every == 0
        i = affordable[0] if fair else max(affordable, key=lambda j: pending[j][0])
        priority, node, a = pending.pop(i)
        node['pending'] -= 1
        calls += 1
        charged = a['ticks']
        event = dict(action=a, priority=priority, fair=fair, requested_ticks=a['ticks'],
                     parent_depth=len(node['path']))
        try:
            response = model.predict(node['handle'], deepcopy(a))
            actual = response['ticks']
            if type(actual) is not int or not 1 <= actual <= a['ticks']:
                raise ValueError('invalid predicted interval')
            if actual != a['ticks'] and not response['terminal']:
                raise ValueError('nonterminal prediction silently shortened command')
            if not finite(response['cost_seconds']) or response['cost_seconds'] < 0:
                raise ValueError('finite nonnegative task cost required')
            h = node['history'].fork()
            if response['observation']['tick'] != h.events[-1]['observation']['tick'] + actual:
                raise ValueError('prediction timestamp mismatch')
            h.append(response['observation'], a)
            charged = actual
            path = node['path'] + [a]
            cost = node['cost'] + response['cost_seconds']
            value = 0. if response['success'] else leaf_seconds(h.fork())
            if not finite(value) or value < 0:
                raise ValueError('shared finite nonnegative leaf seconds required')
            if response['success'] and not response['terminal']:
                raise ValueError('success requires terminal')
            # Failed terminal branches provide evidence, never a cheap solution.
            if not response['terminal'] or response['success']:
                results.append(dict(path=path, cost=cost + value, success=response['success']))
            add(h, response['handle'], path, cost, node['elapsed'] + actual, response['terminal'])
            event.update(status='predicted', actual_ticks=actual, task_seconds=cost,
                         leaf_seconds=value, success=response['success'])
        except Exception as exc:
            event.update(status='error', error=f'{type(exc).__name__}: {exc}')
        ticks += charged
        event['charged_ticks'] = charged
        trace.append(event)
    successful = [r for r in results if r['success']]
    pool = successful or results
    selected = min(pool, key=lambda r: r['cost']) if pool else None
    action = selected['path'][0] if selected else None
    if action is not None:
        validate_action(action, root['history'].events[-1]['observation'])
    return dict(action=action, status='predicted' if selected else 'needs_common_fallback',
                selected=selected, prediction_ticks=ticks, calls=calls, trace=trace,
                wall_seconds=time.monotonic() - started)
