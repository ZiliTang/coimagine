"""Evaluate authored option/continuation contracts and select a feasible root action.

The contract defines units, effect obligations, state transfers, and successor
requirements. The host supplies public metric intervals and physical effect bounds.
Candidate order is the caller's preference, independent of verification status.
"""

from .guarded_options import compile_policy


def check_candidate(contract, candidate, public_cell, effect_bounds, successor_requires,
                    *, quality='empirical', allow_empirical=False):
    first, second = candidate
    requires = {**contract['continuation_requires'], **successor_requires}
    if set(successor_requires) & set(contract['continuation_requires']):
        raise ValueError('route-specific requirements cannot override common constraints')
    cell_id = 'current'
    profile = dict(units=contract['units'], cells={cell_id: public_cell}, options={
        first: dict(requires=contract['requires'], quality=quality, outcomes=[
            dict(observe='end', change=effect_bounds, post=contract['post'])]),
        second: dict(requires=requires)})
    rule = dict(name='choose', when={}, option=first, expect=contract['expect'], then={'end':second})
    compiled = compile_policy(dict(rules=[rule]), profile)
    case = compiled.check()['cases'][0]
    return dict(action=first, continuation=second, status=case['status'],
                admitted=compiled.decide(cell_id,allow_empirical=allow_empirical) is not None,
                cell=public_cell, requires=requires, effects=effect_bounds,
                post=case['branches'][0]['post'], issues=case['issues'])


def first_admitted(checks):
    return next((key for key,case in checks.items() if case['admitted']), None)
