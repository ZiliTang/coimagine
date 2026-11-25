"""Memoize immutable SMT expression DAGs without changing verifier queries."""
from functools import lru_cache


def compile_policy(policy):
    if getattr(policy,'_study_compiled',False):return policy
    original_symbolic=policy._symbolic
    original_selection=policy._selection
    symbols={}
    selections={}
    def symbolic(prefix):
        if prefix not in symbols:symbols[prefix]=original_symbolic(prefix)
        return symbols[prefix]
    def selection(values):
        key=id(values)
        # Retain the values dictionary so object identities cannot be recycled.
        if key not in selections:selections[key]=(values,original_selection(values))
        return selections[key][1]
    policy._symbolic=symbolic
    policy._selection=selection
    policy._study_compiled=True
    return policy
