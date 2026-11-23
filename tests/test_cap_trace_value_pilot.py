import ast
import math

import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.instrumentation import Instrument, HELPERS
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.model import ValueModel


def test_trace_preserves_short_circuit_and_memory_mutation():
    source = '''
def step(memory, x):
    y = x + 2
    if x > 0 and 1 / x > 0:
        memory['n'] += y
        return memory
    return memory
'''
    original = {}
    exec(source, original)
    transform = Instrument({'step'})
    tree = transform.visit(ast.parse(source))
    tree.body.extend(ast.parse(HELPERS).body)
    instrumented = {'math': math}
    exec(compile(ast.fix_missing_locations(tree), '<test trace>', 'exec'), instrumented)
    for x in (0, 2, -1):
        left, right = {'n': 3}, {'n': 3}
        trace = [float('nan')] * len(transform.slots)
        assert original['step'](left, x) == instrumented['step'](right, x, _tb=trace)
        assert left == right
        assert np.isfinite(trace).any()


def test_trace_mask_hides_both_values_and_presence():
    layout = dict(width=12, trace_width=7, trace_widths=[1]*7)
    torch.manual_seed(17)
    x = torch.ones((1, 12))
    changed = x.clone()
    changed[:, 6:] = float('nan')
    for arm in ('GA', 'LOCAL'):
        model = ValueModel(layout, np.zeros(12), np.ones(12), arm)
        assert torch.equal(model(x, torch.tensor([0])), model(changed, torch.tensor([0])))
    model = ValueModel(layout, np.zeros(12), np.ones(12), 'GA')
    changed[:, 5] = float('nan')
    assert torch.equal(model(x, torch.tensor([0])), model(changed, torch.tensor([0])))


def test_same_network_size_and_checkpoint_scores(tmp_path):
    layout = dict(width=12, trace_width=7, trace_widths=[1]*7)
    models = [ValueModel(layout, np.zeros(12), np.ones(12), arm) for arm in ('GA', 'LOCAL', 'GLOBAL')]
    assert len({sum(p.numel() for p in model.parameters()) for model in models}) == 1
    x = torch.randn(1, 12)
    path = tmp_path/'weights.pt'
    torch.save(models[2].state_dict(), path)
    other = ValueModel(layout, np.zeros(12), np.ones(12), 'GLOBAL')
    other.load_state_dict(torch.load(path, weights_only=True))
    assert torch.equal(models[2].all_values(x), other.all_values(x))
