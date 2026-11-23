import numpy as np
import torch

from experiments.planar_tasks.semantic_rl.program_trace_value_v1.model import ValueModel
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.q_model import QModel, all_v, double_dqn_target


def layout():
    return dict(width=12, trace_width=7, trace_widths=[1]*7)


def test_factored_values_match_separate_cap_heads():
    torch.manual_seed(7)
    x = torch.randn(3, 12)
    x[0, 8] = float('nan')
    for arm in ('GA', 'LOCAL', 'GLOBAL'):
        model = ValueModel(layout(), np.zeros(12), np.ones(12), arm)
        expected = torch.stack([model(x, torch.full((3,), i)) for i in range(7)], dim=1)
        torch.testing.assert_close(all_v(model, x), expected, atol=1e-7, rtol=1e-6)


def test_q_factoring_and_multisource_trace_masks():
    torch.manual_seed(8)
    x = torch.randn(3, 12)
    candidates = torch.randn(3, 4, 85)
    candidates[0, 1, 3] = float('nan')
    sources = torch.zeros(3, 4, 7)
    sources[:, 0, 0] = 1
    sources[:, 0, 2] = 1
    sources[:, 1, 3] = 1
    sources[:, 2, 4] = 1
    # Fourth action is a legal public fallback with no CAP source.
    for arm in ('GA', 'LOCAL', 'GLOBAL'):
        model = QModel(layout(), np.zeros(12), np.ones(12), arm, np.zeros(85), np.ones(85))
        z = model.encode(x)[:, None, :].expand(-1, 4, -1).clone()
        if arm != 'GLOBAL':
            mask = sources if arm == 'LOCAL' else torch.zeros_like(sources)
            z[:, :, 5:12] *= mask
            z[:, :, 17:24] *= mask
        finite = torch.isfinite(candidates)
        encoded = torch.cat([torch.where(finite, candidates, 0.).clamp(-10, 10), finite.float(), sources], dim=-1)
        expected = model.net(torch.cat([z, encoded], dim=-1)).squeeze(-1)
        torch.testing.assert_close(model.values(x, candidates, sources), expected, atol=1e-7, rtol=1e-6)


def test_double_dqn_selects_online_and_evaluates_target_and_stops_at_budget():
    class Fixed:
        def __init__(self, values):self.fixed = torch.tensor(values)
        def values(self, *unused):return self.fixed
    online = Fixed([[.9, .8, .99], [.9, .8, .99]])
    target = Fixed([[.1, .7, .95], [.1, .7, .95]])
    batch = dict(x=None, candidates=None, sources=None, valid=torch.tensor([[True, True, False]]*2))
    value = double_dqn_target(online, target, batch, torch.tensor([0., 0.]), torch.tensor([False, True]))
    torch.testing.assert_close(value, torch.tensor([.1, 0.]))
