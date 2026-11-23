import numpy as np
import torch

from tools.cap_regional_actor_iteration import group_margin, slots


def test_group_ranking_matches_ten_logit_greedy_rule():
    scores = torch.tensor([[1., 3., 2.5, 100.]])
    base = torch.tensor([[True, True, False, False]])
    alternate = torch.tensor([[False, False, True, False]])
    assert group_margin(scores, base, alternate).item() == -.5


def test_two_slots_on_a_single_boundary():
    base = dict(step=0, prefix='hash', base='original', x=np.ones(2), base_mask=[True])
    assert slots([], 37) == [None, None]
    first, other = slots([dict(base, alternatives=[('best', [True])])], 37)
    assert first['alternative'] == 'best' and other is None
    first, other = slots([dict(base, alternatives=[('best', [True]), ('lower', [False])])], 37)
    assert first['alternative'] == 'best' and other['alternative'] == 'lower'
