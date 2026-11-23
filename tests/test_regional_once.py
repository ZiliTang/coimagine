"""Regressions for a single intervention and paired acceptance with sparse outcomes."""
import pytest

from tools.cap_regional_once import binom_lower, lower_bound, override_from_branches


def test_first_eligible_crossing_commits_exactly_once():
    room = {'branches': [{'step': 1, 'delta': -1}, {'step': 1, 'delta': 0},
                         {'step': 8, 'delta': 1}, {'step': 31, 'delta': -1}]}
    assert override_from_branches(room, [.2, .24, .9, .3], .2) == (0, 1)
    assert override_from_branches(room, [.2, .2, .9, .3], .2) == (1, 8)
    assert override_from_branches(room, [.2, .2, .1, -.1], .2) == (0, None)


def test_one_sided_bound_handles_tiny_or_empty_discordance():
    assert binom_lower(10, 10, .025) == pytest.approx(.025 ** .1)
    assert lower_bound(0, 0, 200) < 0
    assert lower_bound(1, 0, 200) < 0
    assert lower_bound(10, 0, 200) > 0
    assert lower_bound(10, 10, 200) < 0
