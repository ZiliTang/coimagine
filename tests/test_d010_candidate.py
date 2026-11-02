"""D010 M3 candidate contract tests; pure, no physics."""
from experiments.option_rl_v19.unknown_shape import (
    ContactCandidate, choose_candidate_q, shift_sign_for_error,
)


def test_long_support_uses_inset_and_returns_margin():
    q, margin, status = choose_candidate_q(-.5, .5)
    assert status == 'ok'
    assert abs(q) < 1e-12
    assert margin == .06
    assert -.44 <= q <= .44


def test_short_support_is_flagged_view_insufficient():
    q, margin, status = choose_candidate_q(-.009, .009)
    assert q is None and status == 'view_insufficient'


def test_margin_limited_support_does_not_invent_full_margin():
    q, margin, status = choose_candidate_q(-.05, .05)
    assert status == 'margin_limited'
    assert abs(q) < 1e-12
    assert margin <= .05 + 1e-12


def test_negative_candidate_error_selects_negative_shift_direction():
    assert shift_sign_for_error(-.03) == -1.0


def test_missing_candidate_does_not_default_positive():
    assert shift_sign_for_error(None) == 0.0


def test_candidate_keeps_identity_and_q_until_explicit_update():
    c = ContactCandidate(7, (2.0, 1.0), (1., 0.), (0., 1.), 1.0, 1.0, (-.5, .5), .06)
    assert c.id == 7 and c.q == (2.0, 1.0)
    assert c.valid and c.updates == 0 and c.reason == 'initial'
