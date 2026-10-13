"""Tests for the Phase B legacy-grid candidate-selection audit.

The audit's validity rests on three properties, and each is pinned here: the nested
candidate sets are built without touching the model or the true return, the exact reference
conditions on the consistent worlds rather than on one realised world, and the model branch
does not change when only the hidden truth changes.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from coimagine.diagnostics.grid_selection_audit import (
    build_decision_point,
    candidate_order,
    consistent_worlds,
    joint_score,
    model_outcome,
    nested,
    selection_metrics,
)
from coimagine.learning.belief_model import load_belief_model
from coimagine.learning.data import _joint_for, generate_situations
from coimagine.learning.features import ACTION_IDS

CHECKPOINT = "outputs/grid/world_model/belief.npz"


def test_candidate_order_is_nested_deterministic_and_model_free() -> None:
    first = candidate_order(2, 0)
    second = candidate_order(2, 0)
    other = candidate_order(2, 1)
    assert first == second
    assert first != other
    assert len(first) == 36  # 6**2, below the 128 cap
    assert len(candidate_order(1, 0)) == 6
    assert len(candidate_order(3, 0)) == 128  # capped
    assert set(nested(first, 16)) <= set(nested(first, 64)) <= set(nested(first, 128))
    assert all(len(sequence) == 2 for sequence in first)


def test_selection_metrics_tie_break_and_mock_reference() -> None:
    ids = [5, 2, 9]
    j_hat = np.asarray([1.0, 1.0, 0.0])
    j_ref = np.asarray([0.0, 0.0, 2.0])
    metrics = selection_metrics(ids, j_hat, j_ref)
    # Ties on J_hat break on the smaller stable candidate id.
    assert metrics["chosen_candidate_id"] == 2
    mock = selection_metrics(ids, j_ref, j_ref)
    assert mock["selected_bias"] == pytest.approx(0.0)
    assert mock["candidate_regret"] == pytest.approx(0.0)
    assert mock["selection_excess"] == pytest.approx(0.0)
    assert mock["uniform_candidate_return"] == pytest.approx(float(np.mean(j_ref)))


def test_joint_score_is_higher_for_success() -> None:
    assert joint_score(1.0, 5.0, 0.0) > joint_score(0.0, 0.0, 0.0)
    assert joint_score(0.0, 0.0, 0.0) > joint_score(0.0, 1.0, 0.0)


@pytest.fixture(scope="module")
def situation():
    situations = generate_situations(n_scenes=1, lead_in=2, seed=55_000)
    return situations[0]


def test_consistent_worlds_contain_the_true_physics(situation) -> None:
    worlds = consistent_worlds(situation)
    assert 1 <= len(worlds) <= 16
    assert situation.physics in worlds


def test_model_branch_ignores_hidden_truth(situation) -> None:
    model = load_belief_model(CHECKPOINT)
    point = build_decision_point(situation, model)
    sequence = candidate_order(1, 0)[0]

    baseline, _, _ = model_outcome(model, point, sequence)

    # Change only the hidden physics of the stored situation; the observation and the model
    # inputs are untouched, so the model's fixed-sequence return must not move.
    alternative = None
    for candidate in consistent_worlds(situation):
        if candidate != situation.physics:
            alternative = candidate
            break
    assert alternative is not None
    altered_scene = dataclasses.replace(point.situation.scene, physics=alternative)
    altered = dataclasses.replace(point.situation, scene=altered_scene)
    altered_point = dataclasses.replace(point, situation=altered)
    changed, _, _ = model_outcome(model, altered_point, sequence)
    assert changed == pytest.approx(baseline)

    # The true reference, in contrast, is allowed to move with the hidden truth.
    from coimagine.diagnostics.grid_selection_audit import true_outcome

    true_baseline, _, _ = true_outcome(point, sequence)
    true_changed, _, _ = true_outcome(altered_point, sequence)
    assert set(point.worlds) == set(altered_point.worlds)
    assert true_baseline == pytest.approx(true_changed)  # consistent set is unchanged
