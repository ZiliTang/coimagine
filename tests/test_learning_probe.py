"""Tests for the learning probe.

These guard the two things that make any result from the probe trustworthy:

* the hand-written gradients are correct (checked numerically against finite
  differences), so training is really optimising the stated loss;
* the ablation behaves as claimed -- a model whose training data never contains a
  contact mode cannot even represent that mode at test time.
"""

from __future__ import annotations

import numpy as np
import pytest

from coimagine.diagnostics.learning_probe import (
    ProbeResult,
    format_probe_markdown,
    run_learning_probe,
    teammate_slips,
    write_probe_report,
)
from coimagine.learning import FEATURE_DIM, generate_dataset, sample_to_arrays
from coimagine.learning.data import pose_to_index
from coimagine.learning.mlp import MultiHeadMLP


def test_features_have_the_declared_dimension() -> None:
    dataset = generate_dataset(n_scenes=4, lead_in=1, seed=0)
    x, y, physics_index, targets = sample_to_arrays(dataset.samples)
    assert x.shape[1] == FEATURE_DIM
    assert targets.shape == (x.shape[0], 3)
    assert set(np.unique(targets)) <= {0.0, 1.0}
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert physics_index.min() >= 0 and physics_index.max() < 16
    assert len(dataset.samples) == x.shape[0]


def test_gradients_match_finite_differences() -> None:
    """The MLP is hand-written, so its gradients are verified numerically."""

    rng = np.random.default_rng(0)
    model = MultiHeadMLP(hidden=6, seed=1)
    obs = rng.normal(size=(4, model.obs_dim))
    action = np.zeros((4, model.action_dim))
    action[:, 0] = 1.0
    feasible = rng.integers(0, 2, size=4).astype(float)
    physics_index = rng.integers(0, 16, size=4)
    pose_index = rng.integers(0, 36, size=4)

    _loss, grads, _ = model.loss_and_grads(obs, action, feasible, physics_index, pose_index)
    for key in ("W1", "W2", "Wf1", "Wf2", "Wp", "Wq"):
        flat = model.params[key]
        for _ in range(4):
            index = tuple(rng.integers(0, s) for s in flat.shape)
            original = flat[index]
            step = 1e-6
            flat[index] = original + step
            plus, _, _ = model.loss_and_grads(obs, action, feasible, physics_index, pose_index)
            flat[index] = original - step
            minus, _, _ = model.loss_and_grads(obs, action, feasible, physics_index, pose_index)
            flat[index] = original
            numeric = (plus - minus) / (2 * step)
            analytic = grads[key][index]
            assert numeric == pytest.approx(analytic, rel=1e-4, abs=1e-6), key


def test_pose_target_round_trips() -> None:
    assert pose_to_index((4, 1)) == 1 * 6 + 4
    assert pose_to_index((0, 5)) == 5 * 6


def test_teammate_slips_predicate_matches_the_physics() -> None:
    dataset = generate_dataset(n_scenes=20, lead_in=1, seed=0)
    rare = [s for s in dataset.samples if teammate_slips(s)]
    assert rare, "the generator must produce slipping cases"
    assert all(s.physics.traction[1] == "slip" for s in rare)


@pytest.mark.slow
def test_learning_probe_reports_and_separates_the_modes(tmp_path) -> None:
    """End-to-end smoke test plus the locked ablation finding."""

    result = run_learning_probe(n_scenes=40, epochs=60, seed=0)
    assert isinstance(result, ProbeResult)
    assert result.n_train > 0 and result.n_test > 0
    # The pose decoder should recover the object placement from the scan.
    assert result.in_distribution.pose_accuracy > 0.5
    # The robust property is representational, not a directional accuracy claim:
    # a model that never saw the mode cannot decode it at all.
    assert result.narrow_on_rare_mode.physics_accuracy < 0.05
    assert (
        result.wide_on_rare_mode.physics_accuracy
        > result.narrow_on_rare_mode.physics_accuracy
    )
    paths = write_probe_report(result, str(tmp_path))
    assert "md" in paths and "json" in paths
    assert "balanced accuracy" in format_probe_markdown(result)
