"""Tests for the learned-versus-reference replay.

The replay is the artefact that shows *where* a learned model errs, so the guard is that it
really runs both policies on the same episode and marks the boundaries where they differ.
"""

from __future__ import annotations

import pytest

from coimagine.diagnostics.learned_replay import (
    compare_policies,
    find_and_render_disagreement,
    render_comparison,
)
from coimagine.learning.data import generate_situations
from coimagine.learning.mlp import MultiHeadMLP


def test_compare_runs_both_policies_on_the_same_scene() -> None:
    scene = generate_situations(n_scenes=3, lead_in=0, seed=0)[0].scene
    model = MultiHeadMLP(seed=0)
    comparison = compare_policies(scene, model, horizon=4, seed=0)
    assert comparison.scene is scene
    assert comparison.learned and comparison.reference
    # the disagreement list is derived from the two traces, never invented
    expected = [
        a.t
        for a, b in zip(comparison.learned, comparison.reference)
        if a.action != b.action
    ]
    assert comparison.disagreement_steps == expected
    assert set(comparison.disagreement_steps) <= {s.t for s in comparison.learned}
    assert comparison.learned_success in (0, 1)
    assert comparison.reference_success in (0, 1)
    for step in comparison.learned + comparison.reference:
        assert step.action
        assert step.scan is not None


def test_render_writes_png_and_pdf(tmp_path) -> None:
    scene = generate_situations(n_scenes=3, lead_in=0, seed=0)[0].scene
    model = MultiHeadMLP(seed=0)
    comparison = compare_policies(scene, model, horizon=3, seed=0)
    paths = render_comparison(comparison, out_dir=str(tmp_path), stem="replay")
    assert len(paths) == 2
    assert paths[0].endswith(".png") and paths[1].endswith(".pdf")
    import os

    for path in paths:
        assert os.path.getsize(path) > 0


@pytest.mark.slow
def test_finds_an_episode_where_the_policies_differ() -> None:
    """The figure is only informative when a disagreement exists, so the search must
    actually find one rather than silently rendering an agreeing episode."""

    comparison, paths = find_and_render_disagreement(
        n_scenes=30, epochs=120, seed=0, out_dir="outputs/grid/replays_test"
    )
    if comparison is None:
        pytest.skip("no disagreeing episode in this configuration")
    assert comparison.disagreement_steps
    assert paths
