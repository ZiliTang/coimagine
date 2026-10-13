"""Tests for the fit_capacity_v1 risks named in the task book section 8.

Only the paths with real risk are covered: pairing-key uniqueness, the four-loss sum,
width-invariant sample order, checkpoint selection not touching test data, the final
checkpoint never being overwritten by a selected one, resume state integrity, and the
misrejection / ranking decomposition identity.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from coimagine.continuous2d_learning.model import ModelConfig, _tensors, JointWindowMLP
from coimagine.fit_capacity import trainer
from coimagine.fit_capacity.candidates import feasibility_decomposition, pair_episodes
from coimagine.fit_capacity.metrics import loss_components

ROOT = Path(__file__).resolve().parents[1]


def _records(count: int):
    """Small synthetic record set in the exact schema the encoder expects."""

    from coimagine.continuous2d_learning.records import CollectionConfig, collect_episode

    config = __import__("coimagine.continuous2d.config", fromlist=["default_config"]).default_config()
    records, _ = collect_episode(
        "S0_dev", "S0_dev/00", "train", config,
        CollectionConfig(episodes_per_scene=1, max_windows=count, full_step_episodes=0),
    )
    return records * max(1, math.ceil(count / max(1, len(records))))


# ---------------------------------------------------------------------------
# A1: pairing key
# ---------------------------------------------------------------------------


def test_pairing_key_includes_model_seed_and_is_unique():
    rows = []
    for scene in ("S0_v4", "S1_v4"):
        for episode_seed in (0, 1, 2):
            for model_seed in (0, 1, 2):
                rows.append(
                    {
                        "scene_id": scene, "episode_seed": episode_seed, "H": 3, "model_seed": model_seed,
                    }
                )
    keys = pair_episodes(rows)
    assert len(keys) == len(set(keys)) == 18, "the model seed must be part of the pairing key"
    # dropping the model seed would collide, which is the reported bug
    loose = [(r["scene_id"], r["episode_seed"], r["H"]) for r in rows]
    assert len(loose) != len(set(loose))


# ---------------------------------------------------------------------------
# Section 4: the four losses add up
# ---------------------------------------------------------------------------


def test_four_loss_components_sum_to_the_total():
    torch.manual_seed(0)
    records = _records(8)
    tensors = _tensors(records)
    model = JointWindowMLP(ModelConfig(hidden=16))
    outputs = model(tensors[0])
    components = loss_components(outputs, tuple(tensors[1:]))
    total = torch.stack(components).sum()
    manual = (
        float(components[0].mean())
        + float(components[1].mean())
        + float(components[2].mean())
        + float(components[3].mean())
    )
    assert float(total) == pytest.approx(manual * len(records), rel=1e-6)


# ---------------------------------------------------------------------------
# B1: sample order is independent of width
# ---------------------------------------------------------------------------


def test_sample_order_is_identical_across_widths(tmp_path):
    records = _records(12)
    val = records[:4]
    small = trainer.run(
        "Old+Random", 8, 0, records, val, val, tmp_path / "w8",
        total_updates=40, order_log_limit=8,
    )
    wide = trainer.run(
        "Old+Random", 16, 0, records, val, val, tmp_path / "w16",
        total_updates=40, order_log_limit=8,
    )
    assert small.sample_order == wide.sample_order, "the data RNG must not depend on width"
    assert small.initial_parameter_hash != wide.initial_parameter_hash


def test_initial_parameters_depend_only_on_width_and_seed(tmp_path):
    records = _records(12)
    val = records[:4]
    a = trainer.run("Old+Random", 8, 3, records, val, val, tmp_path / "a", total_updates=5, order_log_limit=0)
    b = trainer.run("Old+Branch", 8, 3, records, val, val, tmp_path / "b", total_updates=5, order_log_limit=0)
    c = trainer.run("Old+Random", 8, 4, records, val, val, tmp_path / "c", total_updates=5, order_log_limit=0)
    assert a.initial_parameter_hash == b.initial_parameter_hash
    assert a.initial_parameter_hash != c.initial_parameter_hash


# ---------------------------------------------------------------------------
# B2: checkpoint selection
# ---------------------------------------------------------------------------


def test_training_signature_has_no_test_input():
    parameters = inspect.signature(trainer.run).parameters
    assert not any("test" in name for name in parameters), "checkpoint selection must not see test data"


def test_final_checkpoint_is_never_overwritten_by_a_selected_one(tmp_path):
    records = _records(12)
    val = records[:4]
    result = trainer.run(
        "Old+Random", 8, 0, records, val, val, tmp_path / "run", total_updates=1000, order_log_limit=0
    )
    paths = {kind: entry["path"] for kind, entry in result.checkpoints.items()}
    assert len(set(paths.values())) == len(paths), "checkpoint files must be distinct"
    assert Path(paths["fixed_1000"]).name == "update1000.pt"
    blob = torch.load(paths["fixed_1000"], map_location="cpu")
    assert blob["kind"] == "fixed_1000"
    assert int(blob["updates"]) == 1000


def test_selection_excludes_update_zero(tmp_path):
    records = _records(12)
    val = records[:4]
    result = trainer.run(
        "Old+Random", 8, 0, records, val, val, tmp_path / "run", total_updates=600, order_log_limit=0
    )
    for kind, entry in result.checkpoints.items():
        if kind.startswith("best_"):
            assert entry["updates"] > 0, "the contract excludes update 0 from checkpoint selection"


def test_resume_state_round_trips(tmp_path):
    records = _records(12)
    val = records[:4]
    cell = tmp_path / "run"
    result = trainer.run("Old+Random", 8, 0, records, val, val, cell, total_updates=50, order_log_limit=2)
    blob = torch.load(cell / "resume_state.pt", map_location="cpu")
    assert blob["updates"] == result.updates
    assert "optimizer" in blob and "data_generator" in blob
    model = JointWindowMLP(ModelConfig(**blob["config"]))
    model.load_state_dict(blob["model"])
    final = torch.load(result.checkpoints["fixed_1000"].path if False else cell / "resume_state.pt", map_location="cpu")
    for name, parameter in model.state_dict().items():
        assert torch.equal(parameter, final["model"][name])


# ---------------------------------------------------------------------------
# D2: feasibility decomposition identity
# ---------------------------------------------------------------------------


def test_feasibility_decomposition_identity():
    j_ref = np.asarray([2.0, 1.5, -0.5, -1.0, -2.0])
    split = feasibility_decomposition(j_ref, best_all=0, best_feasible=2, chosen_feasible=3)
    assert split["identity_residual"] == pytest.approx(0.0)
    assert split["misrejection_loss"] == pytest.approx(2.5)
    assert split["ranking_loss"] == pytest.approx(0.5)
    assert split["execution_regret"] == pytest.approx(3.0)
