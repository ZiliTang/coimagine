"""Acceptance tests for the coverage_scale_v1 256k continuation (items 1-4).

The expensive nesting check needs the extended dataset and is skipped when it is absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

DATA_RUN = Path("outputs/coverage_scale_v1/coverage-scale-20261013")
OUT_RUN = Path("outputs/coverage_scale_v1/coverage-scale-256k")


# ---------------------------------------------------------------------------
# budget and schedule
# ---------------------------------------------------------------------------


def test_uncapped_budget_keeps_sixty_passes():
    from coimagine.coverage_scale.trainer_ext import update_budget_uncapped

    assert update_budget_uncapped(256000) == 120000
    assert update_budget_uncapped(128000) == 60000
    assert update_budget_uncapped(512000) == 240000  # no cap any more
    for windows in (8000, 16000, 256000):
        assert update_budget_uncapped(windows) * 128 / windows == pytest.approx(60.0, rel=0.02)


def test_eval_schedule_matches_the_frozen_interval():
    from coimagine.coverage_scale.trainer_ext import EVAL_STEP, eval_schedule_3000

    assert EVAL_STEP == 3000
    points = eval_schedule_3000(120000)
    assert points[0] == 0 and points[-1] == 120000
    assert 100 in points and 3000 in points and 60000 in points
    assert len(points) == 42


# ---------------------------------------------------------------------------
# grouped metrics
# ---------------------------------------------------------------------------


def _records(n=64):
    from coimagine.coverage_scale.sampler import sample_root
    from coimagine.coverage_scale.recipe import collect_root
    from coimagine.coverage_scale.scene_registry import SceneRegistry
    from coimagine.continuous2d.config import default_config
    from coimagine.continuous2d.scenes import set_scene_resolver

    config = default_config()
    out = []
    roots = []
    ordinal = 0
    while len(out) < n:
        root = sample_root("dev", "S0", "broad", ordinal)
        roots.append(root)
        out.extend(collect_root(root, config)["windows"])
        ordinal += 1
    SceneRegistry.from_scenes(root.scene for root in roots).activate()
    return out[:n]


def test_group_partitions_and_overlap_are_consistent():
    from coimagine.coverage_scale.evaluation import MOTION_GROUPS, record_group

    records = _records()
    flags = [record_group(record) for record in records]
    for flag in flags:
        # translation and rotation are allowed to hold together; stationary is exactly their
        # complement, so the union is exhaustive and the motion members are not disjoint.
        assert flag["stationary"] == (not flag["translation"] and not flag["rotation"])
        assert flag["translation"] or flag["rotation"] or flag["stationary"]
        assert int(flag["contact"]) + int(flag["no_contact"]) == 1
    translation = np.asarray([flag["translation"] for flag in flags])
    rotation = np.asarray([flag["rotation"] for flag in flags])
    stationary = np.asarray([flag["stationary"] for flag in flags])
    assert np.array_equal(translation | rotation, ~stationary)


def test_grouped_summary_reports_each_body_and_its_persistence():
    from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig
    from coimagine.coverage_scale.evaluation import grouped_summary

    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    records = _records(160)
    model = JointWindowMLP(ModelConfig(hidden=32))
    summary = grouped_summary(model, records)
    assert summary["n_windows"] == len(records)
    for name in ("all", "translation", "rotation", "stationary", "contact", "no_contact"):
        group = summary["groups"][name]
        assert group["n_windows"] > 0
        for key in ("object_position", "robot0_position", "robot1_position",
                    "persistence_object", "persistence_robot0", "persistence_robot1",
                    "invalid", "invalid_rate" if "invalid_rate" in group else "invalid"):
            assert key in group or key == "invalid_rate"
        assert 0.0 <= group["invalid"] <= 1.0
    overlap = summary["overlap"]
    assert overlap["translation_and_rotation"] + overlap["translation_only"] == \
        summary["groups"]["translation"]["n_windows"]
    assert overlap["stationary"] == summary["groups"]["stationary"]["n_windows"]


def test_per_record_metrics_include_robot_persistence():
    from coimagine.continuous2d_learning.model import JointWindowMLP, ModelConfig
    from coimagine.coverage_scale.evaluation import per_record_metrics

    torch = pytest.importorskip("torch")
    records = _records(96)
    metrics = per_record_metrics(JointWindowMLP(ModelConfig(hidden=16)), records)
    assert metrics["persistence_robot0"].shape == (len(records),)
    assert np.all(metrics["persistence_robot0"] >= 0)
    assert np.all(metrics["persistence_robot1"] >= 0)


# ---------------------------------------------------------------------------
# extended trainer
# ---------------------------------------------------------------------------


def test_extended_trainer_saves_the_extra_checkpoint_and_resumes(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    from coimagine.coverage_scale.plan import load_registries  # noqa: F401
    from coimagine.coverage_scale.sampler import sample_root
    from coimagine.coverage_scale.scene_registry import SceneRegistry
    from coimagine.coverage_scale.trainer_ext import run_cell_ext

    roots = [sample_root("dev", "S0", "broad", ordinal) for ordinal in range(8)]
    SceneRegistry.from_scenes(root.scene for root in roots).activate()
    from coimagine.continuous2d.config import default_config
    from coimagine.coverage_scale.recipe import collect_root

    config = default_config()
    records = []
    for root in roots:
        records.extend(collect_root(root, config)["windows"])
    out = tmp_path / "cell"
    first = run_cell_ext("t", 32, 0, records, records[:128], records[:64], out,
                         max_updates=150, extra_checkpoints=(120,))
    assert first.updates == 150
    assert set(first.checkpoints) == {"best_new_val_total", "final", "update120"}
    blob = torch.load(out / "update120.pt", map_location="cpu")
    assert blob["updates"] == 120 and blob["selection_rule"] == "best_new_val_total"
    assert (out / "selection_eval.json").exists()
    selection = json.loads((out / "selection_eval.json").read_text())
    assert set(selection) == {"best_new_val_total", "final", "update120"}

    second = run_cell_ext("t", 32, 0, records, records[:128], records[:64], tmp_path / "cell2",
                          max_updates=150, extra_checkpoints=(120,))
    assert second.sample_order_hash == first.sample_order_hash
    assert second.checkpoints["update120"]["parameter_hash"] == first.checkpoints["update120"]["parameter_hash"]

    with pytest.raises(ValueError):
        run_cell_ext("t", 32, 0, records[:-1], records[:128], records[:64], out,
                     max_updates=150, resume=True)


def test_selection_rule_is_still_the_weighted_new_val_total(tmp_path):
    torch = pytest.importorskip("torch")
    torch.set_num_threads(2)
    from coimagine.continuous2d.config import default_config
    from coimagine.coverage_scale.recipe import collect_root
    from coimagine.coverage_scale.sampler import sample_root
    from coimagine.coverage_scale.scene_registry import SceneRegistry
    from coimagine.coverage_scale.trainer_ext import run_cell_ext

    roots = [sample_root("dev", "S1", "wall", ordinal) for ordinal in range(6)]
    SceneRegistry.from_scenes(root.scene for root in roots).activate()
    config = default_config()
    records = []
    for root in roots:
        records.extend(collect_root(root, config)["windows"])
    out = tmp_path / "cell"
    result = run_cell_ext("t", 16, 1, records, records[:128], records[:128], out,
                          max_updates=120, extra_checkpoints=())
    selection = json.loads((out / "selection_eval.json").read_text())
    schedule = result.schedule
    expected = min(row["new_val_total_loss"] for row in schedule)
    assert selection["best_new_val_total"]["new_val"]["weighted"]["total_loss"] == pytest.approx(
        expected, rel=1e-9
    )


# ---------------------------------------------------------------------------
# integration (needs the extended dataset)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not OUT_RUN.exists(), reason="256k continuation not run")
def test_256k_tier_extends_the_128k_prefix():
    from coimagine.coverage_scale.shards import tier_records

    small, small_info = tier_records(DATA_RUN, "train", 128000)
    large, large_info = tier_records(DATA_RUN, "train", 256000)
    assert small_info["reached_target"] and large_info["reached_target"]
    assert [record["transition_id"] for record in small] == \
        [record["transition_id"] for record in large[: len(small)]]
    assert large_info["independent_roots"] > small_info["independent_roots"]


@pytest.mark.slow
@pytest.mark.skipif(not OUT_RUN.exists(), reason="256k continuation not run")
def test_frozen_model_list_covers_both_tiers():
    manifest = json.loads((OUT_RUN / "model_list.json").read_text())
    labels = {entry["label"] for entry in manifest["models"]}
    # 128k kept two points per cell (best + final, final being update 60000);
    # 256k adds update60000 as a third point, so the equal-update comparison is possible.
    assert sum(1 for label in labels if label.startswith("128000/")) == 12
    assert sum(1 for label in labels if label.startswith("256000/")) == 18
    assert len(labels) == 30


# ---------------------------------------------------------------------------
# plan store: extension and scene-id collisions (errata E10, E11)
# ---------------------------------------------------------------------------


def test_plan_extension_appends_only_the_new_block(tmp_path):
    from coimagine.coverage_scale.plan import RootPlan

    plan = RootPlan(tmp_path)
    first = plan.ensure("dev", 24)
    assert sum(1 for _ in (tmp_path / "roots_dev.jsonl").open()) == 24
    second = RootPlan(tmp_path).ensure("dev", 48)
    assert len(second) == 48
    # the file must grow by the 24 new roots, not by a re-sampled block of 48
    assert sum(1 for _ in (tmp_path / "roots_dev.jsonl").open()) == 48
    assert [r.root_id for r in second[:24]] == [r.root_id for r in first]
    assert len({r.root_id for r in second}) == 48
    assert len({r.scene_id for r in second}) == 48


def test_duplicate_plan_rows_are_collapsed_on_read(tmp_path):
    from coimagine.coverage_scale.plan import RootPlan

    plan = RootPlan(tmp_path)
    plan.ensure("dev", 16)
    path = tmp_path / "roots_dev.jsonl"
    original = path.read_text()
    path.write_text(original + original)  # simulate the old append-everything bug
    reread = RootPlan(tmp_path)._read("dev")
    assert [r.root_id for r in reread] == [r.root_id for r in plan.peek("dev")]
    assert len(reread) == 16


def test_scene_id_collision_is_resolved_deterministically(tmp_path):
    from dataclasses import replace

    from coimagine.coverage_scale.plan import resolve_scene
    from coimagine.coverage_scale.sampler import sample_root
    from coimagine.coverage_scale.scene_registry import SceneRegistry, scene_to_dict

    root = sample_root("dev", "S1", "wall", 0)
    registry = SceneRegistry()
    # a foreign scene that already owns this id with different content
    decoy = replace(root.scene, name=root.scene_id, variant=99,
                    object_pose=replace(root.scene.object_pose, x=0.0))
    registry.add(decoy)
    resolved = resolve_scene(registry, root)
    assert resolved.scene_id != root.scene_id
    assert root.family in resolved.scene_id and root.mode in resolved.scene_id
    assert str(root.seed) in resolved.scene_id
    assert scene_to_dict(registry.get(resolved.scene_id)) == scene_to_dict(resolved.scene)
    # the geometry is unchanged; only the id was disambiguated
    without_name = lambda scene: {k: v for k, v in scene_to_dict(scene).items() if k != "name"}
    assert without_name(resolved.scene) == without_name(root.scene)
    assert registry.get(root.scene_id).variant == 99
    # identical content must not be renamed
    same = sample_root("dev", "S0", "broad", 0)
    other = SceneRegistry()
    other.add(same.scene)
    assert resolve_scene(other, same).scene_id == same.scene_id


@pytest.mark.slow
@pytest.mark.skipif(not DATA_RUN.exists(), reason="coverage-scale run not collected")
def test_extended_plan_has_unique_scene_ids_and_keeps_the_old_prefix():
    from coimagine.coverage_scale.plan import RootPlan, load_registries

    plan = RootPlan(DATA_RUN / "roots")
    roots = plan.ensure("train", 16 * 440)
    assert len({r.root_id for r in roots}) == len(roots)
    assert len({r.scene_id for r in roots}) == len(roots)
    registry = load_registries(DATA_RUN / "roots", ("train",))
    assert all(r.scene_id in registry for r in roots)
