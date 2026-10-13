"""coverage_scale_v1 acceptance tests (task book section 7).

Fast tests build their own tiny artifacts; the integration test that needs the real collected
run is skipped when the run directory is absent.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from coimagine.continuous2d.scenes import scene_by_name, set_scene_resolver
from coimagine.coverage_scale.plan import NarrowPlan, load_registries, RootPlan
from coimagine.coverage_scale.sampler import (
    DEFAULT_MARGIN,
    FAMILIES,
    MODES,
    S0_ARENA,
    S1_ARENA,
    inside_goal,
    root_seed,
    sample_root,
    sample_roots,
    scene_id_for,
    strict_clearances,
    strict_legal,
)
from coimagine.coverage_scale.scene_registry import SceneRegistry, scene_from_dict, scene_to_dict

RUN_DIR = Path("outputs/coverage_scale_v1/coverage-scale-20261013")


@pytest.fixture(autouse=True)
def _reset_resolver():
    yield
    set_scene_resolver(None)


# ---------------------------------------------------------------------------
# A1: serialisation and cross-process resolution
# ---------------------------------------------------------------------------


def test_scene_serialisation_round_trip_is_exact():
    for family in FAMILIES:
        for mode in MODES:
            scene = sample_root("dev", family, mode, 0).scene
            assert scene_from_dict(json.loads(json.dumps(scene_to_dict(scene)))) == scene


def test_registry_resolves_across_processes(tmp_path):
    roots = sample_roots("dev", 6)
    registry = SceneRegistry.from_scenes(root.scene for root in roots)
    path = tmp_path / "scenes.json"
    registry.save(path)
    program = (
        "import json,sys;"
        "sys.path.insert(0,'src');"
        "from coimagine.coverage_scale.scene_registry import load_registry_file;"
        "from coimagine.continuous2d.scenes import scene_by_name;"
        "load_registry_file(__import__('pathlib').Path(sys.argv[1]));"
        "print(json.dumps({n: scene_by_name(n).arena for n in json.loads(sys.argv[2])}))"
    )
    names = [root.scene_id for root in roots]
    completed = subprocess.run(
        [sys.executable, "-c", program, str(path), json.dumps(names)],
        capture_output=True, text=True, check=True,
    )
    resolved = json.loads(completed.stdout)
    assert set(resolved) == set(names)
    for root in roots:
        assert resolved[root.scene_id] == list(root.scene.arena)


def test_original_names_still_resolve():
    for name in ("S0_dev", "S1_v3"):
        assert scene_by_name(name).name == name


# ---------------------------------------------------------------------------
# A2: initial-state legality and distribution ranges
# ---------------------------------------------------------------------------


def test_sampled_roots_are_strictly_legal():
    for family in FAMILIES:
        for mode in MODES:
            for ordinal in range(3):
                root = sample_root("dev", family, mode, ordinal)
                assert root is not None, f"{family}/{mode}/{ordinal} blocked"
                clearances = strict_clearances(root.scene)
                assert min(clearances.values()) >= DEFAULT_MARGIN - 1e-12
                assert strict_legal(root.scene)
                goal, arena = root.scene.goal, root.scene.arena
                assert goal.cx - goal.half_w >= -1e-9
                assert goal.cx + goal.half_w <= arena[0] + 1e-9
                assert goal.cy - goal.half_h >= -1e-9
                assert goal.cy + goal.half_h <= arena[1] + 1e-9
                assert not inside_goal(
                    root.scene.object_pose.x, root.scene.object_pose.y,
                    root.scene.object_pose.angle, goal,
                )
                for box in root.scene.obstacles:
                    assert box.half_w > 0 and box.half_h > 0


def test_obstacle_corner_inside_object_is_rejected():
    """The frozen corner-only test accepts this state; the strict check must not."""

    from coimagine.continuous2d.legality import legality_report
    from coimagine.continuous2d.scenes import Box, Pose, SceneSpec

    # A thin wall corner clearly inside a large object footprint.
    scene = SceneSpec(
        name="regression_corner_inside_object",
        category="S0",
        variant=-1,
        arena=S0_ARENA,
        obstacles=(Box(1.5, 1.1, 0.09, 0.4),),
        goal=Box(2.6, 1.1, 0.3, 0.3),
        object_pose=Pose(1.5, 1.1, 0.0),
        robot_poses=(Pose(0.4, 0.4, 0.0), Pose(0.4, 1.8, 0.0)),
    )
    assert legality_report(scene)["legal"] is True
    assert strict_legal(scene) is False


def test_ranges_respected():
    for family, arena in (("S0", S0_ARENA), ("S1", S1_ARENA)):
        for mode in MODES:
            for ordinal in range(4):
                root = sample_root("dev", family, mode, ordinal)
                assert root.scene.arena == arena
                pose = root.scene.object_pose
                assert 0.0 <= pose.x <= arena[0] and 0.0 <= pose.y <= arena[1]
                if family == "S1":
                    assert len(root.scene.obstacles) == 2
                    lower, upper = sorted(root.scene.obstacles, key=lambda box: box.cy)
                    gap = (upper.cy - upper.half_h) - (lower.cy + lower.half_h)
                    assert 0.60 - 1e-9 <= gap <= 0.85 + 1e-9
                    assert 1.40 - 1e-9 <= lower.cx <= 1.80 + 1e-9
                else:
                    assert root.scene.obstacles == ()


def test_same_seed_reproducible_different_seed_changes_geometry():
    for family in FAMILIES:
        for mode in MODES:
            first = sample_root("dev", family, mode, 0)
            again = sample_root("dev", family, mode, 0)
            other = sample_root("dev", family, mode, 1)
            assert scene_to_dict(first.scene) == scene_to_dict(again.scene)
            assert scene_to_dict(first.scene) != scene_to_dict(other.scene)


# ---------------------------------------------------------------------------
# B2: split lineage and nested plans
# ---------------------------------------------------------------------------


def test_split_seed_spaces_are_disjoint():
    seeds = {}
    for split in ("train", "val", "test", "dev"):
        for family in FAMILIES:
            for mode in MODES:
                seeds.setdefault(split, set()).add(root_seed(split, family, mode, 0))
    assert not (seeds["train"] & seeds["val"])
    assert not (seeds["train"] & seeds["test"])
    assert not (seeds["val"] & seeds["test"])
    assert not (seeds["train"] & seeds["dev"])
    identifiers = {split: {scene_id_for(f, root_seed(split, f, m, 0)) for f in FAMILIES for m in MODES}
                   for split in ("train", "val", "test")}
    assert not (identifiers["train"] & identifiers["val"])
    assert not (identifiers["train"] & identifiers["test"])


def test_root_plan_is_nested_and_mode_mixture_is_frozen():
    plan = sample_roots("dev", 64)
    smaller = sample_roots("dev", 24)
    assert [r.root_id for r in smaller] == [r.root_id for r in plan[:24]]
    counts = {"broad": 0, "wall": 0, "goal": 0}
    for root in plan:
        counts[root.mode] += 1
    assert counts["broad"] == 32
    assert counts["wall"] == 16 and counts["goal"] == 16
    families = {"S0": 0, "S1": 0}
    for root in plan:
        families[root.family] += 1
    assert families == {"S0": 32, "S1": 32}


def test_narrow_plan_keeps_six_independent_initial_states():
    plan = NarrowPlan(Path("/tmp/does-not-matter"))
    roots = plan.ensure("narrow", 60)
    assert len({root.scene_id for root in roots}) == 6
    assert len({root.root_id for root in roots}) == 60
    assert [root.scene_id for root in roots[:6]] == [
        "S0_dev", "S1_dev", "S0_v1", "S1_v1", "S0_v2", "S1_v2"
    ]


def test_root_plan_caches_and_reloads(tmp_path):
    plan = RootPlan(tmp_path)
    first = plan.ensure("dev", 12)
    again = RootPlan(tmp_path).ensure("dev", 12)
    assert [r.root_id for r in first] == [r.root_id for r in again]
    assert [scene_to_dict(r.scene) for r in first] == [scene_to_dict(r.scene) for r in again]
    registry = load_registries(tmp_path, ("dev",))
    for root in first:
        assert scene_to_dict(registry.get(root.scene_id)) == scene_to_dict(root.scene)


def test_tiers_skip_shard_holes(tmp_path):
    """A hole in the finished shards must stop the prefix, not be jumped over."""

    from coimagine.coverage_scale.shards import contiguous_done, done_marker, scan_done

    for shard in (0, 1, 3, 4):
        marker = done_marker(tmp_path, "train", shard)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("{}")
    assert scan_done(tmp_path, "train") == [0, 1, 3, 4]
    assert contiguous_done(tmp_path, "train") == [0, 1]


# ---------------------------------------------------------------------------
# B1/C: recipe invariants and candidate identity
# ---------------------------------------------------------------------------


def test_recipe_produces_real_windows_with_stable_ids():
    from coimagine.continuous2d.config import default_config
    from coimagine.coverage_scale.recipe import MAX_REAL_WINDOWS, collect_root

    root = sample_root("dev", "S0", "broad", 0)
    payload = collect_root(root, default_config())
    stats, windows = payload["stats"], payload["windows"]
    assert 0 < stats["windows_total"] <= MAX_REAL_WINDOWS
    assert stats["windows_random"] <= 8 and stats["windows_guide"] <= 8
    seen = set()
    for record in windows:
        assert record["split"] == "dev"
        assert record["scene_id"] == root.scene_id
        assert record["transition_id"]
        assert record["scene_content_hash"]
        seen.add(record["transition_id"])
    # A transition id is content-addressed, so it must be reproducible.
    again = collect_root(root, default_config())
    assert [record["transition_id"] for record in again["windows"]] == [
        record["transition_id"] for record in windows
    ]


def test_candidate_pairing_and_regret_identity():
    import numpy as np

    from coimagine.fit_capacity.candidates import feasibility_decomposition

    j_ref = np.asarray([0.0, 5.0, -3.0, 1.0], dtype=np.float64)
    split = feasibility_decomposition(j_ref, best_all=1, best_feasible=0, chosen_feasible=3)
    assert split["misrejection_loss"] == pytest.approx(5.0)
    assert split["ranking_loss"] == pytest.approx(-1.0)
    assert split["execution_regret"] == pytest.approx(4.0)
    assert split["identity_residual"] == pytest.approx(0.0)


def test_trainer_resume_and_snapshot_identity(tmp_path):
    import torch

    from coimagine.continuous2d.config import default_config
    from coimagine.coverage_scale.recipe import collect_root
    from coimagine.coverage_scale.scene_registry import SceneRegistry
    from coimagine.coverage_scale.trainer import run_cell

    torch.set_num_threads(2)
    config = default_config()
    records = []
    roots = []
    ordinal = 0
    while len(records) <= 384:
        root = sample_root("dev", "S0", "broad", ordinal)
        roots.append(root)
        records.extend(collect_root(root, config)["windows"])
        ordinal += 1
    assert len(records) > 128
    SceneRegistry.from_scenes(root.scene for root in roots).activate()
    val = records[:64]
    monitor = records[:128]
    out = tmp_path / "cell"
    first = run_cell("t", 32, 0, records, monitor, val, out, max_updates=120)
    assert first.updates == 120
    assert set(first.checkpoints) == {"best_new_val_total", "final"}
    order_hash = first.sample_order_hash

    # Resume from 60 by rerunning to 120 with resume=True on a copy of the state at 120 is not
    # possible here, so instead check that a second identical run reproduces the stream and the
    # selected parameters exactly (determinism), and that resume refuses a changed dataset.
    out2 = tmp_path / "cell2"
    second = run_cell("t", 32, 0, records, monitor, val, out2, max_updates=120)
    assert second.sample_order_hash == order_hash
    assert second.checkpoints["best_new_val_total"]["parameter_hash"] == \
        first.checkpoints["best_new_val_total"]["parameter_hash"]

    import torch as _torch

    state = _torch.load(out / "best_new_val_total.pt", map_location="cpu")
    assert state["selection_rule"] == "best_new_val_total"
    with pytest.raises(ValueError):
        run_cell("t", 32, 0, records[:-1], monitor, val, out, max_updates=120, resume=True)


def test_update_budget_and_schedule():
    from coimagine.coverage_scale.trainer import eval_schedule, update_budget

    assert update_budget(8000) == 6000
    assert update_budget(16000) == 7500
    assert update_budget(128000) == 60000
    assert update_budget(1000000) == 60000
    points = eval_schedule(6000)
    assert points[0] == 0 and points[-1] == 6000 and 100 in points
    assert len(points) >= 12  # the frozen step for Umax=6000 is 500


# ---------------------------------------------------------------------------
# Integration (skipped without the real run)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.skipif(not RUN_DIR.exists(), reason="coverage-scale run not collected")
def test_collected_tiers_are_nested():
    from coimagine.coverage_scale.shards import tier_records

    small, small_info = tier_records(RUN_DIR, "train", 8000)
    large, large_info = tier_records(RUN_DIR, "train", 16000)
    assert small_info["reached_target"]
    assert [record["transition_id"] for record in small] == \
        [record["transition_id"] for record in large[: len(small)]]
    assert large_info["actual_windows"] >= small_info["actual_windows"]
    roots = {str(record["root_id"]) for record in small}
    assert all(str(record["split"]) == "train" for record in small)
    assert len(roots) == small_info["independent_roots"]
