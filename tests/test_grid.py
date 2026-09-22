"""Stage 2 grid world, egocentric sensing and scene rendering."""

from __future__ import annotations

import random

import pytest

from coimagine.grid import (
    HiddenPhysics,
    ObjectState,
    Scene,
    SensorNoise,
    StaticMap,
    apply_noise,
    default_scenes,
    information_state,
    ray_cast,
    render_gallery,
    render_noise_comparison,
    scene_from_ascii,
)
from coimagine.grid.sensing import (
    KIND_FREE,
    KIND_OBJECT,
    KIND_ROBOT,
    KIND_WALL,
    line_cells,
)
from coimagine.grid.state import RobotState, euclidean_sq

OCCLUSION = (
    ".....",
    ".#.OO",
    ".#.OO",
    ".#...",
    ".#...",
    "A...B",
)


def test_scene_from_ascii_parses_map_object_and_robots() -> None:
    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    assert scene.static.width == 5 and scene.static.height == 6
    assert (1, 1) in scene.static.walls and (1, 4) in scene.static.walls
    assert scene.obj.origin == (3, 3)
    assert scene.obj.cells == frozenset({(3, 3), (4, 3), (3, 4), (4, 4)})
    assert {r.robot_id: r.pose for r in scene.robots} == {0: (0, 0), 1: (4, 0)}


def test_object_must_form_a_2x2_block() -> None:
    with pytest.raises(ValueError):
        scene_from_ascii((".....", ".OO..", ".....", "A...B"))


def test_scene_rejects_invalid_placements() -> None:
    static = StaticMap(4, 4, walls=frozenset({(0, 0)}))
    obj = ObjectState((1, 1))
    with pytest.raises(ValueError):
        # Robot 0 stands inside a wall.
        Scene(static, obj, (RobotState(0, (0, 0)), RobotState(1, (3, 3))))
    with pytest.raises(ValueError):
        # Robot 0 stands inside the object.
        Scene(static, obj, (RobotState(0, (1, 1)), RobotState(1, (3, 3))))
    with pytest.raises(ValueError):
        # Duplicate robot ids / missing B cannot be expressed, so build directly.
        Scene(static, obj, (RobotState(0, (2, 2)), RobotState(0, (3, 3))))
    with pytest.raises(ValueError):
        scene_from_ascii((".....", ".OO..", ".OO..", "A.A.."), name="duplicate")


def test_static_map_rejects_out_of_bounds_goal() -> None:
    with pytest.raises(ValueError):
        StaticMap(3, 3, target=frozenset({(9, 9)}))
    with pytest.raises(ValueError):
        StaticMap(3, 3, walls=frozenset({(0, 0)}), target=frozenset({(0, 0)}))


def test_hidden_physics_capacity_accounts_for_traction() -> None:
    grip = HiddenPhysics(resistance=2, constraint="free", traction=("grip", "grip"))
    assert grip.effective_capacity((0, 1)) == 2
    slip = HiddenPhysics(resistance=2, constraint="free", traction=("grip", "slip"))
    assert slip.effective_capacity((0, 1)) == 1
    assert slip.effective_capacity((1,)) == 0


def test_line_cells_is_inclusive_and_contiguous() -> None:
    line = line_cells((0, 0), (3, 1))
    assert line[0] == (0, 0) and line[-1] == (3, 1)
    for a, b in zip(line, line[1:]):
        assert max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def test_ray_cast_stays_within_radius() -> None:
    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    scan = ray_cast(scene, 1, radius=3)
    for cell in scan.visible:
        assert euclidean_sq(cell, scan.origin) <= 9
    assert scan.observed[scan.origin] == KIND_ROBOT


def test_ray_cast_is_occluded_by_a_wall() -> None:
    """Robot A is behind a wall column and must not see the object."""

    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    scan_a = ray_cast(scene, 0, radius=6)
    scan_b = ray_cast(scene, 1, radius=6)
    assert scan_a.observed_object_cells == frozenset()
    assert scan_b.observed_object_cells != frozenset()
    # The occluding wall is itself observed.
    assert (1, 1) in scan_a.observed_wall_cells


def test_ray_cast_reports_object_kind() -> None:
    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    scan_b = ray_cast(scene, 1, radius=6)
    for cell in scan_b.observed_object_cells:
        assert scan_b.observed[cell] == KIND_OBJECT


def test_information_state_merges_history_and_keeps_walls_known() -> None:
    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    first = information_state(scene, ray_cast(scene, 0, radius=2))
    second = information_state(scene, ray_cast(scene, 0, radius=6), first)
    assert len(second.observed) >= len(first.observed)
    assert second.step == first.step + 1
    # Walls are part of the prior map and are never "unknown" or "observed".
    assert all(not scene.static.is_wall(c) for c in second.unknown_cells())
    # The object stays unknown for A even after a wider scan, because of occlusion.
    assert second.observed_object_cells == frozenset()


def test_noise_is_seeded_reproducible_and_can_drop_everything() -> None:
    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    clean = ray_cast(scene, 1, radius=6)
    noise = SensorNoise(drop_prob=0.4, false_free_prob=0.3)

    a = apply_noise(clean, noise, random.Random(7))
    b = apply_noise(clean, noise, random.Random(7))
    assert a.observed == b.observed

    total = SensorNoise(drop_prob=0.999999)
    dropped = apply_noise(clean, total, random.Random(0))
    assert set(dropped.observed) <= {clean.origin}
    assert len(dropped.observed) < len(clean.observed)

    assert apply_noise(clean, SensorNoise(), random.Random(0)) is clean


def test_private_views_differ_across_robots() -> None:
    """The sensing constraint must create genuinely different private evidence."""

    scene = scene_from_ascii(OCCLUSION, name="occlusion")
    scan_a = ray_cast(scene, 0, radius=6)
    scan_b = ray_cast(scene, 1, radius=6)
    assert scan_a.observed_object_cells != scan_b.observed_object_cells


def test_default_scenes_are_valid() -> None:
    names = {scene.static.name for scene in default_scenes()}
    assert names == {"open_transport", "pillar_occlusion", "narrow_passage", "corner_goal"}
    for scene in default_scenes():
        assert isinstance(scene.physics, HiddenPhysics)
        assert scene.hidden_label


def test_gallery_renders_png_and_pdf(tmp_path) -> None:
    paths = render_gallery(
        str(tmp_path), scenes=default_scenes()[:1], formats=("png", "pdf")
    )
    assert len(paths) == 2
    for path in paths:
        from pathlib import Path

        assert Path(path).exists() and Path(path).stat().st_size > 1000


def test_noise_comparison_renders(tmp_path) -> None:
    scene = next(s for s in default_scenes() if s.static.name == "pillar_occlusion")
    paths = render_noise_comparison(scene, str(tmp_path), formats=("png",))
    assert len(paths) == 1
    from pathlib import Path

    assert Path(paths[0]).stat().st_size > 1000
