"""Render the Stage 2 scene gallery.

This is the visual deliverable: for every scene it draws

1. the simulator ground truth (walls, true object, robots, true ray scan, hidden
   physics label) -- evaluation only;
2. robot A's information state;
3. robot B's information state,

so the information boundary is visible at a glance: the prior map is shared, the
object is only known where that robot's own rays hit it.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .scenes import default_scenes
from .sensing import (
    NOISELESS,
    OccupancyScan,
    RobotInformationState,
    SensorNoise,
    apply_noise,
    information_state,
    ray_cast,
)
from .state import Scene
from .viz import render_scene_panels, save_figure, use_paper_style


def _scan_and_info(
    scene: Scene, radius: int, noise: SensorNoise, seed: int
) -> tuple:
    rng = random.Random(seed)
    scans: Dict[int, OccupancyScan] = {}
    infos: Dict[int, RobotInformationState] = {}
    for robot in scene.robots:
        scan = ray_cast(scene, robot.robot_id, radius)
        if not noise.is_noiseless:
            scan = apply_noise(scan, noise, rng)
        scans[robot.robot_id] = scan
        infos[robot.robot_id] = information_state(scene, scan)
    return scans, infos


def render_gallery(
    out_dir: str = "outputs/grid/scenes",
    *,
    scenes: Optional[Sequence[Scene]] = None,
    radius: int = 6,
    noise: SensorNoise = NOISELESS,
    seed: int = 0,
    formats: Sequence[str] = ("png", "pdf"),
) -> List[str]:
    """Render one three-panel figure per scene; returns the written paths."""

    scenes = scenes or default_scenes()
    paths: List[str] = []
    for scene in scenes:
        scans, infos = _scan_and_info(scene, radius, noise, seed)
        title = f"Stage 2 scene: {scene.static.name}"
        if not noise.is_noiseless:
            title += " (noisy sensing)"
        paths.extend(
            render_scene_panels(
                scene,
                infos,
                scans,
                out_dir=out_dir,
                stem=f"scene_{scene.static.name}",
                formats=formats,
                suptitle=title,
            )
        )
    return paths


def render_noise_comparison(
    scene: Scene,
    out_dir: str = "outputs/grid/scenes",
    *,
    radius: int = 6,
    noise: Optional[SensorNoise] = None,
    seed: int = 0,
    robot_id: int = 1,
    formats: Sequence[str] = ("png", "pdf"),
) -> List[str]:
    """One robot: noiseless information state versus a noisy realisation."""

    use_paper_style()
    import matplotlib.pyplot as plt

    from .viz import draw_ground_truth_panel, draw_information_panel, shared_legend

    noise = noise or SensorNoise(drop_prob=0.4, false_free_prob=0.3)
    clean_scan = ray_cast(scene, robot_id, radius)
    clean_info = information_state(scene, clean_scan)
    rng = random.Random(seed)
    noisy_scan = apply_noise(clean_scan, noise, rng)
    noisy_info = information_state(scene, noisy_scan)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.35))
    scan_map = {r.robot_id: ray_cast(scene, r.robot_id, radius) for r in scene.robots}
    draw_ground_truth_panel(axes[0], scene, scan_map, letter="a")
    draw_information_panel(axes[1], clean_info, clean_scan, letter="b")
    axes[1].set_title(f"(b) Robot {robot_id}: noiseless sensing", pad=6)
    draw_information_panel(axes[2], noisy_info, noisy_scan, letter="c")
    axes[2].set_title(
        f"(c) Robot {robot_id}: noisy "
        f"(drop={noise.drop_prob:.2f}, false-free={noise.false_free_prob:.2f})",
        pad=6,
    )
    fig.legend(
        handles=shared_legend(), loc="upper center", ncol=4, frameon=False,
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.subplots_adjust(wspace=0.08)
    fig.tight_layout()
    stem = f"noise_comparison_{scene.static.name}"
    return save_figure(fig, out_dir, stem, formats=formats)
