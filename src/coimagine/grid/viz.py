"""Publication-quality scene rendering for the Stage 2 grid.

Two kinds of panel are provided and they must never be confused:

* **ground truth** -- walls, the true object, both robots and the true ray scan.
  This is a simulator/evaluation view and is labelled as such.
* **information state** -- what one robot actually knows: the prior static map,
  both robot poses, and the object only where its *own* rays hit it. Unobserved
  non-wall cells are drawn as unknown. This is the deployment-visible view.

Everything is vector-friendly (PDF/SVG with editable text) and needs no display.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="coimagine-mpl-"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle  # noqa: E402

from .sensing import (  # noqa: E402
    KIND_FREE,
    KIND_OBJECT,
    KIND_ROBOT,
    KIND_WALL,
    OccupancyScan,
    RobotInformationState,
)
from .state import Cell, Direction, ObjectState, RobotState, Scene, StaticMap  # noqa: E402

# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------

COLOR_WALL = "#3f3f3f"
COLOR_FREE = "#ffffff"
COLOR_TARGET = "#e2f1da"
COLOR_TARGET_EDGE = "#7cae6b"
COLOR_UNKNOWN = "#ededed"
COLOR_OBJECT = "#e08a3c"
COLOR_OBJECT_EDGE = "#8a4b12"
COLOR_GRID = "#d8d8d8"
COLOR_SCAN = "#a9cce3"
COLOR_RAY = "#5b8db8"
COLOR_RANGE = "#5b8db8"
COLOR_TEXT = "#1f1f1f"

ROBOT_COLORS = {0: "#2b6cb0", 1: "#b83227"}
ROBOT_LABELS = {0: "A", 1: "B"}

PANEL_TRUTH_TITLE = "Ground truth"
PANEL_INFO_TITLE = "Robot {label} view (own sensing)"

#: Small caption placed inside the ground-truth panel.
TRUTH_CAPTION = "simulator / evaluation only"


def use_paper_style() -> None:
    """Compact serif style with editable vector text."""

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 7.5,
            "figure.dpi": 140,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def _center(cell: Cell) -> Tuple[float, float]:
    return (cell[0] + 0.5, cell[1] + 0.5)


def _cell_rect(cell: Cell, **kwargs) -> Rectangle:
    return Rectangle((cell[0], cell[1]), 1.0, 1.0, **kwargs)


# --------------------------------------------------------------------------
# Layer drawing
# --------------------------------------------------------------------------


def draw_static_map(
    ax,
    static: StaticMap,
    *,
    unknown_cells: Iterable[Cell] = (),
    show_target: bool = True,
) -> None:
    unknown = frozenset(unknown_cells)
    for cell in static.cells():
        if cell in static.walls:
            face, hatch, edge = COLOR_WALL, None, COLOR_WALL
        elif cell in unknown:
            face, hatch, edge = COLOR_UNKNOWN, "//", "#e0e0e0"
        elif show_target and cell in static.target:
            face, hatch, edge = COLOR_TARGET, None, COLOR_GRID
        else:
            face, hatch, edge = COLOR_FREE, None, COLOR_GRID
        ax.add_patch(
            _cell_rect(
                cell,
                facecolor=face,
                edgecolor=edge,
                linewidth=0.6,
                hatch=hatch,
                zorder=1,
            )
        )

    # Dashed outline around the goal region.
    if show_target and static.target:
        for cell in static.target:
            x, y = cell
            for dx, dy, side in (
                (1, 0, ((x + 1, y), (x + 1, y + 1))),
                (-1, 0, ((x, y), (x, y + 1))),
                (0, 1, ((x, y + 1), (x + 1, y + 1))),
                (0, -1, ((x, y), (x + 1, y))),
            ):
                if (x + dx, y + dy) in static.target:
                    continue
                (x0, y0), (x1, y1) = side
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=COLOR_TARGET_EDGE,
                    linewidth=1.1,
                    linestyle=(0, (4, 3)),
                    zorder=2,
                )


def draw_scan(
    ax,
    scan: OccupancyScan,
    *,
    rays: bool = True,
    zorder: float = 2.0,
    skip: Iterable[Cell] = (),
) -> None:
    skip = frozenset(skip)
    origin = _center(scan.origin)
    for cell in scan.visible:
        if cell == scan.origin or cell in skip:
            continue
        ax.add_patch(
            _cell_rect(
                cell, facecolor=COLOR_SCAN, edgecolor="none", alpha=0.38, zorder=zorder
            )
        )
    if rays:
        for cell in scan.hits:
            ax.plot(
                [origin[0], _center(cell)[0]],
                [origin[1], _center(cell)[1]],
                color=COLOR_RAY,
                linewidth=0.45,
                alpha=0.22,
                zorder=zorder + 0.1,
            )
    ax.add_patch(
        Circle(
            origin,
            scan.radius,
            facecolor="none",
            edgecolor=COLOR_RANGE,
            linewidth=0.8,
            linestyle=(0, (5, 3)),
            alpha=0.75,
            zorder=zorder + 0.2,
        )
    )


def draw_object(
    ax,
    obj: ObjectState,
    *,
    cells: Optional[Iterable[Cell]] = None,
    label: bool = True,
    zorder: float = 4.0,
) -> None:
    """Draw the object, optionally only the cells a robot has actually sensed."""

    draw_cells = frozenset(obj.cells if cells is None else cells)
    if not draw_cells:
        return
    if draw_cells == obj.cells:
        x0 = min(c[0] for c in draw_cells)
        y0 = min(c[1] for c in draw_cells)
        patch = FancyBboxPatch(
            (x0 + 0.06, y0 + 0.06),
            obj.width - 0.12,
            obj.height - 0.12,
            boxstyle="round,pad=0,rounding_size=0.14",
            facecolor=COLOR_OBJECT,
            edgecolor=COLOR_OBJECT_EDGE,
            linewidth=1.4,
            zorder=zorder,
        )
        ax.add_patch(patch)
        if label:
            cx = x0 + obj.width / 2.0
            cy = y0 + obj.height / 2.0
            ax.text(
                cx,
                cy,
                "object",
                ha="center",
                va="center",
                color="white",
                fontsize=8,
                fontweight="bold",
                zorder=zorder + 1,
            )
    else:
        for cell in draw_cells:
            ax.add_patch(
                _cell_rect(
                    cell,
                    facecolor=COLOR_OBJECT,
                    edgecolor=COLOR_OBJECT_EDGE,
                    linewidth=1.1,
                    zorder=zorder,
                )
            )


def draw_robots(
    ax, robots: Sequence[RobotState], *, zorder: float = 6.0, show_heading: bool = True
) -> None:
    for robot in robots:
        center = _center(robot.pose)
        color = ROBOT_COLORS.get(robot.robot_id, "#555555")
        if show_heading and robot.heading is not None:
            dx, dy = robot.heading.delta
            ax.annotate(
                "",
                xy=(center[0] + dx * 0.46, center[1] + dy * 0.46),
                xytext=center,
                arrowprops=dict(arrowstyle="-|>", color=color, linewidth=1.7, shrinkA=0, shrinkB=0),
                zorder=zorder + 0.1,
            )
        ax.add_patch(
            Circle(
                center,
                0.33,
                facecolor=color,
                edgecolor="white",
                linewidth=1.4,
                zorder=zorder,
            )
        )
        ax.text(
            center[0],
            center[1],
            ROBOT_LABELS.get(robot.robot_id, str(robot.robot_id)),
            ha="center",
            va="center",
            color="white",
            fontsize=8,
            fontweight="bold",
            zorder=zorder + 1,
        )


def _finish_axes(ax, static: StaticMap, title: str) -> None:
    ax.set_xlim(-0.05, static.width + 0.05)
    ax.set_ylim(-0.05, static.height + 0.05)
    ax.set_aspect("equal")
    ax.set_xticks(range(static.width + 1))
    ax.set_yticks(range(static.height + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_color("#bbbbbb")
        spine.set_linewidth(0.8)
    ax.set_title(title, pad=6)


# --------------------------------------------------------------------------
# Panels
# --------------------------------------------------------------------------


def draw_ground_truth_panel(
    ax,
    scene: Scene,
    scans: Optional[Mapping[int, OccupancyScan]] = None,
    *,
    show_scan: bool = True,
    show_hidden: bool = True,
    letter: Optional[str] = None,
) -> None:
    draw_static_map(ax, scene.static)
    if show_scan and scans:
        for scan in scans.values():
            draw_scan(ax, scan, skip=scene.static.walls)
    draw_object(ax, scene.obj)
    draw_robots(ax, scene.robots)
    title = PANEL_TRUTH_TITLE if letter is None else f"({letter}) {PANEL_TRUTH_TITLE}"
    title = f"{title} — {TRUTH_CAPTION}"
    _finish_axes(ax, scene.static, title)
    # A muted red frame marks the privileged panel, so it cannot be mistaken for a
    # deployable view in a printed figure.
    for spine in ax.spines.values():
        spine.set_color("#c98b8b")
        spine.set_linewidth(1.2)

    if show_hidden and scene.hidden:
        label = scene.hidden.get("label")
        if label:
            ax.text(
                0.98,
                0.98,
                f"hidden: {label}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=6.5,
                color="#7a2f2f",
                family="monospace",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="#fdf1f1", edgecolor="#e0c0c0", linewidth=0.6),
                zorder=20,
            )


def draw_information_panel(
    ax,
    info: RobotInformationState,
    scan: Optional[OccupancyScan] = None,
    *,
    show_scan: bool = True,
    letter: Optional[str] = None,
) -> None:
    static: StaticMap = info.static  # type: ignore[assignment]
    unknown = info.unknown_cells()
    draw_static_map(ax, static, unknown_cells=unknown)
    if show_scan and scan is not None:
        draw_scan(ax, scan, skip=static.walls)
    sensed_object = info.observed_object_cells
    if sensed_object:
        # Reconstruct a minimal 2x2 origin so the rounded body can be drawn.
        ox = min(c[0] for c in sensed_object)
        oy = min(c[1] for c in sensed_object)
        draw_object(
            ax,
            ObjectState((ox, oy)),
            cells=sensed_object,
            label=False,
        )
    robots = (
        RobotState(info.robot_id, info.own_pose, None),
        RobotState(1 - info.robot_id, info.teammate_pose, None),
    )
    draw_robots(ax, robots)
    title = PANEL_INFO_TITLE.format(label=ROBOT_LABELS[info.robot_id])
    if letter is not None:
        title = f"({letter}) {title}"
    _finish_axes(ax, static, title)


def shared_legend() -> List[Line2D]:
    """Proxy artists for a single shared legend under a multi-panel figure."""

    return [
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=COLOR_WALL, markeredgecolor=COLOR_WALL, label="wall (prior)"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=COLOR_TARGET, markeredgecolor=COLOR_TARGET_EDGE, label="goal region"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=COLOR_OBJECT, markeredgecolor=COLOR_OBJECT_EDGE, label="object (sensed)"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=COLOR_UNKNOWN, markeredgecolor=COLOR_GRID, label="unobserved"),
        Line2D([], [], marker="s", linestyle="none", markersize=7,
               markerfacecolor=COLOR_SCAN, markeredgecolor="none", alpha=0.6, label="lidar field of view"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=ROBOT_COLORS[0], markeredgecolor="white", label="robot A"),
        Line2D([], [], marker="o", linestyle="none", markersize=7,
               markerfacecolor=ROBOT_COLORS[1], markeredgecolor="white", label="robot B"),
    ]


# --------------------------------------------------------------------------
# Figure assembly
# --------------------------------------------------------------------------


def render_scene_panels(
    scene: Scene,
    infos: Mapping[int, RobotInformationState],
    scans: Optional[Mapping[int, OccupancyScan]] = None,
    *,
    out_dir: str = "outputs/grid/scenes",
    stem: str = "scene",
    formats: Sequence[str] = ("png", "pdf"),
    suptitle: Optional[str] = None,
    legend: bool = True,
) -> List[str]:
    """Ground truth plus each robot's information state, side by side."""

    use_paper_style()
    scans = scans or {}
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.35))

    draw_ground_truth_panel(axes[0], scene, scans, letter="a")
    for ax, robot_id, letter in zip(axes[1:], (0, 1), ("b", "c")):
        draw_information_panel(ax, infos[robot_id], scans.get(robot_id), letter=letter)

    if suptitle:
        fig.suptitle(suptitle, fontsize=10.5, y=1.02)
    if legend:
        fig.legend(
            handles=shared_legend(),
            loc="upper center",
            ncol=4,
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
    fig.subplots_adjust(wspace=0.08)
    fig.tight_layout()
    return save_figure(fig, out_dir, stem, formats=formats)


def save_figure(fig, out_dir: str, stem: str, *, formats: Sequence[str] = ("png", "pdf")) -> List[str]:
    """Save a figure in several formats and return the written paths."""

    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: List[str] = []
    for fmt in formats:
        path = directory / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        paths.append(str(path))
    plt.close(fig)
    return paths
