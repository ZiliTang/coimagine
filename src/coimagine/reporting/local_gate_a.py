from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from coimagine.core.types import (
    Direction,
    FactorSignature,
    JointOption,
    LocalOption,
    OptionKind,
    PlannerMode,
)
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import ScenarioInstance
from coimagine.envs.grid.state import GridState
from coimagine.evaluation.episodes import EpisodeTrace
from coimagine.evaluation.records import EpisodeResultRecord, ScalingResult
from coimagine.options.generator import GridOptionGenerator
from coimagine.planning.exact_search import ExactTopologySearch
from coimagine.reporting.figures import save_all_formats
from coimagine.reporting.semantics import format_joint_option_compact
from coimagine.visualization.matplotlib_grid import draw_state


def _draw_visibility(ax, state: GridState, observation_radius: int) -> None:
    env = GridEnvironment(observation_radius=observation_radius)
    facecolors = ("#8FBCE6", "#F2B36F", "#9ACD8A")
    for index, robot_id in enumerate(sorted(state.robots)):
        obs = env.observe(state, robot_id)
        for cell in obs.visible_cells:
            ax.add_patch(
                Rectangle(
                    cell,
                    1,
                    1,
                    facecolor=facecolors[index % len(facecolors)],
                    edgecolor="none",
                    alpha=0.075,
                    zorder=-0.2,
                )
            )


def _draw_visibility_rays(ax, state: GridState, observation_radius: int) -> None:
    """Draw a sparse subset of the actual occlusion-terminated sensing rays."""

    env = GridEnvironment(observation_radius=observation_radius)
    ray_colors = ("#2F6FA3", "#B66A18", "#4F8A42")
    ray_count = 16
    for index, robot_id in enumerate(sorted(state.robots)):
        own = state.robots[robot_id].position
        visible = env.visible_cells_from_pose(state, own, exclude_robot=robot_id)
        color = ray_colors[index % len(ray_colors)]
        for angle in np.linspace(0.0, 2.0 * np.pi, ray_count, endpoint=False):
            target = (
                int(round(own[0] + observation_radius * np.cos(angle))),
                int(round(own[1] + observation_radius * np.sin(angle))),
            )
            line = env._line_cells(own, target)  # deterministic Gate-A ray caster
            endpoint = own
            for cell in line[1:]:
                if cell not in visible:
                    break
                endpoint = cell
            ax.plot(
                [own[0] + 0.5, endpoint[0] + 0.5],
                [own[1] + 0.5, endpoint[1] + 0.5],
                color=color,
                linewidth=0.42,
                alpha=0.32,
                zorder=1.5,
            )


def _draw_channel_geometry(ax, state: GridState) -> None:
    channel_y = state.task.metadata.get("channel_y")
    if channel_y is None:
        return
    y = int(channel_y)
    ax.plot(
        [0.15, state.width - 0.15],
        [y, y],
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        color="#666666",
        zorder=4,
    )
    ax.plot(
        [0.15, state.width - 0.15],
        [y + 1, y + 1],
        linestyle=(0, (3, 2)),
        linewidth=0.8,
        color="#666666",
        zorder=4,
    )
    x_arrow = state.width - 1.35
    arrow_style = {
        "arrowstyle": "-|>",
        "lw": 0.38,
        "color": "#555555",
        "mutation_scale": 5.0,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
    }
    ax.annotate(
        "",
        xy=(x_arrow, y + 3.15),
        xytext=(x_arrow, y + 1.08),
        arrowprops=arrow_style,
        zorder=9,
    )
    ax.annotate(
        "",
        xy=(x_arrow - 0.72, y - 2.15),
        xytext=(x_arrow - 0.72, y - 0.08),
        arrowprops=arrow_style,
        zorder=9,
    )


def _draw_reposition_anchors(
    ax,
    state: GridState,
    action: JointOption | None,
    *,
    available_by_robot: Mapping[int, set[tuple[int, int]]] | None = None,
) -> None:
    raw = state.task.metadata.get("reposition_anchors", ())
    selected = {
        option.target
        for option in (action.options if action is not None else ())
        if option.kind == OptionKind.REPOSITION and option.target is not None
    }
    for item in raw:  # type: ignore[union-attr]
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        cell = (int(item[0]), int(item[1]))
        x, y = cell
        if cell in selected:
            ax.scatter(
                [x + 0.5],
                [y + 0.5],
                marker="*",
                s=46,
                facecolor="#2E8B57",
                edgecolor="black",
                linewidth=0.55,
                zorder=13,
            )
        else:
            ax.scatter(
                [x + 0.5],
                [y + 0.5],
                marker="o",
                s=12,
                facecolor="white",
                edgecolor="#555555",
                linewidth=0.55,
                zorder=7,
            )
        if available_by_robot is not None:
            owners = [
                robot_id
                for robot_id, available in sorted(available_by_robot.items())
                if cell in available
            ]
            if owners:
                owner_colors = {0: "#3F7CAC", 1: "#D28B35"}
                for ring_index, robot_id in enumerate(owners):
                    ax.scatter(
                        [x + 0.5],
                        [y + 0.5],
                        marker="o",
                        s=22 + 14 * ring_index,
                        facecolor="none",
                        edgecolor=owner_colors.get(robot_id, "#6C6C6C"),
                        linewidth=0.8,
                        zorder=10 + ring_index,
                    )


def _draw_action(ax, state: GridState, action: JointOption | None) -> None:
    if action is None:
        return
    for option in action.options:
        if option.kind == OptionKind.IDLE:
            continue
        start = state.robots[option.robot_id].position
        sx, sy = start[0] + 0.5, start[1] + 0.5
        if option.kind in {OptionKind.REPOSITION, OptionKind.APPROACH} and option.target:
            tx, ty = option.target[0] + 0.5, option.target[1] + 0.5
            linestyle = "--" if option.kind == OptionKind.REPOSITION else ":"
            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(sx, sy),
                arrowprops={
                    "arrowstyle": "->",
                    "lw": 0.42,
                    "linestyle": linestyle,
                    "mutation_scale": 5.0,
                    "shrinkA": 3.0,
                    "shrinkB": 2.0,
                },
                zorder=8,
            )
        elif option.kind == OptionKind.PUSH and option.direction:
            dx, dy = option.direction.delta
            ax.annotate(
                "",
                xy=(sx + 1.85 * dx, sy + 1.85 * dy),
                xytext=(sx + 0.18 * dx, sy + 0.18 * dy),
                arrowprops={
                    "arrowstyle": "-|>",
                    "lw": 0.48,
                    "mutation_scale": 5.4,
                    "shrinkA": 0.0,
                    "shrinkB": 0.0,
                },
                zorder=8,
            )


def _topology_counts(
    state: GridState,
    *,
    observation_radius: int,
    candidate_budget: int = 8,
    max_arity: int = 3,
) -> tuple[int, int]:
    env = GridEnvironment(observation_radius=observation_radius)
    planner = ExactTopologySearch(
        env,
        GridOptionGenerator(candidate_budget=candidate_budget),
        horizon=1,
        max_arity=max_arity,
    )
    beliefs = planner.initialize_beliefs(state)
    candidates = planner._candidate_sets(beliefs)
    root = planner.admission.build_root_structure(state, candidates)
    admission = planner._admit_at_depth(
        state,
        beliefs,
        candidates,
        PlannerMode.IET,
        root,
        depth=0,
    )
    vertices = sum(len(options) for options in candidates.values())
    return vertices, len(admission.joint_options)


def _panel(
    ax,
    state: GridState,
    action: JointOption | None,
    *,
    title: str,
    observation_radius: int,
    outcome: str | None = None,
    callout: str | None = None,
) -> None:
    _draw_visibility(ax, state, observation_radius)
    draw_state(ax, state, title=title, show_grid=True)
    _draw_action(ax, state, action)
    for robot_id, robot in sorted(state.robots.items()):
        x, y = robot.position
        label_dx = -0.14 if robot_id % 2 == 0 else 0.14
        ax.text(
            x + 0.5 + label_dx,
            y - 0.12,
            f"R{robot_id}",
            ha="center",
            va="top",
            fontsize=6.6,
            color="black",
            fontweight="bold",
            zorder=12,
        )
    vertices, edges = _topology_counts(
        state,
        observation_radius=observation_radius,
    )
    ax.text(
        0.5,
        -0.075,
        f"Topology  |V|={vertices} · |E|={edges}",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.8,
    )
    if action is not None:
        ax.text(
            0.5,
            1.005,
            format_joint_option_compact(action),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=6.1,
        )
    if callout:
        ax.text(
            0.03,
            0.96,
            callout,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.9,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.82},
            zorder=18,
        )
    if outcome:
        ax.text(
            0.5,
            0.5,
            outcome,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9.0,
            fontweight="bold",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.88},
            zorder=20,
        )


def _trace_states(trace: EpisodeTrace) -> list[tuple[GridState, JointOption | None]]:
    panels = [(step.state_before, step.selected_action) for step in trace.steps[:3]]
    while len(panels) < 3:
        panels.append((trace.final_state, None))
    panels.append((trace.final_state, None))
    return panels


def plot_local_imagination_story(
    iet: EpisodeTrace,
    cost: EpisodeTrace,
    output: Path,
    *,
    observation_radius: int = 6,
) -> None:
    """Double-column CST-versus-IET storyboard with observation/topology ribbons."""

    iet_panels = _trace_states(iet)
    cost_panels = _trace_states(cost)
    titles = ("Boundary 0", "Boundary 1", "Boundary 2", "Closed-loop outcome")
    fig, axes = plt.subplots(2, 4, figsize=(7.16, 4.75))
    iet_callouts = (
        "Current support:\none contact",
        "NEW OPTION:\nsecond contact",
        "NEW EDGE:\ncoordinated Push",
        None,
    )
    cost_callouts = (
        "Current support:\none contact",
        "Future support\nremains frozen",
        "No coordinated\nPush edge",
        None,
    )
    for col, ((state, action), title) in enumerate(zip(iet_panels, titles)):
        outcome = None
        if col == 3:
            outcome = (
                f"COMPLETE\n{iet.option_count} executed options"
                if iet.completed
                else f"INCOMPLETE\n{iet.option_count} executed options"
            )
        _panel(
            axes[0, col],
            state,
            action,
            title=title,
            observation_radius=observation_radius,
            outcome=outcome,
            callout=iet_callouts[col],
        )
    for col, ((state, action), title) in enumerate(zip(cost_panels, titles)):
        outcome = None
        if col == 3:
            outcome = (
                f"COMPLETE\n{cost.option_count} executed options"
                if cost.completed
                else f"STALLED\n{cost.option_count} executed options"
            )
        _panel(
            axes[1, col],
            state,
            action,
            title=title,
            observation_radius=observation_radius,
            outcome=outcome,
            callout=cost_callouts[col],
        )

    fig.text(
        0.012,
        0.70,
        "CoImagine–IET (H=3)",
        rotation=90,
        va="center",
        ha="left",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(
        0.012,
        0.27,
        "CST (H=3)",
        rotation=90,
        va="center",
        ha="left",
        fontsize=8.2,
        fontweight="bold",
    )
    fig.suptitle(
        "Gate A: imagined local observations create future option support and coordination topology",
        y=0.995,
        fontsize=8.5,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.505,
        "IET sequence: Reposition/Approach  →  Approach(new contact)  →  coordinated Push",
        ha="center",
        va="center",
        fontsize=6.6,
    )
    fig.text(
        0.5,
        0.055,
        "CST propagates state and executability, but future option vertices remain restricted to current observation support.",
        ha="center",
        va="center",
        fontsize=6.4,
    )
    fig.subplots_adjust(left=0.055, right=0.995, top=0.90, bottom=0.10, wspace=0.12, hspace=0.58)
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_local_imagination_story", dpi=300)
    plt.close(fig)


@dataclass(frozen=True)
class _StoryPanel:
    state: GridState
    action: JointOption | None
    title: str
    callout: str


def _idle(robot_id: int) -> LocalOption:
    return LocalOption(
        option_id=f"story-r{robot_id}-idle",
        robot_id=robot_id,
        kind=OptionKind.IDLE,
        role_token="idle",
        slot_token="idle",
    )


def _move(
    robot_id: int,
    kind: OptionKind,
    target: tuple[int, int],
    *,
    entity_id: str | None = None,
    contact: tuple[int, int] | None = None,
    slot: str = "story",
) -> LocalOption:
    return LocalOption(
        option_id=f"story-r{robot_id}-{kind.value}-{target}",
        robot_id=robot_id,
        kind=kind,
        entity_id=entity_id,
        direction=Direction.NORTH if entity_id is not None else None,
        target=target,
        contact_cell=contact,
        role_token=("reposition" if kind == OptionKind.REPOSITION else "approach_contact"),
        slot_token=slot,
        signature=FactorSignature(
            entity_ids=frozenset({entity_id}) if entity_id else frozenset(),
            resource_tokens=frozenset({f"cell:{target[0]}:{target[1]}"}),
        ),
    )


def _push(
    robot_id: int,
    entity_id: str,
    contact: tuple[int, int],
    support: tuple[int, int],
    *,
    slot: str,
) -> LocalOption:
    return LocalOption(
        option_id=f"story-r{robot_id}-push-{entity_id}-{contact}",
        robot_id=robot_id,
        kind=OptionKind.PUSH,
        entity_id=entity_id,
        direction=Direction.NORTH,
        target=support,
        contact_cell=contact,
        role_token="push",
        slot_token=slot,
        signature=FactorSignature(
            entity_ids=frozenset({entity_id}),
            resource_tokens=frozenset({f"cell:{support[0]}:{support[1]}"}),
            temporal_tokens=frozenset({f"sync:{entity_id}"}),
        ),
    )


def _execute_story(env: GridEnvironment, state: GridState, action: JointOption) -> GridState:
    transition = env.execute(state, action)
    if not transition.result.success:
        raise RuntimeError(
            f"Invalid channel-story action {action.template_key}: "
            f"{transition.result.failure_type}"
        )
    return transition.state


def _channel_storyboards(
    instance: ScenarioInstance,
    observation_radius: int,
) -> tuple[list[_StoryPanel], list[_StoryPanel], GridState]:
    """Construct matched feasible key-boundary sequences for the paper figure.

    The quantitative CST/IET experiments remain planner-generated.  This figure
    intentionally selects five representative exact boundaries so the mechanism,
    factors, and evolving option support remain legible at double-column width.
    """

    env = GridEnvironment(observation_radius=observation_radius)
    s0 = instance.state.clone()
    common = JointOption(
        (
            _push(0, "B0", (4, 7), (4, 6), slot="north:0"),
            _idle(1),
        )
    )
    s1 = _execute_story(env, s0, common)

    iet_reposition = JointOption(
        (
            _idle(0),
            _move(1, OptionKind.REPOSITION, (12, 4), slot="anchor:12:4"),
        )
    )
    i2 = _execute_story(env, s1, iet_reposition)
    iet_approach = JointOption(
        (
            _move(0, OptionKind.APPROACH, (11, 6), entity_id="H", contact=(11, 7), slot="north:0"),
            _move(1, OptionKind.APPROACH, (12, 6), entity_id="H", contact=(12, 7), slot="north:1"),
        )
    )
    i3 = _execute_story(env, i2, iet_approach)
    iet_push = JointOption(
        (
            _push(0, "H", (11, 7), (11, 6), slot="north:0"),
            _push(1, "H", (12, 7), (12, 6), slot="north:1"),
        )
    )
    i4 = _execute_story(env, i3, iet_push)

    cost_approach = JointOption(
        (
            _idle(0),
            _move(1, OptionKind.APPROACH, (7, 6), entity_id="B1", contact=(7, 7), slot="north:0"),
        )
    )
    c2 = _execute_story(env, s1, cost_approach)
    cost_push = JointOption(
        (
            _idle(0),
            _push(1, "B1", (7, 7), (7, 6), slot="north:0"),
        )
    )
    c3 = _execute_story(env, c2, cost_push)
    cost_heavy = JointOption(
        (
            _move(0, OptionKind.APPROACH, (11, 6), entity_id="H", contact=(11, 7), slot="north:0"),
            _idle(1),
        )
    )
    c4 = _execute_story(env, c3, cost_heavy)

    iet = [
        _StoryPanel(s0, common, "Shared start", "Both clear a currently grounded blocker"),
        _StoryPanel(s1, iet_reposition, "Viewpoint choice", "IET values a future observation"),
        _StoryPanel(i2, iet_approach, "New option support", "Second heavy-contact option appears"),
        _StoryPanel(i3, iet_push, "Evolving topology", "New two-robot Push hyperedge"),
        _StoryPanel(i4, None, "Critical outcome", "Heavy gate removed; passage progresses"),
    ]
    cost = [
        _StoryPanel(s0, common, "Shared start", "Both clear a currently grounded blocker"),
        _StoryPanel(s1, cost_approach, "Current-support choice", "CST prepares a visible light blocker"),
        _StoryPanel(c2, cost_push, "Local progress", "Current contact is executed successfully"),
        _StoryPanel(c3, cost_heavy, "Frozen support", "Only one heavy contact is grounded"),
        _StoryPanel(c4, None, "Critical outcome", "No coordinated Push edge for the gate"),
    ]
    return iet, cost, s1


def _option_symbol(option: LocalOption) -> str:
    return {
        OptionKind.REPOSITION: "R",
        OptionKind.APPROACH: "A",
        OptionKind.PUSH: "P",
        OptionKind.HOLD: "H",
        OptionKind.OBSERVE: "O",
        OptionKind.IDLE: "I",
    }[option.kind]


def _factor_group(option: LocalOption) -> str:
    if option.entity_id:
        return option.entity_id
    if option.kind == OptionKind.REPOSITION:
        return "Nav"
    return "Other"


def _draw_topology_strip(
    ax,
    state: GridState,
    action: JointOption | None,
    *,
    observation_radius: int,
    reference_templates: Mapping[int, frozenset[tuple]],
    highlight_new: bool = True,
) -> None:
    env = GridEnvironment(observation_radius=observation_radius)
    planner = ExactTopologySearch(
        env,
        GridOptionGenerator(candidate_budget=10),
        horizon=1,
        max_arity=3,
    )
    beliefs = planner.initialize_beliefs(state)
    candidates = planner._candidate_sets(beliefs)
    root = planner.admission.build_root_structure(state, candidates)
    admission = planner._admit_at_depth(
        state, beliefs, candidates, PlannerMode.IET, root, depth=0
    )
    groups = ("Nav", "B0", "B1", "H", "B3")
    selected = action.by_robot() if action is not None else {}
    ax.set_xlim(-0.8, len(groups))
    ax.set_ylim(-0.15, 2.15)
    ax.axis("off")
    for gx, group in enumerate(groups):
        ax.add_patch(
            Rectangle(
                (gx + 0.03, 0.05),
                0.94,
                1.82,
                facecolor="#FAFAFA",
                edgecolor="#C7C7C7",
                linewidth=0.55,
            )
        )
        ax.text(gx + 0.5, 1.96, group, ha="center", va="bottom", fontsize=4.7, fontweight="bold")
    for lane, robot_id in enumerate((0, 1)):
        y = 1.35 - lane * 0.78
        ax.text(-0.08, y, f"R{robot_id}", ha="right", va="center", fontsize=5.0, fontweight="bold")
        per_group: dict[str, list[LocalOption]] = {group: [] for group in groups}
        for option in candidates.get(robot_id, ()):
            group = _factor_group(option)
            if group in per_group and option.kind != OptionKind.IDLE:
                per_group[group].append(option)
        for gx, group in enumerate(groups):
            options = per_group[group][:3]
            for k, option in enumerate(options):
                x = gx + 0.27 + 0.23 * k
                is_new = highlight_new and option.template_key not in reference_templates.get(robot_id, frozenset())
                is_selected = (
                    robot_id in selected
                    and selected[robot_id].template_key == option.template_key
                )
                ax.scatter(
                    [x],
                    [y],
                    s=34 if is_selected else 25,
                    marker="o",
                    facecolor="#A9D7B5" if is_new else "white",
                    edgecolor="black" if is_selected else ("#2E8B57" if is_new else "#777777"),
                    linewidth=1.0 if is_selected else 0.65,
                    zorder=5,
                )
                ax.text(x, y, _option_symbol(option), ha="center", va="center", fontsize=4.2, zorder=6)

    has_heavy_edge = any(
        sum(
            option.kind == OptionKind.PUSH and option.entity_id == "H"
            for option in joint.options
        )
        >= 2
        for joint in admission.joint_options
    )
    if has_heavy_edge:
        x = groups.index("H") + 0.5
        ax.plot([x, x], [0.58, 1.34], color="#2E8B57", linewidth=1.8, zorder=3)
        ax.text(x + 0.08, 0.96, "2R", ha="left", va="center", fontsize=4.5, color="#2E8B57", fontweight="bold")
    ax.text(
        0.995,
        -0.04,
        f"|V|={sum(len(v) for v in candidates.values())}, |E|={len(admission.joint_options)}, factors={admission.factor_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=4.6,
        color="#444444",
    )


def _draw_channel_panel(
    ax,
    panel: _StoryPanel,
    *,
    observation_radius: int,
) -> None:
    _draw_visibility(ax, panel.state, observation_radius)
    draw_state(
        ax,
        panel.state,
        title="",
        show_grid=True,
        show_robot_ids_inside=False,
        entity_labels_above=True,
    )
    ax.set_title(panel.title, fontsize=7.0, pad=7.0)
    _draw_visibility_rays(ax, panel.state, observation_radius)
    _draw_channel_geometry(ax, panel.state)
    _draw_reposition_anchors(ax, panel.state, panel.action)
    _draw_action(ax, panel.state, panel.action)
    for robot_id, robot in sorted(panel.state.robots.items()):
        label_dx = 0.16 if robot_id % 2 == 0 else -0.16
        ax.text(
            robot.position[0] + 0.5 + label_dx,
            robot.position[1] - 0.12,
            f"R{robot_id}",
            ha="center",
            va="top",
            fontsize=4.6,
            color="black",
            fontweight="bold",
            zorder=15,
        )
    ax.text(
        0.02,
        0.985,
        panel.callout,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=4.2,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "alpha": 0.84, "edgecolor": "#777777", "linewidth": 0.45},
        zorder=20,
    )
    if panel.action is not None:
        ax.text(
            0.5,
            1.005,
            format_joint_option_compact(panel.action),
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=4.6,
        )


def plot_channel_imagination_story(
    instance: ScenarioInstance,
    output: Path,
    *,
    observation_radius: int = 10,
) -> None:
    """Five-boundary multi-blocker CST/IET main figure with factor topology."""

    iet, cost, reference_state = _channel_storyboards(instance, observation_radius)
    env = GridEnvironment(observation_radius=observation_radius)
    planner = ExactTopologySearch(env, GridOptionGenerator(candidate_budget=10), horizon=1)
    reference_beliefs = planner.initialize_beliefs(reference_state)
    reference_candidates = planner._candidate_sets(reference_beliefs)
    reference_templates = {
        rid: frozenset(option.template_key for option in options)
        for rid, options in reference_candidates.items()
    }

    fig = plt.figure(figsize=(7.16, 4.30))
    grid = fig.add_gridspec(
        4,
        5,
        height_ratios=(3.55, 0.92, 3.55, 0.92),
        left=0.052,
        right=0.995,
        top=0.885,
        bottom=0.125,
        wspace=0.095,
        hspace=0.075,
    )
    for col, panel in enumerate(iet):
        map_ax = fig.add_subplot(grid[0, col])
        topo_ax = fig.add_subplot(grid[1, col])
        _draw_channel_panel(map_ax, panel, observation_radius=observation_radius)
        _draw_topology_strip(
            topo_ax,
            panel.state,
            panel.action,
            observation_radius=observation_radius,
            reference_templates=reference_templates,
            highlight_new=col >= 2,
        )
    for col, panel in enumerate(cost):
        map_ax = fig.add_subplot(grid[2, col])
        topo_ax = fig.add_subplot(grid[3, col])
        _draw_channel_panel(map_ax, panel, observation_radius=observation_radius)
        _draw_topology_strip(
            topo_ax,
            panel.state,
            panel.action,
            observation_radius=observation_radius,
            reference_templates=reference_templates,
            highlight_new=False,
        )

    fig.suptitle(
        "Gate A: local-observation imagination creates future option support and an evolving coordination topology",
        y=0.975,
        fontsize=8.2,
        fontweight="bold",
    )
    fig.text(0.012, 0.692, "CoImagine–IET (H=3)", rotation=90, va="center", ha="left", fontsize=7.7, fontweight="bold")
    fig.text(0.012, 0.303, "CST (H=3)", rotation=90, va="center", ha="left", fontsize=7.7, fontweight="bold")
    handles = [
        Patch(facecolor="#8FBCE6", alpha=0.18, edgecolor="none", label="R0 visible region"),
        Patch(facecolor="#F2B36F", alpha=0.18, edgecolor="none", label="R1 visible region"),
        Line2D([0], [0], color="#555555", linestyle=(0, (3, 2)), lw=0.9, label="channel boundary"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#555555", label="reposition anchor"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#A9D7B5", markeredgecolor="#2E8B57", label="new option vertex"),
        Line2D([0], [0], color="#2E8B57", lw=1.8, label="coordinated Push edge"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, fontsize=4.9, bbox_to_anchor=(0.52, 0.010))
    fig.text(
        0.5,
        0.077,
        "Topology strips group robot-owned options by entity/resource factor; green vertices are unsupported by the current observation but appear after IET rollout.",
        ha="center",
        va="center",
        fontsize=5.1,
    )
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_channel_imagination_story", dpi=320)
    plt.close(fig)


def _method_color(method: str) -> str:
    if method == "CST-Frozen-K":
        return "#9AA1A8"
    if method == "CST":
        return "#59636E"
    if method == "CoImagine-IET":
        return "#168F83"
    return "#7656A5"


def _method_label(method: str) -> str:
    if method == "CoImagine-IET":
        return "IET"
    if method.startswith("Privileged"):
        return "Oracle"
    return method


def plot_horizon_summary(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    """Completion and robust timing summaries without twin-axis overlap."""

    aggregate = [row for row in rows if row["scene_type"] == "open"]
    methods = ("CST", "CoImagine-IET", "Privileged IET Oracle")
    horizons = (2, 3)
    lookup = {(str(row["method"]), int(row["horizon"])): row for row in aggregate}
    x = np.arange(len(methods), dtype=float)
    width = 0.33

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.65),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )
    ax = axes[0]
    for offset, horizon in zip((-width / 2, width / 2), horizons):
        values = [float(lookup[(method, horizon)]["completion_rate"]) for method in methods]
        lower = [
            max(0.0, value - float(lookup[(method, horizon)]["completion_ci95_low"]))
            for method, value in zip(methods, values)
        ]
        upper = [
            max(0.0, float(lookup[(method, horizon)]["completion_ci95_high"]) - value)
            for method, value in zip(methods, values)
        ]
        bars = ax.bar(
            x + offset,
            values,
            width,
            color=[_method_color(method) for method in methods],
            alpha=0.62 if horizon == 2 else 0.96,
            hatch="///" if horizon == 2 else None,
            edgecolor="#F7F7F7" if horizon == 2 else "#333333",
            linewidth=0.7,
            yerr=np.asarray([lower, upper]),
            capsize=2.2,
            label=f"H={horizon}",
            zorder=3,
        )
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                min(1.055, value + 0.04),
                f"{value:.0%}",
                ha="center",
                va="bottom",
                fontsize=6.7,
            )
    ax.set_ylim(0.0, 1.12)
    ax.set_ylabel("Completion rate")
    ax.set_xticks(x, tuple(_method_label(method) for method in methods), fontsize=7.3)
    ax.grid(axis="y", alpha=0.22, zorder=0)
    ax.legend(frameon=False, fontsize=7.0, ncol=2, loc="upper left")
    open_runs = int(lookup[(methods[0], horizons[0])]["runs"])
    ax.set_title(f"{open_runs} open randomized tasks (95% Wilson CI)", fontsize=8.1)

    ax = axes[1]
    for method in methods:
        medians = [
            float(lookup[(method, horizon)]["median_planning_seconds"]) * 1000.0
            for horizon in horizons
        ]
        low = [
            (float(lookup[(method, horizon)]["median_planning_seconds"]) - float(lookup[(method, horizon)]["planning_q25_seconds"])) * 1000.0
            for horizon in horizons
        ]
        high = [
            (float(lookup[(method, horizon)]["planning_q75_seconds"]) - float(lookup[(method, horizon)]["median_planning_seconds"])) * 1000.0
            for horizon in horizons
        ]
        ax.errorbar(
            horizons,
            medians,
            yerr=np.asarray([low, high]),
            marker="o",
            markersize=4.5,
            linewidth=1.6,
            capsize=2.5,
            color=_method_color(method),
            label=_method_label(method),
        )
    ax.set_xticks(horizons, ("H=2", "H=3"))
    ax.set_ylabel("Planning time / option [ms]")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, fontsize=6.8)
    ax.set_title("Median and interquartile range", fontsize=8.1)

    fig.suptitle(
        "Horizon benefit and computational cost",
        fontsize=9.0,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.35)
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_horizon_comparison", dpi=320)
    plt.close(fig)


def plot_dependency_summary(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    """Causal-family completion with Wilson intervals instead of a binary heatmap."""

    family_order = (
        "direct_push",
        "approach_push",
        "mixed_approach_push",
        "reposition_approach_push",
    )
    family_labels = (
        "Direct\nPush",
        "Approach\n→ Push",
        "Mixed setup\n→ Push",
        "Reposition → Approach\n→ Push",
    )
    series = (
        ("CST", 2, "o", "--"),
        ("CST", 3, "s", "-"),
        ("CoImagine-IET", 2, "o", "--"),
        ("CoImagine-IET", 3, "s", "-"),
        ("Privileged IET Oracle", 3, "D", "-"),
    )
    lookup = {
        (str(row["method"]), int(row["horizon"]), str(row["scene_family"])): row
        for row in rows
        if row["scene_type"] == "open"
    }
    x = np.arange(len(family_order), dtype=float)
    fig, ax = plt.subplots(figsize=(7.16, 2.82))
    offsets = np.linspace(-0.14, 0.14, len(series))
    for offset, (method, horizon, marker, linestyle) in zip(offsets, series):
        values = []
        lower = []
        upper = []
        for family in family_order:
            row = lookup[(method, horizon, family)]
            value = float(row["completion_rate"])
            values.append(value)
            lower.append(max(0.0, value - float(row["completion_ci95_low"])))
            upper.append(max(0.0, float(row["completion_ci95_high"]) - value))
        label = f"{_method_label(method)} H={horizon}"
        ax.errorbar(
            x + offset,
            values,
            yerr=np.asarray([lower, upper]),
            color=_method_color(method),
            marker=marker,
            linestyle=linestyle,
            linewidth=1.35,
            markersize=4.2,
            capsize=2.0,
            label=label,
            zorder=3,
        )
    ax.set_ylim(-0.04, 1.10)
    ax.set_ylabel("Completion rate")
    ax.set_xticks(x, family_labels, fontsize=7.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=6.4, ncol=3, loc="lower left")
    family_runs = int(lookup[("CST", 2, family_order[0])]["runs"])
    ax.set_title(
        f"Completion by causal dependency class ({family_runs} open instances per family)",
        fontsize=8.7,
        fontweight="bold",
    )
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_dependency_breakdown", dpi=320)
    plt.close(fig)


def plot_support_summary(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    """Compact accuracy/optimism diagnostic for analytic IET."""

    scene_order = ("open", "surprise")
    lookup = {str(row["scene_type"]): row for row in rows}
    metrics = (
        ("root_action_agreement", "Root action\nagreement"),
        ("future_option_precision", "Future option\nprecision"),
        ("future_edge_precision", "Future edge\nprecision"),
    )
    x = np.arange(len(metrics))
    width = 0.34
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.55),
        gridspec_kw={"width_ratios": [1.25, 0.8]},
    )
    ax = axes[0]
    colors = ("#168F83", "#D47A23")
    for idx, scene in enumerate(scene_order):
        row = lookup.get(scene)
        if row is None:
            continue
        values = [float(row[key]) if row[key] is not None else np.nan for key, _ in metrics]
        bars = ax.bar(
            x + (idx - 0.5) * width,
            values,
            width,
            color=colors[idx],
            alpha=0.9,
            label=scene.capitalize(),
        )
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.025,
                f"{value:.0%}",
                ha="center",
                fontsize=6.5,
            )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Rate")
    ax.set_xticks(x, [label for _, label in metrics], fontsize=7.0)
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7.0)
    ax.set_title("Imagination fidelity", fontsize=8.4)

    ax = axes[1]
    false_options = [
        float(lookup.get(scene, {}).get("mean_option_false_positives", 0.0))
        for scene in scene_order
    ]
    false_edges = [
        float(lookup.get(scene, {}).get("mean_edge_false_positives", 0.0))
        for scene in scene_order
    ]
    xx = np.arange(len(scene_order))
    ax.bar(xx - 0.16, false_options, 0.32, color="#E4B55C", label="Options")
    ax.bar(xx + 0.16, false_edges, 0.32, color="#C75B4B", label="Edges")
    ax.set_xticks(xx, ("Open", "Surprise"))
    ax.set_ylabel("False positives / branch")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False, fontsize=7.0)
    ax.set_title("Optimism cost", fontsize=8.4)

    fig.tight_layout(w_pad=1.4)
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_support_quality", dpi=320)
    plt.close(fig)


def plot_episode_distributions(
    records: Sequence[EpisodeResultRecord],
    output: Path,
) -> None:
    """Show return and planning-time distributions for the open benchmark."""

    selected = [
        row
        for row in records
        if row.scene_type == "open"
        and (
            (row.method.rsplit(" H=", 1)[0] == "CST" and row.horizon == 3)
            or (row.method.rsplit(" H=", 1)[0] == "CoImagine-IET" and row.horizon in {2, 3})
            or (row.method.rsplit(" H=", 1)[0] == "Privileged IET Oracle" and row.horizon == 3)
        )
    ]
    order = (
        ("CST", 3, "CST H=3"),
        ("CoImagine-IET", 2, "IET H=2"),
        ("CoImagine-IET", 3, "IET H=3"),
        ("Privileged IET Oracle", 3, "Oracle H=3"),
    )
    return_data = []
    timing_data = []
    colors = []
    labels = []
    for method, horizon, label in order:
        group = [
            row
            for row in selected
            if row.method.rsplit(" H=", 1)[0] == method and row.horizon == horizon
        ]
        return_data.append([row.realized_return for row in group])
        timing_data.append([row.mean_planning_seconds * 1000.0 for row in group])
        colors.append(_method_color(method))
        labels.append(label)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.65))
    for ax, data, ylabel, title in (
        (axes[0], return_data, "Realized return", "Closed-loop utility distribution"),
        (axes[1], timing_data, "Planning time / option [ms]", "Planning-time distribution"),
    ):
        parts = ax.violinplot(data, showmeans=False, showmedians=True, widths=0.75)
        for body, color in zip(parts["bodies"], colors):
            body.set_facecolor(color)
            body.set_edgecolor("#333333")
            body.set_alpha(0.72)
        parts["cmedians"].set_color("#222222")
        parts["cmins"].set_color("#666666")
        parts["cmaxes"].set_color("#666666")
        parts["cbars"].set_color("#666666")
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=18, ha="right", fontsize=6.8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.22)
        ax.set_title(title, fontsize=8.2)
    fig.tight_layout(w_pad=1.3)
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_episode_distributions", dpi=320)
    plt.close(fig)


def plot_paired_summary(rows: Sequence[Mapping[str, object]], output: Path) -> None:
    """Forest plot for paired completion gains on identical instances."""

    open_rows = [row for row in rows if row["scene_type"] == "open"]
    desired = (
        "IET H=3 vs CST H=3",
        "IET H=3 vs IET H=2",
        "Oracle H=3 vs IET H=3",
    )
    lookup = {str(row["comparison"]): row for row in open_rows}
    y = np.arange(len(desired))[::-1]
    values = [float(lookup[label]["completion_delta"]) for label in desired]
    lower = [
        max(0.0, value - float(lookup[label]["completion_delta_ci95_low"]))
        for label, value in zip(desired, values)
    ]
    upper = [
        max(0.0, float(lookup[label]["completion_delta_ci95_high"]) - value)
        for label, value in zip(desired, values)
    ]
    fig, ax = plt.subplots(figsize=(5.3, 2.25))
    ax.axvline(0.0, color="#777777", linewidth=0.8, linestyle="--")
    colors = ("#168F83", "#168F83", "#7656A5")
    for index, (yy, value, lo, hi, color) in enumerate(zip(y, values, lower, upper, colors)):
        ax.errorbar(value, yy, xerr=np.asarray([[lo], [hi]]), fmt="o", color=color, capsize=3.0, markersize=5.0)
        p_value = float(lookup[desired[index]]["mcnemar_exact_p"])
        ax.text(value + hi + 0.015, yy, f"p={p_value:.3g}", va="center", fontsize=6.8)
    ax.set_yticks(y, ("IET H=3 − CST H=3", "IET H=3 − IET H=2", "Oracle H=3 − IET H=3"), fontsize=7.0)
    ax.set_xlim(-0.02, max(0.45, max(value + hi for value, hi in zip(values, upper)) + 0.08))
    ax.set_xlabel("Paired completion-rate difference")
    ax.grid(axis="x", alpha=0.22)
    ax.set_title("Paired open-task comparisons (bootstrap 95% CI)", fontsize=8.4, fontweight="bold")
    fig.tight_layout()
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_paired_comparisons", dpi=320)
    plt.close(fig)


def plot_structural_scaling(rows: Sequence[ScalingResult], output: Path) -> None:
    """Publication-ready object and robot scaling curves on logarithmic axes."""

    entity = sorted(
        (row for row in rows if row.study == "entity_scaling"),
        key=lambda row: row.x_value,
    )
    robot_all = [row for row in rows if row.study == "robot_factor_scaling"]
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.72))

    ax = axes[0]
    ax.plot(
        [row.x_value for row in entity],
        [row.naive_count for row in entity],
        marker="o",
        color="#A54545",
        linewidth=1.7,
        label="Full tuples",
    )
    ax.plot(
        [row.x_value for row in entity],
        [row.factor_count for row in entity],
        marker="s",
        color="#168F83",
        linewidth=1.7,
        label="Entity-factor checks",
    )
    ax.set_yscale("log")
    ax.set_xlabel("Number of task entities M")
    ax.set_ylabel("Candidate checks")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, fontsize=7.0)
    ax.set_title("Object scaling (N=2)", fontsize=8.4)

    ax = axes[1]
    arity_colors = {2: "#3C78A8", 3: "#7656A5", 4: "#D47A23"}
    robot_counts = sorted({row.x_value for row in robot_all})
    # Full Cartesian is independent of arity; draw it once.
    p3 = sorted((row for row in robot_all if row.factor_arity == 3), key=lambda row: row.x_value)
    ax.plot(
        [row.x_value for row in p3],
        [row.naive_count for row in p3],
        marker="o",
        color="#A54545",
        linewidth=1.7,
        label="Full Cartesian",
    )
    for arity in (2, 3, 4):
        group = sorted(
            (row for row in robot_all if row.factor_arity == arity),
            key=lambda row: row.x_value,
        )
        ax.plot(
            [row.x_value for row in group],
            [row.recursive_calls for row in group],
            marker={2: "s", 3: "D", 4: "^"}[arity],
            color=arity_colors[arity],
            linewidth=1.45,
            label=f"Recursive verifier P={arity}",
        )
    ax.set_yscale("log")
    ax.set_xlabel("Number of robots N")
    ax.set_ylabel("Enumeration / verifier calls")
    ax.grid(alpha=0.22, which="both")
    ax.legend(frameon=False, fontsize=6.4)
    ax.set_title("Robot scaling by bounded factor arity", fontsize=8.4)

    fig.suptitle(
        "Bounded-factor structure avoids full joint-option growth",
        fontsize=8.9,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.45)
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_structural_scaling", dpi=320)
    plt.close(fig)


def plot_randomized_channel_debug_gallery(
    instances: Sequence[ScenarioInstance],
    output: Path,
    *,
    columns: int = 4,
) -> None:
    """Render initial-state diagnostics for heterogeneous randomized channels."""

    if not instances:
        return
    columns = max(1, int(columns))
    rows = int(np.ceil(len(instances) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(7.16, 2.12 * rows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, instance in zip(axes.flat, instances):
        state = instance.state
        radius = int(state.task.metadata.get("observation_radius", 6))
        _draw_visibility(ax, state, radius)
        draw_state(
            ax,
            state,
            show_grid=True,
            show_robot_ids_inside=False,
            entity_labels_above=True,
        )
        _draw_visibility_rays(ax, state, radius)
        _draw_channel_geometry(ax, state)
        budget = int(state.task.metadata.get("candidate_budget", 8))
        debug_env = GridEnvironment(observation_radius=radius)
        debug_generator = GridOptionGenerator(candidate_budget=budget)
        debug_planner = ExactTopologySearch(
            debug_env,
            debug_generator,
            horizon=1,
        )
        debug_beliefs = debug_planner.initialize_beliefs(state)
        available_by_robot = {
            robot_id: {
                option.target
                for option in debug_generator.generate(belief)
                if option.kind == OptionKind.REPOSITION and option.target is not None
            }
            for robot_id, belief in debug_beliefs.items()
        }
        _draw_reposition_anchors(
            ax,
            state,
            None,
            available_by_robot=available_by_robot,
        )
        for robot_id, robot in sorted(state.robots.items()):
            ax.text(
                robot.position[0] + 0.5,
                robot.position[1] - 0.08,
                f"R{robot_id}",
                ha="center",
                va="top",
                fontsize=4.5,
                fontweight="bold",
                color="black",
                zorder=15,
            )
        md = state.task.metadata
        ax.set_title(
            f"seed {instance.seed} · {md.get('difficulty_variant')}\n"
            f"M={len(state.blockers)}, heavy={md.get('heavy_blocker_count')}, "
            f"R={radius}, K={md.get('candidate_budget')}, T={md.get('episode_max_steps')}\n"
            f"pose={md.get('initial_position_mode')} · "
            f"N/S={md.get('north_push_entities')}/{md.get('south_push_entities')}",
            fontsize=4.7,
            pad=2.0,
        )
        ax.axis("on")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Debug gallery: unscreened randomized multi-blocker channel tasks",
        fontsize=8.7,
        fontweight="bold",
        y=0.995,
    )
    fig.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="white",
                markeredgecolor="#555555", markersize=4.3, label="Anchor bank",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="none",
                markeredgecolor="#3F7CAC", markersize=5.2, label="R0 currently admissible",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none", markerfacecolor="none",
                markeredgecolor="#D28B35", markersize=5.2, label="R1 currently admissible",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        frameon=False,
        fontsize=5.1,
        ncol=3,
        columnspacing=1.3,
        handletextpad=0.4,
    )
    fig.subplots_adjust(
        left=0.035, right=0.995, bottom=0.025, top=0.90,
        wspace=0.12, hspace=0.42,
    )
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_randomized_channel_debug", dpi=260)
    plt.close(fig)


def plot_randomized_channel_summary(
    rows: Sequence[Mapping[str, object]],
    output: Path,
) -> None:
    """Publication-style completion, progress, and timing by scene difficulty."""

    difficulties = ("easy", "moderate", "hard")
    labels = ("Easy", "Moderate", "Hard")
    preferred_series = (
        ("CST-Frozen-K", 3),
        ("CST", 3),
        ("CoImagine-IET", 3),
        ("Privileged IET Oracle", 3),
    )
    lookup = {
        (str(row["difficulty"]), str(row["method"]), int(row["horizon"])): row
        for row in rows
        if str(row["difficulty"]) != "all"
    }
    series = tuple(
        item
        for item in preferred_series
        if all((difficulty, item[0], item[1]) in lookup for difficulty in difficulties)
    )
    if not series:
        return
    x = np.arange(len(difficulties), dtype=float)
    width = 0.68 / len(series)
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.55))

    def safe_yerr(
        centres: Sequence[float],
        lows: Sequence[float],
        highs: Sequence[float],
    ) -> np.ndarray:
        """Return finite, nonnegative asymmetric errors for Matplotlib.

        Percentile and Wilson summaries can differ from their centres by tiny
        negative round-off (for example, -2e-19). Matplotlib 3.10 rejects any
        negative value, so clamp after converting NaN/Inf to zero.
        """

        centre_array = np.asarray(centres, dtype=float)
        low_array = np.asarray(lows, dtype=float)
        high_array = np.asarray(highs, dtype=float)
        lower = np.maximum(0.0, centre_array - low_array)
        upper = np.maximum(0.0, high_array - centre_array)
        return np.nan_to_num(
            np.vstack([lower, upper]),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

    ax = axes[0]
    offsets = (np.arange(len(series)) - (len(series) - 1) / 2.0) * width
    for offset, (method, horizon) in zip(offsets, series):
        values = []
        ci_lows = []
        ci_highs = []
        for difficulty in difficulties:
            row = lookup[(difficulty, method, horizon)]
            value = float(row["completion_rate"])
            values.append(value)
            ci_lows.append(float(row["completion_ci_low"]))
            ci_highs.append(float(row["completion_ci_high"]))
        ax.bar(
            x + offset,
            values,
            width,
            color=_method_color(method),
            alpha=0.88,
            label=f"{_method_label(method)} H={horizon}",
            zorder=2,
        )
        ax.errorbar(
            x + offset,
            values,
            yerr=safe_yerr(values, ci_lows, ci_highs),
            fmt="none",
            color="#333333",
            linewidth=0.7,
            capsize=1.8,
            zorder=3,
        )
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Completion rate")
    ax.set_xticks(x, labels, fontsize=6.8)
    ax.grid(axis="y", alpha=0.20, zorder=0)
    ax.set_title("Closed-loop completion", fontsize=7.8)
    ax.legend(frameon=False, fontsize=5.7, loc="lower left")

    ax = axes[1]
    marker_by_method = {
        "CST-Frozen-K": "x",
        "CST": "o",
        "CoImagine-IET": "s",
        "Privileged IET Oracle": "D",
    }
    progress_series = tuple((method, horizon, marker_by_method[method]) for method, horizon in series)
    for method, horizon, marker in progress_series:
        medians = []
        q25 = []
        q75 = []
        for difficulty in difficulties:
            row = lookup[(difficulty, method, horizon)]
            median_value = float(row["final_potential_median"])
            medians.append(median_value)
            q25.append(float(row["final_potential_q25"]))
            q75.append(float(row["final_potential_q75"]))
        ax.errorbar(
            x,
            medians,
            yerr=safe_yerr(medians, q25, q75),
            marker=marker,
            linewidth=1.35,
            markersize=4.0,
            capsize=2.0,
            color=_method_color(method),
            label=_method_label(method),
        )
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(x, labels, fontsize=6.8)
    ax.set_ylabel("Final task potential")
    ax.grid(alpha=0.20)
    ax.set_title("Median and IQR progress", fontsize=7.8)

    ax = axes[2]
    for method, horizon, marker in progress_series:
        medians = []
        q25 = []
        q75 = []
        for difficulty in difficulties:
            row = lookup[(difficulty, method, horizon)]
            median_value = float(row["planning_ms_median"])
            medians.append(median_value)
            q25.append(float(row["planning_ms_q25"]))
            q75.append(float(row["planning_ms_q75"]))
        ax.errorbar(
            x,
            medians,
            yerr=safe_yerr(medians, q25, q75),
            marker=marker,
            linewidth=1.35,
            markersize=4.0,
            capsize=2.0,
            color=_method_color(method),
            label=_method_label(method),
        )
    ax.set_xticks(x, labels, fontsize=6.8)
    ax.set_ylabel("Planning time / option [ms]")
    ax.grid(alpha=0.20)
    ax.set_title("Median and IQR cost", fontsize=7.8)

    fig.suptitle(
        "Heterogeneous randomized channel benchmark",
        fontsize=8.8,
        fontweight="bold",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.0)
    output.mkdir(parents=True, exist_ok=True)
    save_all_formats(fig, output / "fig_gate_a_randomized_channel_summary", dpi=320)
    plt.close(fig)
