from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from textwrap import fill
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from coimagine.core.types import Direction, JointOption, LocalOption, OptionKind
from coimagine.envs.grid.state import BlockerState, GridState
from coimagine.evaluation.episodes import EpisodeTrace
from coimagine.evaluation.records import EpisodeResultRecord, ScalingResult
from coimagine.options.library import UNIVERSAL_OPTION_LIBRARY
from coimagine.reporting.figures import save_all_formats

ROBOT_COLORS = {0: "#3366A5", 1: "#2A9D8F", 2: "#7A5195", 3: "#E76F51"}
BLOCKER_COLORS = {"heavy": "#A84343", "medium": "#C46A3A", "light": "#D99A55"}


def _entity_code(entity_id: str) -> str:
    return {
        "heavy-gate": "H",
        "light-front": "F",
        "light-rear": "R",
    }.get(entity_id, entity_id.replace("light-extra-", "E").replace("blocker-", "B"))


def _entity_description(blocker: BlockerState) -> str:
    code = _entity_code(blocker.entity_id)
    return f"{code}: {blocker.mass_class}, {blocker.required_pushers} robot{'s' if blocker.required_pushers != 1 else ''}"


def compact_action(joint: JointOption | None) -> str:
    if joint is None:
        return "—"
    active = [option for option in joint.options if option.kind != OptionKind.IDLE]
    options = active or list(joint.options)
    grouped: dict[tuple, list[int]] = {}
    exemplars: dict[tuple, LocalOption] = {}
    for option in options:
        key = (option.kind, option.entity_id, option.direction, option.role_token)
        grouped.setdefault(key, []).append(option.robot_id)
        exemplars[key] = option
    pieces: list[str] = []
    for key in sorted(grouped, key=lambda item: (item[0].value, str(item[1]), str(item[2]))):
        option = exemplars[key]
        robots = ",".join(f"R{rid}" for rid in sorted(grouped[key]))
        code = _entity_code(option.entity_id) if option.entity_id else ""
        if option.kind == OptionKind.REPOSITION:
            pieces.append(f"{robots} Reposition")
        elif option.kind == OptionKind.APPROACH:
            pieces.append(f"{robots} Approach({code or 'target'})")
        elif option.kind == OptionKind.PUSH:
            arrow = {
                Direction.NORTH: "↑",
                Direction.EAST: "→",
                Direction.SOUTH: "↓",
                Direction.WEST: "←",
                None: "",
            }[option.direction]
            pieces.append(f"{robots} Push({code}){arrow}")
        elif option.kind == OptionKind.HOLD:
            pieces.append(f"{robots} Hold")
        elif option.kind == OptionKind.OBSERVE:
            pieces.append(f"{robots} Observe({code})")
        else:
            pieces.append(f"{robots} Idle")
    return " + ".join(pieces)


def _is_dependency_locked(state: GridState, entity_id: str) -> bool:
    dependency = state.task.activation_dependencies.get(entity_id)
    if dependency is None:
        return False
    blocker = state.blockers.get(dependency)
    return blocker is not None and any(cell in state.task.corridor for cell in blocker.cells)


def _draw_action_overlay(ax, state: GridState, action: JointOption | None) -> None:
    if action is None:
        return
    for option in action.options:
        robot = state.robots[option.robot_id]
        sx, sy = robot.position[0] + 0.5, robot.position[1] + 0.5
        color = ROBOT_COLORS.get(option.robot_id, "#3366A5")
        if option.kind in {OptionKind.REPOSITION, OptionKind.APPROACH} and option.target is not None:
            tx, ty = option.target[0] + 0.5, option.target[1] + 0.5
            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(sx, sy),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, ls="--", alpha=0.9),
                zorder=8,
            )
        elif option.kind == OptionKind.PUSH and option.entity_id in state.blockers:
            blocker = state.blockers[option.entity_id]
            cx = sum(x + 0.5 for x, _ in blocker.cells) / len(blocker.cells)
            cy = sum(y + 0.5 for _, y in blocker.cells) / len(blocker.cells)
            dx, dy = option.direction.delta if option.direction else (0, 0)
            ax.annotate(
                "",
                xy=(cx + 0.85 * dx, cy + 0.85 * dy),
                xytext=(cx, cy),
                arrowprops=dict(arrowstyle="-|>", color="#7B1E1E", lw=2.0),
                zorder=9,
            )


def _draw_story_state(
    ax,
    state: GridState,
    *,
    action: JointOption | None,
    title: str,
    callout: str | None = None,
    callout_tone: str = "positive",
    outcome: str | None = None,
    show_entity_labels: bool = True,
) -> None:
    ax.set_aspect("equal")
    ax.set_xlim(0, state.width)
    ax.set_ylim(0, state.height)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=7.4, pad=4, fontweight="medium")

    for cell in state.task.corridor:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#E9ECEF", edgecolor="none", zorder=0))
    # Task direction is deliberately explicit so the figure can be read without text lookup.
    ax.annotate(
        "clear corridor",
        xy=(state.width - 1.5, 4.65),
        xytext=(1.5, 4.65),
        ha="left",
        va="center",
        fontsize=5.4,
        color="#555555",
        arrowprops=dict(arrowstyle="->", color="#777777", lw=0.8),
        zorder=1,
    )
    for cell in state.walls:
        ax.add_patch(Rectangle(cell, 1, 1, facecolor="#3F3F3F", edgecolor="#252525", lw=0.3, zorder=2))

    for entity_id, blocker in state.blockers.items():
        locked = _is_dependency_locked(state, entity_id)
        face = BLOCKER_COLORS.get(blocker.mass_class, "#D99A55")
        for cell in blocker.cells:
            ax.add_patch(
                Rectangle(
                    cell,
                    1,
                    1,
                    facecolor=face,
                    edgecolor="#652525",
                    lw=0.8,
                    hatch="////" if locked else None,
                    alpha=0.9 if not locked else 0.62,
                    zorder=4,
                )
            )
        cx = sum(x + 0.5 for x, _ in blocker.cells) / len(blocker.cells)
        cy = sum(y + 0.5 for _, y in blocker.cells) / len(blocker.cells)
        ax.text(cx, cy, _entity_code(entity_id), ha="center", va="center", fontsize=7, color="white", fontweight="bold", zorder=5)
        if show_entity_labels:
            label = f"{blocker.mass_class}/{blocker.required_pushers}R"
            ax.text(cx, cy + 0.68, label, ha="center", va="bottom", fontsize=4.35, color="#4A2020", zorder=6)
            if locked:
                ax.text(cx, cy - 0.68, "locked", ha="center", va="top", fontsize=4.1, color="#7A2B2B", fontweight="bold", zorder=6)

    for robot_id, robot in state.robots.items():
        x, y = robot.position
        color = ROBOT_COLORS.get(robot_id, "#3366A5")
        circle = plt.Circle((x + 0.5, y + 0.5), 0.34, facecolor=color, edgecolor="white", linewidth=0.8, zorder=7)
        ax.add_patch(circle)
        ax.text(x + 0.5, y + 0.5, f"R{robot_id}", color="white", ha="center", va="center", fontsize=5.4, fontweight="bold", zorder=8)

    _draw_action_overlay(ax, state, action)
    potential = _weighted_progress(state)
    ax.text(
        0.02,
        0.02,
        f"clearance Φ={potential:.0%}",
        transform=ax.transAxes,
        fontsize=5.7,
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="#BBBBBB", alpha=0.92),
        zorder=12,
    )
    if callout:
        palette = {
            "positive": ("#E7F5EA", "#2E7D32"),
            "negative": ("#FDECEC", "#A33A3A"),
            "neutral": ("#EEF2F5", "#4D5963"),
        }
        face, edge = palette[callout_tone]
        ax.text(
            0.98,
            0.98,
            callout,
            transform=ax.transAxes,
            fontsize=5.4,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=face, edgecolor=edge, linewidth=0.8),
            zorder=12,
        )
    if outcome:
        ax.text(
            0.5,
            0.5,
            outcome,
            transform=ax.transAxes,
            fontsize=7.0,
            ha="center",
            va="center",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#333333", alpha=0.94),
            zorder=15,
        )
    for x in range(state.width + 1):
        ax.plot([x, x], [0, state.height], color="#DADADA", linewidth=0.25, zorder=0)
    for y in range(state.height + 1):
        ax.plot([0, state.width], [y, y], color="#DADADA", linewidth=0.25, zorder=0)


def _weighted_progress(state: GridState) -> float:
    if state.task.initial_blocked_weight > 0:
        blocked = sum(
            blocker.task_weight * len(blocker.cells & state.task.corridor)
            for blocker in state.blockers.values()
        )
        return max(0.0, min(1.0, 1.0 - blocked / state.task.initial_blocked_weight))
    denominator = max(1, state.task.initial_blocked_cells)
    blocked = sum(len(blocker.cells & state.task.corridor) for blocker in state.blockers.values())
    return max(0.0, min(1.0, 1.0 - blocked / denominator))


def _review_footer_text(state: GridState) -> str:
    entities = "; ".join(_entity_description(blocker) for blocker in state.blockers.values())
    templates = "; ".join(f"{item.display_name}: {item.availability_rule}" for item in UNIVERSAL_OPTION_LIBRARY)
    paragraphs = [
        "TASK AND SYMBOLS — Grey cells are the transit corridor that must be cleared. "
        f"{entities}. Hatched R means its contact options are disabled until H leaves the corridor. "
        "Colored circles are robots; dashed arrows are Approach options and solid arrows are Push options.",
        "METHODS — Oracle Evolving Topology regenerates successor option nodes and executable joint-option edges on every imagined branch. "
        "Root-Structure Reuse predicts the same exact successor state and task consequence but projects imagined candidates onto the root option-template slots and root executable edges.",
        "STORY — At t=0 both methods can clear F immediately. Evolving also sends R0 and R1 to H because it predicts the new two-robot Push(H) edge at t=1. Root reuse cannot represent that future edge, leaves both robots idle, and then repeatedly rejects the zero-immediate-reward preparation. H later unlocks R, so only the evolving policy completes the corridor within the episode budget.",
        "UNIVERSAL OPTION LIBRARY — " + templates,
        "VALUES — Φ is weighted corridor clearance. The action ribbon lists the realized receding-horizon macro-options. 'Created edge' refers to a successor executable joint-option template absent at the root, not to an increase in total edge count.",
    ]
    return "\n\n".join(fill(text, width=170) for text in paragraphs)


def _sequence_ribbon(trace: EpisodeTrace, max_items: int = 6) -> str:
    actions = [compact_action(step.selected_action) for step in trace.steps]
    if not actions:
        return "no action"
    # Compress repeated identical suffixes, which is particularly informative for
    # the root-reuse stall rather than visually repeating seven copies of Idle.
    if len(actions) >= 3 and len(set(actions[1:])) == 1:
        actions = [actions[0], f"{actions[1]} ×{len(actions)-1}"]
    elif len(actions) > max_items:
        actions = actions[: max_items - 1] + ["…"]
    return "  →  ".join(actions)


def plot_gate_a_main_story(
    evolving: EpisodeTrace,
    root_reuse: EpisodeTrace,
    output: Path,
) -> None:
    """Create the compact double-column Gate-A paper story and review version."""

    initial = evolving.steps[0].state_before if evolving.steps else evolving.final_state
    footer = _review_footer_text(initial)

    def build(review: bool):
        main_height = 4.25
        footer_height = 3.0 if review else 0.0
        fig = plt.figure(figsize=(7.16, main_height + footer_height))
        if review:
            outer = fig.add_gridspec(2, 1, height_ratios=[main_height, footer_height], hspace=0.08)
            grid = outer[0].subgridspec(2, 4, wspace=0.13, hspace=0.48)
            footer_ax = fig.add_subplot(outer[1])
            footer_ax.axis("off")
            footer_ax.axhline(0.98, color="#AFAFAF", lw=0.7)
            footer_ax.text(0.01, 0.94, footer, ha="left", va="top", fontsize=6.0)
        else:
            grid = fig.add_gridspec(2, 4, wspace=0.13, hspace=0.48)
        axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)])

        # Evolving row: the four decisive boundaries.
        e_steps = list(evolving.steps)
        e0 = e_steps[0]
        e1 = e_steps[1] if len(e_steps) > 1 else e0
        e2 = e_steps[2] if len(e_steps) > 2 else e1
        _draw_story_state(
            axes[0, 0], e0.state_before, action=e0.selected_action,
            title="t=0  Parallel preparation",
            callout="predicts future\nR0+R1 Push(H)", callout_tone="positive",
        )
        _draw_story_state(
            axes[0, 1], e1.state_before, action=e1.selected_action,
            title="t=1  New coalition edge",
            callout="CREATED EDGE\nR0+R1 Push(H)", callout_tone="positive",
            show_entity_labels=False,
        )
        _draw_story_state(
            axes[0, 2], e2.state_before, action=e2.selected_action,
            title="t=2  Rear contact activated",
            callout="H cleared ⇒ R unlocked\nthen R2→R, Push(R)", callout_tone="positive",
            show_entity_labels=False,
        )
        _draw_story_state(
            axes[0, 3], evolving.final_state, action=None,
            title="Outcome",
            outcome=f"COMPLETE\n{evolving.option_count} options · Φ={evolving.final_potential:.0%}",
            show_entity_labels=False,
        )

        # Root-reuse row: show the initial greedy action, the missing future edge,
        # the repeated fixed-structure decision, and final stalled state.
        r_steps = list(root_reuse.steps)
        r0 = r_steps[0]
        r1 = r_steps[1] if len(r_steps) > 1 else r0
        r2 = r_steps[min(2, len(r_steps)-1)] if r_steps else r0
        _draw_story_state(
            axes[1, 0], r0.state_before, action=r0.selected_action,
            title="t=0  Immediate progress only",
            callout="clears F now\nR0,R1 wait", callout_tone="neutral",
        )
        _draw_story_state(
            axes[1, 1], r1.state_before, action=r1.selected_action,
            title="t=1  Fixed future structure",
            callout="Push(H) absent from\nroot rollout ⇒ no value", callout_tone="negative",
            show_entity_labels=False,
        )
        _draw_story_state(
            axes[1, 2], r2.state_before, action=r2.selected_action,
            title="t=2…  Repeats same decision",
            callout=f"wait repeated ×{max(1, root_reuse.option_count-1)}", callout_tone="negative",
            show_entity_labels=False,
        )
        _draw_story_state(
            axes[1, 3], root_reuse.final_state, action=None,
            title="Outcome",
            outcome=f"STALLED\n{root_reuse.option_count} options · Φ={root_reuse.final_potential:.0%}",
            show_entity_labels=False,
        )

        if review:
            y_top_row, y_bottom_row = 0.80, 0.54
            y_task, y_e_ribbon, y_r_ribbon = 0.965, 0.68, 0.425
        else:
            y_top_row, y_bottom_row = 0.70, 0.25
            y_task, y_e_ribbon, y_r_ribbon = 0.935, 0.49, 0.035
        fig.text(0.006, y_top_row, "Evolving topology", rotation=90, va="center", ha="left", fontsize=8.0, fontweight="bold", color="#2E7D32")
        fig.text(0.006, y_bottom_row, "Root-structure reuse", rotation=90, va="center", ha="left", fontsize=8.0, fontweight="bold", color="#A33A3A")
        fig.suptitle(
            "Gate A: future coordination topology changes the first action and task outcome",
            y=0.995,
            fontsize=9.2,
            fontweight="bold",
        )
        fig.text(0.5, y_task, "Task: clear three mixed-mass blockers; H needs two robots and blocks access to R.", ha="center", va="top", fontsize=7.0)
        fig.text(0.5, y_e_ribbon, "Evolving sequence: " + _sequence_ribbon(evolving), ha="center", va="center", fontsize=5.7, color="#245C2A")
        fig.text(0.5, y_r_ribbon, "Root-reuse sequence: " + _sequence_ribbon(root_reuse), ha="center", va="bottom", fontsize=5.7, color="#7A2B2B")
        if review:
            fig.subplots_adjust(left=0.045, right=0.995, top=0.91, bottom=0.015)
        else:
            fig.subplots_adjust(left=0.045, right=0.995, top=0.90, bottom=0.075)
        return fig

    clean = build(False)
    save_all_formats(clean, output / "paper_clean" / "fig_gate_a_main_story", dpi=300)
    plt.close(clean)
    review = build(True)
    save_all_formats(review, output / "paper_review" / "fig_gate_a_main_story", dpi=260)
    plt.close(review)

    # Save every main panel independently for presentation assembly.
    panel_dir = output / "panels" / "main_story"
    panel_specs = []
    e_steps = list(evolving.steps)
    panel_specs.extend(
        [
            ("evolving_t0", e_steps[0].state_before, e_steps[0].selected_action, "Evolving t=0"),
            ("evolving_t1", e_steps[1].state_before, e_steps[1].selected_action, "Evolving t=1"),
            ("evolving_t2", e_steps[2].state_before, e_steps[2].selected_action, "Evolving t=2"),
            ("evolving_final", evolving.final_state, None, "Evolving outcome"),
        ]
    )
    r_steps = list(root_reuse.steps)
    panel_specs.extend(
        [
            ("root_reuse_t0", r_steps[0].state_before, r_steps[0].selected_action, "Root reuse t=0"),
            ("root_reuse_t1", r_steps[1].state_before, r_steps[1].selected_action, "Root reuse t=1"),
            ("root_reuse_repeat", r_steps[min(2, len(r_steps)-1)].state_before, r_steps[min(2, len(r_steps)-1)].selected_action, "Root reuse repeats"),
            ("root_reuse_final", root_reuse.final_state, None, "Root reuse outcome"),
        ]
    )
    for name, state, action, title in panel_specs:
        fig, ax = plt.subplots(figsize=(3.4, 2.25))
        _draw_story_state(ax, state, action=action, title=title)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.90, bottom=0.03)
        save_all_formats(fig, panel_dir / name, dpi=260)
        plt.close(fig)


def plot_gate_a_compact_results(
    episode_records: Sequence[EpisodeResultRecord],
    scaling_records: Sequence[ScalingResult],
    output: Path,
) -> None:
    """One double-column quantitative figure for the compact Gate-A subsection."""

    benchmark = [row for row in episode_records if row.suite == "natural" and row.task_name == "complex_channel_clearing"]
    methods = [
        "Evolving H=2 (exact)",
        "Root-Structure Reuse H=2",
        "Evolving H=1 (myopic)",
    ]
    method_styles = {
        methods[0]: ("o", "-"),
        methods[1]: ("s", "--"),
        methods[2]: ("^", ":"),
    }
    entity_rows = [row for row in scaling_records if row.study == "entity_scaling"]
    robot_rows = [row for row in scaling_records if row.study == "robot_factor_scaling" and row.factor_arity == 3]

    review_text = (
        "PANEL (a) — Mean final weighted corridor clearance over unscreened channel-clearing variants. "
        "M is the number of blockers; failure remains at its achieved Φ rather than being discarded. "
        "Evolving H=1 is a matched myopic control.\n\n"
        "PANEL (b) — Naive cross-entity checking grows as O(M²K²), while entity-local checking grows as O(MK²); both use the same K options per entity.\n\n"
        "PANEL (c) — Full tuples are K^N. Recursive exact factors enumerate only extendable partial tuples up to arity P=3. The plotted pilot points have exact oracle recall 1.0."
    )

    def build(review: bool):
        height = 2.45 + (2.25 if review else 0.0)
        fig = plt.figure(figsize=(7.16, height))
        if review:
            outer = fig.add_gridspec(2, 1, height_ratios=[2.55, 1.95], hspace=0.34)
            grid = outer[0].subgridspec(1, 3, wspace=0.42)
            footer = fig.add_subplot(outer[1])
            footer.axis("off")
            footer.axhline(0.98, color="#AAAAAA", lw=0.7)
            footer_copy = "\n\n".join(
                fill(paragraph, width=160)
                for paragraph in review_text.split("\n\n")
            )
            footer.text(0.01, 0.91, footer_copy, va="top", ha="left", fontsize=6.0)
        else:
            grid = fig.add_gridspec(1, 3, wspace=0.42)
        axes = [fig.add_subplot(grid[0, idx]) for idx in range(3)]

        # (a) Decision outcome across complexity.
        ax = axes[0]
        counts = sorted({row.entity_count for row in benchmark})
        series: dict[str, tuple[list[float], list[float]]] = {}
        for method in methods:
            means: list[float] = []
            errors: list[float] = []
            for count in counts:
                vals = [row.final_potential for row in benchmark if row.method == method and row.entity_count == count]
                means.append(float(np.mean(vals)) if vals else np.nan)
                errors.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0)
            series[method] = (means, errors)

        fixed_same_as_myopic = np.allclose(
            series[methods[1]][0], series[methods[2]][0], equal_nan=True
        )
        display_methods = (methods[0], methods[1]) if fixed_same_as_myopic else tuple(methods)
        for method in display_methods:
            ys, errs = series[method]
            marker, linestyle = method_styles[method]
            label = method.replace(" (exact)", "")
            if method == methods[1] and fixed_same_as_myopic:
                label = "Root reuse H=2 / Evolving H=1"
            ax.errorbar(
                counts, ys, yerr=errs, marker=marker, linestyle=linestyle,
                linewidth=1.25, markersize=4, capsize=2, label=label,
            )
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(counts)
        ax.set_xlabel("Blockers M")
        ax.set_ylabel("Final clearance Φ")
        ax.set_title("(a) Decision outcome", fontsize=8.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.0, loc="center right", frameon=False)
        evolving_rows = [row for row in benchmark if row.method == methods[0]]
        fixed_rows = [row for row in benchmark if row.method == methods[1]]
        ax.text(
            0.97, 0.78,
            f"complete: {sum(row.completed for row in evolving_rows)}/{len(evolving_rows)} vs "
            f"{sum(row.completed for row in fixed_rows)}/{len(fixed_rows)}",
            transform=ax.transAxes, ha="right", va="center", fontsize=5.3,
        )

        # (b) Entity locality.
        ax = axes[1]
        entity = sorted(entity_rows, key=lambda row: row.x_value)
        ax.plot([row.x_value for row in entity], [row.naive_count for row in entity], marker="s", linestyle="--", label="Naive cross-entity")
        ax.plot([row.x_value for row in entity], [row.factor_count for row in entity], marker="o", linestyle="-", label="Entity-local")
        ax.set_yscale("log")
        ax.set_xlabel("Entities M")
        ax.set_ylabel("Compatibility checks")
        ax.set_title("(b) Entity locality", fontsize=8.0)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.5, frameon=False)

        # (c) Bounded coupling.
        ax = axes[2]
        robot = sorted(robot_rows, key=lambda row: row.robot_count)
        ax.plot([row.robot_count for row in robot], [row.naive_count for row in robot], marker="s", linestyle="--", label=r"Full $K^N$")
        ax.plot([row.robot_count for row in robot], [row.recursive_calls for row in robot], marker="o", linestyle="-", label="Recursive P=3")
        ax.set_yscale("log")
        ax.set_xlabel("Robots N")
        ax.set_ylabel("Tuple evaluations")
        ax.set_title("(c) Bounded coupling", fontsize=8.0)
        ax.grid(alpha=0.25)
        recall = min((row.oracle_recall for row in robot), default=1.0)
        ax.text(0.97, 0.05, f"oracle recall={recall:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.7)
        ax.legend(fontsize=5.5, frameon=False)

        if review:
            fig.subplots_adjust(left=0.075, right=0.99, top=0.93, bottom=0.04)
        else:
            fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.18)
        return fig

    clean = build(False)
    save_all_formats(clean, output / "paper_clean" / "fig_gate_a_compact_results", dpi=300)
    plt.close(clean)
    review = build(True)
    save_all_formats(review, output / "paper_review" / "fig_gate_a_compact_results", dpi=260)
    plt.close(review)

    # Save each quantitative panel independently for slide/paper reassembly.
    panel_dir = output / "panels" / "compact_results"
    counts = sorted({row.entity_count for row in benchmark})
    decision_series: dict[str, list[float]] = {}
    for method in methods:
        decision_series[method] = [
            float(np.mean([
                row.final_potential
                for row in benchmark
                if row.method == method and row.entity_count == count
            ]))
            for count in counts
        ]
    fixed_same = np.allclose(
        decision_series[methods[1]], decision_series[methods[2]], equal_nan=True
    )

    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    for method in ((methods[0], methods[1]) if fixed_same else tuple(methods)):
        marker, linestyle = method_styles[method]
        label = method.replace(" (exact)", "")
        if method == methods[1] and fixed_same:
            label = "Root reuse H=2 / Evolving H=1"
        ax.plot(counts, decision_series[method], marker=marker, linestyle=linestyle, label=label)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(counts)
    ax.set_xlabel("Blockers M")
    ax.set_ylabel("Final clearance Φ")
    ax.set_title("Decision outcome")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, frameon=False, loc="center right")
    fig.tight_layout()
    save_all_formats(fig, panel_dir / "decision_outcome", dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    entity = sorted(entity_rows, key=lambda row: row.x_value)
    ax.plot([row.x_value for row in entity], [row.naive_count for row in entity], marker="s", linestyle="--", label="Naive cross-entity")
    ax.plot([row.x_value for row in entity], [row.factor_count for row in entity], marker="o", linestyle="-", label="Entity-local")
    ax.set_yscale("log")
    ax.set_xlabel("Entities M")
    ax.set_ylabel("Compatibility checks")
    ax.set_title("Entity locality")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, frameon=False)
    fig.tight_layout()
    save_all_formats(fig, panel_dir / "entity_locality", dpi=260)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.4, 2.55))
    robot = sorted(robot_rows, key=lambda row: row.robot_count)
    ax.plot([row.robot_count for row in robot], [row.naive_count for row in robot], marker="s", linestyle="--", label=r"Full $K^N$")
    ax.plot([row.robot_count for row in robot], [row.recursive_calls for row in robot], marker="o", linestyle="-", label="Recursive P=3")
    ax.set_yscale("log")
    ax.set_xlabel("Robots N")
    ax.set_ylabel("Tuple evaluations")
    ax.set_title("Bounded coupling")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.2, frameon=False)
    ax.text(0.97, 0.05, f"oracle recall={min((row.oracle_recall for row in robot), default=1.0):.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=6.0)
    fig.tight_layout()
    save_all_formats(fig, panel_dir / "bounded_coupling", dpi=260)
    plt.close(fig)


def write_gate_a_main_table(
    canonical_records: Sequence[EpisodeResultRecord],
    output: Path,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        row
        for row in canonical_records
        if row.suite == "showcase"
    ]
    ordered_names = [
        "Evolving H=2 (exact)",
        "Evolving H=2 (symbolic local belief)",
        "Root-Structure Reuse H=2",
        "Evolving H=1 (myopic)",
    ]
    by_name = {row.method: row for row in rows}
    md = [
        "| Method | Complete | Final Φ | Options | Duration | Return | First action |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    latex = [
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Method & Complete & Final $\Phi$ & Options & Duration & Return & First action \\",
        r"\midrule",
    ]
    for name in ordered_names:
        row = by_name.get(name)
        if row is None:
            continue
        action = row.first_action.replace("_", r"\_")
        md.append(
            f"| {name} | {'yes' if row.completed else 'no'} | {row.final_potential:.2f} | {row.option_count} | {row.total_duration:.1f} | {row.realized_return:.2f} | {row.first_action} |"
        )
        latex.append(
            f"{name} & {'yes' if row.completed else 'no'} & {row.final_potential:.2f} & {row.option_count} & {row.total_duration:.1f} & {row.realized_return:.2f} & {action} \\\\"
        )
    latex.extend([r"\bottomrule", r"\end{tabular}"])
    (output / "gate_a_main_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (output / "gate_a_main_summary.tex").write_text("\n".join(latex) + "\n", encoding="utf-8")
