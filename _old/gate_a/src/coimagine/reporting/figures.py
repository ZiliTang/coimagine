from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from textwrap import fill
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from coimagine.belief.symbolic import RuleBasedBeliefUpdater
from coimagine.core.types import JointOption, LocalOption, OptionKind, PlannerMode
from coimagine.envs.grid.environment import GridEnvironment
from coimagine.envs.grid.generation.procedural import ScenarioInstance
from coimagine.evaluation.bootstrap import bootstrap_mean
from coimagine.evaluation.gate_a import GateAResult
from coimagine.evaluation.records import BaselineResult, ScalingResult, SceneResult
from coimagine.options.generator import GridOptionGenerator
from coimagine.reporting.semantics import (
    BASELINE_DEFINITIONS,
    EVENT_DISPLAY_NAMES,
    METRIC_DEFINITIONS,
    OPTION_COLORS,
    OPTION_DISPLAY_NAMES,
    TASK_DEFINITIONS,
    TOPOLOGY_DEFINITIONS,
    baseline_display_name,
    event_display_name,
    format_joint_option,
    task_display_name,
)
from coimagine.topology.admission import AdmissionResult, TopologyAdmission
from coimagine.visualization.matplotlib_grid import draw_state


@dataclass(frozen=True)
class ExampleBundle:
    instance: ScenarioInstance
    result: GateAResult
    record: SceneResult


def save_all_formats(fig: plt.Figure, base: Path, *, dpi: int = 220) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(base.with_suffix(f".{suffix}"), bbox_inches="tight", dpi=dpi)


def _footer_text(sections: Sequence[tuple[str, Mapping[str, str]]], width: int = 132) -> str:
    blocks: list[str] = []
    for heading, entries in sections:
        lines = [heading.upper()]
        for term, definition in entries.items():
            lines.append(fill(f"{term}: {definition}", width=width, subsequent_indent="    "))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _figure_grid(
    nrows: int,
    ncols: int,
    *,
    width: float,
    main_height: float,
    review_text: str | None = None,
    footer_height: float = 3.0,
):
    if review_text:
        fig = plt.figure(figsize=(width, main_height + footer_height))
        outer = fig.add_gridspec(2, 1, height_ratios=[main_height, footer_height], hspace=0.08)
        inner = outer[0].subgridspec(nrows, ncols, hspace=0.62, wspace=0.34)
        axes = np.asarray(
            [[fig.add_subplot(inner[row, col]) for col in range(ncols)] for row in range(nrows)]
        )
        footer = fig.add_subplot(outer[1])
        footer.axis("off")
        footer.axhline(0.98, color="#B0B0B0", linewidth=0.8)
        footer.text(0.01, 0.94, review_text, va="top", ha="left", fontsize=7.2, family="sans-serif")
        return fig, axes
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, main_height), squeeze=False)
    fig.subplots_adjust(hspace=0.62, wspace=0.34)
    return fig, axes


def _single_axis(
    *, width: float, main_height: float, review_text: str | None = None, footer_height: float = 2.5
):
    fig, axes = _figure_grid(
        1,
        1,
        width=width,
        main_height=main_height,
        review_text=review_text,
        footer_height=footer_height,
    )
    return fig, axes[0, 0]


def _save_clean_and_review(builder, output: Path, name: str) -> None:
    clean_fig = builder(False)
    save_all_formats(clean_fig, output / "paper_clean" / name)
    plt.close(clean_fig)
    review_fig = builder(True)
    save_all_formats(review_fig, output / "paper" / name)
    plt.close(review_fig)


def plot_value_summary(records: list[SceneResult], output: Path) -> None:
    exact = [record for record in records if record.information_mode == "exact_state_oracle"]
    groups: dict[str, list[float]] = defaultdict(list)
    for record in exact:
        groups[record.task_name].append(record.oracle_gain)
    labels = sorted(groups)
    intervals = [bootstrap_mean(groups[label]) for label in labels]
    task_entries = {
        task_display_name(label): TASK_DEFINITIONS.get(
            task_display_name(label),
            "A Gate-A task family reported as a separate scene distribution.",
        )
        for label in labels
    }
    review = _footer_text(
        [
            ("Task families", task_entries),
            ("Metrics", {key: METRIC_DEFINITIONS[key] for key in ("Oracle gain", "First-action regret", "First-action reversal")}),
            ("Reading", {
                "Exact structural value": "The value difference is measured with exact state, exact transition, exact task consequence, and exact executability; no learned-model error is included.",
                "Error bar": "Scene-level bootstrap 95% confidence interval for the mean. Pilot intervals remain provisional because each task distribution is intentionally small.",
            }),
        ],
        width=150,
    )

    def build(with_review: bool):
        fig, ax = _single_axis(
            width=9.2,
            main_height=5.6,
            review_text=review if with_review else None,
            footer_height=5.4,
        )
        positions = np.arange(len(labels))
        means = np.asarray([item.mean for item in intervals])
        errors = np.asarray(
            [
                [item.mean - item.lower for item in intervals],
                [item.upper - item.mean for item in intervals],
            ]
        )
        ax.barh(positions, means, xerr=errors, capsize=3)
        ax.axvline(0.0, linewidth=0.8, color="black")
        ax.set_yticks(positions, [task_display_name(label) for label in labels])
        ax.invert_yaxis()
        ax.set_xlabel(r"Oracle gain $\Delta_{\mathrm{oracle}}$")
        ax.set_title("Exact structural value by task family")
        ax.grid(axis="x", alpha=0.25)
        max_x = max([0.0, *[item.upper for item in intervals]])
        pad = max(0.04, 0.08 * max(1.0, max_x))
        ax.set_xlim(min(-0.03, min(means, default=0.0) - pad), max_x + pad)
        for position, interval, values in zip(positions, intervals, (groups[label] for label in labels)):
            ax.text(
                interval.upper + 0.012,
                position,
                f"n={len(values)}",
                va="center",
                ha="left",
                fontsize=8,
            )
        fig.subplots_adjust(left=0.31, right=0.97, top=0.90, bottom=0.10)
        return fig

    _save_clean_and_review(build, output, "fig_gate_a_value")

    panel_dir = output / "panels" / "gate_a_value"
    for label, interval in zip(labels, intervals):
        fig, ax = plt.subplots(figsize=(3.8, 2.8))
        ax.barh([0], [interval.mean], xerr=[[interval.mean - interval.lower], [interval.upper - interval.mean]], capsize=4)
        ax.axvline(0.0, linewidth=0.8, color="black")
        ax.set_yticks([0], [task_display_name(label)])
        ax.set_xlabel(r"$\Delta_{\mathrm{oracle}}$")
        ax.grid(axis="x", alpha=0.25)
        fig.subplots_adjust(left=0.42, right=0.96, bottom=0.20, top=0.92)
        save_all_formats(fig, panel_dir / f"oracle_gain_{label}")
        plt.close(fig)


def plot_event_summary(records: list[SceneResult], output: Path) -> None:
    groups: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if record.information_mode != "exact_state_oracle":
            continue
        if record.task_name in {"positive_higher_order_synergy", "higher_order_conflict"}:
            continue
        groups[record.dominant_event].append(record.oracle_gain)
    labels = sorted(groups)
    if not labels:
        return
    intervals = [bootstrap_mean(groups[label]) for label in labels]
    review = _footer_text(
        [
            ("Exclusive event classes", {
                "Pure edge creation": "At least one successor executable joint-option template is new and no root edge is destroyed.",
                "Pure edge destruction": "At least one root executable joint-option template disappears and no successor edge is created.",
                "Mixed edge turnover": "Created and destroyed edge sets are both nonempty. This is an exclusive realized-event class, not the diagnostic task-family name Coordination-edge replacement.",
                "No topology change": "The template-level executable complete-joint-option set is unchanged.",
            }),
            ("Scope", {
                "Realized topology event": "The exclusive class computed after the evolving planner's selected first transition by comparing root and exact successor template-level edge sets.",
                "Event stratification": "Horizon-one higher-order factor diagnostics are excluded because they test factor admission rather than a temporal topology transition.",
                "Bar height": "Mean oracle gain over all exact-state scene records assigned to the realized event class; the sample count is printed next to each bar.",
            }),
        ],
        width=150,
    )

    def build(with_review: bool):
        fig, ax = _single_axis(
            width=8.4,
            main_height=4.2,
            review_text=review if with_review else None,
            footer_height=3.8,
        )
        positions = np.arange(len(labels))
        means = [item.mean for item in intervals]
        ax.barh(positions, means)
        ax.axvline(0.0, linewidth=0.8, color="black")
        ax.set_yticks(positions, [event_display_name(label) for label in labels])
        ax.invert_yaxis()
        ax.set_xlabel(r"Mean oracle gain $\Delta_{\mathrm{oracle}}$")
        ax.set_title("Oracle gain by exclusive realized topology event")
        ax.grid(axis="x", alpha=0.25)
        max_x = max([0.0, *means])
        ax.set_xlim(min(-0.02, min(means, default=0.0) - 0.02), max_x + max(0.05, 0.12 * max(1.0, max_x)))
        for position, label, mean in zip(positions, labels, means):
            ax.text(mean + 0.012, position, f"n={len(groups[label])}", va="center", ha="left", fontsize=8)
        fig.subplots_adjust(left=0.30, right=0.97, top=0.88, bottom=0.14)
        return fig

    _save_clean_and_review(build, output, "fig_gate_a_events")


def plot_scaling(results: list[ScalingResult], output: Path) -> None:
    entity = sorted([result for result in results if result.study == "entity_scaling"], key=lambda item: item.x_value)
    if entity:
        review = _footer_text(
            [("Terms", {
                "Naive joint checks": "Unstructured comparison count (MK)^N for N robots, M entities, and K options per entity.",
                "Entity-local checks": "Sum of per-entity checks, approximately M K^N when unrelated entities need no direct compatibility test.",
                "Log scale": "The vertical axis is logarithmic; equal vertical spacing represents multiplicative growth.",
            })]
        )

        def build_entity(with_review: bool):
            fig, ax = _single_axis(width=5.8, main_height=3.8, review_text=review if with_review else None, footer_height=2.3)
            ax.plot([item.x_value for item in entity], [item.naive_count for item in entity], marker="o", label="Naive joint checks")
            ax.plot([item.x_value for item in entity], [item.factor_count for item in entity], marker="s", label="Entity-local checks")
            ax.set_yscale("log")
            ax.set_xlabel("Number of entities M")
            ax.set_ylabel("Checks")
            ax.set_title("Entity-factor scaling")
            ax.grid(alpha=0.25)
            ax.legend()
            return fig

        _save_clean_and_review(build_entity, output, "fig_entity_scaling")

        for metric in ("naive_count", "factor_count"):
            fig, ax = plt.subplots(figsize=(3.4, 2.8))
            ax.plot([item.x_value for item in entity], [getattr(item, metric) for item in entity], marker="o")
            ax.set_yscale("log")
            ax.set_xlabel("M")
            ax.set_ylabel(metric.replace("_", " "))
            ax.grid(alpha=0.25)
            save_all_formats(fig, output / "panels" / "scaling" / metric)
            plt.close(fig)

    robot = [result for result in results if result.study == "robot_factor_scaling"]
    if robot:
        review = _footer_text(
            [("Terms", {
                "N": "Total robot count.",
                "P": "Maximum irreducible factor arity used by recursive verification.",
                "Recursive verifier calls": "Number of exact partial-tuple executability/extendability evaluations in the synthetic factor benchmark.",
                "Full joint tuples K^N": "Complete Cartesian enumeration reference, shown as a dashed curve.",
                "Oracle recall": "All plotted recursive settings retain the exact high-value completions; recall is reported separately in the data table.",
            })]
        )

        def build_robot(with_review: bool):
            fig, ax = _single_axis(width=6.2, main_height=4.0, review_text=review if with_review else None, footer_height=2.6)
            arities = sorted({item.factor_arity for item in robot})
            for arity in arities:
                rows = sorted([item for item in robot if item.factor_arity == arity], key=lambda item: item.robot_count)
                ax.plot([item.robot_count for item in rows], [item.recursive_calls for item in rows], marker="o", label=f"Recursive, P={arity}")
            refs = {}
            for item in robot:
                refs[item.robot_count] = item.naive_count
            ax.plot(sorted(refs), [refs[n] for n in sorted(refs)], linestyle="--", linewidth=1.5, label=r"Full tuples $K^N$")
            ax.set_yscale("log")
            ax.set_xlabel("Number of robots N")
            ax.set_ylabel("Tuple evaluations")
            ax.set_title("Bounded-arity recursive scaling")
            ax.grid(alpha=0.25)
            ax.legend()
            return fig

        _save_clean_and_review(build_robot, output, "fig_robot_scaling")


def _effective_budget(instance: ScenarioInstance, default: int) -> int:
    return int(instance.state.task.metadata.get("candidate_budget", default))


def _exact_admission(state, candidate_budget: int, max_arity: int) -> AdmissionResult:
    if GridEnvironment(observation_radius=max(state.width, state.height)).is_terminal(state):
        return AdmissionResult(
            candidate_options={rid: tuple() for rid in state.robots},
            joint_options=tuple(),
            total_cartesian=0,
            admitted_count=0,
            factor_count=0,
            verifier_calls=0,
            retained_tuples=0,
        )
    env = GridEnvironment(observation_radius=max(state.width, state.height))
    generator = GridOptionGenerator(candidate_budget=candidate_budget)
    updater = RuleBasedBeliefUpdater()
    beliefs = {rid: updater.initialize(env.observe(state, rid), state.width, state.height) for rid in state.robots}
    candidates = {rid: generator.generate(belief) for rid, belief in beliefs.items()}
    return TopologyAdmission(env, max_arity=max_arity).admit(state, candidates, PlannerMode.FULL_TUPLE_ORACLE)


def _successor_admission(bundle: ExampleBundle, candidate_budget: int, max_arity: int) -> AdmissionResult:
    trace = bundle.result.evolving.trace
    if len(trace) >= 2:
        return trace[1].admission
    successor = trace[0].state_after if trace else bundle.instance.state
    return _exact_admission(successor, _effective_budget(bundle.instance, candidate_budget), max_arity)


def _short_entity(entity_id: str | None) -> str:
    if entity_id is None:
        return ""
    return entity_id.replace("blocker-", "B").replace("hidden-blocker", "hidden").replace("visible-decoy", "decoy")


def _option_label(option: LocalOption) -> str:
    parts = [OPTION_DISPLAY_NAMES[option.kind]]
    if option.entity_id:
        parts.append(_short_entity(option.entity_id))
    if option.direction:
        parts.append(option.direction.value[0].upper())
    if option.slot_token and option.slot_token not in {"idle", option.entity_id}:
        slot = option.slot_token.replace("north:", "N").replace("east:", "E").replace("south:", "S").replace("west:", "W")
        parts.append(slot)
    return "·".join(parts)


def _edge_styles_for_phase(
    admission: AdmissionResult,
    other: AdmissionResult,
    *,
    phase: str,
):
    current = {joint.template_key for joint in admission.joint_options}
    other_set = {joint.template_key for joint in other.joint_options}
    if phase == "root":
        changed = current - other_set
    else:
        changed = current - other_set
    persistent = current & other_set
    return changed, persistent


def _draw_topology(
    ax,
    admission: AdmissionResult,
    other: AdmissionResult,
    *,
    title: str,
    phase: str,
    selected: JointOption | None,
    edge_cap: int = 42,
) -> None:
    candidates = admission.candidate_options
    joints = admission.joint_options
    if not any(candidates.values()):
        ax.text(0.5, 0.56, "Terminal boundary", ha="center", va="center", fontsize=11, fontweight="medium")
        ax.text(0.5, 0.43, "No future candidate-induced topology", ha="center", va="center", fontsize=8)
        ax.set_title(title)
        ax.axis("off")
        return

    robot_ids = sorted(candidates)
    positions: dict[str, tuple[float, float]] = {}
    for column, rid in enumerate(robot_ids):
        options = list(candidates[rid])
        ys = np.linspace(0.94, 0.08, max(1, len(options)))
        for y, option in zip(ys, options):
            positions[option.option_id] = (column, float(y))
            ax.scatter([column], [y], s=60, color=OPTION_COLORS[option.kind], edgecolors="white", linewidths=0.5, zorder=4)
            ax.text(column + 0.035, y, _option_label(option), va="center", fontsize=5.6)

    changed, persistent = _edge_styles_for_phase(admission, other, phase=phase)
    selected_key = selected.template_key if selected is not None else None

    def priority(joint: JointOption):
        key = joint.template_key
        return (0 if key == selected_key else 1, 0 if key in changed else 1, tuple(map(str, key)))

    visible_joints = [joint for joint in sorted(joints, key=priority) if all(option.option_id in positions for option in joint.options)]
    visible_joints = visible_joints[:edge_cap]
    for joint in visible_joints:
        key = joint.template_key
        points = [positions[option.option_id] for option in sorted(joint.options, key=lambda item: item.robot_id)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        if key == selected_key:
            color, linewidth, linestyle, alpha, zorder = "#111111", 2.2, "-", 0.95, 3
        elif key in changed and phase == "root":
            color, linewidth, linestyle, alpha, zorder = "#C44E52", 1.1, "--", 0.78, 2
        elif key in changed and phase == "successor":
            color, linewidth, linestyle, alpha, zorder = "#2A9D55", 1.2, "-", 0.80, 2
        else:
            color, linewidth, linestyle, alpha, zorder = "#A8A8A8", 0.55, "-", 0.32, 1
        ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle, alpha=alpha, zorder=zorder)

    node_count = sum(len(options) for options in candidates.values())
    shown = len(visible_joints)
    ax.set_xlim(-0.25, max(1, len(robot_ids) - 1) + 0.75)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(robot_ids)), [f"Robot {rid}" for rid in robot_ids])
    ax.set_yticks([])
    ax.set_title(
        f"{title}\n{node_count} nodes · {len(joints)} edges ({shown} shown)",
        fontsize=9.2,
        pad=7,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_qualitative_examples(
    examples: Mapping[str, ExampleBundle],
    output: Path,
) -> None:
    temporal_names = [name for name in ("edge_creation", "edge_replacement", "hold_pass", "reveal_act") if name in examples]
    if not temporal_names:
        return
    task_defs = {task_display_name(name): TASK_DEFINITIONS[task_display_name(name)] for name in temporal_names}
    review = _footer_text(
        [
            ("Task families", task_defs),
            ("Methods", {
                "Oracle Evolving Topology": BASELINE_DEFINITIONS["Oracle Evolving Topology"],
                "Root-Structure Reuse": BASELINE_DEFINITIONS["Root-Structure Reuse"],
            }),
            ("Values", {"Number in panel title": "Finite-horizon cumulative task-potential improvement minus option-duration penalties."}),
        ],
        width=150,
    )

    def build(with_review: bool):
        fig, axes = _figure_grid(len(temporal_names), 3, width=10.4, main_height=3.0 * len(temporal_names), review_text=review if with_review else None, footer_height=3.6)
        for row, name in enumerate(temporal_names):
            bundle = examples[name]
            root = bundle.instance.state
            evolving_after = bundle.result.evolving.trace[0].state_after if bundle.result.evolving.trace else root
            fixed_after = bundle.result.fixed.trace[0].state_after if bundle.result.fixed.trace else root
            draw_state(axes[row, 0], root, title=f"{task_display_name(name)}: root")
            draw_state(axes[row, 1], evolving_after, title=f"Evolving successor, J={bundle.result.evolving.value:.2f}")
            draw_state(axes[row, 2], fixed_after, title=f"Root reuse successor, J={bundle.result.fixed.value:.2f}")
        return fig

    _save_clean_and_review(build, output, "fig_gate_a_examples")

    panel_dir = output / "panels" / "gate_a_examples"
    for name, bundle in examples.items():
        states = {
            "root": bundle.instance.state,
            "evolving_successor": bundle.result.evolving.trace[0].state_after if bundle.result.evolving.trace else bundle.instance.state,
            "fixed_successor": bundle.result.fixed.trace[0].state_after if bundle.result.fixed.trace else bundle.instance.state,
        }
        for suffix, state in states.items():
            fig, ax = plt.subplots(figsize=(3.5, 3.0))
            draw_state(ax, state, title=f"{task_display_name(name)} — {suffix.replace('_', ' ')}")
            save_all_formats(fig, panel_dir / f"{name}_{suffix}")
            plt.close(fig)


def plot_topology_examples(
    examples: Mapping[str, ExampleBundle],
    output: Path,
    *,
    candidate_budget: int,
    max_arity: int,
) -> None:
    names = [name for name in ("edge_creation", "edge_replacement", "hold_pass", "reveal_act") if name in examples]
    if not names:
        return
    task_defs = {task_display_name(name): TASK_DEFINITIONS[task_display_name(name)] for name in names}
    topo_defs = {
        key: TOPOLOGY_DEFINITIONS[key]
        for key in ("Root topology", "Regenerated successor topology", "Local-option node", "Executable joint-option edge", "Created edge", "Destroyed edge", "Persistent edge", "Selected edge")
    }
    option_defs = {
        "Approach": "Reposition to a task-defined support, contact-preparation, observation, or passage target without intentionally moving a blocker.",
        "Push": "Move a named blocker by one grid cell in the indicated direction from a valid support/contact slot.",
        "Hold": "Remain at the designated gate cell for the whole macro-option to enable a synchronized pass.",
        "Observe": "Reveal the geometry of a semantically known hidden entity from a designated observation cell.",
        "Idle": "Take no active task action while preserving the option-boundary contract.",
    }
    review = _footer_text(
        [
            ("Topology terms and line styles", topo_defs),
            ("Option labels", option_defs),
            ("Diagnostic rows", task_defs),
            ("Display rule", {"Displayed edges": "All nodes are shown. Edges are deterministically prioritized as selected, changed, then persistent and capped for legibility; the title states displayed/total counts. Counts always refer to the complete exact topology."}),
        ],
        width=155,
    )

    def build(with_review: bool):
        fig, axes = _figure_grid(len(names), 2, width=9.4, main_height=3.65 * len(names), review_text=review if with_review else None, footer_height=6.0)
        fig.subplots_adjust(top=0.945)
        for row, name in enumerate(names):
            bundle = examples[name]
            root = bundle.result.evolving.root_admission
            successor = _successor_admission(bundle, candidate_budget, max_arity)
            root_selected = bundle.result.evolving.joint_option
            successor_selected = bundle.result.evolving.trace[1].joint_option if len(bundle.result.evolving.trace) > 1 else None
            _draw_topology(
                axes[row, 0], root, successor,
                title=f"{task_display_name(name)} — root topology",
                phase="root",
                selected=root_selected,
            )
            _draw_topology(
                axes[row, 1], successor, root,
                title="Regenerated successor topology",
                phase="successor",
                selected=successor_selected,
            )
        legend = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=OPTION_COLORS[kind], markeredgecolor="white", label=OPTION_DISPLAY_NAMES[kind])
            for kind in OptionKind
        ] + [
            Line2D([0], [0], color="#111111", linewidth=2.2, label="Selected edge"),
            Line2D([0], [0], color="#2A9D55", linewidth=1.2, label="Created edge"),
            Line2D([0], [0], color="#C44E52", linewidth=1.1, linestyle="--", label="Destroyed edge"),
            Line2D([0], [0], color="#A8A8A8", linewidth=0.8, label="Persistent edge"),
        ]
        fig.legend(
            handles=legend,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.992),
            ncol=len(legend),
            fontsize=6.3,
            frameon=False,
            handlelength=1.6,
            columnspacing=0.9,
        )
        return fig

    _save_clean_and_review(build, output, "fig_gate_a_topologies")

    panel_dir = output / "panels" / "gate_a_topologies"
    for name, bundle in examples.items():
        root = bundle.result.evolving.root_admission
        successor = _successor_admission(bundle, candidate_budget, max_arity)
        panels = (
            ("root", root, successor, "root", bundle.result.evolving.joint_option),
            ("successor", successor, root, "successor", bundle.result.evolving.trace[1].joint_option if len(bundle.result.evolving.trace) > 1 else None),
        )
        for suffix, admission, other, phase, selected in panels:
            fig, ax = plt.subplots(figsize=(4.4, 3.2))
            _draw_topology(ax, admission, other, title=f"{task_display_name(name)} — {suffix}", phase=phase, selected=selected)
            save_all_formats(fig, panel_dir / f"{name}_{suffix}")
            plt.close(fig)


def plot_factor_diagnostics(
    examples: Mapping[str, ExampleBundle],
    baseline_records: Sequence[BaselineResult],
    output: Path,
) -> None:
    names = [name for name in ("positive_higher_order_synergy", "higher_order_conflict") if name in examples]
    if not names:
        return
    by_key = {(row.task_name, row.mode): row for row in baseline_records}
    review = _footer_text(
        [
            ("Mechanisms", {task_display_name(name): TASK_DEFINITIONS[task_display_name(name)] for name in names}),
            ("Baselines", {
                "Pairwise-Clique Verification": BASELINE_DEFINITIONS["Pairwise-Clique Verification"],
                "Recursive Exact Factors": BASELINE_DEFINITIONS["Recursive Exact Factors"],
                "Exact Full-Tuple Oracle": BASELINE_DEFINITIONS["Exact Full-Tuple Oracle"],
            }),
            ("Metrics", {"Oracle recall": METRIC_DEFINITIONS["Oracle recall"], "Oracle precision": METRIC_DEFINITIONS["Oracle precision"]}),
        ],
        width=145,
    )

    def build(with_review: bool):
        fig, axes = _figure_grid(len(names), 2, width=8.4, main_height=3.2 * len(names), review_text=review if with_review else None, footer_height=4.0)
        for row_index, name in enumerate(names):
            bundle = examples[name]
            admission = bundle.result.evolving.root_admission
            _draw_topology(axes[row_index, 0], admission, admission, title=f"{task_display_name(name)} — exact root topology", phase="successor", selected=bundle.result.evolving.joint_option, edge_cap=20)
            ax = axes[row_index, 1]
            methods = [PlannerMode.PAIRWISE_ONLY.value, PlannerMode.RECURSIVE_FACTORS.value, PlannerMode.FULL_TUPLE_ORACLE.value]
            rows = [by_key.get((name, mode)) for mode in methods]
            labels = [baseline_display_name(mode).replace(" ", "\n") for mode in methods]
            recall = [item.exact_root_recall if item else 0.0 for item in rows]
            precision = [item.exact_root_precision if item else 0.0 for item in rows]
            x = np.arange(len(labels))
            ax.bar(x - 0.18, recall, width=0.36, label="Oracle recall")
            ax.bar(x + 0.18, precision, width=0.36, label="Oracle precision")
            ax.set_ylim(0, 1.08)
            ax.set_xticks(x, labels)
            ax.set_ylabel("Fraction")
            ax.set_title("Admission fidelity")
            ax.grid(axis="y", alpha=0.25)
            ax.legend(fontsize=7)
            for idx, item in enumerate(rows):
                if item:
                    ax.text(idx, 0.04, f"calls={item.verifier_calls}", ha="center", va="bottom", fontsize=6, rotation=90)
        return fig

    _save_clean_and_review(build, output, "fig_factor_diagnostics")
