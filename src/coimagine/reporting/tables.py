from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from coimagine.evaluation.bootstrap import bootstrap_mean
from coimagine.evaluation.records import BaselineResult, SceneResult
from coimagine.reporting.semantics import (
    BASELINE_DEFINITIONS,
    METRIC_DEFINITIONS,
    TASK_DEFINITIONS,
    TOPOLOGY_DEFINITIONS,
    event_display_name,
    task_display_name,
)


def summarize_scenes(records: list[SceneResult]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[SceneResult]] = defaultdict(list)
    for record in records:
        groups[(record.suite, record.task_name, record.information_mode)].append(record)
    rows = []
    for (suite, task, information), items in sorted(groups.items()):
        gain = bootstrap_mean([item.oracle_gain for item in items])
        regret = bootstrap_mean([item.first_action_regret for item in items])
        rows.append(
            {
                "suite": suite,
                "task": task_display_name(task),
                "information": information,
                "n": len(items),
                "oracle_gain": gain.mean,
                "oracle_gain_ci_low": gain.lower,
                "oracle_gain_ci_high": gain.upper,
                "first_action_regret": regret.mean,
                "positive_gain_fraction": sum(item.positive_gain for item in items) / len(items),
                "reversal_fraction": sum(item.first_action_reversal for item in items) / len(items),
                "rank_agreement": sum(item.rank_agreement for item in items) / len(items),
                "mechanism_valid_fraction": sum(item.mechanism_valid for item in items) / len(items),
                "created_edges": sum(item.created_edge_count for item in items) / len(items),
                "destroyed_edges": sum(item.destroyed_edge_count for item in items) / len(items),
                "planning_seconds": sum(item.total_planning_seconds for item in items) / len(items),
            }
        )
    return rows


def summarize_events(records: list[SceneResult]) -> list[dict[str, object]]:
    groups: dict[str, list[SceneResult]] = defaultdict(list)
    for record in records:
        if record.information_mode != "exact_state_oracle":
            continue
        # Horizon-one higher-order factor diagnostics are not temporal-event evidence.
        if record.task_name in {"positive_higher_order_synergy", "higher_order_conflict"}:
            continue
        groups[record.dominant_event].append(record)
    rows: list[dict[str, object]] = []
    for event, items in sorted(groups.items()):
        gain = bootstrap_mean([item.oracle_gain for item in items])
        rows.append(
            {
                "dominant_event": event_display_name(event),
                "n": len(items),
                "oracle_gain": gain.mean,
                "ci_low": gain.lower,
                "ci_high": gain.upper,
                "positive_gain_fraction": sum(item.positive_gain for item in items) / len(items),
                "reversal_fraction": sum(item.first_action_reversal for item in items) / len(items),
                "created_edges": sum(item.created_edge_count for item in items) / len(items),
                "destroyed_edges": sum(item.destroyed_edge_count for item in items) / len(items),
            }
        )
    return rows


def summarize_baselines(records: list[BaselineResult]) -> list[dict[str, object]]:
    groups: dict[str, list[BaselineResult]] = defaultdict(list)
    for record in records:
        groups[record.method_name].append(record)
    rows = []
    for method, items in sorted(groups.items()):
        rows.append(
            {
                "method": method,
                "value": sum(item.value for item in items) / len(items),
                "admitted": sum(item.admitted_count for item in items) / len(items),
                "verifier_calls": sum(item.verifier_calls for item in items) / len(items),
                "retained_tuples": sum(item.retained_tuples for item in items) / len(items),
                "oracle_recall": sum(item.exact_root_recall for item in items) / len(items),
                "oracle_precision": sum(item.exact_root_precision for item in items) / len(items),
            }
        )
    return rows


def _fmt(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def write_markdown_table(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("No results.\n", encoding="utf-8")
        return
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    lines.extend("| " + " | ".join(_fmt(row[column]) for column in columns) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: object) -> str:
    text = _fmt(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def write_latex_table(path: Path, rows: list[dict[str, object]], caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("% No results.\n", encoding="utf-8")
        return
    columns = list(rows[0])
    escaped = [_latex_escape(column) for column in columns]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{" + "l" * len(columns) + "}",
        "\\hline",
        " & ".join(escaped) + " \\\\",
        "\\hline",
    ]
    for row in rows:
        lines.append(" & ".join(_latex_escape(row[column]) for column in columns) + " \\\\")
    lines.extend([
        "\\hline",
        "\\end{tabular}%",
        "}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\end{table*}",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_semantics_glossary(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sections = [
        ("Topology and visualization terms", TOPOLOGY_DEFINITIONS),
        ("Diagnostic task families", TASK_DEFINITIONS),
        ("Baseline methods", BASELINE_DEFINITIONS),
        ("Metrics", METRIC_DEFINITIONS),
    ]
    md: list[str] = ["# Gate A terminology and mechanism glossary", ""]
    for title, entries in sections:
        md.extend([f"## {title}", "", "| Term | Definition |", "|---|---|"])
        for term, definition in entries.items():
            md.append(f"| {term} | {definition} |")
        md.append("")
    (output_dir / "gate_a_glossary.md").write_text("\n".join(md), encoding="utf-8")

    tex = ["% Auto-generated Gate-A glossary.", "\\begin{description}"]
    for _, entries in sections:
        for term, definition in entries.items():
            tex.append(f"\\item[{_latex_escape(term)}] {_latex_escape(definition)}")
    tex.append("\\end{description}")
    (output_dir / "gate_a_glossary.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
