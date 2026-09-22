"""Run the M0-M3 oracle suite and write the oracle-gap report.

The diagnostics are deterministic: there is no sampling, so a run is exactly
reproducible from the recorded git revision and configuration.

Gap decomposition follows the research plan and deliberately does **not** add up
non-independent gaps:

* ``physical_invisibility``  = ``J_central* - J_full*``
* ``finite_sharing``         = ``J_B*(B) - J_central*``           (single boundary)
* ``message_representation`` = ``J_codec(codec) - J_B*(B)``       (single boundary)
* ``public_only``            = ``J_codec(null) - J_central*``     (any horizon)
* ``computation``            = ``TrueModelLimitedSearch - J_central*``
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import __version__
from ..exact import (
    Codec,
    ConsequenceCodec,
    NullCodec,
    ObservationCodec,
    OracleResult,
    PosteriorModeCodec,
    PosteriorThresholdCodec,
    o_b_exact,
    o_central,
    o_codec,
    o_full,
    true_model_limited_search,
)
from ..toy import MicroModel, all_instances
from ..toy.model import validate_model


@dataclass(frozen=True)
class DiagnosticConfig:
    horizons: Tuple[int, ...] = (1, 3)
    bits: Tuple[int, ...] = (0, 1, 2)
    limited_search_depth: int = 1
    output_dir: Optional[str] = None
    name: str = "micro"

    def codecs(self) -> Tuple[Codec, ...]:
        return (
            NullCodec(),
            ObservationCodec(bits=1),
            PosteriorModeCodec(bits=1),
            ConsequenceCodec(bits=1),
            PosteriorThresholdCodec(threshold=0.8),
        )


@dataclass
class OracleRecord:
    instance: str
    mechanism: str
    horizon: int
    oracle: str
    codec: str
    bits: int
    expected_cost: float
    success_prob: float
    strategy_class: str
    fixed_codec: bool
    nodes: int
    seconds: float
    exact: bool
    note: str = ""


@dataclass
class InstanceReport:
    instance: str
    mechanism: str
    records: List[OracleRecord] = field(default_factory=list)
    gaps: Dict[str, float] = field(default_factory=dict)
    root_actions: Dict[str, str] = field(default_factory=dict)


def _record(result: OracleResult, mechanism: str) -> OracleRecord:
    return OracleRecord(
        instance=result.instance,
        mechanism=mechanism,
        horizon=result.horizon,
        oracle=result.name,
        codec=result.codec,
        bits=result.bits,
        expected_cost=result.expected_cost,
        success_prob=result.success_prob,
        strategy_class=result.strategy_class,
        fixed_codec=result.fixed_codec,
        nodes=result.nodes,
        seconds=result.seconds,
        exact=result.exact,
        note=result.note,
    )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def run_micro_diagnostics(
    config: Optional[DiagnosticConfig] = None,
    models: Optional[Sequence[MicroModel]] = None,
) -> Tuple[List[InstanceReport], Dict[str, object]]:
    """Run every oracle on every instance at every configured horizon."""

    config = config or DiagnosticConfig()
    max_horizon = max(config.horizons)
    models = models if models is not None else all_instances(horizon=max_horizon)

    reports: List[InstanceReport] = []
    started = time.perf_counter()

    for model in models:
        validate_model(model)
        mechanism = str(model.metadata.get("expectation", ""))
        report = InstanceReport(instance=model.name, mechanism=mechanism)

        for horizon in config.horizons:
            limited = _resize(model, horizon)

            full = o_full(limited)
            central = o_central(limited)
            report.records.append(_record(full, mechanism))
            report.records.append(_record(central, mechanism))

            codec_results: Dict[str, OracleResult] = {}
            for codec in config.codecs():
                result = o_codec(limited, codec)
                codec_results[codec.name] = result
                report.records.append(_record(result, mechanism))

            if horizon == 1:
                for bits in config.bits:
                    result = o_b_exact(limited, bits)
                    report.records.append(_record(result, mechanism))

            limited_result = true_model_limited_search(
                limited, NullCodec(), depth_limit=config.limited_search_depth
            )
            report.records.append(_record(limited_result, mechanism))

            # -- gap decomposition ------------------------------------------
            prefix = f"H{horizon}"
            report.gaps[f"{prefix}.J_full"] = full.expected_cost
            report.gaps[f"{prefix}.J_central"] = central.expected_cost
            report.gaps[f"{prefix}.physical_invisibility"] = (
                central.expected_cost - full.expected_cost
            )
            report.gaps[f"{prefix}.public_only"] = (
                codec_results["null"].expected_cost - central.expected_cost
            )
            best_codec = min(
                value.expected_cost
                for name, value in codec_results.items()
                if name != "null" and name != "posterior-threshold"
            )
            report.gaps[f"{prefix}.best_message_value"] = (
                codec_results["null"].expected_cost - best_codec
            )
            report.gaps[f"{prefix}.computation"] = (
                limited_result.expected_cost - central.expected_cost
            )
            for name, value in codec_results.items():
                report.gaps[f"{prefix}.J_codec[{name}]"] = value.expected_cost
            if horizon == 1:
                o_b = o_b_exact(limited, 1)
                report.gaps[f"{prefix}.J_B(bits=1)"] = o_b.expected_cost
                report.gaps[f"{prefix}.finite_sharing"] = (
                    o_b.expected_cost - central.expected_cost
                )
                for name, value in codec_results.items():
                    report.gaps[f"{prefix}.representation[{name}]"] = (
                        value.expected_cost - o_b.expected_cost
                    )

        reports.append(report)

    summary: Dict[str, object] = {
        "package_version": __version__,
        "git_revision": _git_revision(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "config": {
            "horizons": list(config.horizons),
            "bits": list(config.bits),
            "limited_search_depth": config.limited_search_depth,
            "codecs": [codec.name for codec in config.codecs()],
        },
        "instances": [model.name for model in models],
        "wall_clock_seconds": time.perf_counter() - started,
    }
    return reports, summary


def _resize(model: MicroModel, horizon: int) -> MicroModel:
    import dataclasses

    return model if model.horizon == horizon else dataclasses.replace(model, horizon=horizon)


# --------------------------------------------------------------------------
# Report writing
# --------------------------------------------------------------------------


def _all_records(reports: Sequence[InstanceReport]) -> List[OracleRecord]:
    out: List[OracleRecord] = []
    for report in reports:
        out.extend(report.records)
    return out


def format_results_markdown(reports: Sequence[InstanceReport]) -> str:
    lines: List[str] = []
    lines.append("| instance | H | oracle | codec | bits | expected cost | success | exact | nodes | seconds |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for report in reports:
        for record in report.records:
            cost = "n/a" if record.expected_cost != record.expected_cost else f"{record.expected_cost:.4f}"
            success = "n/a" if record.success_prob != record.success_prob else f"{record.success_prob:.3f}"
            lines.append(
                f"| {record.instance} | {record.horizon} | {record.oracle} | "
                f"{record.codec} | {record.bits} | {cost} | {success} | "
                f"{'yes' if record.exact else 'no'} | {record.nodes} | "
                f"{record.seconds:.3f} |"
            )
    return "\n".join(lines)


def format_gap_markdown(reports: Sequence[InstanceReport]) -> str:
    lines: List[str] = []
    lines.append(
        "Single-boundary decomposition (bits=1): "
        "physical = J_central − J_full; sharing = J_B − J_central; "
        "representation = J_codec − J_B. Gaps are not additive."
    )
    lines.append("")
    lines.append("| instance | physical | finite sharing | repr[observation] | repr[posterior-mode] | repr[consequence] | repr[threshold] |")
    lines.append("|---|---|---|---|---|---|---|")
    for report in reports:
        g = report.gaps
        if "H1.J_B(bits=1)" not in g:
            continue
        lines.append(
            "| {name} | {phys:.4f} | {share:.4f} | {r_obs:.4f} | {r_mode:.4f} | "
            "{r_cons:.4f} | {r_thr:.4f} |".format(
                name=report.instance,
                phys=g["H1.physical_invisibility"],
                share=g["H1.finite_sharing"],
                r_obs=g.get("H1.representation[observation]", float("nan")),
                r_mode=g.get("H1.representation[posterior-mode]", float("nan")),
                r_cons=g.get("H1.representation[consequence]", float("nan")),
                r_thr=g.get("H1.representation[posterior-threshold]", float("nan")),
            )
        )
    lines.append("")
    lines.append(
        "Multi-step (H=3): J_central is the observation-sharing reference, "
        "`public_only` is the value of private messaging over public information, "
        "`computation` is the depth-limited receding-horizon penalty."
    )
    lines.append("")
    lines.append("| instance | J_full | J_central | public_only | best_message_value | computation |")
    lines.append("|---|---|---|---|---|---|")
    for report in reports:
        g = report.gaps
        if "H3.J_central" not in g:
            continue
        lines.append(
            "| {name} | {full:.4f} | {central:.4f} | {pub:.4f} | {msg:.4f} | {comp:.4f} |".format(
                name=report.instance,
                full=g["H3.J_full"],
                central=g["H3.J_central"],
                pub=g["H3.public_only"],
                msg=g["H3.best_message_value"],
                comp=g["H3.computation"],
            )
        )
    return "\n".join(lines)


def write_report(
    reports: Sequence[InstanceReport],
    summary: Dict[str, object],
    output_dir: str,
) -> Dict[str, str]:
    """Write CSV, JSON and Markdown artefacts; return their paths."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    records = _all_records(reports)
    csv_path = out / "oracle_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    paths["oracle_results_csv"] = str(csv_path)

    json_path = out / "oracle_results.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "instances": [
                    {
                        "instance": report.instance,
                        "mechanism": report.mechanism,
                        "records": [asdict(record) for record in report.records],
                        "gaps": report.gaps,
                    }
                    for report in reports
                ],
            },
            indent=2,
            default=str,
        )
    )
    paths["oracle_results_json"] = str(json_path)

    md_path = out / "oracle_results.md"
    md_path.write_text(
        "# Micro oracle results\n\n"
        f"- git revision: `{summary['git_revision']}`\n"
        f"- package version: `{summary['package_version']}`\n"
        f"- horizons: {summary['config']['horizons']}\n"  # type: ignore[index]
        f"- message budgets: {summary['config']['bits']}\n\n"  # type: ignore[index]
        + format_results_markdown(reports)
        + "\n\n"
        + format_gap_markdown(reports)
        + "\n"
    )
    paths["oracle_results_md"] = str(md_path)

    gap_json = out / "oracle_gap_decomposition.json"
    gap_json.write_text(
        json.dumps(
            {report.instance: report.gaps for report in reports}, indent=2, default=str
        )
    )
    paths["oracle_gap_json"] = str(gap_json)

    return paths
