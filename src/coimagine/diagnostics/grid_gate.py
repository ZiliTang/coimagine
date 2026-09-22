"""Stage 2 design gate: are C1 and C2 decidable in the grid's hidden structure?

Run before committing to a full grid benchmark. It answers two questions on the
finite grid abstraction:

* **C1 room** -- at a matched one-bit budget, does the choice of message matter?
  ``J_codec(posterior-mode) - J_B(1) > 0`` means a naive compact posterior wastes the
  bit and a better representation exists.
* **C2 value** -- does explicitly anticipating the next round's message change the
  *first* decision? ``J_replan_no_message_branch - J_codec > 0`` for a useful codec.

Both are exact, because the abstraction is a finite tabular model.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import __version__
from ..exact import (
    Codec,
    ConsequenceCodec,
    NullCodec,
    ObservationCodec,
    PosteriorModeCodec,
    o_b_exact,
    o_central,
    o_codec,
    o_full,
    o_replan_no_message_branch,
)
from ..toy.grid_abstraction import (
    build_grid_abstraction,
    build_grid_split_abstraction,
)

GATE_CONFIGS: Tuple[Tuple[str, Callable[..., object], bool], ...] = (
    ("same_face_informed", build_grid_abstraction, True),
    ("same_face_blind", build_grid_abstraction, False),
    ("split_informed", build_grid_split_abstraction, True),
    ("split_blind", build_grid_split_abstraction, False),
)


@dataclass
class GateRecord:
    config: str
    horizon: int
    planner: str
    codec: str
    bits: int
    expected_cost: float
    exact: bool


@dataclass
class GateReport:
    config: str
    informed: bool
    records: List[GateRecord] = field(default_factory=list)
    gaps: Dict[str, float] = field(default_factory=dict)


def _codecs() -> Tuple[Codec, ...]:
    return (
        NullCodec(),
        ObservationCodec(bits=1),
        PosteriorModeCodec(bits=1),
        ConsequenceCodec(bits=1),
    )


def _record(config: str, horizon: int, planner: str, codec: str, bits: int, result) -> GateRecord:
    return GateRecord(
        config=config,
        horizon=horizon,
        planner=planner,
        codec=codec,
        bits=bits,
        expected_cost=result.expected_cost,
        exact=result.exact,
    )


def run_grid_gate(
    configs: Sequence[Tuple[str, Callable[..., object], bool]] = GATE_CONFIGS,
) -> Tuple[List[GateReport], Dict[str, object]]:
    """Run the gate for every configuration and horizon."""

    started = time.perf_counter()
    reports: List[GateReport] = []

    for name, builder, informed in configs:
        report = GateReport(config=name, informed=informed)
        model1 = builder(horizon=1, initial_informative=informed)
        model3 = builder(horizon=3, initial_informative=informed)

        full1, central1 = o_full(model1), o_central(model1)
        report.records.append(_record(name, 1, "O_full", "n/a", -1, full1))
        report.records.append(_record(name, 1, "O_central", "n/a", -1, central1))
        for bits in (0, 1):
            result = o_b_exact(model1, bits)
            report.records.append(
                _record(name, 1, "O_B", "enumerated", bits, result)
            )

        codec1: Dict[str, float] = {}
        for codec in _codecs():
            result = o_codec(model1, codec)
            report.records.append(_record(name, 1, "O_codec", codec.name, codec.bits, result))
            codec1[codec.name] = result.expected_cost

        full3, central3 = o_full(model3), o_central(model3)
        report.records.append(_record(name, 3, "O_full", "n/a", -1, full3))
        report.records.append(_record(name, 3, "O_central", "n/a", -1, central3))

        codec3: Dict[str, float] = {}
        replan3: Dict[str, float] = {}
        for codec in _codecs():
            result = o_codec(model3, codec)
            report.records.append(_record(name, 3, "O_codec", codec.name, codec.bits, result))
            codec3[codec.name] = result.expected_cost
            control = o_replan_no_message_branch(model3, codec)
            report.records.append(
                _record(name, 3, "O_replan_no_message_branch", codec.name, codec.bits, control)
            )
            replan3[codec.name] = control.expected_cost

        informative = [n for n in ("observation", "consequence") if n in codec3]
        j_b0 = next(
            r.expected_cost
            for r in report.records
            if r.horizon == 1 and r.planner == "O_B" and r.bits == 0
        )
        j_b1 = next(
            r.expected_cost
            for r in report.records
            if r.horizon == 1 and r.planner == "O_B" and r.bits == 1
        )
        report.gaps = {
            "H1.J_full": full1.expected_cost,
            "H1.J_central": central1.expected_cost,
            "H1.J_B(bits=0)": j_b0,
            "H1.J_B(bits=1)": j_b1,
            # > 0 means the naive compact posterior wastes the bit.
            "H1.message_representation_loss": codec1["posterior-mode"] - j_b1,
            "H3.J_full": full3.expected_cost,
            "H3.J_central": central3.expected_cost,
            # > 0 means a better message representation exists at H = 3.
            "H3.c1_room": (
                codec3["posterior-mode"] - min(codec3[n] for n in informative)
                if informative
                else float("nan")
            ),
        }
        for name_codec, value in codec3.items():
            report.gaps[f"H3.J_codec[{name_codec}]"] = value
            report.gaps[f"H3.c2_value[{name_codec}]"] = replan3[name_codec] - value
        reports.append(report)

    summary: Dict[str, object] = {
        "package_version": __version__,
        "git_revision": _git_revision(),
        "configs": [name for name, _b, _i in configs],
        "wall_clock_seconds": time.perf_counter() - started,
    }
    return reports, summary


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_gate_markdown(reports: Sequence[GateReport]) -> str:
    lines: List[str] = []
    lines.append("## C1 room: does the message representation matter? (H = 1 and H = 3)")
    lines.append("")
    lines.append("| config | J_B(0) | J_B(1) | J_central H1 | mode | observation | consequence | repr. loss | C1 room H3 |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for report in reports:
        g = report.gaps
        h1 = {r.codec: r.expected_cost for r in report.records if r.horizon == 1 and r.planner == "O_codec"}
        lines.append(
            "| {cfg} | {b0:.4f} | {b1:.4f} | {c1:.4f} | {mode:.4f} | {obs:.4f} | {cons:.4f} | {loss:.4f} | {room:.4f} |".format(
                cfg=report.config,
                b0=g["H1.J_B(bits=0)"],
                b1=g["H1.J_B(bits=1)"],
                c1=g["H1.J_central"],
                mode=h1.get("posterior-mode", float("nan")),
                obs=h1.get("observation", float("nan")),
                cons=h1.get("consequence", float("nan")),
                loss=g["H1.message_representation_loss"],
                room=g["H3.c1_room"],
            )
        )
    lines.append("")
    lines.append("## C2 value: does anticipating the next message change the first decision? (H = 3)")
    lines.append("")
    lines.append("| config | codec | O_codec | O_replan | C2 value |")
    lines.append("|---|---|---|---|---|")
    for report in reports:
        for codec in ("null", "observation", "posterior-mode", "consequence"):
            key = f"H3.c2_value[{codec}]"
            if key not in report.gaps:
                continue
            lines.append(
                "| {cfg} | {codec} | {jc:.4f} | {jr:.4f} | {c2:.4f} |".format(
                    cfg=report.config,
                    codec=codec,
                    jc=report.gaps[f"H3.J_codec[{codec}]"],
                    jr=report.gaps[f"H3.J_codec[{codec}]"] + report.gaps[key],
                    c2=report.gaps[key],
                )
            )
    return "\n".join(lines)


def write_gate_report(
    reports: Sequence[GateReport], summary: Dict[str, object], output_dir: str
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    records = [record for report in reports for record in report.records]
    csv_path = out / "grid_gate_results.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    paths["csv"] = str(csv_path)

    json_path = out / "grid_gate.json"
    json_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "configs": [
                    {
                        "config": report.config,
                        "informed": report.informed,
                        "gaps": report.gaps,
                        "records": [asdict(r) for r in report.records],
                    }
                    for report in reports
                ],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)

    md_path = out / "grid_gate.md"
    md_path.write_text(
        "# Stage 2 design gate\n\n"
        f"- git revision: `{summary['git_revision']}`\n"
        f"- configs: {', '.join(summary['configs'])}\n\n"  # type: ignore[arg-type]
        + format_gate_markdown(reports)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
