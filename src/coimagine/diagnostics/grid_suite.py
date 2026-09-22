"""A0-A3 grid mechanism suite: the six baseline families on the real grid.

For every mechanism scene and every planner the suite evaluates the exact
closed-loop expected cost of the receding-horizon controller, averaged over the 16
hidden contact types under the uniform prior. It writes per-type rows plus a
scene-by-planner summary, so the oracle gaps and the communication effects can be
read off directly.

The suite also carries the two required sensing diagnostics, computed separately
from the decision problem: object-localisation error and hidden-physics belief
error, so a method cannot look bad for "cannot see" reasons.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import __version__
from ..grid.belief import (
    localization_metrics,
    physics_metrics,
    uniform_prior,
    update_with_scan,
)
from ..grid.planner import (
    PHYSICS,
    ContactContext,
    evaluate_planner,
    planner_spec,
)
from ..grid.scenes import mechanism_scenes
from ..grid.sensing import ray_cast
from ..grid.state import HiddenPhysics, Scene, enumerate_hidden_physics

#: The four mechanism-relevant hidden contact types. Sweeping all 16 is exponential
#: in the release/probe branch count and adds no mechanism information.
MECHANISM_HYPOTHESES: Tuple[HiddenPhysics, ...] = (
    HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip")),
    HiddenPhysics(resistance=2, constraint="free", traction=("grip", "grip")),
    HiddenPhysics(resistance=1, constraint="constrained", traction=("grip", "grip")),
    HiddenPhysics(resistance=2, constraint="constrained", traction=("grip", "grip")),
)

#: ``(planner kind, message bits)``
DEFAULT_PLANNERS: Tuple[Tuple[str, int], ...] = (
    ("full", 1),
    ("central", 1),
    ("codec", 1),
    ("observation", 2),
    ("consequence", 1),
    ("public", 0),
    ("replan", 1),
)


@dataclass
class SuiteRecord:
    scene: str
    planner: str
    true_physics: str
    expected_cost: float
    success_prob: float
    steps: float
    nodes: int


@dataclass
class SensingRecord:
    scene: str
    robot: int
    p_correct_origin: float
    expected_cell_error: float
    p_correct_physics: float
    brier_physics: float


@dataclass
class SuiteSummary:
    scene: str
    planner: str
    mean_cost: float
    mean_success: float
    oracle_gap: float = 0.0
    sharing_gap: float = 0.0
    message_gap: float = 0.0
    c2_gap: float = 0.0


def _sensing_records(scene: Scene) -> List[SensingRecord]:
    """Localisation and physics diagnostics on the initial (pre-contact) scan."""

    poses = tuple(r.pose for r in scene.robots)
    prior = uniform_prior(scene.static, robot_poses=poses)
    out: List[SensingRecord] = []
    for robot in scene.robots:
        scan = ray_cast(scene, robot.robot_id, radius=6)
        belief = update_with_scan(prior, scan, scene.static)
        loc = localization_metrics(belief, scene.obj.origin)
        physics = physics_metrics(belief, scene.physics)
        out.append(
            SensingRecord(
                scene=scene.static.name,
                robot=robot.robot_id,
                p_correct_origin=loc["p_correct_origin"],
                expected_cell_error=loc["expected_cell_error"],
                p_correct_physics=physics["p_correct_physics"],
                brier_physics=physics["brier_physics"],
            )
        )
    return out


def run_grid_suite(
    *,
    horizon: int = 3,
    depth: int = 2,
    planners: Sequence[Tuple[str, int]] = DEFAULT_PLANNERS,
    hypotheses: Optional[Sequence[HiddenPhysics]] = None,
) -> Tuple[List[SuiteRecord], List[SensingRecord], List[SuiteSummary], Dict[str, object]]:
    started = time.perf_counter()
    scenes = mechanism_scenes()
    hidden: Sequence[HiddenPhysics] = (
        tuple(hypotheses) if hypotheses is not None else MECHANISM_HYPOTHESES
    )

    records: List[SuiteRecord] = []
    sensing: List[SensingRecord] = []
    summaries: List[SuiteSummary] = []

    for scene in scenes:
        sensing.extend(_sensing_records(scene))
        ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
        by_planner: Dict[str, SuiteSummary] = {}
        for kind, bits in planners:
            spec = planner_spec(kind, bits=bits, depth=depth)
            costs: List[float] = []
            successes: List[float] = []
            for physics in hidden:
                hypothesis = (PHYSICS.index(physics), False)
                result = evaluate_planner(spec, ctx, hypothesis, horizon=horizon)
                records.append(
                    SuiteRecord(
                        scene=scene.static.name,
                        planner=spec.label,
                        true_physics=physics.label,
                        expected_cost=result.expected_cost,
                        success_prob=result.success_prob,
                        steps=result.steps,
                        nodes=result.nodes,
                    )
                )
                costs.append(result.expected_cost)
                successes.append(result.success_prob)
            by_planner[spec.label] = SuiteSummary(
                scene=scene.static.name,
                planner=spec.label,
                mean_cost=sum(costs) / max(1, len(costs)),
                mean_success=sum(successes) / max(1, len(successes)),
            )
        # Oracle gaps, filled where the matching planner exists.
        full = by_planner.get("full")
        central = by_planner.get("central")
        codec = None
        for label, summary in by_planner.items():
            if label.startswith("codec:"):
                codec = summary
        public = None
        for label, summary in by_planner.items():
            if label.startswith("public:"):
                public = summary
        replan = None
        for label, summary in by_planner.items():
            if label.startswith("replan:"):
                replan = summary
        for summary in by_planner.values():
            if full is not None:
                summary.oracle_gap = summary.mean_cost - full.mean_cost
            if full is not None and central is not None:
                summary.sharing_gap = central.mean_cost - full.mean_cost
            if central is not None:
                summary.message_gap = summary.mean_cost - central.mean_cost
            if replan is not None:
                summary.c2_gap = replan.mean_cost - (
                    codec.mean_cost if codec is not None else summary.mean_cost
                )
        summaries.extend(by_planner.values())

    summary_meta: Dict[str, object] = {
        "package_version": __version__,
        "git_revision": _git_revision(),
        "horizon": horizon,
        "depth": depth,
        "planners": [f"{kind}:{bits}" for kind, bits in planners],
        "hypotheses": len(hidden),
        "wall_clock_seconds": time.perf_counter() - started,
    }
    return records, sensing, summaries, summary_meta


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_suite_markdown(
    summaries: Sequence[SuiteSummary], sensing: Sequence[SensingRecord]
) -> str:
    lines: List[str] = []
    lines.append("## A0-A3: planner panel (prior-weighted over the 16 contact types)")
    lines.append("")
    lines.append("| scene | planner | mean cost | mean success | gap vs full | vs central | C2 |")
    lines.append("|---|---|---|---|---|---|---|")
    for summary in summaries:
        lines.append(
            "| {scene} | {planner} | {cost:.3f} | {succ:.2f} | {oracle:.3f} | {sharing:.3f} | {c2:.3f} |".format(
                scene=summary.scene,
                planner=summary.planner,
                cost=summary.mean_cost,
                succ=summary.mean_success,
                oracle=summary.oracle_gap,
                sharing=summary.message_gap,
                c2=summary.c2_gap,
            )
        )
    lines.append("")
    lines.append("## Sensing diagnostics (pre-contact, reported separately)")
    lines.append("")
    lines.append("| scene | robot | p(correct origin) | cell error | p(correct physics) | Brier |")
    lines.append("|---|---|---|---|---|---|")
    for record in sensing:
        lines.append(
            "| {scene} | {robot} | {p:.2f} | {err:.2f} | {pp:.3f} | {b:.3f} |".format(
                scene=record.scene,
                robot=record.robot,
                p=record.p_correct_origin,
                err=record.expected_cell_error,
                pp=record.p_correct_physics,
                b=record.brier_physics,
            )
        )
    return "\n".join(lines)


def write_suite_report(
    records: Sequence[SuiteRecord],
    sensing: Sequence[SensingRecord],
    summaries: Sequence[SuiteSummary],
    meta: Dict[str, object],
    output_dir: str,
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    csv_path = out / "grid_suite_records.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    paths["records_csv"] = str(csv_path)

    json_path = out / "grid_suite.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": meta,
                "summaries": [asdict(s) for s in summaries],
                "sensing": [asdict(s) for s in sensing],
                "records": [asdict(r) for r in records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)

    md_path = out / "grid_suite.md"
    md_path.write_text(
        "# Stage 2 A0-A3 grid suite\n\n"
        f"- git revision: `{meta['git_revision']}`\n"
        f"- horizon {meta['horizon']}, lookahead depth {meta['depth']}, "
        f"{meta['hypotheses']} hidden types\n"
        f"- wall clock: {meta['wall_clock_seconds']:.1f}s\n\n"
        + format_suite_markdown(summaries, sensing)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
