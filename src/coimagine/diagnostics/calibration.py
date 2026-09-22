"""Calibration testbed: does a non-privileged predictor measurably lose to the ceiling?

The comparison is deliberately narrow and fair:

* **ceiling** -- the true simulator state and the true contact physics, same planner,
  same candidate set, same lookahead depth;
* **non-privileged** -- the known transition rules, but the object's pose and the
  contact physics are the robot's own estimate from its own ray scan, and it never
  reads a privileged quantity.

Both planners use the pose-agnostic public candidate template set and the identical
search heuristic, so the cost difference is attributable to what the robot knows, not
to how it searches.

The report separates two losses:

* **pose loss** -- belief support larger than one placement (the robot does not know
  where the object is);
* **physics loss** -- belief support exactly one placement, yet the contact physics is
  still unknown.

If neither is measurable the testbed is too easy and no learned model is justified.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from .. import __version__
from ..grid import scene_from_ascii
from ..grid.belief import localization_metrics, physics_metrics, uniform_prior, update_with_scan
from ..grid.planner import (
    PHYSICS,
    ContactContext,
    belief_from_posterior,
    evaluate_planner,
    planner_spec,
)
from ..grid.sensing import SensorNoise, apply_noise, ray_cast
from ..grid.state import Cell, HiddenPhysics, Scene

#: Goal region used by the calibration scenes: two cells wide, three tall, so that a
#: two-cell east push from either ambiguous placement lands inside it.
CALIBRATION_TARGET = frozenset(
    {(4, 1), (5, 1), (4, 2), (5, 2), (4, 3), (5, 3)}
)

#: The public template set used at the decision point. It is map-derived and
#: pose-agnostic: it contains no information about where the object is.
CALIBRATION_CANDIDATES: Tuple[str, ...] = (
    "reposition:0:1,1",
    "reposition:0:1,2",
    "reposition:1:1,1",
    "reposition:1:1,2",
    "co_push:E",
    "push:0:E",
    "push:1:E",
    "probe:0:E",
    "release:0:E",
    "wait",
)

PHYSICS_CALIBRATION = HiddenPhysics(resistance=1, constraint="free", traction=("grip", "grip"))


def build_calibration_scene(
    origin: Cell, target=frozenset(CALIBRATION_TARGET)
) -> Scene:
    """A 6x6 scene with the object at ``origin`` and the robots in the lower-left."""

    grid = [["."] * 6 for _ in range(6)]
    for dx in (0, 1):
        for dy in (0, 1):
            grid[origin[1] + dy][origin[0] + dx] = "O"
    for (x, y) in target:
        grid[y][x] = "T"
    grid[2][0] = "A"
    grid[3][0] = "B"
    rows = ["".join(grid[y]) for y in range(5, -1, -1)]
    return scene_from_ascii(
        rows, name=f"calibration_{origin[0]}_{origin[1]}", hidden=PHYSICS_CALIBRATION
    )


@dataclass
class CalibrationRecord:
    true_origin: Cell
    radius: int
    noise: float
    seed: int
    belief_origins: int
    p_correct_origin: float
    p_correct_physics: float
    ceiling_cost: float
    non_privileged_cost: float
    gap: float
    success: float


def run_calibration(
    *,
    radii: Sequence[int] = (2, 3, 6),
    origins: Sequence[Cell] = ((2, 1), (2, 2)),
    seeds: Sequence[int] = (0,),
    noise: float = 0.0,
    depth: int = 3,
    horizon: int = 5,
) -> Tuple[List[CalibrationRecord], Dict[str, object]]:
    started = time.perf_counter()
    hypothesis = (PHYSICS.index(PHYSICS_CALIBRATION), False)
    records: List[CalibrationRecord] = []

    for origin in origins:
        scene = build_calibration_scene(origin)
        ctx = ContactContext(scene.static, scene.obj.origin, scene.robots)
        ceiling = evaluate_planner(
            planner_spec("full", depth=depth), ctx, hypothesis,
            horizon=horizon, public_candidates=True,
            candidate_ids=CALIBRATION_CANDIDATES,
        )
        for radius in radii:
            for seed in seeds:
                sensor_noise = SensorNoise(drop_prob=noise)
                prior = uniform_prior(
                    scene.static, robot_poses=tuple(r.pose for r in scene.robots)
                )
                scan = apply_noise(
                    ray_cast(scene, 0, radius=radius), sensor_noise, __import__("random").Random(seed)
                )
                posterior = update_with_scan(prior, scan, scene.static, sensor_noise)
                belief = belief_from_posterior(posterior, scene.robots)
                result = evaluate_planner(
                    planner_spec("public", 0, depth=depth), ctx, hypothesis,
                    horizon=horizon, initial_belief=belief,
                    public_candidates=True, candidate_ids=CALIBRATION_CANDIDATES,
                )
                loc = localization_metrics(posterior, scene.obj.origin)
                phy = physics_metrics(posterior, scene.physics)
                records.append(
                    CalibrationRecord(
                        true_origin=scene.obj.origin,
                        radius=radius,
                        noise=noise,
                        seed=seed,
                        belief_origins=len({h.origin for h in posterior}),
                        p_correct_origin=loc["p_correct_origin"],
                        p_correct_physics=phy["p_correct_physics"],
                        ceiling_cost=ceiling.expected_cost,
                        non_privileged_cost=result.expected_cost,
                        gap=result.expected_cost - ceiling.expected_cost,
                        success=result.success_prob,
                    )
                )

    summary: Dict[str, object] = {
        "package_version": __version__,
        "git_revision": _git_revision(),
        "radii": list(radii),
        "origins": [list(o) for o in origins],
        "depth": depth,
        "horizon": horizon,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    return records, summary


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=False
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_calibration_markdown(records: Sequence[CalibrationRecord]) -> str:
    lines: List[str] = []
    lines.append("| radius | belief placements | p(correct origin) | p(correct physics) | ceiling | non-privileged | gap |")
    lines.append("|---|---|---|---|---|---|---|")
    for record in records:
        lines.append(
            "| {r} | {b} | {po:.2f} | {pp:.3f} | {c:.2f} | {d:.2f} | {g:.2f} |".format(
                r=record.radius,
                b=record.belief_origins,
                po=record.p_correct_origin,
                pp=record.p_correct_physics,
                c=record.ceiling_cost,
                d=record.non_privileged_cost,
                g=record.gap,
            )
        )
    if records:
        mean_gap = sum(r.gap for r in records) / len(records)
        lines.append("")
        lines.append(f"Mean gap over {len(records)} runs: **{mean_gap:.2f}**.")
    return "\n".join(lines)


def write_calibration_report(
    records: Sequence[CalibrationRecord], summary: Dict[str, object], output_dir: str
) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}

    csv_path = out / "calibration_records.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    paths["csv"] = str(csv_path)

    json_path = out / "calibration.json"
    json_path.write_text(
        json.dumps({"summary": summary, "records": [asdict(r) for r in records]}, indent=2)
    )
    paths["json"] = str(json_path)

    md_path = out / "calibration.md"
    md_path.write_text(
        "# Calibration: true-state ceiling vs a non-privileged estimate\n\n"
        f"- git revision: `{summary['git_revision']}`\n"
        f"- depth {summary['depth']}, horizon {summary['horizon']}\n\n"
        + format_calibration_markdown(records)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
