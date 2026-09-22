"""Does a message pay when the teammate is *necessary*?

Round 21's rule: message value shows up only where the teammate's contribution is
**required**, not merely cheaper. This module builds exactly that decision and measures the
value of one bit.

The construction, using only physics values the model already has:

* robot A is **slipping** -- A knows this about itself, and it means A can never move the
  object alone at any force, so "push solo" is not an option;
* robot B's traction is **private** to B;
* the object is light, so a **co-push works exactly when B grips**.

So the only way to succeed is to co-push, and whether that works is decided by a fact that
lives entirely in the teammate's private state. Without a message the team must hedge
between attempting a push that may be impossible and declining a task it could have won.
With one bit, it knows.

Costs are the ones the project already uses: a push at force ``f`` costs ``1 + f``; a failed
push costs an extra penalty; and declining a task that was winnable costs the same penalty,
so "do nothing" cannot look free.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .. import __version__
from ..grid import scene_from_ascii
from ..grid.continuous import FORCE_MIN, push_outcome
from ..grid.state import HiddenPhysics, Scene

#: A push at force f costs this plus f.
PUSH_BASE_COST = 1.0
#: Charged for a push that fails, and for declining a task that was winnable.
#:
#: It must clearly exceed the cost of a successful attempt, otherwise declining a winnable
#: task breaks even with winning it and the task objective disappears. An earlier value of
#: 2.0 was exactly the cost of a successful co-push, which made "do nothing" as good as
#: "succeed" and hid the value of the message. The project's planner uses a failure cost of
#: 20 against action costs of order 1, and this follows the same convention.
FAIL_PENALTY = 10.0
#: The angle the team pushes along (inside the east angular window).
ANGLE = 0.0

def necessity_scene(teammate_grip: bool, resistance: int = 1) -> Scene:
    """A slipping, both robots on the west face, goal to the east.

    The grid is indexed ``[y][x]`` and emitted top row first, so the object sits at
    world origin ``(2, 1)``, robot A at ``(1, 1)`` and B at ``(1, 2)``: both are on the
    object's west face, so a co-push is immediately executable.
    """

    grid = [["."] * 6 for _ in range(6)]
    for dx in (0, 1):
        for dy in (0, 1):
            grid[1 + dy][2 + dx] = "O"
    for y in range(6):
        grid[y][4] = "T"
        grid[y][5] = "T"
    grid[1][1] = "A"
    grid[2][1] = "B"
    rows = ["".join(grid[y]) for y in range(5, -1, -1)]
    return scene_from_ascii(
        rows,
        name="necessity",
        hidden=HiddenPhysics(
            resistance=resistance,
            constraint="free",
            traction=("slip", "grip" if teammate_grip else "slip"),
        ),
    )


@dataclass
class NecessityRecord:
    scene_id: int
    teammate_grips: int
    solvable: int
    cost: Dict[str, float]
    success: Dict[str, int]
    regret: Dict[str, float]


@dataclass
class PolicySummary:
    name: str
    n: int
    mean_regret: float
    success_rate: float


@dataclass
class NecessityResult:
    summaries: List[PolicySummary]
    by_teammate: Dict[str, List[PolicySummary]]
    message_value: float
    success_gain: float
    records: List[NecessityRecord] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


POLICIES: Tuple[str, ...] = ("oracle", "silent", "one_bit", "always_co", "always_decline")


def _play(scene: Scene, mode: str, force: float) -> Tuple[float, int]:
    """Cost and success of one choice in the simulator."""

    if mode == "decline":
        return 0.0, 0
    forces = {0: force} if mode == "solo" else {0: force, 1: force}
    outcome = push_outcome(scene, forces, ANGLE)
    if outcome.moved:
        return PUSH_BASE_COST + force, 1
    return PUSH_BASE_COST + force + FAIL_PENALTY, 0


def run_necessity_comparison(
    *, n_scenes: int = 200, seed: int = 0, resistance: int = 1
) -> NecessityResult:
    started = time.perf_counter()
    records: List[NecessityRecord] = []
    rng = np.random.default_rng(seed)

    for scene_id in range(n_scenes):
        teammate_grips = bool(rng.integers(0, 2))
        scene = necessity_scene(teammate_grips, resistance=resistance)

        # What is actually available? A slips, so solo is impossible at any force.
        co_threshold = 1.0 if teammate_grips else float("inf")
        solvable = int(np.isfinite(co_threshold))
        oracle_cost, oracle_success = _play(
            scene, "co", co_threshold if solvable else FORCE_MIN
        )
        if not solvable:
            oracle_cost, oracle_success = 0.0, 0

        # A silent team hedges. It knows its own traction (slipping) and the resistance, so
        # it knows the co-push needs the teammate; it puts probability 1/2 on the teammate
        # gripping, and declines if that looks better than risking a failed attempt.
        p_grip = 0.5
        co_force = 1.0
        co_expected = (
            p_grip * (PUSH_BASE_COST + co_force)
            + (1 - p_grip) * (PUSH_BASE_COST + co_force + FAIL_PENALTY)
        )
        decline_expected = p_grip * FAIL_PENALTY  # misses a winnable task half the time
        if co_expected <= decline_expected:
            silent_cost, silent_success = _play(scene, "co", co_force)
        else:
            silent_cost, silent_success = 0.0, 0

        # One bit: the teammate's traction is known exactly.
        if teammate_grips:
            onebit_cost, onebit_success = _play(scene, "co", co_force)
        else:
            onebit_cost, onebit_success = 0.0, 0

        always_co_cost, always_co_success = _play(scene, "co", co_force)
        always_decline_cost, always_decline_success = 0.0, 0

        def regret(cost: float, success: int) -> float:
            """Regret against the oracle, charging the penalty for missing a win."""

            if solvable and not success:
                return FAIL_PENALTY + (cost - oracle_cost)
            return cost - oracle_cost

        records.append(
            NecessityRecord(
                scene_id=scene_id,
                teammate_grips=int(teammate_grips),
                solvable=solvable,
                cost={
                    "oracle": oracle_cost,
                    "silent": silent_cost,
                    "one_bit": onebit_cost,
                    "always_co": always_co_cost,
                    "always_decline": always_decline_cost,
                },
                success={
                    "oracle": oracle_success,
                    "silent": silent_success,
                    "one_bit": onebit_success,
                    "always_co": always_co_success,
                    "always_decline": always_decline_success,
                },
                regret={
                    "oracle": 0.0,
                    "silent": regret(silent_cost, silent_success),
                    "one_bit": regret(onebit_cost, onebit_success),
                    "always_co": regret(always_co_cost, always_co_success),
                    "always_decline": regret(
                        always_decline_cost, always_decline_success
                    ),
                },
            )
        )

    def summarise(name: str, subset: Sequence[NecessityRecord]) -> List[PolicySummary]:
        out: List[PolicySummary] = []
        for policy in POLICIES:
            if not subset:
                out.append(PolicySummary(policy, 0, float("nan"), float("nan")))
                continue
            out.append(
                PolicySummary(
                    name=policy,
                    n=len(subset),
                    mean_regret=float(np.mean([r.regret[policy] for r in subset])),
                    success_rate=float(np.mean([r.success[policy] for r in subset])),
                )
            )
        return out

    groups = {
        "teammate_grips": [r for r in records if r.teammate_grips],
        "teammate_slips": [r for r in records if not r.teammate_grips],
    }
    overall = summarise("all", records)
    by_name = {s.name: s for s in overall}

    return NecessityResult(
        summaries=overall,
        by_teammate={k: summarise(k, v) for k, v in groups.items()},
        message_value=by_name["silent"].mean_regret - by_name["one_bit"].mean_regret,
        success_gain=by_name["one_bit"].success_rate - by_name["silent"].success_rate,
        records=records,
        meta={
            "package_version": __version__,
            "git_revision": _git_revision(),
            "n_scenes": n_scenes,
            "resistance": resistance,
            "push_base_cost": PUSH_BASE_COST,
            "fail_penalty": FAIL_PENALTY,
            "wall_clock_seconds": time.perf_counter() - started,
        },
    )


def _git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip() or "unknown"
    except OSError:
        return "unknown"


def format_necessity_markdown(result: NecessityResult) -> str:
    lines = [
        "The teammate is **necessary**: this robot slips, so only a co-push can move the "
        "object, and a co-push works exactly when the teammate grips.",
        "",
        f"Value of one bit: **{result.message_value:.3f}** regret; success gain "
        f"**{result.success_gain:+.3f}**.",
        "",
        "| group | policy | n | mean regret | success rate |",
        "|---|---|---|---|---|",
    ]
    for summary in result.summaries:
        lines.append(
            f"| all | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
            f"{summary.success_rate:.3f} |"
        )
    for group, summaries in result.by_teammate.items():
        for summary in summaries:
            lines.append(
                f"| {group} | {summary.name} | {summary.n} | {summary.mean_regret:.3f} | "
                f"{summary.success_rate:.3f} |"
            )
    return "\n".join(lines)


def write_necessity_report(result: NecessityResult, output_dir: str) -> Dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, str] = {}
    json_path = out / "necessity.json"
    json_path.write_text(
        json.dumps(
            {
                "meta": result.meta,
                "summaries": [asdict(s) for s in result.summaries],
                "by_teammate": {
                    k: [asdict(s) for s in v] for k, v in result.by_teammate.items()
                },
                "message_value": result.message_value,
                "success_gain": result.success_gain,
                "records": [asdict(r) for r in result.records],
            },
            indent=2,
        )
    )
    paths["json"] = str(json_path)
    md_path = out / "necessity.md"
    md_path.write_text(
        "# Message value when the teammate is necessary\n\n"
        f"- git revision: `{result.meta['git_revision']}`\n"
        f"- {result.meta['n_scenes']} scenes, resistance "
        f"{result.meta['resistance']}\n"
        f"- push costs {result.meta['push_base_cost']} + force; failure or a missed win "
        f"costs {result.meta['fail_penalty']}\n\n"
        + format_necessity_markdown(result)
        + "\n"
    )
    paths["md"] = str(md_path)
    return paths
