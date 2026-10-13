"""The single frozen payoff used by every phase-C entry point.

CONTRACT section 3.3 defines the scalar payoff as

    completion  = 1 if the object satisfies the existing success rule else 0
    progress    = min(1, (d_start - d_end) / max(d_start, 1e-6))
    time_cost   = total wall-clock seconds of the skill sequence
    action_cost = sum(|F| * dt) / (F_max * T_total)
    Q           = completion + 0.5 * progress - 0.05 * time_cost - 0.10 * action_cost

``progress`` is **clamped only from above**. An action that pushes the object further from the goal
earns negative progress; that is the contract, and it is what makes the task signal able to
distinguish actions that make things worse.

The first phase-C run additionally clamped ``progress`` from below at 0. That clamp was never in the
contract, it made every action's task term identically zero in the mechanism scene, and it is what
produced the apparent "the two worlds prefer different actions" difference. It is kept here only as
``progress_clamped_historically`` so the recomputation can quantify the mistake; **no entry point
may use it for a reported result**.
"""

from __future__ import annotations

from typing import Dict

#: names of the two progress terms, deliberately distinct so a table cannot conflate them
PROGRESS_TERM = "progress"
PROGRESS_WEIGHTED_TERM = "0.5 * progress"


def progress_contract(start_distance: float, end_distance: float) -> float:
    """Contract 3.3 verbatim: upper clamp at 1, **no** lower clamp."""

    return min(1.0, (start_distance - end_distance) / max(start_distance, 1e-6))


def progress_clamped_historically(start_distance: float, end_distance: float) -> float:
    """The unauthorised lower clamp used by the first run. Diagnostic only."""

    return min(1.0, max(0.0, (start_distance - end_distance) / max(start_distance, 1e-6)))


def task_outcome(completion: float, progress: float) -> float:
    """The task part of the payoff, without time or action cost."""

    return completion + 0.5 * progress


def payoff(completion: float, progress: float, time_cost: float, action_cost: float) -> float:
    return completion + 0.5 * progress - 0.05 * time_cost - 0.10 * action_cost


def score_run(run: Dict[str, object], *, clamp_progress: bool = False) -> Dict[str, float]:
    """Score one saved run dict (``start_distance``, ``end_distance``, ``duration``, ...)."""

    progress = (progress_clamped_historically if clamp_progress else progress_contract)(
        float(run["start_distance"]), float(run["end_distance"]))
    completion = 1.0 if run.get("success") else 0.0
    q = payoff(completion, progress, float(run["duration"]), float(run["action_cost"]))
    return {"completion": completion, PROGRESS_TERM: progress,
            PROGRESS_WEIGHTED_TERM: 0.5 * progress, "q": q,
            "task_outcome": task_outcome(completion, progress)}


def analyse(worlds: Dict[str, list], delta: float, *, clamp_progress: bool = False) -> dict:
    """Payoff matrix, near-optimal sets, regret and the 50/50 information gap."""

    names = [run["command"] for run in worlds["A"]]
    q: Dict[str, Dict[str, float]] = {"A": {}, "B": {}}
    task: Dict[str, Dict[str, float]] = {"A": {}, "B": {}}
    for world in ("A", "B"):
        for run in worlds[world]:
            scored = score_run(run, clamp_progress=clamp_progress)
            q[world][run["command"]] = scored["q"]
            task[world][run["command"]] = scored["task_outcome"]
    best = {world: max(q[world].values()) for world in ("A", "B")}
    near = {world: sorted(name for name in names if best[world] - q[world][name] <= delta + 1e-12)
            for world in ("A", "B")}
    argmax = {world: max(names, key=lambda name: q[world][name]) for world in ("A", "B")}
    regret = max(best["A"] - q["A"][argmax["B"]], best["B"] - q["B"][argmax["A"]])
    all_knowing = 0.5 * (best["A"] + best["B"])
    common = max(0.5 * (q["A"][name] + q["B"][name]) for name in names)
    return {
        "q_A": q["A"], "q_B": q["B"],
        "task_outcome_A": task["A"], "task_outcome_B": task["B"],
        "near_optimal_A": near["A"], "near_optimal_B": near["B"],
        "near_optimal_disjoint": not (set(near["A"]) & set(near["B"])),
        "argmax_A": argmax["A"], "argmax_B": argmax["B"],
        "mutual_regret": regret, "delta": delta,
        "all_knowing_50_50": all_knowing, "common_action_50_50": common,
        "information_gap_50_50": all_knowing - common,
        "positive": bool(not (set(near["A"]) & set(near["B"])) and regret > delta),
        "any_positive_progress_A": any(value > 0 for value in task["A"].values()),
        "any_positive_progress_B": any(value > 0 for value in task["B"].values()),
        "best_task_outcome_A": max(task["A"].values()),
        "best_task_outcome_B": max(task["B"].values()),
        "worst_task_outcome_A": min(task["A"].values()),
        "worst_task_outcome_B": min(task["B"].values()),
        "needs_different_actions": bool(
            not (set(near["A"]) & set(near["B"])) and regret > delta),
        # "the same optimal action" does NOT mean "the same full ranking"
        "same_argmax": argmax["A"] == argmax["B"],
    }
