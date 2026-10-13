"""D.2 continuation test: 12 test roots, three consecutive skills each.

    python experiments/local_contact_option/run_continuation.py --workers 6

The velocity is inherited between skills and the controller's stage state is reset by the
protocol; nothing is snapped back to a canonical pose. A skill that does not reach a locally
settled terminal state is **not** forced to continue as if it were rest-to-rest: the chain stops
there and the failure is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.continuous2d.config import default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.local_contact_option.contract import OptionConfig  # noqa: E402
from coimagine.local_contact_option.metrics import execution_metrics, to_initial_frame  # noqa: E402
from coimagine.local_contact_option.option import OptionExecutor  # noqa: E402
from coimagine.local_contact_option.roots import VariantSpec, build_roots  # noqa: E402
from coimagine.local_contact_option.runner import (  # noqa: E402
    build_snapshot, root_metadata, root_from_metadata,
)
from coimagine.local_contact_option.skill import LocalSurfacePush  # noqa: E402

SELECTION_SALT = "local-contact-option-continuation-v1"
SKILLS_PER_ROOT = 3
ROOTS_SELECTED = 12


def select_roots(metadata, count: int = ROOTS_SELECTED):
    ordered = sorted(
        metadata,
        key=lambda row: hashlib.sha256(f"{SELECTION_SALT}|{row['root_id']}".encode()).hexdigest(),
    )
    return ordered[:count]


def run_chain(root, arm: str, option: OptionConfig, skills: int = SKILLS_PER_ROOT) -> dict:
    baseline = VariantSpec("baseline")
    snapshot, config, initial_object, initial_robots, _, velocities = build_snapshot(
        root, baseline, default_config()
    )
    from coimagine.local_contact_option.roots import apply_initial_velocities

    env = Continuous2DEnv(config)
    env.restore(snapshot, exact=True)
    apply_initial_velocities(env, velocities)
    start_object = env.backend.object_state().as_tuple()
    start_robots = [state.as_tuple() for state in env.backend.robot_states()]

    steps = []
    cumulative = 0.0
    broken_at = None
    entry_qualified = _entry_qualification(env, option)
    for index in range(skills):
        step_start_object = env.backend.object_state().as_tuple()
        step_start_robots = [state.as_tuple() for state in env.backend.robot_states()]
        specs = {}
        for robot_id in (0, 1):
            if arm in ("O", "P"):
                specs[robot_id] = root.oracle_spec(robot_id)
            else:
                parameters = root.local_parameters(robot_id)
                specs[robot_id] = LocalSurfacePush(**parameters) if parameters else None
        executor = OptionExecutor(option, arm)
        result = executor.run(
            env, root_id=f"{root.root_id}#{index}", variant="continuation",
            joint_specs=specs, active_robots=root.active_robots,
        )
        metrics = execution_metrics(
            result, initial_object=start_object, initial_robots=start_robots,
            force_limit=option.force_limit, physics=config.physics, option=option,
        )
        final_object = result.terminal_object or env.backend.object_state().as_tuple()
        step_relative = to_initial_frame(final_object, step_start_object)
        steps.append({
            "skill": index, "exit_reason": result.exit_reason,
            "step_displacement": math.hypot(step_relative[0], step_relative[1]),
            "step_rotation": abs(step_relative[2]),
            "push_completed": bool(result.push_completed),
            "settled_after_abort": bool(result.settled_after_abort),
            "wall_clock_seconds": result.wall_clock_steps * config.physics.control_dt,
            "settled_observed": result.exit_reason == "settled_observed",
            "final_speed": math.hypot(final_object[3], final_object[4]),
            "final_omega": abs(final_object[5]),
            "reached_effective_push": metrics.get("reached_effective_push"),
            "contact_period_ratio": metrics.get("contact_period_ratio"),
            "displacement_this_skill": metrics.get("displacement"),
            "initial_state_for_next_skill": bool(
                result.exit_reason == "settled_observed"
                and math.hypot(final_object[3], final_object[4]) < option.settle_linear
                and abs(final_object[5]) < option.settle_angular
            ),
            "next_entry_qualification": _entry_qualification(env, option),
        })
        cumulative += result.wall_clock_steps * config.physics.control_dt
        if result.exit_reason != "settled_observed":
            broken_at = index
            break
        if index + 1 < skills:
            # the protocol resets the stage state; the physical velocity is inherited untouched
            env.controller_state = [{}, {}]
    final_object = result.terminal_object or env.backend.object_state().as_tuple()
    terminal = to_initial_frame(final_object, start_object)
    normal_completion = bool(
        len(steps) == skills and all(step["settled_observed"] for step in steps)
    )
    return {
        "root_id": root.root_id, "split": root.split, "arm": arm,
        "skills_requested": skills, "skills_attempted": len(steps),
        "skills_completed": len(steps),
        "skills_push_completed": sum(1 for step in steps if step["push_completed"]),
        "normal_completion": normal_completion,
        "entry_qualified": entry_qualified,
        "false_stable_then_continued": [
            step["skill"] for step in steps[:-1]
            if step["settled_observed"] and not step.get("next_entry_qualification", {}).get("qualified", True)
        ],
        "broken_at": broken_at, "steps": steps,
        "cumulative_step_displacement": sum(step["step_displacement"] for step in steps),
        "cumulative_wall_clock_seconds": cumulative,
        "terminal_relative_object": list(terminal),
        "terminal_speed": math.hypot(final_object[3], final_object[4]),
        "terminal_omega": abs(final_object[5]),
        "start_object": list(start_object),
    }


def _entry_qualification(env, option: OptionConfig) -> dict:
    """Evaluator-side entry check: both robots and the object must actually be at rest.

    This is used to *report* whether a terminal state is a usable next initial state. It reads
    ground truth, so it can never feed the controller: the controller only ever sees the
    protocol's own observations.
    """

    object_state = env.backend.object_state()
    robots = env.backend.robot_states()
    object_speed = math.hypot(object_state.vx, object_state.vy)
    robot_speed = max(math.hypot(state.vx, state.vy) for state in robots)
    robot_omega = max(abs(state.omega) for state in robots)
    return {
        "object_speed": object_speed, "object_omega": abs(object_state.omega),
        "robot_speed": robot_speed, "robot_omega": robot_omega,
        "qualified": bool(object_speed < option.settle_linear
                          and abs(object_state.omega) < option.settle_angular
                          and robot_speed < option.settle_linear
                          and robot_omega < option.settle_angular),
    }


def _worker(job):
    root = root_from_metadata(job["root"])
    option = OptionConfig(**job["option_config"])
    return run_chain(root, job["arm"], option)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="local-position-push-v1")
    parser.add_argument("--out-root", default="outputs/local_position_push_v1")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out_dir = ROOT / args.out_root / args.run_id
    metadata = json.loads((out_dir / "roots_test.json").read_text())["roots"]
    frozen = json.loads((out_dir / "roots_test.json").read_text())["frozen_config"]
    option = OptionConfig(**frozen["baseline"])
    chosen = select_roots(metadata)

    jobs = [
        {"root": row, "arm": arm, "option_config": option.as_dict()}
        for row in chosen for arm in ("V", "X")
    ]
    import multiprocessing as mp

    context = mp.get_context("fork")
    with context.Pool(processes=args.workers) as pool:
        results = list(pool.imap_unordered(_worker, jobs, chunksize=1))

    by_arm: Dict[str, List[dict]] = defaultdict(list)
    for row in results:
        by_arm[row["arm"]].append(row)
    summary = {}
    for arm, rows in sorted(by_arm.items()):
        completed = [row for row in rows if row["skills_completed"] == SKILLS_PER_ROOT]
        normal = [row for row in rows if row["normal_completion"]]
        push_all = [row for row in rows if row["skills_push_completed"] == SKILLS_PER_ROOT]
        summary[arm] = {
            "roots": len(rows),
            "chains_attempting_3_skills": len(completed),
            "chains_with_3_push_completed": len(push_all),
            "chains_normal_completed": len(normal),
            "normal_completion_ratio": len(normal) / len(rows),
            "false_stable_then_continued_cases": sum(
                len(row["false_stable_then_continued"]) for row in rows),
            "entry_qualified_roots": sum(1 for row in rows if row["entry_qualified"]["qualified"]),
            "mean_skills_attempted": sum(row["skills_attempted"] for row in rows) / len(rows),
            "mean_cumulative_seconds": sum(row["cumulative_wall_clock_seconds"] for row in rows) / len(rows),
            "settled_terminal_ratio": sum(
                1 for row in rows if row["steps"] and row["steps"][-1]["settled_observed"]
            ) / len(rows),
            "terminal_speed_median": sorted(
                row["terminal_speed"] for row in rows)[len(rows) // 2],
            "usable_initial_state_ratio": (
                sum(1 for row in rows for step in row["steps"][:-1]
                    if step["initial_state_for_next_skill"])
                / max(1, sum(len(row["steps"]) - 1 for row in rows))
            ),
            "transitions_attempted": sum(len(row["steps"]) - 1 for row in rows),
        }
    payload = {
        "selection_salt": SELECTION_SALT, "roots": [row["root_id"] for row in chosen],
        "frozen_kv": option.kv, "skills_per_root": SKILLS_PER_ROOT,
        "summary": summary, "chains": results,
    }
    (out_dir / "continuation.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))
    print("CONTINUATION_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
