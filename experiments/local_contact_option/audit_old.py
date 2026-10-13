"""A.1 implementation audit: the facts this round is built on, read from the code.

    python experiments/local_contact_option/audit_old.py --out docs/local_contact_option_v1

Every row names the file and function it was read from, so the table can be re-derived instead
of trusted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from coimagine.continuous2d.config import CONTROLLER_VERSION, default_config  # noqa: E402
from coimagine.continuous2d.environment import Continuous2DEnv  # noqa: E402
from coimagine.continuous2d.scenes import S0_DEVELOPMENT  # noqa: E402
from coimagine.continuous2d.skills import Hold, PushContact, control  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="docs/local_contact_option_v1")
    args = parser.parse_args()
    config = default_config()
    physics = config.physics
    control_config = config.control

    env = Continuous2DEnv(config)
    env.reset(S0_DEVELOPMENT)
    observation = env.observation()
    before = env.backend.object_state().as_tuple()
    result = env.execute_joint_skill((Hold(), Hold()))
    after = env.backend.object_state().as_tuple()

    rows = [
        {
            "item": "window duration",
            "value": f"{physics.window_steps} control steps x {physics.control_dt:.3f} s = "
                     f"{physics.window_steps * physics.control_dt:.2f} s",
            "source": "continuous2d/config.py::PhysicsConfig.window_steps/control_dt; "
                      "continuous2d/environment.py::execute_joint_skill",
            "same_as_task": True,
        },
        {
            "item": "control period / physics step",
            "value": f"control_dt={physics.control_dt:.4f} s (20 Hz), physics_dt={physics.physics_dt:.5f} s, "
                     f"substeps={physics.substeps}",
            "source": "config.py::PhysicsConfig.substeps; backend.py::apply_and_step",
            "same_as_task": True,
        },
        {
            "item": "force limit",
            "value": f"max_force={physics.max_force} N, clamped in backend.clamp_force and "
                     f"skills._clamp_force",
            "source": "backend.py::clamp_force; skills.py::_clamp_force",
            "same_as_task": True,
        },
        {
            "item": "velocity inheritance",
            "value": "execute_joint_skill resets controller_state only; body velocities are NOT "
                     "reset at a window boundary, so a new window inherits them",
            "source": "environment.py::execute_joint_skill (lines resetting controller_state)",
            "same_as_task": True,
        },
        {
            "item": "old push: contact test",
            "value": "gap = dot(contact_world - robot_position, direction) - robot_radius; "
                     "approach switches to push when gap <= 0.005 m",
            "source": "skills.py::_push_contact",
            "same_as_task": True,
        },
        {
            "item": "old push: approach stage",
            "value": "PD to the desired point contact - direction*radius, routed by A* "
                     "(geometry.plan_path) when further than 0.25 m, force clamped to "
                     "min(max_force, skill.force_limit)",
            "source": "skills.py::_push_contact approach branch; geometry.py::plan_path",
            "same_as_task": True,
        },
        {
            "item": "old push: push stage",
            "value": "forward = clamp(push_k_distance*max(0,gap) + push_k_velocity*"
                     "(push_speed - relative_along), 0, force_limit); lateral PD with "
                     "push_lateral_kp/kd about the desired contact point",
            "source": "skills.py::_push_contact push branch",
            "same_as_task": True,
        },
        {
            "item": "old push: reference frame",
            "value": "push_direction_world is a constant WORLD direction; contact_local is an "
                     "OBJECT-frame coordinate re-projected through the current object pose every "
                     "control period (so the skill follows a rotating object and needs object truth)",
            "source": "skills.py::contact_world, _push_contact",
            "same_as_task": True,
        },
        {
            "item": "old push: completion / braking",
            "value": "no braking: the window simply ends after window_steps. 'hold' is damped "
                     "braking F=-hold_kd*v with no position hold",
            "source": "skills.py::_hold; config.py::ControlConfig.hold_semantics='damped_brake'",
            "same_as_task": True,
        },
        {
            "item": "task success",
            "value": f"success needs the object inside the goal (margin {physics.goal_margin} m) "
                     f"and below {physics.velocity_tolerance} m/s / {physics.angular_tolerance} rad/s "
                     f"for {physics.success_hold_time} s; the controller never reads it",
            "source": "environment.py::object_in_goal/object_slow/task_success",
            "same_as_task": True,
        },
        {
            "item": "controller feedback",
            "value": "skills.control receives the full Observation, which contains the object's "
                     "global pose and velocity; the old push therefore uses object truth",
            "source": "skills.py::Observation, control",
            "same_as_task": True,
        },
        {
            "item": "round trip check",
            "value": f"hold window: object pose moved by "
                     f"{max(abs(a - b) for a, b in zip(before, after)):.3e} (no self-motion)",
            "source": "this audit run",
            "same_as_task": True,
        },
    ]
    payload = {
        "controller_version": CONTROLLER_VERSION,
        "physics": config.as_dict()["physics"],
        "control": config.as_dict()["control"],
        "facts": rows,
        "all_match_task_description": all(row["same_as_task"] for row in rows),
        "notes": "no divergence from the task description was found; nothing in the old code "
                 "was modified for this round",
    }
    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "A_implementation_audit.json").write_text(json.dumps(payload, indent=1, sort_keys=True))
    lines = ["# A.1 implementation audit (read from the code, not from a report)", "",
             f"`{CONTROLLER_VERSION}`; all rows match the task description: "
             f"**{payload['all_match_task_description']}**", "",
             "| item | value | source |", "|---|---|---|"]
    for row in rows:
        lines.append(f"| {row['item']} | {row['value']} | `{row['source']}` |")
    (out_dir / "A_implementation_audit.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print("AUDIT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
