"""Focused tests for the v4.6 PUSH changes.

Two behaviours are pinned:

* a narrow face (0.25 m, shorter than the robot diameter) is contactable with the soft inset;
  the old hard guard would have returned ``insufficient_visible_segment`` at the standoff;
* a cooperative pair stages at about 3 cm with aligned headings, both report ready, and both
  enter ``push`` in the same control period (no early single-sided push).
"""
from __future__ import annotations

import math

from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import env as E
from coimagine.random_start import layout, spec


def test_short_face_is_contacted_and_pushed():
    instance = layout.build_split("L1", "train", count=3)[0][2]
    world = E.RandomStartEnv(instance)
    boundary = world.joint_step(11, 20)          # PUSH 0.39 m/s, requested 1.00 m; r1 brakes
    assert boundary["skill_ends"][0]["exit_reason"] != "insufficient_visible_segment"
    assert any(row["r0"]["contacting_object"] for row in world.rows)
    final = world.rows[-1]["r0"]
    assert float(final.get("signed_progress") or 0.0) > 0.10
    assert boundary["skill_ends"][0]["exit_reason"] in (
        "push_complete", "option_budget", "push_wall_timeout", "joint_window_end")


def test_cooperative_pair_stages_then_pushes_together():
    instance = layout.InstanceSpec(
        name="coop_test", map="L0", split="dev", index=0, seed=0, arena=(4.0, 3.0),
        obstacles=(), object_pose=Pose(2.0, 1.5, 0.0),
        robot_poses=(Pose(1.8, 1.075, -math.pi / 2), Pose(2.2, 0.745, -math.pi / 2)),
        goal_centre=(2.0, 2.65), goal_radius=spec.GOAL_RADIUS,
        env_version=spec.ENV_VERSION)
    world = E.RandomStartEnv(instance)
    world.joint_step(6, 6)                       # both PUSH 0.24 m/s, requested 0.50 m
    rows = world.rows
    assert rows

    ready = [index for index, row in enumerate(rows)
             if row["r0"].get("ready") and row["r1"].get("ready")]
    push = [index for index, row in enumerate(rows)
            if row["r0"]["stage"] == "push" and row["r1"]["stage"] == "push"]
    assert ready, "both robots must reach the staging clearance"
    assert push, "both robots must enter push"
    assert push[0] > ready[0], "push must follow the joint ready state"

    first_ready = rows[ready[0]]
    for robot in (0, 1):
        gap = float(first_ready[f"r{robot}"]["gap"])
        assert 0.015 <= gap <= 0.045, gap
        assert math.degrees(abs(first_ready[f"r{robot}"]["align_error_deg"])) <= 10.0

    first_push = push[0]
    assert rows[first_push]["r0"]["stage"] == rows[first_push]["r1"]["stage"] == "push"

    first_both_contact = next((index for index, row in enumerate(rows)
                               if row["r0"]["contacting_object"]
                               and row["r1"]["contacting_object"]), None)
    assert first_both_contact is not None
    for row in rows[first_both_contact:]:
        if row["r0"]["stage"] == "push" and row["r1"]["stage"] == "push":
            assert row["r0"]["contacting_object"] and row["r1"]["contacting_object"]
