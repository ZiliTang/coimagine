"""REPOSITION crosses observed corners and measures signed boundary-path travel."""
from dataclasses import replace
import math

import pytest

from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import env as E, layout, spec
from coimagine.random_start.skills import RepositionSkill


def corner_instance(direction="CCW"):
    rx, mx = (2.45, 1.9) if direction == "CCW" else (1.55, 2.1)
    return layout.InstanceSpec(
        name=f"reposition_corner_{direction}", map="L0", split="dev", index=0,
        seed=0, arena=(4.0, 3.0), obstacles=(), object_pose=Pose(2.0, 1.5, 0.0),
        robot_poses=(Pose(rx, 1.075, -math.pi / 2), Pose(mx, 1.075, -math.pi / 2)),
        goal_centre=(2.0, 2.65), goal_radius=spec.GOAL_RADIUS,
        env_version=spec.ENV_VERSION)


@pytest.fixture(scope="module")
def local_observation():
    return E.RandomStartEnv(corner_instance()).boundary_observations()[0]


def observation_at(base, x, y, normal=(0.0, 1.0), *, visible=True, contact=False):
    patch = replace(base.patch, normal_world=normal,
                    tangent_world=(-normal[1], normal[0]),
                    nearest_distance=spec.ROBOT_RADIUS + spec.REPOSITION_GAP)
    return replace(base, own=replace(base.own, x=x, y=y, vx=0.0, vy=0.0),
                   patch=patch if visible else None, contacting_object=contact,
                   static_nearest=None, teammate_nearest=None)


def moving_skill():
    skill = RepositionSkill(spec.ROBOT_ACTIONS[15])  # CCW, 1 m
    skill.stage = "move"
    return skill


def test_corner_continues_and_progress_uses_preceding_tangent(local_observation):
    skill = moving_skill()
    skill.step(observation_at(local_observation, 0.0, 0.0), spec.CONTROL_DT)
    # Travel right under the old command; the new face commands upward motion next.
    _, corner = skill.step(observation_at(local_observation, 0.1, 0.0, (-1.0, 0.0)),
                           spec.CONTROL_DT)
    assert corner["corner_event"]
    assert not skill.done
    assert skill.signed_progress == pytest.approx(0.1)
    assert skill.tangent == pytest.approx((0.0, 1.0))
    skill.step(observation_at(local_observation, 0.1, 0.1, (-1.0, 0.0)), spec.CONTROL_DT)
    assert skill.signed_progress == pytest.approx(0.2)
    # The declaration remains the initial heading, but boundary travel can turn.
    assert skill.declared_direction == pytest.approx((1.0, 0.0))
    skill.step(observation_at(local_observation, 0.1, 0.05, (-1.0, 0.0)), spec.CONTROL_DT)
    assert skill.signed_progress == pytest.approx(0.15)
    assert skill.positive_travel == pytest.approx(0.2)
    assert skill.backward_travel == pytest.approx(0.05)
    assert skill.progress == pytest.approx(skill.signed_progress)


@pytest.mark.parametrize("interruption", ["missing", "contact"])
def test_interrupted_motion_not_backfilled_on_reacquisition(local_observation, interruption):
    skill = moving_skill()
    skill.step(observation_at(local_observation, 0.0, 0.0), spec.CONTROL_DT)
    skill.step(observation_at(local_observation, 0.1, 0.0,
                              visible=interruption != "missing",
                              contact=interruption == "contact"), spec.CONTROL_DT)
    skill.step(observation_at(local_observation, 0.2, 0.0), spec.CONTROL_DT)
    assert skill.signed_progress == 0.0
    skill.step(observation_at(local_observation, 0.3, 0.0), spec.CONTROL_DT)
    assert skill.signed_progress == pytest.approx(0.1)


def test_corner_keeps_safety_checks(local_observation):
    skill = moving_skill()
    skill.step(observation_at(local_observation, 0.0, 0.0), spec.CONTROL_DT)
    corner = observation_at(local_observation, 0.1, 0.0, (-1.0, 0.0))
    corner = replace(corner, static_nearest=(spec.ROBOT_RADIUS + 0.01, (1.0, 0.0)))
    skill.step(corner, spec.CONTROL_DT)
    assert skill.exit_reason == "safety_abort"


@pytest.mark.parametrize("guard", ["patch_lost", "blocked_contact", "option_budget"])
def test_corner_continuation_retains_other_exit_guards(local_observation, guard):
    skill = moving_skill()
    skill.step(observation_at(local_observation, 0.0, 0.0), spec.CONTROL_DT)
    if guard == "option_budget":
        skill.skill_time = skill.time_budget - spec.CONTROL_DT / 2
    for _ in range(100):
        obs = observation_at(local_observation, 0.0, 0.0, (-1.0, 0.0),
                             visible=guard != "patch_lost", contact=guard == "blocked_contact")
        skill.step(obs, spec.CONTROL_DT)
        if skill.done:
            break
    assert skill.exit_reason == guard


@pytest.mark.parametrize("direction,offset", [("CCW", 12), ("CW", 16)])
@pytest.mark.parametrize("scale_index", [1, 2, 3])
def test_real_corner_reaches_requested_path_distance(direction, offset, scale_index):
    action = spec.ROBOT_ACTIONS[offset + scale_index]
    world = E.RandomStartEnv(corner_instance(direction))
    boundary = world.joint_step(action.index, 20)
    assert boundary["skill_ends"][0]["exit_reason"] == "reposition_complete"
    active = [row["r0"] for row in world.rows if row["r0"].get("signed_progress") is not None]
    assert active[-1]["signed_progress"] >= action.distance
    assert any(row["corner_event"] and not row["done"] for row in active)
    assert max(row["penetration"] for row in world.rows) < 1e-3
    assert world.total_periods == world.world_advances
