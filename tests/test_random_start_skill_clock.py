"""A newly started skill gets its own seek window at every episode time."""
from dataclasses import replace

import pytest

from coimagine.random_start.env import RandomStartEnv
from coimagine.random_start.layout import build_split
from coimagine.random_start.skills import make_skill
from coimagine.random_start.spec import CONTROL_DT, FIND_SECONDS, ROBOT_ACTIONS


def no_target_observation():
    instance = build_split("L0", "train", count=1)[0][0]
    obs = RandomStartEnv(instance).boundary_observations()[0]
    scan = replace(obs.scan, distances=(0.0,)*72, valid=(False,)*72,
                   category=(0,)*72, raw_labels=(None,)*72)
    return replace(obs, scan=scan, patch=None, patches=(), static_nearest=None,
                   teammate_nearest=None, contacting_object=False,
                   contacting_static=False, contacting_teammate=False)


@pytest.mark.parametrize("absolute_start", [0.0, 7.95, 8.0, 70.0])
def test_push_seek_duration_uses_skill_clock(absolute_start):
    obs = no_target_observation()
    skill = make_skill(ROBOT_ACTIONS[0])
    count = 0
    while not skill.done:
        now = absolute_start + count * CONTROL_DT
        current = replace(obs, time=now, elapsed_seconds=now,
                          tick=round(now / CONTROL_DT), remaining_seconds=90-now)
        skill.step(current, CONTROL_DT)
        count += 1
        assert count <= 40
    assert skill.exit_reason == "not_observed"
    assert count * CONTROL_DT == pytest.approx(FIND_SECONDS)
