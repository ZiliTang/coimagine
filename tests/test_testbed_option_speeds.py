"""Current speed commands reach controllers, including legacy preset aliases."""
import json
import math
from dataclasses import replace

import pytest

from experiments.option_rl_v13 import controller as old
from experiments.option_rl_v19 import spec as S, testbed_server as server
from experiments.option_rl_v19.session import TestbedSession
from experiments.option_rl_v19.unknown_shape import ContactCandidate, make_unknown_skill
from experiments.option_rl_v19.synthetic_shapes import polygon_observation
from coimagine.random_start.lidar import SurfacePatch
from coimagine.random_start.skills import RobotKinematics


def test_reposition_legacy_ids_resolve_to_one_selectable_speed_per_direction():
    menu = server._option_list()
    repos = [s for s in menu if s['family'] == 'REPOSITION']
    assert len(menu) == 43 and {s['action_index'] for s in repos} == {40, 43}
    assert [s['action_index'] for s in menu if s['family'] == 'YIELD'] == [46]
    assert len(repos) == 2
    obs = TestbedSession(server.SCENES['contact_0']).env.boundary_observations()[0]
    mask = S.mask(obs)
    assert {i for i in range(38, 44) if mask[i]} == {40, 43}
    for index in range(38, 44):
        skill = make_unknown_skill(old.ACTIONS[index])
        assert skill.action.index == (40 if index < 41 else 43)
        assert skill.action.speed == .30
        # Exercise the real factory's tangent scaling wrapper, not just labels.
        skill.stage = 'move'; skill.tangent = (1., 0.)
        reference = skill.reference.update((.30, 0.), 1.)
        assert reference == pytest.approx((.30, 0.))


def test_push_speed_updates_reach_live_controller_without_reinitializing_q():
    inst = server.SCENES['contact_0']
    env = TestbedSession(inst).env
    slot = env.slots[0]
    # Factory receives the current codebook even when caller holds an old action.
    slot.action = S.ACTIONS[0]; slot.skill = make_unknown_skill(old.ACTIONS[0])
    skill = slot.skill
    for index, speed in enumerate((.10, .20, .30, .40)):
        slot.select(S.ACTIONS[index])
        assert slot.skill is skill and slot.pending is None
        assert skill.speed_command == speed and skill.action.speed == speed
    assert env.actions == S.ACTIONS
    assert env.action_codebook_hash == S.CODEBOOK_HASH
    assert old.ACTIONS[0].speed == .04  # frozen experiments retain their table


def test_legacy_preset_loading_maps_commands_without_rewriting_history(tmp_path, monkeypatch):
    monkeypatch.setattr(server, 'PRESETS', tmp_path)
    record = dict(id='legacy', scene='contact_0', created=1.,
                  sequence=[dict(a0=42, a1=44, decisions=10)])
    path = tmp_path/'legacy.json'; path.write_text(json.dumps(record))
    original = path.read_bytes()
    loaded = server._preset_get('legacy')
    assert loaded['sequence'][0]['a0'] == 43
    assert loaded['sequence'][0]['a1'] == 44
    assert path.read_bytes() == original
    assert S.canonical_sequence(loaded['sequence']) == loaded['sequence']
    with pytest.raises(ValueError):
        S.canonical_index(46)


@pytest.mark.parametrize('index,speed', tuple(enumerate((.10, .20, .30, .40))))
@pytest.mark.parametrize('tangent_error', (0., .05))
def test_real_contact_candidate_branch_obeys_speed_with_lateral_error(index, speed, tangent_error):
    points = tuple((.18, -.3 + i*.1) for i in range(7))
    patch = SurfacePatch(points_robot=points, points_world=points, distance=.18,
                         centroid_distance=.18, nearest_distance=.18,
                         normal_world=(1., 0.), tangent_world=(0., 1.),
                         confidence=1., arc_rays=7, span_m=.6, rms_m=0., valid=True)
    obs = replace(polygon_observation('triangle', 0, (0., 0.), (3., 3.)),
                  own=RobotKinematics(0., 0., 0., 0., 0., 0.),
                  patch=patch, patches=(patch,), contacting_object=True)
    skill = make_unknown_skill(S.ACTIONS[index])
    skill.candidate = ContactCandidate(1, (.18, tangent_error), (1., 0.), (0., 1.),
                                       0., 0., (-.3, .3), .02)
    _, info = skill.step(obs, .05)
    assert info['stage'] == 'push'
    vx, vy = info['desired_velocity']
    assert math.hypot(vx, vy) == pytest.approx(speed)
    if tangent_error == 0.:
        assert vx == pytest.approx(speed) and vy == pytest.approx(0.)
    else:
        assert vx > 0 and vy > 0
