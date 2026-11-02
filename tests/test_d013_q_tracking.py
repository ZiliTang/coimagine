"""Finite visible-segment tracking and the actual allocator/skill integration."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from experiments.option_rl_v13 import controller as C
from experiments.option_rl_v19 import unknown_shape as U
from experiments.option_rl_v19.synthetic_shapes import edge_patch, polygon_observation


def observation(patches):
    obs = polygon_observation('octagon', 0, (1.7, 2.0), (1.7, 2.8))
    return replace(obs, patches=tuple(patches), patch=patches[0] if patches else None)


def edges():
    low = edge_patch((2.0, 2.0), (2.1464466094, 1.6464466094), np.array((1.7, 2.0)))
    high = edge_patch((2.0, 2.0), (2.1464466094, 2.3535533906), np.array((1.7, 2.0)))
    return low, high


def environment():
    sk = U.UnknownShapePush(C.ACTIONS[2])
    slots = [SimpleNamespace(skill=sk, action=C.ACTIONS[2]),
             SimpleNamespace(skill=None, action=C.ACTIONS[44])]
    env = SimpleNamespace(slots=slots, overrides={}, _coordinate=lambda *args: None)
    U.UnknownShapeEnvMixin.install_prior_free(env)
    return env, sk


def test_committed_selector_not_called_again_when_adjacent_faces_reorder(monkeypatch):
    low, high = edges()
    env, sk = environment()
    calls = []
    original = U.select_simple_point

    def select(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(U, 'select_simple_point', select)
    for k in range(8):
        obs = observation([low, high] if k % 2 == 0 else [high, low])
        env._coordinate(k * .05, [obs, replace(obs, robot_id=1)])
        if k == 0:
            q0 = np.asarray(sk.allocated_q)
            normal0 = sk.allocated_face.normal_world
        assert np.linalg.norm(np.asarray(sk.allocated_q)-q0) < 1e-8
        assert sk.allocated_face.normal_world == normal0
    assert len(calls) == 1


def test_support_visibility_changes_do_not_recenter_material_q():
    low, _ = edges()
    obs = observation([low])
    f = U.visible_faces(obs)[0]
    q = U._face_world(f, .5*(f[4]+f[5]))
    anchor = U._face_anchor(f, float(q @ f[2]))
    shorter = replace(low, points_world=low.points_world[:4], span_m=low.span_m*.75)
    tracked, face, _, debug = U.track_contact_q(observation([shorter]), q, anchor)
    assert face is not None and debug['matched']
    np.testing.assert_allclose(tracked, q, atol=1e-9)


def test_adjacent_or_disjoint_segment_cannot_replace_committed_target():
    low, high = edges()
    obs = observation([low]); f = U.visible_faces(obs)[0]
    q = U._face_world(f, .5*(f[4]+f[5])); anchor = U._face_anchor(f, float(q @ f[2]))
    disjoint = replace(low, points_world=tuple(tuple(np.asarray(p)+np.asarray(f[2])*1.0)
                                              for p in low.points_world))
    for other in [high, disjoint]:
        tracked, face, _, debug = U.track_contact_q(observation([other]), q, anchor)
        assert face is None and not debug['matched']
        np.testing.assert_allclose(tracked, q)


def test_disjoint_collinear_fragments_are_not_one_visible_segment():
    p = edge_patch((2., 1.), (2., 1.3), np.array((1.7, 1.5)))
    p2 = edge_patch((2., 1.8), (2., 2.1), np.array((1.7, 1.5)))
    assert len(U.visible_faces(observation([p, p2]))) == 2
    assert all(abs(f[5]-f[4]) < .31 for f in U.visible_faces(observation([p,p2])))


def test_initial_q_uses_own_support_even_when_peer_sees_more():
    own_patch = edge_patch((2.,1.),(2.,1.4),np.array((1.5,1.2)))
    peer_patch = edge_patch((2.,1.2),(2.,2.),np.array((1.5,1.8)))
    own = observation([own_patch])
    peer = replace(observation([peer_patch]),robot_id=1)
    env,sk = environment()
    env._coordinate(0.,[own,peer])
    np.testing.assert_allclose(sk.allocated_q,(2.,1.2),atol=1e-12)
    assert not sk.allocated_wait
    assert not hasattr(env,'support_memory')


def test_reposition_observations_do_not_expand_later_push_support():
    wide = edge_patch((2.,1.),(2.,2.),np.array((1.5,1.2)))
    narrow = edge_patch((2.,1.),(2.,1.4),np.array((1.5,1.2)))
    env,sk = environment()
    env.slots[0].skill=None;env.slots[0].action=C.ACTIONS[42]
    observed=observation([wide]);peer=replace(observed,robot_id=1)
    env._coordinate(0.,[observed,peer])
    env.slots[0].skill=sk;env.slots[0].action=C.ACTIONS[2]
    current=observation([narrow])
    env._coordinate(.05,[current,replace(current,robot_id=1)])
    np.testing.assert_allclose(sk.allocated_q,(2.,1.2),atol=1e-12)


def test_paired_allocation_keeps_each_q_inside_its_owners_current_support():
    own_patch = edge_patch((2.,1.),(2.,1.4),np.array((1.5,1.2)))
    peer_patch = edge_patch((2.,1.6),(2.,2.),np.array((1.5,1.8)))
    env,sk = environment()
    env.slots[1].skill=U.UnknownShapePush(C.ACTIONS[2])
    env.slots[1].action=C.ACTIONS[2]
    env._coordinate(0.,[observation([own_patch]),replace(observation([peer_patch]),robot_id=1)])
    for slot,(lo,hi) in zip(env.slots,[(1.,1.4),(1.6,2.)]):
        assert slot.skill.allocated_q is not None and not slot.skill.allocated_wait
        x,y=slot.skill.allocated_q
        assert abs(x-2.)<1e-12 and lo-1e-12<=y<=hi+1e-12


def test_tracking_advects_target_and_corrects_normal_without_recentering():
    p = edge_patch((2., 1.), (2., 2.), np.array((1.7, 1.5)))
    f = U.visible_faces(observation([p]))[0]
    q = np.array((2., 1.4)); anchor = U._face_anchor(f, float(q @ f[2]))
    moved = replace(p, points_world=tuple((x+.01, y+.02) for x, y in p.points_world))
    tracked, face, _, debug = U.track_contact_q(observation([moved]), q, anchor,
                                               velocity=(.2, .4), dt=.05)
    assert face is not None and debug['matched']
    np.testing.assert_allclose(tracked, (2.01, 1.42), atol=1e-9)


def test_missing_segment_waits_then_reacquires_same_identity():
    low, high = edges(); env, sk = environment()
    obs = observation([low]); other = replace(obs, robot_id=1)
    env._coordinate(0., [obs, other]); identity = sk.allocated_id; q = sk.allocated_q
    env._coordinate(.05, [observation([high]), other])
    assert sk.allocated_wait and sk.allocated_face is None
    assert sk.allocated_q == q
    env._coordinate(.1, [obs, other])
    assert not sk.allocated_wait and sk.allocated_id == identity


def test_new_skill_of_same_action_gets_new_commit(monkeypatch):
    low, _ = edges(); env, sk = environment(); obs = observation([low])
    env._coordinate(0., [obs, replace(obs, robot_id=1)]); old_id = sk.allocated_id
    env.slots[0].skill = U.UnknownShapePush(C.ACTIONS[2])
    env._coordinate(.05, [obs, replace(obs, robot_id=1)])
    assert env.slots[0].skill.allocated_id != old_id


def test_actual_skill_step_uses_target_segment_not_nearest_other_patch():
    low, high = edges(); env, sk = environment()
    low = replace(low, nearest_distance=.6)
    high = replace(high, nearest_distance=.19)
    obs = observation([low, high]); env._coordinate(0., [obs, replace(obs, robot_id=1)])
    chosen = sk.allocated_face
    # Make a different patch closer after allocation. Actual step must keep
    # q and its normal together rather than independently choosing that patch.
    decoy = high if chosen is low else low
    decoy = replace(decoy, nearest_distance=.18)
    obs = replace(obs, patches=(decoy, chosen), patch=decoy)
    _, info = sk.step(obs, .05)
    np.testing.assert_allclose(info['normal_world'], chosen.normal_world)
    np.testing.assert_allclose(info['candidate']['n'], chosen.normal_world)
    assert info['candidate']['id'] == 1
