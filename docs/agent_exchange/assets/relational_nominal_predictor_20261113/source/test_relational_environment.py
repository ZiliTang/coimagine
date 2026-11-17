"""Partial-view Option integration; no policy-learning performance claims."""
from dataclasses import replace
import numpy as np
import pytest
from experiments.planar_tasks.tasks import task_catalog
from experiments.option_rl_v21.system import OptionSystemEnv
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv, MovableBox
from experiments.planar_tasks.relational_search.contract import public_observation,validate_action,action_space


def instance():
    return task_catalog()['planar_open_anchor'].instance


def command(target='entity-a',a=44,ticks=10):
    return dict(target_id=target,a0=a,a1=a,ticks=ticks)


def test_target_filtered_scan_and_patch_provenance():
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    public=env.public()
    assert {e['id'] for e in public['entities']}=={'entity-a','b'}
    for i in (0,1):
        raw,labels=env._rays(i,0,0.)
        for target in ('entity-a','b'):
            scan=env._target_scan(i,0,0.,target)
            assert scan.distances==raw.distances and scan.valid==raw.valid
            assert all((category==1)==(label==target and valid)
                       for category,label,valid in zip(scan.category,labels,scan.valid))
            sensor=env._sensor(i,0,0.,target)
            assert not sensor.forbidden_fields()
            hits=[(raw.origin[0]+d*np.cos(a),raw.origin[1]+d*np.sin(a))
                  for a,d,v,label in zip(raw.ray_angles,raw.distances,raw.valid,labels) if v and label==target]
            for patch in sensor.patches:
                for point in patch.points_world:
                    assert min(np.linalg.norm(np.asarray(point)-p) for p in hits)<1e-10
    assert public_observation(public)==public


def test_occluded_body_and_hidden_mass_do_not_change_public_input():
    light=MultiObjectOptionEnv(instance(),[MovableBox('hidden',2.8,2.,mass=3.)])
    heavy=MultiObjectOptionEnv(instance(),[MovableBox('hidden',2.8,2.,mass=100.)])
    assert light.public()==heavy.public()
    assert 'hidden' not in [e['id'] for e in light.public()['entities']]
    with pytest.raises(ValueError): validate_action(command('hidden'),light.public())


def test_target_specific_menu_survives_projection():
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    o=env.public()
    assert set(o['entity_legal_pairs'])=={e['id'] for e in o['entities']}
    for target,menu in o['entity_legal_pairs'].items():
        for pair in menu: validate_action(dict(target_id=target,a0=pair[0],a1=pair[1],ticks=20),o)
    unsupported=[p for p in o['legal_pairs'] if p not in o['entity_legal_pairs']['b']]
    assert unsupported
    with pytest.raises(ValueError):
        validate_action(dict(target_id='b',a0=unsupported[0][0],a1=unsupported[0][1],ticks=20),o)
    for n,a in enumerate(action_space(o)):
        validate_action(a,o)
        if n>=100:break


def test_single_object_forced_command_compatibility():
    old=OptionSystemEnv(instance()); old.decision_ticks=1
    new=MultiObjectOptionEnv(instance())
    for a,ticks in [(2,20),(2,20),(44,20)]:
        new.step_command(command(a=a,ticks=ticks))
        for _ in range(ticks): old.joint_step(a,a)
        np.testing.assert_array_equal(old.object_state().as_tuple(),new.object_state().as_tuple())
        for left,right in zip(old.robot_states(),new.robot_states()):
            np.testing.assert_array_equal(left.as_tuple(),right.as_tuple())
        assert [s.last_info for s in old.slots]==[s.last_info for s in new.slots]


def test_switch_rebuilds_estimator_and_allocations_without_teleport():
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    before=env.offline_audit()
    estimator=env.motion_estimator
    assert env.step_command(command('b'))['handoff']['phase']=='switch_brake'
    assert env.total_periods==10
    assert env.step_command(command('b'))['handoff']['commit']
    assert env.active_target=='b' and env.motion_estimator is not estimator
    assert env.step_command(command('entity-a'))['handoff']['phase']=='switch_brake'
    assert env.step_command(command('entity-a'))['handoff']['commit']
    assert env.active_target=='entity-a'
    # Zero-speed brake switching must leave both object poses unchanged.
    assert env.offline_audit()['entities']==before['entities']
    replay=env.replay_commands()
    assert replay.offline_audit()==env.offline_audit()


def test_repeated_push_preserves_controller_and_switch_uses_brake():
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    env.step_command(command(a=2,ticks=20))
    skills=[s.skill for s in env.slots]
    env.step_command(command(a=2,ticks=20))
    assert all(s.skill is sk for s,sk in zip(env.slots,skills))
    result=env.step_command(command('b',ticks=10))
    assert result['executed']['target_id'] is None and result['executed']['a0']==44
    assert env.total_periods==50


def test_task_success_always_uses_transport_object():
    goal=instance().goal_centre
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',goal[0],goal[1],width=.3,height=.3)])
    distance=env.object_distance()
    env._commit_target('b')  # Direct lifecycle unit test, no execution or body writes.
    assert env.object_distance()==distance and distance>1.
    assert not env.success_conditions()['distance_ok']


def test_extra_contacts_and_penetration_not_omitted():
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',1.55,1.45,width=.3,height=.3)])
    contacts=env.env.backend.contacts()
    assert any(c.get('entity_id')=='b' and c['pair']=='robot_object' for c in contacts)
    a=env._sensor(0,0,0.,'entity-a'); b=env._sensor(0,0,0.,'b')
    assert not a.contacting_object and a.contacting_static
    assert b.contacting_object and b.contact_point_robot is not None
    assert env.env.backend.penetration_by_pair()['robot_object']>0
    outside=MultiObjectOptionEnv(instance(),[MovableBox('b',8.,2.)])
    assert not outside.env.backend.in_bounds()
    with pytest.raises(NotImplementedError): env.env.backend.export_state()


def test_global_brake_without_current_target_support():
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    result=env.step_command(command(None))
    assert result['executed']['target_id'] is None and result['end_tick']==10


def test_entity_relabeling_preserves_sensor_geometry_and_menus():
    one=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    two=MultiObjectOptionEnv(instance(),[MovableBox('other',2.,2.85,width=.3,height=.3)],primary_id='target')
    a,b=one.public(),two.public()
    assert [r['scan']['distances'] for r in a['robots']]==[r['scan']['distances'] for r in b['robots']]
    assert a['entity_legal_pairs']['entity-a']==b['entity_legal_pairs']['target']
    assert a['entity_legal_pairs']['b']==b['entity_legal_pairs']['other']


def test_target_switch_waits_for_public_reference_to_stop():
    from experiments.planar_tasks.relational_search.target_binding import TargetBinding
    binding=TargetBinding();binding.current='a'
    env=MultiObjectOptionEnv(instance(),[MovableBox('b',2.,2.85,width=.3,height=.3)])
    for tick in (0,10):
        o=env.public();o['tick']=tick
        for entity in o['entities']:entity['last_seen_tick']=tick
        for robot in o['robots']:
            robot['own']['vx']=robot['own']['vy']=0.
            robot['execution']['reference_velocity']=[.3,0.]
        result=binding.select(command('b'),o)
        assert not result['commit'] and result['phase']=='switch_brake'
