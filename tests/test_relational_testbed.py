"""Multi-object editor persistence, execution boundary and replay integration."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest
from coimagine.continuous2d.scenes import Pose
from experiments.option_rl_v19 import testbed_server as server
from experiments.planar_tasks.relational_search import testbed as T
from experiments.planar_tasks.relational_search.environment import MultiObjectOptionEnv
from experiments.planar_tasks.relational_search.motion import move_id


@pytest.fixture
def store(tmp_path,monkeypatch):
    for name,folder in [('PRESETS','presets'),('RUNS','runs')]:
        path=tmp_path/folder;path.mkdir();monkeypatch.setattr(server,name,path)
    monkeypatch.setattr(server,'RUN_INDEX',tmp_path/'index.json')
    monkeypatch.setattr(server,'_RUN_INDEX_CACHE',None)
    return tmp_path


def body():
    meta=server._scene_dict(T.scene())
    return dict(scene=T.SCENE_ID,object=meta['initial']['object'],robots=meta['initial']['robots'],
        goal=meta['goal'],movable_objects=T.default_boxes(),controller_version=T.CONTROLLER,
        sequence=[dict(target_id=None,a0=move_id('W'),a1=44,decisions=2)],name='roundtrip')


def test_movable_editor_roundtrip_keeps_geometry_goal_target_and_timing(store):
    request=body();request['movable_objects'][0].update(x=3.2,angle=.2,width=.7)
    request['goal']['y']=2.2
    saved=server._preset_save(request);loaded=server._preset_get(saved['id'])
    assert loaded['movable_objects']==request['movable_objects']
    assert loaded['scene_meta']['goal']==request['goal']
    assert loaded['sequence']==request['sequence']
    assert loaded['controller_version']==T.CONTROLLER
    assert any(p['id']==saved['id'] for p in server._preset_list())


def test_new_seed_does_not_rewrite_saved_presets(store):
    server._seed_presets()
    path=server.PRESETS/(T.PRESET_ID+'.json');content=path.read_bytes()
    server._seed_presets()
    assert path.read_bytes()==content
    p=server._preset_get(T.PRESET_ID)
    assert p['sequence']==T.default_sequence() and p['movable_objects']==T.default_boxes()
    assert len(server._option_list())==43
    assert len(server._option_list(relational=True))==75


def test_replay_records_both_bodies_and_exact_poststep_scans(store):
    request=body();request['sequence']=[dict(target_id='entity-b',a0=44,a1=10,decisions=12)]
    run=server._run(request)
    assert run['execution_error'] is None and run['periods']==48
    assert len(run['frames'])==len(run['strip'])==48
    assert all(set(f['entities'])=={'entity-a','entity-b'} for f in run['frames'])
    assert run['frames'][0]['t']==pytest.approx(.05) and run['frames'][-1]['t']==pytest.approx(2.4)
    assert run['events'][0]['handoff']['phase']=='switch_brake'
    assert any(e['handoff']['commit'] for e in run['events'])
    assert all(f['scan'][0]['tick']==f['tick'] for f in run['frames'])
    assert any(x['id']==run['id'] for x in server._run_list())
    # Recording must not alter the underlying controller/physical trajectory.
    plain=MultiObjectOptionEnv(T.scene(),T.parse_boxes(T.default_boxes()))
    for _ in range(12):plain.step_command(dict(target_id='entity-b',a0=44,a1=10,ticks=4))
    assert plain.offline_audit()==run['final_audit']
    assert json.loads((server.RUNS/(run['id']+'.json')).read_text())['movable_objects']==request['movable_objects']


def test_illegal_visible_action_stops_and_retains_partial_replay(store):
    request=body()
    request['sequence']=[dict(target_id=None,a0=44,a1=44,decisions=2),
                         dict(target_id='entity-a',a0=18,a1=18,decisions=2)]
    # Both robots cannot push west from the required visible right faces here.
    run=server._run(request)
    assert run['execution_error']['segment']==1
    assert run['periods']==8 and len(run['frames'])==8
    assert not run['success'] and (server.RUNS/(run['id']+'.json')).exists()


@pytest.mark.parametrize('mutation', ['duplicate','nan','overlap','target_move','mixed','progress'])
def test_invalid_editor_requests_are_rejected(store,mutation):
    request=body()
    if mutation=='duplicate':request['movable_objects'][0]['label']='entity-a'
    if mutation=='nan':request['movable_objects'][0]['x']=float('nan')
    if mutation=='overlap':request['movable_objects'][0].update(x=2.,y=2.)
    if mutation=='target_move':request['sequence'][0]['target_id']='entity-a'
    if mutation=='mixed':request['sequence'][0]['a1']=2
    if mutation=='progress':request['sequence'][0]['budget_m']=.4
    with pytest.raises(ValueError):server._preset_save(request)
    assert not list(server.PRESETS.glob('*.json'))


def test_old_single_object_save_run_remains_legacy(store):
    inst=server.SCENES['contact_0'];meta=server._scene_dict(inst)
    request=dict(scene=inst.name,object=meta['initial']['object'],robots=meta['initial']['robots'],
                 sequence=[dict(a0=44,a1=44,decisions=1)])
    saved=server._preset_save(request)
    assert saved['movable_objects']==[] and saved['controller_version']=='m3b'
    run=server._run(request)
    assert run['periods']==4 and run['controller_version']=='m3b'
    assert 'entities' not in run['frames'][0]
