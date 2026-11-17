from copy import deepcopy
from dataclasses import replace
import json
import pytest
from experiments.planar_tasks.relational_search import transfer_tasks as T
from experiments.planar_tasks.relational_search import pipeline_execution as X
from experiments.planar_tasks.relational_search import testbed as RT
from experiments.option_rl_v19 import testbed_server as server


@pytest.mark.parametrize('case',T.cases(),ids=lambda c:c['id'])
def test_fixture_geometry_goal_and_public_information_boundary(case):
    scene,boxes=X.fixture(case['fixture']);env=RT.validate_geometry(scene,boxes)
    RT.validate_geometry(replace(scene,object_pose=replace(scene.object_pose,x=scene.goal_centre[0],y=scene.goal_centre[1])),boxes)
    o=env.public()
    assert {e['id'] for e in o['entities']}=={'entity-a','entity-b'}
    assert o['goal']['radius']==.2
    encoded=json.dumps(o)
    for forbidden in ['straight_corridor_overlap','recommended_action','host_annotation','fixture_sha256',case['id']]:
        assert forbidden not in encoded


def test_relation_pairs_only_move_the_same_obstacle():
    rows=T.cases()
    for name in T.DIRECTIONS:
        a,b=[deepcopy(c) for c in rows if c['layout']==name]
        assert a['host_annotation']['lateral_separation_margin_m']<0<b['host_annotation']['lateral_separation_margin_m']
        for c in [a,b]:
            del c['fixture']['movable_objects'][0]['x'];del c['fixture']['movable_objects'][0]['y']
        assert a['fixture']==b['fixture']


def test_preset_seed_roundtrip_and_no_overwrite(tmp_path,monkeypatch):
    monkeypatch.setattr(server,'PRESETS',tmp_path)
    server._seed_presets()
    for c in T.cases():
        pid=T.preset_id(c);p=tmp_path/(pid+'.json');before=p.read_bytes();r=server._preset_get(pid);f=c['fixture']
        assert r['object']==f['object'] and r['robots']==f['robots']
        assert r['movable_objects']==f['movable_objects'] and r['scene_meta']['goal']==f['goal']
        assert r['sequence']==T.preview_sequence()
        saved=server._preset_save(r)
        assert server._preset_get(saved['id'])['scene_meta']['goal']==f['goal']
        server._seed_presets();assert p.read_bytes()==before
