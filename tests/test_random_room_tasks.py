from copy import deepcopy
import math
import pytest
from coimagine.random_start.spec import ROBOT_DIAMETER
from experiments.planar_tasks.authoring_v2.random_room_tasks import generate,polygon,separated,inside
from experiments.planar_tasks.relational_search import pipeline_execution as X

def test_fixed_panel_geometry_and_split():
    data=generate();assert data==generate();assert len(data['cases'])==200
    assert {s:sum(r['split']==s for r in data['cases']) for s in ['dev','validation','test']}=={'dev':160,'validation':20,'test':20}
    for row in data['cases']:
        f=row['fixture'];p=f['object'];g=f['goal']
        assert math.hypot(g['x']-p['x'],g['y']-p['y'])>=1.8
        shapes=[polygon(p['x'],p['y'],.25,1.,p['angle'])]
        shapes += [polygon(b['x'],b['y'],b['width'],b['height'],b['angle']) for b in f['movable_objects']]
        shapes += [polygon(b['cx'],b['cy'],2*b['half_w'],2*b['half_h'],b['angle']) for b in f['room']['obstacles']]
        goal=polygon(g['x'],g['y'],.25,1.,p['angle'])
        assert inside(goal) and all(separated(goal,s,.1) for s in shapes)
        shapes += [polygon(p['x'],p['y'],ROBOT_DIAMETER,ROBOT_DIAMETER,p['angle']) for p in f['robots']]
        assert all(inside(p) for p in shapes)
        assert all(separated(p,q) for i,p in enumerate(shapes) for q in shapes[i+1:])
        scene,boxes=X.fixture(f);assert len(boxes)+1==row['movable_count']
        assert len(scene.obstacles)==len(f['room']['obstacles'])

def test_schema_does_not_change_old_fixtures():
    f=X.seed_fixture();s,_=X.fixture(f);assert len(s.obstacles)==2
    bad=deepcopy(f);bad['room']={'arena':[6.,4.8],'obstacles':[]}
    with pytest.raises(ValueError):X.fixture(bad)
    f=generate()['cases'][0]['fixture'];f['room']['arena']=[7,5]
    with pytest.raises(ValueError):X.fixture(f)
