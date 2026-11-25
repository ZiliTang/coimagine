from copy import deepcopy
import math
import shutil

import numpy as np
import pytest
import shapely
from shapely.geometry import Point, box

from experiments.planar_tasks.authoring_v2.four_stage.native_geometry import NativeEvidenceMap, build, kernel
from experiments.planar_tasks.authoring_v2.symbolic_general.simulation import replay_boundary, observation
from tools.cap_unknown_safety import fixture


@pytest.fixture(scope="module", autouse=True)
def compiled_kernel():
    if not shutil.which("g++"):
        pytest.skip("native geometry requires explicit C++ build")
    build()


def initial(blocked=False):
    world, _, _ = replay_boundary(fixture(blocked), [])
    return observation(world)


def test_hidden_obstacle_does_not_change_native_public_map():
    a,b=initial(False),initial(True)
    assert a == b
    x,y=NativeEvidenceMap(),NativeEvidenceMap()
    x.update(a);y.update(b)
    assert np.array_equal(x.mask(),y.mask())
    assert x.query(Point(5,4).buffer(.1)).status == "unknown"


def test_passive_object_hits_are_occupied_but_manipulated_target_is_not():
    a=initial();b=deepcopy(a);target=a["goal"]["target_id"]
    point=None
    for robot in b["robots"]:
        s=robot["scan"]
        for i,label in enumerate(s["entity_ids"]):
            if label==target:
                assert s["category"][i]==1
                s["entity_ids"][i]="passive-box"
                angle=s["ray_angles"][i];d=s["distances"][i]
                point=Point(s["origin"][0]+d*math.cos(angle),s["origin"][1]+d*math.sin(angle))
    assert point is not None
    x,y=NativeEvidenceMap(),NativeEvidenceMap();x.update(a);y.update(b)
    assert x.query(point.buffer(.002)).status!="occupied"
    assert y.query(point.buffer(.002)).status=="occupied"


def test_partial_cells_are_unknown_and_clone_has_independent_storage():
    m=NativeEvidenceMap();m.update(initial())
    # Deliberately use a remote cell where the current rays supply no closed
    # certificate. Cover only half, then cover the other half in a clone.
    unknown=np.argwhere(m.mask()==0)
    y,x=next((int(y),int(x)) for y,x in unknown if x*.025>4 and y*.025>3)
    h=m.resolution;left,bottom=x*h,y*h
    m.add_initial_certificate(box(left-1e-9,bottom-1e-9,left+h/2,bottom+h+1e-9),source="declared_initial_separation")
    assert not m.mask()[y,x]
    c=deepcopy(m)
    c.add_initial_certificate(box(left+h/2-1e-9,bottom-1e-9,left+h+1e-9,bottom+h+1e-9),source="declared_initial_separation")
    assert c.mask()[y,x]
    assert not m.mask()[y,x]


def test_native_closed_ray_certificates_do_not_fill_synthetic_obstacle():
    # Independent ray/rectangle intersections. A sufficiently large obstacle
    # cannot fit in any accepted closed-ray cell without crossing a free ray.
    o=initial()
    obstacle=box(2.7,1.7,3.3,2.3)
    walls=box(0,0,6,4.8).boundary
    m=NativeEvidenceMap(scan_stride_ticks=1)
    for t in range(4):
        q=deepcopy(o);q["tick"]=t
        for i,r in enumerate(q["robots"]):
            origin=(1.2+.1*t,1.3+1.2*i)
            s=r["scan"];s["origin"]=list(origin)
            angles=np.arange(72)*math.pi/36
            distances=[]
            for angle in angles:
                line=shapely.LineString([origin,(origin[0]+10*math.cos(angle),origin[1]+10*math.sin(angle))])
                hits=line.intersection(shapely.union_all([walls,obstacle.boundary]))
                distances.append(Point(origin).distance(hits))
            s.update(ray_angles=angles.tolist(),distances=distances,valid=[True]*72,category=[2]*72)
        m.update(q)
    ys,xs=np.nonzero(m.mask());h=m.resolution
    cells=shapely.box(xs*h,ys*h,(xs+1)*h,(ys+1)*h)
    assert np.max(shapely.area(shapely.intersection(cells,obstacle)),initial=0) == 0
    assert m.query(obstacle).status != "free"


def test_kernel_rejects_invalid_arena_and_requires_explicit_certificate_origin():
    assert not kernel().cap_grid_new(float("nan"),4.8,.025,.04)
    m=NativeEvidenceMap()
    with pytest.raises(ValueError,match="unrecognized"):
        m.add_initial_certificate(box(0,0,6,4.8),source="unseen_is_free")
