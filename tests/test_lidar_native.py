"""The fused kernel must preserve the existing NumPy observations exactly."""
import json
import random
import math
import shutil

import numpy as np
import pytest
from coimagine.random_start import lidar as L, lidar_native as N
from coimagine.random_start.geometry import box_polygon
from experiments.option_rl_v19.synthetic_shapes import SHAPES


@pytest.fixture(scope='module', autouse=True)
def built():
    if shutil.which('gcc') is None:
        pytest.skip('native kernel needs gcc')
    N.build()
    N.kernel.cache_clear()
    assert N.kernel() is not None


def equal(args, kwargs):
    reference = L.scan_lidar_batch(*args, **kwargs)
    actual = N.scan_compiled(*args, **kwargs)
    assert reference == actual
    assert np.asarray(reference.distances).tobytes() == np.asarray(actual.distances).tobytes()


@pytest.mark.parametrize('shape', ['T', 'triangle', 'octagon', 'box'])
def test_exact_random_geometry(shape):
    rng = random.Random(137)
    for n in range(100):
        arena = (5., 4.)
        obstacles = tuple(box_polygon(rng.uniform(.3,4.7), rng.uniform(.3,3.7),
                         rng.uniform(-3.,3.), .2, .35) for _ in range(n % 5))
        robots = [(rng.uniform(0.,5.),rng.uniform(0.,4.),rng.uniform(-math.pi,math.pi),0.,0.,0.),
                  (rng.uniform(0.,5.),rng.uniform(0.,4.),0.,0.,0.,0.)]
        pose = (rng.uniform(.2,4.8),rng.uniform(.2,3.8),rng.uniform(-math.pi,math.pi))
        args = (arena,obstacles,pose,robots,0,n,n*.05)
        kwargs = dict(scan_rays=(4,72,288)[n%3],scan_range=(.05,1.2,5.,20.)[n%4],
                      object_polygon=None if shape=='box' else SHAPES[shape])
        equal(args,kwargs)


@pytest.mark.parametrize('origin', [(0.,0.),(.02,.02),(2.,2.),(1.5,2.),(5.,4.),(2.,0.)])
@pytest.mark.parametrize('heading', [0.,math.pi/2,math.pi,-math.pi/2])
def test_corner_inside_tangent_parallel_and_coincident_hits(origin,heading):
    # Coincident obstacle/object geometry tests entry order; repeated vertex is degenerate.
    poly=((-0.5,-0.5),(0.5,-0.5),(0.5,0.5),(0.5,0.5),(-0.5,0.5))
    args=((5.,4.),(box_polygon(2.,2.,0.,.5,.5),),(2.,2.,0.),
          [(*origin,heading,0.,0.,0.),(2.,2.,0.,0.,0.,0.)],0,0,0.)
    equal(args,dict(scan_rays=72,object_polygon=poly))


def test_override_and_state_changes_have_no_cross_frame_cache():
    args=((5.,4.),(),(2.,2.,.3),[(1.,2.,0.,0.,0.,0.),(4.,3.,0.,0.,0.,0.)],0,0,0.)
    for angle in (0.,.7,-1.,0.):equal(args,dict(object_angle_override=angle))


def test_missing_binary_fallback_and_required_mode(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv('COIMAGINE_LIDAR_CACHE',str(tmp_path))
    try:
        monkeypatch.setenv('COIMAGINE_LIDAR_KERNEL','auto');N.kernel.cache_clear()
        assert N.kernel() is None
        assert 'native_not_built' in capsys.readouterr().err
        monkeypatch.setenv('COIMAGINE_LIDAR_KERNEL','native');N.kernel.cache_clear()
        with pytest.raises(RuntimeError,match='not built'):N.kernel()
        monkeypatch.setenv('COIMAGINE_LIDAR_KERNEL','numpy');N.kernel.cache_clear()
        assert N.kernel() is None
    finally:N.kernel.cache_clear()


def test_corrupt_binary_fails_closed(tmp_path):
    dest,identity=N._location()
    (tmp_path/'manifest.json').write_bytes((dest/'manifest.json').read_bytes())
    (tmp_path/'lidar.so').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='mismatch'):N._load_at(tmp_path,identity)


def test_parallel_calls_use_independent_outputs():
    from concurrent.futures import ThreadPoolExecutor
    args=((5.,4.),(),(2.,2.,.3),[(1.,2.,0.,0.,0.,0.),(4.,3.,0.,0.,0.,0.)],0,0,0.)
    angles=[j*.1 for j in range(20)]
    expected=[L.scan_lidar_batch(*args,object_angle_override=a) for a in angles]
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual=list(pool.map(lambda a:N.scan_compiled(*args,object_angle_override=a),angles))
    assert actual==expected


def test_new_campaign_snapshots_native_source_and_freezes(tmp_path):
    from experiments.planar_tasks.relational_search.pipeline import Campaign,read
    c=Campaign.create(tmp_path/'campaign',sequence=[dict(target_id=None,a0=44,a1=44,ticks=1)])
    rel='src/coimagine/random_start/lidar_kernel.c'
    assert read(c.path/'sources.json')[rel]==N._sha(N.SOURCE)
    assert (c.path/'sources'/rel).read_bytes()==N.SOURCE.read_bytes()
    assert c.freeze()['status']=='verified'
    assert c.verify_frozen()['status']=='verified'


def test_runtime_summary_and_fallback_scan(monkeypatch,tmp_path):
    assert N.runtime_info()['engine']=='native'
    args=((5.,4.),(),(2.,2.,.3),[(1.,2.,0.,0.,0.,0.),(4.,3.,0.,0.,0.,0.)],0,0,0.)
    monkeypatch.setenv('COIMAGINE_LIDAR_CACHE',str(tmp_path))
    monkeypatch.setenv('COIMAGINE_LIDAR_KERNEL','auto')
    try:
        N.kernel.cache_clear()
        assert N.scan_batch(*args)==L.scan_lidar_batch(*args)
        assert N.runtime_info()==dict(version=N.VERSION,engine='numpy',build=None)
    finally:N.kernel.cache_clear()
