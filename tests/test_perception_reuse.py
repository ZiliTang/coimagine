"""Same-instant sensor reuse must preserve fresh state/contact and causal timing."""
import math
from dataclasses import replace

import pytest

from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import env as E, lidar as L
from coimagine.random_start.layout import InstanceSpec


@pytest.fixture
def world(monkeypatch):
    inst = InstanceSpec(name='perception-cache', map='L0', split='test', index=0, seed=11,
                        arena=(5.,4.), obstacles=(), object_pose=Pose(2.5,2.,0.),
                        robot_poses=(Pose(1.5,2.,0.),Pose(4.,3.5,0.)), goal_centre=(3.,2.))
    env = E.RandomStartEnv(inst, E.EnvConfig(lidar_backend='numpy_batch'))
    counts = {'scan':0, 'patch':0}
    scan, patches = E.scan_lidar_batch, E.scan_patches
    def cast(*a,**kw):
        counts['scan'] += 1
        return scan(*a,**kw)
    def fit(*a,**kw):
        counts['patch'] += 1
        return patches(*a,**kw)
    monkeypatch.setattr(E,'scan_lidar_batch',cast)
    monkeypatch.setattr(E,'scan_patches',fit)
    return env,counts


def test_same_instant_reuses_geometry_and_does_not_repeat_tracker_update(world):
    env,counts = world
    a = env.boundary_observations()
    before = [t.updates for t in env.anchor_trackers.values()]
    b = env.boundary_observations()
    assert a == b and a[0].scan is b[0].scan and a[0].patches is b[0].patches
    assert counts == {'scan':2,'patch':2}
    assert [t.updates for t in env.anchor_trackers.values()] == before


@pytest.mark.parametrize('changed',['tick','timestamp','heading','object','own','peer'])
def test_new_time_or_changed_geometry_at_same_tick_is_not_stale(world,changed):
    env,counts = world
    first=env.observe(0,0,0.)
    tick,stamp=0,0.
    if changed=='tick': tick=1
    elif changed=='timestamp': stamp=.05
    elif changed=='heading': env.heading[0]=.2
    elif changed=='object': env.env.backend.object_body.angle=.2
    elif changed=='own': env.env.backend.robot_bodies[0].position=(1.6,2.)
    elif changed=='peer': env.env.backend.robot_bodies[1].position=(3.8,3.5)
    second=env.observe(0,tick,stamp)
    assert second.scan is not first.scan
    assert counts == {'scan':2,'patch':2}
    assert second.scan.tick==tick and second.scan.sample_time==stamp


def test_contact_feedback_is_rebuilt_even_when_scan_is_reused(world):
    env,counts=world
    no_contact={'0':[],'1':[]}
    a=env.observe(0,0,0.,no_contact)
    touching={'0':[('object',0,1.68,2.)],'1':[]}
    b=env.observe(0,0,0.,touching)
    assert b.scan is a.scan and counts=={'scan':1,'patch':1}
    assert not a.contacting_object and b.contacting_object
    assert b.contact_point_robot is not None


def test_branch_restore_clears_sensor_cache(world):
    env,counts=world
    env.observe(0,0,0.);saved=env.export_branch_state()
    env.restore_branch_state(saved)
    assert not env._scan_cache and not env._patch_cache
    env.observe(0,0,0.)
    assert counts=={'scan':2,'patch':2}


def test_nearest_preserves_first_tie_and_world_coordinates():
    scan=L.LidarScan(0,0,0.,(2.,3.),0.,(0.,math.pi/2,math.pi),
                     (3.,1.,1.),(True,True,True),(L.CATEGORY_STATIC,)*3,(None,)*3)
    distance,point,index=scan.nearest(L.CATEGORY_STATIC)
    assert distance==1. and index==1
    assert point==(2.+math.cos(math.pi/2),4.)
    assert scan.nearest(L.CATEGORY_OBJECT) is None


def test_patch_grouping_skips_only_invalid_singletons(monkeypatch):
    angles=tuple(i*math.pi/4 for i in range(8))
    good={0,1,7}
    scan=L.LidarScan(0,0,0.,(0.,0.),0.,angles,
                     tuple(1/math.cos(angles[i]) if i in good else 0. for i in range(8)),
                     tuple(i in good for i in range(8)),
                     tuple(L.CATEGORY_OBJECT if i in good else L.CATEGORY_NONE for i in range(8)),
                     (None,)*8)
    groups=[];original=L._split_runs
    def fit(points,*a):
        groups.append(points)
        return original(points,*a)
    monkeypatch.setattr(L,'_split_runs',fit)
    patches=L.scan_patches(scan,(0.,0.))
    assert len(groups)==1 and len(groups[0])==3
    assert len(patches)==1 and patches[0].arc_rays==3
