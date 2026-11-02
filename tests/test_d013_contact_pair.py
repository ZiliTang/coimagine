import math
from coimagine.random_start.lidar import LidarScan, SurfacePatch
from coimagine.random_start.skills import RobotKinematics, SkillObservation
from experiments.option_rl_v13 import controller as C
from experiments.option_rl_v19.unknown_shape import select_contact_pair


def patch(x0=-.5, x1=.5, y=1.875):
    pts=tuple((x0+(x1-x0)*i/6,y) for i in range(7))
    return SurfacePatch(points_robot=pts,points_world=pts,distance=abs(y),centroid_distance=abs(y),
        nearest_distance=abs(y),normal_world=(0.,1.),tangent_world=(-1.,0.),confidence=1.,
        arc_rays=7,span_m=x1-x0,rms_m=.001,valid=True)


def obs(x,y,p):
    own=RobotKinematics(x,y,0.,0.,0.,0.); mate=RobotKinematics(0.,0.,0.,0.,0.,0.)
    scan=LidarScan(0,0,0.,(0.,0.),0.,(),(),(),())
    return SkillObservation(robot_id=0,tick=0,time=0.,own=own,teammate=mate,scan=scan,
        patches=(p,),patch=p,contacting_object=False,contact_point_robot=None,contacting_static=False,
        contacting_teammate=False,goal_xy=(2.5,1.5),goal_radius=.2,arena=(5.,4.),force_limit=40.,
        remaining_seconds=10.,elapsed_seconds=0.,static_nearest=None,teammate_nearest=None)


def action(): return C.ACTIONS[10]


def test_selector_uses_nearest_and_separates():
    r=select_contact_pair(obs(2.10,1.5,patch()),action(),obs(2.90,1.5,patch()),action())
    assert r is not None
    assert r['same_face']
    assert math.dist(r['q0'],r['q1']) >= .38
    # same-face ordering must not cross
    assert r['q0'][0] <= r['q1'][0] + 1e-6
    assert -.5 <= r['q0'][0] <= r['q1'][0] <= .5


def test_selector_is_deterministic_without_multistart():
    a=select_contact_pair(obs(2.10,1.5,patch()),action(),obs(2.90,1.5,patch()),action())
    b=select_contact_pair(obs(2.10,1.5,patch()),action(),obs(2.90,1.5,patch()),action())
    assert a['q0']==b['q0'] and a['q1']==b['q1'] and a['cost']==b['cost']


def test_selector_respects_short_support():
    r=select_contact_pair(obs(2.40,1.5,patch(-.05,.05)),action(),obs(2.60,1.5,patch(-.05,.05)),action())
    assert r is not None
    assert r['q0'][0] <= r['q1'][0] + 1e-6


def test_selector_local_radius_stays_near_previous_q():
    p=patch()
    # previous q0/q1 near the support centre; local search must stay within .1
    r=select_contact_pair(obs(2.10,1.5,p),action(),obs(2.90,1.5,p),action(),
                          prev_q0=(0.0,1.875), prev_q1=(0.2,1.875), local_radius=.1)
    assert r is not None
    assert abs(r['q0'][0]-0.0) <= .101
    assert abs(r['q1'][0]-0.2) <= .101


def test_support_anchor_maps_relative_q_to_shifted_face():
    from experiments.option_rl_v19.unknown_shape import visible_faces, _face_anchor, _map_anchor
    p0=patch(); f0=visible_faces(obs(0.,0.,p0))[0]
    a=_face_anchor(f0, .20)  # u=.20 from support centre
    # a shifted/equivalent visible face (same line, new points)
    p1=SurfacePatch(points_robot=p0.points_robot,points_world=tuple((x+1.0,y) for x,y in p0.points_world),
        distance=p0.distance,centroid_distance=p0.centroid_distance,nearest_distance=p0.nearest_distance,
        normal_world=p0.normal_world,tangent_world=p0.tangent_world,confidence=1.,arc_rays=7,
        span_m=p0.span_m,rms_m=p0.rms_m,valid=True)
    q=_map_anchor(obs(1.,0.,p1),a,(-9.,-9.))
    assert abs(q[0] - 0.80) < 1e-6 and abs(q[1]-1.875) < 1e-6
