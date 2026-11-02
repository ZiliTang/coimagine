"""D011 same-face union / dual contact allocation tests; pure, no physics."""
import math
from coimagine.random_start.lidar import LidarScan, SurfacePatch
from coimagine.random_start.skills import RobotKinematics, SkillObservation
from experiments.option_rl_v13 import controller as C
from experiments.option_rl_v19.unknown_shape import allocate_pair_targets


def patch(x0=-.5, x1=.5, y=1.875):
    pts = tuple((x0 + (x1-x0)*i/6, y) for i in range(7))
    return SurfacePatch(points_robot=pts, points_world=pts, distance=abs(y), centroid_distance=abs(y),
        nearest_distance=abs(y), normal_world=(0., 1.), tangent_world=(-1., 0.),
        confidence=1., arc_rays=7, span_m=x1-x0, rms_m=.001, valid=True)


def obs(x, y, p):
    own=RobotKinematics(x,y,0.,0.,0.,0.); mate=RobotKinematics(0.,0.,0.,0.,0.,0.)
    scan=LidarScan(0,0,0.,(0.,0.),0.,(),(),(),())
    return SkillObservation(robot_id=0,tick=0,time=0.,own=own,teammate=mate,scan=scan,
        patches=(p,),patch=p,contacting_object=False,contact_point_robot=None,contacting_static=False,
        contacting_teammate=False,goal_xy=(2.5,1.5),goal_radius=.2,arena=(5.,4.),force_limit=40.,
        remaining_seconds=10.,elapsed_seconds=0.,static_nearest=None,teammate_nearest=None)


def action(): return C.ACTIONS[10]  # PUSH_N, +y


def test_same_face_union_separates_targets():
    a=allocate_pair_targets(obs(2.1,1.5,patch()),action(),obs(2.9,1.5,patch()),action())
    assert a and a['reason']=='separated'
    assert abs(math.dist(a['q0'],a['q1']) - .42) < .02
    assert a['q0'][1] == a['q1'][1]


def test_insufficient_union_uses_one_leader_and_one_wait():
    a=allocate_pair_targets(obs(2.4,1.5,patch(-.1,.1)),action(),obs(2.6,1.5,patch(-.1,.1)),action())
    assert a and a['reason']=='insufficient_union'
    assert (a['q0'] is None) != (a['q1'] is None)


def test_different_faces_are_not_merged():
    p2=patch(-.5,.5)
    # normal opposite (top face) => different face
    top=SurfacePatch(points_robot=p2.points_robot,points_world=p2.points_world,distance=p2.distance,
        centroid_distance=p2.centroid_distance,nearest_distance=p2.nearest_distance,
        normal_world=(0.,-1.),tangent_world=(1.,0.),confidence=1.,arc_rays=7,span_m=1.,rms_m=.001,valid=True)
    a=allocate_pair_targets(obs(2.1,2.5,p2),action(),obs(2.9,2.5,top),action())
    assert a is None
