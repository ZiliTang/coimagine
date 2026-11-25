"""Public rectangle pose sets and explicit whole-motion envelopes.

The test profile declares a 1 x .25 m rectangular target CAD prior. Without a
supported long face the estimate is unknown; no nominal unseen depth is filled.
Envelope sizes are versioned host assumptions, audited separately from geometry.
"""
import math
from dataclasses import dataclass
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.ops import unary_union


def disc(x,y,r):
    # Circumscribed polygon, so the queried envelope contains the circular body.
    return Point(x,y).buffer(r/math.cos(math.pi/128),quad_segs=32)


@dataclass
class PoseSet:
    outer: object
    inner: object
    centre: tuple
    normal: tuple
    tangent: tuple
    width: float
    length: float
    support: int


def rectangle_pose(observation, *, width=.25, length=1., error=.003):
    target=observation['goal']['target_id']
    entity=next((x for x in observation['entities']if x['id']==target and x['last_seen_tick']==observation['tick']),None)
    if entity is None:return None
    points=entity['points'];possibilities=[]
    for robot in observation['robots']:
        for patch in robot['patches']:
            if not patch.get('valid')or patch.get('span_m',0)<=width+2*error or patch.get('rms_m',1)>error/3:continue
            n=patch['normal_world'];t=(-n[1],n[0]);face=sum(p[0]*n[0]+p[1]*n[1]for p in patch['points_world'])/len(patch['points_world'])
            line=[p for p in points if abs(p[0]*n[0]+p[1]*n[1]-face)<error]
            if len(line)<4:continue
            ts=[p[0]*t[0]+p[1]*t[1]for p in line]
            low=max(ts)-length/2;high=min(ts)+length/2
            if low>high+2*error:continue
            def rect(a,b,c,d):return Polygon([(u*t[0]+v*n[0],u*t[1]+v*n[1])for u,v in [(a,c),(b,c),(b,d),(a,d)]])
            outer=rect(low-length/2-error,high+length/2+error,face-error,face+width+error)
            if not all(outer.buffer(1e-8).covers(Point(p))for p in points):continue
            inner=rect(high-length/2+error,low+length/2-error,face+error,face+width-error)
            if inner.is_empty or not inner.is_valid:continue
            centre=tuple((low+high)/2*t[k]+(face+width/2)*n[k]for k in (0,1))
            possibilities.append(PoseSet(outer,inner,centre,tuple(n),tuple(t),width,length,len(line)))
    if not possibilities:return None
    # Every long-face constraint describes the same declared rectangle.
    outer=possibilities[0].outer
    for item in possibilities[1:]:outer=outer.intersection(item.outer)
    if outer.is_empty:return None
    result=max(possibilities,key=lambda x:x.support)
    result.outer=outer;result.inner=unary_union([p.inner for p in possibilities])
    return result


@dataclass(frozen=True)
class MotionContract:
    version: str='bounded-short-command-plus-brake-v1'
    move_ticks: int=4
    push_ticks: int=4
    brake_ticks: int=4
    robot_radius: float=.10
    move_robot_path: float=.045
    push_robot_path: float=.075
    robot_stop_path: float=.120
    target_push_boundary_path: float=.120
    target_stop_boundary_path: float=.080
    initial_static_separation: float=.10
    minimum_static_component_area: float=.04
    pose_error: float=.003
    quality: str='explicit host envelope assumption; independently audited, not global dynamics theorem'


def envelopes(observation,pose,kind,robot,contract=MotionContract(),direction=None,*,target_stationary=False):
    """Whole command + following brake, including rotations via boundary travel.

    No commanded endpoint is substituted for the trajectory bound. Circles
    overapproximate every robot centre path; buffering the pose set covers every
    target boundary path under the stated motion contract.
    """
    robots=[]
    for i,r in enumerate(observation['robots']):
        if kind!='push'or i!=robot:
            # MOVE raw reference lies on its declared world direction, with
            # speed in [0,.2]. Public reference and velocity determine a cone
            # containing the reference ramp, velocity tracking and braking tail.
            # KV=40, mass=4, force cap=15, A_ref=.5; the horizon deliberately
            # includes the FULL reference ramp-down plus two tracking constants.
            own=r['own'];v=(own.get('vx',0.),own.get('vy',0.))
            ref=(r.get('execution',{}).get('reference_velocity')or(0.,0.))
            u=(.2*math.cos(direction*math.pi/4),.2*math.sin(direction*math.pi/4))if kind=='observe'and i==robot else (0.,0.)
            vmax=max(math.hypot(*v),math.hypot(*ref),math.hypot(*u));rmax=max(math.hypot(*ref),math.hypot(*u))
            kmin=min(40.,15./max(vmax+rmax,1e-9))
            horizon=(contract.move_ticks*.05 if kind=='observe'and i==robot else 0.)+rmax/.5+2*4./kmin+.1
            points=[(own['x']+horizon*q[0],own['y']+horizon*q[1])for q in ((0.,0.),v,ref,u)]
            robots.append(MultiPoint(points).convex_hull.buffer((contract.robot_radius+.003)/math.cos(math.pi/128),quad_segs=32))
            continue
        motion=contract.robot_stop_path
        if kind=='push'and i==robot:motion+=contract.push_robot_path
        robots.append(disc(r['own']['x'],r['own']['y'],contract.robot_radius+motion))
    target_motion=0. if target_stationary and kind!='push'else contract.target_stop_boundary_path+(contract.target_push_boundary_path if kind=='push'else 0.)
    target=pose.outer.buffer(target_motion)if pose else None
    return dict(robots=robots,target=target,static=unary_union([*robots,*([target]if target is not None else [])]))
