"""Small, reproducible manual scenes for reviewing V2 option behavior."""
import math
from coimagine.continuous2d.scenes import Pose
from coimagine.random_start.layout import InstanceSpec


def review_scenes():
    def instance(name, obj, robots, goal):
        return InstanceSpec(name=name, map='L0', split='diagnostic', index=2100,
            seed=20261103, arena=(5., 4.), obstacles=(), object_pose=Pose(*obj),
            robot_poses=tuple(Pose(*p) for p in robots), goal_centre=goal, goal_radius=.20)
    return {
        'v2_front_blocker': instance('v2_front_blocker', (2.5, 2., -math.pi/2),
            ((2.08, 2., 0.), (2.92, 1.78, math.pi)), (3.6, 2.)),
        'v2_wall_yield': instance('v2_wall_yield', (4.20, 2., -math.pi/2),
            ((3.78, 2., 0.), (4.62, 1.78, math.pi)), (4.55, 2.)),
        'v2_opposed': instance('v2_opposed', (2.5, 2., 0.),
            ((1.70, 2., 0.), (3.30, 2., math.pi)), (3.6, 2.)),
        'v2_dual_north': instance('v2_dual_north', (2.5, 2., 0.),
            ((2.75, 1.595, math.pi/2+.12), (2.25, 1.595, math.pi/2-.12)), (2.5, 2.8)),
        'v2_adjacent_ne': instance('v2_adjacent_ne', (2.5, 2., 0.),
            ((2.48, 1.595, math.pi/2), (1.70, 2., 0.)), (3.2, 2.7)),
        'v2_idle': instance('v2_idle', (2.5, 2., 0.),
            ((.8, .8, 0.), (4.2, .8, math.pi)), (3.6, 2.)),
    }


def review_presets():
    def segment(a0, a1, seconds):
        return dict(a0=a0, a1=a1, decisions=round(seconds/.2))
    return [
        ('v2_blocker_brake', 'V2 · 前方阻挡 · PUSH + BRAKE', 'v2_front_blocker',
         [segment(2, 44, 7.), segment(44, 44, 2.)]),
        ('v2_blocker_yield', 'V2 · 前方让路 · PUSH + YIELD', 'v2_front_blocker',
         [segment(2, 46, 7.), segment(44, 44, 2.)]),
        ('v2_wall_yield', 'V2 · 近墙侧让 · PUSH + YIELD', 'v2_wall_yield',
         [segment(2, 46, 4.), segment(44, 44, 2.)]),
        ('v2_opposed', 'V2 · 对向施力 · 保留物理效果', 'v2_opposed',
         [segment(2, 18, 5.), segment(44, 44, 2.)]),
        ('v2_wrong_direction', 'V2 · 反面错误方向拒绝 → 改为让路', 'v2_front_blocker',
         [segment(2, 2, 1.6), segment(2, 46, 7.), segment(44, 44, 2.)]),
        ('v2_dual_north', 'V2 · 同面双北推回归', 'v2_dual_north',
         [segment(10, 10, 5.), segment(44, 44, 2.)]),
        ('v2_adjacent_ne', 'V2 · N + E 协推回归', 'v2_adjacent_ne',
         [segment(10, 2, 5.), segment(44, 44, 2.)]),
        ('v2_yield_idle', 'V2 · 无威胁应静止 · YIELD + BRAKE', 'v2_idle',
         [segment(46, 44, 3.)]),
    ]
