"""Versioned legal simple scenes: dynamic per-episode train roots, fixed val/test.

The old run9 generator stays untouched. This generator conditions the same random
sector proposals on actual disc/rotated-box and wall clearances; it never clips a
robot coordinate into the arena. Scene geometry is a simulator reset input only.
"""
from __future__ import annotations
import math
from collections import Counter
import numpy as np
from coimagine.continuous2d.scenes import Pose
from coimagine.random_start import geometry as G, layout, spec

VERSION='simple-legal-v2'
ARENA=(4.,3.)
OBJECT_MARGIN=.30
ROBOT_OBJECT_GAP=(.06,.37)
ROBOT_CLEARANCE=.05
SPLITS={'train':101,'validation':202,'test':303}


def geometry_audit(instance):
    obj=instance.object_pose
    poly=G.box_polygon(obj.x,obj.y,obj.angle,spec.OBJECT_WIDTH/2,spec.OBJECT_HEIGHT/2)
    body_gaps=[G.circle_polygon_gap((r.x,r.y),spec.ROBOT_RADIUS,poly) for r in instance.robot_poses]
    wall_gaps=[G.wall_distance((r.x,r.y),instance.arena)-spec.ROBOT_RADIUS
               for r in instance.robot_poses]
    pair_gap=math.dist([(instance.robot_poses[0].x),(instance.robot_poses[0].y)],
                       [(instance.robot_poses[1].x),(instance.robot_poses[1].y)])-2*spec.ROBOT_RADIUS
    object_wall_gap=min(G.wall_distance(p,instance.arena) for p in poly)
    reasons=[]
    if min(body_gaps)<0:reasons.append('robot_object_overlap')
    if min(wall_gaps)<0:reasons.append('robot_wall_overlap')
    if pair_gap<0:reasons.append('robot_robot_overlap')
    if object_wall_gap<0:reasons.append('object_wall_overlap')
    return dict(legal=not reasons,reasons=reasons,robot_object_gaps=body_gaps,
                robot_wall_gaps=wall_gaps,robot_robot_gap=pair_gap,object_wall_gap=object_wall_gap)


def scene(seed, index, split='train', *, rejection_counts=None):
    if split not in SPLITS:raise ValueError('explicit train/validation/test split required')
    if index<0 or seed<0:raise ValueError('seed and ordinal must be nonnegative')
    # Root k is reproducible independently of policy sampling and prior rejection counts.
    rng=np.random.default_rng(np.random.SeedSequence([seed,SPLITS[split],2,index]))
    counts=Counter()
    for _ in range(2000):
        angle=float(rng.uniform(-math.pi,math.pi));c,s=abs(math.cos(angle)),abs(math.sin(angle))
        hx,hy=spec.OBJECT_WIDTH/2,spec.OBJECT_HEIGHT/2
        ex,ey=c*hx+s*hy,s*hx+c*hy
        ox=float(rng.uniform(OBJECT_MARGIN+ex,ARENA[0]-OBJECT_MARGIN-ex))
        oy=float(rng.uniform(OBJECT_MARGIN+ey,ARENA[1]-OBJECT_MARGIN-ey))
        base=float(rng.uniform(0,2*math.pi));spread=float(rng.uniform(.9,1.6));robots=[]
        for k in (0,1):
            a=base+k*spread;relative=a-angle
            ca,sa=abs(math.cos(relative)),abs(math.sin(relative))
            radial=min(hx/max(ca,1e-12),hy/max(sa,1e-12))
            distance=radial+float(rng.uniform(.24,.55))
            x,y=ox+distance*math.cos(a),oy+distance*math.sin(a)
            robots.append(Pose(x,y,math.atan2(oy-y,ox-x)))
        # Exactly the old full-uniform goal rectangle. No goal-distance curriculum.
        margin=.60+spec.GOAL_RADIUS
        goal=(float(rng.uniform(margin,ARENA[0]-margin)),float(rng.uniform(margin,ARENA[1]-margin)))
        instance=layout.InstanceSpec(name=f'{VERSION}_{split}_{index:07d}',map='L0',split=split,index=index,
            seed=int(rng.integers(0,1<<30)),arena=ARENA,obstacles=(),object_pose=Pose(ox,oy,angle),
            robot_poses=tuple(robots),goal_centre=goal,goal_radius=spec.GOAL_RADIUS,env_version=spec.ENV_VERSION)
        audit=geometry_audit(instance)
        reason=(audit['reasons'][0] if not audit['legal'] else
                'robot_object_margin' if min(audit['robot_object_gaps'])<ROBOT_OBJECT_GAP[0] or max(audit['robot_object_gaps'])>ROBOT_OBJECT_GAP[1] else
                'robot_wall_margin' if min(audit['robot_wall_gaps'])<ROBOT_CLEARANCE else
                'robot_pair_margin' if audit['robot_robot_gap']<ROBOT_CLEARANCE else None)
        if reason is not None:counts[reason]+=1;continue
        if rejection_counts is not None:rejection_counts.update(counts)
        return instance
    raise RuntimeError(f'legal scene rejection budget exhausted: {dict(counts)}')


def panel(seed,count,split):
    return [scene(seed,i,split) for i in range(count)]
