"""Current public evidence for translating the task object directly to the goal."""
import numpy as np

from coimagine.random_start.geometry import box_polygon
from coimagine.random_start.lidar import CATEGORY_OBJECT, CATEGORY_STATIC
from coimagine.random_start.spec import LIDAR_RANGE
from experiments.planar_tasks.relational_search.estimator import fit_box

COVERAGE_REQUIRED = .80
FEATURE_NAMES = ('target_estimated','static_hit','movable_hit','swept_visibility',
                 'observed_clear','unknown','visible_other_movables','goal_wall_distance_m')


def blocker_type(static, movable):
    if static and movable:
        return 'both'
    if static:
        return 'static'
    return 'movable' if movable else 'clear'


def sweep_bounds(corners, delta):
    """Vertices and normals of an oriented box swept by a straight translation."""
    corners = np.asarray(corners)
    vectors = np.vstack([corners[1]-corners[0], corners[2]-corners[1], delta])
    vectors = vectors[np.linalg.norm(vectors,axis=1)>1e-9]
    axes = vectors[:,[1,0]] * [-1.,1.]
    axes /= np.linalg.norm(axes,axis=1)[:,None]
    vertices = np.vstack([corners, corners+delta])
    return vertices, axes


def scan_support(scan, probes):
    origin = np.asarray(scan['origin'])
    angles = np.asarray(scan['ray_angles'])
    ranges = np.asarray(scan['distances'])
    valid = np.asarray(scan['valid'],dtype=bool)
    ends = origin + np.column_stack((np.cos(angles),np.sin(angles)))*ranges[:,None]
    # Each probe uses the nearer reach of its two bracketing rays.
    order = np.argsort(angles % (2*np.pi))
    directions = angles[order] % (2*np.pi)
    reach = np.where(valid,ranges,LIDAR_RANGE)[order]
    relative = probes-origin
    right = np.searchsorted(directions,np.arctan2(relative[:,1],relative[:,0])%(2*np.pi))%len(order)
    visible = np.linalg.norm(relative,axis=1) <= np.minimum(reach[right],reach[right-1])+1e-8
    return ends,valid,visible


def corridor_semantics(observation):
    """Measure the task-target corridor at this boundary, independently of CAP proposals."""
    tick=observation['tick']
    goal=np.asarray(observation['goal']['xy'])
    label=observation['goal']['target_id']
    robots=observation['robots']
    arena=np.asarray(robots[0]['arena'])
    entities={e['id']:e['points'] for e in observation['entities'] if e['last_seen_tick']==tick and e['points']}
    features=dict(target_estimated=0,static_hit=0,movable_hit=0,swept_visibility=0.,
                  observed_clear=0,unknown=1,visible_other_movables=len(set(entities)-{label}),
                  goal_wall_distance_m=float(np.min(np.r_[goal,arena-goal])))
    box=fit_box(entities.get(label,[]),[r['scan']['origin'] for r in robots])
    if box is None:
        return dict(kind='unknown',**features)
    centre=np.array([box['x'],box['y']])
    delta=goal-centre
    corners=np.array(box_polygon(*centre,box['angle'],box['width']/2,box['height']/2))
    vertices,axes=sweep_bounds(corners,delta)
    spans=vertices@axes.T
    # Nine area samples on each of 20 translated footprints beyond the starting centre.
    offsets=np.array([u*(corners[1]-corners[0])+v*(corners[3]-corners[0])
                      for u in (-.4,0.,.4) for v in (-.4,0.,.4)])
    probes=(centre+np.linspace(.05,1.,20)[:,None,None]*delta+offsets[None]).reshape(-1,2)
    seen=np.zeros(len(probes),bool)
    static=movable=False
    for robot in robots:
        scan=robot['scan']
        ends,valid,visible=scan_support(scan,probes)
        seen |= visible
        projection=ends@axes.T
        inside=valid & ((projection>=spans.min(0)-1e-8)&(projection<=spans.max(0)+1e-8)).all(1)
        categories=np.asarray(scan['category'])
        ids=np.asarray(scan['entity_ids'],dtype=object)
        # The reference concerns internal obstacles; exclude arena boundary hits.
        interior=np.minimum(ends,arena-ends).min(1)>.05
        static |= bool((inside & (categories==CATEGORY_STATIC) & interior).any())
        movable |= bool((inside & (categories==CATEGORY_OBJECT) & (ids!=None) & (ids!=label)).any())  # noqa: E711
    fraction=float(seen.mean())
    kind=blocker_type(static,movable) if static or movable or fraction>=COVERAGE_REQUIRED else 'unknown'
    features.update(target_estimated=1,static_hit=int(static),movable_hit=int(movable),swept_visibility=fraction,
                    observed_clear=int(kind=='clear'),unknown=int(kind=='unknown'))
    return dict(kind=kind,**features)
