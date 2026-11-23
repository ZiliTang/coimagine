"""Public, per-candidate geometric and previous-execution cues at an option boundary."""
import numpy as np

from coimagine.random_start.geometry import box_polygon
from coimagine.random_start.lidar import CATEGORY_OBJECT, CATEGORY_STATIC, CATEGORY_TEAMMATE
from coimagine.random_start.spec import CONTROL_DT, ROBOT_RADIUS, SAFETY_GAP
from experiments.planar_tasks.relational_search.estimator import fit_box
from experiments.planar_tasks.relational_search.motion import ACTIONS, DIRECTIONS
from experiments.planar_tasks.semantic_rl.corridor_semantics import sweep_bounds


def direction(action):
    if action.kind == 'move':
        angle = DIRECTIONS.index(action.direction)*np.pi/4
        return np.array([np.cos(angle),np.sin(angle)])
    if getattr(action,'effect_kind',None) == 'translate':
        return np.asarray(action.effect_direction)
    return np.zeros(2)


def fitted(obs, label):
    if obs is None:
        return None
    points = next((e['points'] for e in obs['entities'] if e['id'] == label and e['last_seen_tick'] == obs['tick']), [])
    return fit_box(points,[r['scan']['origin'] for r in obs['robots']])


def endpoints(robot):
    s = robot['scan']
    angle = np.asarray(s['ray_angles'])
    keep = np.asarray(s['valid'],bool)
    point = np.asarray(s['origin']) + np.stack([np.cos(angle),np.sin(angle)],1)*np.asarray(s['distances'])[:,None]
    return point[keep], np.asarray(s['category'])[keep], np.asarray(s['entity_ids'],object)[keep]


def sweep_hits(box, delta, label, robots):
    corners = np.asarray(box_polygon(box['x'],box['y'],box['angle'],box['width']/2,box['height']/2))
    verts, axes = sweep_bounds(corners,delta)
    limits = verts @ axes.T
    static = movable = False
    for robot in robots:
        points, category, ids = endpoints(robot)
        proj = points @ axes.T
        inside = np.all((proj >= limits.min(0)-1e-8) & (proj <= limits.max(0)+1e-8),axis=1)
        static |= bool(np.any(inside & (category==CATEGORY_STATIC)))
        movable |= bool(np.any(inside & (category==CATEGORY_OBJECT) & (ids!=label)))
    return float(static),float(movable)


def candidate_semantics(obs, action, previous=None, previous_action=None):
    """Evaluate the proposed command without executing it or reading its future feedback."""
    robots = obs['robots']
    label = action['target_id'] or obs['goal']['target_id']
    box = fitted(obs,label)
    old_box = fitted(previous,label)
    centre = None if box is None else np.array([box['x'],box['y']])
    seconds = action['ticks']*CONTROL_DT
    features = dict(push_box_seen=0., sweep_static=0., sweep_movable=0., push_goal_progress=0.,
                    contact_alignment=0., move_close_target=0., move_surface_shortfall=0.,
                    move_teammate_shortfall=0., repeat_blocked=0., repeat_robot_progress=0.,
                    repeat_box_observed=0., repeat_box_progress=0.)
    for i,robot in enumerate(robots):
        a = ACTIONS[action[f'a{i}']]
        u = direction(a)
        if not np.any(u):
            continue
        xy = np.array([robot['own']['x'],robot['own']['y']])
        travel = seconds*a.speed
        delta = travel*u
        if a.kind == 'move':
            points, cats, _ = endpoints(robot)
            rel = points-xy
            forward = rel @ u
            side = np.abs(rel @ np.array([-u[1],u[0]]))
            radius = ROBOT_RADIUS+SAFETY_GAP
            close = (forward>0) & (side<=radius)
            for mask,name in [(cats!=CATEGORY_TEAMMATE,'move_surface_shortfall'),(cats==CATEGORY_TEAMMATE,'move_teammate_shortfall')]:
                ok = close & mask
                if np.any(ok):
                    gap = np.min(forward[ok]-np.sqrt(np.maximum(0,radius**2-side[ok]**2)))
                    features[name] += float(np.clip(travel-gap,0,travel))
            if centre is not None:
                features['move_close_target'] += np.linalg.norm(centre-xy)-np.linalg.norm(centre-xy-delta)
        elif centre is not None:
            features['push_box_seen'] += 1.
            static, movable = sweep_hits(box,delta,label,robots)
            features['sweep_static'] += static
            features['sweep_movable'] += movable
            offset = centre-xy
            features['contact_alignment'] += (offset@u)/max(np.linalg.norm(offset),.01)
            if label == obs['goal']['target_id']:
                goal = np.asarray(obs['goal']['xy'])-centre
                features['push_goal_progress'] += np.linalg.norm(goal)-np.linalg.norm(goal-delta)
        if previous is None or previous_action is None:
            continue
        last = ACTIONS[previous_action[f'a{i}']]
        if last.kind!=a.kind or (a.kind=='push' and previous_action['target_id']!=action['target_id']):
            continue
        align = float(direction(last) @ u)
        if align < .7:
            continue
        prior = previous['robots'][i]['own']
        elapsed = (obs['tick']-previous['tick'])*CONTROL_DT
        if elapsed <= 0:
            raise ValueError('public boundaries out of order')
        report = robot['execution']
        blocked = report.get('direction_unavailable') or report.get('stage') in ('blocked','move_blocked','yield_blocked')
        features['repeat_blocked'] += align*bool(blocked)
        features['repeat_robot_progress'] += align*((xy-[prior['x'],prior['y']])@u)/elapsed
        if a.kind=='push' and centre is not None and old_box is not None:
            features['repeat_box_observed'] += align
            features['repeat_box_progress'] += align*((centre-[old_box['x'],old_box['y']])@u)/elapsed
    return features
