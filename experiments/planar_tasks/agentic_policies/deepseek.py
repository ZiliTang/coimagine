import json
import math
import sys

# Global persistent state (within episode)
STATE = {
    'phase': 'INIT',
    'timer': 0,
    'last_action': None,
    'object_est': None,
    'object_vel': (0.0, 0.0),
    'robot0_brake_until': 0,
    'robot1_brake_until': 0,
}

# Action IDs
ACTIONS = {
    'PUSH_E_1': 0, 'PUSH_E_2': 1, 'PUSH_E_3': 2, 'PUSH_E_4': 3,
    'PUSH_NE_1': 4, 'PUSH_NE_2': 5, 'PUSH_NE_3': 6, 'PUSH_NE_4': 7,
    'PUSH_N_1': 8, 'PUSH_N_2': 9, 'PUSH_N_3': 10, 'PUSH_N_4': 11,
    'PUSH_NW_1': 12, 'PUSH_NW_2': 13, 'PUSH_NW_3': 14, 'PUSH_NW_4': 15,
    'PUSH_W_1': 16, 'PUSH_W_2': 17, 'PUSH_W_3': 18, 'PUSH_W_4': 19,
    'PUSH_SW_1': 20, 'PUSH_SW_2': 21, 'PUSH_SW_3': 22, 'PUSH_SW_4': 23,
    'PUSH_S_1': 24, 'PUSH_S_2': 25, 'PUSH_S_3': 26, 'PUSH_S_4': 27,
    'PUSH_SE_1': 28, 'PUSH_SE_2': 29, 'PUSH_SE_3': 30, 'PUSH_SE_4': 31,
    'ROTATE_CCW_1': 32, 'ROTATE_CCW_2': 33, 'ROTATE_CCW_3': 34,
    'ROTATE_CW_1': 35, 'ROTATE_CW_2': 36, 'ROTATE_CW_3': 37,
    'REPOSITION_CCW': 40, 'REPOSITION_CW': 43,
    'BRAKE_HOLD': 44, 'YIELD_CLEARANCE': 46
}

def angle_between(ang1, ang2):
    """Shortest signed angle from ang1 to ang2 in radians."""
    diff = (ang2 - ang1 + math.pi) % (2 * math.pi) - math.pi
    return diff

def normalize_angle(ang):
    while ang > math.pi:
        ang -= 2 * math.pi
    while ang < -math.pi:
        ang += 2 * math.pi
    return ang

def clamp(v, low, high):
    return max(low, min(high, v))

def compute_object_state(observations):
    """Estimate object position and velocity from both robots' scans.
    Returns (x, y, vx, vy) or None if not confident."""
    obj_points = []
    for obs in observations:
        scan = obs.get('scan')
        if not scan:
            continue
        # Use category == 1 (object)
        categories = scan.get('category')
        distances = scan.get('distances')
        angles = scan.get('ray_angles')
        origin = scan.get('origin')
        if not all([categories, distances, angles, origin]):
            continue
        for i, cat in enumerate(categories):
            if cat == 1 and distances[i] is not None and distances[i] < 3.0:
                world_angle = angles[i]
                # ray_angles are world angles; origin is world xy
                x = origin[0] + distances[i] * math.cos(world_angle)
                y = origin[1] + distances[i] * math.sin(world_angle)
                obj_points.append((x, y))
    if len(obj_points) < 3:
        return None
    # Cluster points: assume single object, but remove outliers by distance from median
    xs = [p[0] for p in obj_points]
    ys = [p[1] for p in obj_points]
    median_x = sorted(xs)[len(xs)//2]
    median_y = sorted(ys)[len(ys)//2]
    # Keep points within 0.5 of median
    filtered = [(x,y) for x,y in obj_points if math.hypot(x-median_x, y-median_y) < 0.5]
    if len(filtered) < 3:
        return None
    x = sum(p[0] for p in filtered)/len(filtered)
    y = sum(p[1] for p in filtered)/len(filtered)
    # Estimate velocity using history
    if STATE['object_est'] is not None:
        dt = 0.05  # approximate; could use time diff but constant
        vx = (x - STATE['object_est'][0]) / dt
        vy = (y - STATE['object_est'][1]) / dt
        # Low pass filter
        STATE['object_vel'] = (0.7*STATE['object_vel'][0] + 0.3*vx,
                               0.7*STATE['object_vel'][1] + 0.3*vy)
    else:
        STATE['object_vel'] = (0.0, 0.0)
    STATE['object_est'] = (x, y)
    return (x, y, STATE['object_vel'][0], STATE['object_vel'][1])

def select_push_direction(goal_vec, obj_vel, robot_pos, normal_vec=None):
    """Choose best push action to move object towards goal.
    Returns action ID for one robot."""
    # Desired push direction in world
    dx = goal_vec[0] - 0.3 * obj_vel[0]  # damp velocity
    dy = goal_vec[1] - 0.3 * obj_vel[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return ACTIONS['BRAKE_HOLD']
    dx /= length
    dy /= length
    # If robot is on other side, maybe push opposite? but choose best among available angles
    # Map desired direction to one of 8 compass directions
    angle = math.atan2(dy, dx)
    # Decide best action index based on angle
    # PUSH_E, NE, N, NW, W, SW, S, SE
    dirs = [
        (0.0, 0, 'E'),
        (math.pi/4, 4, 'NE'),
        (math.pi/2, 8, 'N'),
        (3*math.pi/4, 12, 'NW'),
        (math.pi, 16, 'W'),
        (-3*math.pi/4, 20, 'SW'),
        (-math.pi/2, 24, 'S'),
        (-math.pi/4, 28, 'SE')
    ]
    best_diff = 999
    best_action = None
    for d, base, name in dirs:
        diff = abs(angle_between(angle, d))
        if diff < best_diff:
            best_diff = diff
            best_action = base  # speed 0.2 chosen for moderate force; we can adjust later
    # Choose speed: defaults to 0.2 (index offset 1)
    action = best_action + 1
    return action

def choose_actions(observations, legal_pairs):
    """Main decision logic. Returns (a0, a1, ticks)."""
    STATE['timer'] += 1
    # Extract per-robot observations
    obs0 = observations[0]
    obs1 = observations[1]
    # Goal
    goal = obs0.get('goal_xy')
    if not goal:
        goal = (3.6, 2.0)
    # Estimate object state
    obj_state = compute_object_state(observations)
    if obj_state is None:
        # No confident object, maybe use initial guess or just brake
        return (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1)
    obj_x, obj_y, obj_vx, obj_vy = obj_state
    # Compute vector to goal
    gx, gy = goal
    to_goal = (gx - obj_x, gy - obj_y)
    dist_to_goal = math.hypot(*to_goal)
    if dist_to_goal < 0.25:
        # Near goal: brake to avoid overshoot
        return (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1)
    # Determine roles: robot closer to object? Actually both can push if positioned.
    # We'll have both robots push towards goal, but if one is poorly positioned, use repositioning.
    # For simplicity, each robot independently chooses push direction based on its view.
    # Robot 0
    own0 = obs0['own']
    pos0 = (own0['x'], own0['y'])
    # Check if robot is in contact with object (patch distance small)
    patch0 = obs0.get('patch')
    contact0 = obs0.get('contacting_object', False)
    # If not in contact, need to approach object first. Use repositioning around object or push?
    # Since we don't have direct approach actions except reposition, but reposition moves around object.
    # We'll attempt to use push actions in direction towards object if not contacting, but push without contact does nothing?
    # Actually push actions may move robot and contact object later. So we can command push direction that moves robot towards object.
    # Determine vector from robot to object
    to_obj0 = (obj_x - pos0[0], obj_y - pos0[1])
    dist_to_obj0 = math.hypot(*to_obj0)
    if not contact0 and dist_to_obj0 > 0.35:
        # Not in contact, need to approach. Use push action towards object (which will move robot until contact)
        # Choose push direction along to_obj
        angle_req = math.atan2(to_obj0[1], to_obj0[0])
        a0 = select_push_for_approach(angle_req)
    else:
        # Probably in contact, push towards goal
        a0 = select_push_direction(to_goal, (obj_vx, obj_vy), pos0)
    # Robot 1 similar
    own1 = obs1['own']
    pos1 = (own1['x'], own1['y'])
    patch1 = obs1.get('patch')
    contact1 = obs1.get('contacting_object', False)
    to_obj1 = (obj_x - pos1[0], obj_y - pos1[1])
    dist_to_obj1 = math.hypot(*to_obj1)
    if not contact1 and dist_to_obj1 > 0.35:
        angle_req = math.atan2(to_obj1[1], to_obj1[0])
        a1 = select_push_for_approach(angle_req)
    else:
        a1 = select_push_direction(to_goal, (obj_vx, obj_vy), pos1)
    # Check legal and adjust if necessary
    pair = (a0, a1)
    if pair not in legal_pairs:
        # fallback: find similar push with lower speed or brake
        # Try reducing speed (if we used speed 2, try speed 1, etc.) but keep safety.
        # Simplest: if not legal, choose nearest legal pair? But for safety use BRAKE.
        if (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD']) in legal_pairs:
            return (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1)
        else:
            # Pick any legal pair with brake for one
            for lp in legal_pairs:
                if lp[0] == ACTIONS['BRAKE_HOLD'] or lp[1] == ACTIONS['BRAKE_HOLD']:
                    return lp[0], lp[1], 1
            return legal_pairs[0][0], legal_pairs[0][1], 1
    return a0, a1, 1

def select_push_for_approach(angle):
    """Select push action to move robot in given world angle (approach)."""
    # Map to compass with speed 0.2
    dirs = [
        (0.0, 0),
        (math.pi/4, 4),
        (math.pi/2, 8),
        (3*math.pi/4, 12),
        (math.pi, 16),
        (-3*math.pi/4, 20),
        (-math.pi/2, 24),
        (-math.pi/4, 28)
    ]
    best_diff = 999
    best_base = 0
    for d, base in dirs:
        diff = abs(angle_between(angle, d))
        if diff < best_diff:
            best_diff = diff
            best_base = base
    return best_base + 1  # speed 0.2

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except:
            continue
        # The observation might contain robots and legal_pairs
        if 'robots' in data and 'legal_pairs' in data:
            robots = data['robots']
            legal_pairs = data['legal_pairs']
            # Reset detection: if decision==0 and maybe time==0
            if data.get('decision', 0) == 0 and STATE['timer'] > 0 and data.get('time', 0) == 0:
                # New episode reset
                STATE['phase'] = 'INIT'
                STATE['timer'] = 0
                STATE['object_est'] = None
                STATE['object_vel'] = (0.0, 0.0)
                STATE['robot0_brake_until'] = 0
                STATE['robot1_brake_until'] = 0
            a0, a1, ticks = choose_actions(robots, legal_pairs)
            # Output JSON line with actions and ticks
            out = {'a0': a0, 'a1': a1, 'ticks': ticks}
            sys.stdout.write(json.dumps(out) + '\n')
            sys.stdout.flush()
        else:
            # Unknown format, output brake
            out = {'a0': 44, 'a1': 44, 'ticks': 1}
            sys.stdout.write(json.dumps(out) + '\n')
            sys.stdout.flush()

if __name__ == '__main__':
    main()
