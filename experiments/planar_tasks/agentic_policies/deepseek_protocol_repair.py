import json
import math
import sys

# Persistent state within episode
STATE = {
    'object_est': None,
    'object_vel': (0.0, 0.0),
    'timer': 0,
    'last_a0': None,
    'last_a1': None,
    'last_ticks': 1,
}

# Action IDs
ACTIONS = {
    'BRAKE_HOLD': 44,
    'YIELD_CLEARANCE': 46,
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
}

# Speed level to index offset for push actions (base + speed_index)
SPEED_OFFSETS = [0, 1, 2, 3]  # speed 0.1, 0.2, 0.3, 0.4
DEFAULT_SPEED_INDEX = 1  # 0.2 m/s moderate

# Compass directions: angle (rad), base action index for speed 0.1
COMPASS = [
    (0.0, 0),          # E
    (math.pi/4, 4),    # NE
    (math.pi/2, 8),    # N
    (3*math.pi/4, 12), # NW
    (math.pi, 16),     # W
    (-3*math.pi/4, 20),# SW
    (-math.pi/2, 24),  # S
    (-math.pi/4, 28),  # SE
]

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

def estimate_object_state(robots):
    """Estimate object position and velocity from both robots' scans.
    Returns (x, y, vx, vy) or None if not confident."""
    obj_points = []
    for obs in robots:
        scan = obs.get('scan')
        if not scan:
            continue
        categories = scan.get('category')
        distances = scan.get('distances')
        angles = scan.get('ray_angles')
        origin = scan.get('origin')
        if not all([categories, distances, angles, origin]):
            continue
        # Collect points classified as object (category 1) within a reasonable range
        for i, cat in enumerate(categories):
            if cat == 1 and distances[i] is not None and distances[i] < 3.0:
                x = origin[0] + distances[i] * math.cos(angles[i])
                y = origin[1] + distances[i] * math.sin(angles[i])
                obj_points.append((x, y))
    if len(obj_points) < 3:
        return None
    # Median-based outlier removal to handle multiple viewpoints
    xs = [p[0] for p in obj_points]
    ys = [p[1] for p in obj_points]
    median_x = sorted(xs)[len(xs)//2]
    median_y = sorted(ys)[len(ys)//2]
    filtered = [(x, y) for x, y in obj_points if math.hypot(x - median_x, y - median_y) < 0.5]
    if len(filtered) < 3:
        return None
    x = sum(p[0] for p in filtered) / len(filtered)
    y = sum(p[1] for p in filtered) / len(filtered)
    # Velocity estimate using previous position (assuming dt ~ 0.05s per decision)
    dt = 0.05
    if STATE['object_est'] is not None:
        vx = (x - STATE['object_est'][0]) / dt
        vy = (y - STATE['object_est'][1]) / dt
        # Low-pass filter
        STATE['object_vel'] = (0.6 * STATE['object_vel'][0] + 0.4 * vx,
                               0.6 * STATE['object_vel'][1] + 0.4 * vy)
    else:
        STATE['object_vel'] = (0.0, 0.0)
    STATE['object_est'] = (x, y)
    return (x, y, STATE['object_vel'][0], STATE['object_vel'][1])

def select_push_action(desired_world_angle, speed_index=1):
    """Return action ID for push in given world angle (radians)."""
    # Find closest compass direction
    best_diff = 999
    best_base = 0
    for d, base in COMPASS:
        diff = abs(angle_between(desired_world_angle, d))
        if diff < best_diff:
            best_diff = diff
            best_base = base
    return best_base + speed_index  # speed offset maps directly

def can_use_pair(legal_pairs_set, a0, a1):
    """Check if pair (a0, a1) is in legal_pairs_set (set of tuples)."""
    return (a0, a1) in legal_pairs_set

def normalize_pair(legal_pairs_set, a0, a1):
    """If pair not legal, try to adjust speed downward or use BRAKE."""
    if can_use_pair(legal_pairs_set, a0, a1):
        return a0, a1
    # Try reducing speed of both pushes by one, two, three steps
    for step in [1, 2, 3]:
        # Determine base and speed for each push action
        # All push actions are contiguous: 0..31
        def reduce_speed(action):
            if 0 <= action <= 31:
                base = (action // 4) * 4  # base for direction (multiple of 4)
                speed = action % 4
                new_speed = speed - step
                if new_speed >= 0:
                    return base + new_speed
            return None
        for new_a0 in [reduce_speed(a0)]:
            if new_a0 is None:
                new_a0 = a0
            for new_a1 in [reduce_speed(a1)]:
                if new_a1 is None:
                    new_a1 = a1
                if can_use_pair(legal_pairs_set, new_a0, new_a1):
                    return new_a0, new_a1
    # Fallback: try BRAKE for one or both
    if can_use_pair(legal_pairs_set, ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD']):
        return ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD']
    # Try BRAKE for robot0 while keeping a1
    if can_use_pair(legal_pairs_set, ACTIONS['BRAKE_HOLD'], a1):
        return ACTIONS['BRAKE_HOLD'], a1
    # Try BRAKE for robot1 while keeping a0
    if can_use_pair(legal_pairs_set, a0, ACTIONS['BRAKE_HOLD']):
        return a0, ACTIONS['BRAKE_HOLD']
    # Last resort: first legal pair
    if legal_pairs_set:
        first = next(iter(legal_pairs_set))
        return first[0], first[1]
    return ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD']

def decide_actions(robots, legal_pairs_set):
    """Main policy. Returns (a0, a1, ticks)."""
    STATE['timer'] += 1
    # Get robots
    if len(robots) < 2:
        return (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1)
    obs0 = robots[0]
    obs1 = robots[1]
    # Goal
    goal = obs0.get('goal_xy')
    if goal is None:
        goal = (3.6, 2.0)  # fallback, but should be present
    # Estimate object state
    obj_state = estimate_object_state(robots)
    if obj_state is None:
        # No object info, hold
        return (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1)
    obj_x, obj_y, obj_vx, obj_vy = obj_state
    to_goal = (goal[0] - obj_x, goal[1] - obj_y)
    dist_to_goal = math.hypot(*to_goal)
    # Success rule: if object is within 0.2 of goal and slow, hold
    if dist_to_goal < 0.25 and math.hypot(obj_vx, obj_vy) < 0.1:
        return (ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1)
    # Determine desired push direction: toward goal, with velocity damping
    desired_angle = math.atan2(to_goal[1], to_goal[0])
    # For each robot, decide if pushing or repositioning
    # Get robot positions and contact status
    own0 = obs0.get('own')
    own1 = obs1.get('own')
    contacting0 = obs0.get('contacting_object', False)
    contacting1 = obs1.get('contacting_object', False)
    # Distance from robot to object
    if own0 and 'x' in own0:
        to_obj0 = (obj_x - own0['x'], obj_y - own0['y'])
        dist_obj0 = math.hypot(*to_obj0)
    else:
        dist_obj0 = 999
    if own1 and 'x' in own1:
        to_obj1 = (obj_x - own1['x'], obj_y - own1['y'])
        dist_obj1 = math.hypot(*to_obj1)
    else:
        dist_obj1 = 999
    # Threshold for being in contact: patch distance small or contacting flag true
    in_contact0 = contacting0 or dist_obj0 < 0.35
    in_contact1 = contacting1 or dist_obj1 < 0.35
    # If not in contact, robot should approach object using push actions (which move robot)
    if not in_contact0:
        approach_angle0 = math.atan2(to_obj0[1], to_obj0[0])
        a0 = select_push_action(approach_angle0, DEFAULT_SPEED_INDEX)
    else:
        # In contact: push in desired_angle, but maybe adjust based on robot relative position
        # Simple: both push in same desired_angle
        a0 = select_push_action(desired_angle, DEFAULT_SPEED_INDEX)
    if not in_contact1:
        approach_angle1 = math.atan2(to_obj1[1], to_obj1[0])
        a1 = select_push_action(approach_angle1, DEFAULT_SPEED_INDEX)
    else:
        a1 = select_push_action(desired_angle, DEFAULT_SPEED_INDEX)
    # Normalize to legal pair
    a0, a1 = normalize_pair(legal_pairs_set, a0, a1)
    # Determine ticks: use 1 for quick reactions, but could be longer if stable; keep 1 for now
    ticks = 1
    return (a0, a1, ticks)

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except Exception as e:
            # On parse error, output safe brake
            out = {'a0': ACTIONS['BRAKE_HOLD'], 'a1': ACTIONS['BRAKE_HOLD'], 'ticks': 1}
            sys.stdout.write(json.dumps(out) + '\n')
            sys.stdout.flush()
            continue
        # Extract observation and maybe reset flag
        obs = data.get('observation', data)  # if no 'observation' key, assume flat (but contract says observation wrapper)
        reset = data.get('reset', False)
        # On reset, clear persistent state
        if reset:
            STATE['object_est'] = None
            STATE['object_vel'] = (0.0, 0.0)
            STATE['timer'] = 0
            STATE['last_a0'] = None
            STATE['last_a1'] = None
            STATE['last_ticks'] = 1
        robots = obs.get('robots')
        legal_pairs = obs.get('legal_pairs')
        if not robots or not legal_pairs:
            # Missing data, brake
            out = {'a0': ACTIONS['BRAKE_HOLD'], 'a1': ACTIONS['BRAKE_HOLD'], 'ticks': 1}
            sys.stdout.write(json.dumps(out) + '\n')
            sys.stdout.flush()
            continue
        # Convert legal_pairs (list of lists) to set of tuples
        try:
            legal_set = set((int(p[0]), int(p[1])) for p in legal_pairs)
        except Exception:
            legal_set = set()
        # Decide actions
        try:
            a0, a1, ticks = decide_actions(robots, legal_set)
        except Exception as e:
            # On unexpected exception, output brake and write error to stderr
            a0, a1, ticks = ACTIONS['BRAKE_HOLD'], ACTIONS['BRAKE_HOLD'], 1
            sys.stderr.write(f"Exception in decide_actions: {e}\n")
        # Ensure ticks within 1..40
        ticks = max(1, min(40, int(ticks)))
        # Output
        out = {'a0': a0, 'a1': a1, 'ticks': ticks}
        sys.stdout.write(json.dumps(out) + '\n')
        sys.stdout.flush()

if __name__ == '__main__':
    main()
