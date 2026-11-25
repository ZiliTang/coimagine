"""Portable public-history summaries shared by generated CAPs and the host.

No environment, fixture, body state, reward or hidden shape enters this module.
Visible-point fitting is performed in the points' local frame. Static hit memory
is a bounded observed map, not a guarantee that unseen space is empty.
"""
from math import atan2, cos, sin, hypot, pi, sqrt


def initial_memory():
    return dict(static=[], centre=None, stagnation=0., previous_distance=None, evidence={})


def estimate_target(observation, prior=None):
    target = observation['goal']['target_id']
    points = next((e['points'] for e in observation['entities'] if e['id'] == target
                   and e['last_seen_tick'] == observation['tick']), [])
    if len(points) < 3:
        return None
    origin = [sum(p[k] for p in points)/len(points) for k in (0, 1)]
    local = [[p[0]-origin[0], p[1]-origin[1]] for p in points]
    xx = sum(p[0]**2 for p in local); yy = sum(p[1]**2 for p in local)
    xy = sum(p[0]*p[1] for p in local)
    theta = .5*atan2(2*xy, xx-yy)
    t = [cos(theta), sin(theta)]; n = [-t[1], t[0]]
    along = [p[0]*t[0]+p[1]*t[1] for p in local]
    across = [p[0]*n[0]+p[1]*n[1] for p in local]
    lo, hi = [min(along), min(across)], [max(along), max(across)]
    mid = [(lo[k]+hi[k])/2 for k in (0, 1)]
    depth = hi[1]-lo[1]
    if depth < .045:
        # A declared nominal unseen depth; its uncertainty is not a hard bound.
        observers = [r['scan']['origin'] for r in observation['robots']]
        ox = sum(p[0]-origin[0] for p in observers)/len(observers)
        oy = sum(p[1]-origin[1] for p in observers)/len(observers)
        sign = -1 if ox*n[0]+oy*n[1] > 0 else 1
        mid[1] += sign*.125
        depth = .25
    centre = [origin[k]+t[k]*mid[0]+n[k]*mid[1] for k in (0, 1)]
    # Keep the prior in the SAME local coordinates when checking a sparse fit.
    if prior is not None and len(points) < 6:
        p = [prior[k]-origin[k] for k in (0, 1)]
        px, py = p[0]*t[0]+p[1]*t[1], p[0]*n[0]+p[1]*n[1]
        if lo[0]-.15 <= px <= hi[0]+.15 and lo[1]-.30 <= py <= hi[1]+.30:
            centre = [origin[k]+t[k]*px+n[k]*py for k in (0, 1)]
    return dict(centre=centre, angle=theta, width=max(.05, hi[0]-lo[0]),
                depth=depth, uncertainty=.15 if hi[1]-lo[1] < .045 else .08,
                points=points)


def update_memory(memory, event):
    o = event['observation']
    if memory.get('tick') == o['tick']:
        return memory
    fit = estimate_target(o, memory.get('centre'))
    if fit:
        memory['centre'] = fit['centre']
        d = hypot(fit['centre'][0]-o['goal']['xy'][0], fit['centre'][1]-o['goal']['xy'][1])
        previous = memory.get('previous_distance')
        elapsed = max(0, o['tick']-memory.get('tick', o['tick']))*.05
        memory['stagnation'] = memory.get('stagnation', 0.)+elapsed if previous is not None and previous-d < .002 else 0.
        memory['previous_distance'] = d
    cells = {(round(p[0]/.08), round(p[1]/.08)): p for p in memory.get('static', [])}
    for r in o['robots']:
        s = r['scan']; ox, oy = s['origin']
        if len(s['valid']) != 72:
            raise ValueError('symbolic profile requires 72 body-frame rays')
        for valid, kind, a, distance in zip(s['valid'], s['category'], s['ray_angles'], s['distances']):
            if valid and kind == 2:
                p = [ox+distance*cos(a), oy+distance*sin(a)]
                cells[(round(p[0]/.08), round(p[1]/.08))] = p
    # Deterministic bounded spatial sample; no future evidence or truth labels.
    points = list(cells.values())
    c = memory.get('centre') or o['goal']['xy']
    points.sort(key=lambda p: (hypot(p[0]-c[0], p[1]-c[1]), p[0], p[1]))
    memory['static'] = points[:96]
    memory['tick'] = o['tick']
    memory['evidence'] = {}  # never carry model permissions over a real action
    return memory


def option_table():
    out = {}
    for d in range(8):
        code = d*4+1
        for role, pair in [('both', [code, code]), ('r0', [code, 46]), ('r1', [46, code])]:
            for ticks in (6, 12):
                out['push_%s_%d_%d' % (role, d, ticks)] = dict(pair=pair, ticks=ticks, heading=d, family='push', target=True)
    for robot in (0, 1):
        for side, code in [('ccw', 40), ('cw', 43)]:
            pair = [46, 46]; pair[robot] = code
            out['orbit_%d_%s' % (robot, side)] = dict(pair=pair, ticks=12, heading=None, family='orbit', target=True)
    for d in range(8):
        out['move_%d' % d] = dict(pair=[47+4*d+1, 47+4*d+1], ticks=12, heading=d, family='move', target=False)
    out['brake'] = dict(pair=[44, 44], ticks=4, heading=None, family='brake', target=False)
    return out


def semantic_packet(memory, observation):
    o = observation; fit = estimate_target(o, memory.get('centre'))
    seen = fit is not None
    c = fit['centre'] if fit else memory.get('centre') or o['goal']['xy']
    gx, gy = o['goal']['xy'][0]-c[0], o['goal']['xy'][1]-c[1]
    distance = hypot(gx, gy)
    z = dict(seen=seen, distance=distance, gx=gx, gy=gy,
             uncertainty=fit['uncertainty'] if fit else 1., stagnation=memory.get('stagnation', 0.),
             contact0=bool(o['robots'][0]['contacting_object']), contact1=bool(o['robots'][1]['contacting_object']),
             static_contact=any(r['contacting_static'] for r in o['robots']),
             peer_contact=any(r['contacting_teammate'] for r in o['robots']),
             speed=max(hypot(r['own']['vx'], r['own']['vy']) for r in o['robots']))
    points = fit['points'] if fit else []
    radius = max([hypot(p[0]-c[0], p[1]-c[1]) for p in points]+[.25])
    for d in range(8):
        ux, uy = cos(d*pi/4), sin(d*pi/4)
        z['align_%d' % d] = (gx*ux+gy*uy)/max(distance, 1e-9)
        front = max([(p[0]-c[0])*ux+(p[1]-c[1])*uy for p in points]+[.125])
        gaps = [(p[0]-c[0])*ux+(p[1]-c[1])*uy-front for p in memory.get('static', [])
                if abs((p[1]-c[1])*ux-(p[0]-c[0])*uy) < radius+.15
                and (p[0]-c[0])*ux+(p[1]-c[1])*uy > 0]
        z['gap_%d' % d] = max(-2., min(gaps+[5.]))
        z['wall_seen_%d' % d] = bool(gaps)
    menu = {tuple(p) for p in o['entity_legal_pairs'].get(o['goal']['target_id'], [])}
    global_menu = {tuple(p) for p in o.get('global_legal_pairs', [[44, 44]])}
    commands = {}
    evidence = memory.get('evidence', {})
    for name, opt in option_table().items():
        legal = tuple(opt['pair']) in (menu if opt['target'] else global_menu)
        z['legal:'+name] = bool(legal and (seen or not opt['target']))
        # Epistemic inputs computed by the frozen predictor/online rollout host.
        # Before evidence acquisition, all candidates remain available for inquiry.
        z['permit:'+name] = bool(evidence.get(name, {}).get('permit', True))
        commands[name] = dict(target_id=o['goal']['target_id'] if opt['target'] else None,
                              a0=opt['pair'][0], a1=opt['pair'][1], ticks=opt['ticks'])
    return z, commands
