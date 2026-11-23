"""Protect the base-policy approach and use B for precision or a sustained stall."""


def route(memory, obs, a, b, ctx):
    tick = ctx['tick']
    gap = ctx['gap']
    contact = bool(ctx['touch'])
    if 'mode' not in memory:
        memory.update(mode='A', start=tick, mark=gap, advanced=tick, trigger=None)
    if gap >= 0 and (memory['mark'] < 0 or gap < memory['mark'] - .14):
        memory['mark'] = gap
        memory['advanced'] = tick
    if not ctx['triggered']:
        return 'A'
    if memory['trigger'] is None:
        memory['trigger'] = tick
        if contact and gap > 1.65:
            memory.update(mode='B', start=tick, mark=gap, advanced=tick)

    mode = memory['mode']
    a_pair = (a['a0'], a['a1'])
    b_pair = (b['a0'], b['a1'])
    if a_pair == b_pair:
        return 'B' if b['ticks'] < a['ticks'] else mode

    progress = ctx['progress']
    moving_goalward = progress is not None and progress > .11
    elapsed = tick - max(memory['start'], memory['advanced'])
    if mode == 'A':
        a_push = min(a_pair) < 32
        b_push = min(b_pair) < 32
        same_drive = any(x < 32 and y < 32 and x // 4 == y // 4 for x, y in zip(a_pair, b_pair))
        b_clearance = same_drive and 46 in b_pair and 44 in a_pair
        # A long approach can include no object progress while the robots line up.
        trigger_wait = 460 if gap > 2.2 and not contact else 300
        can_try_b = tick - memory['trigger'] >= trigger_wait
        if b_clearance or (contact and b_push and not a_push) or 0 <= gap < .62:
            can_try_b = True
        if not a_push and not b_push and 0 <= gap < 1.75:
            can_try_b = True
        if a_push and not b_push:
            wait = 355
        elif b_clearance or not a_push and not b_push:
            wait = 205
        else:
            wait = 230
        switch = can_try_b and elapsed >= wait
    else:
        switch = elapsed >= 290
    if switch and not moving_goalward:
        mode = 'B' if mode == 'A' else 'A'
        memory.update(mode=mode, start=tick, mark=gap, advanced=tick)
    return mode
