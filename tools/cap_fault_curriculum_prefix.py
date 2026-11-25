"""Scan frozen composed CAPs, then select equal-format development examples.

The policy sees public observations only. Offline topology and terminal outcomes
select training examples. They are never passed to an executing policy.
"""
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import random

from tools import cap_abc_event_ppo as event
from tools import cap_authored_handoff as handoff
from tools import cap_topology_graph as geometry

s = event.s
RESERVE_TICKS = 240
WINDOW_TICKS = 120
MIN_ADVANCE = .035


def _node(bits):
    station, approach, staged, _, rear, peer, fixed, movable = bits
    access = ('unavailable' if not station else 'peer' if peer else 'inaccessible'
              if not approach else 'contact' if rear else 'staged' if staged else 'approaching')
    obstacle = 'both' if fixed and movable else 'fixed' if fixed else 'movable' if movable else 'open'
    return f'{access}/{obstacle}'


def run_member(job, member, preview, boundary=None, *, push_takeover=None, progress_m=.10):
    """Run one CAP using the same execution contract with either proposal backend."""
    room, fixture, record = job
    counts, history, trace = Counter(), deque(maxlen=5), []

    def decide(world, obs, action, left, reason):
        elapsed = obs['tick'] - s.observe(world)['tick']
        replacement = preview(member, obs, action, elapsed)
        counts['check_' + reason] += 1
        if replacement is None or event.same_control(replacement, action):
            return None
        counts['changed_' + reason] += 1
        return replacement

    def takeover(world, action):
        from experiments.planar_tasks.semantic_rl.push_takeover import push_checkpoints
        return push_checkpoints(world,action,push_takeover,progress_m=progress_m,
                                max_decisions=s.MAX_DECISIONS)

    with s.phase.world(fixture) as (world, _):
        member.update(world.history)
        review = event.PushReview('progress', decide, progress_m=.20, contact_ticks=3)
        pending = None
        while not s.ended(world):
            obs = s.observe(world)
            action = pending if pending is not None else member.propose(obs)
            if boundary is not None:
                action = boundary(world, obs, action, pending is not None)
            if record:
                public = handoff.sample(obs, history)
                del public['flags'], public['z']
                public.update(action=action, n=len(trace), from_event=pending is not None,
                              public_history=world.history.sha256,
                              execution=[r['execution'].get('stage') for r in obs['robots']],
                              static_contact=[bool(r['contacting_static']) for r in obs['robots']],
                              visible_other=[e['id'] for e in obs['entities']
                                             if e['id'] != obs['goal']['target_id'] and e['points']
                                             and e['last_seen_tick'] == obs['tick']])
                bits, gap, _ = geometry.topology(world, obs)
                trace.append(dict(public=public, node=_node(bits), true_gap=round(gap, 5)))
            pending = s.tick_action(world, action, 'split_precision', member.update, counts, review=review,
                                    takeover=takeover if push_takeover is not None else None)
        result = s.output(world, counts)
    return room, dict(result=result, trace=trace)


def _episode(job):
    return run_member(job, s.phase.FrozenCAP(MODULE), event.preview)


def scan_panel(source, fixtures, *, workers=12, trace=True):
    """Caller supplies the verified, composed, fixed72-local source text and fixture dict."""
    global MODULE
    MODULE = s.old.dq.load_native(source)[0]
    jobs = [(k, f, trace) for k, f in fixtures.items()]
    if workers == 1:
        return dict(map(_episode, jobs))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork')) as pool:
        return dict(pool.map(_episode, jobs))


def _tick(trace, i):
    return trace[i]['public']['tick']


def _interventions(trace):
    """Locate low-progress dwell/return windows, point to incoming command if possible."""
    found = {}
    for i, row in enumerate(trace):
        tick = _tick(trace, i)
        if i == 0 or tick > s.MAX_TICKS-RESERVE_TICKS:
            continue
        node = row['node']
        if node in found:
            continue
        # Use the first actual command boundary at or beyond the physical window.
        j = i+1
        while j < len(trace) and _tick(trace, j)-tick < WINDOW_TICKS:
            j += 1
        if j == len(trace):
            continue
        segment = trace[i:j+1]
        visits = sum(r['node'] == node for r in segment)
        if (segment[-1]['node'] == node and visits >= 2 and
                row['true_gap'] - segment[-1]['true_gap'] < MIN_ADVANCE):
            entry = i
            while entry and trace[entry-1]['node'] == node:
                entry -= 1
            found[node] = dict(prefix=max(0, entry-1), entry=entry, window_start=i,
                               window_end=j, visits=visits)
    return found


def _public_example(rows, room, i, label):
    trace = rows[room]['trace']
    return dict(room=room, label=label, baseline_success=bool(rows[room]['result']['success']),
                takeover_at=i, remaining_ticks=s.MAX_TICKS-_tick(trace,i),
                prefix=trace[i]['public']['public_history'],
                public_lead=[r['public'] for r in trace[max(0,i-4):i+1]],
                baseline_follow=[r['public'] for r in trace[i+1:i+5]],
                final_public=trace[-1]['public'])


def select_packet(rows, arm, *, seed, failed=8, recovered=4):
    """Choose distinct rooms, using one identical-case-schema packet in either arm.

    Recovery matches are based on a witnessed duplicate bottleneck on the successful
    trajectory. When unavailable, ordinary successes are explicitly marked controls.
    """
    if arm not in ('graph','random'):
        raise ValueError('arm must be graph or random')
    if not all(r['trace'] for r in rows.values()):
        raise ValueError('selection needs nonempty development command traces')
    rng = random.Random(seed)
    shuffled = sorted(rows)
    rng.shuffle(shuffled)
    loses = [k for k in shuffled if not rows[k]['result']['success']]
    wins = [k for k in shuffled if rows[k]['result']['success']]
    hits = {k:_interventions(r['trace']) for k,r in rows.items()}
    def arbitrary(k):
        trace = rows[k]['trace']
        options = [j for j in range(1,len(trace))
                   if _tick(trace,j) <= s.MAX_TICKS-RESERVE_TICKS]
        return rng.choice(options) if options else 0

    support = defaultdict(lambda: [0,0])
    for k in shuffled:
        for node in hits[k]:
            support[node][int(bool(rows[k]['result']['success']))] += 1
    ranking = sorted(support, key=lambda n:(-support[n][0],-int(support[n][1]>0),n))
    choices = {}
    wanted = min(failed,len(loses))
    if arm == 'graph':
        targeted = wanted - int(wanted >= 3)
        for node in ranking:
            for k in loses:
                if len(choices) >= targeted:
                    break
                if k not in choices and node in hits[k]:
                    choices[k] = (hits[k][node]['prefix'], node, 'failure')
            if len(choices) >= targeted:
                break
    for k in loses:
        if len(choices) == wanted:
            break
        if k not in choices:
            i = arbitrary(k)
            choices[k] = (i,rows[k]['trace'][i]['node'],'failure')

    # Match positives to the actual failed packet, never another arm's held runs.
    failure_nodes = [v[1] for v in choices.values()]
    n_positive = min(len(wins), recovered)
    target_count = len(choices) + n_positive
    # Match a spread of the shown failures. Heterogeneous random failures should
    # not receive four controls all illustrating the first failure's mechanism.
    for j in range(n_positive):
        if not failure_nodes:
            break
        anchor = failure_nodes[j * len(failure_nodes) // n_positive]
        preferred = list(dict.fromkeys([anchor, *failure_nodes]))
        match = next(((k, node) for node in preferred for k in wins
                      if k not in choices and node in hits[k]), None)
        if match:
            k, node = match
            choices[k] = (hits[k][node]['prefix'], node, 'natural_recovery')
    for k in wins:
        if len(choices) >= target_count:
            break
        if k not in choices:
            i = arbitrary(k)
            choices[k]=(i,rows[k]['trace'][i]['node'],'successful_control')

    cases = [_public_example(rows,k,i,label) for k,(i,_,label) in choices.items()]
    edges = defaultdict(set)
    for k,row in rows.items():
        path = [v['node'] for v in row['trace']] + ['SUCCESS' if row['result']['success'] else 'FAILURE']
        for before,after in zip(path,path[1:]):
            edges[(before,after)].add(k)
    audit = dict(arm=arm, seed=seed, possible_failures=len(loses), possible_successes=len(wins),
                 support={n:dict(fail=support[n][0],success=support[n][1]) for n in ranking},
                 edges=[dict(source=a,destination=b,rooms=len(ks)) for (a,b),ks in sorted(edges.items())],
                 selected={k:dict(prefix=i,node=n,label=lab,witness=hits[k].get(n))
                           for k,(i,n,lab) in choices.items()})
    packet = dict(schema='public-case-packet-v1', cases=cases,
                  description='Each lead ends at the proposed intervention boundary. Follow and final samples show the old program outcome for development only. gap is public scan-centroid distance in metres; -1 means target unseen; progress is its change over four previous command boundaries, touch means task target contact. A successful_control is a success without a matching diagnosed stall.')
    return packet, audit
