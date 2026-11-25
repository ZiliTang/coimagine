"""Observe the frozen P保守推进.v1 at existing ordinary P PUSH prefixes.

The legacy program predates the checker. Run its original code in an isolated
worker, then attach observers to a hash-locked diagnostic copy. Never import
authored code into the host. Every public snapshot must match the archived
prefix; the observer must reproduce the original complete decision permit.
"""
import argparse
from pathlib import Path
from unittest.mock import patch

from experiments.planar_tasks.relational_search.contract import digest
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source
from tools import cap_safe_nominal_risk_forks as forks
from tools import cap_safe_nominal_study as study
from tools.cap_entry_residual_study import immutable


OUT = study.OUT.parent / 'cap_policy_checker_v1'
CONTROL = 'diag_dev_nat_003'
COMPOSITION = 'P保守推进.v1@G程序.v1'
BASE_COMPOSITION = 'P绕行.v1@G程序.v1'
SELECTION = ('First 2 room IDs ascending per recorded initial geometry stratum. '
             'Each uses frozen P-only history. Select ordinary non-pending goal '
             'translation pushes, tick > 0, ranks 1 and 6 counted from one. '
             'Preserve missing ranks. Select independently of trial03 output, outcomes and checker.')


def selection(out):
    records = study.read(study.OUT/'room_records.json')
    strata = {s: sorted(k for k, r in records.items() if r['stratum'] == s)[:2]
              for s in sorted({r['stratum'] for r in records.values()})}
    plan = dict(schema_version=1, development_control=dict(room=CONTROL, tick=50),
                panel=study.HOLDOUT, previously_evaluated_panel=True, selection=SELECTION,
                strata=strata, push_ranks=[1, 6],
                room_records_sha256=study.sha(study.OUT/'room_records.json'),
                source_panels=str(study.OUT.relative_to(study.ROOT)))
    immutable(out/'selection.json', plan)
    return plan


def sources():
    locked = study.read(study.OUT/'locked.json')
    names = ('P', locked['name'])
    expected = (study.BASE_SHA, forks.FINAL)
    result = []
    for name, checksum in zip(names, expected):
        path = study.OUT/'sealed'/name
        study.Campaign(path).verify_frozen()
        source = (path/'composed.py').read_text()
        if study.host.digest(source) != checksum:
            raise ValueError('legacy adapter requires the exact frozen program')
        result.append(source)
    return tuple(result)


def select_boundaries(room, ranks):
    panel = 'dev' if room == CONTROL else 'confirm'
    data = study.read(study.OUT/'runs'/panel/'P'/'full'/(room+'.json'))
    actions = [s for s in data['steps'] if s['tick'] > 0 and not s['pending']
               and s['action']['target_id'] == s['target_id']
               and min(s['action']['a0'], s['action']['a1']) < 32]
    if room == CONTROL:
        chosen = [(None, next(s for s in actions if s['tick'] == 50))]
        missing = []
    else:
        chosen = [(n, actions[n-1]) for n in ranks if n <= len(actions)]
        missing = [n for n in ranks if n > len(actions)]
    history = {b['tick']: b['public_history'] for b in data['boundaries']}
    return {s['tick']:dict(rank=n, action=s['action'], history=history[s['tick']])
            for n, s in chosen}, missing


def snapshot_name(room, tick):
    return f'{room}_t{tick:04d}.json'


class PrefixComplete(Exception):
    """Internal stop after collecting every selected boundary, before the suffix."""


def capture(job):
    room, fixture, old_source, safe_source, targets, out = job
    original = study.host.old.prefix.run_member
    safe_worker = StreamProgram(safe_source)
    shadow = study.ISO.IsolatedCAP(safe_worker)
    seen = []

    def with_shadow(job, member, preview, boundary, **kwargs):
        class Both:
            def update(self, history):
                member.update(history)
                shadow.update(history)

            def propose(self, obs):
                return member.propose(obs)

        def collect(world, obs, action, pending):
            issued = boundary(world, obs, action, pending)
            t = obs['tick']
            if t in targets:
                target = targets[t]
                if pending or issued != target['action'] or world.history.sha256 != target['history']:
                    raise ValueError('frozen replay does not reproduce the original public prefix')
                action = shadow.propose(obs)
                if action != shadow.memory['permit']['command'] or shadow.memory['permit']['tick'] != t:
                    raise ValueError('shadow did not bind its permit to this boundary')
                snapshot = dict(room=room, tick=t, rank=target['rank'], observation=obs,
                                observation_sha256=digest(obs), memory=shadow.memory, action=action,
                                history_sha256=world.history.sha256, source_sha256=forks.FINAL,
                                composition=COMPOSITION, prefix_composition=BASE_COMPOSITION)
                if room == CONTROL:
                    anchor = study.read(forks.OUT/'shadow'/(room+'.json'))['first']
                    if snapshot['observation_sha256'] != anchor['observation'] or action != anchor['new_action']:
                        raise ValueError('positive control did not reproduce the earlier diagnostic')
                immutable(out/'snapshots'/snapshot_name(room,t), snapshot)
                seen.append(t)
                if len(seen) == len(targets):
                    raise PrefixComplete
            return issued
        return original(job, Both(), preview, collect, **kwargs)

    try:
        if targets:
            with patch.object(study.host.old.prefix, 'run_member', with_shadow):
                try:
                    study.host.episode((room, fixture, True, None, 'boundary', False,
                                        study.host.enabled(old_source), None))
                except PrefixComplete:
                    pass
    finally:
        safe_worker.close()
    if sorted(seen) != sorted(targets):
        raise ValueError(f'missing public prefixes in {room}')
    return room, seen


def capture_panel(plan, out, workers, base, safe):
    fixtures = study.read(study.OUT/'fixtures.json')
    fixtures[CONTROL] = study.dev_rooms()[CONTROL]
    rooms = [CONTROL, *(k for keys in plan['strata'].values() for k in keys)]
    jobs, missing, expected = [], {}, {}
    for room in rooms:
        targets, missing[room] = select_boundaries(room, plan['push_ranks'])
        expected[room] = sorted(targets)
        jobs.append((room, fixtures[room], base, safe, targets, out))
    study.host.old.collect(base, jobs, capture, workers, out/'capture_rooms')
    for room in rooms:
        for tick in expected[room]:
            if not (out/'snapshots'/snapshot_name(room,tick)).is_file():
                raise ValueError(f'missing snapshot for {room} at {tick}')
    immutable(out/'prefixes.json', dict(ticks=expected, missing_ranks=missing))


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('audited frozen implementation is incompatible with this adapter')
    return source.replace(before, after)


def observer_source(source):
    """Preserve the controlling gate. Traverse it normally and once per forced public pair."""
    if study.host.digest(source) != forks.FINAL:
        raise ValueError('legacy attribution supports only the registered trial03')
    src = source[:source.index('def update_run(')]
    src = replace_once(src, 'def available_pushes(memory,o,scene,nominal):',
        """TRACK = {}
def available_pushes(memory,o,scene,nominal):
  if TRACK['only'] is not None:
    a,b = TRACK['only']
    yield make(o['goal']['target_id'],a,b,40)
    return""")
    src = replace_once(src, '  for act in available_pushes(memory,o,scene,proposals):',
        """  for act in available_pushes(memory,o,scene,proposals):
    TRACK['proposed'].append([act['a0'],act['a1']])""")
    src = replace_once(src, '      q=joint_points(f0,f1)',
        """      q=joint_points(f0,f1)
      if TRACK['only'] is None and not q and dot(f0[0],f1[0])>.97 and abs(f0[2]-f1[2])<.04:
        span=max(0.,min(f0[4],f1[4])-max(f0[3],f1[3])-.08)
        key=[a,b,span]
        if key not in TRACK['widths']:
          TRACK['widths'].append(key)""")
    src = replace_once(src, '      future_peer = shift(peer,delta,1.) if peer_drives else peer',
        """      future_peer = shift(peer,delta,1.) if peer_drives else peer
      who=1-i
      TRACK['peers'][who]=1 if peer_drives else 0
      if (all(.222<station[k]<arena[k]-.222 for k in (0,1)) and
          mag(sub(station,future_peer))<.325):
        TRACK['hits'][who]=1""")
    src = replace_once(src, '  base = proposals[0]', "  TRACK['last_reason']=reason\n  base = proposals[0]")
    return src + """
def update_memory(memory,event):
  o=event['observation']
  def inspect(pair):
    TRACK.update(only=pair,proposed=[],widths=[],peers=[-1,-1],hits=[0,0],last_reason=None)
    packet=candidate_packet(owned(memory),o)
    return packet
  if event['batch'] is None:
    packet=inspect(None)
    return dict(packet=packet,proposed=TRACK['proposed'],widths=TRACK['widths'])
  rows=[]
  for pair in event['batch']:
    packet=inspect(pair)
    passed=packet['decision']=='permit_push'
    reason=packet['reason'] if passed else TRACK['last_reason'] or packet['reason']
    rows.append([pair,passed,reason,TRACK['peers'],TRACK['hits']])
  return dict(rows=rows)
def prepare_context(m,o):
  return dict(values=[0],state=[None]*5,proposals=[])
"""


def inspect_snapshot(worker, snapshot):
    obs, memory = snapshot['observation'], snapshot['memory']
    goal = obs['goal']['target_id']
    menu = [p for p in obs['entity_legal_pairs'].get(goal, []) if min(p) < 32]

    def request(batch):
        return worker.request(dict(op='advance', memory=memory,
                                   event=dict(observation=obs, batch=batch)))['memory']

    observed = request(None)
    if observed['packet'] != memory['permit'] or observed['packet']['command'] != snapshot['action']:
        raise ValueError('instrumentation disagrees with the original complete decision')
    proposed = []
    for pair in observed['proposed']:
        if pair not in proposed:
            proposed.append(pair)
    selected = None
    if observed['packet']['decision'] == 'permit_push':
        action = snapshot['action']
        selected = [action['a0'], action['a1']]
    reviews, predictions, widths = [], [], []
    omitted_widths = 0
    for a, b, span in observed['widths']:
        if [a,b] not in menu:
            omitted_widths += 1
            continue
        widths.append(dict(pair=[a,b], source='visible_overlap', lower_m=span, upper_m=None,
                           required_m=.42, conclusion='unknown', pruned=True))
    for offset in range(0,len(menu),16):
        for pair, passed, reason, models, hits in request(menu[offset:offset+16])['rows']:
            active = [i for i, model in enumerate(models) if model >= 0]
            reviews.append(dict(pair=pair, verdict='pass' if passed else 'reject',
                                reason=reason, peer_robots=active))
            for i in active:
                predictions.append(dict(pair=pair, robot=i, model='co_moving' if models[i] else 'stationary',
                                        scope='hypothesis', used_to_veto=bool(
                                            pair in proposed and not passed and hits[i])))
    packet = dict(version=1, proposed=proposed, claim='selected' if selected else 'abstain',
                  selected=selected, reviews=reviews, widths=widths, peer_predictions=predictions)
    independently_passed = {tuple(r['pair']) for r in reviews if r['verdict'] == 'pass'}
    visited_passes = [pair for pair in proposed if tuple(pair) in independently_passed]
    if visited_passes != ([selected] if selected is not None else []):
        raise ValueError('forced menu review disagrees with the real first-passing decision loop')
    return dict(observation=obs, menu=menu, action=snapshot['action'], packet=packet, selected=selected,
                metadata=dict(room=snapshot['room'], tick=snapshot['tick'], rank=snapshot['rank'],
                              composition=COMPOSITION, prefix_composition=BASE_COMPOSITION,
                              observation_sha256=snapshot['observation_sha256'],
                              history_sha256=snapshot['history_sha256'], source_sha256=forks.FINAL,
                              width_calls_outside_public_push_menu=omitted_widths))


def extract_panel(out, source):
    source = observer_source(source)
    manifest = dict(policy_sha256=forks.FINAL, observer_sha256=study.host.digest(source),
                    composition=COMPOSITION, prefix_composition=BASE_COMPOSITION, checker_version=1,
                    meaning_of_reject='current implemented PUSH gate declines; no assertion of physical impossibility',
                    width='logged only at the same-face pruning site reached in nominal or alternative generation; lower bound is shared scan interval after the two end margins, floored at zero',
                    peer='logged only where the endpoint loop predicted the teammate pose; used_to_veto requires a tested-and-declined action where at least one sampled endpoint actually crossed the peer threshold',
                    omitted='early returns may leave passing actions unvisited when another push was selected',
                    units='candidate identity is target PUSH actuator pair; all speeds retained; time window capped after admission by frozen program')
    immutable(out/'adapter.json',manifest)
    compile_source(source)
    worker = StreamProgram(source)
    try:
        counts = {}
        for room, ticks in study.read(out/'prefixes.json')['ticks'].items():
            for tick in ticks:
                filename = snapshot_name(room,tick)
                snapshot = study.read(out/'snapshots'/filename)
                record = inspect_snapshot(worker,snapshot)
                immutable(out/'packets'/filename, record)
                counts[filename] = len(record['menu'])
        print(f'{len(counts)} prefixes, {sum(counts.values())} menu actions checked',flush=True)
    finally:
        worker.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    plan = selection(args.out)
    base, safe = sources()
    capture_panel(plan, args.out, args.workers, base, safe)
    extract_panel(args.out, safe)


if __name__ == '__main__':
    main()
