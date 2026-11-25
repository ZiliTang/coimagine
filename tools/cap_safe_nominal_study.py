"""Bounded development and paired blind evaluation of the diagonal nominal CAP."""
import argparse
from collections import Counter
from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from unittest.mock import patch

from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.relational_search.motion import ACTIONS, CODEBOOK_HASH
from tools import cap_pr_graph_round as prior
from tools import cap_open_recovery_round as host
from tools.cap_entry_residual_study import immutable
from tools.cap_fault_curriculum_rooms import panel, SEEDS, STRATA, content_hash, VERSION

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/semantic_rl/cap_safe_nominal_v1'
PUBLIC = OUT / 'author'
ORIGIN = OUT.parent / 'cap_free_repair_20261124_01/sealed/trial999'
HOLDOUT = 'safe_nominal_v1_confirm'
SEED = 781353107
BASE_SHA = '660e2f3a1a13eff1abb8126c48435c277364e8b6576dac97cc6b750ed893755b'
BUDGET = dict(wall_seconds=1200, max_candidates=3, rooms_per_version=16)
read, save = host.read, host.save
Campaign = host.old.author.host.RoomCampaign
S = host.old.prefix.s
ISO = host.old.isolated


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@contextmanager
def lock(name):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / ('.' + name + '.lock')).open('a') as file:
        fcntl.flock(file, fcntl.LOCK_EX)
        yield


def dev_rooms():
    pool = prior.dev_pool()
    keys = read(PUBLIC / 'rooms.json')
    if len(keys) != 16 or len(set(keys)) != 16:
        raise ValueError('expected 16 distinct development room IDs')
    return {key: pool[key] for key in keys}


def known_rooms():
    paths = [*sorted(OUT.parent.glob('cap_*/fixtures.json')),
             *sorted(prior.prior.PREV.glob('fixtures/*.json')),
             host.old.study.OUT / 'rooms.json']
    known_ids, known_content, inventory = set(), set(), {}
    for path in paths:
        if path.parent == OUT:
            continue
        obj = read(path)
        if 'records' in obj:
            records = {k: v['content_sha256'] for k, v in obj['records'].items()}
        else:
            records = {k: content_hash(v[1] if isinstance(v, list) else v) for k, v in obj.items()}
        inventory[str(path.relative_to(ROOT))] = dict(count=len(records), sha256=sha(path))
        known_ids.update(records)
        known_content.update(records.values())
    return known_ids, known_content, inventory


def init():
    with lock('init'):
        dev = dev_rooms()
        Campaign(ORIGIN).verify_frozen()
        if sha(ORIGIN/'composed.py') != BASE_SHA or sha(PUBLIC/'P.py') != BASE_SHA:
            raise ValueError('parent source differs from the frozen trial999')
        # The public room list was written when the author packet was opened.
        opened = (PUBLIC / 'rooms.json').stat().st_mtime
        immutable(OUT/'budget.json', dict(**BUDGET, start=opened, deadline=opened+1200,
                  start_evidence='author/rooms.json creation mtime'))
        SEEDS[HOLDOUT] = SEED
        fixtures, records = panel(HOLDOUT, 64)
        ids, contents, inventory = known_rooms()
        ids.update(dev)
        contents.update(content_hash(f) for f in dev.values())
        hashes = {v['content_sha256'] for v in records.values()}
        if len(hashes) != 64 or hashes & contents or set(fixtures) & ids:
            raise ValueError('blind room identity/content overlap')
        strata = dict(Counter(r['stratum'] for r in records.values()))
        if strata != {s:16 for s in STRATA}:
            raise ValueError('unbalanced holdout')
        immutable(OUT/'fixtures.json', fixtures)
        immutable(OUT/'room_records.json', records)
        immutable(OUT/'plan.json', dict(parent=BASE_SHA, parent_path=str(ORIGIN/'composed.py'),
                  budgets=BUDGET, panel=HOLDOUT, seed=SEED, generator=VERSION, strata=strata,
                  source='frozen full-start, fixed72-local-v1, split_precision, normal and PUSH progress callbacks',
                  development={k:content_hash(f) for k,f in dev.items()},
                  fixture_sha256=sha(OUT/'fixtures.json'), record_sha256=sha(OUT/'room_records.json'),
                  excluded_panels=inventory, excluded_unique_ids=len(ids), excluded_unique_contents=len(contents),
                  codebook_sha256=CODEBOOK_HASH, max_ticks=S.MAX_TICKS, max_decisions=S.MAX_DECISIONS,
                  author_knowledge={p.name:sha(p) for p in sorted((PUBLIC/'knowledge').glob('*')) if p.is_file()}))
        if not (OUT/'git_init.json').exists():
            save(OUT/'git_init.json', git_state())


def git_state():
    def git(*args):
        return subprocess.check_output(['git',*args],cwd=ROOT,text=True).rstrip()
    return dict(head=git('rev-parse','HEAD'), status=git('status','--short').splitlines())


def runtime_sources():
    files = ['tools/cap_safe_nominal_study.py','tools/cap_open_recovery_round.py',
             'tools/cap_cluster_module.py','tools/cap_fault_curriculum_isolated.py',
             'tools/cap_fault_curriculum_prefix.py','tools/cap_option_semantic_factorial.py',
             'tools/cap_abc_event_ppo.py','tools/cap_fault_curriculum_rooms.py',
             'experiments/planar_tasks/authoring_v2/random_room_author.py',
             'experiments/planar_tasks/relational_search/environment.py',
             'experiments/planar_tasks/relational_search/motion.py',
             'experiments/planar_tasks/semantic_rl/stream_features.py',
             'experiments/planar_tasks/semantic_rl/typed_compiler.py',
             'experiments/planar_tasks/semantic_rl/environment.py',
             'experiments/planar_tasks/semantic_rl/phase_aware_cap.py',
             'experiments/planar_tasks/semantic_rl/push_phases.py',
             'experiments/planar_tasks/semantic_rl/push_takeover.py',
             'experiments/planar_tasks/semantic_rl/duration_execution.py']
    return {f:sha(ROOT/f) for f in files}


def seal(name, *, final=False):
    if not re.fullmatch('P|trial[0-9]{2,3}',name):
        raise ValueError('expected P or trialNN')
    srcfile = PUBLIC / (name+'.py')
    source = srcfile.read_text()
    checksum = host.digest(source)
    path = OUT/'sealed'/name
    with lock('submissions'):
        if name == 'P' and checksum != BASE_SHA:
            raise ValueError('modified parent')
        budget = read(OUT/'budget.json')
        ledger_file = OUT/'submissions.json'
        ledger = read(ledger_file) if ledger_file.exists() else {}
        if name != 'P':
            if name in ledger and ledger[name]['sha256'] != checksum:
                raise ValueError('submitted version was edited')
            if name not in ledger:
                if len(ledger) >= budget['max_candidates']:
                    raise ValueError('three author candidates already submitted')
                if srcfile.stat().st_mtime > budget['deadline'] or (not final and time.time() > budget['deadline']):
                    raise ValueError('author submission deadline exceeded')
                ledger[name] = dict(sha256=checksum, source_mtime=srcfile.stat().st_mtime, registered=time.time())
                save(ledger_file, ledger)
        if not path.exists():
            campaign = Campaign.create(path,dev_rooms())
            with campaign.locked():
                campaign.state['model'].update(name='gpt-6.2-sol',provider='collaboration')
                campaign.state['author_protocol'].update(version='safe-nominal-v1',**BUDGET)
                campaign.state['deadline'] = budget['deadline']
                campaign._save()
            (path/'composed.py').write_text(source)
            campaign.freeze()
        Campaign(path).verify_frozen()
        if sha(path/'composed.py') != checksum:
            raise ValueError('frozen program changed')
    return source


def truth(world):
    backend = world.env.env.backend
    target = world.env.primary_id
    body = backend.bodies[target]
    contacts = backend.contacts()
    return dict(object=[*body.position,body.angle], object_static=any(
        c['pair']=='object_wall' and c.get('entity_id')==target for c in contacts))


def public_state(obs):
    target = obs['goal']['target_id']
    return dict(tick=obs['tick'], target_id=target, active_target=obs['active_target'], goal=obs['goal']['xy'],
                seen=any(e['id']==target and e['points'] and e['last_seen_tick']==obs['tick'] for e in obs['entities']),
                xy=[[r['own']['x'],r['own']['y']] for r in obs['robots']],
                contact=[r['contacting_object'] for r in obs['robots']],
                static_contact=[r['contacting_static'] for r in obs['robots']],
                execution=[{k:r['execution'].get(k) for k in ('stage','direction_valid','direction_unavailable')} for r in obs['robots']])


def directions(action):
    return [getattr(ACTIONS[action[k]], 'effect_direction', None) for k in ('a0','a1')]


def episode(job):
    room, fixture, hooks = job
    steps, reviews = [], []
    pending_proposals = {}
    original_tick = S.tick_action
    original_request = ISO.BROKER.request

    def remember(obs, memory, suggested, running, kind, left=None):
        entry = dict(**public_state(obs), kind=kind, permit=memory.get('permit'), compact=memory.get('compact'),
                     proposal=suggested, running=running, remaining_ticks=left)
        reviews.append(entry)
        if suggested and not host.old.prefix.event.same_control(suggested,running):
            pending_proposals[obs['tick']] = entry

    def observed_request(request):
        response = original_request(request)
        event = request.get('event', {})
        review = event.get('push_checkpoint')
        if review is not None:
            suggestions = response['proposals']
            remember(event['observation'], response['memory'], suggestions[0] if suggestions else None,
                     event['action'], 'push_'+review['reason'], review['remaining_ticks'])
        return response

    class Audit:
        def boundary(self,obs,memory,action,pending):
            origin = pending_proposals.pop(obs['tick'],None) if pending else None
            if pending and (origin is None or origin['proposal']!=action):
                raise ValueError('pending command has no matching live preview')
            steps.append(dict(**public_state(obs), action=action, pending=pending,
                     direction=directions(action), origin=origin['kind'] if pending else 'ordinary',
                     permit=origin['permit'] if pending else memory.get('permit'),
                     compact=origin['compact'] if pending else memory.get('compact'),
                     committed_permit=memory.get('permit')))

        def checkpoint(self,obs,memory,suggested,running,kind):
            # The broker observer also records declines at these checkpoints.
            pass

    def preview(member,obs,action,elapsed):
        row=member.advance(dict(observation=obs,action=action,executed_ticks=elapsed))
        try:
            proposal=validate_action(row['proposals'][0],obs)
        except ValueError:
            proposal=None
        remember(obs,row['memory'],proposal,action,'single_push_review')
        return proposal

    def traced_tick(world, action, *args, **kwargs):
        row=steps[-1]
        row.update(truth=truth(world), segments=[], phase_truth=[])
        original_step=world.step
        def step(*a,**kw):
            before = truth(world)
            event=original_step(*a,**kw)
            row['segments'].append({k:event[k] for k in ('requested','executed','start_tick','end_tick','segments','handoffs','reason')})
            row['phase_truth'].append(dict(start_tick=event['start_tick'],truth=before,
                                          end_tick=event['end_tick'],end_truth=truth(world)))
            return event
        with patch.object(world,'step',step):
            result=original_tick(world,action,*args,**kwargs)
        row.update(end_tick=world.env.total_periods,end_truth=truth(world))
        return result

    with patch.object(S,'tick_action',traced_tick), patch.object(ISO,'preview',preview), \
            patch.object(ISO.BROKER,'request',observed_request):
        _, result=host.episode((room,fixture,False,None,'boundary',False,hooks,None),Audit())
    if len(steps)!=len(result['boundaries']) or steps[-1]['end_tick']!=result['result']['ticks']:
        raise ValueError('incomplete audit trace')
    return room, dict(result,audit_schema=2,steps=steps,reviews=reviews,
                      terminal=dict(tick=steps[-1]['end_tick'],truth=steps[-1]['end_truth']))


def run(name, phase, worker_count, keys=None):
    if worker_count < 1:
        raise ValueError('positive worker count required')
    source=seal(name,final=phase=='confirm')
    fixtures=dev_rooms() if phase=='dev' else read(OUT/'fixtures.json')
    keys=keys or list(fixtures)
    if len(keys)!=len(set(keys)) or any(k not in fixtures for k in keys):
        raise ValueError('room list must be a distinct subset of the declared panel')
    dest=OUT/'runs'/phase/name/'full'
    with lock('run-'+phase+'-'+name):
        if phase=='dev' and name!='P':
            if time.time()>read(OUT/'budget.json')['deadline'] or (OUT/'locked.json').exists():
                raise ValueError('author development window closed')
        jobs=[(k,fixtures[k],host.enabled(source)) for k in keys]
        host.old.collect(source,jobs,episode,min(len(keys),worker_count),dest)
        if phase=='dev' and (name=='P' or time.time()<=read(OUT/'budget.json')['deadline']):
            for k in keys:
                row=read(dest/(k+'.json'))
                # Explicit whitelist; no true poses, topology, fixture or holdout results.
                public_fields=('tick','target_id','active_target','goal','seen','xy','contact','static_contact',
                               'execution','action','pending','direction','origin','permit','compact',
                               'committed_permit','segments','end_tick')
                public_steps=[{field:step[field] for field in public_fields}
                              for step in row['steps']]
                save(PUBLIC/'feedback'/name/'full'/(k+'.json'),dict(result=row['result'],
                     boundaries=row['boundaries'],steps=public_steps,reviews=row['reviews']))
    return dest


def freeze(name=None):
    finalfile=PUBLIC/'FINAL.json'
    decision=read(finalfile)
    name=name or decision.get('name') or decision.get('selected')
    if name != (decision.get('name') or decision.get('selected')):
        raise ValueError('selection differs from author FINAL.json')
    if finalfile.stat().st_mtime>read(OUT/'budget.json')['deadline']:
        raise ValueError('final submitted after deadline')
    submissions = {p.stem:dict(sha256=sha(p),mtime=p.stat().st_mtime)
                   for p in PUBLIC.glob('trial*.py') if re.fullmatch('trial[0-9]{2,3}', p.stem)}
    if len(submissions)>BUDGET['max_candidates'] or any(
            sub['mtime']>read(OUT/'budget.json')['deadline'] for sub in submissions.values()):
        raise ValueError('authored versions outside the declared budget')
    source=seal(name,final=True)
    if host.digest(source)!=(decision.get('source_sha256') or decision.get('sha256')):
        raise ValueError('final source fingerprint mismatch')
    seal('P',final=True)
    data=dict(name=name, sha256=host.digest(source), parent=BASE_SHA, source='sealed/'+name+'/composed.py',
              final_sha256=sha(finalfile), authored_submissions=submissions, fixture_sha256=sha(OUT/'fixtures.json'),
              protocol_sha256=sha(OUT/'plan.json'),runtime_sources=runtime_sources(),
              campaigns={v:Campaign(OUT/'sealed'/v).verify_frozen() for v in {'P',name}})
    with lock('freeze'):
        immutable(OUT/'locked.json',data)
        if not (OUT/'git_freeze.json').exists():
            save(OUT/'git_freeze.json',git_state())
    return data


def evaluate(workers):
    with lock('evaluate'):
        return paired_evaluation(workers)


def paired_evaluation(workers):
    locked=read(OUT/'locked.json')
    if freeze(locked['name'])!=locked:
        raise ValueError('confirmation seal mismatch')
    plan=read(OUT/'plan.json')
    if sha(OUT/'fixtures.json')!=plan['fixture_sha256'] or sha(OUT/'room_records.json')!=plan['record_sha256']:
        raise ValueError('holdout changed')
    rows={}
    records=read(OUT/'room_records.json')
    status_path=OUT/'evaluation_status.json'
    status=dict(started=time.time(),workers=workers,expected_rooms=len(records),status='running',completed={})
    save(status_path,status)
    for label,name in [('parent','P'),('candidate',locked['name'])]:
        status['active']=name
        save(status_path,status)
        directory=run(name,'confirm',workers)
        for k,record in records.items():
            row=read(directory/(k+'.json'))
            if row.get('audit_schema')!=2 or row['terminal']['tick']!=row['result']['ticks']:
                raise ValueError('missing confirmed trace or terminal sample for '+k)
            rows.setdefault(k,dict(room=k,stratum=record['stratum'],content_sha256=record['content_sha256']))[label]=row['result']
        status['completed'][label]=len(records)
        save(status_path,status)
    if freeze(locked['name'])!=locked or sha(OUT/'room_records.json')!=plan['record_sha256']:
        raise ValueError('confirmation inputs changed during evaluation')
    summary=dict(rooms=len(rows),parent_success=sum(v['parent']['success'] for v in rows.values()),
                 candidate_success=sum(v['candidate']['success'] for v in rows.values()),
                 rescues=sum(v['candidate']['success']>v['parent']['success'] for v in rows.values()),
                 losses=sum(v['candidate']['success']<v['parent']['success'] for v in rows.values()))
    immutable(OUT/'paired_rooms.json',rows)
    immutable(OUT/'summary.json',summary)
    status.update(status='complete',active=None,finished=time.time(),summary_sha256=sha(OUT/'summary.json'))
    save(status_path,status)
    print(json.dumps(summary))


def cli():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='phase',required=True)
    commands.add_parser('init')
    dev=commands.add_parser('dev')
    dev.add_argument('name')
    dev.add_argument('--rooms',nargs='+')
    dev.add_argument('--workers',type=int,default=12)
    final=commands.add_parser('freeze'); final.add_argument('name',nargs='?')
    ev=commands.add_parser('evaluate'); ev.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.phase=='init': init()
    elif args.phase=='dev': print(run(args.name,'dev',args.workers,args.rooms))
    elif args.phase=='freeze': print(json.dumps(freeze(args.name),indent=2))
    else: evaluate(args.workers)

if __name__=='__main__': cli()
