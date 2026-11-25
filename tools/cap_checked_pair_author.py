"""Matched public author packets, metered development trials, and checked v2 freezes."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import shutil
import time
from unittest.mock import patch

from experiments.planar_tasks.authoring_v2.checked_submission import submit_checked, verify_checked
from tools import cap_safe_nominal_study as study
from tools.cap_checked_pair_author_scaffold import PARENT, scaffold, submission_text

OUT = study.OUT.parent / 'cap_checked_pair_v1' / 'authoring_verified'
ARMS = ('control', 'checker')
LIMITS = dict(wall_seconds=1200, max_tickets=3, rooms_per_ticket=16, workers=8)
read, save, sha = study.read, study.save, study.sha
ROOT = study.ROOT
PUBLIC = OUT / 'authors'
HISTORY = study.OUT / 'sealed/trial03'
GUIDE = ROOT / 'experiments/planar_tasks/authoring_v2'
PUBLIC_STEPS = ('tick', 'target_id', 'active_target', 'goal', 'seen', 'xy', 'contact', 'static_contact',
                'execution', 'action', 'pending', 'direction', 'origin', 'permit', 'compact',
                'committed_permit', 'segments', 'end_tick')


def checked_paths():
    files = sorted((OUT/'development_inputs').glob('*.json'))
    if len(files) != 3:
        raise ValueError('expected three host development boundaries')
    return files


def check_receipt(path):
    receipt = verify_checked(path)
    expected = [read(p) for p in checked_paths()]
    actual = [read(path / (row['case']+'.input.json')) for row in receipt['cases']]
    if actual != expected:
        raise ValueError('wrong host-selected development boundaries')
    return receipt


def campaign(path, checked, arm):
    receipt = check_receipt(checked)
    source = checked/'policy.py'
    if not path.exists():
        camp = study.Campaign.create(path,study.dev_rooms())
        with camp.locked():
            camp.state['model'].update(name='gpt-6.2-sol',provider='codex',arm=arm)
            camp.state['author_protocol'].update(version='paired-planar-checker-v1', **LIMITS)
            camp.state['checked_submission'] = dict(source_sha256=receipt['source_sha256'],
                                                     manifest_sha256=sha(checked/'manifest.json'))
            camp._save()
        shutil.copyfile(source,path/'composed.py')
        camp.freeze()
    study.Campaign(path).verify_frozen()
    if sha(path/'composed.py') != receipt['source_sha256']:
        raise ValueError('checked source differs from campaign')
    return source.read_text()


def public_record(row):
    return dict(result=row['result'], boundaries=row['boundaries'], reviews=row['reviews'],
                steps=[{k:step[k] for k in PUBLIC_STEPS} for step in row['steps']])


def capture(room):
    historical = study.read(study.OUT/'locked.json')['parent_path'] if 'parent_path' in study.read(study.OUT/'locked.json') else study.ORIGIN/'composed.py'
    src = Path(historical).read_text()
    if study.host.digest(src) != study.BASE_SHA:
        raise ValueError('unexpected original prefix generator')
    fixture = study.dev_rooms()[room]
    study.ISO.initialize(src)
    events = []
    original_update = study.ISO.IsolatedCAP.update
    class Complete(Exception):
        pass
    def recording(member, history):
        events.append(history.events[0])
        original_update(member,history)
    class AtFirstPush:
        def boundary(self,obs,memory,action,pending):
            if (not pending and obs['tick']>0 and min(action['a0'],action['a1'])<32 and
                action['target_id']==obs['goal']['target_id']):
                raise Complete()
        def checkpoint(self,*args):
            pass
    try:
        with patch.object(study.ISO.IsolatedCAP,'update',recording):
            try:
                study.host.episode((room,fixture,False,None,'boundary',False,study.host.enabled(src),None),AtFirstPush())
            except Complete:
                return dict(events=events)
    finally:
        study.ISO.BROKER.close()
    raise ValueError('preselected development room did not reach a noninitial PUSH '+room)


def write_effect_report(folder, name):
    from tools.cap_checked_pair_feedback import write_public_checks
    write_public_checks(folder, name)


def activate():
    if list((OUT/'ledger').glob('*/clock.json')):
        raise ValueError('authoring already started')
    if not read(OUT/'conformance/result.json')['all_match']:
        raise ValueError('parent conformance required before activation')
    folder = PUBLIC/'checker'
    write_effect_report(folder, 'parent')
    for arm in ARMS:
        (PUBLIC/arm/'README.md').write_text(instructions(arm))
    files = [Path(__file__), ROOT/'tools/cap_checked_pair_author_scaffold.py',
             ROOT/'tools/cap_checked_pair_feedback.py']
    return study.immutable(OUT/'activation.json', dict(
        protocol_sha256=sha(OUT/'protocol.json'),
        conformance_sha256=sha(OUT/'conformance/result.json'),
        diagnostic_feedback='individualized finite-menu review and public trajectory checks of expressed effects and continuation; all official receipts use the same three ordinary development boundaries',
        tools={str(p.relative_to(ROOT)):sha(p) for p in files},
        note='Initial harness compatibility smoke excluded. Current host reports only public information. Clocks start independently at service activation.'))


def public_diagnostics(folder, name, checked):
    report = []
    for i,path in enumerate(checked_paths()):
        row = read(checked/f'{i:04d}.report.json')
        report.append(dict(prefix=path.stem, status=row['status'],counts=row['counts'],
                           findings=row['findings'], omitted_passing=row['omitted_passing']))
    save(folder/'checks'/f'{name}.json',report)


def init():
    study.Campaign(HISTORY).verify_frozen()
    source = (HISTORY/'composed.py').read_text()
    text = scaffold(source)
    nominated = submission_text(text)
    boundaries = OUT/'development_inputs'
    boundaries.mkdir(parents=True,exist_ok=True)
    for i in (1,3):
        room = f'diag_dev_nat_{i:03d}'
        snap = boundaries/(room+'_first_noninitial_push.json')
        if not snap.exists():
            save(snap,capture(room))
    initial = boundaries/'diag_dev_nat_002_initial.json'
    if not initial.exists():
        with study.S.phase.world(study.dev_rooms()['diag_dev_nat_002']) as (world, _):
            save(initial,dict(observation=world.history.events[0]['observation']))
    checked = OUT/'submissions/parent'
    source_path = OUT/'parent_source.py'
    if source_path.exists() and source_path.read_text()!=nominated:
        raise ValueError('parent scaffold changed since initialization')
    source_path.write_text(nominated)
    if not checked.exists():
        submit_checked(source_path,checked_paths(),checked)
    campaign(OUT/'sealed/parent',checked,'common_parent')
    for arm in ARMS:
        folder = PUBLIC/arm
        folder.mkdir(parents=True,exist_ok=True)
        (folder/'requests').mkdir(exist_ok=True)
        (folder/'P.py').write_text(text)
        shutil.copytree(study.PUBLIC/'knowledge',folder/'knowledge',dirs_exist_ok=True)
        for name in ('checked_policy_guide.md','checked_policy_example.py'):
            shutil.copyfile(GUIDE/name,folder/'knowledge'/name)
        shutil.copytree(study.PUBLIC/'feedback/trial03/full',folder/'feedback/parent',dirs_exist_ok=True)
        shutil.copytree(boundaries,folder/'public_prefixes',dirs_exist_ok=True)
        save(folder/'rooms.json',list(study.dev_rooms()))
        (folder/'README.md').write_text(instructions(arm))
        if arm=='checker':
            public_diagnostics(folder,'parent',checked)
            write_effect_report(folder, 'parent')
    manifest=dict(parent_historical=PARENT,parent_checked=sha(checked/'policy.py'),human_scaffold=study.host.digest(text),
                  dev_rooms={k:study.content_hash(v) for k,v in study.dev_rooms().items()},
                  host_boundaries={p.name:sha(p) for p in checked_paths()}, budgets=LIMITS,
                  shared_knowledge={p.name:sha(p) for p in (PUBLIC/'control/knowledge').glob('*') if p.is_file()},
                  feedback='public per-room outcomes and ordinary/event logs for both; only checker receives personalized pre-final reports',
                  author_visibility='arm-local packet by instruction; isolated native authored policy worker; author browsing is not OS isolated unless the invoking transport enforces it')
    study.immutable(OUT/'protocol.json',manifest)
    return manifest


def instructions(arm):
    return """Improve this two-robot general pushing program's success in new rooms from the same distribution.
Preserve the public-only information boundary, deterministic bounded planner, 72-ray local-centre convention,
event previews and action codebook. Develop whichever algorithm you judge useful. Read knowledge/ including
the required checked policy guide/example. Keep the three hooks and truthful reports linked to the real
decision gate. P.py is a behavior-preserving API refactor of the conservative parent (human-readable).
feedback/parent and public_prefixes/ contain real development
examples. No other directories, rooms, agents, external tools or previous study may be consulted. Read and
write files only in this author directory. Do not run simulations/compiler/host yourself. The host checks
submission, compiles via bubblewrap and returns bounded trial results. Source is packed by redundant whitespace
only for the 48KiB carrier and must still fit; the AST must be identical.
Each independently timed author has 20 minutes after activation, at most three immutable source tickets,
and at most 16 distinct rooms per ticket (48 total). To submit a version write trial01.py, trial02.py or
trial03.py (full program), THEN atomically publish requests/trial01.json or corresponding number, e.g.
{"rooms":["diag_dev_nat_001","diag_dev_nat_003"], "intent":"What this changes and why"}.
Omit rooms to run the complete 16 rooms. A ticket is consumed even on a rejected source. Each ticket is run once.
Observe replies/trialNN.json and feedback/trialNN/room.json; editing a submitted source does not change its seal.
Record your first concrete diagnosis and later discoveries in worklog.jsonl (time, evidence, proposed change).
For final nomination write FINAL.json atomically with {"name":"trial01","reason":"..."}. The named run must have
been received and passed the common submission. A nomination of the verified unchanged parent ("parent") is
also valid. Early finish is fine. No new source is accepted after deadline. The host owns evaluation.
""" + ('You receive personalized menu/claim diagnostics in checks/parent.json and checks/trialNN.json, plus the measured outcome/continuation reports in checks/.\n' if arm=='checker' else
       'You receive public trajectories, outcomes and the common submission gate status for every attempted version.\n')


def conformance():
    source = campaign(OUT/'sealed/parent', OUT/'submissions/parent','common_parent')
    fixtures = study.dev_rooms()
    jobs = [(k,v,study.host.enabled(source)) for k,v in fixtures.items()]
    dest = OUT/'conformance/runs'
    study.host.old.collect(source,jobs,study.episode,LIMITS['workers'],dest)
    rows={}
    for key in fixtures:
        previous=read(study.PUBLIC/'feedback/trial03/full'/(key+'.json'))
        current=public_record(read(dest/(key+'.json')))
        rows[key]=dict(result_equal=current['result']==previous['result'],
                       commands_equal=current['boundaries']==previous['boundaries'],
                       permits_and_events_equal=current['steps']==previous['steps'] and current['reviews']==previous['reviews'],
                       ticks=current['result']['ticks'], success=current['result']['success'])
    result=dict(rooms=rows,all_match=all(all(row[k] for k in ('result_equal','commands_equal','permits_and_events_equal')) for row in rows.values()))
    save(OUT/'conformance/result.json',result)
    if not result['all_match']:
        raise ValueError('historical parent conformance mismatch')
    return result


def trial(arm, name, request):
    folder = PUBLIC/arm
    account = OUT/'ledger'/arm
    path = folder/(name+'.py')
    if path.is_symlink() or not path.is_file() or path.stat().st_size>150000:
        raise ValueError('expected ordinary bounded candidate source')
    if not isinstance(request,dict) or set(request)-{'rooms','intent'}:
        raise ValueError('unsupported trial request fields')
    fixtures = study.dev_rooms()
    keys = request.get('rooms',list(fixtures))
    if (not isinstance(keys,list) or not 1<=len(keys)<=16 or len(set(keys))!=len(keys)
            or any(k not in fixtures for k in keys)):
        raise ValueError('rooms must be a unique development subset of size 1..16')
    src = path.read_text()
    ticket = account/name
    ticket.mkdir(exist_ok=False)
    (ticket/'readable.py').write_text(src)
    save(ticket/'request.json',dict(request=request,registered=time.time(),source_sha256=study.host.digest(src),
                                    rooms=keys,reserved_episodes=len(keys)))
    packed = submission_text(src)
    file = ticket/'submitted.py'
    file.write_text(packed)
    checked = OUT/'submissions'/arm/name
    submit_checked(file,checked_paths(),checked)
    source = campaign(OUT/'sealed'/arm/name,checked,arm)
    if arm == 'checker':
        public_diagnostics(folder,name,checked)
    reply = dict(status='running',accepted=True,rooms=keys,started=time.time(),source_sha256=sha(checked/'policy.py'))
    save(folder/'replies'/(name+'.json'), reply)
    dest = OUT/'dev_runs'/arm/name
    jobs = [(key,fixtures[key],study.host.enabled(source)) for key in keys]
    study.host.old.collect(source,jobs,study.episode,min(LIMITS['workers'],len(keys)),dest)
    results = {}
    deadline=read(account/'clock.json')['deadline']
    for key in keys:
        record = public_record(read(dest/(key+'.json')))
        results[key]=record['result']
        if time.time()<=deadline:
            save(folder/'feedback'/name/(key+'.json'),record)
    if arm == 'checker' and time.time() <= deadline:
        write_effect_report(folder, name)
    reply.update(status='complete',finished=time.time(),success=sum(r['success'] for r in results.values()),results=results)
    save(ticket/'result.json',reply)
    return reply


def serve(arm):
    folder = PUBLIC/arm
    account = OUT/'ledger'/arm
    account.mkdir(parents=True,exist_ok=True)
    with (account/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        activation=read(OUT/'activation.json')
        if activation['protocol_sha256']!=sha(OUT/'protocol.json'):
            raise ValueError('activated protocol mismatch')
        manifest=read(OUT/'protocol.json')
        if not (folder/'README.md').exists():
            raise ValueError('public packet is missing')
        clock_path=account/'clock.json'
        if not clock_path.exists():
            now=time.time()
            save(clock_path,dict(start=now,deadline=now+manifest['budgets']['wall_seconds']))
        clock=read(clock_path)
        save(folder/'clock.json',clock)
        while True:
            final=folder/'FINAL.json'
            if final.is_file() and final.stat().st_mtime<=clock['deadline']:
                note=read(final)
                name=note.get('name')
                if name not in ('parent','trial01','trial02','trial03'):
                    save(folder/'final_status.json',dict(status='rejected',reason='invalid nomination'))
                else:
                    candidate=Path('parent') if name=='parent' else Path(arm)/name
                    checked=OUT/'submissions'/candidate
                    camp=OUT/'sealed'/candidate
                    if checked.exists() and camp.exists():
                        campaign(camp,checked,arm)
                        result=dict(arm=arm,name=name,source_sha256=sha(checked/'policy.py'),
                                    checked=str(checked),campaign=str(camp),nomination=note,
                                    received=time.time(),nomination_mtime=final.stat().st_mtime,
                                    campaign_manifest_sha256=sha(camp/'manifest.json'))
                        study.immutable(OUT/'frozen'/f'{arm}.json',result)
                        save(folder/'final_status.json',dict(status='frozen',name=name))
                        return result
                    save(folder/'final_status.json',dict(status='unaccepted',name=name))
            if time.time()>clock['deadline']:
                save(account/'closed.json',dict(status='incomplete',reason='no timely accepted nomination'))
                return read(account/'closed.json')
            for request_path in sorted((folder/'requests').glob('trial*.json')):
                name=request_path.stem
                if request_path.is_symlink() or not re.fullmatch('trial0[1-3]',name) or (account/f'{name}.claimed.json').exists():
                    continue
                if request_path.stat().st_mtime>clock['deadline'] or time.time()>clock['deadline']:
                    continue
                save(account/f'{name}.claimed.json',dict(registered=time.time()))
                try:
                    answer=trial(arm,name,read(request_path))
                except Exception as exc:
                    answer=dict(status='error',message=str(exc),finished=time.time())
                    save(account/f'{name}.error.json',answer)
                if time.time()<=clock['deadline']:
                    save(folder/'replies'/(name+'.json'),answer)
            time.sleep(.5)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    subs=cli.add_subparsers(dest='verb',required=True)
    subs.add_parser('init')
    subs.add_parser('conformance')
    subs.add_parser('activate')
    run=subs.add_parser('serve')
    run.add_argument('arm',choices=ARMS)
    args=cli.parse_args()
    result=serve(args.arm) if args.verb=='serve' else {'init':init,'conformance':conformance,'activate':activate}[args.verb]()
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
