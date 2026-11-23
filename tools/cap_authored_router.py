"""Meter a public A/B routing program and test it against frozen CAPs and graph."""
import ast
from collections import Counter, deque
import copy
import math
from pathlib import Path
import shutil
import sys
import time

from tools import cap_authored_handoff as e
from tools import cap_authored_handoff_graph as graph

OUT = e.OUT.parent / 'cap_authored_router_20261123_01'
DOC = e.DOC.parent / '2026-11-23_authored_router'
WORK = e.WORK.parent / 'router_ab_1123'
CAM = OUT / 'campaign'
SEED = 662316251
START = 1795414319.55
DEADLINE = START + 720
ARMS = ('A', 'B', 'lock', 'graph', 'router')


def setup():
    dev = e.dev()
    packet = e.read(WORK / 'dev.json')
    eligible = [row for row in packet if row['trigger']]
    rooms = sorted((r['room'] for r in eligible), key=lambda k: e.key('routerdev/' + k))[:24]
    assert len(rooms) == 24 and len(packet) == 176
    c = e.base.host.RoomCampaign.create(CAM, {k: dev[k] for k in rooms})
    with c.locked():
        c.state['model'].update(name='gpt-6.2-sol')
        c.state['deadline'] = DEADLINE
        c.state['limits'].update(attempts=72, resets=72, execution_ticks=129600, feature_batches=21672)
        c.state['author_protocol'].update(version='public-ab-router-v1', wall_seconds=720,
                                         prior='frozen A and B and 176 development rooms', max_revisions=3,
                                         max_library_policies=0,no_prior_semantics=False,offline_access='public guides, frozen member source, development packet and metered probes',
                                         final='select one immutable probed router in selection.json')
        c._save()
    e.save(CAM/'knowledge/transfer_execution_protocol.json', {k:c.state[k] for k in ('stage1','limits','author_protocol')})
    e.save(CAM / 'knowledge/router_protocol.json', dict(seed=SEED, arms=ARMS, primary='router vs A (best pure expert on 176 development rooms)', secondary='router vs graph, lock, B', gate='A until same first eligible stall as graph; both may choose A or B thereafter', development=rooms, start=START, deadline=DEADLINE, packet_sha=graph.file_sha(WORK/'dev.json')))
    DOC.mkdir(exist_ok=True, parents=True)


def initialize(*, profile=e.perception.DEFAULT):
    member = e.read(e.OUT/'member.json')
    assert e.base.host.RoomCampaign(e.CAM).verify_frozen() == member['seal']
    source = Path(member['path']).read_text()
    assert e.base.digest(source) == member['sha']
    version = e.init(source, profile=profile)
    graph.load_policy()
    return version


def load_router(source):
    tree = ast.parse(source)
    allowed = (ast.FunctionDef, ast.Import, ast.ImportFrom, ast.Assign, ast.Expr)
    if not all(isinstance(n, allowed) for n in tree.body):
        raise ValueError('router must be a deterministic function')
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(v.name != 'math' for v in node.names):
            raise ValueError('only math import is permitted')
        if isinstance(node, ast.ImportFrom) and (node.module != 'math' or node.level):
            raise ValueError('only math import is permitted')
        if isinstance(node, ast.Attribute) and node.attr.startswith('_'):
            raise ValueError('private attribute')
        if isinstance(node, ast.Name) and node.id.startswith('__'):
            raise ValueError('private identifier')
    import builtins
    permitted = 'abs all any bool dict enumerate float int len list max min range round set sorted str sum tuple zip isinstance reversed'
    env = {'__builtins__': {k:getattr(builtins,k) for k in permitted.split()}}
    env['__builtins__']['__import__'] = lambda name,*args: math if name=='math' else None
    exec(compile(tree,'frozen_router','exec'),env)
    return env['route']


def play(f, *, stall_gate=True):
    members = [e.s.phase.FrozenCAP(m) for m in (e.AM, e.BM)]
    hist, memory, counts = deque(maxlen=5), {}, Counter()
    hit = previous = None
    n = chosen_b = different = switches = 0
    examples = []
    with e.s.phase.world(f) as (w, _):
        def update(h):
            for cap in members:
                cap.update(h)
        update(w.history)
        while not e.s.ended(w):
            obs = e.s.observe(w)
            v = e.sample(obs,hist)
            if hit is None and e.eligible(v,n,'stall'):
                hit = dict(n=n,tick=v['tick'],h=w.history.sha256,z=v['z'])
            a,b = [cap.propose(obs) for cap in members]
            ctx = dict(v, n=n, triggered=hit is not None, previous=previous)
            who = ROUTE(memory, copy.deepcopy(obs), dict(a), dict(b), ctx)
            if who not in ('A','B'):
                raise ValueError('invalid choice')
            if stall_gate and hit is None:
                who = 'A'
            if who == 'B':
                chosen_b += 1
                different += int(a != b)
            switches += int(previous is not None and who != previous)
            if a != b and hit and len(examples)<10:
                examples.append(dict(tick=v['tick'],who=who,gap=v['gap'],touch=v['touch'],progress=v['progress'],a=a,b=b))
            previous = who
            e.s.tick_action(w, a if who == 'A' else b, e.g.prep.prior.EXECUTOR, update, counts)
            n += 1
        result = e.s.output(w,counts)
    return dict(result=result,hit=hit,b=chosen_b,different=different,switches=switches,proposals=n,examples=examples)


def probe_one(job):
    k, idx = job
    f = e.read(CAM/'tasks'/k/'fixture.json')
    row = play(f)
    ref = e.read(e.OUT/'train'/(k+'.json'))
    assert row['hit']==ref['A']['hit']
    if not row['different']:
        assert row['result']['history']==ref['A']['result']['history']
    row['reference'] = {m:ref[m]['result']['success'] for m in ('A','B','lock')}
    e.save(OUT/f'probes/v{idx}'/(k+'.json'), row)


def probe(idx):
    global ROUTE
    if idx not in (1,2,3):
        raise ValueError('slots 1 through 3')
    src = (WORK/f'v{idx}.py').read_text()
    ROUTE = load_router(src)
    c = e.base.host.RoomCampaign(CAM)
    slot = CAM/f'router_v{idx}.py'
    with c.locked():
        if slot.exists():
            assert slot.read_text()==src
            print('Slot already reserved. See results file when complete.')
            return
        if time.time()>DEADLINE or (CAM/'selection.json').exists():
            raise ValueError('authoring closed')
        rooms = c.state['tasks']
        c._charge(attempts=len(rooms), resets=len(rooms), execution_ticks=1800*len(rooms),
                  feature_batches=301*len(rooms))
        slot.write_text(src)
        c._save()
    started=time.monotonic()
    initialize(profile=e.perception.LEGACY)
    e.pool(probe_one,[(k,idx) for k in rooms],12)
    rows = {k:e.read(OUT/f'probes/v{idx}'/(k+'.json')) for k in rooms}
    ticks=sum(r['result']['ticks'] for r in rows.values())
    decisions=sum(r['result']['decisions'] for r in rows.values())
    with c.locked():
        c.state['used']['execution_ticks'] -= 1800*len(rooms)-ticks
        c.state['used']['feature_batches'] -= 301*len(rooms)-decisions
        c._save()
    info=dict(revision=idx,sha=graph.file_sha(slot),success=sum(r['result']['success'] for r in rows.values()),rooms=len(rooms),ticks=ticks,seconds=time.monotonic()-started,baseline={m:sum(r['reference'][m] for r in rows.values()) for m in ('A','B','lock')})
    e.save(WORK/f'results{idx}.json',dict(summary=info,rows=rows))
    e.save(CAM/f'probe{idx}.json',info)
    print(info,flush=True)


def freeze():
    selection=e.read(WORK/'selection.json')
    idx=selection['revision']
    assert idx in (1,2,3)
    c=e.base.host.RoomCampaign(CAM)
    if e.read(CAM/'campaign.json')['status']!='frozen':
        assert (WORK/'selection.json').stat().st_mtime <= DEADLINE
        assert graph.file_sha(CAM/f'router_v{idx}.py')==e.read(CAM/f'probe{idx}.json')['sha']
        e.save(CAM/'selection.json',selection)
        e.save(CAM/'host_shas.json',{p.name:graph.file_sha(p) for p in map(Path,(__file__,e.__file__,graph.__file__,e.s.__file__,e.s.phase.__file__))})
        shutil.copyfile(CAM/f'router_v{idx}.py',DOC/'router.py')
        c.freeze()
    seal=c.verify_frozen()
    e.save(OUT/'frozen.json',dict(seal=seal,router_sha=graph.file_sha(CAM/f'router_v{idx}.py'),selection=selection))
    return idx


def panel():
    make=e.s.phase.cap.rooms.generate
    rooms={f"router_{SEED}_{r['id']}":r['fixture'] for r in make(SEED)['cases']}
    prior=e.s.collect_identities()
    for p in e.read(e.old.previous.OUT/'panels.json')['rooms'].values():
        prior.update(p.values())
    for path in (e.OUT/'confirmation_ids.json',e.old.OUT/'fresh_panel.json'):
        prior.update(e.read(path).values())
    for name in ('development','held'):
        prior.update(e.s.phase.input_hash(f) for f in e.read(e.s.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,e.g.prep.SEED):
        prior.update(e.s.phase.input_hash(r['fixture']) for r in make(seed)['cases'])
    ids={k:e.s.phase.input_hash(f) for k,f in rooms.items()}
    assert len(set(ids.values()))==200 and not prior.intersection(ids.values())
    e.save(OUT/'confirmation_ids.json',ids)
    return rooms


def test_one(job):
    k,f=job
    p=OUT/'test'/(k+'.json')
    if p.exists():
        return
    rows={}
    for m in ARMS[:-1]:
        if m in ('lock','graph') and rows['A']['hit'] is None:
            rows[m]=dict(rows['A'],identical_prefix=True,cpu_seconds=0.)
            continue
        start=time.process_time()
        row=e.simulate(f,m,'stall',chooser=graph.choose)
        rows[m]={key:row[key] for key in ('result','hit','changed','switches')}
        rows[m]['cpu_seconds']=time.process_time()-start
    start=time.process_time()
    router=play(f)
    router['cpu_seconds']=time.process_time()-start
    assert router['hit']==rows['A']['hit']==rows['graph']['hit']==rows['lock']['hit']
    if not router['different']:
        assert router['result']['history']==rows['A']['result']['history']
    del router['examples']
    rows['router']=router
    e.save(p,rows)


def evaluate():
    global ROUTE
    idx=freeze()
    initialize(profile=e.perception.LEGACY)
    ROUTE=load_router((CAM/f'router_v{idx}.py').read_text())
    rooms=panel()
    started=time.time()
    e.pool(test_one,rooms.items(),18)
    e.save(OUT/'evaluation_time.json',dict(start=started,end=time.time(),workers=18))


if __name__=='__main__':
    if sys.argv[1]=='setup':
        setup()
    elif sys.argv[1]=='probe':
        probe(int(sys.argv[2]))
    elif sys.argv[1]=='evaluate':
        evaluate()
