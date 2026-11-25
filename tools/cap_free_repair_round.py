"""Free choice of a general fix or independently switchable recovery, starting at trial11."""
import argparse
import shutil
import time
from collections import defaultdict
from tools import cap_pr_graph_round as prev
from tools.cap_entry_residual_study import immutable

host = prev.host
OUT = prev.OUT.parent / 'cap_free_repair_20261124_01'
PUBLIC = OUT / 'author'
DOC = prev.DOC.parent / '2026-11-24_free_repair'
read, save = host.read, host.save

def configure():
    host.OUT, host.PUBLIC, host.DOC, host.dev_rooms = OUT, PUBLIC, DOC, prev.dev_pool

def setup():
    prev.configure()
    source = host.seal('trial11')
    assert host.digest(source) == read(prev.OUT/'locked.json')['sha256']
    configure()
    PUBLIC.mkdir(parents=True, exist_ok=True)
    immutable(OUT/'plan.json', dict(parent=host.digest(source), wall_seconds=1800, token_cap=None,
        dev=160, confirm=200, endpoint='paired full room success, rescues, losses; ablations by frozen change type'))
    (PUBLIC/'P.py').write_text(source)
    pool=prev.dev_pool()
    save(PUBLIC/'rooms.json',sorted(pool))
    prev.SEEDS['freerepair_confirm']=842105733
    fixtures,_=prev.panel('freerepair_confirm',200)
    used={prev.content_hash(f) for f in pool.values()}
    used.update(prev.content_hash(f) for _,f in read(prev.OLD/'fixtures.json').values())
    for path in [prev.OUT/'fixtures.json',prev.OLD_AUTHOR.parent/'fixtures.json',host.old.OUT/'fixtures.json',*prev.prior.PREV.glob('fixtures/*.json')]:
        used.update(prev.content_hash(f) for f in read(path).values())
    used.update(x['content_sha256'] for x in read(host.old.study.OUT/'rooms.json')['records'].values())
    assert len({prev.content_hash(f) for f in fixtures.values()}-used)==200
    immutable(OUT/'fixtures.json',fixtures)
    shutil.copytree(prev.PUBLIC/'knowledge',PUBLIC/'knowledge',dirs_exist_ok=True)
    host.run('P','dev',workers=16)
    # A room can visit several modes. Count rooms, not repeated boundaries.
    index=defaultdict(lambda: {'success':[], 'failure':[]})
    matrix={}
    for room in sorted(pool):
        row=read(OUT/'runs/dev/P/full'/(room+'.json'))
        won=bool(row['result']['success'])
        trace=[dict(node=e['node'],true_gap=e['gap'],public=b) for e,b in zip(row['effects'],row['boundaries'],strict=True)]
        modes=prev.PREFIX._interventions(trace)
        matrix[room]={'success':won,'stalls':{n:trace[v['window_end']]['public']['tick'] for n,v in modes.items()}}
        for n in {step['node'] for step in trace}:
            index[n]['success' if won else 'failure'].append(dict(room=room,stall=matrix[room]['stalls'].get(n)))
    save(PUBLIC/'overview.json',dict(rooms=matrix,mode_visits=index))
    save(OUT/'READY.json',dict(dev=len(pool),success=sum(r['success'] for r in matrix.values())))

def cli():
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['setup','dev'])
    p.add_argument('name',nargs='?',default='P')
    p.add_argument('--rooms',nargs='+')
    p.add_argument('--start',type=int)
    p.add_argument('--kind',default='boundary',choices=['boundary','contact','progress'])
    p.add_argument('--workers',type=int,default=12)
    p.add_argument('--force',action='store_true')
    a=p.parse_args()
    if a.phase=='setup':
        setup()
    else:
        configure()
        if time.time()>read(OUT/'budget.json')['deadline']:
            raise ValueError('author time expired')
        host.run(a.name,'dev',a.rooms,a.start,a.kind,a.force,workers=a.workers)

if __name__=='__main__':
    cli()
