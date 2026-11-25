"""Graph-selected failure curriculum for the frozen P+R pair, using the v2 CAP host."""
import argparse
from collections import Counter
from pathlib import Path
import json
import random
import shutil
import time

from tools import cap_open_recovery_round as host, cap_entry_residual_engine as frozen
from tools import cap_handoff_opportunities as prior
from tools.cap_entry_residual_study import immutable
from tools.cap_fault_curriculum_rooms import panel, SEEDS, content_hash
from tools.cap_recovery_round_prompt import make_prompt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/semantic_rl/cap_pr_graph_20261124_01'
PUBLIC = OUT/'author'
DOC = ROOT/'docs/research/key_findings/2026-11-24_pr_graph'
OLD = prior.OUT
read, save = host.read, host.save
PREFIX = host.old.prefix
ISO = host.old.isolated
OLD_AUTHOR = frozen.baseline.OUT/'author'

def dev_pool():
    return {k:f for k,(g,f) in read(OLD/'fixtures.json').items() if g=='dev_nat'}

def dev_rooms():
    pool=dev_pool()
    return {k:pool[k] for k in read(PUBLIC/'rooms.json')}

def configure():
    host.OUT,host.PUBLIC,host.DOC,host.dev_rooms=OUT,PUBLIC,DOC,dev_rooms
    if (OUT/"selection.json").exists():
        host.old.NODE=read(OUT/"selection.json")["node"]

def scan(job):
    k,f,record=job
    member=ISO.IsolatedCAP(ISO.BROKER)
    roles=[]
    def annotate(world,obs,action,pending):
        recovery=member.memory['recovery']
        roles.append(dict(tick=obs['tick'],active=bool(recovery.get('plan')),offered=bool(recovery.get('pending')),
                          has_command=bool(recovery.get('command')),released=recovery.get('release')==obs['tick']))
        return action
    _,r=PREFIX.run_member((k,f,record),member,ISO.preview,annotate)
    for step,role in zip(r['trace'],roles,strict=True):
        assert role.pop('tick')==step['public']['tick']
        step['public']['recovery']=role
    return k,r

def setup():
    PUBLIC.mkdir(parents=True,exist_ok=True)
    DOC.mkdir(parents=True,exist_ok=True)
    eng=frozen
    eng.prepare()
    source=eng.source('trial06')
    configure()
    (PUBLIC/'P.py').write_text(source)
    immutable(OUT/'plan.json',dict(parent=host.digest(source),wall_seconds=1800,token_cap=None,confirm=200,
        endpoint='paired room completion; frozen baseline cluster membership; fixed-prefix 120 tick 3.5 cm local advance',
        selection='top recurrent low-advance node by distinct failures on development; select using development only'))
    SEEDS['prgraph_confirm']=753706923
    confirm,_=panel('prgraph_confirm',200)
    previous={content_hash(f) for _,f in read(OLD/'fixtures.json').values()}
    for path in list(prior.PREV.glob('fixtures/*.json'))+[host.old.OUT/'fixtures.json',OLD_AUTHOR.parent/'fixtures.json']:
        previous.update(content_hash(f) for f in read(path).values())
    previous.update(x['content_sha256'] for x in read(host.old.study.OUT/'rooms.json')['records'].values())
    assert len({content_hash(f) for f in confirm.values()}-previous)==200
    immutable(OUT/'fixtures.json',confirm)
    pool=dev_pool()
    dest=OUT/'graph'
    host.old.collect(source,[(k,f,True) for k,f in pool.items()],scan,16,dest)
    rows={k:read(dest/(k+'.json')) for k in pool}
    for k,r in rows.items():
        assert r['result']==read(OLD/'runs'/(k+'.json'))['baseline']
    _,audit=PREFIX.select_packet(rows,'graph',seed=671445)
    counts=audit['support']
    node=sorted(counts,key=lambda n:(-counts[n]['fail'],-int(counts[n]['success']>0),n))[0]
    hits={k:PREFIX._interventions(r['trace']).get(node) for k,r in rows.items()}
    rng=random.Random(9321)
    keys=sorted(rows)
    rng.shuffle(keys)
    failures=[k for k in keys if hits[k] and not rows[k]['result']['success']]
    wins=[k for k in keys if hits[k] and rows[k]['result']['success']]
    cases=failures[:12]+wins[:6]
    cases += [k for k in keys if k not in cases and not hits[k] and rows[k]['result']['success']][:22-len(cases)]
    save(PUBLIC/'rooms.json',cases)
    def example(k):
        h=hits[k]
        ex=PREFIX._public_example(rows,k,h['prefix'] if h else 0,'matched' if h else 'outside_control')
        if h:
            ex['stall_confirmed_at']=rows[k]['trace'][h['window_end']]['public']['tick']
            ex['trace_until_stall']=[r['public'] for r in rows[k]['trace'][h['prefix']:h['window_end']+1]]
        return ex
    save(PUBLIC/'timeline.json',[example(k) for k in cases if not rows[k]['result']['success']])
    controls=[]
    for k in cases:
        if not rows[k]['result']['success']:
            continue
        visits=[j for j,step in enumerate(rows[k]['trace']) if step['node']==node]
        controls.append(PREFIX._public_example(rows,k,max(0,visits[0]-1),'visited_same_node_then_succeeded') if visits and not hits[k] else example(k))
    save(PUBLIC/'controls.json',controls)
    save(OUT/'selection.json',dict(node=node,support=counts,edges=audit['edges'],cases={k:hits[k] for k in cases},n=len(pool)))
    shutil.copytree(OLD_AUTHOR/'knowledge',PUBLIC/'knowledge',dirs_exist_ok=True)
    interface=(OLD_AUTHOR/'interface.md').read_text()
    interface=interface.replace('tools.cap_open_recovery_round','tools.cap_pr_graph_round')
    i=interface.index('Goal:'); j=interface.index('Submit complete',i)
    interface=interface[:i]+f'Goal: improve the current P+R system on the recurrent {node} stall in the graph-selected development packet. P.py is the complete incumbent P+R; its recovery functions already implement R. Keep two control roles; you may improve normal planning, recovery behaviour or their interface, including preventive event interventions. Compare against naturally recovering cases. Mark which role changed. Use only observations for runtime decisions.\n\n'+interface[j:]
    if node=='unavailable/open':
        interface+='\nGraph label unavailable/open means none of three sampled pushing stations on the goal-directed rear face was clear for a robot, even though the short forward object sweep was open. The 120-tick repeated low-progress window was seen in 20 failed and 0 successful development rooms. Two of the supplied successful controls transiently visit the same node and still finish; they are marked in controls.json. These offline hints may inform the author, not deployed inputs. When inspecting a memory summary, active is an acknowledged R plan, offered is a pending R plan. In FINAL please specify measurable local claim, anchor, horizon, criterion, and whether direct activation without the soft gate is available as a development probe.\n'
    (PUBLIC/'interface.md').write_text(interface)
    (PUBLIC/'prompt.md').write_text(make_prompt(PUBLIC/'P.py',PUBLIC/'timeline.json',PUBLIC/'controls.json',PUBLIC/'interface.md'))
    host.run('P','dev',workers=14)

def run_dev(args):
    configure()
    if args.name!='P':
        clock=read(OUT/'budget.json')
        p=PUBLIC/(args.name+'.py')
        if not p.is_file() or max(p.stat().st_mtime,time.time())>clock['deadline']:
            raise ValueError('author time expired')
    host.run(args.name,'dev',args.rooms,args.start,args.kind,args.force,workers=args.workers)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['setup','dev','base-confirm'])
    parser.add_argument('name',nargs='?')
    parser.add_argument('--rooms',nargs='+')
    parser.add_argument('--start',type=int)
    parser.add_argument('--kind',default='boundary',choices=['boundary','contact','progress'])
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--force',action='store_true')
    args=parser.parse_args()
    if args.phase=='setup':setup()
    elif args.phase=='dev':run_dev(args)
    else:
        configure();host.run('P','confirm',workers=16)
