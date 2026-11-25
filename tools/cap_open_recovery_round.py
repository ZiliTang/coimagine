"""Paired open CAP authoring round. Development and confirmation outputs are disjoint."""
import argparse
import ast
import hashlib
import json
import re
from pathlib import Path

from tools import cap_cluster_module as old
from tools import cap_fault_curriculum_rooms as rooms
from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram

OUT = old.OUT.parent / 'cap_open_recovery_20261124_01'
PUBLIC = OUT / 'author'
DOC = Path('docs/research/key_findings/2026-11-24_open_recovery')
read, save = old.read, old.save


def digest(source):
    return hashlib.sha256(source.encode()).hexdigest()


def dev_rooms():
    allowed = read(PUBLIC / 'rooms.json')
    dev = read(old.study.OUT / 'rooms.json')['rooms']['dev']
    return {k:dev[k] for k in allowed}


def seal(name):
    if not re.fullmatch('P|trial[0-9]{2,3}',name):
        raise ValueError('expected P or trialNN')
    path = OUT / 'sealed' / name
    source = (PUBLIC / (name+'.py')).read_text()
    if name == 'P' and digest(source)!=read(OUT/'plan.json')['parent']:
        raise ValueError('parent mismatch')
    if not path.exists():
        camp = old.author.host.RoomCampaign.create(path,dev_rooms())
        with camp.locked():
            camp.state['model'].update(name='gpt-6.2-sol',provider='collaboration')
            camp.state['author_protocol'].update(version='open-recovery-v1',wall_seconds=1800,token_cap=None)
            camp._save()
        (path/'composed.py').write_text(source)
        camp.freeze()
    old.author.host.RoomCampaign(path).verify_frozen()
    if (path/'composed.py').read_text()!=source:
        raise ValueError('candidate name already sealed with different contents')
    return source


def enabled(source):
    return any(isinstance(n,ast.Assign) and any(getattr(t,'id','')=='USE_PUSH_REVIEW' for t in n.targets)
               and isinstance(n.value,ast.Constant) and n.value.value is True for n in ast.parse(source).body)


def episode(job, audit=None):
    key, fixture, detail, start, kind, force, events, parent_src = job
    iso, prefix = old.isolated, old.prefix
    member = iso.IsolatedCAP(iso.BROKER)
    base_worker = StreamProgram(parent_src) if start is not None else None
    base = iso.IsolatedCAP(base_worker) if base_worker else None
    active = start is None
    reached = None
    timeline = []
    world_ref = None
    class Members:
        def update(self,history):
            member.update(history)
            if base and not active:
                base.update(history)
        def propose(self,obs):
            return (member if active else base).propose(obs)

    def observe_event(obs,action,reason,left):
        if detail:
            timeline.append(dict(tick=obs['tick'],kind=reason,remaining=left,action=action,
                contact=[r['contacting_object'] for r in obs['robots']],
                xy=[[r['own']['x'],r['own']['y']] for r in obs['robots']]))
    def switch(obs,reason):
        nonlocal active,reached
        if not active and obs['tick']==start and reason==kind:
            active=True
            reached=dict(tick=start,kind=kind,history=world_ref.history.sha256,
                         observation_sha256=digest(json.dumps(obs,sort_keys=True,separators=(',',':'))))
            return True
        return False
    def boundary(world,obs,action,pending):
        nonlocal world_ref
        world_ref=world
        observe_event(obs,action,'boundary',None)
        if switch(obs,'boundary'):
            proposal=member.propose(obs)
            if not pending or proposal!=base.propose(obs):
                action=proposal
        if audit is not None:
            audit.boundary(obs,member.memory,action,pending)
        return action
    def checkpoint(obs,action,left,reason):
        observe_event(obs,action,reason,left)
        first=switch(obs,reason)
        if not active or not events:
            return None
        # This preview, including the extra checkpoint fields, is never committed.
        tick=prefix.s.observe(world_ref)['tick']
        event=dict(observation=obs,action=action,executed_ticks=obs['tick']-tick,
                   push_checkpoint=dict(reason=reason,remaining_ticks=left,force=bool(force and first)))
        response=member.broker.request(dict(op='advance',memory=member.memory,event=event))
        props=response['proposals']
        if audit is not None and props:
            audit.checkpoint(obs,response['memory'],props[0],action,reason)
        if len(props)>1:
            raise ValueError('checkpoint requires zero or one proposal')
        if props:
            return validate_action(props[0],obs)
        return None
    def preview(_,obs,action,elapsed):
        return iso.preview(member if active else base,obs,action,elapsed)
    try:
        _, result=prefix.run_member((key,fixture,True),Members(),preview,boundary,
                                    push_takeover=checkpoint if detail or events or start is not None else None)
    finally:
        if base_worker: base_worker.close()
    if start is not None and reached is None:
        raise ValueError(f'{key}: requested prefix event not found')
    traces=result.pop('trace')
    if detail or audit is not None:
        result['boundaries']=[r['public'] for r in traces]
    if detail:
        result['events']=timeline
    result['effects']=[dict(tick=r['public']['tick'],node=r['node'],gap=r['true_gap']) for r in traces]
    result['anchor']=prefix._interventions(traces).get(old.NODE)
    result['reached']=reached
    return key,result


def run(name, phase, keys=None, start=None, kind='boundary', force=False, workers=12):
    if force and (start is None or kind=='boundary'):
        raise ValueError('forced checkpoint requires an explicit PUSH event')
    source=seal(name)
    if phase=='confirm':
        if name!='P' and read(OUT/'locked.json')['sha256']!=digest(source):
            raise ValueError('confirmation requires selected frozen candidate')
        fixtures=read(OUT/'fixtures.json')
    else:
        fixtures=dev_rooms()
    keys=keys or list(fixtures)
    if start is not None and len(keys)!=1:
        raise ValueError('one room per specified fork')
    dest=OUT/'runs'/phase/name/(f'{kind}_{start}_{int(force)}' if start is not None else 'full')
    parent=(PUBLIC/'P.py').read_text()
    jobs=[(k,fixtures[k],phase=='dev',start,kind,force,enabled(source),parent) for k in keys]
    old.collect(source,jobs,episode,min(len(keys),workers),dest)
    if phase=='dev':
        for k in keys:
            row=read(dest/(k+'.json'))
            # Only the public boundary data and actual outcomes go to the author.
            feedback={f:row[f] for f in ('result','boundaries','events','reached')}
            save(PUBLIC/'feedback'/name/dest.name/(k+'.json'),feedback)
    return dest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('phase',choices=['dev','confirm'])
    p.add_argument('name')
    p.add_argument('--rooms',nargs='+')
    p.add_argument('--start',type=int)
    p.add_argument('--kind',choices=['boundary','contact','progress'],default='boundary')
    p.add_argument('--force',action='store_true')
    p.add_argument('--workers',type=int,default=12)
    a=p.parse_args()
    run(a.name,a.phase,a.rooms,a.start,a.kind,a.force,a.workers)
