"""Same-prefix timing probes of frozen trial06's recovery logic."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from math import dist
import multiprocessing as mp
import random
import time
import traceback

from tools import cap_entry_residual_engine as eng, cap_entry_residual_study as prior, cap_entry_room_yield as selection
from tools.cap_entry_residual_study import OUT as PREV, rooms, read, save, immutable
from experiments.planar_tasks.relational_search.contract import digest, validate_action
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source

OUT=PREV.parent/'cap_handoff_opportunities_20261124_01'
DOC=prior.ROOT/'docs/research/key_findings/2026-11-24_handoff_opportunities'
HOST=eng.host
BROKER=None


@lru_cache(maxsize=1)
def instrument():
    s=eng.source('trial06')
    # Only relax the original evidence window on demand. All collision/feasibility
    # and time-left constraints and all candidate construction remain unchanged.
    for old,new in [
        ("  older=[x for x in hist if tick-x[0]>=140]", "  probe=memory.get('timing_probe','')\n  wait=60 if probe=='early' else 140\n  older=[x for x in hist if tick-x[0]>=wait]"),
        ("tick-x[0]<=145]", "tick-x[0]<=wait+5]"),
        ("if not older or len(recent)<4 or gap<.85","if not older or len(recent)<(2 if probe=='early' else 4) or gap<.85"),
        ("if mag(sub(older[-1][1:3],goal))-gap>.14:", "if mag(sub(older[-1][1:3],goal))-gap>(.06 if probe=='early' else .14):"),
        ("  state=memory.pop('recovery')\n  memory=original_update_memory", "  state=memory.pop('recovery')\n  probe=event.get('timing_probe','')\n  memory=original_update_memory"),
        ("  previous=state.pop('command',None)","  if probe:\n    memory['timing_probe']=probe\n  previous=state.pop('command',None)"),
        ("      if ((advance>=plan['length'] and not extend)", "      extra=(probe=='extend' and advance<=plan['length']+.10)\n      if ((advance>=plan['length'] and not extend and not extra)"),
        ("if cmd and advance<plan['length']:", "if cmd and (advance<plan['length'] or extra):"),
        ("  memory['recovery']=state\n  return memory","  if probe=='giveback' and state.get('plan') and state.get('command'):\n    state.pop('plan')\n    state.pop('command')\n    state['release']=tick\n  memory.pop('timing_probe',None)\n  memory['recovery']=state\n  return memory")]:
        if s.count(old)!=1:raise ValueError(old)
        s=s.replace(old,new)
    return s


def init():
    global BROKER
    BROKER=StreamProgram(instrument())


class Member:
    def __init__(self, targets):
        self.targets=list(targets)
        self.matched=[]
        self.memory=None
        self.history=None
        self.offers=[]
        self.legacy_entries=0
        self.carry=None

    def query(self, before, event, probe=None):
        return BROKER.request(dict(op='advance',memory=before,event=dict(event,timing_probe=probe) if probe else event))

    def update(self,history):
        event=HOST.phase.cap.current_event(history)
        tick=event['observation']['tick']
        self.ev=dict(event,executed_ticks=None if history.previous_tick is None else tick-history.previous_tick)
        self.before=self.memory
        self.history=history.sha256
        probe=self.carry[1] if self.carry and self.carry[0]==tick else None
        self.row=self.query(self.before,self.ev,probe)
        self.memory=self.row['memory']
        self.carry=None

    def choose(self,before,event,row,kind):
        obs=event['observation']
        now=obs['tick']
        review=kind!='boundary'
        state=row['memory']['recovery']
        def action(reply):
            try:a=validate_action(reply['proposals'][0],obs)
            except ValueError:
                if not review:raise
                return None
            return None if review and eng.event.same_control(a,event['action']) else a
        current=action(row)
        prev=((before or {}).get('recovery') or {})
        if state.get('pending'):
            self.legacy_entries+=1
            return current
        if state.get('plan') and state.get('command'):
            probe='giveback'
        elif state.get('release')==now and prev.get('plan'):
            probe='extend'
        elif not state.get('plan') and not state.get('command') and not row['memory']['slot'].get('plan') and now>=60 and now-state.get('release',-200)>=170:
            probe='early'
        else:return current
        other=self.query(before,event,probe)
        alt=action(other)
        altstate=other['memory']['recovery']
        if alt==current or (review and alt is None):return current
        if probe=='early' and not altstate.get('pending'):return current
        if probe=='extend' and not (altstate.get('plan') and altstate.get('command')):return current
        plan=altstate.get('pending') if probe=='early' else (state.get('plan') or altstate.get('plan'))
        anchor=dict(tick=now,kind=kind,probe=probe,start=plan['start'],prefix=self.history,
                    observation=digest(obs),memory=digest(before),default=current,alternative=alt)
        if probe=='giveback':
            parent,rec=other['proposals'][0],row['proposals'][0]
        else:parent,rec=row['proposals'][0],other['proposals'][0]
        feat=eng.prepare_features(obs,row['memory'],plan,rec,parent,review)
        feat.update(plan_age=now-plan['start'],plan_best=plan['best'])
        anchor['features']=feat
        self.offers.append(anchor)
        if self.targets and anchor==self.targets[0]:
            self.matched.append(self.targets.pop(0))
            if review:self.carry=now,probe
            else:self.row=other;self.memory=other['memory']
            return alt
        return current

    def propose(self,obs):return self.choose(self.before,self.ev,self.row,'boundary')


def run(key,fixture,targets):
    member=Member(targets)
    counts=Counter()
    def decide(world,obs,action,left,reason):
        ev=dict(observation=obs,action=action,executed_ticks=obs['tick']-HOST.observe(world)['tick'])
        chosen=member.choose(member.memory,ev,member.query(member.memory,ev),reason)
        counts['check_'+reason]+=1
        if chosen is not None:counts['changed_'+reason]+=1
        return chosen
    with HOST.phase.world(fixture) as (world,_):
        obs=HOST.observe(world)
        body=world.env.env.backend.bodies[obs['goal']['target_id']]
        goal=obs['goal']['xy'];gaps=[dist(body.position,goal)]
        old=world.detector.update
        def observe(packet):
            result=old(packet)
            gaps.append(dist(body.position,goal))
            return result
        world.detector.update=observe
        member.update(world.history)
        pending=None
        reviews=eng.event.PushReview('progress',decide,progress_m=.20,contact_ticks=3)
        while not HOST.ended(world):
            act=member.propose(HOST.observe(world)) if pending is None else pending
            pending=HOST.tick_action(world,act,'split_precision',member.update,counts,review=reviews)
        result=HOST.output(world,counts)
        assert len(gaps)==1+result['ticks'] and not member.targets
    return dict(result=result,offers=member.offers,matched=member.matched,legacy_pending=member.legacy_entries),gaps


def choose(rows,probe):
    arr=[r for r in rows if r['probe']==probe]
    if not arr:return []
    if probe=='extend' or len(arr)==1:return arr[:1]
    return [arr[0],arr[-1]] if arr[0]!=arr[-1] else arr[:1]


def evaluate(job):
    k,f,group=job
    base,gaps=run(k,f,[])
    pairs=[]
    def branch(route,alt_anchor,default,old_gaps,origin):
        trial,new_gaps=run(k,f,[*route,alt_anchor])
        t=alt_anchor['tick']
        assert abs(old_gaps[t]-new_gaps[t])<1e-8
        def gain(g):return round(g[t]-g[min(t+120,len(g)-1)],5)
        pairs.append(dict(origin=origin,anchor=alt_anchor,default=default['result']['success'],altered=trial['result']['success'],
                          default_advance=gain(old_gaps),altered_advance=gain(new_gaps)))
        return trial,new_gaps
    early=choose(base['offers'],'early')
    for ix,a in enumerate(early):
        trial,alterg=branch([],a,base,gaps,'original')
        if ix==0:
            # Only the plan started by this particular earlier entry.
            exits=[x for x in trial['offers'] if x['tick']>a['tick'] and x['start']==a['tick']]
            for typ in ['giveback','extend']:
                for b in choose(exits,typ):
                    branch([a],b,trial,alterg,'after_early')
    for typ in ['giveback','extend']:
        for a in choose(base['offers'],typ):
            branch([],a,base,gaps,'original')
    if group=='silent':
        historical=read(PREV/'runs/train/zero'/(k+'.json'))
        assert base['result']==historical['result'] and not historical['entries']
    save(OUT/'runs'/(k+'.json'),dict(room=k,group=group,baseline=base['result'],legacy_pending=base['legacy_pending'],pairs=pairs))


def panels():
    previous=read(PREV/'fixtures/train.json'); keys=sorted(previous)
    positives=selection.labels(PREV,'train',keys)[:,0]
    x=selection.positions(previous,keys)
    seen=set()
    for p in [*list((PREV/'fixtures').glob('*.json')),prior.PREV/'fixtures.json',
              PREV.parent/'cap_cluster_module_20261124_01/fixtures.json']:
        seen.update(rooms.content_hash(f) for f in read(p).values())
    seen.update(r['content_sha256'] for r in read(PREV.parent/'cap_fault_curriculum_20261124_01/rooms.json')['records'].values())
    chosen={}
    for split,n,seed,enrich in [('dev_nat',160,741113911,False),('dev_geo',160,741113913,True),
                               ('val_nat',160,741113917,False),('val_geo',80,741113919,True)]:
        rooms.SEEDS[split]=seed
        pool,_=rooms.panel(split,n*5 if enrich else n)
        names=sorted(pool)
        if enrich:names=[names[i] for i in selection.candidates(x,positives,selection.positions(pool,names))]
        for k in names:
            f=pool[k];h=rooms.content_hash(f)
            if h in seen:raise ValueError('duplicate new room')
            seen.add(h);chosen[k]=[split,f]
    silent=[k for k in keys if not (r:=read(PREV/'runs/train/zero'/(k+'.json')))['entries'] and not r['result']['success']]
    for k in random.Random(71323).sample(silent,80):chosen[k]=['silent',previous[k]]
    immutable(OUT/'fixtures.json',chosen)
    immutable(OUT/'protocol.json',dict(counts=dict(dev_nat=160,dev_geo=160,silent=80,val_nat=160,val_geo=80),
        base=eng.FROZEN,probe_sha=eng.baseline.digest(instrument()),entry='140 to 60 ticks, 2 recent observations, progress <.06 m rather than .14; preserve other predicates',
        exits='early voluntary release on legal differing proposal while R active; late release only at displacement goal within .10m overrun, one legal next R decision, budgets and protections unchanged',
        scheduling='original natural and after first early intervention; pick first/last qualifying early handoff and early return plus first eligible extension; compare same prefix, tick, public obs, memory, proposals; identical original switching after intervention',
        criterion='terminal success paired, count distinct rescued and lost rooms; physical distance advance at +120 ticks or terminal secondary. Ridge on dev only, conditional held one-opportunity ranking if both label signs observed; no adoption from offline labels.'))
    return chosen


def study():
    eng.prepare()
    fixture_pack=panels()
    compile_source(instrument())
    init()
    reference=read(eng.baseline.old.study.OUT/'rooms.json')['rooms']['dev']
    checked=[]
    for k in ['diag_dev_027','diag_dev_084','diag_dev_126','diag_dev_030']:
        f=reference[k]
        record,_=run(k,f,[])
        assert record['result']==read(prior.PREV/'runs/dev/trial06/full'/(k+'.json'))['result']
        checked.append(k)
    BROKER.close()
    save(OUT/'parity.json',checked)
    with ProcessPoolExecutor(18,mp_context=mp.get_context('fork'),initializer=init) as pool:
        jobs=[(k,f,g) for k,(g,f) in fixture_pack.items() if not (OUT/'runs'/(k+'.json')).exists()]
        random.Random(713).shuffle(jobs)
        for future in as_completed([pool.submit(evaluate,j) for j in jobs]):future.result()
    if not (OUT/'silent_parent.json').exists():
        silent={k:f for k,(group,f) in fixture_pack.items() if group=='silent'}
        parent=eng.ISO.scan_panel(eng.source('P'),silent,workers=6,trace=False)
        save(OUT/'silent_parent.json',{k:r['result'] for k,r in parent.items()})
    from tools.cap_handoff_opportunities_report import build
    build()


if __name__=='__main__':
    try:study()
    except Exception:
        save(OUT/'error.json',dict(at=time.time(),traceback=traceback.format_exc()))
        raise
