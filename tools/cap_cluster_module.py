"""Single-cluster CAP development and paired testing on the fixed-diagonal task."""
import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

from tools import cap_fault_curriculum as study
from tools import cap_fault_curriculum_author as author
from tools import cap_fault_curriculum_isolated as isolated
from tools import cap_fault_curriculum_prefix as prefix
from experiments.planar_tasks.relational_search.contract import validate_action
from experiments.planar_tasks.semantic_rl.typed_compiler import compile_source

OUT = study.OUT.parent / 'cap_cluster_module_20261124_01'
PUBLIC = OUT / 'author'
PARENT = study.OUT / 'mother'
NODE = 'staged/fixed'
read, save = study.read, study.save


def source_parent():
    author.host.RoomCampaign(PARENT / 'campaign').verify_frozen()
    source = (PARENT / 'policy.py').read_text()
    assert author.sha(source) == read(OUT/'plan.json')['mother']
    assert source == (PARENT/'campaign/composed.py').read_text()
    return source


def run_base(job):
    key, fixture, wanted = job
    member = isolated.IsolatedCAP(isolated.BROKER)
    def snapshot(world, observation, action, pending):
        if wanted and world.history.sha256 == wanted:
            save(PUBLIC/'snapshots'/f'{key}.json', dict(observation=observation,
                 memory=member.memory, action=action, public_history=wanted))
        return action
    _, row = prefix.run_member((key, fixture, True), member, isolated.preview, snapshot)
    if wanted and not (PUBLIC/'snapshots'/f'{key}.json').is_file():
        raise ValueError(f'{key}: recorded history was not reproduced')
    witness = prefix._interventions(row['trace']).get(NODE)
    anchor = None
    if witness:
        record = row['trace'][witness['window_end']]['public']
        anchor = dict(tick=record['tick'], history=record['public_history'], index=record['n'])
    return key, dict(result=row['result'], anchor=anchor)


def collect(source, jobs, worker, workers, dest):
    compile_source(source)
    dest.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context('fork'),
                             initializer=isolated.initialize, initargs=(source,)) as pool:
        futures = [pool.submit(worker, job) for job in jobs if not (dest/f'{job[0]}.json').is_file()]
        for done in as_completed(futures):
            key, row = done.result()
            save(dest/f'{key}.json',row)


def base_phase(phase, workers):
    src=source_parent()
    if phase=='snapshots':
        dev=read(study.OUT/'rooms.json')['rooms']['dev']
        jobs=[(r['room'],dev[r['room']],r['prefix']) for r in read(PUBLIC/'cases.json')]
    else:
        jobs=[(k,f,None) for k,f in read(OUT/'fixtures.json').items()]
    collect(src,jobs,run_base,workers,OUT/phase)


def composed(snippet):
    mother=source_parent()
    author.check_response(dict(source=snippet,rationale=''),source_limit=49152)
    for fn in ('initial_memory','update_memory','prepare_context'):
        mother=mother.replace('def '+fn+'(', 'def parent_'+fn+'(')
    return mother+'\n'+snippet+'''
from copy import deepcopy as owned

def initial_memory() -> dict:
    memory=parent_initial_memory()
    memory['slot']={}
    return memory

def update_memory(memory: dict, event: dict) -> dict:
    slot=memory.pop('slot')
    memory.pop('force_test',None)
    force=event.pop('force_test',False)
    memory=parent_update_memory(memory,event)
    slot=module_update(slot,owned(event),owned(memory))
    slot.pop('_force',None)
    memory['slot']=slot
    memory['force_test']=force
    return memory

def prepare_context(memory: dict, observation: dict) -> dict:
    slot=memory.pop('slot')
    slot['_force']=memory.pop('force_test',False)
    ctx=parent_prepare_context(memory,observation)
    base=ctx['proposals'][0]
    action=module_act(slot,owned(memory),owned(observation),dict(base))
    if action is not None:
        ctx['proposals'].append(base)
        ctx['proposals'][0]=action
        tid=action['target_id']
        idx=next((j for j,e in enumerate(observation['entities']) if e['id']==tid),-1)
        ctx['values']=[idx,action['a0'],action['a1'],action['ticks']]
    return ctx
'''


def sealed(name, fixtures):
    path=OUT/'trials'/name/'campaign'
    source=composed((PUBLIC/(name+'.py')).read_text())
    if not path.exists():
        camp=author.host.RoomCampaign.create(path,fixtures)
        with camp.locked():
            camp.state['model'].update(name=author.MODEL, provider='collaboration')
            camp.state['author_protocol'].update(version='single-cluster-v1',wall_seconds=1800,token_cap=None,
                final='single author-selected revision before deadline',case_access='public development cases only')
            camp._save()
        (path/'composed.py').write_text(source)
        camp.freeze()
    author.host.RoomCampaign(path).verify_frozen()
    if (path/'composed.py').read_text()!=source:
        raise ValueError('trial source changed after freeze')
    return source


class BranchMember(isolated.IsolatedCAP):
    def __init__(self, broker, mode, anchor):
        super().__init__(broker)
        self.mode, self.anchor = mode, anchor
        self.phase = 'full' if mode=='full' else 'before'
        self.seen = False
        self.takes = 0
        self.changed = 0
        self.first_tick = None
        self.local_effect = []

    def advance(self,event):
        self.input_memory=self.memory
        self.input_event=event
        row=super().advance(event)
        self.committed=row
        return row

    def choose(self,row,obs,preview=False):
        props=row['proposals']
        module=len(props)==3
        base=props[-1] if module else props[0]
        if self.phase in ('before','after'):
            return validate_action(base,obs)
        if not module:
            if self.phase=='once':
                self.phase='after'
            return validate_action(base,obs)
        try:
            action=validate_action(props[0],obs)
        except ValueError:
            if preview:
                return None
            raise
        self.takes+=1
        self.changed+=action!=base
        if self.first_tick is None:
            self.first_tick=obs['tick']
        return action

    def propose(self,obs):
        return self.choose(self.committed,obs)

    def push_checkpoint(self,obs,action,left,reason):
        """Preview only the optional module, using the sealed program's stream ABI.

        module_update can pass the public checkpoint context to module_act via its
        temporary slot. The running history is updated only by real execution.
        """
        if self.phase!='full':
            raise ValueError('module PUSH checkpoints require full deployment mode')
        tick=self.input_event['observation']['tick']
        event=dict(observation=obs,action=action,executed_ticks=obs['tick']-tick,
                   push_checkpoint=dict(reason=reason,remaining_ticks=left))
        row=self.broker.request(dict(op='advance',memory=self.memory,event=event))
        # The composed ABI appends the parent action as a third proposal only
        # when the module actually requests control at this checkpoint.
        if len(row['proposals'])==3:
            return row['proposals'][0]
        return None

    def boundary(self,world,obs,action,pending):
        if self.mode in ('parent','authored','forced') and self.anchor:
            elapsed=obs['tick']-self.anchor['tick']
            if elapsed >= 0 and (not self.local_effect or
                    (len(self.local_effect)==1 and elapsed>=120)):
                bits,gap,_=prefix.geometry.topology(world,obs)
                self.local_effect.append(dict(tick=obs['tick'],gap=gap,node=prefix._node(bits)))
        if hasattr(self,'log'):
            row=self.committed
            props=row['proposals']
            m=row['memory']
            plan=m['slot'].get('plan') or {}
            self.log.append(dict(tick=obs['tick'], from_event=pending,
                action=action, base=props[-1] if len(props)==3 else props[0], override=len(props)==3,
                xy=[[r['own']['x'],r['own']['y']] for r in obs['robots']],
                compact=m.get('compact'), local_plan=plan,
                stages=[r['execution'].get('stage') for r in obs['robots']],
                contact=[bool(r['contacting_object']) for r in obs['robots']],
                static=[bool(r['contacting_static']) for r in obs['robots']]))
        if self.phase!='before' or self.anchor is None or self.anchor['tick']!=obs['tick']:
            return action
        if self.anchor['history']!=world.history.sha256:
            raise ValueError('prefix history mismatch')
        self.seen=True
        if self.mode=='parent':
            self.phase='after'
            return action
        self.phase='once'
        row=self.committed
        if self.mode=='forced':
            row=self.broker.request(dict(op='advance',memory=self.input_memory,
                 event=dict(self.input_event,force_test=True)))
        if len(row['proposals'])!=3:
            self.phase='after'
            return action
        return self.choose(row,obs)


def preview(member,obs,action,elapsed):
    row=member.broker.request(dict(op='advance',memory=member.memory,
             event=dict(observation=obs,action=action,executed_ticks=elapsed)))
    try:
        return member.choose(row,obs,True)
    except ValueError:
        return None


def run_trial(job):
    key,fixture,mode,anchor,feedback_file=job
    member=BranchMember(isolated.BROKER,'full' if mode=='public-trace' else mode,anchor)
    if mode=='public-trace' or feedback_file:
        member.log=[]
    _,row=prefix.run_member((key,fixture,False),member,preview,member.boundary)
    if mode=='public-trace':
        return key,dict(result=row['result'],boundaries=member.log)
    if feedback_file:
        save(feedback_file,dict(result=row['result'],boundaries=member.log))
    if mode!='full' and not member.seen:
        raise ValueError('anchor never reached: '+key)
    return key,dict(result=row['result'],takes=member.takes,changed=member.changed,
                    first_tick=member.first_tick,verified_anchor=member.seen,local_effect=member.local_effect)


def trial_phase(name,phase,mode,workers):
    if not name or not name.startswith('trial') or not name[5:].isdigit():
        raise ValueError('expected trialNN')
    cases=read(PUBLIC/'cases.json')
    dev=read(study.OUT/'rooms.json')['rooms']['dev']
    source=sealed(name,{r['room']:dev[r['room']] for r in cases})
    if phase=='packet':
        fixtures={r['room']:dev[r['room']] for r in cases}
        base_dir=OUT/'snapshots'
    else:
        final=read(OUT/'locked.json')
        if name!=final['name'] or author.sha(source)!=final['sha256']:
            raise ValueError('final candidate required')
        fixtures=read(OUT/'fixtures.json')
        base_dir=OUT/'baseline'
    jobs=[]
    for key,fixture in fixtures.items():
        anchor=read(base_dir/f'{key}.json')['anchor']
        if anchor or mode in ('full','public-trace'):
            feedback_file=(PUBLIC/(name+'_traces')/(key+'.json')
                           if phase=='packet' and mode=='full' else None)
            jobs.append((key,fixture,mode,anchor,feedback_file))
    if mode=='public-trace' and phase!='packet':
        raise ValueError('author feedback is development-only')
    dest=PUBLIC/(name+'_traces') if mode=='public-trace' else OUT/'trials'/name/phase/mode
    collect(source,jobs,run_trial,workers,dest)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['snapshots','baseline','packet','confirmation'])
    parser.add_argument('--name')
    parser.add_argument('--mode',choices=['full','forced','authored','parent','public-trace'],default='full')
    parser.add_argument('--workers',type=int,default=12)
    args=parser.parse_args()
    if args.phase in ('snapshots','baseline'):
        base_phase(args.phase,args.workers)
    else:
        trial_phase(args.name,args.phase,args.mode,args.workers)
