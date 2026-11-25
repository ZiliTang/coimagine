"""Trace the incumbent recovery selector's first rejection on actual development histories."""
from collections import Counter
from tools import cap_pr_graph_round as r

LABELS=('evidence_or_gap_or_time','recently_advanced','recent_distance_varied','forward_path_open',
        'no_matching_push_pose','parent_has_other_proposals','no_contact_or_pair_push',
        'parent_repositioning','unchanged_parent_push')

def instrument(source):
    a=source.index('def recovery_select('); b=source.index('original_initial_memory=',a)
    lines=source[a:b].splitlines(True)
    which=iter(LABELS)
    for i,line in enumerate(lines):
        if line.strip()=='return None':
            indent=line[:len(line)-len(line.lstrip())]
            name=next(which)
            reason=repr(name)
            if name=='evidence_or_gap_or_time':
                reason="('history_short' if not older else 'too_few_recent_observations' if len(recent)<4 else 'within_0.85m_of_goal' if gap<.85 else 'under_11_seconds')"
            lines[i]=indent+"memory['trace_guard']="+reason+"\n"+line
        if line.strip()=='return best':
            lines[i]="  memory['trace_guard']='admitted' if best else 'no_legal_plan'\n"+line
    if next(which,None) is not None:
        raise ValueError('changed selector')
    source=source[:a]+''.join(lines)+source[b:]
    old="  state=memory.pop('recovery')\n  memory=original_update_memory(memory,event)"
    assert source.count(old)==1
    return source.replace(old,"  memory.pop('trace_guard',None)\n"+old+"\n  memory['trace_guard']='not_called'")

def worker(job):
    key,fixture,start,end=job
    states=[]
    member=r.ISO.IsolatedCAP(r.ISO.BROKER)
    def boundary(world,obs,action,pending):
        tick=obs['tick']
        if start<=tick<=end:
            why=member.memory['trace_guard']
            if why=='not_called' and member.memory['slot'].get('plan'):
                why='existing_parent_plan'
            states.append(dict(tick=tick,first_rejection=why))
        return action
    _,row=r.PREFIX.run_member((key,fixture,False),member,r.ISO.preview,boundary)
    assert row['result']==r.read(r.OUT/'graph'/(key+'.json'))['result']
    return key,dict(observations=states,exact_replay=True)

def main():
    pool=r.dev_pool()
    cases=r.read(r.OUT/'graph_diagnosis.json')
    src=instrument((r.PUBLIC/'P.py').read_text())
    jobs=[(x['room'],pool[x['room']],x['first_tick'],x['confirm_tick']) for x in cases]
    r.host.old.collect(src,jobs,worker,10,r.OUT/'guards_detail')
    count=Counter(); support=Counter()
    for k,_,_,_ in jobs:
        entries=r.read(r.OUT/'guards_detail'/(k+'.json'))['observations']
        signs=[e['first_rejection'] for e in entries]
        count.update(signs);support.update(set(signs))
    r.save(r.OUT/'guard_detail_summary.json',dict(boundaries=count,rooms=support,n=len(jobs)))
if __name__=='__main__':main()
