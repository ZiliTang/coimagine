"""Query the nominal planner in shadow at actual decision boundaries, without changing actions."""
from experiments.planar_tasks.semantic_rl.stream_features import StreamProgram
from tools.cap_shared_failure_audit import DEST,r

def probe_source(source):
    return source.split('def module_update(')[0]+'''
def initial_memory():return {}
def update_memory(m,event):
    o=event['observation']
    tid=o['goal']['target_id']
    c,faces,points=geometry(o,m['c'])
    pairs=o['entity_legal_pairs'].get(tid,[])
    if not c:return {'why':'no_pose'}
    if not pairs or o['active_target']!=tid:return {'why':'no_pairs_or_binding'}
    goal=o['goal']['xy']
    dst=m['route']
    heading=norm(sub(dst,c)) if dst and mag(sub(goal,c))>.63 else route_goal(c,goal,o)
    def reason():
        best,score=push_plan(m,o,c,faces,heading,pairs)
        if not best:return 'no_candidate'
        if best[2]<=(.47 if dst and m['stale_ticks']>160 else .59):return 'alignment'
        if score<=.44:return 'score'
        return 'accepted'
    why=reason()
    result=dict(why=why,faces=sum(len(x) for x in faces),heading=heading,nominal=parent_prepare_context(m,o)['proposals'][0])
    if m['avoid'] and why!='accepted':
        m['avoid']={}
        result['without_avoid']=reason()
    return result
def prepare_context(m,o):
    return dict(values=[0],state=[0]*5,proposals=[])
'''
def run(job):
    k,f,src=job
    answer=[]
    class Audit:
        def boundary(self,o,m,a,pending):
            diag=worker.request(dict(op='advance',memory=m,event=dict(observation=o)))['memory']
            answer.append(dict(t=o['tick'],pending=pending,**diag))
        def checkpoint(self,*args):pass
    worker=StreamProgram(probe_source(src))
    try:
        _,res=r.host.episode((k,f,True,None,'boundary',False,r.host.enabled(src),src),Audit())
    finally:worker.close()
    assert res['result']==r.read(r.OUT/'evaluation/parent'/(k+'.json'))['result']
    return k,answer

if __name__=='__main__':
    r.configure()
    src=r.host.seal('P')
    keys=r.read(DEST/'groups.json')['pairs']
    fixtures=r.read(r.OUT/'fixtures.json')
    r.host.old.collect(src,[(k,fixtures[k],src) for k in keys],run,12,DEST/'shadow')
