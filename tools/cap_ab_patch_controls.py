"""At every boundary accept one source of B's action changes and retain A elsewhere."""
from collections import Counter
from pathlib import Path
from tools import cap_ab_patch_gate as test

exp=test.exp


def rollout(fixture,enable_recovery):
    a,b=[exp.s.phase.FrozenCAP(m) for m in (exp.AM,exp.BM)]
    counts=Counter()
    changes=0
    with exp.s.phase.world(fixture) as (w,_):
        def update(h):
            a.update(h)
            b.update(h)
        update(w.history)
        while not exp.s.ended(w):
            obs=exp.s.observe(w)
            baseline=a.propose(obs)
            command=baseline
            if (b.memory['recovery']['proposal'] is not None)==enable_recovery:
                command=b.propose(obs)
                changes+=command!=baseline
            exp.s.tick_action(w,command,exp.g.prep.prior.EXECUTOR,update,counts)
        result=exp.s.output(w,counts)
        return dict(success=result['success'],history=result['history'],changed=changes)


def room(job):
    key,f=job
    path=test.OUT/'controls'/(key+'.json')
    if path.exists():return
    src=exp.read(exp.OUT/'test'/(key+'.json'))
    result={}
    for name,value in [('recovery',True),('default',False)]:
        r=rollout(f,value)
        if not r['changed']:assert r['history']==src['A']['result']['history']
        result[name]=r
    exp.save(path,result)


if __name__=='__main__':
    exp.init(Path(exp.read(exp.OUT/'member.json')['path']).read_text())
    exp.pool(room,test.graph.fresh_panel().items(),8)
