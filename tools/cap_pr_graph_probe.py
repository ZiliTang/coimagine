"""Paired one-command activation at the first PUSH event in a frozen stall window."""
from tools import cap_pr_graph_round as r
from tools.cap_pr_graph_report import write_csv


def list_events(job):
    k,f,start,end=job
    _,rec=r.host.episode((k,f,True,None,'boundary',False,False,None))
    assert rec['result']==r.read(r.OUT/'runs/confirm/P/full'/(k+'.json'))['result']
    windows=[x for x in rec['events'] if x['kind'] in ('contact','progress') and start<=x['tick']<=end]
    return k,dict(event=windows[0] if windows else None,count=len(windows))


def probe_event(job):
    k,f,t,kind,force,hook,parent=job
    class Receipt:
        def __init__(self):self.command=None
        def boundary(self,*args):pass
        def checkpoint(self,obs,memory,proposed,running,reason):
            if obs['tick']==t and reason==kind:
                self.command=proposed
    rc=Receipt()
    _,v=r.host.episode((k,f,False,t,kind,force,hook,parent),rc)
    return k,dict(reached=v['reached'],result=v['result'],at_anchor=rc.command)


def native():
    r.configure()
    lock=r.read(r.OUT/'locked.json')
    src=r.host.seal(lock['name'])
    if not r.host.enabled(src):
        raise ValueError('submitted program has no PUSH checkpoint hook')
    parent=r.host.seal('P')
    fixtures=r.read(r.OUT/'fixtures.json')
    jobs=[]
    for k,f in fixtures.items():
        a=r.read(r.OUT/'runs/confirm/P/full'/(k+'.json'))
        h=a['anchor']
        if h:
            jobs.append((k,f,a['effects'][h['prefix']]['tick'],a['effects'][h['window_end']]['tick']))
    r.host.old.collect(parent,jobs,list_events,12,r.OUT/'native_anchors')
    anchors={k:e for k,*_ in jobs if (e:=r.read(r.OUT/'native_anchors'/(k+'.json'))['event']) is not None}
    for arm,prog,force in [('old',parent,False),('new',src,False),('force',src,True)]:
        runs=[(k,fixtures[k],a['tick'],a['kind'],force,r.host.enabled(prog),parent) for k,a in anchors.items()]
        if runs:r.host.old.collect(prog,runs,probe_event,12,r.OUT/'native_probes'/arm)
    rows=[]
    for k,anchor in anchors.items():
        a,b,c=[r.read(r.OUT/'native_probes'/tag/(k+'.json')) for tag in ['old','new','force']]
        assert a['reached']==b['reached']==c['reached'] and a['result']==r.read(r.OUT/'runs/confirm/P/full'/(k+'.json'))['result']
        rows.append(dict(room=k,tick=anchor['tick'],kind=anchor['kind'],prefix=a['reached']['history'],
            observation=a['reached']['observation_sha256'],old_win=a['result']['success'],natural_win=b['result']['success'],
            forced_win=c['result']['success'],forced_proposal=c['at_anchor'] is not None,
            changed=int(b['result']['history']!=c['result']['history'])))
    write_csv('native_probe.csv',rows)
    r.save(r.OUT/'native_probe.json',dict(cluster_rooms=len(jobs),applicable=len(anchors),
        force_offers=sum(x['forced_proposal'] for x in rows),changed=sum(x['changed'] for x in rows),
        wins=[sum(x[col] for x in rows) for col in ('old_win','natural_win','forced_win')]))


if __name__=='__main__':native()
