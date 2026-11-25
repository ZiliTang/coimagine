"""Evaluate trial11's declared contact-plus-progress contract at its first accepted offer."""
from unittest.mock import patch
from tools import cap_pr_graph_round as r
from tools.cap_pr_graph_report import write_csv


def episode(job):
    k,f,tag,src_parent,anchor=job
    natural=anchor is None
    accepted=[]
    offers=[]
    commits=[]
    original=r.ISO.IsolatedCAP.update
    def update(self,h):
        if self.broker is not r.ISO.BROKER:return original(self,h)
        before=(self.memory or {}).get('recovery',{})
        pending=before.get('pending') or {}
        ev=r.PREFIX.s.phase.cap.current_event(h)
        t=ev['observation']['tick']
        requested=ev.get('requested_action') or ev.get('action')
        if ('offset' in pending and h.previous_tick is not None and t>h.previous_tick
            and before.get('command')==requested):
            accepted.append(dict(t=pending['start'],ack=t))
        original(self,h)
        state=self.memory['recovery']
        p=state.get('pending') or {}
        if 'offset' in p and self.proposal==state.get('command'):
            offers.append(dict(t=t,history=h.sha256,proposal=self.proposal))
        plan=state.get('plan') or {}
        commits.append(dict(t=t,plan=plan.get('start') if 'offset' in plan else None,
                            release=state.get('release'),offer=bool(state.get('pending')),
                            recovery=bool(state.get('command') or state.get('plan'))))
    with patch.object(r.ISO.IsolatedCAP,'update',update):
        _,row=r.host.episode((k,f,True,anchor,'boundary',False,tag!='P',src_parent))
    boundaries=[{kk:x[kk] for kk in ('tick','gap','touch','execution','public_history','action')} for x in row['boundaries']]
    if natural:
        assert row['result']==r.read(r.OUT/'runs/confirm'/tag/'full'/(k+'.json'))['result']
    return k,dict(offers=offers,accepted=accepted,commits=commits,boundaries=boundaries,result=row['result'],reached=row['reached'])

def passing(boundaries,t,g0,end):
    touched=False
    first=None
    for b in boundaries:
        if not t<b['tick']<=end:continue
        if b['touch'] and 'push' in b['execution']:touched=True
        if touched and b['gap']>=0 and round((g0-b['gap'])*1000)>=180:
            if first is None:first=b['tick']
            if b['tick']-first>=15:return b['tick']
    return None

def main():
    r.configure()
    lock=r.read(r.OUT/'locked.json');name=lock['name']
    assert lock['sha256']=='e193d645615a77b13811d97d69ae387c304d67b79b8ccbdc041c4067e23d5d80'
    fixtures=r.read(r.OUT/'fixtures.json')
    src=r.host.seal(name);old=r.host.seal('P')
    root=r.OUT/'contract'
    r.host.old.collect(src,[(k,f,name,None,None) for k,f in fixtures.items()],episode,16,root/'natural')
    data={k:r.read(root/'natural'/(k+'.json')) for k in fixtures}
    first={k:x['accepted'][0]['t'] for k,x in data.items() if x['accepted']}
    for tag,prog in [('P',old),(name,src)]:
        jobs=[(k,fixtures[k],tag,src,t) for k,t in first.items()]
        if jobs:r.host.old.collect(prog,jobs,episode,12,root/tag)
    rows=[]
    for k,t in first.items():
        natural=data[k]; a,b=[r.read(root/tag/(k+'.json')) for tag in ('P',name)]
        assert a['reached']==b['reached'] and b['result']==natural['result']
        offer=next(x for x in natural['offers'] if x['t']==t)
        assert a['reached']['history']==offer['history']
        start=next(x for x in b['boundaries'] if x['tick']==t)
        base=next(x for x in a['boundaries'] if x['tick']==t)
        assert base['gap']==start['gap'] and start['gap']>=0
        ap=passing(a['boundaries'],t,start['gap'],t+265)
        bp=passing(b['boundaries'],t,start['gap'],t+265)
        release=next((v['t'] for v in b['commits'] if v['t']>t and v['plan']!=t and v['release']==v['t']),None)
        pgain=None
        if release is not None:
            next_offer=next((v['t'] for v in b['commits'] if v['t']>release and (v['offer'] or v['recovery'])),10**9)
            after=[v for v in b['boundaries'] if release<=v['tick']<=release+120 and v['tick']<next_offer and v['gap']>=0]
            if len(after)>=2 and after[0]['tick']==release:pgain=round(after[0]['gap']-after[-1]['gap'],3)
        rows.append(dict(room=k,tick=t,ack=natural['accepted'][0]['ack'],history=offer['history'],gap=start['gap'],
            parent_pass=ap is not None,candidate_pass=bp is not None,parent_confirm_tick=ap,candidate_confirm_tick=bp,
            release=release,pass_after_release=bp is not None and release is not None and release<=bp,
            P_resume_gain_m=pgain,parent_success=a['result']['success'],candidate_success=b['result']['success'],
            parent_unconfirmed_early=ap is None and a['result']['ticks']<t+265,
            candidate_unconfirmed_early=bp is None and b['result']['ticks']<t+265))
    write_csv('claim.csv',rows)
    offered=sum(bool(x['offers']) for x in data.values())
    summary=dict(rooms=len(data),offered=offered,accepted=len(first),offered_not_accepted=offered-len(first),
        matched=len(rows),old=sum(x['parent_pass'] for x in rows),new=sum(x['candidate_pass'] for x in rows),
        rescued=sum(x['candidate_pass'] and not x['parent_pass'] for x in rows),
        lost=sum(x['parent_pass'] and not x['candidate_pass'] for x in rows),
        released=sum(x['release'] is not None for x in rows),pass_after_release=sum(x['pass_after_release'] for x in rows),
        p_resume_seen=sum(x['P_resume_gain_m'] is not None for x in rows),
        p_resume_10cm=sum(x['P_resume_gain_m'] is not None and x['P_resume_gain_m']>=.10 for x in rows))
    r.save(r.OUT/'contract.json',summary)

if __name__=='__main__':main()
