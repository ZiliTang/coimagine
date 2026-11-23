"""Finite-budget reachability over measured, public, single-option transitions."""
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from tools import cap_authored_handoff as e

QUANTUM=5
STATES=48
FAIL,SUCCESS=48,49
MIN_SUPPORT=20
PSEUDO=16
TIME=1800//QUANTUM
BUDGET=300


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_member():
    selection=e.read(e.WORK/'selection.json')
    rev=selection['revision']
    probe=e.read(e.OUT/f'probe{rev}.json')
    path=e.CAM/'semantics'/(probe['sid']+'.py')
    src=path.read_text()
    assert e.base.digest(src)==probe['sha'] and src==(e.WORK/f'v{rev}.py').read_text()
    c=e.base.host.RoomCampaign(e.CAM)
    seal=c.verify_frozen() if e.read(e.CAM/'campaign.json')['status']=='frozen' else c.freeze()
    lock=dict(revision=rev,sid=probe['sid'],sha=probe['sha'],path=str(path),
              seal=seal,reason=selection['reason'],A=e.A)
    if (e.OUT/'member.json').exists():assert e.read(e.OUT/'member.json')==lock
    e.save(e.OUT/'member.json',lock)
    revisions=[e.read(p) for p in sorted(e.OUT.glob('probe[1-3].json'))]
    info='\n'.join(f"| {r['revision']} | {r['rescued']}/16 | {r['lost']}/8 |" for r in revisions)
    (e.DOC/'author.md').write_text(f"""# B 的开发与冻结

作者为 gpt-6.2-sol。参照 A 为 usual s003。本轮真实 A 前缀为 16 个最终失败和 8 个最终自行恢复，B 在同一时刻接手并执行到底。预算取最多三版，每版测试同一批 24 个房间。作者选定第 {rev} 版，选择理由如下。

{selection['reason']}

| 版本 | 救回原失败 | 丢失原成功 |
|---|---|---|
{info}

B 源码标识 `{probe['sha']}`，authoring_v2 清单标识 `{seal['sha256']}`。重复查询这批开发案例会带来选择偏差。泛化由独立房间检验。
""")
    return lock


def train_one(job):
    k,f,family=job
    path=e.OUT/'train'/(k+'.json')
    if path.exists():return
    modes={}
    ref=e.read(e.OUT/'baseline'/(k+'.json'))
    for arm in ('A','B','lock','mixed'):
        row=e.simulate(f,arm,family,record=True,seed=int(e.key(k)[:8],16))
        if arm=='A':assert row['result']['history']==ref['result']['history']
        if arm in ('lock','mixed'):assert row['hit']==modes['A']['hit']
        modes[arm]=row
    e.save(path,modes)


def fit_tables(counts):
    # Rows index (node,expert); column 49 is environment success.
    pooled=defaultdict(Counter)
    for (z,a),dest in counts.items():pooled[z].update(dest)
    mats=defaultdict(lambda:([],[],[]))
    support=np.zeros((STATES,2),int)
    for z in range(STATES):
        prior=pooled[z]
        if not prior:continue
        prior_n=sum(prior.values())
        for a in (0,1):
            obs=counts.get((z,a),{})
            total=sum(obs.values());support[z,a]=total
            for (dest,qt,dec),num in prior.items():
                if dest==FAIL:continue
                rr,cc,vv=mats[qt,dec]
                rr.append(z*2+a);cc.append(dest)
                vv.append((obs.get((dest,qt,dec),0)+PSEUDO*num/prior_n)/(total+PSEUDO))
    matrices={}
    for cost,(rows,cols,weights) in mats.items():
        matrix=np.zeros((STATES*2,50),np.float32)
        np.add.at(matrix,(rows,cols),weights)
        matrices[cost]=matrix
    return matrices,support


def solve(matrices,support,tmax=TIME,dmax=BUDGET,commit=None):
    value=np.zeros((tmax+1,STATES+2,dmax+1),np.float32)
    value[:,SUCCESS,:]=1
    policy=np.zeros((tmax+1,STATES,dmax+1),np.uint8)
    eligible=(support.min(1)>=MIN_SUPPORT)[:,None]
    for t in range(1,tmax+1):
        q=np.zeros((STATES*2,dmax+1),np.float32)
        for (dt,dc),m in matrices.items():
            if dt<=t and dc<=dmax:q[:,dc:]+=m@value[t-dt,:,:dmax+1-dc]
        a,b=q[0::2],q[1::2]
        take=eligible & (b>a+1e-6) if commit is None else bool(commit)
        policy[t]=take
        value[t,:STATES]=np.where(take,b,a)
    return policy if commit is None else value[:,:STATES]


def load_policy():
    global POLICY
    lock=e.read(e.OUT/'graph.json')
    assert file_sha(e.OUT/'policy.npy')==lock['policy_sha']
    POLICY=np.load(e.OUT/'policy.npy')


def choose(node,ticks,decisions):
    return POLICY[min(TIME,ticks//QUANTUM),node,min(BUDGET,decisions)]


def lock_graph():
    panel=e.dev();family=e.read(e.OUT/'family.json')
    expert=freeze_member()
    e.init(Path(expert['path']).read_text(), profile=e.perception.LEGACY)
    fitting={k:f for k,f in panel.items() if k not in family['refs']}
    assert len(fitting)==176
    e.pool(train_one,[(k,f,family['family']) for k,f in fitting.items()])
    counts=defaultdict(Counter)
    for k in sorted(fitting):
        arms=e.read(e.OUT/'train'/(k+'.json'))
        # Merge duplicate real boundaries reached under multiple behavior policies.
        # Distinct actions remain distinct even with a shared state history.
        seen=set()
        for arm,row in arms.items():
            assert len(row['edges'])==len(row['origin_hashes'])
            for h,(z,a,d,ticks,dc) in zip(row['origin_hashes'],row['edges']):
                if (h,a) in seen:continue
                seen.add((h,a))
                assert ticks>0 and dc>0
                counts[z,a][d,math.ceil(ticks/QUANTUM),dc]+=1
    matrices,support=fit_tables(counts)
    policy=solve(matrices,support)
    np.save(e.OUT/'policy.npy',policy)
    for expert_id in (0,1):
        estimates=solve(matrices,support,commit=expert_id)
        np.save(e.OUT/f'commit{expert_id}.npy',estimates.astype(np.float16))
    e.save(e.OUT/'transition_counts.json',[dict(z=z,expert=a,outcomes=[[*k,v] for k,v in cnt.items()]) for (z,a),cnt in counts.items()])
    lock=dict(member=expert['sha'],family=family['family'],rooms=len(fitting),
              transitions=int(support.sum()),support=support.tolist(),enabled_nodes=int(sum(support.min(1)>=MIN_SUPPORT)),
              quantum=QUANTUM,pseudo=PSEUDO,min_support=MIN_SUPPORT,policy_sha=file_sha(e.OUT/'policy.npy'),
              fixed_shas=[file_sha(e.OUT/f'commit{i}.npy') for i in (0,1)],model_sha=file_sha(e.OUT/'transition_counts.json'),code_sha=file_sha(__file__),runtime_sha=file_sha(e.__file__))
    e.save(e.OUT/'graph.json',lock)
    print('graph locked',lock['transitions'],lock['enabled_nodes'],flush=True)


def fresh_panel():
    fixtures={f"recovery_{e.SEED}_{r['id']}":r['fixture'] for r in e.s.phase.cap.rooms.generate(e.SEED)['cases']}
    seen=e.s.collect_identities()
    prior=e.read(e.old.previous.OUT/'panels.json')
    for p in prior['rooms'].values():seen.update(p.values())
    seen.update(e.read(e.old.OUT/'fresh_panel.json').values())
    for name in ('development','held'):seen.update(e.s.phase.input_hash(f) for f in e.read(e.s.OUT/(name+'.json')).values())
    for seed in (662315862,662315879,e.g.prep.SEED):seen.update(e.s.phase.input_hash(r['fixture']) for r in e.s.phase.cap.rooms.generate(seed)['cases'])
    ids={k:e.s.phase.input_hash(v) for k,v in fixtures.items()}
    assert len(set(ids.values()))==200 and not set(ids.values())&seen
    e.save(e.OUT/'confirmation_ids.json',ids)
    return fixtures


def test_one(job):
    k,f,family=job
    path=e.OUT/'test'/(k+'.json')
    if path.exists():return
    rows={}
    for mode in ('A','B','lock','graph'):
        if mode in ('lock','graph') and rows['A']['hit'] is None:
            rows[mode]=rows['A'];continue
        row=e.simulate(f,mode,family,chooser=choose)
        if mode in ('lock','graph'):
            assert row['hit']==rows['A']['hit']
            if row['first_take'] is None:assert row['result']['history']==rows['A']['result']['history']
        rows[mode]=row
    e.save(path,rows)


def evaluate():
    member=e.read(e.OUT/'member.json')
    seal=e.base.host.RoomCampaign(e.CAM).verify_frozen()
    assert seal==member['seal']
    src=Path(member['path']).read_text();assert e.base.digest(src)==member['sha']
    e.init(src, profile=e.perception.LEGACY);load_policy()
    fixtures=fresh_panel()
    e.pool(test_one,[(k,f,e.read(e.OUT/'graph.json')['family']) for k,f in fixtures.items()])


if __name__=='__main__':
    import sys
    if sys.argv[1]=='train':lock_graph()
    elif sys.argv[1]=='evaluate':evaluate()
