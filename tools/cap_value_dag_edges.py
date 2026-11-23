"""Audit one-window transition prediction and the support of the fitted suffix cells."""
from collections import defaultdict
import numpy as np

from tools import cap_value_dag as f
from tools.cap_value_dag_report import stat


def transitions(cases):
    result=[[],[]]
    for c in cases:
        for r,b in c['branches'].items():
            states=b['states']
            for i in range(min(len(states),2)):
                if i==0 and r[1]=='B':continue
                dest=states[i+1][0] if len(states)>i+1 else 6+int(not b['y'][2])
                result[i].append((c['room'],states[i],r[i],dest))
    return result


def audit(fold,training):
    cases=f.dataset(fold)
    res={}
    for stage,(a,b) in enumerate(zip(training,transitions(cases))):
        for mode in ('physical','physical_history','public','public_history'):
            tab={}
            for _,src,cap,j in a:f.add(tab,src,cap,mode,np.eye(8)[j])
            out=[]
            for room,src,cap,j in b:
                p=f.lookup(tab,src,cap,mode); marginal=f.lookup(tab,src,cap,'constant')
                obs=np.eye(8)[j]
                n=tab.get(f.keys(src,cap,mode)[-1],[0])[0]
                out.append((room,sum((p-obs)**2),sum((marginal-obs)**2),int(p.argmax()==j),n))
            ids=np.array([row[0] for row in out]);v=np.array([row[1:] for row in out])
            res[f'{stage+1}_{mode}']=dict(brier=stat(v[:,0],ids),gain_vs_cap=stat(v[:,1]-v[:,0],ids),
                        accuracy=stat(v[:,2],ids),zero_cell=int(sum(v[:,3]==0)),ge8=int(sum(v[:,3]>=8)),
                        median_cell=float(np.median(v[:,3])),n=len(v))
    return res


if __name__=='__main__':
    training=transitions(f.dataset('dev'))
    f.save(f.DOC/'transitions.json',{k:audit(k,training) for k in ('diagnostic','confirm')})
