"""D008 offline endpoint/reference audit.  No env, no Pymunk, no scan_lidar."""
from __future__ import annotations
import csv, hashlib, json, math, statistics as S
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "outputs/agent_exchange/D007_20261030_002"
OUT = ROOT / "outputs/agent_exchange/D008_20261030_003"
WIDTH, HEIGHT = 1.0, 0.25


def dot(a,b): return a[0]*b[0]+a[1]*b[1]
def norm(a): return math.hypot(*a)
def unit(a):
    z=norm(a); return (a[0]/z,a[1]/z) if z>1e-12 else (0.,0.)
def rot(a): return (math.cos(a),math.sin(a))
def t90(a): return (-a[1],a[0])


def truth_faces(obj):
    x,y,th=obj[0],obj[1],obj[2]
    faces=[]
    for name, mid_local, n_local, t_local, L in (
        ('bottom',(0,-HEIGHT/2),(0,1),(1,0),WIDTH),
        ('top',(0,HEIGHT/2),(0,-1),(1,0),WIDTH),
        ('right',(WIDTH/2,0),(-1,0),(0,1),HEIGHT),
        ('left',(-WIDTH/2,0),(1,0),(0,1),HEIGHT),
    ):
        c,s=math.cos(th),math.sin(th)
        mid=(x+c*mid_local[0]-s*mid_local[1], y+s*mid_local[0]+c*mid_local[1])
        n=(c*n_local[0]-s*n_local[1], s*n_local[0]+c*n_local[1])
        t=(c*t_local[0]-s*t_local[1], s*t_local[0]+c*t_local[1])
        eps=[(mid[0]+t[0]*L/2,mid[1]+t[1]*L/2),(mid[0]-t[0]*L/2,mid[1]-t[1]*L/2)]
        eps=sorted(eps,key=lambda p:dot(p,t))
        faces.append(dict(name=name,mid=mid,n=n,t=t,L=L,eps=eps))
    return faces


def choose_face(e,obj):
    n=e.get('face_normal'); pts=e.get('face_points') or []
    if not n or not pts: return None, 'no_face'
    n=unit(n); th=math.atan2(n[1],n[0])
    cand=[]
    for f in truth_faces(obj):
        ang=abs(math.atan2(math.sin(th-math.atan2(f['n'][1],f['n'][0])),
                           math.cos(th-math.atan2(f['n'][1],f['n'][0]))))
        plane=sum(dot(p,f['n']) for p in pts)/len(pts)-dot(f['mid'],f['n'])
        cand.append((ang,abs(plane),f))
    cand.sort(key=lambda z:z[0])
    if not cand: return None,'no_face'
    best=cand[0]
    second=cand[1] if len(cand)>1 else (math.pi,0,None)
    if best[0]>math.radians(15) or (second[0]-best[0])<math.radians(10) or best[1]>.01:
        return None,'ambiguous'
    return best[2],'ok'


def load_runs():
    runs=[]
    for p in sorted((SRC/'runs').glob('*.json')):
        d=json.loads(p.read_text())
        if d['tag']=='nonint': continue
        runs.append(d)
    return runs


def effect(t,side):
    return ((t.get('p3',{}).get(str(side),{}) or {}).get('info',{}) or {}).get('effect') or {}


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    runs=load_runs()
    total_ticks=sum(len(r['result']['ticks']) for r in runs)
    side_ticks=total_ticks*2
    ledger=[]; missing=[]; ambiguous=[]; meas=[]
    for run in runs:
        for t in run['result']['ticks']:
            obj=(t.get('p4') or {}).get('before',{}).get('object')
            if obj is None: continue
            for side in (0,1):
                e=effect(t,side)
                rec=e.get('endpoint')
                row=dict(run=run['id'],case=run['case'],profile=run['profile'],tick=t['tick'],
                         side=side,skill_uid=(e.get('skill_uid','')),stage=e.get('stage'))
                if not rec:
                    missing.append(row); continue
                f,status=choose_face(e,obj)
                if status!='ok':
                    ambiguous.append(dict(row)); row['face']=status; ledger.append(row); continue
                row['face']=f['name']
                n=unit(e['face_normal']); t_=t90(n); L=math.dist(rec['endpoints'][0],rec['endpoints'][1]) if all(rec.get('endpoints') or []) else 2*abs(e.get('anchor_distance') or 0)
                eps_true=sorted(f['eps'],key=lambda p:dot(p,t_))  # same +t as fitted face
                # measurement intervals
                for m in rec.get('measurements') or []:
                    s=m.get('side'); true=eps_true[s]
                    if not m.get('observed') or m.get('point') is None:
                        meas.append(dict(row, **{'m_side':s,'kind':m.get('kind'),'observed':0}))
                        continue
                    q0,q1=dot(m['lower'],t_),dot(m['upper'],t_); qt=dot(true,t_)
                    lo,hi=min(q0,q1),max(q0,q1)
                    cov=bool(lo-1e-9<=qt<=hi+1e-9)
                    outside=max(0.,lo-qt,qt-hi)
                    mid=dot(m['point'],t_)
                    meas.append(dict(row, **{'m_side':s,'kind':m.get('kind'),'observed':1,
                        'width':abs(hi-lo),'cover':int(cov),'outside':outside,
                        'mid_err':abs(mid-qt),'true_q':qt}))
                # tracking/reference
                eps=rec.get('endpoints') or [None,None]
                A=rec.get('anchor'); s=1 if rec.get('anchor_side')==0 else -1
                Mhat=(A[0]+t_[0]*s*L/2, A[1]+t_[1]*s*L/2) if A else None
                Atrue=eps_true[rec.get('anchor_side',0)]
                Eanc=math.dist(A,Atrue) if A else None
                Mtrue=f['mid']; Ltrue=f['L']; ttrue=f['t'] if dot(f['t'],t_)>=0 else (-f['t'][0],-f['t'][1])
                Mtrue_hat=(Atrue[0]+ttrue[0]*s*Ltrue/2, Atrue[1]+ttrue[1]*s*Ltrue/2)
                # use true midpoint directly; identity residual
                Evec=(Mhat[0]-Mtrue[0],Mhat[1]-Mtrue[1]) if Mhat else None
                refmid=e.get('face_midpoint'); ref_res=math.dist(Mhat,refmid) if Mhat and refmid else None
                tgt=e.get('target_before_visibility'); q=e.get('offset',0)+e.get('role_offset',0)+e.get('q_rot',0)
                target_rec=(tgt[0]-t_[0]*q,tgt[1]-t_[1]*q) if tgt else None
                tgt_res=math.dist(Mhat,target_rec) if Mhat and target_rec else None
                row.update(dict(face=f['name'],L_hat=L,L_true=Ltrue,anchor_err=Eanc,
                    ref_mid_err=ref_res,target_recon_err=tgt_res,
                    tangent_ref_err=dot(Evec,t_) if Evec else None,
                    normal_ref_err=dot(Evec,n) if Evec else None,
                    interval_cover=sum(1 for m in meas[-2:] if m.get('cover')) if len(meas)>=2 else None,
                    anchor_switch=bool(rec.get('anchor_switched')),endpoint_status=rec.get('status')))
                ledger.append(row)
    # stats
    def q(vals,p):
        vals=[v for v in vals if v is not None]
        if not vals: return None
        vals=sorted(vals); return vals[min(len(vals)-1,max(0,int(round(p*(len(vals)-1)))))]
    def stat(key):
        vals=[r.get(key) for r in ledger if r.get(key) is not None]
        return dict(n=len(vals),median=q(vals,.5),p95=q(vals,.95),max=max(vals) if vals else None)
    obs=[m for m in meas if m.get('observed')]
    summary=dict(
        runs=len(runs),control_ticks=total_ticks,side_ticks=side_ticks,
        endpoint_rows=len([r for r in ledger if r.get('endpoint_status')]),
        side_without_endpoint=len(missing),ambiguous_or_unbound=len(ambiguous),
        measured_rows=len(obs),measurement_coverage=sum(m.get('cover',0) for m in obs)/max(1,len(obs)),
        outside_gt_1cm=sum(1 for m in obs if (m.get('outside') or 0)>.01),
        outside_gt_2cm=sum(1 for m in obs if (m.get('outside') or 0)>.02),
        width=stat('width') if False else None,
        anchor_err=stat('anchor_err'),tangent_ref_err=stat('tangent_ref_err'),
        normal_ref_err=stat('normal_ref_err'),ref_mid_err=stat('ref_mid_err'),
        target_recon_err=stat('target_recon_err'),L_hat=stat('L_hat'),
        anchor_switch_events=sum(1 for r in ledger if r.get('anchor_switch')),
    )
    # interval widths/nonfinite missing
    widths=[m.get('width') for m in obs if m.get('width') is not None]
    summary['measurement_width']=dict(n=len(widths),median=q(widths,.5),p95=q(widths,.95),max=max(widths) if widths else None)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=1))
    with (OUT/'side_ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in ledger for k in r})); w.writeheader(); w.writerows(ledger)
    with (OUT/'measurement_ledger.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=sorted({k for r in meas for k in r})); w.writeheader(); w.writerows(meas)
    (OUT/'missing.json').write_text(json.dumps(dict(missing=missing,ambiguous=ambiguous),indent=1))
    (OUT/'manifest.json').write_text(json.dumps(dict(task='D008',head=__import__('subprocess').check_output(['git','rev-parse','HEAD']).decode().strip(),
        input_sha256={str(p.relative_to(SRC)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((SRC/'runs').glob('*.json'))},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),summary=summary),indent=1))
    print(json.dumps(summary,indent=1))


if __name__=='__main__':
    main()
