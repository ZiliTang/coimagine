import json, math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/agent_exchange/D007_20261030_002'
CH=OUT/'chains'; CH.mkdir(exist_ok=True)

def load(rid):
    return json.loads((OUT/'runs'/f'{rid}.json').read_text())

def series(d):
    ts=[]; ang=[]; tx=[]; rx=[]; et=[]; lat=[]
    for t in d['result']['ticks']:
        a=t.get('p4',{}).get('after');
        if not a: continue
        ts.append(t['t_start']); ang.append(a['object'][2]); rx.append(a['robots'][0][0])
        e=t['p3']['0']['info'].get('effect',{})
        tg=e.get('target_final') or e.get('contact_target')
        tx.append(tg[0] if tg else math.nan)
        et.append(e.get('e_t',math.nan)); lat.append(e.get('lateral',math.nan))
    return ts,ang,tx,rx,et,lat

for rid in ('013_probe','014_probe','017_probe','018_probe','021_probe','022_probe'):
    d=load(rid); ts,ang,tx,rx,et,lat=series(d)
    fig,ax=plt.subplots(3,1,figsize=(7,6),sharex=True)
    ax[0].plot(ts,tx,label='target x'); ax[0].plot(ts,rx,label='robot x'); ax[0].set_ylabel('x (m)'); ax[0].legend(fontsize=8)
    ax[1].plot(ts,et,label='e_t'); ax[1].plot(ts,lat,label='lateral'); ax[1].set_ylabel('m / m/s'); ax[1].legend(fontsize=8)
    ax[2].plot(ts,ang,label='object angle'); ax[2].set_ylabel('rad'); ax[2].legend(fontsize=8); ax[2].set_xlabel('t (s)')
    ax[0].set_title(f"{d['id']} {d['case']} {d['profile']}")
    fig.tight_layout(); fig.savefig(CH/f'{rid}.png',dpi=120); plt.close(fig)
print('charts',len(list(CH.glob('*.png'))))
