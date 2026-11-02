"""Compare the completed run13 short trial with the first matching run12 episodes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
P12=ROOT/'outputs/option_rl_v12/continuous_5hz_pilot_20261029/run12_001'
P13=ROOT/'outputs/option_rl_v13/short_lifecycle_20261029_130435/run13_001'

def read(p):return json.loads(p.read_text())
def mean(rows,key):
    values=[r[key] for r in rows if r.get(key) is not None]
    return float(np.mean(values)) if values else None

def collect(path,n):
    history=read(path/'history.json')[:n];evaluations=[];retention=Counter()
    for h in history:
        if h.get('success_rate') is None:continue
        ev=read(path/'evaluations'/f"iteration_{h['iteration']:06d}.json")
        phases=Counter();failed=waiting=total=0;gain=[]
        for s in ev['roots']:
            if s['initial_distance']>.2:
                gain.append(s['initial_distance']-s['final_distance'])
                if s['success']:retention[s['name']]+=1
            for a in s['actions']:
                for ex,phase in zip(a['exits'],a['phases']):
                    phases[phase]+=1;total+=1;failed+=ex!='continuing'
                    waiting+=ex!='continuing' or phase in ('peer_wait','perception_wait')
        row={k:h[k] for k in ('iteration','success_rate','eval_outside_goal_success','mean_progress')}
        row.update(failed=failed/total,failed_or_wait=waiting/total,
            push=phases['push']/total,brake=phases['brake']/total,peer_wait=phases['peer_wait']/total,
            mean_distance_gain_outside=float(np.mean(gain)))
        evaluations.append(row)
    best=max(evaluations,key=lambda r:(r['success_rate'],r['mean_progress']))
    summary=dict(iterations=len(history),episodes=sum(h['attempted_episodes'] for h in history),
        transitions=sum(h['steps'] for h in history),seconds_last_history=history[-1]['seconds'],
        simulated_seconds=sum(h['steps']*h['option_mean_seconds'] for h in history),
        train_anomalies=sum(h['anomaly_episodes'] for h in history),
        eval_anomalies=sum(h.get('eval_anomalies',0) for h in history),
        best=best,last_validation=evaluations[-1],
        positive_progress_evaluations=sum(e['mean_progress']>0 for e in evaluations),
        eval_count=len(evaluations),retention=dict(retention),windows={})
    for name,train,ev in [('all',history,evaluations),('last50',history[-50:],evaluations[-5:])]:
        summary['windows'][name]=dict(train={k:mean(train,k) for k in ['explore_progress',
            'train_outside_goal_success','mode_switch_commit_rate','switching_state_rate','entropy']},
            validation={k:mean(ev,k) for k in ['eval_outside_goal_success','mean_progress',
                'failed','failed_or_wait','push','brake','peer_wait','mean_distance_gain_outside']})
    return summary,history,evaluations

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);n=len(read(P13/'history.json'))
    old,h12,e12=collect(P12,n);new,h13,e13=collect(P13,n)
    manifests=[read(p/'manifest.json') for p in (P12,P13)]
    same_panels=all(manifests[0][k]==manifests[1][k] for k in ('validation_instances','test_instances'))
    assert same_panels and manifests[0]['reward_config']==manifests[1]['reward_config']
    mismatch={str(path):[s for s,d in m['source_sha256'].items() if hashlib.sha256((ROOT/s).read_bytes()).hexdigest()!=d]
        for path,m in zip((P12,P13),manifests)}
    assert not any(mismatch.values())
    budget=read(P13/'result.json')['seconds'];wall_rows=[h for h in read(P12/'history.json') if h['seconds']<=budget]
    report=dict(run12=old,run13=new,same_validation_test_panels=True,same_reward_config=True,
        source_mismatches=mismatch,equal_wall_run12_iterations=len(wall_rows),
        equal_wall_run12_episodes=sum(h['attempted_episodes'] for h in wall_rows),
        notes=['single training seed; fixed validation repeated, not independent replicates',
               'phase push is not an independently measured physical-contact duration',
               'run13 mode changes include explicitly changed failed-intent retries'])
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2))
    fig,axs=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    for label,h,e,color in [('run12',h12,e12,'#1d78b5'),('run13',h13,e13,'#d5632b')]:
        x=[r['iteration'] for r in e]
        for ax,key,title in [(axs[0,0],'eval_outside_goal_success','Fixed validation: success outside goal'),
                             (axs[0,1],'mean_progress','Fixed validation: goal progress')]:
            ax.plot(x,[r[key] for r in e],'.-',label=label,color=color);ax.set_title(title)
        axs[1,0].plot(x,[r['failed_or_wait'] for r in e],'.-',label=label,color=color)
        axs[1,1].plot(x,[r['push'] for r in e],'.-',label=label,color=color)
    axs[0,1].axhline(0,color='gray',lw=.8)
    axs[1,0].set_title('Validation: failed OR waiting state share')
    axs[1,1].set_title('Validation: PUSH phase share (not contact time)')
    for ax in axs.flat:
        ax.grid(alpha=.2);ax.legend();ax.set_xlabel('Training iteration (8 episodes each)')
    fig.savefig(a.out/'run13_vs_run12.png',dpi=160)
    print(json.dumps({k:report[k] for k in ('equal_wall_run12_iterations','same_validation_test_panels','same_reward_config')},indent=2))

if __name__=='__main__':main()
