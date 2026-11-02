"""Plot the frozen-policy diagnostic; no policy or experiment changes."""
import argparse,json
from pathlib import Path
import numpy as np


def plot(source):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    source=Path(source);summary=json.loads((source/'summary.json').read_text())
    episodes=[json.loads(p.read_text()) for p in sorted((source/'episodes').glob('*.json'))]
    fig,axes=plt.subplots(1,3,figsize=(15,4))
    for e in episodes:
        if e['greedy'] and e['name'] in ('contact_0','approach_2','dual_0'):
            r=[r for r in e['records'] if r['time']<=16]
            axes[0].plot([a['time'] for a in r],[a['distance'] for a in r],label=e['name'],marker='.')
    axes[0].axhline(.2,color='black',linestyle='--',label='goal radius')
    axes[0].set(title='Greedy policy passes the goal',xlabel='Physical seconds',ylabel='Distance to goal (m)');axes[0].legend(fontsize=8)
    samples=[r for e in episodes if not e['greedy'] for r in e['records'] if r['distance']<.35 and r['closing_speed']>.01]
    axes[1].scatter([r['distance'] for r in samples],[r['brake_pair_probability'] for r in samples],
        c=['#e14d43' if r['both_brake'] else '#8ba8c3' for r in samples],s=12,alpha=.6)
    axes[1].set(title=f'Near-goal joint BRAKE probability; draws {sum(r["both_brake"] for r in samples)}/{len(samples)}',
        xlabel='Distance to goal (m)',ylabel='P(BRAKE, BRAKE)')
    forks=[r for r in summary['forks'] if r['threshold']==.18 and not r['baseline_success']]
    x=np.arange(len(forks));axes[2].bar(x-.18,[r['baseline_return'] for r in forks],.36,label='greedy baseline')
    axes[2].bar(x+.18,[r['brake_return'] for r in forks],.36,label='offline BRAKE fork')
    axes[2].set_xticks(x,[r['name'] for r in forks],rotation=15);axes[2].set(title='Same-prefix counterfactuals',ylabel='Unchanged task return');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Frozen v14 final policy; developer tasks; truth only for offline labels and fork selection')
    fig.tight_layout();fig.savefig(source/'stopping_diagnosis.png',dpi=150);plt.close(fig)
    return source/'stopping_diagnosis.png'


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    print(plot(p.parse_args().source))
