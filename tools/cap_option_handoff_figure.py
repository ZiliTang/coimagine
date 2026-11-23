"""Plot counts and paired outcomes of the independently tested handoff."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools import cap_option_handoff as exp


def plot(metrics):
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained',
                          gridspec_kw={'width_ratios':[1,1.15]})
    fresh=metrics['fresh']
    labels=['Development','New rooms']
    for ax,k,name in [(axes[0],'formation','First proposal establishes Y'),
                      (axes[1],'local','One proposal advances object ≥2 cm\nafter A actually established Y')]:
        groups=[metrics[s][k] for s in ('dev','fresh')]
        for i,key in enumerate(('first','second')):
            xx=[j+(-.17 if i==0 else .17) for j in range(2)]
            rates=[100*r[key]/max(r['n'],1) for r in groups]
            bars=ax.bar(xx,rates,.31,color=('#226C72','#C97F36')[i],
                        label=(('A','B') if k=='formation' else ('B takes over','A continues'))[i])
            ax.bar_label(bars,labels=[f"{r[key]}/{r['n']}" for r in groups],padding=3,size=9)
        ax.set(title=name,xticks=[0,1],xticklabels=labels,ylim=(0,115),ylabel='Share of paired rooms (%)')
        ax.spines[['right','top']].set_visible(False)
        ax.legend(loc='upper right',frameon=False)
    fig.suptitle(f"Single-option handoff   ·   {fresh['terminal']['plus']} rescued / {fresh['terminal']['minus']} lost vs A in 200 new rooms")
    fig.savefig(exp.DOC/'handoff.png',dpi=155)
    plt.close(fig)
