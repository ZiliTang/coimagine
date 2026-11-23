"""Show the paired routing differences and calibration of takeover estimates."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(rows,paired,path):
    fig,axs=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    ax=axs[0]
    for i,(k,label) in enumerate([('lock_A','At X, B vs A'),('graph_A','Graph vs A'),('graph_lock','Graph vs B at X')]):
        r=paired[k]
        pct=100*(r['plus']-r['minus'])/r['n']
        lo,hi=100*np.asarray(r['ci'])
        ax.errorbar(pct,i,xerr=[[max(0,pct-lo)],[max(0,hi-pct)]],fmt='o',color='#186861',capsize=3)
        ax.annotate(f"{r['plus']} rescued / {r['minus']} lost",(pct,i),xytext=(0,10),
                    textcoords='offset points',ha='center',fontsize=8)
    ax.axvline(0,color='.6',ls='--',lw=1)
    ax.set(yticks=range(3),yticklabels=['B at X vs A','Graph vs A','Graph vs B at X'],ylim=(-.6,2.65),
           xlabel='Net success difference (percentage points)',title='200 independent rooms')
    ax.spines[['top','right']].set_visible(False)
    act=[r for r in rows if r['trigger']]
    delta=np.array([r['estimate_B']-r['estimate_A'] for r in act])
    real=np.array([r['lock']-r['A'] for r in act])
    ids=np.argsort(delta,kind='stable')
    ax=axs[1]
    for i,batch in enumerate(np.array_split(ids,4)):
        if not len(batch):continue
        ax.scatter(i,100*delta[batch].mean(),marker='o',color='#287478',label='Model' if i==0 else None)
        ax.scatter(i,100*real[batch].mean(),marker='s',color='#b46132',label='Paired outcomes' if i==0 else None)
        ax.plot([i,i],100*np.array([delta[batch].mean(),real[batch].mean()]),lw=1,color='.65',zorder=0)
    ax.axhline(0,color='.6',ls='--',lw=1)
    ax.set(xticks=range(4),xticklabels=['Low','2','3','High'],xlabel='Predicted B advantage quartile',
           ylabel='B minus A success (percentage points)',title=f'Fixed continuations at first X (n={len(act)})')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(path,dpi=150);plt.close(fig)
