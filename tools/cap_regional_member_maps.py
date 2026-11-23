"""Regional development coverage by frozen member on the dashboard's shared PCA."""
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import font_manager
import numpy as np

from tools import cap_regional_dashboard as ui

COLORS={'previous_failure':'#d1d5d4','lost':'#b86a41','retained':'#177b79','added':'#5146aa'}
LEGEND={'previous_failure':'旧库与本轮都未解决','lost':'旧库成功、本轮失败',
        'retained':'旧库和本轮均成功','added':'本轮新增成功（相对旧库）'}

def selection_map(sweeps,directory):
    data=ui.read(ui.OUT/'regions.json')
    ids=data['ids'];x=np.array(data['x'])
    z=(x-x.mean(axis=0))/np.maximum(.1,x.std(axis=0))
    _,singular,basis=np.linalg.svd(z,full_matrices=False)
    for component in basis:
        if component[np.argmax(np.abs(component))]<0:component*=-1
    projection=z@basis[:2].T
    variance=100*singular**2/(singular**2).sum()
    distances=np.linalg.norm(z[:,None]-z[None,:],axis=2)
    neighbors=np.argsort(distances,axis=1,kind='stable')[:,:24]
    radius=distances[np.arange(len(ids)),neighbors[:,-1]]
    eligible=radius<=np.median(radius)
    membership=np.array([[int(k in row['focus']) for k in ids] for row in sweeps])
    overlap=membership@membership.T
    fig,(left,right)=plt.subplots(1,2,figsize=(12,5.3),gridspec_kw={'width_ratios':[1.4,1]})
    fig.subplots_adjust(top=.82,bottom=.16,wspace=.27)
    fig.suptitle('五次选区在开发时使用的九维公开特征中有多接近？',fontsize=17)
    for mask,color,label in [(~eligible,'#c9cfcd','其他房间'),(eligible,'#4b8985','密度合格的 80 个候选中心')]:
        left.scatter(*projection[mask].T,s=22,color=color,label=label)
    centers={}
    stats=[];past=set();seen=set()
    for i,row in enumerate(sweeps):
        ci=ids.index(row['center'])
        centers.setdefault(ci,[]).append(str(i+1))
        opened=np.array([k not in past for k in ids]);counts=opened[neighbors].sum(axis=1)
        assert set(row['focus'])=={ids[j] for j in neighbors[ci]}
        stats.append(dict(round=i+1,center=ids[ci],center_unresolved=bool(opened[ci]),max_open=int(counts[eligible].max()),
                          tie_count=int((counts[eligible]==counts[ci]).sum()),first_time_in_focus=len(set(row['focus'])-seen)))
        past.update(row['wins']);seen.update(row['focus'])
    for ci,rounds in centers.items():
        p=projection[ci];left.scatter(*p,marker='*',s=160,c='#e9a03f',edgecolors='#6f4618',zorder=4)
        left.annotate('、'.join(rounds),p,xytext=(5,7),textcoords='offset points',fontsize=11,weight='bold',zorder=5)
    left.set(xlabel=f'选区 PC1（{variance[0]:.1f}%）',ylabel=f'选区 PC2（{variance[1]:.1f}%）',title='星号旁数字为轮次')
    left.grid(alpha=.12);left.legend(fontsize=9,frameon=False)
    right.imshow(overlap,cmap='GnBu',vmin=0,vmax=24)
    for i,j in np.ndindex(overlap.shape):
        right.text(j,i,str(overlap[i,j]),ha='center',va='center',color='white' if overlap[i,j]>15 else '#102f35',fontsize=13)
    right.set(xticks=range(5),yticks=range(5),xticklabels=range(1,6),yticklabels=range(1,6),
              xlabel='轮次',ylabel='轮次',title='两轮重点名单共有多少房间（各 24 个）')
    fig.text(.095,.025,'左图投影的是实际选区特征。右图直接比较名单，保留全部九维距离给出的近邻关系。',fontsize=10)
    fig.savefig(directory/'regional_selection_geometry.png',dpi=140);plt.close(fig)
    (directory/'regional_selection_geometry.json').write_text(json.dumps(dict(rounds=stats,overlap=overlap.tolist(),
        unique_focus=len(seen),selection_pca_variance=variance[:2].tolist()),indent=2)+'\n')


def main():
    sweep=ui.read(ui.OUT/'regional/study.json')['sweeps']
    assert len(sweep)==5
    with (ui.ROOT/'docs/research/key_findings/2026-11-19_cap_geometry/scenes.csv').open() as f:
        pts={r['case_id']:(float(r['pc1']),float(r['pc2'])) for r in csv.DictReader(f)
             if re.fullmatch(r'dev_[2-5]_\d+',r['case_id'])}
    assert len(pts)==160
    names=sorted(pts);xy=np.array([pts[k] for k in names]);root=set(names)
    wins=[set(r['wins']) for r in sweep]
    font=Path.home()/'.local/share/fonts/NotoSansSC-Regular.otf'
    if font.is_file():font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':['Noto Sans SC','DejaVu Sans'],'axes.unicode_minus':False})
    fig,axes=plt.subplots(2,3,figsize=(15,9.4),sharex=True,sharey=True)
    fig.subplots_adjust(left=.058,right=.982,top=.82,bottom=.09,wspace=.13,hspace=.44)
    fig.suptitle('区域作者五个 CAP 的成功分布',x=.057,y=.982,ha='left',fontsize=23,weight='bold')
    fig.text(.059,.927,'同一批 160 个开发初态 · 每个点代表一个房间 · 黑圈为当轮 24 个重点近邻，星号为中心',fontsize=12)
    legend=[Line2D([],[],marker='o',ls='',color=COLORS[k],markersize=7,label=LEGEND[k]) for k in ['added','retained','lost','previous_failure']]
    fig.legend(handles=legend,loc='upper left',bbox_to_anchor=(.053,.913),ncol=4,frameon=False,fontsize=11)
    cov=set();summary=[];rows=[]
    pad=np.ptp(xy,axis=0)*.065
    for i,(r,win,ax) in enumerate(zip(sweep,wins,axes.flat)):
        focus=set(r['focus']); added=win-cov
        category={k:('retained' if k in cov else 'added') if k in win else ('lost' if k in cov else 'previous_failure') for k in names}
        for name,color in COLORS.items():
            data=np.array([pts[k] for k in names if category[k]==name]).reshape(-1,2)
            ax.scatter(*data.T,s=23 if name!='added' else 36,facecolor=color,linewidth=.4,edgecolor='white',zorder=2)
        marks=np.array([pts[k] for k in focus])
        ax.scatter(*marks.T,s=78,facecolor='none',linewidth=.84,edgecolor='#202727',zorder=3)
        ax.scatter(*pts[r['center']],s=152,marker='*',edgecolor='#101010',facecolor='#fff',linewidth=1.1,zorder=4)
        ax.set_xlim(xy[:,0].min()-pad[0],xy[:,0].max()+pad[0]);ax.set_ylim(xy[:,1].min()-pad[1],xy[:,1].max()+pad[1])
        ax.set_title(f"第 {i+1} 轮  {r['sid']}   ·   {len(win)}/160",loc='left',fontsize=15,weight='bold',pad=23)
        ax.text(0,1.025,f'重点区成功 {len(win&focus)}/24    本轮新增 {len(added&focus)} 区内 / {len(added-focus)} 区外',
                transform=ax.transAxes,fontsize=10)
        ax.grid(alpha=.13);ax.spines[['top','right']].set_visible(False)
        if i%3==0:ax.set_ylabel('PC2（8.66%）')
        if i in (2,3,4):ax.set_xlabel('PC1（9.94%）');ax.tick_params(axis='x',labelbottom=True)
        other=set().union(*(w for j,w in enumerate(wins) if j!=i))
        summary.append(dict(round=i+1,sid=r['sid'],success=len(win),new=len(added),retained=len(win&cov),
                            lost=len(cov-win),new_focus=len(added&focus),new_outside=len(added-focus),
                            focus_success=len(win&focus),focus_previous_member=len(wins[i-1]&focus) if i else None,
                            old_unresolved_focus=len(focus-cov),unique=len(win-other)))
        for k in names:
            rows.append(dict(round=i+1,sid=r['sid'],case_id=k,pc1=pts[k][0],pc2=pts[k][1],
                             focus=int(k in focus),center=int(k==r['center']),success=int(k in win),
                             category=category[k],unique=int(k in win-other)))
        cov|=win
    ax=axes.flat[-1];ax.axis('off')
    ax.text(0,.98,'同一重点区内的单成员表现',transform=ax.transAxes,va='top',fontsize=15,weight='bold')
    cells=[[r['round'],'—' if r['focus_previous_member'] is None else r['focus_previous_member'],r['focus_success'],r['unique']] for r in summary]
    table=ax.table(cellText=cells,colLabels=['轮次','上轮 / 24','本轮 / 24','全池独胜'],cellLoc='center',bbox=[0,.18,1,.63])
    table.auto_set_font_size(False);table.set_fontsize(11)
    for (rr,cc),cell in table.get_celld().items():
        cell.set_edgecolor('#d8dedd');cell.set_linewidth(.6)
        if rr==0:cell.set_facecolor('#ecf4f2')
    ax.text(0,.05,'独胜为五个成员里仅该成员成功的房间数。',transform=ax.transAxes,fontsize=10)
    fig.text(.059,.026,'选区使用 9 维公开几何。此图沿用 28 维离线几何的前两个主成分，圈住的近邻在图上可能分散。',fontsize=11)
    d=ui.FINDINGS;d.mkdir(parents=True,exist_ok=True)
    fig.savefig(d/'regional_member_maps.png',dpi=150)
    plt.close(fig)
    with (d/'regional_member_scenes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    (d/'regional_member_counts.json').write_text(json.dumps(summary,indent=2)+'\n')
    selection_map(sweep,d)
    print(summary)

if __name__=='__main__':main()
