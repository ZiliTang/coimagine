"""Render empirical transition graphs and side-by-side recorded boundary positions."""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np

from tools import cap_topology_graph as g


def describe(profile):
    if profile[4]>.5:return 'Rear contact','后侧接触'
    if profile[6]>.7:return 'Blocked sweep','输运段静态受阻'
    if profile[0]<.5:return 'No station','缺少空闲后侧站位'
    if profile[1]<.5:return 'Indirect','站位可用但直达受限'
    if profile[2]>.6:return 'Staged','已就位、多未接触'
    return 'Approaching','可直达、尚未就位'


def matrix(rows):
    fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    for row,m in enumerate(('physical','supervised','public')):
        for col,expert in enumerate('AB'):
            ax=axes[row,col]
            a=np.zeros((6,8))
            den=np.zeros(6,int)
            for r in rows:
                if r['panel']=='fresh' and r['method']==m:
                    a[r['source'],r['dest']]=r[expert]/max(1,r['n'])
                    den[r['source']]=r['n']
            ax.imshow(a,vmax=1,vmin=0,cmap='Blues',aspect='auto')
            for i in range(6):
                for j in range(8):
                    if a[i,j] >= .005:
                        ax.text(j,i,f'{a[i,j]*100:.0f}',ha='center',va='center',color='white' if a[i,j]>.48 else '#283741',fontsize=9)
            ax.set(xticks=range(8),xticklabels=['0','1','2','3','4','5','Win','End'],
                   yticks=range(6),yticklabels=[f'{k}  ({n})' for k,n in enumerate(den)],xlabel='Destination after 120 ticks',
                   ylabel='Source (paired prefixes)',title=f'{m}   |   CAP {expert}')
    fig.suptitle('Fresh rooms   |   Transition probabilities (%)   |   public-only IDs have their own partition')
    fig.savefig(g.DOC/'transition_matrix.png',dpi=140)
    plt.close(fig)


def graph(rows,centres):
    fig,axs=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    xy=np.array([[np.cos(t),np.sin(t)] for t in np.deg2rad([100,40,-20,-80,-140,160])]+[[1.6,-.74],[1.6,.82]])
    for ax,expert,ink in zip(axs,'AB',['#236e75','#93522b']):
        ax.set_aspect('equal')
        for i in range(6):
            rr=[r for r in rows if r['panel']=='fresh' and r['method']=='physical' and r['source']==i]
            n=rr[0]['n']
            dest=max([r for r in rr if r['dest']!=i],key=lambda r:r[expert])
            j=dest['dest']
            p=dest[expert]/max(1,n)
            if p:
                line=FancyArrowPatch(xy[i],xy[j],arrowstyle='-|>',mutation_scale=15,shrinkA=28,shrinkB=28,
                                     connectionstyle='arc3,rad=.14',linewidth=1+5*p, color=ink,alpha=.78)
                ax.add_patch(line)
            stay=next(r[expert] for r in rr if r['dest']==i)/max(1,n)
            ax.scatter(*xy[i],s=1700,color='#ecf2f2',edgecolor=ink,zorder=3)
            ax.text(*xy[i],f'{i}\n{stay:.0%}',ha='center',va='center',fontsize=10,zorder=4)
        for i,lab in [(6,'Win'),(7,'End')]:
            ax.text(*xy[i],lab,ha='center',va='center',bbox=dict(boxstyle='round',facecolor='#f6ebd8',edgecolor='.5'))
        ax.set(xlim=(-1.45,1.85),ylim=(-1.4,1.45),title=f'CAP {expert}')
        ax.axis('off')
    fig.suptitle('Physical topology in fresh rooms   |   strongest outgoing destination other than self\nNode percentages = stayed in node; arrow width = transition probability')
    legend='    ·    '.join(f'{i} {describe(c)[0]}' for i,c in enumerate(centres))
    fig.text(.5,.012,legend,ha='center',fontsize=9)
    fig.savefig(g.DOC/'graph.png',dpi=150)
    plt.close(fig)


def representatives(rooms,labels,points,centres):
    candidates={i:[] for i in range(6)}
    for r in rooms:
        for rec in r['records']:
            for stage,aa,bb in [(0,'AA','BA'),(1,'AA','AB'),(1,'BA','BB')]:
                runs=[rec['branches'][a] for a in (aa,bb)]
                if len(runs[0]['states'])<=stage:continue
                h=runs[0]['states'][stage]
                key=(r['room'],h)
                i=labels[key]
                s=points[key]
                branches=[]
                for rr in runs:
                    trace=[f for f in rr['trace'] if f[1]==stage+1]
                    dest=None
                    if len(rr['states'])>stage+1:
                        end=rec['points'][rr['states'][stage+1]]
                        endbox=np.mean(end['draw']['box'],0)
                        robots=end['draw']['robots']
                        trace.append([end['tick'],stage+1,*endbox,*robots[0],*robots[1]])
                        dest=str(labels[(r['room'],rr['states'][stage+1])])
                    else:dest='任务成功' if rr['success'] else '终止'
                    branches.append(dict(trace=trace,dest=dest))
                val=dict(room=r['room'],node=i,name=describe(centres[i])[1],tick=s['tick'],draw=s['draw'],branches=branches,bits=s['y'])
                candidates[i].append((float(np.linalg.norm(centres[i]-s['y'])),r['room'],s['tick'],val))
    chosen=[]
    for i,rows in candidates.items():
        seen=set()
        for _,room,_,v in sorted(rows,key=lambda r:r[:3]):
            if room not in seen:
                chosen.append(v)
                seen.add(room)
            if len(seen)==2:break
    template="""<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>同前缀转移回放</title><style>body{font:16px system-ui;margin:2rem auto;max-width:1080px;padding:0 1rem;background:#fafaf7;color:#192b2b}canvas{width:100%;height:auto;background:white;border:1px solid #aaa}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}input{width:min(45vw,460px)}select,button{padding:.4rem}p{line-height:1.6}
@media(max-width:600px){.grid{grid-template-columns:1fr}}</style><h1>同前缀的两个真实后继</h1>
<p>每类最多取两个不同新房间，按起点到物理簇中心的距离挑选。两面板共享时钟，位置取最近一次已记录的命令边界。轮廓标出分叉时的物体与障碍物，后续只显示目标物中心与两机器人位置。边界间的连线仅辅助阅读。</p>
<label>代表局面 <select id="sel"></select></label> <button id="play">播放</button> <input id="time" type="range" min="0" max="120" value="0" aria-label="分叉后 tick"><span id="info"></span>
<div class="grid"><section><h2 id="a"></h2><canvas width="600" height="480"></canvas></section><section><h2 id="b"></h2><canvas width="600" height="480"></canvas></section></div>
<p>绿色为目标物，橙、蓝为机器人，星号为终点，红线为当前朝目标方向最有利的物体后侧面。灰色为静态障碍，米色为其他可动物体的起始位置。</p>
<script>const cases=DATA;const sel=document.querySelector('#sel'),time=document.querySelector('#time');let timer=null;
cases.forEach((r,i)=>sel.add(new Option('簇 '+r.node+' '+r.name+' · '+r.room+' · tick '+r.tick,i)));
function draw(){let v=cases[+sel.value], t=+time.value;document.querySelector('#info').textContent=' +'+t+' tick';
v.branches.forEach((b,ix)=>{let el=document.querySelectorAll('canvas')[ix],ctx=el.getContext('2d');ctx.clearRect(0,0,600,480);const pt=([x,y])=>[x*100,480-y*100];
function line(p,color,fill=false,width=2){ctx.beginPath();for(let i=0;i<p.length;i++){let [x,y]=pt(p[i]);if(i)ctx.lineTo(x,y);else ctx.moveTo(x,y)}ctx.lineWidth=width;if(fill){ctx.closePath();ctx.fillStyle=color;ctx.fill()}else{ctx.strokeStyle=color;ctx.stroke()}}
v.draw.static.forEach(p=>line(p,'#a4a7a4',true));v.draw.movable.forEach(p=>line(p,'#dbcaad',true));line(v.draw.box,'#abc9be',true);line(v.draw.face,'#a54740',false,3);
let [gx,gy]=pt(v.draw.goal);ctx.fillStyle='#242925';ctx.fillText('★',gx-5,gy+4);let frames=b.trace.filter(f=>f[0]<=v.tick+t);let inks=['#148160','#2271a0','#c1742f'];if(frames.length){for(let j=0;j<3;j++){let q=2+j*2;let p=frames.map(f=>[f[q],f[q+1]]);line(p,inks[j]);let [x,y]=pt(p[p.length-1]);ctx.beginPath();ctx.arc(x,y,j?7:5,0,7);ctx.fillStyle=inks[j];ctx.fill()}}
document.querySelector(ix?'#b':'#a').textContent='CAP '+(ix?'B':'A')+' → '+b.dest})}
function stop(){clearInterval(timer);timer=null;document.querySelector('#play').textContent='播放'}
sel.onchange=()=>{stop();time.value=0;draw()};time.oninput=draw;document.querySelector('#play').onclick=()=>{if(timer){stop();return}if(+time.value===120)time.value=0;document.querySelector('#play').textContent='暂停';timer=setInterval(()=>{time.value=+time.value+2;draw();if(+time.value>=120)stop()},100)};draw();</script></html>"""
    (g.DOC/'replays.html').write_text(template.replace('DATA',json.dumps(chosen,separators=(',',':'))))


def make(panels,labels,points,centres,frequencies):
    graph(frequencies,centres)
    matrix(frequencies)
    representatives(panels['fresh'],labels['fresh']['physical'],points['fresh'],centres)
