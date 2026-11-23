"""Replay opposite paired outcomes and compare observable target ranges."""
from pathlib import Path
from unittest.mock import patch
import csv
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools import cap_authored_handoff as e
from tools import cap_authored_handoff_graph as graph


def example(job):
    key,fixture,family=job
    original=e.s.tick_action
    branches={}
    expected=e.read(e.OUT/'test'/(key+'.json'))
    for arm in ('A','lock','graph'):
        trace=[]
        def observe(w,action,controller,update,counts):
            obs=e.s.observe(w);f=e.g.live_frame(obs)
            trace.append(dict(tick=obs['tick'],h=w.history.sha256,gap=float(f['gap']) if f['seen'] else None,
                              xy=f['xy'].round(3).tolist(),action=action))
            original(w,action,controller,update,counts)
        with patch.object(e.s,'tick_action',observe):
            result=e.simulate(fixture,arm,family,chooser=graph.choose)
        assert result['result']['history']==expected[arm]['result']['history']
        branches[arm]=dict(trace=trace,result=result['result'],hit=result['hit'])
    e.save(e.OUT/'examples'/(key+'.json'),branches)


def main():
    with (e.DOC/'routes.csv').open() as f:rows=list(csv.DictReader(f))
    panels=[]
    for key,one,two in [('B rescues A','lock','A'),('B loses A success','A','lock'),
                        ('Graph rescues B at X','graph','lock'),('Graph loses B-at-X success','lock','graph')]:
        cs=[r['room'] for r in rows if r[one]=='1' and r[two]=='0' and r['room'] not in [p[1] for p in panels]]
        if cs:panels.append((key,min(cs,key=lambda k:e.key('examples/'+k))))
    if not panels:return
    graph.load_policy();member=e.read(e.OUT/'member.json')
    e.init(Path(member['path']).read_text(), profile=e.perception.LEGACY);rooms=graph.fresh_panel()
    family=e.read(e.OUT/'graph.json')['family']
    e.pool(example,[(k,rooms[k],family) for _,k in panels],4)
    fig,axes=plt.subplots(2,2,figsize=(10,6),layout='constrained',squeeze=False)
    for ax in axes.flat:ax.set_visible(False)
    body=['# 三种路线从相同初态执行的实际公开观测','',
          '从四类相反的成败结果各按预定字符串哈希顺序取一个不同房间，用同一冻结版本重新执行，并核对完整历史哈希。曲线为当时可见的目标扫描中心到终点的距离，目标不可见时留空。竖线为三种方案共享的第一次触发点。单例不估计频率。','']
    import numpy as np
    for ax,(criterion,k) in zip(axes.flat,panels):
        ax.set_visible(True);data=e.read(e.OUT/'examples'/(k+'.json'))
        for arm,color in [('A','#284e76'),('lock','#b17726'),('graph','#298576')]:
            r=data[arm]
            ax.plot([s['tick'] for s in r['trace']],[np.nan if s['gap'] is None else s['gap'] for s in r['trace']],
                    color=color,lw=1.3,alpha=.88,label=f"{arm} ({r['result']['success']})")
        hit=data['A']['hit']
        if hit:ax.axvline(hit['tick'],color='.5',ls=':',lw=1)
        ax.set(title=criterion,xlabel='Physical ticks',ylabel='Observed goal range (m)')
        ax.spines[['right','top']].set_visible(False);ax.legend(frameon=False,fontsize=8)
        body+=['- '+k+'，首次 X 在第 '+str(hit['tick'])+' tick，整局成功标记和终止时刻为 '+ '、'.join(f"{a} {v['result']['success']}（{v['result']['ticks']} tick）" for a,v in data.items())+'。']
    fig.savefig(e.DOC/'examples.png',dpi=150);plt.close(fig)
    (e.DOC/'examples.md').write_text('\n'.join(body)+'\n\n![实际范围曲线](examples.png)\n')
    p=e.DOC/'README.md';s=p.read_text()
    if '[成败翻转案例]' not in s:p.write_text(s+'\n[成败翻转案例](examples.md) · [实际公开距离曲线](examples.png)\n')
    p=Path('docs/research/experiment_journal/entries.json');es=json.loads(p.read_text());entry=next(r for r in es if r['id']=='authored-handoff')
    for name,title in [('examples.png','成败翻转的实际轨迹'),('examples.md','案例编号与结局')]:
        if not any(x[0]==name for x in entry['assets']):entry['assets'].append([name,title])
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(es,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)


if __name__=='__main__':main()
