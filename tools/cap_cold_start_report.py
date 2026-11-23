"""Paired closed-loop results and retention of the four rounds' offline labels."""
import csv
import json
from tools import cap_cold_start as run
from tools.cap_option_handoff_opportunity_report import paired

exp,d,e=run.exp,run.d,run.e
DOC,OUT=run.DOC,run.OUT


def write_csv(name,rows):
    with (DOC/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def pref_stats():
    d.load('semantic')
    warm=d.MODELS['semantic']
    cold=run.weights(OUT/'cold0.npz')
    stats=[]
    for i in range(1,5):
        anchors,x,y,_=run.p.counts(sorted((exp.stage_dir(i)/'fit_data').glob('*.npz')))
        for mode in ('warm_replay','self','warm_anchor'):
            base=warm if mode=='warm_replay' else cold
            for checkpoint in {'initial':0,'after':i,'final':4}.items():
                k,step=checkpoint
                d.MODELS['semantic']=run.weights(OUT/mode/f'actor{step}.npz') if step else base
                pred=d.logits(x,'semantic')>0
                stats.append(dict(arm=mode,round=i,checkpoint=k,correct=int((pred==y).sum()),n=len(y),
                                  chose_B=int((d.logits(anchors,'semantic')>0).sum()),anchors=len(anchors)))
    return stats


def render(result,prefs):
    names={'warm4_vs_warm0':'蒸馏起点，保留自身','self4_vs_cold0':'随机起点，保留自身','warm_anchor4_vs_cold0':'随机起点，复用蒸馏组软标签'}
    panels=result['panels']
    comparisons=['| 初始化与保持目标 | 训练 272 间 | 固定诊断 64 间，第 0 至 4 轮 | 自然 200 间 | 自然房间救回 / 丢失 | 自然净差 95% 区间（百分点） |',
                 '|---|---:|---|---:|---:|---:|']
    for key,mode in zip(names,('warm','self','warm_anchor'),strict=True):
        a=panels['train'][key]
        b=panels['held'][key]
        diag=' → '.join(map(str,result['diagnostic'][mode]))
        lo,hi=[100*z for z in b['ci']]
        comparisons.append(f"| {names[key]} | {a['second']} → {a['first']} | {diag} | {b['second']} → {b['first']} | {b['plus']} / {b['minus']} | [{lo:+.1f}, {hi:+.1f}] |")
    cross=panels['held']['self4_vs_warm4']
    sections=['# 同一批分叉监督下的冷启动对照',
        '固定 seed17，用 PyTorch 的默认初始化取得与原蒸馏前一致的 349→64→64→1 网络。随机初始化仅作用于路由 actor。几何特征、两位 CAP 的提案、停滞触发前使用 A、执行器及感知均沿用前次实验。冷启动组跳过教师蒸馏，在同一批已保存的四轮分叉样本上更新，未重新采样训练轨迹。全部测试按既有房间内容哈希逐例闭环执行。',
        '\n'.join(comparisons),
        f"冷启动保留自身的最终版本与蒸馏起点的最终版本配对，独胜 {cross['plus']} 个房间、独败 {cross['minus']} 个，净差的 95% 区间为 [{100*cross['ci'][0]:+.1f}, {100*cross['ci'][1]:+.1f}] 个百分点。区间按房间配对重采样 4,000 次。200 房间在前次实验后已查看，当前对照是事后诊断；只测试了一个初始化种子。",
        '## “相同数据”的两种对照',
        '所有组读取同一批按房间排序的文件，复用相同的硬偏好标签、保持项的输入、每轮 300 次更新、同样的小批次索引、学习率和损失系数。先从蒸馏权重按此回放，四个 checkpoint 的全部权重均与原训练逐值相等。每轮 Adam 依照原实现重新初始化。',
        '冷启动保留自身组，保持目标由该冷启动 actor 在每轮更新前的概率给出。冷启动复用软标签组，保持目标也与原实验逐项相同，用原来蒸馏组在该轮更新前的概率给出，因此还能通过这些软标签获得蒸馏组的信息。三组的硬偏好都来自原蒸馏分支当时执行的两条后缀。',
        '## 标签与闭环的关系',
        '| 策略 | 各轮更新后命中当轮偏好，合计 /188 | 最终模型命中四轮全部偏好 /188 |\n|---|---:|---:|',
    ]
    for arm,label in [('warm_replay','蒸馏起点'),('self','冷启动、自身保持'),('warm_anchor','冷启动、原软标签')]:
        hits=lambda check:sum(r['correct'] for r in prefs if r['arm']==arm and r['checkpoint']==check)
        sections[-1]+=f"\n| {label} | {hits('after')} | {hits('final')} |"
    sections+=['7,075 个已记录的可路由训练边界上，随机初始网络贪心选择 B 的次数为 0；行为仍有基础专家能力。188 条硬标签来自 107 个训练房间，其中 167 条来自仅一位专家成功的房间。',
               '原蒸馏组在这 107 个有硬标签的房间从 46 提高到 74 个成功；其余 165 间为 111→111。这个分层由采样结果决定，不能用来估计给同一房间增加标签的因果效应。仅 B 成功的亚组在训练房间为 38→61/92，在自然房间为 13→11/28。因此除了训练时特意多采独胜房间，还要考虑同一亚组内的迁移。',
               '一个分叉的正标签要求在当时的公开历史改选一步、继而交还当时的 actor 后成功。部署换成新的 actor 时，后续到达的状态和其上的路由选择都可能变化。软概率保持约束只覆盖记录到的状态，无法保证其他状态的路由和回报。这个机制和标签局部性一致，此实验没有分离状态分布偏移、动作后果敏感、标签稀少各自的贡献。[Ross 等，2011](https://proceedings.mlr.press/v15/ross11a.html) · [Schulman 等，2015](https://proceedings.mlr.press/v37/schulman15.html)',
               '[逐房间训练与自然闭环](endpoints.csv) · [偏好与分叉选向统计](preferences.csv) · [训练与评测实现](../../../../tools/cap_cold_start.py)']
    sections.insert(4,'随机起点在 200 间中的成败逐间与 A 相同。保留自身的随机起点经四轮学习，在仅 B 成功的 28 间中从 0 到 15 间，在两位专家都失败的 81 间中从 0 到 13 间。在仅 A 成功的 13 间和两位都成功的 78 间分别从 13 到 8、从 78 到 76。')
    (DOC/'README.md').write_text('\n\n'.join(sections)+'\n')


def main():
    DOC.mkdir(parents=True,exist_ok=True)
    prefs=pref_stats()
    write_csv('preferences.csv',prefs)
    signal={r['room'] for r in csv.DictReader((exp.DOC/'visits.csv').open()) if int(r['informative'])}
    rows=[]
    panels={}
    details={}
    for panel,old in [('train','train_check.csv'),('held','held.csv')]:
        src=list(csv.DictReader((exp.DOC/old).open()))
        assert len(src)==(272 if panel=='train' else 200)
        for r in src:
            k=r['room']
            rows.append(dict(panel=panel,room=k,stratum=r['stratum'],signal=k in signal,warm0=int(r['initial']),warm4=int(r['final']),
                             **{m:e.read(OUT/panel/m/(k+'.json'))['success'] for m in ('cold0','self4','warm_anchor4')}))
        sub=[r for r in rows if r['panel']==panel]
        def contrast(a,b):
            return paired([r[a] for r in sub],[r[b] for r in sub])
        panels[panel]={a+'_vs_'+b:contrast(a,b) for a,b in [('warm4','warm0'),('self4','cold0'),('warm_anchor4','cold0'),('self4','warm4'),('warm_anchor4','warm4'),('warm_anchor4','self4')]}
        for group in ('all','A_only','B_only','both','neither','supplied_signal','no_signal'):
            if group.endswith('signal') and panel!='train':
                continue
            g=[r for r in sub if group=='all' or r['stratum']==group or group=='supplied_signal' and r['signal'] or group=='no_signal' and not r['signal']]
            details[panel+'/'+group]=dict(n=len(g),**{m:sum(r[m] for r in g) for m in ('warm0','warm4','cold0','self4','warm_anchor4')})
    write_csv('endpoints.csv',rows)
    diag={}
    for mode in ('self','warm_anchor'):
        diag[mode]=[]
        for i in range(5):
            model=f'{mode}{i}' if i else 'cold0'
            paths=list((OUT/'diagnostic'/model).glob('*.json'))
            assert len(paths)==64
            diag[mode].append(sum(e.read(f)['success'] for f in paths))
    diag['warm']=e.read(exp.OUT/'FINISHED.json')['diagnostic']
    stats=dict(panels=panels,groups=details,diagnostic=diag)
    render(stats,prefs)
    e.save(OUT/'summary.json',stats)
    return stats,prefs


if __name__=='__main__':
    summary,_=main()
    print(json.dumps(summary,indent=2))
