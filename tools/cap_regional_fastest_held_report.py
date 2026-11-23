"""Summarize paired frozen-checkpoint rollouts and CAPS' observed successful histories."""
import csv
from statistics import median

from tools import cap_regional_fastest_held as run

study = run.study
METHODS = ('directq','best','fixed','polish')


def rows():
    sids = study.read(run.OUT/'protocol.json')['sids']
    fixtures = study.read(study.OUT/'held.json')
    with (study.DOCS/'per_scene.csv').open() as f:
        references = {r['case_id']:r for r in csv.DictReader(f) if r['panel'] in run.PANELS}
    assert set(fixtures)==set(references)
    for cid, r in references.items():
        panel = r['panel']
        success = {}
        for sid in sids:
            expert = study.read(study.OUT/'evaluation'/sid/(cid+'.json'))
            assert expert['complete'] and expert['id']==cid and int(expert['success'])==int(r[sid])
            if expert['success']:
                success[sid]=expert
        fastest = min(success,key=lambda sid:(success[sid]['ticks'],sid)) if success else None
        hist = {e['history'] for e in success.values()}
        entry = dict(panel=panel,case_id=cid,parent=r['parent'],objects=int(r['objects']),
                     ncap=len(success),best=int(r['best']),fastest=fastest)
        assert entry['best']==int(r['n_s002']) and bool(success)==bool(int(r['union']))
        for key in ('directq','fixed','polish'):
            directory = run.ex.BASE if key=='directq' else run.OUT/key
            res = study.read(directory/'evaluation'/panel/(cid+'.json'))
            assert res['case_id']==cid and 0<res['boundaries']<=300 and 0<res['ticks']<=1800
            if key!='directq':
                assert res['cap_id']==key
            entry[key]=int(res['success'])
            entry[key+'_ticks']=res['ticks']
            entry[key+'_exact_fastest']=int(bool(fastest) and res['history']==success[fastest]['history'])
            entry[key+'_exact_success']=int(res['history'] in hist)
            assert entry[key+'_exact_fastest']<=entry[key+'_exact_success']<=entry[key]
        assert entry['directq']==int(r['directq'])
        yield entry


def summarize(group):
    counts = {}
    for method in METHODS:
        counts[method] = dict(success=sum(r[method] for r in group),
                              retained=sum(r[method] for r in group if r['ncap']),
                              single=sum(r[method] for r in group if r['ncap']==1),
                              outside=sum(r[method] for r in group if not r['ncap']))
        if method!='best':
            counts[method]['exact_fastest']=sum(r[method+'_exact_fastest'] for r in group)
            counts[method]['exact_success']=sum(r[method+'_exact_success'] for r in group)
    shared = [r for r in group if r['polish'] and r['directq']]
    return dict(n=len(group),families=len({r['parent'] for r in group}),
                union=sum(bool(r['ncap']) for r in group),singles=sum(r['ncap']==1 for r in group),
                counts=counts,paired={k:study.compare(group,k,'polish') for k in ('directq','best','fixed')},
                shared_successes=len(shared),shared_median_tick_difference=median(
                    r['polish_ticks']-r['directq_ticks'] for r in shared) if shared else None)


def report():
    records=list(rows())
    results={}
    for panel,n in run.PANELS.items():
        group=[r for r in records if r['panel']==panel]
        assert len(group)==n
        results[panel]=summarize(group)
    docs=run.ex.DOCS
    with (docs/'held_per_scene.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    study.save(docs/'held_results.json',dict(panels=results,protocol=study.read(run.OUT/'protocol.json')))
    lines=['# 最快成功教师 actor 的留出集闭环','',
           '主结果使用在训练集上保留 161/161 的 seed 17 模型（24,000 步后追加 4,500 步）。同时评估原 24,000 步模型。两者使用同一套冻结的十个区域专家、归一化与合法动作，均逐边界取最高分动作。比较对象是同房间已归档的原二元收益 directQ 和开发集选定单程序 n_s002。','',
           '| 面板 | 方法 | 全体成功 | 并集内保留 | 独胜保留 | 并集外成功 |','|---|---|---:|---:|---:|---:|']
    labels=dict(fresh='自然留出',near='困难族扰动',directq='原 directQ',best='开发选定单 CAP',fixed='24,000 步 actor',polish='训练选定 actor')
    for panel,m in results.items():
        for k in METHODS:
            c=m['counts'][k]
            lines.append(f"| {labels[panel]} | {labels[k]} | {c['success']}/{m['n']} | {c['retained']}/{m['union']} | {c['single']}/{m['singles']} | {c['outside']}/{m['n']-m['union']} |")
    fresh=results['fresh']
    main=fresh['counts']['polish']
    lines+=['',f"训练选定模型在训练并集上成功 161/161，在自然留出并集上成功 {main['retained']}/{fresh['union']}（{100*main['retained']/fresh['union']:.1f}%）。其中只有一个成功 CAP 的房间从训练 16/16 到留出 {main['single']}/{fresh['singles']}。这些分母对应不同初态。", '', '## 同房间的成功差异','',
            '| 面板 | 参照 → 训练选定 actor | 救回 / 丢失 | 净差（百分点） | 配对 95% 区间 |','|---|---|---:|---:|---:|']
    for panel,m in results.items():
        for k,p in m['paired'].items():
            lines.append(f"| {labels[panel]} | {labels[k]} → actor | {p['rescued']} / {p['lost']} | {p['pp']:+.1f} | [{p['ci'][0]:+.1f}, {p['ci'][1]:+.1f}] |")
    lines += ['','两套模型共新增 580 次执行，对应 290 个留出初态。自然面板含 200 个房间。困难面板含原 45 个开发父房间的各两个未见扰动。配对区间按物体数分层重抽 10,000 次；困难面板每次以父房间为单位保留两个扰动。两个面板此前均已用于诊断，模型使用相同的一个训练种子。','',
        '训练时每个可解房间选一条最快成功示范，在每个示范边界给实际动作标签 1。评分器输入各 CAP 提案的显式参数、来源及公开几何历史，同一动作的多个来源合并。合法动作上的 softmax 给出模仿分布，执行时取 argmax。','',
        '[逐房间成败与用时](held_per_scene.csv) · [模型哈希与统计](held_results.json) · [同模型训练集结果](README.md)']
    (docs/'held.md').write_text('\n'.join(lines)+'\n')
    study.save(run.OUT/'FINISHED.json',dict(completed=len(records)*2,report=str(docs/'held.md')))


if __name__=='__main__':
    report()
