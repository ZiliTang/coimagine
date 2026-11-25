"""Count paired terminal outcomes by room and test a single, causal timing edit."""
import csv
import json
from collections import Counter
from math import comb
from statistics import mean

from tools import cap_entry_residual_fit as ridge
from tools.cap_handoff_opportunities import DOC, OUT, read, save


PANELS = {'dev_nat': '开发 · 自然', 'dev_geo': '开发 · 定向', 'silent': '开发 · 已知无接管提议的失败',
          'val_nat': '验证 · 自然', 'val_geo': '验证 · 定向'}
PROBES = {'early': '提前接管', 'giveback': '提前交还', 'extend': '稍晚交还'}
ORIGINS = {'original': '原路线', 'after_early': '提前接管后的路线'}


def summarize(rows):
    signs = Counter((r['take'], r['defer']) for r in rows)
    rescued = {r['room'] for r in rows if r['defer'] > r['take']}
    lost = {r['room'] for r in rows if r['defer'] < r['take']}
    delta = [r['advance_delta'] for r in rows]
    signed = [(r['defer']-r['take'])*r['advance_delta'] for r in rows if r['take']!=r['defer']]
    return dict(rooms=len({r['room'] for r in rows}), pairs=len(rows),
                rescue_pairs=signs[0, 1], loss_pairs=signs[1, 0],
                both_win=signs[1, 1], both_fail=signs[0, 0],
                rescue_rooms=len(rescued), loss_rooms=len(lost), both_directions=len(rescued & lost),
                mean_120tick_gain_m=mean(delta) if delta else None,
                short_progress_matches_terminal_winner=sum(d>0 for d in signed),
                short_progress_ties=sum(d==0 for d in signed))


def first_policy(rows, gate, panel):
    # A deployed policy can identify the first eligible boundary without future information.
    first = {}
    for row in sorted(rows, key=lambda r: r['tick']):
        first.setdefault(row['room'], row)
    changed = [r for r in first.values() if ridge.should_defer(gate, r['features'])]
    for r in first.values():
        assert r['take'] == panel[r['room']]['baseline']['success']
    orig = sum(r['baseline']['success'] for r in panel.values())
    gains = sum(r['defer'] > r['take'] for r in changed)
    losses = sum(r['defer'] < r['take'] for r in changed)
    discordant = gains + losses
    sign_p = min(1.0, 2 * sum(comb(discordant,i) for i in range(min(gains,losses)+1)) / 2**discordant)
    return dict(n=len(panel), baseline=orig, chosen=orig+gains-losses,
                opportunities=len(first), intervened=len(changed), rescued=gains, lost=losses,
                paired_exact_two_sided_p=sign_p)


def build():
    fixtures = read(OUT/'fixtures.json')
    protocol = read(OUT/'protocol.json')
    paths = {p.stem:p for p in (OUT/'runs').glob('*.json')}
    if set(fixtures) != set(paths):
        raise ValueError('Room set incomplete')
    panels = {g:{} for g in PANELS}
    rows = []
    for k, path in sorted(paths.items()):
        r = read(path)
        group = r['group']
        assert fixtures[k][0] == group and k == r['room']
        panels[group][k] = r
        for pair in r['pairs']:
            a = pair['anchor']
            if pair['origin'] == 'original':
                assert pair['default'] == r['baseline']['success']
            rows.append(dict(room=k, group=group, origin=pair['origin'], probe=a['probe'],
                             tick=a['tick'], event=a['kind'], start=a['start'], prefix=a['prefix'],
                             observation=a['observation'], memory=a['memory'], features=a['features'],
                             original_action=a['default'], alternative_action=a['alternative'],
                             take=int(pair['default']), defer=int(pair['altered']),
                             default_advance=pair['default_advance'], altered_advance=pair['altered_advance'],
                             advance_delta=round(pair['altered_advance']-pair['default_advance'],5)))
    assert {g:len(p) for g,p in panels.items()} == protocol['counts']
    assert len({(r['room'],r['origin'],r['probe'],r['tick'],r['event'],r['prefix']) for r in rows}) == len(rows)
    stats, models, broad = {}, {}, {}
    for g,panel in panels.items():
        subset = [r for r in rows if r['group']==g and r['origin']=='original']
        early = [r for r in subset if r['probe']=='early']
        untouched = {k for k,f in panel.items() if not f['legacy_pending']}
        never = [r for r in early if r['room'] in untouched]
        broad[g] = dict(n=len(panel),baseline=sum(r['baseline']['success'] for r in panel.values()),
                        original=summarize(subset),no_pending_rooms=len(untouched),
                        no_pending_failures=sum(not panel[k]['baseline']['success'] for k in untouched),
                        early_on_no_pending=summarize(never))
        rescued_once = {r['room'] for r in subset if r['defer']>r['take']}
        rescued_all = {r['room'] for r in rows if r['group']==g and r['defer']
                       and not panel[r['room']]['baseline']['success']}
        broad[g]['extra_rescued_only_with_two_edits'] = len(rescued_all - rescued_once)
    parent_file = OUT/'silent_parent.json'
    if parent_file.exists():
        parents = read(parent_file)
        assert set(parents)==set(panels['silent'])
        broad['silent']['standalone_parent_successes'] = sum(r['success'] for r in parents.values())
    for origin in ORIGINS:
        for typ in PROBES:
            if origin=='after_early' and typ=='early':
                continue
            key = origin + '_' + typ
            subsets = {g:[r for r in rows if r['group']==g and r['origin']==origin and r['probe']==typ] for g in PANELS}
            stats[key] = {g:summarize(rr) for g,rr in subsets.items()}
            dev = subsets['dev_nat'] + subsets['dev_geo'] + subsets['silent']
            if not all(any((r['defer']-r['take'])*s>0 for r in dev) for s in [-1,1]):
                models[key] = {'reason':'开发分叉未同时出现正、负终局差异'}
                continue
            gate = dict(ridge._fit_linear(dev),threshold=0.0)
            held = {}
            for g in ['val_nat','val_geo']:
                rr = subsets[g]
                held[g] = dict(single_fork=ridge._summary(rr,gate), conditional_auc=ridge._ranking(rr,gate))
                if origin=='original':
                    held[g]['first_policy']=first_policy(rr,gate,panels[g])
                    held[g]['always_change_first']=first_policy(rr,ridge.ALWAYS_DEFER,panels[g])
            models[key] = dict(gate=gate,train=summarize(dev),held=held)
    DOC.mkdir(parents=True,exist_ok=True)
    save(DOC/'results.json',dict(groups=broad,by_timing=stats,models=models))
    with (DOC/'pairs.csv').open('w',newline='') as f:
        w = csv.DictWriter(f,fieldnames=rows[0].keys())
        w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v,sort_keys=True) if k in ('features','original_action','alternative_action') else v for k,v in r.items()})
    md = ['# 提前接管与改变交还时机','',
          '母策略 P 与恢复程序 R 冻结。每个分叉从相同的可见观测、物理前缀和程序记忆出发，只改变一次当前决定，再用原规则跑到终局。'+
          '本轮比较基准是作者的 P＋R 完整程序。','',
          '## 哪些房间存在机会','',
          '| 面板 | 房间 | 原程序成功 | 有任一原路线分叉 | 至少能救回一回的房间 | 至少会丢失一回的房间 |',
          '|---|---:|---:|---:|---:|---:|']
    for g,x in broad.items():
        v=x['original']
        md.append(f"|{PANELS[g]}|{x['n']}|{x['baseline']}|{v['rooms']}|{v['rescue_rooms']}|{v['loss_rooms']}|")
    md+=['','每间计一次。从事后知道终局的视角，任取一条已测分叉可以多救回哪些房间。救回与丢失两列相对于同一条原路线，实际选择需要只看当时的公开输入。','',
         '自然抽样的新房间用来判断总体收益空间。定向组从独立候选池取最可能触发旧接管的五分之一。已知失败组来自旧开发集，仅用于寻找漏触发。验证两组与开发房间按布局内容去重。','',
         '## 各个交接时刻','',
         '| 起点 | 改动 | 面板 | 有机会的房间 | 分叉 | 改动独胜 / 原决定独胜（分叉） | 双胜 / 双败 | 救回 / 丢失（去重房间） |',
         '|---|---|---|---:|---:|---:|---:|---:|']
    for key,groups in stats.items():
        origin,typ=key.rsplit('_',1)
        for g,v in groups.items():
            md.append(f"|{ORIGINS[origin]}|{PROBES[typ]}|{PANELS[g]}|{v['rooms']}|{v['pairs']}|{v['rescue_pairs']} / {v['loss_pairs']}|{v['both_win']} / {v['both_fail']}|{v['rescue_rooms']} / {v['loss_rooms']}|")
    md+=['','提前接管将停滞证据窗口从 140 ticks 缩到 60 ticks，要求最近至少两个观测，物体向目标前进不超过 6 cm。原有几何、可行动作和剩余时间条件继续生效。提前交还要求 P 已经提出不同的合法动作。稍晚交还允许 R 在达到位移目标、超出不超过 10 cm 时再选一次动作，保留超时与避障限制。所有条件仅在原有 option 边界、单推手连续接触 3 ticks 或接触中推手机器人沿推动轴每前进 0.2 m 的复核点检查。复核点只纳入能实际切换控制的替代提议。','',
          '在原轨迹上，每类取首末两个合格时刻，稍晚交还只取第一个。另在首次提前接管实际生成的计划内取同样的交还分叉。后一类成败均与已经提前接管的配对路线比较，不能和原路线增益相加。提前接管指此刻早于原规则，房间更早时可能已有其他恢复计划。','',
          '## 公开信息能否判断何时改动','',
          '分时机拟合线性回归。输入为当时可见的几何、接触、进展与计划信息；目标为改动后的终局成功减去默认终局成功，轨迹内按房间等权。开发两种胜负方向都出现才拟合。验证前固定阈值为零。以下整局策略只在该类**第一次**合格时按预测决定是否干预，后续完全沿用原规则。两条结局均已实跑，可从配对记录精确还原该策略的逐房间终局。','',
          '| 改动 | 验证面板 | 原程序 → 单次选择 | 执行改动 | 救回 / 丢失 | 首次一律改动的成功数 | 全部分叉条件 AUC |',
          '|---|---|---:|---:|---:|---:|---:|']
    for typ in PROBES:
        model=models['original_'+typ]
        for g,v in model.get('held',{}).items():
            p=v['first_policy']; auc=v['conditional_auc']
            auc_text = '—' if auc is None else f'{auc:.3f}'
            md.append(f"|{PROBES[typ]}|{PANELS[g]}（{p['n']} 间）|{p['baseline']} → {p['chosen']}|{p['intervened']}|{p['rescued']} / {p['lost']}|{v['always_change_first']['chosen']}|{auc_text}|")
        if 'reason' in model:
            md.append(f"|{PROBES[typ]}|{model['reason']}|—|—|—|—|—|")
    md+=['','AUC 仅比较终局翻转的正负分叉，缺少任一方向时未定义。单次选择的房间增益与这个条件排序指标分母不同。','',
          '## 原规则完全没有提议的房间','']
    for g,v in broad.items():
        u=v['early_on_no_pending']
        md.append(f"- {PANELS[g]}中，{v['no_pending_rooms']} 间在整条原路线均无恢复提议，其中 {v['no_pending_failures']} 间失败。提前接管在 {u['rooms']} 间可执行，救回 {u['rescue_rooms']} 间、丢失 {u['loss_rooms']} 间。")
    if parent_file.exists():
        md+=['',f"另外从头运行母策略 P。80 间已知失败中，P 单独成功 {broad['silent']['standalone_parent_successes']} 间。"]
    md+=['',f"首次提前接管后再调整一次交还，与最初未经干预的整局比较。自然验证另救回 {broad['val_nat']['extra_rescued_only_with_two_edits']} 间、定向验证另救回 {broad['val_geo']['extra_rescued_only_with_two_edits']} 间；这些房间在已测的单次改动中均未获救。"]
    md+=['','## 配对实例','',
          '物理距离仅用于事后核对。下表给出原决定和替代决定在随后 120 ticks（或更早终局）缩短了多少物体至目标距离，负数表示远离。每类各列首个满足条件的验证案例。','',
          '| 面板与房间 | 改动 | tick | 终局（原 → 改） | 目标距离缩短（原 → 改，m） |',
          '|---|---|---:|---:|---:|']
    for typ in PROBES:
        for sign in [1,-1]:
            r=next((r for r in rows if r['group'].startswith('val') and r['origin']=='original' and
                    r['probe']==typ and r['defer']-r['take']==sign),None)
            if r:
                md.append(f"|{r['room']}|{PROBES[typ]}|{r['tick']}|{r['take']} → {r['defer']}|{r['default_advance']:.2f} → {r['altered_advance']:.2f}|")
    md+=['','[结果解读与配对案例](interpretation.md) · [逐分叉记录](pairs.csv) · [统计、特征权重和逐组结果](results.json)','']
    (DOC/'README.md').write_text('\n'.join(md))
    save(OUT/'finished.json',dict(rooms=len(fixtures),pairs=len(rows),report=str(DOC/'README.md')))


if __name__=='__main__':
    build()
