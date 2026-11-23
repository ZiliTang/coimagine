"""Fit matched augmented GA-Q models and compare frozen greedy policies on new roots."""
import argparse
import html
import multiprocessing as mp
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import cap_family_augmentation as study
from . import cap_family_fit
from .cap_trace_value_long import sha
from experiments.planar_tasks.semantic_rl.program_trace_value_v1 import full_data
from experiments.planar_tasks.semantic_rl.program_trace_value_v1.dataset import read, save
from tools.cap_trace_value_long_report import write_csv

OUT, PARENT, ARMS, CAPS = study.OUT, study.PARENT, study.ARMS, study.CAPS
POLICIES = ('baseline', *ARMS)


def assemble(arm, rows):
    """Append real variants to the original arrays, keeping successor indices local to each rollout."""
    base = PARENT/'arrays/train'
    dest = OUT/arm
    original = read(base/'manifest.json')
    assert len({(r['case_id'],r['cap']) for r in rows}) == len(rows)
    assert all(n==7 for n in Counter(r['case_id'] for r in rows).values())
    assert all(r['shard']['behavior_mismatches']==0 for r in rows)
    expanded = dest/'arrays/train'
    expanded.mkdir(exist_ok=True, parents=True)
    if not (expanded/'manifest.json').exists():
        start = original['boundaries']
        size = start+sum(r['shard']['boundaries'] for r in rows)
        arrays={}
        for key in (*full_data.KEYS, 'next'):
            source=np.load(base/(key+'.npy'), mmap_mode='r')
            array=np.lib.format.open_memmap(expanded/(key+'.npy'), mode='w+',dtype=source.dtype,
                                             shape=(size,*source.shape[1:]))
            for i in range(0,start,2048):
                j=min(i+2048,start)
                array[i:j]=source[i:j]
            arrays[key]=array
        assert max(arrays['root'][:start])<10000
        offset=start
        for row in rows:
            with np.load(row['shard']['path']) as shard:
                n=len(shard['x'])
                assert n==row['shard']['boundaries'] and shard['done'][-1] and shard['done'].sum()==1
                assert shard['reward'][-1]==row['success'] and np.all(shard['cap']==CAPS.index(row['cap']))
                end=offset+n
                for key in full_data.KEYS:
                    arrays[key][offset:end]=shard[key]
                arrays['next'][offset:end]=np.minimum(np.arange(offset+1,end+1),end-1)
                offset=end
        assert offset==size
        for array in arrays.values():
            array.flush()
        save(expanded/'manifest.json',dict(boundaries=size,width=original['width'],episodes=original['episodes']+len(rows)))
    for link,target in [(dest/'arrays/validation',PARENT/'arrays/validation'),(dest/'dataset.json',PARENT/'dataset.json'),
                        (dest/'normalizers.npz',PARENT/'normalizers.npz')]:
        if not link.exists():
            link.symlink_to(target)
        assert link.resolve()==target.resolve()


def training_state():
    state={}
    for arm in ARMS:
        model=OUT/arm/'models/GA_s17'
        if (model/'offline_complete.json').exists():
            state[arm]='24,000 / 24,000'
        elif (model/'progress.json').exists():
            item=read(model/'progress.json')
            offset=16000 if item['stage']=='Q_init' else 0
            state[arm]=f"{item['stage']} · {offset+item['updates']:,} / 24,000"
        else:
            state[arm]='准备中'
    return state


def fit_jobs(collection):
    from contextlib import ExitStack
    with ExitStack() as stack:
        processes={}
        for arm in ARMS:
            if (OUT/arm/'models/GA_s17/offline_complete.json').exists():
                continue
            cmd=[sys.executable,'-u','-m',__spec__.name,'--fit',arm]
            log=stack.enter_context((OUT/arm/'fit.log').open('a'))
            processes[arm]=stack.enter_context(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in processes.values()):
            stats=training_state()
            study.publish('组装数据与拟合离线 GA-Q',phases={'匹配采集批次':str(collection['rounds']),
                  '普通组拟合':stats['uniform'],'定向组拟合':stats['targeted'],'新面板执行':'0 / 2000'},
                  detail='每组从头做 16,000 次 V 更新和 8,000 次 Q 更新，使用原训练归一化。')
            time.sleep(5)
        for arm,process in processes.items():
            if process.returncode:
                raise RuntimeError(f'fit {arm} failed with exit {process.returncode}; see {arm}/fit.log')


def weights():
    return {'baseline':PARENT/'models/GA_s17/offline.pt',
            **{arm:OUT/arm/'models/GA_s17/offline.pt' for arm in ARMS}}


def evaluate():
    if (OUT/'evaluation_complete.json').exists():
        return
    models=weights()
    seal={k:sha(p) for k,p in models.items()}
    assert seal['baseline']==read(OUT/'protocol.json')['baseline_sha256']
    if (OUT/'frozen.json').exists():
        assert seal==read(OUT/'frozen.json')
    else:
        save(OUT/'frozen.json',seal)
    cases=read(OUT/'evaluation_panel.json')
    jobs=[]
    for name in [*CAPS,*POLICIES]:
        for case in cases:
            params=dict(cap=name) if name in CAPS else dict(checkpoint=str(models[name]),epsilon=0)
            jobs.append(dict(case=case,condition=name,directory=str(OUT/'evaluation'/name/case['case_id']),**params))
    rows=[]
    with ProcessPoolExecutor(12,mp_context=mp.get_context('spawn'),initializer=study.initialize) as pool:
        for future in as_completed([pool.submit(study.worker,j) for j in jobs]):
            item=future.result()
            assert item['complete'] and not item['error']
            rows.append(dict(case_id=item['case_id'],condition=item['condition'],success=bool(item['success']),
                             ticks=item['ticks'], stop=item['stop']))
            if len(rows)%10==0:
                study.publish('冻结模型闭环评估',phases={'采集':'完成','拟合':'2 / 2','新面板执行':f'{len(rows)} / {len(jobs)}'},
                              detail='200 个新初态，每例评估七个单 CAP 和三个冻结 GA-Q 策略。')
    assert len(rows)==len({(r['case_id'],r['condition']) for r in rows})==2000
    save(OUT/'evaluation_complete.json',sorted(rows,key=lambda r:(r['case_id'],r['condition'])))


def summarize():
    collection=read(OUT/'collection.json')
    rows=collection['records']
    plans={r['case_id']:r for path in sorted(OUT.glob('round_*_plan.json')) for r in read(path)}
    families={r['case_id']:len(r['winners']) for r in read(OUT/'anchors.json')}
    summaries={}
    runs=defaultdict(list)
    for r in rows:
        runs[r['case_id']].append(r)
    for arm in ARMS:
        selected=[p for p in plans.values() if p['arm']==arm]
        per_family=Counter()
        single_source=single_still_wins=still_unique=success=0
        for p in selected:
            winners={r['cap'] for r in runs[p['case_id']] if r['success']}
            success+=len(winners)
            per_family[p['parent']]+=len(winners)
            if len(p['base_winners'])==1:
                single_source+=1
                original=p['base_winners'][0]
                single_still_wins+=original in winners
                still_unique+=winners=={original}
        exposure=np.array([k+per_family[root] for root,k in families.items()])
        summaries[arm]=dict(runs=7*len(selected), variants=len(selected), families=len(per_family),
                            successes=success, distinct_sequences=collection['unique_sequences'][arm], added_boundaries=sum(r['shard']['boundaries'] for r in rows if r['arm']==arm),
                            singleton_variants=single_source, sole_cap_still_succeeds=single_still_wins,
                            sole_cap_still_unique=still_unique, success_total_sd=float(exposure.std()),
                            success_total_min=int(exposure.min()),success_total_max=int(exposure.max()))
    write_csv(OUT/'family_exposure.csv',[dict(parent=k,original_successes=count,**{arm:sum(r['success'] for r in rows if r['parent']==k and r['arm']==arm) for arm in ARMS}) for k,count in families.items()])
    outcomes=defaultdict(dict)
    for row in read(OUT/'evaluation_complete.json'):
        outcomes[row['case_id']][row['condition']]=int(row['success'])
    panel=read(OUT/'evaluation_panel.json')
    data=[dict(case_id=c['case_id'],movable_count=c['movable_count'],ncap=sum(outcomes[c['case_id']][cap] for cap in CAPS),
               **outcomes[c['case_id']]) for c in panel]
    write_csv(OUT/'per_scene.csv',data)
    ncap=np.array([r['ncap'] for r in data])
    count=np.array([r['movable_count'] for r in data])
    rng=np.random.default_rng(study.SEED)
    strata=[np.flatnonzero(count==k) for k in range(2,6)]
    draws=np.concatenate([rng.choice(idx,(10000,len(idx))) for idx in strata],axis=1)
    metrics={}
    for k,mask in [('unique',ncap==1),('all',ncap>=0),('union',ncap>0),('multiple',ncap>1),('outside',ncap==0)]:
        result=dict(n=int(mask.sum()),wins={a:sum(r[a] for r,m in zip(data,mask) if m) for a in POLICIES},comparisons={})
        for newer,older in [('targeted','uniform'),('targeted','baseline'),('uniform','baseline')]:
            delta=np.array([r[newer]-r[older] for r in data])
            denom=mask[draws].sum(1)
            samples=(delta*mask)[draws].sum(1)[denom>0]/denom[denom>0]
            result['comparisons'][newer+'-'+older]=dict(delta_pp=float(100*delta[mask].mean()),ci_pp=np.quantile(100*samples,[.025,.975]).tolist(),
                        gained=int(((delta>0)&mask).sum()),lost=int(((delta<0)&mask).sum()))
        metrics[k]=result
    save(OUT/'summary.json',dict(collection=summaries,evaluation=metrics,original_family_sd=float(np.std(list(families.values())))))
    lines=['# 场景族增广与稀有 CAP 能力保留','',
           '两组共享从开发训练划分中选出的 200 个可解基础场景。定向组按原场景和已采邻居的累计成功轨迹数优先补少数族，普通组均匀轮转。两组使用相同的扰动分布，保留所有成功、失败轨迹，每个新初态运行七个 CAP。','',
           '| 采集组 | 基础族 | 新初态 | 执行 | 成功轨迹 / 不同动作序列 | 新决策边界 | 独有族邻居 | 原专家仍成功 / 仍独有 |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm,name in [('uniform','普通'),('targeted','定向')]:
        r=summaries[arm]
        lines.append(f"| {name} | {r['families']} | {r['variants']} | {r['runs']} | {r['successes']} / {r['distinct_sequences']} | {r['added_boundaries']} | {r['singleton_variants']} | {r['sole_cap_still_succeeds']} / {r['sole_cap_still_unique']} |")
    lines+=['', '原始 608 个训练初态共有 4,256 条轨迹，成功 1,425 条、失败 2,831 条；另外 152 个验证初态单独留出。']
    for arm,title in [('uniform','普通增广'),('targeted','定向增广')]:
        row=summaries[arm]
        lines.append(f"- {title}合计 {4256+row['runs']:,} 条，成功 {1425+row['successes']:,} 条、失败 {2831+row['runs']-row['successes']:,} 条。")
    lines+=['',f"200 个基础族的累计成功数标准差从 {np.std(list(families.values())):.2f} 变为普通组 {summaries['uniform']['success_total_sd']:.2f}、定向组 {summaries['targeted']['success_total_sd']:.2f}。每个训练邻居均完整采集七个 CAP，只有原专家继续成功且六个竞争 CAP 均失败时，才计为仍独有。"]
    lines+=['','| 新面板 | 场景数 | 原始 Q 成功 | 普通增广 Q | 定向增广 Q | 定向减普通（百分点，95% 区间） | 新增 / 失去 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for k,label in [('unique','恰一 CAP 成功，主指标'),('all','全体'),('union','七 CAP 成功并集'),('multiple','多个 CAP 成功'),('outside','并集外')]:
        m=metrics[k]
        s=m['comparisons']['targeted-uniform']
        lo,hi=s['ci_pp']
        lines.append(f"| {label} | {m['n']} | {m['wins']['baseline']} | {m['wins']['uniform']} | {m['wins']['targeted']} | {s['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | {s['gained']} / {s['lost']} |")
    lines+=['','评估使用预先冻结的 200 个新场景族。每个模型在每例独立闭环贪心执行。主比较条件于训练 seed17，95% 区间按物体数分层，配对重抽场景 10,000 次。原始 Q 为归档的未增广离线权重，两种增广模型各从头拟合 16,000 步 V 和 8,000 步 Q。训练均按决策边界抽样，因此等执行预算下的边界数和成功轨迹数可不同。','',
            '[逐场景结果](per_scene.csv) · [逐族采集量](family_exposure.csv) · [统计汇总](summary.json) · [冻结协议](protocol.json)']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    collection_table=[]
    for arm,title in [('uniform','普通增广'),('targeted','定向增广')]:
        r=summaries[arm]
        collection_table.append([title, r['variants'], r['runs'], r['successes'], r['distinct_sequences'],
                                 4256+r['runs'], 1425+r['successes'], 2831+r['runs']-r['successes']])
    result_table=[]
    for k,title in [('unique','恰一 CAP 成功'),('all','全体'),('union','七 CAP 并集'),('multiple','多个 CAP 成功'),('outside','并集外')]:
        r=metrics[k]
        effect=r['comparisons']['targeted-uniform']
        lo,hi=effect['ci_pp']
        result_table.append([title,r['n'],*[r['wins'][a] for a in POLICIES],
                             f"{effect['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"])
    def table(headers,body):
        head=''.join('<th>'+html.escape(h)+'</th>' for h in headers)
        cells=['<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in body]
        return '<div class="scroll"><table><thead><tr>'+head+'</tr></thead><tbody>'+''.join(cells)+'</tbody></table></div>'
    webpage=('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
             '<title>场景增广结果</title><style>body{max-width:1040px;margin:4% auto;padding:20px;font:16px/1.65 system-ui,sans-serif;'
             'color:#162a29;background:#f3f4ee}.scroll{overflow:auto}table{border-collapse:collapse;background:white}'
             'td,th{padding:9px 12px;text-align:left;border-bottom:1px solid #d4dbd1;white-space:nowrap}a{color:#006655}</style>'
             '<a href="index.html">返回进度</a><h1>场景族增广结果</h1><h2>训练轨迹</h2>'
             +table(['组别','新初态','新增执行','新增成功','不同成功动作序列','训练总轨迹','成功','失败'],collection_table)
             +'<h2>新场景闭环成功数</h2>'
             +table(['场景','例数','原始离线 Q','普通增广 Q','定向增广 Q','定向减普通，百分点及 95% 区间'],result_table)
             +'<p>三个模型共享 200 个新场景，训练 seed17。区间按物体数分层，对场景配对重抽 10,000 次。</p></html>')
    (OUT/'live/report.html').write_text(webpage)
    save(OUT/'FINISHED.json',dict(complete=True,collection=len(rows),evaluations=len(data)*10,
           hashes={k:sha(OUT/k) for k in ['REPORT.md','summary.json','per_scene.csv','family_exposure.csv','frozen.json','protocol.json']}))
    study.publish('完成',finished=True,detail='闭环计数和场景配对区间已生成。')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',choices=ARMS)
    args=parser.parse_args()
    if args.fit:
        records=read(OUT/'collection.json')['records']
        assemble(args.fit,[r for r in records if r['arm']==args.fit])
        cap_family_fit.train(OUT/args.fit)
        return
    study.collect(require_distinct=True)
    fit_jobs(read(OUT/'collection.json'))
    evaluate()
    summarize()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        study.publish('执行中断',error=traceback.format_exc())
        raise
