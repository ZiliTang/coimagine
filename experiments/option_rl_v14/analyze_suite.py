"""Read-only training comparison and an offline dashboard index."""
import argparse,json
from pathlib import Path
import numpy as np
from experiments.option_rl_v14.live import atomic_json,read_json


def summarize(batch):
    batch=Path(batch).resolve();launch=read_json(batch/'launch.json');rows=[];histories={}
    for job in launch['jobs']:
        name=job['name'];run=batch/name/name;h=read_json(run/'history.json',[]) or [];histories[name]=h
        ev=[r for r in h if r.get('success_rate') is not None]
        result=read_json(run/'result.json',{}) or {}
        row=dict(name=name,iterations=len(h),episodes=sum(r['valid_episodes'] for r in h),
            actor_samples=sum(r['steps'] for r in h),dense_samples=sum(r.get('dense_samples',0) for r in h),
            simulated_seconds=sum(r.get('simulated_seconds',0) for r in h),
            anomaly_episodes=sum(r['anomaly_episodes'] for r in h),status=result.get('status','running'),
            best_train_success=max((r.get('dev_train_success',0) for r in ev),default=None),
            final_train_success=ev[-1].get('dev_train_success') if ev else None,
            best_perturb_success=max((r['success_rate'] for r in ev),default=None),
            final_perturb_success=ev[-1]['success_rate'] if ev else None,
            final_train_progress=ev[-1].get('dev_train_progress') if ev else None,
            final_perturb_progress=ev[-1]['mean_progress'] if ev else None,
            eval_iterations=[r['iteration'] for r in ev],
            max_logp_error=max((r.get('initial_logp_max_error',0) for r in h),default=0),
            stopped_kl_updates=sum(r.get('kl_early_stop',0) for r in h),
            wall_seconds=result.get('seconds'))
        evals=[read_json(p) for p in sorted((run/'evaluations').glob('*.json'))]
        root_retention={}
        for e in evals:
            for r in (e.get('development_training') or {}).get('roots',[]):
                root_retention.setdefault(r['name'],[]).append(bool(r['success']))
        row['train_root_retention']={k:dict(successes=sum(v),evaluations=len(v),last3=v[-3:]) for k,v in root_retention.items()}
        rows.append(row)
    common=min((len(h) for h in histories.values()),default=0)
    eval_sets=[{r['iteration'] for r in h if r.get('success_rate') is not None} for h in histories.values()]
    matched=max(set.intersection(*eval_sets),default=None) if eval_sets else None
    for row in rows:
        point=next((r for r in histories[row['name']] if r['iteration']==matched),None)
        row['matched_evaluation']={k:point.get(k) for k in ('iteration','dev_train_success','dev_train_progress','success_rate','mean_progress')} if point else None
    result=dict(scope='8 fixed developer train tasks; 8 1cm tangent perturbations; one seed; repeated evaluations are not independent trials',
        all_finished=all(r['status']!='running' for r in rows),common_completed_iterations=common,matched_evaluation_iteration=matched,rows=rows)
    atomic_json(batch/'comparison.json',result)
    table=['| 变体 | 轮次 | 固定训练峰值/末次 | 扰动峰值/末次 | 末次扰动progress |',
           '|---|---:|---:|---:|---:|']
    def fmt(x):return '—' if x is None else f'{x:.3f}'
    for r in rows:
        table.append(f"| {r['name']} | {r['iterations']} | {fmt(r['best_train_success'])}/{fmt(r['final_train_success'])} | {fmt(r['best_perturb_success'])}/{fmt(r['final_perturb_success'])} | {fmt(r['final_perturb_progress'])} |")
    note='本对照只测试开发小集合的可学性，不代表全随机任务泛化；训练场景固定，动作随机数按episode序号持续变化。4组并行运行，墙钟受竞争影响，不能作为独立的吞吐性能对照。'
    (batch/'SUMMARY.md').write_text('# V14 开发小任务对照\n\n'+note+'\n\n'+'\n'.join(table)+'\n')
    links=''.join(f'<li><b>{r["name"]}</b> · {r["iterations"]}轮 · {r["status"]} · '
        f'<a href="{r["name"]}/{r["name"]}/dashboard.html">逐轮曲线</a> · '
        f'<a href="{r["name"]}/{r["name"]}/episodes.html">末轮回放</a></li>' for r in rows)
    (batch/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>V14 小任务对照</title>'
        '<style>body{background:#101820;color:#dce8fa;font:16px system-ui;margin:3em;line-height:1.8}a{color:#82caff}img{max-width:100%}</style>'
        '<h1>V14 小任务可学性对照</h1><p>'+note+'</p><ul>'+links+'</ul><p><a href="SUMMARY.md">汇总</a> · '
        '<a href="comparison.json">原始比较JSON</a></p><img src="comparison.png" alt="相同迭代曲线"></html>')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    fields=[('dev_train_success','Fixed developer train success'),('success_rate','1cm perturbation success'),
            ('mean_progress','Perturbation progress'),('explore_success','Stochastic train success'),
            ('mode_switch_commit_rate','Commits per robot / 5Hz sample'),('explained_variance','Critic explained variance')]
    for ax,(key,title) in zip(axes.flat,fields):
        for name,h in histories.items():
            points=[(r['iteration'],r[key]) for r in h if r.get(key) is not None]
            if points:ax.plot(*zip(*points),label=name,marker='.' if key in ('dev_train_success','success_rate','mean_progress') else None,linewidth=1)
        ax.set(title=title,xlabel='PPO iteration');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7);fig.suptitle('8 fixed developer tasks + 8 small perturbations; one seed; not generalization')
    fig.tight_layout();fig.savefig(batch/'comparison.png',dpi=160);plt.close(fig)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--batch',type=Path,required=True)
    r=summarize(p.parse_args().batch);print(json.dumps(r,indent=2))
