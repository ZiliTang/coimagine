"""Pre-holdout amendment: obtain repair states outside supervised initial roots.

The original implementation/evidence remain immutable. This extension adds a
separate developer cohort; it does not change any controller, model or test root.
"""
import collections, time, traceback
from . import random_room_coverage_v2 as N
from experiments.planar_tasks.relational_search.pipeline import Campaign,read,save


def prepare_amendment():
    p=N.BASE/'repair_development_panel'
    if N.frozen(p):return
    if (N.BASE/'evaluation').exists():raise RuntimeError('cannot amend after fresh evaluation begins')
    c=N.make(p);cases=N.panel_rows(2026111743,'repairdev',100)
    previous=read(N.PARENT/'development.json')['cases']+read(N.PARENT/'fresh_test.json')['cases']+read(N.PARENT/'old_diagnostic.json')['cases']
    seen={N.digest(r['fixture']) for r in previous}
    for r in cases:
        if r['fixture_sha256'] in seen:raise ValueError('repair development overlaps prior roots')
        seen.add(r['fixture_sha256']);N.X.fixture(r['fixture'])
    save(p/'cases.json',dict(seed=2026111743,cases=cases))
    save(p/'amendment.json',dict(reason='Perfect imitation on supervised roots would conceal recovery-generalization gaps; use independent developer roots.',
        created_before_fresh_test=True,repair_roots=100,all_CAP_reference_episodes=1000,learner_probe_episodes=100,
        unchanged_original_training_probe=200,maximum_total_formal_episodes=7990,declared_ceiling=8000,
        fresh_test_unchanged_sha256=N.sha(N.PARENT/'fresh_test.json'),old_source_unchanged_sha256=N.sha(N.ROOT/'experiments/planar_tasks/authoring_v2/random_room_coverage_v2.py'),
        data_use='repairdev root trajectories are NOT added as ordinary demonstrations; only successful takeover suffixes may be added',
        selection='same fixed up-to6 misses per movable-count stratum and predeclared cuts; no fresh-test outcome used'))
    N.finish(c);save(N.HOST/'AMENDMENT_PREPARED.json',dict(panel=str(p),count=100,geometry_checks=100))


def repair_collect():
    # Keep the original200-root probe to measure training-root retention.
    cases=read(N.PARENT/'development.json')['cases'];out=N.BASE/'expanded_training_rollouts'
    N.physical_phase(out,[N.job_for(c,'expanded_GDA',out) for c in cases],'expanded_GDA_newdev')
    cases=read(N.BASE/'repair_development_panel/cases.json')['cases'];out=N.BASE/'repair_development_CAPs'
    teachers=N.physical_phase(out,[N.job_for(c,e['name'],out) for e in N.entries() for c in cases],'repair_development_CAPs')
    out=N.BASE/'expanded_repair_rollouts'
    learned=N.physical_phase(out,[N.job_for(c,'expanded_GDA',out) for c in cases],'expanded_GDA_repairdev')
    out=N.BASE/'repair'
    if N.frozen(out):return
    jobs,selection=N.handoff_jobs(cases,learned,teachers,out,max_per_stratum=6)
    N.make(out);save(out/'selection.json',selection);N.physical_phase(out,jobs,'repair_handoffs')


def summarize():
    ev=read(N.BASE/'evaluation/result.json')['rows'];old=read(N.O.HOST/'RESULT.json')['rows'];rows=list(ev)
    for r in old:
        name=r['condition'] if r['condition'].startswith('CAP-') else 'original_'+r['condition']
        rows.append(dict(r,condition=name,imported=True))
    stats={};names=sorted({r['condition'] for r in rows});paired={}
    for split in ('dev','validation','test','fresh_test'):
        sub=[r for r in rows if r['split']==split];union={r['case_id'] for r in sub if r['condition'].startswith('CAP-') and r['success']};stats[split]={'teacher_union':len(union)}
        for name in names:
            rr=[r for r in sub if r['condition']==name];wins={r['case_id'] for r in rr if r['success']}
            stats[split][name]=dict(n=len(rr),success=len(wins),retained=len(wins&union),lost=sorted(union-wins),outside=sorted(wins-union))
        for tier in ('expanded','repaired'):
            for arm in N.ARMS:
                left=tier+'_'+arm;right=('control' if tier=='expanded' else 'expanded')+'_'+arm
                a={r['case_id'] for r in sub if r['condition']==left and r['success']};b={r['case_id'] for r in sub if r['condition']==right and r['success']}
                paired[split+':'+left+' vs '+right]=dict(gained=sorted(a-b),lost=sorted(b-a))
    phases=[p for p in N.BASE.iterdir() if (p/'manifest.json').exists()];seals={str(p):Campaign(p).verify_frozen() for p in phases}
    executions=[r for phase in ('diagnosis','collection','expanded_training_rollouts','repair_development_CAPs','expanded_repair_rollouts','repair','evaluation') for r in read(N.BASE/phase/'result.json')['rows']]
    result=dict(status='complete',summary=stats,paired=paired,rows=rows,seals=seals,new_episodes=len(executions),new_ticks=sum(r['ticks'] for r in executions),fits=9,no_new_cap=True,no_rl=True,
        amendment=read(N.BASE/'repair_development_panel/amendment.json'),preflight=read(N.HOST/'PREFLIGHT.json'),
        datasets={t:read(N.data_path(t)/'extraction.json')['demonstrations'] for t in N.TIERS},
        boundaries={t:read(N.data_path(t)/'extraction.json')['boundaries'] for t in N.TIERS},
        fit_accuracy={t+'_'+a:read(N.BASE/('fit_'+t)/a/'complete.json')['accuracy'] for t in N.TIERS for a in N.ARMS})
    if result['new_episodes']>8000 or result['new_ticks']>14400000:raise ValueError('declared total ceiling exceeded')
    save(N.HOST/'RESULT.json',result);N.report(result)
    p=N.ROOT/'docs/agent_exchange/reviews/Random_Room_Coverage_Repair_20261117.md'
    p.write_text(p.read_text()+'\n## 预先修正的接管采样设计\n\n原计划在已监督的200初态上找接管失败，会受完整模仿造成的选择偏差影响。因此在扩充训练期间、任何新最终留出执行之前，另冻结100个repairdev初态（seed2026111743），执行10CAP和expanded GDA，仅将其真实接管成功后缀用于repaired。原200初态仍执行训练覆盖检查。新最终留出100个不变。原驱动只在编排层停止；在途优化完成后由新增版本接续，未改旧源码或重跑优化；原退出与修正记录保留。\n\n新增接管数据与expanded的比较并非等标签数量对照，因此若改善，只能说明这批追加恢复数据有帮助，不能单独证明定向接管比等量随机数据更高效。工程预检另有1执行/218ticks，不混入正式覆盖统计。\n')
    save(N.HOST/'FINISHED.json',dict(status='complete',episodes=result['new_episodes'],ticks=result['new_ticks'],sealed_phases=len(seals),amended_repair_cohort=True))


if __name__=='__main__':
    try:
        prepare_amendment();N.repair_collect=repair_collect;N.summarize=summarize;N.run_all()
    except BaseException:
        save(N.HOST/'AMENDED_FAILURE.json',dict(at=time.time(),traceback=traceback.format_exc()));N.update(stage='failed',status='failed');raise
