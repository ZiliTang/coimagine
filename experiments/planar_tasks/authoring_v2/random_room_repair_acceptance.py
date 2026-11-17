"""Predeclared post-fit mechanism checks on repair-development roots/prefixes."""
from . import random_room_coverage_v2 as N
from experiments.planar_tasks.relational_search.pipeline import read,save,Campaign


def run():
    primary=read(N.HOST/'RESULT.json');cases=read(N.BASE/'repair_development_panel/cases.json')['cases'];byid={r['id']:r for r in cases}
    selected=read(N.BASE/'data_repaired/selected.json');selection={r['case_id']:r for r in read(N.BASE/'repair/selection.json')}
    out=N.BASE/'repair_acceptance';jobs=[N.job_for(c,'repaired_GDA',out) for c in cases]
    for r in selected:
        summary=read(N.Path(r['result_path']).parent/'summary.json');case=byid[summary['case_id']];ref=selection[case['id']]['learner_path']
        jobs.append(N.job_for(case,'repaired_GDA',out,cut=r['start_boundary'],reference=ref,reference_sha256=N.sha(ref)))
    if primary['new_episodes']+len(jobs)>8000:raise RuntimeError('combined predeclared episode ceiling')
    Campaign(N.BASE/'fit_repaired').verify_frozen();rows=N.physical_phase(out,jobs,'repair_mechanism_acceptance')
    roots=[r for r in rows if not r['cut']];suffixes=[r for r in rows if r['cut']]
    baseline=read(N.BASE/'expanded_repair_rollouts/result.json')['rows'];teachers=read(N.BASE/'repair_development_CAPs/result.json')['rows']
    old={r['case_id'] for r in baseline if r['success']};new={r['case_id'] for r in roots if r['success']};union={r['case_id'] for r in teachers if r['success']}
    result=dict(episodes=len(rows),ticks=sum(r['ticks'] for r in rows),combined_formal_episodes=primary['new_episodes']+len(rows),combined_formal_ticks=primary['new_ticks']+sum(r['ticks'] for r in rows),
        roots=dict(n=len(roots),before=len(old),after=len(new),gained=sorted(new-old),lost=sorted(old-new),teacher_union=len(union),retained=len(new&union),outside=sorted(new-union)),
        suffixes=dict(n=len(suffixes),success=sum(r['success'] for r in suffixes),failed=[dict(case=r['case_id'],cut=r['cut']) for r in suffixes if not r['success']],prefix_checks=sum(bool(r['prefix_audit']['physics_equal'] and r['prefix_audit']['public_equal']) for r in suffixes)),
        seal=Campaign(out).verify_frozen(),scope='repair-development mechanism/retention checks; not heldout generalization; no further model updates')
    save(N.HOST/'REPAIR_ACCEPTANCE.json',result)
    p=N.ROOT/'docs/agent_exchange/reviews/Random_Room_Coverage_Repair_20261117.md';q=result['roots'];s=result['suffixes']
    p.write_text(p.read_text()+f"\n## 预注册修复机制验收\n\n同一真实失败前缀后，repaired GDA完成{s['success']}/{s['n']}个已提供成功后缀的接管任务；{s['prefix_checks']}个公开/物理前缀一致。从100个接管开发初态重新执行，expanded→repaired为{q['before']}→{q['after']}，新增{len(q['gained'])}、丢失{len(q['lost'])}；教师并集{q['teacher_union']}。这是开发机制验收，不是独立泛化结果。\n\n该项追加{result['episodes']}执行/{result['ticks']}ticks；与主实验合计{result['combined_formal_episodes']}执行/{result['combined_formal_ticks']}ticks，未超过原8000次上限。另有工程预检1执行/218ticks。选定检查在新最终留出执行之前记录于REPAIR_ACCEPTANCE_PROTOCOL.json；验收没有训练或修改任何模型。\n")
    N.update(stage='complete',status='complete',repair_acceptance=result['suffixes'],combined_episodes=result['combined_formal_episodes'])


if __name__=='__main__':run()
