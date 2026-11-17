"""Validate/export development tasks or run one explicitly selected policy episode."""
import argparse
import json
from pathlib import Path

from .tasks import task_catalog, geometry_audit, scene_metadata, VERSION


def export_tasks(directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    from coimagine.random_start import geometry as G, spec as R
    from .runner import write_json
    from .interface import contract

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    tasks = task_catalog()
    write_json(directory/'catalog.json', dict(version=VERSION, split='development',
        tasks=[dict(**t.record(), sha256=t.sha256, audit=geometry_audit(t.instance))
               for t in tasks.values()]))
    write_json(directory/'policy_contract.json', contract())
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    for ax, task in zip(axes.flat, tasks.values()):
        inst = task.instance
        for b in inst.obstacles:
            ax.add_patch(Polygon(G.box_polygon(b.cx, b.cy, b.angle, b.half_w, b.half_h),
                                 color='#555d68'))
        obj = inst.object_pose
        ax.add_patch(Polygon(G.box_polygon(obj.x, obj.y, obj.angle,
                                          R.OBJECT_WIDTH/2, R.OBJECT_HEIGHT/2), color='#3498db'))
        for i, r in enumerate(inst.robot_poses):
            ax.add_patch(Circle((r.x, r.y), R.ROBOT_RADIUS, color=('#e6a23c', '#2eaf7d')[i]))
            ax.text(r.x, r.y, str(i), ha='center', va='center', fontsize=8)
        ax.add_patch(Circle(inst.goal_centre, inst.goal_radius, fill=False, color='#a868cc', lw=2))
        ax.set(xlim=(0, inst.arena[0]), ylim=(0, inst.arena[1]), aspect='equal',
               title=inst.name, xlabel='x (m)', ylabel='y (m)')
        ax.grid(alpha=.15)
    fig.suptitle('CoImagine planar development v1 — geometric audit only; solvability unverified')
    fig.savefig(directory/'task_atlas.png', dpi=160)
    plt.close(fig)
    return dict(directory=str(directory), tasks=len(tasks), all_geometrically_legal=all(
        geometry_audit(t.instance)['legal'] for t in tasks.values()))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='operation', required=True)
    sub.add_parser('validate', help='geometry checks only; no physics rollout')
    export = sub.add_parser('export', help='catalog, policy contract and task atlas')
    export.add_argument('--out', required=True)
    run = sub.add_parser('run', help='one bounded local JSONL policy episode; no automatic API calls')
    run.add_argument('--task', choices=tuple(task_catalog()), required=True)
    run.add_argument('--out', required=True)
    run.add_argument('--max-decisions', type=int, default=300)
    run.add_argument('--wall-seconds', type=float, default=120.)
    run.add_argument('--response-seconds', type=float, default=20.)
    run.add_argument('--publish-testbed', action='store_true', help='copy this replay into local testbed history')
    run.add_argument('--command', nargs=argparse.REMAINDER, required=True)
    args = p.parse_args()
    if args.operation == 'validate':
        audits = {name: geometry_audit(t.instance) for name, t in task_catalog().items()}
        print(json.dumps(audits, indent=2))
        return 0 if all(a['legal'] for a in audits.values()) else 1
    if args.operation == 'export':
        print(json.dumps(export_tasks(args.out), indent=2))
        return 0
    if not args.command:
        p.error('--command requires an executable and arguments')
    from .runner import RunBudget, run_episode
    result = run_episode(task_catalog()[args.task], args.command, args.out,
                         RunBudget(args.max_decisions, args.wall_seconds, args.response_seconds))
    if args.publish_testbed:
        # Publishing is explicit; IDs cannot overwrite old runs.
        import uuid
        from .runner import ROOT
        path = ROOT/'outputs/testbed/runs'/f'planar_{uuid.uuid4().hex[:12]}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.loads((Path(args.out)/'replay.json').read_text())
        data['id'] = path.stem
        with path.open('x') as f:
            json.dump(data, f)
        result['testbed_url'] = f'http://127.0.0.1:8765/?run={path.stem}'
    print(json.dumps(result, indent=2))
    return 0 if result['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
