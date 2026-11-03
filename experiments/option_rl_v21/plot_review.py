"""Offline figures from recorded V2 runs; no simulation or controller calls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    rows = json.loads(args.metrics.read_text())
    runs = {r['preset_id']: json.loads((Path('outputs/testbed/runs')/(r['id']+'.json')).read_text()) for r in rows}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for name, color, label in [('v2_blocker_brake', '#d45a4b', 'PUSH + BRAKE'),
                               ('v2_blocker_yield', '#168d9c', 'PUSH + YIELD')]:
        run = runs[name]; f = run['frames']; t = np.array([x['t'] for x in f])
        obj = np.array([x['object'] for x in f]); robot = np.array([x['robots'][1] for x in f])
        axes[0, 0].plot(obj[:, 0], obj[:, 1], color=color, label=label+' object')
        axes[0, 0].plot(robot[:, 0], robot[:, 1], '--', color=color, label=label+' robot 1')
        axes[0, 0].scatter(robot[0, 0], robot[0, 1], color=color, marker='o')
        axes[0, 0].scatter(robot[-1, 0], robot[-1, 1], color=color, marker='x')
        yaw = np.unwrap(obj[:, 2])-run['object']['angle']
        axes[0, 1].plot(t, yaw, color=color, label=label)
        contact = np.array([x['physical_contact'][1] for x in run['strip']], float)
        axes[1, 0].step(t, contact + (1.5 if name.endswith('brake') else 0), where='post', color=color, label=label)
    axes[0, 0].set(title='Same initial scene: object and yielding robot paths', xlabel='world x (m)', ylabel='world y (m)')
    axes[0, 0].axis('equal'); axes[0, 0].legend(fontsize=8)
    axes[0, 1].set(title='Object yaw relative to start', xlabel='simulation time (s)', ylabel='rad'); axes[0, 1].legend()
    axes[1, 0].set(title='Robot 1 physical contact (tick-start sensor)', xlabel='simulation time (s)', yticks=[0, 1, 1.5, 2.5], yticklabels=['YIELD off', 'YIELD on', 'BRAKE off', 'BRAKE on'])
    for name, color, label in [('v2_blocker_yield', '#168d9c', 'open space'), ('v2_wall_yield', '#cd9544', 'near wall')]:
        run = runs[name]
        xs = [(f['t'], f['mb'][1].get('yield_debug')) for f in run['frames']]
        xs = [(t, d) for t, d in xs if d and d.get('min_gap') is not None]
        axes[1, 1].plot([t for t, _ in xs], [d['min_gap'] for _, d in xs], color=color, label=label)
        blocked = [(t, d['min_gap']) for t, d in xs if d['blocked']]
        if blocked:
            axes[1, 1].scatter(*zip(*blocked), color=color, marker='x', s=18)
    axes[1, 1].axhline(.20, color='gray', ls=':', label='enter gap')
    axes[1, 1].set(title='YIELD minimum observed gap; x = blocked', xlabel='simulation time (s)', ylabel='m'); axes[1, 1].legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
