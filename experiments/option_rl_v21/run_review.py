"""Publish a bounded set of V2 controller review examples; no policy training."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np

from experiments.option_rl_v19 import testbed_server as server
from experiments.option_rl_v21.review_cases import review_presets


def metrics(run):
    frames = run['frames']; initial = run['object']
    objects = np.asarray([f['object'] for f in frames], float)
    robots = np.asarray([f['robots'] for f in frames], float)
    angle = np.unwrap(objects[:, 2])-initial['angle']
    debug = [f['mb'][1].get('yield_debug') for f in frames if f['mb'][1].get('yield_debug')]
    # Controller motor-force requests (before the coordinator gate), NOT the
    # contact forces transmitted to the object. Keep gate_post and actual
    # contact/motion evidence in the replay for interpretation.
    forces = np.asarray([r['forces'] for r in run['strip']], float)
    norms = np.linalg.norm(forces, axis=2)
    cancellation = 1.-np.linalg.norm(forces.sum(axis=1), axis=1)/np.maximum(norms.sum(axis=1), 1e-9)
    cancellation[norms.sum(axis=1) < .1] = 0.
    return dict(id=run['id'], name=run['name'], periods=run['periods'], task_success=run['success'],
        termination=run['termination'], object_dx=float(objects[-1, 0]-initial['x']),
        object_dy=float(objects[-1, 1]-initial['y']), yaw_peak=float(np.abs(angle).max()),
        robot1_path=float(np.linalg.norm(np.diff(robots[:, 1, :2], axis=0), axis=1).sum()),
        robot1_object_contact_s=.05*sum(r['physical_contact'][1] for r in run['strip']),
        both_object_contact_s=.05*sum(all(r['physical_contact']) for r in run['strip']),
        motor_force_cancellation_mean=float(cancellation.mean()),
        yield_active_s=.05*sum(d['active'] for d in debug),
        yield_blocked_s=.05*sum(d['blocked'] for d in debug),
        direction_unavailable_s=[.05*sum(bool(f['mb'][i].get('direction_unavailable')) for f in frames) for i in (0, 1)],
        wall_seconds=run['timing']['session_seconds'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    server._seed_presets()
    rows = []
    started = time.monotonic()
    for preset_id, title, scene, sequence in review_presets():
        meta = server._scene_dict(server.SCENES[scene])
        body = dict(scene=scene, name=title, sequence=sequence,
                    object=meta['initial']['object'], robots=meta['initial']['robots'])
        run = server._run(body)
        row = metrics(run); row['preset_id'] = preset_id
        rows.append(row)
        (args.out/'metrics.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(row, ensure_ascii=False), flush=True)
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(Path('experiments/option_rl_v21').glob('*.py'))}
    sources.update(server._LOADED_RUNTIME_HASHES)
    (args.out/'manifest.json').write_text(json.dumps(dict(
        version=server.S.VERSION, attempts=len(rows), seconds=time.monotonic()-started,
        source_sha256=sources, training=False, note='Manual option review; task labels supplementary.'), indent=2)+'\n')


if __name__ == '__main__':
    main()
