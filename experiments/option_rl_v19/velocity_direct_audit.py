"""Direct contact-velocity diagnostic vs EKF, using recorded testbed frames."""
from __future__ import annotations
import json, math, statistics
from pathlib import Path
import numpy as np

RUNS = {
    'N+E goal': '20261102_221100_197ae0',
    'N+E': '20261102_221100_80c863',
    'south 1.9m': '20261102_221101_c09c9a',
    'approach east': '20261102_221102_26080c',
    'single east': '20261102_221102_704097',
    'dual north': '20261102_221102_512353',
}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def true_vel(fr, i):
    a, b = fr[i-1]['object'], fr[i+1]['object']; dt = fr[i+1]['t']-fr[i-1]['t']
    return np.array([(b[0]-a[0])/dt, (b[1]-a[1])/dt]), wrap(b[2]-a[2])/dt


def solve(mbs, true=None):
    rows, z = [], []
    for m in mbs:
        if not m or not m.get('contact'):
            continue
        q = (m.get('candidate') or {}).get('q')
        n = m.get('face_normal'); v = m.get('own_v')
        if not q or not n or not v:
            continue
        q = np.asarray(q, float); n = np.asarray(n, float)
        n = n/max(1e-9, np.linalg.norm(n))
        # J q = (-qy, qx)
        rows.append([n[0], n[1], n[0]*(-q[1]) + n[1]*(q[0])])
        z.append(float(n @ np.asarray(v, float)))
    if not rows:
        return None
    H = np.asarray(rows, float); z = np.asarray(z, float)
    # weak zero prior on all three components; this is a diagnostic, not a controller
    lam = .02
    A = H.T @ H + lam*np.eye(3)
    x = np.linalg.solve(A, H.T @ z)
    return x


def main():
    out = []
    for name, rid in RUNS.items():
        d = json.load(open(f'outputs/testbed/runs/{rid}.json')); fr = d['frames']
        errs = []; derr = []; angs = []; sp = []
        for i in range(1, len(fr)-1):
            vt, wt = true_vel(fr, i)
            xe = None
            for m in fr[i].get('mb') or []:
                if m and m.get('v_est'):
                    xe = np.asarray(m['v_est'][:2], float); we = float(m['v_est'][2]); break
            xd = solve(fr[i].get('mb') or [])
            if xd is None:
                continue
            errs.append(float(np.linalg.norm(xd[:2]-vt)))
            angs.append(abs(wrap(float(xd[2])-wt)))
            sp.append(float(np.linalg.norm(vt)))
            if xe is not None:
                derr.append(float(np.linalg.norm(xe-vt)))
        def stats(a):
            if not a: return None
            return dict(n=len(a), mean=round(float(statistics.mean(a)),4),
                        median=round(float(statistics.median(a)),4),
                        p90=round(sorted(a)[int(.9*(len(a)-1))],4))
        out.append(dict(name=name, id=rid, periods=d['periods'], success=d['success'],
                        true_speed_mean=round(float(statistics.mean(sp)),4) if sp else None,
                        ekf=stats(derr), direct=stats(errs), direct_ang=stats(angs)))
        print(name, 'direct', stats(errs), 'ekf', stats(derr), 'truev', out[-1]['true_speed_mean'])
    Path('outputs/testbed/direct_velocity_audit.json').write_text(json.dumps(out, indent=2))

if __name__ == '__main__':
    main()
