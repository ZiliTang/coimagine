"""D009 v2 prototype: prior-free rigid-motion EKF + surface-velocity PUSH.

No object size, centre, rectangle, endpoint, or face-ID prior.  The maintained
controller uses only the robot's legal scan/patch observations, its own velocity,
and actual contact feedback.  The EKF state is a rigid twist at a fixed world
reference point; lines are local geometric features, not object faces.
"""
from __future__ import annotations
import math
import os
from dataclasses import dataclass
import numpy as np

from coimagine.random_start import spec
from coimagine.random_start.skills import SkillObservation, patch_gap
from experiments.option_rl_v13.push_controller import ContinuousPush
from experiments.option_rl_v13 import controller as C
from experiments.option_rl_v19 import spec as S

USE_QPRIME = os.environ.get('D013_USE_QPRIME', '1') != '0'

Vec = tuple[float, float]
J = np.array([[0., -1.], [1., 0.]])


def unit(v):
    z = math.hypot(v[0], v[1])
    return (v[0]/z, v[1]/z) if z > 1e-12 else (0., 0.)


def t90(v):
    return (-v[1], v[0])


def dot(a, b):
    return a[0]*b[0] + a[1]*b[1]


class RigidMotionEKF:
    """Twist at a fixed world reference; no object centre/size/endpoints."""
    def __init__(self):
        self.x = np.zeros(3)                 # vx, vy, omega at origin
        self.P = np.diag([.20**2, .20**2, .50**2])
        self.nullspace = np.zeros((3, 0))
        self.history = []
        self.last_line = {}

    def predict(self, dt):
        # random-walk velocity model: q has units m^2/s^3, so the one-step
        # covariance is q*dt, not the position-integrated dt^3/3 form.
        self.P = self.P + np.diag([.5**2*dt, .5**2*dt, 1.0**2*dt])
        # constant velocity; the material reference point is not reset

    def update(self, H, z, R):
        H = np.asarray(H, float); z = float(z); R = float(R)
        S = float(H @ self.P @ H.T + R)
        if not np.isfinite(S) or S <= 1e-12:
            return None
        y = z - float(H @ self.x)
        nis = y*y/S
        if nis > 9.0:
            return dict(rejected=True, nis=nis, innovation=y)
        K = self.P @ H.T / S
        self.x = self.x + K*y
        self.P = self.P - np.outer(K, H @ self.P)
        self.P = 0.5*(self.P + self.P.T)
        return dict(rejected=False, nis=nis, innovation=y, K=K)

    def apply_nullspace_prior(self, rows, sigma=.05):
        """Pull unobservable twist directions to zero with a soft prior."""
        H = np.asarray(rows, float)
        if H.ndim != 2 or H.shape[0] == 0:
            return None
        try:
            _, sv, vt = np.linalg.svd(H, full_matrices=True)
        except np.linalg.LinAlgError:
            return None
        tol = max(H.shape) * (sv[0] if len(sv) else 0.) * 1e-6
        rank = int(np.sum(sv > tol))
        if rank >= H.shape[1]:
            self.nullspace = np.zeros((H.shape[1], 0))
            return None
        N = vt[rank:].T
        self.nullspace = N
        # zero the unobservable mean; keep a finite prior covariance
        self.x = self.x - N @ (N.T @ self.x)
        Pn = N.T @ self.P @ N
        target = np.eye(N.shape[1]) * float(sigma)**2
        self.P = self.P + N @ (target - Pn) @ N.T
        self.P = 0.5*(self.P + self.P.T)
        return N

    def velocity_at(self, p):
        v = self.x[0:2] + self.x[2] * (J @ np.asarray(p, float))
        return (float(v[0]), float(v[1]))

    def contact(self, p, n, v_robot):
        p = np.asarray(p, float); n = np.asarray(n, float)
        z = float(n @ np.asarray(v_robot, float))
        H = np.array([n[0], n[1], float(n @ (J @ p))])
        return self.update(H, z, .02**2)

    def line(self, n, rho, phi, drho, dphi, dt):
        n = np.asarray(n, float)
        # rho_dot = n . v at the closest point of the infinite line
        self.update(np.array([n[0], n[1], 0.]), drho, .01**2)
        # line orientation rate is a direct omega observation
        self.update(np.array([0., 0., 1.]), dphi, (math.radians(2.0))**2)


@dataclass
class Line:
    n: Vec
    rho: float
    phi: float
    stamp: float
    hits: int = 1


@dataclass
class ContactCandidate:
    id: int
    q: tuple
    n: tuple
    t: tuple
    selected_time: float
    verified_time: float
    support: tuple
    margin: float
    valid: bool = True
    updates: int = 0
    reason: str = 'initial'
    view_ticks: int = 0


def shift_sign_for_error(e):
    if e is None:
        return 0.0
    return 1.0 if e >= 0 else -1.0


def choose_candidate_q(lo, hi, m=.06, min_span=.02):
    """Return (q, margin, status) from a visible support interval."""
    span = float(hi - lo)
    if span < min_span:
        return None, 0.0, 'view_insufficient'
    if span >= 2*m + .02:
        q = float(min(max((lo+hi)/2, lo+m), hi-m))
        return q, m, 'ok'
    q = (lo+hi)/2
    return q, max(0., min(m, span/2)), 'margin_limited'




def face_support(obs, action):
    """Best visible support patch aligned with the requested push direction."""
    d = np.asarray(action.effect_direction, float)
    best = None
    for patch in obs.patches:
        if not patch.valid or len(patch.points_world) < 3:
            continue
        n = np.asarray(patch.normal_world, float)
        n = n / max(1e-12, np.linalg.norm(n))
        align = float(n @ d)
        if align < math.cos(math.radians(35)):
            continue
        pts = np.asarray(patch.points_world, float)
        t = np.asarray(t90(n), float)
        proj = pts @ t
        lo, hi = float(proj.min()), float(proj.max())
        score = (align, patch.confidence, hi-lo)
        if best is None or score > best[0]:
            best = (score, n, t, float(n @ pts[0]), lo, hi, pts)
    return best




def _support_qt(face, own, margin=.04, local_q=None, local_radius=None):
    _, n, t, rho, lo, hi, pts = face
    lo2, hi2 = lo + margin, hi - margin
    if lo2 > hi2:
        lo2 = hi2 = .5*(lo+hi)
    qt = float(np.dot(np.asarray(own, float), np.asarray(t, float)))
    if local_q is not None and local_radius is not None:
        lq = np.asarray(local_q, float)
        if abs(float(lq @ np.asarray(n, float)) - rho) <= .05:
            lqt = float(lq @ np.asarray(t, float))
            lo2 = max(lo2, lqt - float(local_radius))
            hi2 = min(hi2, lqt + float(local_radius))
            if lo2 > hi2:
                lo2 = hi2 = min(max(lqt, lo+margin), hi-margin)
            qt = lqt
    return min(max(qt, lo2), hi2), lo2, hi2




def visible_faces(obs):
    """All visible support faces, without filtering by the robot action."""
    faces = []
    for patch in obs.patches:
        if not patch.valid or len(patch.points_world) < 3:
            continue
        pts = np.asarray(patch.points_world, float)
        n = np.asarray(patch.normal_world, float)
        n = n / max(1e-12, float(np.linalg.norm(n)))
        t = np.asarray(t90(n), float)
        rho = float(n @ pts[0])
        proj = pts @ t
        lo, hi = float(proj.min()), float(proj.max())
        merged = False
        for f in faces:
            # Same infinite line is insufficient: disjoint fragments can be
            # different polygon segments. Compare intervals in ONE basis.
            fp = pts @ np.asarray(f[2], float)
            if (float(np.asarray(f[1]) @ n) >= .98 and abs(f[3]-rho) <= .03
                    and float(fp.min()) <= float(f[5]) + .04
                    and float(fp.max()) >= float(f[4]) - .04):
                f[4] = min(f[4], float(fp.min())); f[5] = max(f[5], float(fp.max()))
                f[6] = tuple(np.vstack([np.asarray(f[6]), pts]))
                merged = True
                break
        if not merged:
            faces.append([patch, n, t, rho, lo, hi, tuple(pts)])
    return faces


def _face_world(f, qt):
    return np.asarray(f[1], float)*float(f[3]) + np.asarray(f[2], float)*float(qt)


def _face_bounds(f, own, margin=.04, prev_q=None, local_radius=None):
    lo, hi = float(f[4])+margin, float(f[5])-margin
    if lo > hi:
        lo = hi = .5*(float(f[4])+float(f[5]))
    t = np.asarray(f[2], float); n = np.asarray(f[1], float)
    init = float(np.asarray(own, float) @ t)
    if prev_q is not None and local_radius is not None:
        pq = np.asarray(prev_q, float)
        if abs(float(pq @ n) - float(f[3])) <= .05:
            init = float(pq @ t)
            lo = max(lo, init - float(local_radius))
            hi = min(hi, init + float(local_radius))
            if lo > hi:
                lo = hi = min(max(init, float(f[4])+margin), float(f[5])-margin)
    return min(max(init, lo), hi), lo, hi


def _same_face(f0, f1):
    return (float(np.asarray(f0[1]) @ np.asarray(f1[1])) >= math.cos(math.radians(15))
            and abs(float(f0[3])-float(f1[3])) <= .04)


def _pair_loss(f0, f1, q0t, q1t, own0, own1, desired_force, desired_torque, sep, o_ref_override=None):
    q0 = _face_world(f0, q0t); q1 = _face_world(f1, q1t)
    n0 = np.asarray(f0[1], float); n1 = np.asarray(f1[1], float)
    if o_ref_override is not None:
        o_ref = np.asarray(o_ref_override, float)
    else:
        pts = np.vstack([np.asarray(f0[6], float), np.asarray(f1[6], float)])
        o_ref = np.array([(pts[:,0].min()+pts[:,0].max())*.5,
                          (pts[:,1].min()+pts[:,1].max())*.5], float)
    F = n0+n1
    M = float((q0[0]-o_ref[0])*n0[1] - (q0[1]-o_ref[1])*n0[0]
              + (q1[0]-o_ref[0])*n1[1] - (q1[1]-o_ref[1])*n1[0])
    fd = np.asarray(desired_force, float); fd = fd/max(1e-9, float(np.linalg.norm(fd)))
    force_err = 1.0 - float(F @ fd)/max(1e-9, float(np.linalg.norm(F)))
    torque_err = (M-float(desired_torque))**2
    sep_err = max(0.0, float(sep)-float(np.linalg.norm(q0-q1)))**2
    return 2.0*force_err + 20.0*torque_err + 30.0*sep_err, q0, q1, o_ref, M, F


def _optimize_face_pair(f0, f1, obs0, obs1, desired_force, desired_torque, sep,
                        prev_q0=None, prev_q1=None, local_radius=None,
                        steps=(.20, .10, .05), o_ref_override=None):
    q0t, lo0, hi0 = _face_bounds(f0, (obs0.own.x, obs0.own.y), .04, prev_q0, local_radius)
    q1t, lo1, hi1 = _face_bounds(f1, (obs1.own.x, obs1.own.y), .04, prev_q1, local_radius)
    s0 = float(np.asarray((obs0.own.x, obs0.own.y)) @ np.asarray(f0[2], float))
    s1 = float(np.asarray((obs1.own.x, obs1.own.y)) @ np.asarray(f1[2], float))
    order = 1 if s0 <= s1 else -1
    same = _same_face(f0, f1)
    best, q0, q1, o_ref, M, F = _pair_loss(f0, f1, q0t, q1t, obs0.own, obs1.own,
                                            desired_force, desired_torque, sep, o_ref_override)
    for step in steps:
        for _ in range(5):
            improved = False
            for side in (0, 1):
                if side == 0:
                    cand = [q0t-step, q0t, q0t+step]
                    if same:
                        cand = [min(x, q1t-1e-4) if order > 0 else max(x, q1t+1e-4) for x in cand]
                    cand = [min(max(x, lo0), hi0) for x in cand]
                    for x in cand:
                        c, a, b, o, m, ff = _pair_loss(f0, f1, x, q1t, obs0.own, obs1.own,
                                                        desired_force, desired_torque, sep, o_ref_override)
                        if c < best-1e-12:
                            best, q0t, q0, q1, o_ref, M, F, improved = c, x, a, b, o, m, ff, True
                else:
                    cand = [q1t-step, q1t, q1t+step]
                    if same:
                        cand = [max(x, q0t+1e-4) if order > 0 else min(x, q0t-1e-4) for x in cand]
                    cand = [min(max(x, lo1), hi1) for x in cand]
                    for x in cand:
                        c, a, b, o, m, ff = _pair_loss(f0, f1, q0t, x, obs0.own, obs1.own,
                                                        desired_force, desired_torque, sep, o_ref_override)
                        if c < best-1e-12:
                            best, q1t, q0, q1, o_ref, M, F, improved = c, x, a, b, o, m, ff, True
            if not improved:
                break
    return dict(cost=float(best), q0=q0, q1=q1, o_ref=o_ref,
                separation=float(np.linalg.norm(q0-q1)), same_face=bool(same),
                face0=f0, face1=f1, q0t=float(q0t), q1t=float(q1t),
                M=float(M), F=(float(F[0]), float(F[1])),
                support0=(float(f0[4]), float(f0[5])), support1=(float(f1[4]), float(f1[5])))






def select_contact_point(obs, desired_force=(0., -1.), prev_q=None, local_radius=None,
                         margin=.04, steps=(.20, .10, .05), o_ref_override=None):
    """Single-contact version of the visible-support selector."""
    faces = visible_faces(obs)
    if local_radius is not None and prev_q is not None:
        faces = [f for f in faces if abs(float(np.asarray(prev_q, float) @ np.asarray(f[1], float))-float(f[3])) <= .05]
    if not faces:
        return None
    fd = np.asarray(desired_force, float); fd = fd/max(1e-9, float(np.linalg.norm(fd)))
    best = None
    for f in faces:
        q, lo, hi = _face_bounds(f, (obs.own.x, obs.own.y), margin, prev_q, local_radius)
        pts = np.asarray(f[6], float)
        if o_ref_override is not None:
            o_ref = np.asarray(o_ref_override, float)
        else:
            o_ref = np.array([(pts[:,0].min()+pts[:,0].max())*.5, (pts[:,1].min()+pts[:,1].max())*.5])
        def cost(qt):
            w = _face_world(f, qt); n = np.asarray(f[1], float)
            force_err = 1.0 - float(n @ fd)
            M = float((w[0]-o_ref[0])*n[1] - (w[1]-o_ref[1])*n[0])
            return 2.0*force_err + 20.0*M*M, w
        c, w = cost(q)
        for step in steps:
            for _ in range(5):
                improved = False
                for x in (q-step, q, q+step):
                    x = min(max(x, lo), hi)
                    cc, ww = cost(x)
                    if cc < c-1e-12:
                        c, q, w, improved = cc, x, ww, True
                if not improved:
                    break
        if best is None or c < best['cost']:
            best = dict(q=(float(w[0]), float(w[1])), cost=float(c),
                        support=(float(f[4]), float(f[5])),
                        n=(float(f[1][0]), float(f[1][1])),
                        t=(float(f[2][0]), float(f[2][1])))
    return best





def contour_center(faces):
    """Area centroid of the convex hull of observed/memory boundary points."""
    pts = []
    for f in faces:
        for q in np.asarray(f[6], float):
            if np.all(np.isfinite(q)):
                pts.append((float(q[0]), float(q[1])))
    if len(pts) < 3:
        if pts:
            a = np.asarray(pts, float); return (float(a[:,0].mean()), float(a[:,1].mean()))
        return None
    pts = sorted(set(pts))
    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        a = np.asarray(pts, float); return (float(a[:,0].mean()), float(a[:,1].mean()))
    area2 = 0.; cx = 0.; cy = 0.
    for i in range(len(hull)):
        x0, y0 = hull[i]; x1, y1 = hull[(i+1) % len(hull)]
        cr = x0*y1 - x1*y0
        area2 += cr; cx += (x0+x1)*cr; cy += (y0+y1)*cr
    if abs(area2) < 1e-12:
        a = np.asarray(hull, float); return (float(a[:,0].mean()), float(a[:,1].mean()))
    return (float(cx/(3.*area2)), float(cy/(3.*area2)))



SIMPLE_SELECTOR = os.environ.get('D013_SIMPLE_SELECTOR', '1') != '0'


def _best_action_face(obs, action):
    faces = visible_faces(obs)
    if not faces:
        return None
    d = np.asarray(action.effect_direction, float)
    d = d/max(1e-9, float(np.linalg.norm(d)))
    return max(faces, key=lambda f: abs(float(np.asarray(f[1], float) @ d)))


def _face_center_q(f):
    qt = .5*(float(f[4])+float(f[5]))
    q = np.asarray(f[1], float)*float(f[3]) + np.asarray(f[2], float)*qt
    return q, qt


def _simple_pair_solution(f0, f1, desired_force):
    sep = 2*spec.ROBOT_RADIUS + .06
    same = (float(np.asarray(f0[1]) @ np.asarray(f1[1])) >= math.cos(math.radians(15))
            and abs(float(f0[3])-float(f1[3])) <= .04)
    if same:
        lo = max(float(f0[4]), float(f1[4])) + .04
        hi = min(float(f0[5]), float(f1[5])) - .04
        span = hi-lo
        if span < sep:
            return None
        q0t = lo + span/3.; q1t = lo + 2.*span/3.
        if q1t-q0t < sep:
            mid=.5*(lo+hi); q0t=mid-sep/2; q1t=mid+sep/2
        q0 = np.asarray(f0[1])*float(f0[3]) + np.asarray(f0[2])*q0t
        q1 = np.asarray(f1[1])*float(f1[3]) + np.asarray(f1[2])*q1t
    else:
        q0 = np.asarray(f0[1])*float(f0[3]) + np.asarray(f0[2])*.5*(f0[4]+f0[5])
        q1 = np.asarray(f1[1])*float(f1[3]) + np.asarray(f1[2])*.5*(f1[4]+f1[5])
        if float(np.linalg.norm(q0-q1)) < sep:
            return None
    pts = np.vstack([np.asarray(f0[6],float), np.asarray(f1[6],float)])
    o_ref = np.array([(pts[:,0].min()+pts[:,0].max())*.5,
                      (pts[:,1].min()+pts[:,1].max())*.5])
    n0 = np.asarray(f0[1],float); n1 = np.asarray(f1[1],float)
    F = n0+n1
    M = float((q0[0]-o_ref[0])*n0[1]-(q0[1]-o_ref[1])*n0[0]
              +(q1[0]-o_ref[0])*n1[1]-(q1[1]-o_ref[1])*n1[0])
    fd = np.asarray(desired_force,float); fd=fd/max(1e-9,float(np.linalg.norm(fd)))
    force_err = 1.0-float(F@fd)/max(1e-9,float(np.linalg.norm(F)))
    cost = 2.0*force_err + 20.0*M*M
    return dict(q0=(float(q0[0]),float(q0[1])), q1=(float(q1[0]),float(q1[1])),
                support0=(float(f0[4]),float(f0[5])), support1=(float(f1[4]),float(f1[5])),
                sep=float(np.linalg.norm(q0-q1)), cost=float(cost), M=M,
                n0=(float(n0[0]),float(n0[1])), n1=(float(n1[0]),float(n1[1])))


def select_simple_pair(obs0, action0, obs1, action1):
    faces0 = visible_faces(obs0)
    faces1 = visible_faces(obs1)
    if not faces0 or not faces1:
        return None
    d0=np.asarray(action0.effect_direction,float); d1=np.asarray(action1.effect_direction,float)
    df=d0+d1; df=df/max(1e-9,float(np.linalg.norm(df)))
    best=None
    for f0 in faces0:
        for f1 in faces1:
            r=_simple_pair_solution(f0,f1,df)
            if r is None: continue
            if best is None or r['cost'] < best['cost']:
                best=r
    if best is None:
        return None
    return dict(q0=best['q0'], q1=best['q1'], support0=best['support0'],
                support1=best['support1'], sep=best['sep'], reason='simple_support')

def select_simple_point(obs, action):
    f = _best_action_face(obs, action)
    if f is None:
        return None
    q, qt = _face_center_q(f)
    return dict(q=(float(q[0]),float(q[1])), support=(float(f[4]),float(f[5])),
                n=(float(f[1][0]),float(f[1][1])), t=(float(f[2][0]),float(f[2][1])))



def _face_for_q(obs, q):
    q = np.asarray(q, float)
    for f in visible_faces(obs):
        n = np.asarray(f[1], float); t = np.asarray(f[2], float)
        if abs(float(q @ n) - float(f[3])) > .05:
            continue
        qt = float(q @ t)
        if float(f[4])-.05 <= qt <= float(f[5])+.05:
            return f
    return None




def _nearest_visible_q(obs, current_q, other_obs=None, other_q=None):
    faces = visible_faces(obs)
    if not faces:
        return None
    q0 = np.asarray(current_q, float)
    other_faces = visible_faces(other_obs) if other_obs is not None else []
    best = None
    for f in faces:
        n = np.asarray(f[1], float); t = np.asarray(f[2], float)
        lo, hi = float(f[4])+.04, float(f[5])-.04
        if lo >= hi:
            continue
        candidates = [.5*(lo+hi)]
        # If the other robot is on the same line/face, use 1/3 and 2/3.
        if other_q is not None:
            for g in other_faces:
                if (float(np.asarray(g[1]) @ n) >= .98
                        and abs(float(g[3])-float(f[3])) <= .04):
                    span = hi-lo
                    if span > .02:
                        candidates += [lo+span/3., lo+2.*span/3.]
                    break
        for qt in candidates:
            q = n*float(f[3]) + t*qt
            d = float(np.linalg.norm(q-q0))
            if best is None or d < best[0]:
                best = (d, q, f)
    if best is None:
        return None
    # Hysteresis: never jump to a candidate far from the current q.  A face
    # disappearing is handled by holding q, not by teleporting to a neighbour.
    if best[0] > .10:
        return None
    return tuple(float(x) for x in best[1]), best[2]

def _match_anchor_face(obs, anchor, prev_q=None):
    if anchor is None:
        return None
    n0 = np.asarray(anchor['n'], float)
    candidates = []
    for f in visible_faces(obs):
        n = np.asarray(f[1], float)
        if float(n @ n0) < .98:
            continue
        if anchor.get('support') is not None:
            overlap = max(0.0, min(float(f[5]), float(anchor['support'][1]))
                          - max(float(f[4]), float(anchor['support'][0])))
            if overlap <= .01:
                continue
        candidates.append(f)
    if not candidates:
        return None
    def score(f):
        n = np.asarray(f[1], float); t = np.asarray(f[2], float)
        center = .5*(float(f[4])+float(f[5]))
        q = n*float(f[3]) + t*(center+float(anchor.get('u',0.)))
        overlap = 0.0
        if anchor.get('support') is not None:
            overlap = max(0.0, min(float(f[5]), float(anchor['support'][1]))
                          - max(float(f[4]), float(anchor['support'][0])))
        if prev_q is not None:
            return (float(np.linalg.norm(q-np.asarray(prev_q,float)))
                    + .05*abs(float(f[3])-float(anchor['rho']))
                    - .5*overlap)
        return abs(float(f[3])-float(anchor['rho'])) - .5*overlap
    return min(candidates, key=score)

def _face_anchor(f, qt):
    lo, hi = float(f[4]), float(f[5]); center = .5*(lo+hi)
    n = np.asarray(f[1], float); t = np.asarray(f[2], float)
    return dict(n=(float(n[0]), float(n[1])), t=(float(t[0]), float(t[1])),
                rho=float(f[3]), u=float(qt-center), support=(lo, hi), span=float(hi-lo))


def track_contact_q(obs, q, anchor, velocity=None, dt=0.0):
    """Associate the committed FINITE segment, then correct its material q.

    Visible-support midpoints are acquisition candidates, not measurements of
    a material point. Preserve tangent position; only correct line-normal error.
    Missing/adjacent/disjoint supports cannot silently trigger reselection.
    All inputs are observed geometry or the existing contact-velocity estimate.
    """
    previous = np.asarray(q, float)
    predicted = previous.copy()
    if velocity is not None:
        v = np.asarray(velocity, float)
        if v.shape == (2,) and np.all(np.isfinite(v)):
            predicted += v * float(dt)
    debug = dict(previous_q=previous.tolist(), predicted_q=predicted.tolist(),
                 source='support_unobserved_hold', matched=False, matches=[])
    matches = []
    for f in visible_faces(obs):
        n, t = np.asarray(f[1], float), np.asarray(f[2], float)
        normal_match = float(n @ np.asarray(anchor['n'], float)) if anchor else 1.0
        normal_error = abs(float(predicted @ n) - float(f[3]))
        qt = float(predicted @ t)
        outside = max(float(f[4])-qt, qt-float(f[5]), 0.0)
        accepted = normal_match >= .98 and normal_error <= .05 and outside <= .05
        debug['matches'].append(dict(n=n.tolist(), rho=float(f[3]),
                                     support=[float(f[4]), float(f[5])],
                                     normal_match=normal_match, normal_error=normal_error,
                                     outside_support=outside, accepted=accepted))
        if accepted:
            matches.append((normal_error + outside + .05*(1-normal_match), f))
    if not matches:
        debug['final_q'] = predicted.tolist()
        return tuple(predicted), None, anchor, debug
    f = min(matches, key=lambda item: item[0])[1]
    n, t = np.asarray(f[1], float), np.asarray(f[2], float)
    qt = float(predicted @ t)
    corrected = n * float(f[3]) + t * qt
    debug.update(source='same_segment_normal_correct', matched=True,
                 final_q=corrected.tolist(), normal=n.tolist(),
                 support=[float(f[4]), float(f[5])])
    return tuple(float(x) for x in corrected), f, _face_anchor(f, qt), debug


def _map_anchor(obs, anchor, world_q):
    """Map a previous support-relative q to the current visible face; world fallback."""
    if anchor is None:
        return world_q
    for f in visible_faces(obs):
        n = np.asarray(f[1], float); t = np.asarray(f[2], float)
        if float(n @ np.asarray(anchor['n'], float)) < .98 or abs(float(f[3])-float(anchor['rho'])) > .05:
            continue
        span = float(f[5]-f[4])
        if abs(span-float(anchor.get('span', span))) > .20:
            return world_q
        center = .5*(float(f[4])+float(f[5]))
        q = n*float(f[3]) + t*(center + float(anchor.get('u', 0.)))
        return (float(q[0]), float(q[1]))
    return world_q

def select_contact_pair(obs0, action0, obs1, action1, *, desired_force=(0., -1.),
                        desired_torque=0., margin=.04, separation=None, steps=(.20, .10, .05),
                        prev_q0=None, prev_q1=None, local_radius=None,
                        o_ref=None):
    """Deterministic visible-face pair selection.

    Approach: all visible face pairs are considered (no random starts).  Once
    pushing, local_radius restricts each robot to the face/neighbourhood of its
    previous q, so the selector cannot jump across a face/endpoint.
    """
    faces0 = visible_faces(obs0)
    faces1 = visible_faces(obs1)
    if local_radius is not None:
        if prev_q0 is not None:
            faces0 = [f for f in faces0 if abs(float(np.asarray(prev_q0, float) @ np.asarray(f[1], float))-float(f[3])) <= .05]
        if prev_q1 is not None:
            faces1 = [f for f in faces1 if abs(float(np.asarray(prev_q1, float) @ np.asarray(f[1], float))-float(f[3])) <= .05]
        if not faces0 or not faces1:
            return None
    if not faces0 or not faces1:
        return None
    sep = float(2*spec.ROBOT_RADIUS + .06) if separation is None else float(separation)
    best = None
    for f0 in faces0:
        for f1 in faces1:
            r = _optimize_face_pair(f0, f1, obs0, obs1, desired_force, desired_torque, sep,
                                    prev_q0, prev_q1, local_radius, steps, o_ref)
            if best is None or r['cost'] < best['cost']:
                best = r
    return dict(q0=(float(best['q0'][0]), float(best['q0'][1])),
                q1=(float(best['q1'][0]), float(best['q1'][1])),
                cost=float(best['cost']), separation=float(best['separation']),
                same_face=bool(best['same_face']),
                o_ref=(float(best['o_ref'][0]), float(best['o_ref'][1])),
                n0=(float(best['face0'][1][0]), float(best['face0'][1][1])),
                n1=(float(best['face1'][1][0]), float(best['face1'][1][1])),
                selector_debug={k: best[k] for k in ('q0t','q1t','M','F','support0','support1','o_ref')},
                anchor0=_face_anchor(best['face0'], best['q0t']),
                anchor1=_face_anchor(best['face1'], best['q1t']),
                face0_id=id(best['face0']), face1_id=id(best['face1']))

def allocate_pair_targets(obs0, action0, obs1, action1, *, margin=.06):
    """Allocate two tangential contact targets on the same observed face.

    Returns None when the faces are not confidently the same.  If the
    observed union is too short for two separated contacts, the leader gets
    the support centre and the follower is told to wait; no missing endpoint
    is invented.
    """
    f0, f1 = face_support(obs0, action0), face_support(obs1, action1)
    if f0 is None or f1 is None:
        return None
    _, n0, _, rho0, lo0, hi0, p0 = f0
    _, n1, _, rho1, lo1, hi1, p1 = f1
    if float(n0 @ n1) < math.cos(math.radians(15)) or abs(rho0-rho1) > .04:
        return None
    n = n0 + n1
    n = n / max(1e-12, np.linalg.norm(n))
    t = np.asarray(t90(n), float)
    proj = np.concatenate([p0 @ t, p1 @ t])
    lo, hi = float(proj.min()), float(proj.max())
    rho = .5*(rho0+rho1)
    sep = 2*spec.ROBOT_RADIUS + margin
    inset = .02
    center = .5*(lo+hi)
    if hi-lo >= sep + 2*inset:
        q0t, q1t = center-sep/2, center+sep/2
        s0 = float(np.dot([obs0.own.x, obs0.own.y], t))
        s1 = float(np.dot([obs1.own.x, obs1.own.y], t))
        if s0 <= s1:
            targets = (q0t, q1t)
        else:
            targets = (q1t, q0t)
        qs = []
        for q in targets:
            q = max(lo+inset, min(hi-inset, q))
            p = n*rho + t*q
            qs.append((float(p[0]), float(p[1])))
        return dict(q0=qs[0], q1=qs[1], n=(float(n[0]),float(n[1])), t=(float(t[0]),float(t[1])),
                    support=(lo,hi), sep=sep, leader=None, reason='separated')
    # insufficient union: one leader at centre, follower waits
    s0 = float(np.dot([obs0.own.x, obs0.own.y], t))
    s1 = float(np.dot([obs1.own.x, obs1.own.y], t))
    leader = 0 if s0 <= s1 else 1
    p = n*rho + t*center
    return dict(q0=(float(p[0]),float(p[1])) if leader==0 else None,
                q1=(float(p[0]),float(p[1])) if leader==1 else None,
                n=(float(n[0]),float(n[1])), t=(float(t[0]),float(t[1])),
                support=(lo,hi), sep=sep, leader=leader, reason='insufficient_union')



@dataclass
class OptionBudget:
    budget_m: float | None = None
    time_budget_s: float | None = None
    progress_kind: str = 'object_normal'
    progress_m: float = 0.0
    start_time: float | None = None
    last_own: tuple | None = None
    contact_started: bool = False
    reached: bool = False
    exit_reason: str | None = None

    def configure(self, progress_m, time_s, kind, now):
        self.budget_m = None if progress_m is None else float(progress_m)
        self.time_budget_s = None if time_s is None else float(time_s)
        self.progress_kind = str(kind)
        self.progress_m = 0.0
        self.start_time = float(now)
        self.last_own = None
        self.contact_started = False
        self.reached = False
        self.exit_reason = None

    def update(self, obs, now, dt):
        own = (float(obs.own.x), float(obs.own.y))
        if self.last_own is None:
            self.last_own = own
        patch = getattr(obs, 'patch', None)
        gap = patch_gap(patch) if patch is not None else None
        n = None if patch is None else np.asarray(patch.normal_world, float)
        if self.budget_m is not None:
            if self.progress_kind == 'robot_path':
                self.progress_m += float(np.hypot(own[0]-self.last_own[0], own[1]-self.last_own[1]))
            elif gap is not None and gap <= .02:
                if not self.contact_started:
                    self.contact_started = True
                    self.last_own = own
                elif n is not None:
                    dx = own[0]-self.last_own[0]; dy = own[1]-self.last_own[1]
                    self.progress_m += max(0.0, dx*float(n[0]) + dy*float(n[1]))
            if self.progress_m >= self.budget_m - 1e-9:
                self.reached = True; self.exit_reason = 'progress_budget'
        if self.time_budget_s is not None and self.start_time is not None:
            if now - self.start_time >= self.time_budget_s - 1e-9:
                self.reached = True; self.exit_reason = self.exit_reason or 'time_budget'
        self.last_own = own

    def info(self):
        return dict(progress_m=float(self.progress_m), budget_m=self.budget_m,
                    time_budget_s=self.time_budget_s, budget_reached=bool(self.reached),
                    progress_kind=self.progress_kind, exit_reason=self.exit_reason)



class SharedMotionEstimator:
    """Legal per-env shared motion estimator; survives option switches.

    Each observer keeps its own frozen line history.  A rate is produced only
    by comparing the current observation of that observer with its own previous
    tick; observations of the same tick from different robots never become a
    time difference.  Same-tick multi-view evidence is deduplicated (highest
    quality line wins) rather than multiplied as independent rates.
    """
    def __init__(self):
        self.ekf = RigidMotionEKF()
        self.histories = {0: [], 1: []}
        self.lines = []
        self.last_now = None
        self.direct_info = None
        self.direct_prev = None

    def _match(self, prev, n, rho, phi):
        best = None
        for line in prev:
            da = abs(math.atan2(math.sin(phi-line.phi), math.cos(phi-line.phi)))
            dr = abs(rho-line.rho)
            if da <= math.radians(20) and dr <= .10:
                best = line; break
        return best

    def update(self, observations, now, dt):
        if self.last_now is not None and abs(now - self.last_now) <= 1e-12:
            return self                          # idempotent for a repeated tick
        self.last_now = now
        self.ekf.predict(dt)
        if isinstance(observations, dict):
            items = list(observations.items())
        else:
            items = list(enumerate(observations))
        rates = []
        for side, obs in items:
            prev = self.histories.get(side, [])
            current = []
            for patch in [p for p in obs.patches if p.valid and len(p.points_world) >= 3]:
                n = np.asarray(patch.normal_world, float)
                n = n / max(1e-12, np.linalg.norm(n))
                p0 = np.asarray(patch.points_world[0], float)
                rho = float(n @ p0)
                phi = math.atan2(n[1], n[0])
                line = Line((float(n[0]), float(n[1])), rho, phi, now)
                old = self._match(prev, n, rho, phi)
                if old is not None and now - old.stamp > 1e-9:
                    dtime = now - old.stamp
                    drho = (rho - old.rho) / dtime
                    dphi = math.atan2(math.sin(phi-old.phi), math.cos(phi-old.phi)) / dtime
                    rates.append((float(patch.confidence), line, drho, dphi))
                current.append(line)
            self.histories[side] = current
            # prior-free: no true contact_point feedback; use fitted-line rates only

        # Deduplicate the same physical line seen by both robots at this tick.
        accepted = []
        Hrows = []
        for quality, line, drho, dphi in sorted(rates, key=lambda z: -z[0]):
            duplicate = False
            for _, other, _, _ in accepted:
                da = abs(math.atan2(math.sin(line.phi-other.phi), math.cos(line.phi-other.phi)))
                if da <= math.radians(5) and abs(line.rho-other.rho) <= .02:
                    duplicate = True; break
            if duplicate:
                continue
            accepted.append((quality, line, drho, dphi))
            n = np.asarray(line.n, float)
            Hrows.append([float(n[0]), float(n[1]), 0.])
            Hrows.append([0., 0., 1.])
            self.ekf.line(n, line.rho, line.phi, drho, dphi, dt)
        if Hrows:
            self.ekf.apply_nullspace_prior(Hrows)
        self.lines = [line for side in (0, 1) for line in self.histories.get(side, [])]
        self.debug = dict(
            accepted=[dict(quality=float(q), n=[float(v) for v in l.n], rho=float(l.rho),
                           phi=float(l.phi), drho=float(dr), dphi=float(dp))
                      for q, l, dr, dp in accepted],
            ekf=[float(v) for v in self.ekf.x], time=float(now))
        self._try_direct(observations, accepted)
        self.debug['direct'] = self.direct_info
        return self

    def _try_direct(self, observations, accepted=None):
        entries = []
        items = observations.items() if isinstance(observations, dict) else enumerate(observations)
        for side, obs in items:
            patch = obs.patch
            if patch is None or patch_gap(patch) > .02 or not patch.valid:
                continue
            n = np.asarray(patch.normal_world, float)
            n = n / max(1e-12, float(np.linalg.norm(n)))
            q = np.asarray((obs.own.x, obs.own.y), float) + spec.ROBOT_RADIUS*n
            v_robot = np.asarray((obs.own.vx, obs.own.vy), float)
            entries.append(dict(side=side, n=n, q=q, z=float(n @ v_robot),
                                w=max(.05, float(patch.confidence))))
        if len(entries) < 1:
            self.direct_info = dict(accepted=False, reason='contact points<1')
            return None
        q_ref = entries[0]['q']
        H, z, w = [], [], []
        for e in entries:
            qr = e['q'] - q_ref
            n = e['n']
            H.append([float(n[0]), float(n[1]), float(n @ np.array([-qr[1], qr[0]]))])
            z.append(float(e['z'])); w.append(float(e['w']))
        # line orientation supplies a direct omega row (median dphi)
        acc = [v for v in (accepted or []) if math.isfinite(float(v[3]))]
        if acc:
            H.append([0., 0., 1.]); z.append(float(np.median([float(v[3]) for v in acc]))); w.append(.15)
        H = np.asarray(H, float); z = np.asarray(z, float); w = np.asarray(w, float)
        sw = np.sqrt(w); Hw = H*sw[:,None]; zw = z*sw
        try:
            _, sv, vt = np.linalg.svd(Hw, full_matrices=False)
        except np.linalg.LinAlgError:
            self.direct_info = dict(accepted=False, reason='svd')
            return None
        rank = int(np.sum(sv > max(Hw.shape)*sv[0]*1e-6)) if len(sv) else 0
        cond = float(sv[0]/max(sv[rank-1], 1e-12)) if rank else float('inf')
        if rank < 2 or cond > 20.:
            self.direct_info = dict(accepted=False, reason='rank/cond', rank=rank, cond=cond)
            return None
        lam = .02
        x = np.linalg.solve(Hw.T @ Hw + lam*np.eye(3), Hw.T @ zw)
        res = float(np.linalg.norm(Hw @ x - zw) / max(1., np.linalg.norm(zw)))
        if res > .15:
            self.direct_info = dict(accepted=False, reason='residual', rank=rank, cond=cond, res=res)
            return None
        v_ref = x[:2]; om = float(x[2])
        # convert q_ref twist to the fixed world-origin twist used by the EKF/UI
        v_origin = v_ref - om * np.array([-q_ref[1], q_ref[0]])
        if getattr(self, 'direct_prev', None) is None:
            sm = np.array([v_origin[0], v_origin[1], om], float)
        else:
            sm = .30*np.array([v_origin[0], v_origin[1], om], float) + .70*np.asarray(self.direct_prev, float)
        self.direct_prev = sm.copy()
        self.ekf.x = sm
        self.ekf.P = np.diag([.05**2, .05**2, .10**2])
        contact_v = []
        for e in entries:
            qr = e['q'] - q_ref
            v = v_ref + om * np.array([-qr[1], qr[0]])
            contact_v.append(dict(side=e['side'], q=[float(e['q'][0]), float(e['q'][1])],
                                  n=[float(e['n'][0]), float(e['n'][1])],
                                  v=[float(v[0]), float(v[1])]))
        self.contact_solution = dict(q_ref=[float(q_ref[0]), float(q_ref[1])],
                                     v_ref=[float(v_ref[0]), float(v_ref[1])], omega=om,
                                     contacts=contact_v, res=res, rank=rank)
        self.direct_info = dict(accepted=True, rank=rank, cond=cond, res=res,
                                x=[float(v) for v in sm], rows=len(entries),
                                contact_solution=self.contact_solution)
        return sm


    def velocity_at_contact(self, side, q):
        sol = getattr(self, 'contact_solution', None)
        if not sol:
            return None
        e = next((c for c in sol.get('contacts', []) if int(c.get('side', -1)) == int(side)), None)
        if e is None:
            return None
        qc = np.asarray(e['q'], float); v = np.asarray(e['v'], float); om = float(sol['omega'])
        dq = np.asarray(q, float) - qc
        vq = v + om * np.array([-dq[1], dq[0]])
        return (float(vq[0]), float(vq[1]))


class UnknownShapePush(ContinuousPush):
    """PUSH translate with no shape/endpoint/centre target chain."""
    __test__ = False

    def __init__(self, action, force_limit=spec.MAX_FORCE, config=None, estimator=None):
        super().__init__(action, force_limit, config if config is not None else C.CONFIG)
        self.estimator = estimator or SharedMotionEstimator()
        self.face = None
        self.speed_command = float(action.speed)
        self.progress = 0.0
        self.effective_contact = 0.0
        self.contact_seen = False
        self.last_contact_time = -1.0
        self.contact_hold_s = .20
        self.debug_ref = None
        self.debug_qprime = None
        self.debug_offset = None
        self.budget_m = None
        self.time_budget_s = None
        self.progress_m = 0.0
        self.budget_reached = False
        self.budget_start_time = None
        self.budget_last_own = None
        self.budget_contact_started = False
        self.ready = False
        self.contact_ready = False
        self.release = False
        self.joint_active = False
        self.recovery_requested = False
        self.recovery_ready = False
        self.pause_reason = None
        self.gap = None
        self.paired = False
        self.align_time = 0.0
        self.align_settle = 0.0
        self.wait_until = -1.0
        self.rotation_time = 0.0
        self.retreat_until = -1.0
        self.contact_phase = 'approach'
        self.retry_count = 0
        self.shift_sign = 0.0
        self.shift_until = -1.0
        self.last_n = None
        self.last_t = None
        self.wait_until = -1.0
        self.nav_qid = None
        self.nav_stable = 0
        self.allocated_q = None
        self.allocated_id = None
        self.allocated_wait = False
        self.allocated_sep = None
        self.allocated_reason = None
        self.allocated_debug = None
        self.allocated_support = None
        self.selector_committed = False
        self.committed_q = None
        self.committed_support = None
        self.committed_anchor = None
        self.raw_pre = None
        self.peer_contact = False
        self.pair_push_ready = False
        self.base_normal_angle = None
        self.contact_offset = 0.0
        self.ff_t = 0.0
        self.u_adjust = 0.0
        self.contact_err_t = None
        self.candidate_err_t = None
        self.candidate = None
        self.candidate_id = 0
        self.candidate_last_span = 0.0
        self.candidate_pending = 0

    def _observe_face(self, obs):
        # compatibility with Slot's sensor-missing check; no prior face estimate
        self.face = obs.patch
        return obs.patch

    def _line_track(self, patch, t):
        n = np.asarray(patch.normal_world, float)
        n = n / max(1e-12, np.linalg.norm(n))
        p = np.asarray(patch.points_world[0], float)
        rho = float(n @ p)
        phi = math.atan2(n[1], n[0])
        best = None
        for line in self.lines:
            da = abs(math.atan2(math.sin(phi-line.phi), math.cos(phi-line.phi)))
            dr = abs(rho-line.rho)
            if da <= math.radians(20) and dr <= .10:
                best = line; break
        if best is None:
            line = Line((float(n[0]), float(n[1])), rho, phi, t)
            self.lines.append(line); self.lines = self.lines[-8:]
            return line, None
        dt = max(1e-6, t - best.stamp)
        out = dict(drho=(rho-best.rho)/dt, dphi=math.atan2(math.sin(phi-best.phi),
                                                           math.cos(phi-best.phi))/dt)
        best.n = (float(n[0]), float(n[1])); best.rho = rho; best.phi = phi; best.stamp = t
        return best, out

    def step(self, obs: SkillObservation, dt: float):
        force, info = self._step_impl(obs, dt)
        if self.budget_m is not None or self.time_budget_s is not None:
            own = (float(obs.own.x), float(obs.own.y))
            if self.budget_last_own is None:
                self.budget_last_own = own
            if self.budget_m is not None and (bool(info.get('contacting_object')) or (info.get('gap') is not None and float(info['gap']) <= .02)):
                if not self.budget_contact_started:
                    self.budget_contact_started = True
                    self.budget_last_own = own
                n = info.get('face_normal') or (info.get('candidate') or {}).get('n')
                if n is not None:
                    dx = own[0] - self.budget_last_own[0]
                    dy = own[1] - self.budget_last_own[1]
                    # robot displacement projected on the observed support normal
                    self.progress_m += max(0.0, dx*float(n[0]) + dy*float(n[1]))
            if self.budget_m is not None and self.progress_m >= self.budget_m - 1e-9:
                self.budget_reached = True
                self.exit_reason = 'progress_budget'
            if self.time_budget_s is not None and self.budget_start_time is not None:
                if self.skill_time - self.budget_start_time >= self.time_budget_s - 1e-9:
                    self.budget_reached = True
                    self.exit_reason = self.exit_reason or 'time_budget'
            info = {**info, 'progress_m': self.progress_m,
                    'budget_m': self.budget_m, 'budget_reached': self.budget_reached,
                    'time_budget_s': self.time_budget_s}
        self.budget_last_own = own if 'own' in locals() else self.budget_last_own
        return force, info

    def _step_impl(self, obs: SkillObservation, dt: float):
        self.skill_time += dt
        self._note_position(obs)
        patches = [p for p in obs.patches if p.valid and len(p.points_world) >= 3]
        # choose a patch aligned with the requested push direction; no size prior
        d = np.asarray(self.action.effect_direction, float)
        scored = []
        for p in patches:
            n = np.asarray(p.normal_world, float)
            n = n / max(1e-12, np.linalg.norm(n))
            align = float(n @ d)
            gap = patch_gap(p)
            if align >= math.cos(math.radians(35)):
                scored.append((gap, -align, p))
        scored.sort(key=lambda x: (x[0], x[1]))
        # No direction-compatible face: do not silently push the nearest
        # (possibly orthogonal/opposite) surface.  Wait for a valid view.
        patch = (getattr(self, 'allocated_face', None)
                 if getattr(self, 'allocation_managed', False)
                 else (scored[0][2] if scored else None))
        # Publish the sole allocator result each tick, including visibility
        # loss. Candidate identity follows the committed segment, not rounded q.
        if self.allocated_q is not None and getattr(self, 'allocated_normal', None) is not None:
            an = tuple(self.allocated_normal); at = t90(an)
            if self.candidate is None or self.allocated_id != getattr(self, '_alloc_last_id', None):
                self.candidate_id += 1
                self._alloc_last_id = self.allocated_id
                self.candidate = ContactCandidate(
                    self.candidate_id, tuple(self.allocated_q), an, at,
                    self.skill_time, self.skill_time, tuple(self.allocated_support), .02)
            self.candidate.q = tuple(self.allocated_q)
            self.candidate.n, self.candidate.t = an, at
            self.candidate.support = tuple(self.allocated_support)
            self.candidate.valid = not self.allocated_wait
            self.candidate.reason = self.allocated_reason
            if self.candidate.valid:
                self.candidate.verified_time = self.skill_time
        raw = np.zeros(2); info = {}; cp = None
        if patch is None:
            self.stage = "find"
            self.ready = self.contact_ready = False
            self.face = None
        else:
            n = np.asarray(patch.normal_world, float)
            n = n / max(1e-12, np.linalg.norm(n))
            t = np.array(t90(n))
            self.last_n, self.last_t = n, t
            cp = None
            # raw contact/own/candidate error must be fresh before any
            # RETREAT/SHIFT early return
            self.contact_ready = bool(patch is not None and patch_gap(patch) <= .02)
            self.contact_seen = self.contact_seen or self.contact_ready
            if self.contact_ready:
                self.effective_contact += dt
            if self.candidate is not None:
                own0 = np.array([obs.own.x, obs.own.y])
                self.candidate_err_t = float(np.dot(np.asarray(self.candidate.q) - own0, t))
            else:
                self.candidate_err_t = None
            gap = patch_gap(patch); self.gap = gap
            if self.paired and self.contact_ready and not self.pair_push_ready:
                hdg = math.atan2(n[1], n[0])
                herr = math.atan2(math.sin(hdg - obs.own.angle), math.cos(hdg - obs.own.angle))
                if abs(herr) > .18:
                    self.contact_phase = 'orient'
                    self.stage = 'orient'
                    raw = np.zeros(2)
                    self.face = patch
                    raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                    raw = np.asarray(raw, float)
                    ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                    force, saturated = self._force_from_reference(ref, obs)
                    return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                self.contact_phase = 'wait_peer_contact'
                self.stage = 'wait_peer_contact'
                raw = np.zeros(2)
                self.face = patch
                raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                raw = np.asarray(raw, float)
                ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                force, saturated = self._force_from_reference(ref, obs)
                return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
            if self.paired and self.allocated_wait:
                self.stage = 'wait_peer_contact'
                raw = np.zeros(2)
                self.face = patch
                raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                raw = np.asarray(raw, float)
                ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                force, saturated = self._force_from_reference(ref, obs)
                return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
            if self.allocated_q is not None:
                # publish the shared q as this controller's candidate
                if self.allocated_id != getattr(self, '_alloc_last_id', None):
                    self.candidate_id += 1
                    self._alloc_last_id = self.allocated_id
                    self.candidate = ContactCandidate(self.candidate_id, tuple(self.allocated_q),
                        (float(n[0]), float(n[1])), (float(t[0]), float(t[1])),
                        self.skill_time, self.skill_time, (tuple(self.allocated_support) if self.allocated_support is not None else (-1.0, 1.0)), (.02 if self.allocated_support is not None else 0.0), True, 0, (self.allocated_reason or 'pair_alloc'))
                own0 = np.array([obs.own.x, obs.own.y])
                self.candidate_err_t = float(np.dot(np.asarray(self.candidate.q) - own0, t))
                if not self.contact_ready:
                    return self._allocated_navigate(obs, dt, patch, n, t, cp)
            if self.contact_phase in ('retreat', 'shift') and self.candidate is None:
                self.contact_phase = 'wait'
                self.shift_sign = 0.0
                raw = np.zeros(2)
                self.stage = 'wait'
                self.face = patch
                raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                raw = np.asarray(raw, float)
                ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                force, saturated = self._force_from_reference(ref, obs)
                return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
            if self.contact_phase in ('retreat', 'shift'):
                if self.contact_phase == 'retreat' and self.skill_time >= self.retreat_until:
                    self.contact_phase = 'shift'
                    self.shift_until = self.skill_time + 1.0   # .08m at .08m/s
                    self.stage = 'shift'
                elif self.contact_phase == 'shift' and self.skill_time >= self.shift_until:
                    self.contact_phase = 'approach'
                    self.align_time = 0.0
                    self.stage = 'approach'
                phase, cmd = persistent_phase_command(
                    self.contact_phase, self.skill_time, self.retreat_until,
                    self.shift_until, self.shift_sign, n, t)
                if cmd is None:
                    raw = np.zeros(2)
                else:
                    raw = np.asarray(cmd, float)
                self.stage = phase if phase != 'approach' else 'approach'
                raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                raw = np.asarray(raw, float)
                ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                force, saturated = self._force_from_reference(ref, obs)
                return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
            # motion estimation is owned by the shared per-env estimator; this
            # controller only consumes the current snapshot.
            contact = bool(self.contact_ready)
            if contact:
                self.last_contact_time = self.skill_time
            elif (self.contact_phase == 'push' and self.last_contact_time >= 0.0
                  and self.skill_time - self.last_contact_time <= self.contact_hold_s):
                # contact sensor chatter while physically pushing must not
                # bounce the stage back to touch/approach
                contact = True
            cp = None
            if contact and self.candidate is not None:
                cp = tuple(float(x) for x in self.candidate.q)
            self.contact_seen = self.contact_seen or contact
            self.contact_ready = contact
            self.ready = bool(contact or gap <= .05)
            self.stage = "push" if contact else ("ready" if self.ready else "approach")
            if not contact:
                # approach the observed line, no endpoint or centre target
                pts = np.asarray(patch.points_world, float)
                own = np.array([obs.own.x, obs.own.y])
                # persistent contact candidate from the current visible
                # support; only an explicit retarget event replaces it.
                proj = pts @ t
                lo, hi = float(proj.min()), float(proj.max())
                q, margin, status = choose_candidate_q(lo, hi)
                if status == 'view_insufficient':
                    self.candidate = None
                    self.candidate_err_t = None
                    self.stage = 'view_insufficient'
                    raw = np.zeros(2)
                    self.face = patch
                    self.progress = self.effective_contact
                    raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                    raw = np.asarray(raw, float)
                    ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                    force, saturated = self._force_from_reference(ref, obs)
                    return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                q_world = n*float(n @ pts[0]) + t*q
                old_q = None if self.candidate is None else np.asarray(self.candidate.q, float)
                if self.candidate is None:
                    self.candidate_id += 1
                    self.candidate = ContactCandidate(self.candidate_id, (float(q_world[0]), float(q_world[1])),
                        (float(n[0]), float(n[1])), (float(t[0]), float(t[1])),
                        self.skill_time, self.skill_time, (lo, hi), margin, True, 0, 'initial')
                    self.candidate_last_span = hi-lo
                    self.candidate_pending = 0
                else:
                    c = self.candidate
                    q_now = float(np.dot(np.asarray(c.q), t))
                    support_ext = hi - lo > self.candidate_last_span + .05
                    line_err = abs(float(np.dot(np.asarray(c.q), n)) - float(n @ pts[0]))
                    if line_err > .02 or q_now < lo - .02 or q_now > hi + .02:
                        self.candidate_id += 1
                        self.candidate = ContactCandidate(self.candidate_id, (float(q_world[0]), float(q_world[1])),
                            (float(n[0]), float(n[1])), (float(t[0]), float(t[1])),
                            self.skill_time, self.skill_time, (lo, hi), margin, True, 0, 'support_invalid')
                        self.candidate_last_span = hi-lo
                        self.candidate_pending = 0
                    elif support_ext:
                        self.candidate_pending += 1
                        if self.candidate_pending >= 3:
                            self.candidate_id += 1
                            self.candidate = ContactCandidate(self.candidate_id, (float(q_world[0]), float(q_world[1])),
                                (float(n[0]), float(n[1])), (float(t[0]), float(t[1])),
                                self.skill_time, self.skill_time, (lo, hi), margin, True, 0, 'support_extension')
                            self.candidate_last_span = hi-lo
                            self.candidate_pending = 0
                    else:
                        self.candidate_pending = 0
                        c.verified_time = self.skill_time
                        c.support = (lo, hi)
                if old_q is not None and self.candidate is not None:
                    nq = np.asarray(self.candidate.q, float)
                    filt = .30*nq + .70*old_q
                    step = filt - old_q
                    sn = float(np.linalg.norm(step))
                    if sn > .02:
                        filt = old_q + step/sn*.02
                    self.candidate.q = (float(filt[0]), float(filt[1]))
                e_t = float(np.dot(np.asarray(self.candidate.q) - own, t))
                self.candidate_err_t = e_t
                if self.candidate is not None:
                    q = np.asarray(self.candidate.q, float)
                    outward = -n
                    e_t = float(np.dot(q - own, t))
                    if abs(e_t) > .05:
                        target = q + outward*(spec.ROBOT_RADIUS + .15)
                        self.contact_phase = 'standoff'
                        self.stage = 'standoff'
                    else:
                        target = q + outward*spec.ROBOT_RADIUS
                        self.contact_phase = 'final'
                        self.stage = 'final'
                    self.debug_ref = tuple(float(x) for x in target)
                    raw = 6.0*(target - own)
                    rn = float(np.linalg.norm(raw))
                    if rn > .30:
                        raw = raw/rn*.30
                    self.face = patch
                    self.progress = self.effective_contact
                    raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                    raw = np.asarray(raw, float)
                    ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                    force, saturated = self._force_from_reference(ref, obs)
                    return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                if not self.paired and self.candidate is not None and self.contact_phase not in ('wait',):
                    # Approach is navigation in a safe distance band, with a
                    # dynamic contact candidate q.  Do not touch merely because
                    # the robot reached a q-relative standoff; only the final
                    # approach to the target robot pose q + outward*R may touch.
                    q = np.asarray(self.candidate.q, float)
                    outward = -n
                    goal = q + outward*spec.ROBOT_RADIUS       # robot centre at contact
                    safe = q + outward*(spec.ROBOT_RADIUS + .20)  # navigation band
                    dgoal = float(np.linalg.norm(goal - own))
                    dsafe = float(np.linalg.norm(safe - own))
                    if self.nav_qid != self.candidate.id:
                        self.nav_qid = self.candidate.id
                        self.nav_stable = 0
                    else:
                        self.nav_stable += 1
                    # q must be stable for 3 ticks and we must be near the safe
                    # band before leaving it for the final normal approach.
                    inside = dgoal <= spec.ROBOT_RADIUS + .20 + .02
                    final = (abs(e_t) <= .05 and (inside or (self.nav_stable >= 3 and (dsafe <= .10 or dgoal <= .24))))
                    if final and dgoal > .035:
                        self.contact_phase = 'final'
                        self.stage = 'final'
                        raw = np.asarray(unit(goal - own), float)*min(.30, max(.10, 2.5*dgoal))
                    elif final:
                        self.contact_phase = 'touch'
                        self.stage = 'touch'
                        raw = n*.10 + t*float(np.clip(.4*e_t, -.02, .02))
                    else:
                        self.contact_phase = 'standoff'
                        self.stage = 'standoff'
                        raw = np.asarray(unit(safe - own), float)*min(.45, max(.10, 2.8*dsafe))
                    self.face = patch
                    self.progress = self.effective_contact
                    raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                    raw = np.asarray(raw, float)
                    ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                    force, saturated = self._force_from_reference(ref, obs)
                    return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                if not self.paired and self.contact_phase == 'wait' and self.skill_time < self.wait_until:
                    raw = np.zeros(2)
                    self.stage = 'wait'
                elif not self.paired and self.contact_phase == 'wait':
                    self.contact_phase = 'approach'
                    self.align_time = 0.0
                    self.align_settle = 0.0
                elif not self.paired and gap < .08:
                    own_vt = float(np.dot(np.array([obs.own.vx, obs.own.vy]), t))
                    surf_vt = float(np.dot(np.asarray(self.estimator.ekf.velocity_at(own)), t))
                    rel_vt = abs(own_vt - surf_vt)
                    if abs(e_t) <= .015 and rel_vt <= .03:
                        self.align_settle += dt
                        if self.align_settle >= .15:
                            self.contact_phase = 'touch'
                            self.stage = 'touch'
                            raw = n*.06 + t*float(np.clip(.5*e_t, -.02, .02))
                        else:
                            self.contact_phase = 'align'
                            self.stage = 'align'
                            raw = t*float(np.clip(2.5*e_t, -.20, .20))
                    else:
                        self.align_settle = 0.0
                        self.align_time += dt
                        if self.align_time > .60:
                            self.contact_phase = 'wait'
                            self.wait_until = self.skill_time + .30
                            raw = np.zeros(2)
                            self.stage = 'wait'
                        else:
                            self.contact_phase = 'align'
                            self.stage = 'align'
                            raw = t*float(np.clip(2.5*e_t, -.20, .20))
                elif not self.paired and gap <= .06:

                    self.contact_phase = 'touch'
                    self.stage = 'touch'
                    raw = n*.04
                else:
                    radial = min(self.config.approach_normal_speed,
                                 max(.06, self.config.gap_gain*max(0., gap-.03)))
                    raw = n*radial + t*float(np.clip(2.5*e_t, -.25, .25))
            else:
                if self.candidate is not None:
                    own0 = np.array([obs.own.x, obs.own.y])
                    outward = -n
                    qq = np.asarray(self.candidate.q, float)
                    na = math.atan2(n[1], n[0])
                    if self.base_normal_angle is None:
                        self.base_normal_angle = na
                    yaw_err = math.atan2(math.sin(na - self.base_normal_angle),
                                         math.cos(na - self.base_normal_angle))
                    if USE_QPRIME:
                        self.contact_offset = float(np.clip(self.contact_offset + .02*yaw_err, -.06, .06))
                        qp = qq + np.asarray(t, float)*self.contact_offset
                    else:
                        self.contact_offset = 0.0
                        qp = qq
                    self.debug_qprime = tuple(float(x) for x in qp)
                    self.debug_offset = float(self.contact_offset)
                    dist_q = float(np.linalg.norm(qp - own0))
                    surf_gap = max(0.0, dist_q - spec.ROBOT_RADIUS)
                    # Preserve the existing contact-position feedback. With
                    # gain 6/s, a v/6 inward reference offset requests speed v
                    # when centered on the surface. The old fixed .05 offset
                    # plus .30 cap made every action behave like the .30 level.
                    penetration_reference = float(self.speed_command)/6.0
                    rr = qp + outward*((spec.ROBOT_RADIUS - penetration_reference)
                                       if surf_gap <= .05 else spec.ROBOT_RADIUS)
                    ee = rr - own0
                    self.debug_ref = tuple(float(x) for x in rr)
                    self.candidate_err_t = float(np.dot(qp - own0, t))
                    hdg = na
                    herr = math.atan2(math.sin(hdg - obs.own.angle), math.cos(hdg - obs.own.angle))
                    if abs(herr) > .18:
                        self.contact_phase = 'orient'
                        self.stage = 'orient'
                        raw = np.zeros(2)
                    else:
                        raw = 6.0*ee
                        rn = float(np.linalg.norm(raw))
                        if rn > self.speed_command:
                            raw = raw/rn*self.speed_command
                        self.contact_phase = 'push'
                        self.stage = 'push'
                    self.face = patch
                    self.progress = self.effective_contact
                    raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=False)
                    raw = np.asarray(raw, float)
                    ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                    force, saturated = self._force_from_reference(ref, obs)
                    return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                if self.contact_phase not in ('push',):
                    hdg = math.atan2(n[1], n[0])
                    herr = math.atan2(math.sin(hdg - obs.own.angle), math.cos(hdg - obs.own.angle))
                    if abs(herr) > .18:
                        self.contact_phase = 'orient'
                        self.stage = 'orient'
                        raw = np.zeros(2)
                        self.face = patch
                        self.progress = self.effective_contact
                        raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                        raw = np.asarray(raw, float)
                        ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                        force, saturated = self._force_from_reference(ref, obs)
                        return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                # bounded response: a sustained estimated rotation is a legal
                # signal to stop full push and retry the contact later.
                if not self.paired:
                    omega = float(self.estimator.ekf.x[2])
                    if abs(omega) > .12:
                        self.rotation_time += dt
                    else:
                        self.rotation_time = 0.0
                    if (self.rotation_time > .15 and self.retry_count < 3
                            and self.contact_phase not in ('retreat', 'shift')):
                        self.retry_count += 1
                        et = self.candidate_err_t
                        if et is None:
                            self.contact_phase = 'approach'
                            self.rotation_time = 0.0
                        else:
                            self.shift_sign = shift_sign_for_error(et)
                        self.contact_phase = 'retreat'
                        self.retreat_until = self.skill_time + .30
                        self.rotation_time = 0.0
                    if self.contact_phase == 'retreat':
                        raw = -n*.08
                        if self.skill_time >= self.retreat_until:
                            self.contact_phase = 'shift'
                            self.shift_until = self.skill_time + 1.0
                        self.stage = 'retreat'
                        self.face = patch
                        self.progress = self.effective_contact
                        raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                        raw = np.asarray(raw, float)
                        ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                        force, saturated = self._force_from_reference(ref, obs)
                        return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                if self.contact_phase == 'shift':
                    raw = t*self.shift_sign*.08
                    self.stage = 'shift'
                    if self.skill_time >= self.shift_until:
                        self.contact_phase = 'approach'
                        self.align_time = 0.0
                    raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
                    raw = np.asarray(raw, float)
                    ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
                    force, saturated = self._force_from_reference(ref, obs)
                    return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
                # follow surface twist at the contact point; bounded lateral trim only
                v_s = self.estimator.ekf.velocity_at(cp) if cp is not None else (0., 0.)
                # Follow the full estimated surface velocity, bounded by the
                # existing physical velocity/force limits rather than a tiny
                # arbitrary FF cap.  Active relative trim is a separate term and
                # is zero until a calibrated contact strategy exists.
                ff_on = os.environ.get('D009_SURFACE_FF', '0') != '0'
                vt = float(np.clip(float(t @ np.asarray(v_s)), -.16, .16)) if ff_on else 0.0
                # bounded lateral trim toward the persistent candidate q;
                # this is the contact-point position feedback, separate from
                # the surface-velocity feedforward.
                if self.candidate is not None and cp is not None:
                    self.contact_err_t = float(np.dot(np.asarray(self.candidate.q) - np.asarray(cp), t))
                else:
                    self.contact_err_t = None
                self.ff_t = vt
                self.u_adjust = 0.0 if self.contact_err_t is None else float(np.clip(1.5*self.contact_err_t, -.08, .08))
                self.contact_phase = 'push'
                self.stage = 'push'
                vn = float(self.speed_command)
                raw = n*vn + t*(vt + self.u_adjust)
            self.face = patch
            self.progress = self.effective_contact
        raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
        raw = np.asarray(raw, float)
        ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
        force, saturated = self._force_from_reference(ref, obs)
        info = self._info_unknown(obs, force, saturated, ref, raw, patch, cp)
        if self.done:
            force = (0., 0.)
        return force, info

    def stop(self, reason):
        if not self.done:
            self.done = True
            self.exit_reason = reason
            self.stage = "done"

    def _allocated_navigate(self, obs, dt, patch, n, t, cp):
        q = np.asarray(self.allocated_q, float)
        own = np.array([obs.own.x, obs.own.y])
        outward = -n
        e_t = float(np.dot(q - own, t))
        if abs(e_t) > .05:
            target = q + outward*(spec.ROBOT_RADIUS + .15)
            self.contact_phase = 'standoff'
            self.stage = 'standoff'
        else:
            target = q + outward*spec.ROBOT_RADIUS
            self.contact_phase = 'final'
            self.stage = 'final'
        self.debug_ref = tuple(float(x) for x in target)
        self.candidate_err_t = e_t
        raw = 6.0*(target - own)
        rn = float(np.linalg.norm(raw))
        if rn > .30:
            raw = raw/rn*.30
        self.face = patch
        self.progress = self.effective_contact
        raw = self._obstacle_avoidance(obs, (float(raw[0]), float(raw[1])), include_teammate=True)
        raw = np.asarray(raw, float)
        ref = self.reference.update((float(raw[0]), float(raw[1])), dt)
        force, saturated = self._force_from_reference(ref, obs)
        return force, self._info_unknown(obs, force, saturated, ref, raw, patch, cp)

    def _desired_heading(self, n, t, raw):
        if n is None:
            return None
        # Contact acquisition (standoff/final/touch) is an independent
        # omni-directional lateral adjustment: face the support normal instead
        # of binding heading to the tangential travel direction.  Heading and
        # linear velocity are then regulated by separate feedback loops.
        if self.contact_ready or self.contact_phase in ('standoff', 'final', 'touch', 'push', 'orient'):
            return n
        return None

    def _info_unknown(self, obs, force, saturated, ref, raw, patch, cp):
        n = tuple(float(x) for x in patch.normal_world) if patch is not None else None
        t = t90(n) if n else None
        st = self.stage
        info = {
            "skill": self.action.label, "kind": self.kind, "stage": st,
            "exit_reason": self.exit_reason, "done": self.done,
            "desired_velocity": [float(x) for x in raw], "reference_velocity": list(ref),
            "force": list(force), "force_norm": math.hypot(*force),
            "force_saturated": saturated, "gap": self.gap, "gap_target": 0.,
            "normal_world": n, "tangent_world": t, "normal_valid": n is not None,
            "patch_valid": patch is not None, "contacting_object": bool(self.contact_ready),
            "contact_ready": bool(self.contact_ready), "ready": bool(self.ready),
            "joint_active": bool(self.joint_active), "release": bool(self.release),
            "effective_contact": self.effective_contact, "progress": self.progress,
            "target": None, "contact_target": cp,
            "candidate_err_t": self.candidate_err_t,
            "ff_t": self.ff_t, "u_adjust": self.u_adjust, "contact_err_t": self.contact_err_t,
            "selector_debug": self.allocated_debug, "desired_pre": self.raw_pre, "avoidance_active": self.avoidance_active, "avoidance_direction": self.avoidance_direction,
            "gap": self.gap, "ref_center": self.debug_ref, "q_prime": self.debug_qprime, "contact_offset": self.debug_offset, "contact_ready": bool(self.contact_ready),
            "peer_contact": self.peer_contact, "pair_push_ready": self.pair_push_ready,
            "candidate": (None if self.candidate is None else {
                "id": self.candidate.id, "q": list(self.candidate.q),
                "n": list(self.candidate.n), "t": list(self.candidate.t),
                "support": list(self.candidate.support), "margin": self.candidate.margin,
                "valid": self.candidate.valid, "updates": self.candidate.updates,
                "reason": self.candidate.reason,
                "selected_time": self.candidate.selected_time,
                "verified_time": self.candidate.verified_time}),
            "own": [float(obs.own.x), float(obs.own.y)],
            "own_v": [float(obs.own.vx), float(obs.own.vy)],
            "contact_phase": self.contact_phase,
                "desired_heading": self._desired_heading(n, t, raw),
                "ff_t": self.ff_t, "u_adjust": self.u_adjust, "contact_err_t": self.contact_err_t,
            "selector_debug": self.allocated_debug, "desired_pre": self.raw_pre, "avoidance_active": self.avoidance_active, "avoidance_direction": self.avoidance_direction,
            "gap": self.gap, "ref_center": self.debug_ref, "q_prime": self.debug_qprime, "contact_offset": self.debug_offset, "contact_ready": bool(self.contact_ready),
            "peer_contact": self.peer_contact, "pair_push_ready": self.pair_push_ready,
            "retreat_until": self.retreat_until, "shift_until": self.shift_until,
            "retry_count": self.retry_count, "shift_sign": self.shift_sign,
            "effect": {
                "version": "unknown-shape-motion-v2", "kind": "translate",
                "estimator_debug": getattr(self.estimator, 'debug', None),
                "direction": list(self.action.effect_direction),
                "target": None, "unit": "none",
                "face_normal": n, "face_points": list(patch.points_world) if patch is not None else [],
                "estimated_centre": None, "contact_target": cp,
                "geometry_age": 0.0, "endpoint": None,
                "v_est": [float(self.estimator.ekf.x[0]), float(self.estimator.ekf.x[1]), float(self.estimator.ekf.x[2])],
                "v_cov": self.estimator.ekf.P.tolist(), "line_count": len(self.estimator.lines),
                "candidate_err_t": self.candidate_err_t,
            "ff_t": self.ff_t, "u_adjust": self.u_adjust, "contact_err_t": self.contact_err_t,
            "selector_debug": self.allocated_debug, "desired_pre": self.raw_pre, "avoidance_active": self.avoidance_active, "avoidance_direction": self.avoidance_direction,
            "gap": self.gap, "ref_center": self.debug_ref, "q_prime": self.debug_qprime, "contact_offset": self.debug_offset, "contact_ready": bool(self.contact_ready),
            "peer_contact": self.peer_contact, "pair_push_ready": self.pair_push_ready,
                "candidate": (None if self.candidate is None else {"id": self.candidate.id, "q": list(self.candidate.q),
                    "support": list(self.candidate.support), "margin": self.candidate.margin,
                    "reason": self.candidate.reason, "verified_time": self.candidate.verified_time}),
                "own": [float(obs.own.x), float(obs.own.y)],
                "own_v": [float(obs.own.vx), float(obs.own.vy)],
                "contact_point_robot": None,
                "contact_point_world": (None if cp is None else list(cp)),
                "contact_phase": self.contact_phase,
                "desired_heading": self._desired_heading(n, t, raw),
                "ff_t": self.ff_t, "u_adjust": self.u_adjust, "contact_err_t": self.contact_err_t,
            "selector_debug": self.allocated_debug, "desired_pre": self.raw_pre, "avoidance_active": self.avoidance_active, "avoidance_direction": self.avoidance_direction,
            "gap": self.gap, "ref_center": self.debug_ref, "q_prime": self.debug_qprime, "contact_offset": self.debug_offset, "contact_ready": bool(self.contact_ready),
            "peer_contact": self.peer_contact, "pair_push_ready": self.pair_push_ready,
                "feedback": True, "prior_free": True,
            },
            "feedback": {"v_est": [float(self.estimator.ekf.x[0]), float(self.estimator.ekf.x[1]), float(self.estimator.ekf.x[2])]},
        }
        return info


class UnknownShapeEnvMixin:
    """OptionSystemEnv factory replacement for the prior-free PUSH."""
    def install_prior_free(self):
        self.motion_estimator = SharedMotionEstimator()
        self.budget_trackers = {0: OptionBudget(), 1: OptionBudget()}
        self._q_generation = 0
        self._commit_valid = False
        self._commit_key = None
        self._commit_q = {0: None, 1: None}
        self._commit_support = {0: None, 1: None}
        self._commit_anchor = {0: None, 1: None}
        self._commit_alloc = None
        self.factory = lambda action: make_unknown_skill(action, self.overrides, self.motion_estimator)
        for slot in self.slots:
            slot.factory = self.factory
        orig = self._coordinate
        def coordinate(now, observations=None):
            observations = observations if observations is not None else self.boundary_observations()
            _items = observations.items() if isinstance(observations, dict) else enumerate(observations)
            for _i, _o in _items:
                self.budget_trackers[int(_i)].update(_o, now, spec.CONTROL_DT)
            o_ref_est = None  # contour-center experiment disabled (stale/partial memory biased it)
            if observations is None:
                observations = self.boundary_observations()
            self.motion_estimator.update(observations, now, spec.CONTROL_DT)
            paired = all(isinstance(slot.skill, UnknownShapePush) for slot in self.slots)
            def _aligned(o):
                p = getattr(o, 'patch', None)
                if p is None or not getattr(p, 'valid', False) or patch_gap(p) > .02:
                    return False
                nn = np.asarray(p.normal_world, float)
                hh = math.atan2(float(nn[1]), float(nn[0]))
                ee = math.atan2(math.sin(hh - o.own.angle), math.cos(hh - o.own.angle))
                return abs(ee) <= .18
            both_ready = bool(paired and _aligned(observations[0]) and _aligned(observations[1]))
            if both_ready:
                self._pair_contact_latch = True
            if not paired:
                self._pair_contact_latch = False
            for i, slot in enumerate(self.slots):
                if isinstance(slot.skill, UnknownShapePush):
                    slot.skill.paired = paired
                    slot.skill.peer_contact = bool((observations[1-i].patch is not None and patch_gap(observations[1-i].patch) <= .02) if paired else False)
                    slot.skill.pair_push_ready = bool(getattr(self, '_pair_contact_latch', False))
                    slot.skill.allocated_q = None
                    slot.skill.allocated_wait = False
                    slot.skill.allocated_id = None
            # Commit once per actual skill lifetime (not merely action index).
            # All later q writes go through finite-segment tracking below.
            _akey = tuple((int(getattr(slot.action, 'index', -1)), id(slot.skill))
                          for slot in self.slots)
            if _akey != self._commit_key:
                self._commit_valid = False
                self._commit_key = _akey
                self._commit_q = {0: None, 1: None}
                self._commit_support = {0: None, 1: None}
                self._commit_anchor = {0: None, 1: None}
                self._commit_alloc = None
                self._q_generation += 1
            selected_now = False
            if not self._commit_valid:
                if paired:
                    alloc = (select_simple_pair(observations[0], self.slots[0].action,
                                                observations[1], self.slots[1].action)
                             if SIMPLE_SELECTOR else None)
                    if alloc is None:
                        d0 = np.asarray(self.slots[0].action.effect_direction, float)
                        d1 = np.asarray(self.slots[1].action.effect_direction, float)
                        df = d0 + d1
                        df = df / max(1e-9, float(np.linalg.norm(df)))
                        alloc = select_contact_pair(observations[0], self.slots[0].action,
                                                    observations[1], self.slots[1].action,
                                                    desired_force=tuple(df),
                                                    o_ref=o_ref_est)
                    if alloc is not None:
                        self._commit_alloc = alloc
                        for i in (0, 1):
                            self._commit_q[i] = alloc.get(f'q{i}')
                else:
                    for i, slot in enumerate(self.slots):
                        if not isinstance(slot.skill, UnknownShapePush):
                            continue
                        if SIMPLE_SELECTOR:
                            sel = select_simple_point(observations[i], slot.action)
                        else:
                            df = tuple(slot.action.effect_direction)
                            sel = select_contact_point(observations[i], desired_force=df,
                                                       o_ref_override=o_ref_est)
                        self._commit_q[i] = None if sel is None else sel['q']
                self._commit_valid = any(q is not None for q in self._commit_q.values())
                selected_now = self._commit_valid
                for i, q in self._commit_q.items():
                    if q is not None:
                        f = _face_for_q(observations[i], q)
                        if f is not None:
                            self._commit_anchor[i] = _face_anchor(f, float(np.asarray(q) @ f[2]))
                            self._commit_support[i] = (float(f[4]), float(f[5]))
            for i, slot in enumerate(self.slots):
                sk = slot.skill
                if not isinstance(sk, UnknownShapePush):
                    continue
                sk.allocation_managed = True
                sk.allocated_face = None
                sk.allocated_wait = True
                q = self._commit_q.get(i)
                if q is None:
                    sk.allocated_debug = dict(q_tracking=dict(source='selector_unavailable', matched=False))
                    continue
                # A lock identifies a material contact point/segment; it does
                # not freeze its world coordinates while the object moves.
                velocity = self.motion_estimator.velocity_at_contact(i, q)
                q, f, anchor, debug = track_contact_q(
                    observations[i], q, self._commit_anchor.get(i),
                    velocity=velocity, dt=(0.0 if selected_now else spec.CONTROL_DT))
                debug.update(selector_evaluated=selected_now,
                             segment_id=f'{self._q_generation}:{i}',
                             commit_source=('pair_selector' if paired else 'single_selector'))
                self._commit_q[i] = q
                self._commit_anchor[i] = anchor
                if f is not None:
                    self._commit_support[i] = (float(f[4]), float(f[5]))
                    sk.allocated_face = f[0]
                    sk.allocated_wait = False
                sk.allocated_q = q
                sk.allocated_support = self._commit_support.get(i)
                sk.allocated_normal = None if anchor is None else anchor['n']
                sk.allocated_id = (self._q_generation, i)
                sk.allocated_reason = debug['source']
                sk.allocated_debug = dict(q_tracking=debug)
            return orig(now, observations)
        self._coordinate = coordinate
        return self


def make_unknown_skill(action, overrides=None, estimator=None):
    action = S.ACTIONS[S.canonical_index(action.index)]
    if action.kind == "push" and getattr(action, "effect_kind", None) == "translate":
        sk = UnknownShapePush(action, estimator=estimator)
        ov = (overrides or {}).get("PUSH", {})
        # no shape/prior fields are accepted from override; only control gains are allowed
        allowed = {"approach_normal_speed", "gap_gain", "tangential_gain", "tangential_speed"}
        cfg = {k: v for k, v in ov.items() if k in allowed and hasattr(sk.config, k)}
        if cfg:
            sk.config = __import__("dataclasses").replace(sk.config, **cfg)
        return sk
    return S.make_skill(action, overrides)
