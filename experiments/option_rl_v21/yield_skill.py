"""Local clearance control. No target, full contour, object centre or truth state.

Sampled velocities are checked against the current measured surface points and
short segments. Object motion comes from the existing legal shared estimator;
the teammate velocity is shared odometry. This is a bounded local heuristic,
not a global planner or a collision-free guarantee for unobserved geometry.
"""
from dataclasses import asdict, dataclass
import math
import numpy as np

from coimagine.random_start import spec as R
from coimagine.random_start.lidar import CATEGORY_OBJECT, CATEGORY_TEAMMATE
from coimagine.random_start.skills import Skill


@dataclass(frozen=True)
class YieldConfig:
    speed: float = .40
    enter_gap: float = .20
    exit_gap: float = .32
    clearance: float = .12
    horizon: float = .80
    side_bias: float = 1.30
    point_padding: float = .015

    def __post_init__(self):
        if not all(math.isfinite(v) and v > 0 for v in asdict(self).values()):
            raise ValueError('YIELD parameters must be positive and finite')
        if self.exit_gap <= self.enter_gap or self.clearance >= self.exit_gap:
            raise ValueError('invalid YIELD hysteresis/clearance')


class YieldSkill(Skill):
    kind = 'yield'

    def __init__(self, action, *, estimator=None, config=YieldConfig()):
        super().__init__(action)
        self.estimator = estimator
        self.config = config
        self.time_budget = R.TIME_BUDGET_SECONDS + R.CONTROL_DT
        self.active = False
        self.side = 0
        self.escape_direction = None
        self.stage = 'yield_idle'

    def _surfaces(self, obs):
        scan = obs.scan
        indices = np.flatnonzero(np.asarray(scan.valid, bool))
        if not len(indices):
            return np.empty((0, 2)), np.empty((0, 2)), np.empty(0, int)
        angles = np.asarray(scan.ray_angles)[indices]
        distances = np.asarray(scan.distances)[indices]
        points = np.asarray(scan.origin) + distances[:, None]*np.column_stack((np.cos(angles), np.sin(angles)))
        categories = np.asarray(scan.category)[indices]
        # Densify only adjacent, nearby returns of one measured class. Never
        # close an unseen contour or join across visibility discontinuities.
        extra, labels = [], []
        for j in range(len(indices)-1):
            span = float(np.linalg.norm(points[j+1]-points[j]))
            if indices[j+1]-indices[j] == 1 and categories[j] == categories[j+1] and .045 < span < .25:
                count = int(math.ceil(span/.045))
                for k in range(1, count):
                    extra.append(points[j] + (points[j+1]-points[j])*k/count)
                    labels.append(categories[j])
        if extra:
            points = np.vstack((points, extra)); categories = np.concatenate((categories, labels))
        velocities = np.zeros_like(points)
        velocities[categories == CATEGORY_TEAMMATE] = (obs.teammate.vx, obs.teammate.vy)
        if self.estimator is not None:
            for i in np.flatnonzero(categories == CATEGORY_OBJECT):
                value = self.estimator.velocity_at_contact(obs.robot_id, points[i])
                if value is None:
                    value = self.estimator.ekf.velocity_at(points[i])
                velocity = np.asarray(value, float)
                if velocity.shape == (2,) and np.all(np.isfinite(velocity)):
                    length = float(np.linalg.norm(velocity))
                    velocities[i] = velocity*min(1., .60/max(length, 1e-12))
        return points, velocities, categories

    def choose_velocity(self, obs, dt):
        cfg = self.config
        own = np.asarray((obs.own.x, obs.own.y))
        own_v = np.asarray((obs.own.vx, obs.own.vy))
        points, velocities, categories = self._surfaces(obs)
        if not len(points):
            self.active = False
            self.escape_direction = None
            return (0., 0.), dict(active=False, blocked=False, min_gap=None,
                                  predicted_min_gap=None, candidates=[], side=self.side)
        relative = own-points
        ranges = np.linalg.norm(relative, axis=1)
        normals = relative/np.maximum(ranges[:, None], 1e-9)
        gaps = ranges - R.ROBOT_RADIUS
        closing = np.sum((velocities-own_v)*normals, axis=1)
        # Threat selection includes looming returns, not only the nearest point.
        predicted_gaps = gaps - np.maximum(0., closing)*cfg.horizon
        threat = int(np.argmin(predicted_gaps))
        gap = float(gaps.min())
        danger = gap < cfg.enter_gap or float(predicted_gaps.min()) < cfg.clearance
        if danger:
            self.active = True
        elif gap > cfg.exit_gap and float(predicted_gaps.min()) > cfg.enter_gap:
            self.active = False
            self.escape_direction = None
        if not self.active:
            return (0., 0.), dict(active=False, blocked=False, min_gap=gap,
                                  predicted_min_gap=float(predicted_gaps.min()), candidates=[], side=self.side)
        away = normals[threat]
        tangent = np.array((-away[1], away[0]))
        # Fixed robot-ID tie break; the selected side persists unless its local
        # clearance becomes worse. Sideways bias helps exit a moving object's lane.
        if self.escape_direction is not None:
            self.side = 1 if float(tangent @ self.escape_direction) >= 0. else -1
        elif self.side == 0:
            self.side = 1 if obs.robot_id == 0 else -1
        preferred = away + cfg.side_bias*self.side*tangent
        preferred *= cfg.speed/max(float(np.linalg.norm(preferred)), 1e-9)
        dirs = np.column_stack((np.cos(np.arange(16)*math.pi/8), np.sin(np.arange(16)*math.pi/8)))
        candidates = np.vstack((np.zeros((1, 2)), cfg.speed*.5*dirs, cfg.speed*dirs))
        times = np.arange(1, 9)*cfg.horizon/8
        # Roll out the SAME acceleration bound as ReferenceVelocity; a candidate
        # cannot instantly reverse the robot's motion in the collision check.
        reference = np.asarray(self.reference.value, float)
        pos = np.zeros_like(candidates); ref = np.broadcast_to(reference, candidates.shape).copy()
        samples = []
        for step, time in enumerate(times):
            delta = candidates-ref
            norms = np.linalg.norm(delta, axis=1)
            ref += delta*np.minimum(1., R.A_REF_MAX*(cfg.horizon/8)/np.maximum(norms, 1e-9))[:, None]
            pos += ref*(cfg.horizon/8)
            obstacles = points + velocities*time
            clearance = np.linalg.norm((own+pos)[:, None, :]-obstacles[None, :, :], axis=2)
            clearance = clearance.min(axis=1)-R.ROBOT_RADIUS-cfg.point_padding
            # Arena limits are public, irrespective of sparse wall returns.
            proposed = own+pos
            walls = np.column_stack((proposed[:, 0], proposed[:, 1],
                                     obs.arena[0]-proposed[:, 0], obs.arena[1]-proposed[:, 1])).min(axis=1)-R.ROBOT_RADIUS-R.BOUNDARY_RADIUS
            samples.append(np.minimum(clearance, walls))
        samples = np.stack(samples, axis=1)
        minimum, final = samples.min(axis=1), samples[:, -1]
        # In existing contact permit escape, but no candidate that worsens the
        # observed penetration/clearance is labelled feasible.
        threshold = min(cfg.clearance, gap-cfg.point_padding-.005)
        feasible = minimum >= threshold
        deficit = np.maximum(0., cfg.clearance-samples)
        cost = (np.sum((candidates-preferred)**2, axis=1)
                + 12.*np.mean(deficit**2, axis=1) - .5*np.minimum(final, cfg.exit_gap))
        if feasible.any():
            choice = int(np.argmin(np.where(feasible, cost, np.inf)))
        else:
            # Report blocked while applying the least-worsening escape, rather
            # than freezing an already endangered robot in place.
            choice = int(np.argmax(minimum-.02*cost))
        chosen = candidates[choice]
        escape_available = feasible & (np.linalg.norm(candidates, axis=1) > .05)
        blocked = (not bool(feasible.any()) or
                   (gap < cfg.clearance and not bool(escape_available.any())))
        lateral = float(chosen @ tangent)
        if abs(lateral) > .05:
            self.side = 1 if lateral > 0 else -1
            self.escape_direction = self.side*tangent
        return tuple(float(x) for x in chosen), dict(
            active=True, blocked=blocked, min_gap=gap,
            predicted_min_gap=float(minimum[choice]), predicted_final_gap=float(final[choice]),
            threat_point=points[threat].tolist(), threat_velocity=velocities[threat].tolist(),
            threat_category=int(categories[threat]), closing_rate=float(closing[threat]),
            preferred_velocity=preferred.tolist(), chosen_velocity=chosen.tolist(), side=self.side,
            escape_direction=None if self.escape_direction is None else self.escape_direction.tolist(),
            candidates=[dict(v=v.tolist(), min_gap=float(m), feasible=bool(f), cost=float(c))
                        for v, m, f, c in zip(candidates, minimum, feasible, cost)])

    def step(self, observation, dt):
        self.skill_time += dt
        desired, debug = self.choose_velocity(observation, dt)
        self.stage = ('yield_blocked' if debug['blocked'] else 'yield_escape' if self.active else 'yield_idle')
        reference = self.reference.update(desired, dt)
        force, saturated = self._force_from_reference(reference, observation)
        return force, dict(skill=self.action.label, kind='yield', stage=self.stage,
            done=False, exit_reason=None, desired_velocity=list(desired),
            reference_velocity=list(reference), force=list(force), force_norm=math.hypot(*force),
            force_saturated=saturated, yield_debug=debug, contacting_object=bool(observation.contacting_object),
            avoidance_active=self.active, avoidance_direction=(list(desired) if self.active else None),
            progress=0., progress_target=None)
