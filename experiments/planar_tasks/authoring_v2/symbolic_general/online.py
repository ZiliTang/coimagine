"""Bounded runtime rollout from public-history geometric hypotheses only."""
from copy import deepcopy
import math
import time

from experiments.planar_tasks.relational_search.contract import History, public_observation, digest
from experiments.planar_tasks.relational_search.estimator import estimate
from experiments.planar_tasks.relational_search.predictor import build
from experiments.planar_tasks.semantic_rl.environment import Environment, EventConfig, Detector
from . import frontend
from .prediction import outcome_valid


class SyntheticWorld(Environment):
    def __init__(self, hypothesis):
        self.env = build(hypothesis)
        self.env.config.time_budget = 90.
        self.config = EventConfig(mode='duration')
        self.record_details = False
        self.decision_count = 0
        self.fast_feedback = True
        self.env.cache_public_sensors = True
        self.detector = Detector(self.config)
        self.history = History(); self._append_public(self.env.public())
        self.events = []


def compatibility(real, synthetic, tolerance=.08):
    """Reject hypotheses contradicting current measured range/category evidence.

    Exact robot poses/ray headings are shared. All observed target hits must have
    target support within tolerance; no hypothetical hit may occupy observed free
    ray space. Unseen shape completeness and cold controller state remain unknown.
    """
    target = real['goal']['target_id']; errors = []; support = 0; free_violations = 0
    for r, s in zip(real['robots'], synthetic['robots']):
        a, b = r['scan'], s['scan']
        for ok, d, label, sok, sd, slabel in zip(a['valid'], a['distances'], a['entity_ids'],
                                                b['valid'], b['distances'], b['entity_ids']):
            if sok and sd < d-tolerance: free_violations += 1
            if ok and label == target:
                support += 1
                errors.append(abs(d-sd) if sok and slabel == target else 100.)
    return dict(compatible=bool(support >= 3 and free_violations == 0 and max(errors, default=100.) <= tolerance),
                target_rays=support, max_target_range_error=max(errors, default=None),
                free_space_violations=free_violations, tolerance_m=tolerance)


def hypotheses(memory, observation):
    h = History(); h.append(public_observation(observation))
    base = estimate(h)
    target = observation['goal']['target_id']
    fit = frontend.estimate_target(observation, memory.get('centre'))
    if fit is None: raise ValueError('insufficient public target geometry')
    # Replace the generic fit with the versioned local-frame fit.
    entity = next(e for e in base['entities'] if e['id'] == target)
    entity.update(x=fit['centre'][0], y=fit['centre'][1], angle=fit['angle'],
                  width=fit['width']+.04, height=fit['depth'])
    # Persist observed obstacles without borrowing the execution map. Thin cells
    # are explicit geometric hypotheses, not evidence of complete free space.
    base['obstacles'] = [[p[0], p[1], .015, .015, 0.] for p in memory.get('static', [])]
    root = observation['tick']
    for delta in (0., -.025, .025):
        proposal = deepcopy(base)
        obj = next(e for e in proposal['entities'] if e['id'] == target)
        # Vary unknown depth while holding the observer-facing side fixed.
        ox = sum(r['own']['x'] for r in observation['robots'])/2
        oy = sum(r['own']['y'] for r in observation['robots'])/2
        nx, ny = -math.sin(obj['angle']), math.cos(obj['angle'])
        sign = 1 if (obj['x']-ox)*nx+(obj['y']-oy)*ny >= 0 else -1
        obj['height'] = max(.04, obj['height']+delta)
        obj['x'] += sign*nx*delta/2; obj['y'] += sign*ny*delta/2
        yield proposal, root


class OnlineRollout:
    def __init__(self, *, max_branches=600, max_ticks=12000, max_seconds=60., per_query=3):
        self.limits = dict(branches=max_branches, ticks=max_ticks, seconds=max_seconds, per_query=per_query)
        self.ledger = dict(branches=0, ticks=0, seconds=0., models=0, incompatible=0, errors=0, budget_exhausted=0)

    def run(self, memory, observation, action, command):
        results = []; details = []; started = time.monotonic()
        if (self.ledger['branches'] >= self.limits['branches'] or
                self.ledger['ticks']+command['ticks'] > self.limits['ticks'] or
                self.ledger['seconds'] >= self.limits['seconds']):
            self.ledger['budget_exhausted'] += 1
            return dict(samples=[],details=[dict(status='budget_exhausted_before_construction')],
                        wall_seconds=0.,coverage='no new evidence; online budget exhausted')
        try:
            proposals = hypotheses(memory, observation)
            for i, (hypothesis, offset) in enumerate(proposals):
                if i >= self.limits['per_query']: break
                if (self.ledger['branches'] >= self.limits['branches'] or
                        self.ledger['ticks']+command['ticks'] > self.limits['ticks'] or
                        self.ledger['seconds']+time.monotonic()-started >= self.limits['seconds']):
                    self.ledger['budget_exhausted'] += 1; break
                world = SyntheticWorld(hypothesis); self.ledger['models'] += 1
                check = compatibility(observation, world.history.events[-1]['observation'])
                detail = dict(hypothesis_sha256=digest(hypothesis), compatibility=check)
                details.append(detail)
                if not check['compatible']:
                    self.ledger['incompatible'] += 1; continue
                self.ledger['branches'] += 1
                # Reserve worst-case work first; actual completion settles it.
                self.ledger['ticks'] += command['ticks']
                try:
                    from .simulation import measured_step
                    measured = measured_step(world, command)
                    self.ledger['ticks'] -= command['ticks']-measured['event']['periods']
                    end = deepcopy(world.history.events[-1]['observation'])
                    end['tick'] += offset
                    for e in end['entities']: e['last_seen_tick'] += offset
                    # Only public synthetic output enters the semantic predictor.
                    next_memory = frontend.update_memory(deepcopy(memory), dict(observation=end))
                    z, _ = frontend.semantic_packet(next_memory, end)
                    results.append(dict(values=z, path_safe=not measured['object_static'],
                                        source='public_hypothesis_rollout', hypothesis_sha256=detail['hypothesis_sha256']))
                    detail.update(status='complete', ticks=measured['event']['periods'])
                except (ValueError, KeyError, RuntimeError) as exc:
                    self.ledger['errors'] += 1
                    detail.update(status='error', error=str(exc))
        except (ValueError, KeyError, RuntimeError) as exc:
            self.ledger['errors'] += 1; details.append(dict(error=str(exc)))
        elapsed = time.monotonic()-started; self.ledger['seconds'] += elapsed
        return dict(samples=results, details=details, wall_seconds=elapsed,
                    coverage='finite observed-compatible geometric hypotheses; unknown space and cold controller remain unproved')


class Controller:
    def __init__(self, model, *, risk=.10, max_queries=2, rollout=None):
        self.model = model; self.risk = risk; self.max_queries = max_queries
        self.rollout = rollout or OnlineRollout()
        self.ledger = dict(boundaries=0, model_admissions=0, model_rejections=0,
                           online_queries=0, online_admissions=0, withheld=0,
                           rollout=self.rollout.ledger)

    def decide(self, memory, observation, z, commands, policy):
        self.ledger['boundaries'] += 1
        permissions = {}; checks = []; local = dict(z); queries = 0
        for _ in range(len(commands)):
            selected = policy.select(local); action = selected['option']
            if action is None: break
            if action == 'brake':
                # A brake is an explicit fallback, not productive coverage or a
                # claim that stopping guarantees no subsequent physical collision.
                permissions[action] = dict(permit=True, source='fallback_stop'); break
            if not selected['executable']:
                permissions[action] = dict(permit=False, source='current_precondition')
                local['permit:'+action] = False; continue
            pred = self.model.predict(action, local, policy.host)
            verdicts = [outcome_valid(policy, action, s['values'], path_safe=s['path_safe'])[0] for s in pred['samples']]
            risk = 1-sum(verdicts)/len(verdicts) if verdicts else None
            allowed = False; source = 'insufficient'; report = None; online = None
            if pred['supported'] and pred['calibrated']:
                # The full declared interval model, not only sample votes, must
                # pass actual successor selection. Empirical quality is retained.
                box = {k: (v if type(v) is bool else [v, v]) for k, v in local.items()}
                report = policy.check(box, action, self.model.interval_outcomes(pred, policy.host),
                                      quality='empirical', enumerate_rules=False)
                allowed = report['status'] == 'conditional' and risk is not None and risk <= self.risk
                source = 'model'
                self.ledger['model_admissions' if allowed else 'model_rejections'] += 1
            if not allowed and queries < self.max_queries and (risk is None or risk < .8):
                online = self.rollout.run(memory, observation, action, commands[action])
                queries += 1; self.ledger['online_queries'] += 1
                tests = [outcome_valid(policy, action, s['values'], path_safe=s['path_safe'])[0] for s in online['samples']]
                allowed = len(tests) >= 2 and all(tests)
                source = 'online_empirical' if allowed else 'online_insufficient'
                if allowed: self.ledger['online_admissions'] += 1
            if not allowed: self.ledger['withheld'] += 1
            permissions[action] = dict(permit=allowed, source=source)
            checks.append(dict(action=action, allowed=allowed, source=source, predicted_risk=risk,
                               support_rooms=pred['local_rooms'], calibrated=pred['calibrated'],
                               symbolic=report, online=online))
            if allowed: break
            local['permit:'+action] = False
        return dict(permissions=permissions, checks=checks, model_sha256=self.model.sha256)
