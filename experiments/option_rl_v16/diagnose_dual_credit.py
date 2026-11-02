"""D001: dual-push success vs frozen greedy, advantage, and local gradient.

Read-only diagnostic. No optimizer.step, no normalizer.update, no checkpoint write.
It loads the two frozen v15 checkpoints, runs the task-book episode budget, records
per-decision legal inputs/masks/actions/logp/value/reward and the full joint
distribution, then reconstructs four 8-episode PPO batches for GAE/advantage and
first-order actor-gradient diagnostics.

Object/robot true states are recorded only for offline replay/effect labels. They
never enter the policy, mask, or gradient inputs.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, subprocess, sys, time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coimagine.random_start import spec
from experiments.option_rl_v16.train import Step, load_checkpoint, returns_and_advantages, run_episode
from experiments.option_rl_v16.interface import ACTIONS, TYPE_COUNT, SCALE_COUNT, make_env, observe
from experiments.option_rl_v16.learnability import load_panel
from experiments.option_rl_v16.live import atomic_json

TASK_VERSION = 'D001 v1'
EXPECTED = {
    'panel': 'a60bdcc120923a8091d8e8930eb4717406075aeeb55b5291e3f924dd48227126',
    'independent': '1e8a0c88ab65ef9a9a42b58a11b625094804a7b5127650ce8f4504ae05b19449',
    'joint_types': 'dac637e6d21f64d6fa3bf98c658c176d12b1ac7409b66fe6ff84186e5121e6c5',
}
PANEL_ORDER = ['contact_2', 'contact_0', 'approach_2', 'approach_0',
               'dual_2', 'dual_0', 'reposition_2', 'reposition_0']
DUAL_ROOTS = ['dual_2', 'dual_0']
OTHER_ROOTS = ['contact_2', 'contact_0', 'approach_2', 'approach_0',
               'reposition_2', 'reposition_0']
RANDOM_SEEDS = list(range(1000, 1016))
BATCH_SEEDS = list(range(1000, 1004))
CATEGORY_NAMES = ('same_push', 'same_speed', 'push_reposition', 'both_brake', 'other', 'all')
TASK_NAMES = ('dual', 'single', 'reposition', 'all')
GROUPS = ('body', 'base_heads', 'joint_types', 'all')
ENTROPY_COEF = 0.02
TRACE_SECONDS = 10.0


# ---------------------------------------------------------------------------
# frozen action categories
# ---------------------------------------------------------------------------

def _is_push(a):
    return ACTIONS[a].kind == 'push'


def _is_translate_push(a):
    x = ACTIONS[a]
    return x.kind == 'push' and x.effect_kind == 'translate'


def _is_reposition(a):
    return ACTIONS[a].kind == 'reposition'


def _is_brake(a):
    return ACTIONS[a].kind == 'wait'


def _same_push(a, b):
    return (_is_translate_push(a) and _is_translate_push(b)
            and ACTIONS[a].type_index == ACTIONS[b].type_index)


def _same_speed(a, b):
    return _same_push(a, b) and abs(float(ACTIONS[a].speed) - float(ACTIONS[b].speed)) < 1e-9


def _push_reposition(a, b):
    return ((_is_translate_push(a) and _is_reposition(b))
            or (_is_translate_push(b) and _is_reposition(a)))


def _both_brake(a, b):
    return _is_brake(a) and _is_brake(b)


def _category_masks():
    n = len(ACTIONS)
    same_push = np.zeros((n, n), dtype=bool)
    same_speed = np.zeros((n, n), dtype=bool)
    push_reposition = np.zeros((n, n), dtype=bool)
    both_brake = np.zeros((n, n), dtype=bool)
    for a in range(n):
        for b in range(n):
            same_push[a, b] = _same_push(a, b)
            same_speed[a, b] = _same_speed(a, b)
            push_reposition[a, b] = _push_reposition(a, b)
            both_brake[a, b] = _both_brake(a, b)
    return same_push, same_speed, push_reposition, both_brake


SAME_PUSH, SAME_SPEED, PUSH_REPO, BOTH_BRAKE = _category_masks()


# ---------------------------------------------------------------------------
# policy helpers
# ---------------------------------------------------------------------------

def _joint_logits(policy, x, masks):
    h = policy.body(x)
    log0 = policy._logits(x, h, masks, 0)
    log1 = policy._logits(x, h, masks, 1)
    if policy.model_config.get('coupled'):
        scores = policy.joint_types(h).reshape(-1, TYPE_COUNT, TYPE_COUNT)
        residual = scores[:, policy.type_ids[:, None], policy.type_ids[None, :]]
        joint = log0[:, :, None] + log1[:, None, :] + residual
    else:
        joint = log0[:, :, None] + log1[:, None, :]
    return joint.masked_fill(~policy.legal_pairs[None, :, :], -torch.inf)


def joint_probabilities(policy, obs, mask0, mask1):
    """Return (46x46 probabilities, 46x46 log-probs) for one encoded observation."""
    x = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :])
    masks = (torch.tensor(np.asarray(mask0, dtype=bool)[None, :]),
             torch.tensor(np.asarray(mask1, dtype=bool)[None, :]))
    with torch.no_grad():
        joint = _joint_logits(policy, x, masks)
        log_z = torch.logsumexp(joint.reshape(1, -1), dim=-1)
        logp = joint - log_z[:, None, None]
        probs = logp.exp()
    return probs[0].numpy(), logp[0].numpy()


def same_push_logprob(policy, obs, mask0, mask1):
    """Differentiable log P(both translate PUSH, same direction) for one input."""
    x = torch.tensor(np.asarray(obs, dtype=np.float32)[None, :])
    masks = (torch.tensor(np.asarray(mask0, dtype=bool)[None, :]),
             torch.tensor(np.asarray(mask1, dtype=bool)[None, :]))
    joint = _joint_logits(policy, x, masks)
    selected = torch.tensor(SAME_PUSH[None, :, :], dtype=torch.bool) & policy.legal_pairs[None, :, :]
    values = joint[0][selected[0]]
    if not bool(torch.isfinite(values).any()):
        return None
    log_z = torch.logsumexp(joint.reshape(1, -1), dim=-1)
    return torch.logsumexp(values, dim=-1) - log_z


def batch_logp_entropy(policy, obs, masks, a0, a1):
    ds, _ = policy.distributions(obs, masks, a0=a0)
    logp = ds[0].log_prob(a0) + ds[1].log_prob(a1)
    entropy = ds[0].entropy() + ds[1].entropy()
    return logp, entropy


# ---------------------------------------------------------------------------
# recording environment / normalizer
# ---------------------------------------------------------------------------

class TraceEnv:
    """Delegating env wrapper that records 5 Hz true states for offline replay only."""

    def __init__(self, env):
        self.env = env
        self.frames = []
        self.min_distance = float(env.object_distance())
        self.entered_goal = bool(self.min_distance <= spec.GOAL_RADIUS)
        self._record(initial=True)

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _record(self, initial=False):
        env = self.env
        obj = env.object_state()
        robots = env.robot_states()
        observations = env.boundary_observations()
        stages = []
        forces = []
        for slot in env.slots:
            info = slot.last_info or {}
            stages.append('initial' if initial else str(info.get('stage', slot.skill.stage)))
            force = info.get('force') or (0., 0.)
            forces.append(float(math.hypot(float(force[0]), float(force[1]))))
        self.frames.append(dict(
            t=0. if initial else float(env.total_periods) * spec.CONTROL_DT,
            o=[float(obj.x), float(obj.y), float(obj.angle)],
            r=[[float(r.x), float(r.y), float(r.angle)] for r in robots],
            ct=[bool(o.contacting_object) for o in observations],
            st=stages,
            fo=forces,
            distance=float(env.object_distance()),
        ))

    def joint_step(self, a0, a1, record=False):
        boundary = self.env.joint_step(a0, a1, record=record)
        self._record()
        distance = float(self.env.object_distance())
        self.min_distance = min(self.min_distance, distance)
        self.entered_goal = bool(self.entered_goal or distance <= spec.GOAL_RADIUS)
        return boundary


class RecorderNormalizer:
    """Delegates normalisation and records the exact raw observation used by the policy."""

    def __init__(self, base):
        self.base = base
        self.raw = []

    def normalise(self, x):
        self.raw.append(np.asarray(x, dtype=np.float32).copy())
        return self.base.normalise(x)

    def __getattr__(self, name):
        return getattr(self.base, name)


# ---------------------------------------------------------------------------
# worker
# ---------------------------------------------------------------------------

_CACHE = {}


def _worker_init():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _load_policy(model, checkpoint):
    key = (model, str(checkpoint))
    if key not in _CACHE:
        policy, normalizer, data = load_checkpoint(checkpoint)
        policy.eval()
        _CACHE[key] = (policy, normalizer, data)
    return _CACHE[key]


def _load_instances(panel_path):
    key = ('panel', str(panel_path))
    if key not in _CACHE:
        train, validation = load_panel(panel_path)
        _CACHE[key] = {i.name: i for i in train}
    return _CACHE[key]


def _jsonable_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def run_job(job):
    """One bounded episode; returns a small episode record and per-decision rows."""
    model = job['model']
    checkpoint = job['checkpoint']
    panel_path = job['panel']
    root = job['root']
    mode = job['mode']
    seed = job['seed']
    out = Path(job['out'])
    policy, base_normalizer, data = _load_policy(model, checkpoint)
    instance = _load_instances(panel_path)[root]
    recorder = RecorderNormalizer(base_normalizer)
    holder = {}

    def factory(instance_):
        env = make_env(instance_, context=policy.rollout_config['context'])
        if job.get('time_budget') is not None:
            from dataclasses import replace
            env.config = replace(env.config, time_budget=float(job['time_budget']))
        trace = TraceEnv(env)
        holder['trace'] = trace
        return trace

    generator = torch.Generator().manual_seed(seed if seed >= 0 else 1000)
    steps, summary = run_episode(
        policy, instance, recorder,
        generator=generator, deterministic=(mode == 'greedy'),
        update_stats=False, record=False, env_factory=factory,
    )
    trace = holder['trace']
    raw_obs = recorder.raw
    if len(raw_obs) != len(steps):
        raise RuntimeError(f'raw observation count {len(raw_obs)} != steps {len(steps)}')
    if len(summary['actions']) != len(steps):
        raise RuntimeError(f'decision count {len(summary["actions"])} != steps {len(steps)}')
    if len(trace.frames) != len(summary['execution_samples']) + 1:
        raise RuntimeError(f'trace frames {len(trace.frames)} != execution samples+1 {len(summary["execution_samples"]) + 1}')

    encoded = np.stack([s.obs for s in steps]).astype(np.float32)
    raw = np.stack(raw_obs).astype(np.float32)
    mask0 = np.stack([s.mask0 for s in steps]).astype(bool)
    mask1 = np.stack([s.mask1 for s in steps]).astype(bool)
    a0 = np.asarray([s.a0 for s in steps], dtype=np.int16)
    a1 = np.asarray([s.a1 for s in steps], dtype=np.int16)
    logp = np.asarray([s.logp for s in steps], dtype=np.float32)
    value = np.asarray([s.value for s in steps], dtype=np.float32)
    reward = np.asarray([s.reward for s in steps], dtype=np.float32)
    periods = np.asarray([s.periods for s in steps], dtype=np.int16)
    terminal = np.asarray([s.terminal for s in steps], dtype=bool)

    probs = np.empty((len(steps), len(ACTIONS), len(ACTIONS)), dtype=np.float32)
    logp_matrix = np.empty_like(probs)
    max_logp_error = 0.0
    for i, step in enumerate(steps):
        p, lp = joint_probabilities(policy, step.obs, step.mask0, step.mask1)
        probs[i] = p
        logp_matrix[i] = lp
        err = abs(float(lp[step.a0, step.a1]) - float(step.logp))
        max_logp_error = max(max_logp_error, err)
        if abs(float(p.sum()) - 1.0) > 1e-5:
            raise RuntimeError(f'joint probability sum {p.sum()} at decision {i}')
        if not np.isfinite(p).all() or (p < -1e-8).any():
            raise RuntimeError(f'nonfinite/negative joint probability at decision {i}')
        if (p[~policy.legal_pairs.numpy()] > 1e-8).any():
            raise RuntimeError(f'probability mass on illegal joint pair at decision {i}')
    if max_logp_error > 2e-5:
        raise RuntimeError(f'old logp recompute error {max_logp_error} > 2e-5')

    # Group 5 Hz execution samples by actor decision.
    samples = summary['execution_samples']
    groups = []
    si = 0
    for decision in summary['actions']:
        end = float(decision['time'])
        group = []
        while si < len(samples) and float(samples[si]['time']) <= end + 1e-9:
            group.append(samples[si])
            si += 1
        groups.append(group)
    if len(groups) != len(steps) or si != len(samples):
        raise RuntimeError('execution sample grouping mismatch')

    episode_id = f'{model}__{root}__{mode}__{seed if seed >= 0 else "greedy"}'
    decision_rows = []
    starts = []
    previous = 0.0
    for i, (step, decision, group) in enumerate(zip(steps, summary['actions'], groups)):
        start = previous
        end = float(decision['time'])
        starts.append((start, end))
        previous = end
        probs_i = probs[i]
        p_same = float(probs_i[SAME_PUSH].sum())
        p_same_speed = float(probs_i[SAME_SPEED].sum())
        p_push_repo = float(probs_i[PUSH_REPO].sum())
        p_both_brake = float(probs_i[BOTH_BRAKE].sum())
        p_other = float(1.0 - p_same - p_push_repo - p_both_brake)
        phases = decision.get('phases') or []
        exits = decision.get('exits') or []
        seconds = Counter()
        first_both_push_delay = None
        for sample in group:
            dt = float(sample['dt'])
            phase = sample.get('phases') or []
            if all(p == 'push' for p in phase):
                seconds['both_push'] += dt
            elif any(p == 'push' for p in phase):
                seconds['one_push'] += dt
            elif any(p in ('peer_wait', 'perception_wait') for p in phase):
                seconds['wait'] += dt
            elif any(p == 'blocked' for p in phase):
                seconds['blocked'] += dt
            elif any(p == 'switching' for p in phase):
                seconds['switching'] += dt
            elif any(p in ('find', 'approach', 'ready', 'align', 'move') for p in phase):
                seconds['pre_contact'] += dt
            elif any(p == 'brake' for p in phase):
                seconds['brake'] += dt
            else:
                seconds['other'] += dt
            if first_both_push_delay is None and all(p == 'push' for p in phase):
                delay = float(sample['time']) - float(sample['dt']) - start
                first_both_push_delay = max(0.0, delay)
        actual_seconds = float(sum(float(s['dt']) for s in group))
        decision_rows.append(dict(
            episode_id=episode_id, model=model, root=root,
            task=('dual' if root.startswith('dual') else 'single' if root.startswith(('contact', 'approach')) else 'reposition'),
            mode=mode, seed=seed, decision_index=i, t_start=start, t_end=end,
            dt=float(step.periods) * spec.CONTROL_DT, periods=int(step.periods),
            a0=int(step.a0), a1=int(step.a1),
            label0=ACTIONS[step.a0].label, label1=ACTIONS[step.a1].label,
            kind0=ACTIONS[step.a0].kind, kind1=ACTIONS[step.a1].kind,
            type0=int(ACTIONS[step.a0].type_index), type1=int(ACTIONS[step.a1].type_index),
            scale0=int(getattr(ACTIONS[step.a0], 'scale_index', -1)),
            scale1=int(getattr(ACTIONS[step.a1], 'scale_index', -1)),
            speed0=float(getattr(ACTIONS[step.a0], 'speed', 0.)),
            speed1=float(getattr(ACTIONS[step.a1], 'speed', 0.)),
            same_push=_same_push(step.a0, step.a1),
            same_speed=_same_speed(step.a0, step.a1),
            push_reposition=_push_reposition(step.a0, step.a1),
            both_brake=_both_brake(step.a0, step.a1),
            p_same_push=p_same, p_same_speed=p_same_speed,
            p_push_reposition=p_push_repo, p_both_brake=p_both_brake, p_other=p_other,
            p_selected=float(probs_i[step.a0, step.a1]),
            logp=float(step.logp), value=float(step.value), reward=float(step.reward),
            terminal=bool(step.terminal),
            phase0=phases[0] if len(phases) > 0 else None,
            phase1=phases[1] if len(phases) > 1 else None,
            exit0=exits[0] if len(exits) > 0 else None,
            exit1=exits[1] if len(exits) > 1 else None,
            events=';'.join(decision.get('events') or []),
            actual_seconds=actual_seconds,
            both_push_seconds=float(seconds['both_push']),
            one_push_seconds=float(seconds['one_push']),
            wait_seconds=float(seconds['wait']),
            blocked_seconds=float(seconds['blocked']),
            switching_seconds=float(seconds['switching']),
            pre_contact_seconds=float(seconds['pre_contact']),
            brake_seconds=float(seconds['brake']),
            other_seconds=float(seconds['other']),
            first_both_push_delay=first_both_push_delay,
            max_logp_error=max_logp_error,
        ))

    # Continuous selected same-push runs.
    runs = []
    current = None
    for row in decision_rows:
        if row['same_push']:
            if current is None:
                current = dict(start=row['decision_index'], end=row['decision_index'],
                               length=0, seconds=0.0, same_speed=0, direction=row['type0'],
                               speed0=row['speed0'], speed1=row['speed1'])
            current['end'] = row['decision_index']
            current['length'] += 1
            current['seconds'] += row['dt']
            current['same_speed'] += int(row['same_speed'])
        else:
            if current is not None:
                runs.append(current)
                current = None
    if current is not None:
        runs.append(current)
    for j, run in enumerate(runs):
        run.update(episode_id=episode_id, model=model, root=root,
                   task=decision_rows[0]['task'] if decision_rows else None,
                   mode=mode, seed=seed, run_id=j)

    success = bool(summary['success'])
    entered_goal = bool(trace.entered_goal)
    episode = dict(
        episode_id=episode_id, model=model, root=root, task=decision_rows[0]['task'] if decision_rows else None,
        mode=mode, seed=seed, success=success, return_=float(summary['return']),
        seconds=float(summary['seconds']), initial_distance=float(summary['initial_distance']),
        final_distance=float(summary['final_distance']), min_distance=float(trace.min_distance),
        goal=list(instance.goal_centre), radius=float(instance.goal_radius), arena=list(instance.arena),
        entered_goal=entered_goal, entered_goal_not_success=bool(entered_goal and not success),
        anomaly=bool(not summary['eligible_for_ppo']), termination=summary['termination'],
        decisions=int(summary['decisions']), actor_transitions=len(steps),
        simulated_seconds=float(summary['seconds']),
        same_push_prob_mean=float(np.mean([r['p_same_push'] for r in decision_rows])) if decision_rows else None,
        same_push_selected=int(sum(r['same_push'] for r in decision_rows)),
        same_push_selected_seconds=float(sum(r['dt'] for r in decision_rows if r['same_push'])),
        same_speed_selected=int(sum(r['same_speed'] for r in decision_rows)),
        push_reposition_selected=int(sum(r['push_reposition'] for r in decision_rows)),
        both_brake_selected=int(sum(r['both_brake'] for r in decision_rows)),
        both_push_seconds=float(sum(r['both_push_seconds'] for r in decision_rows)),
        one_push_seconds=float(sum(r['one_push_seconds'] for r in decision_rows)),
        wait_seconds=float(sum(r['wait_seconds'] for r in decision_rows)),
        blocked_seconds=float(sum(r['blocked_seconds'] for r in decision_rows)),
        switching_seconds=float(sum(r['switching_seconds'] for r in decision_rows)),
        pre_contact_seconds=float(sum(r['pre_contact_seconds'] for r in decision_rows)),
        brake_seconds=float(sum(r['brake_seconds'] for r in decision_rows)),
        other_seconds=float(sum(r['other_seconds'] for r in decision_rows)),
        same_push_both_push_seconds=float(sum(r['both_push_seconds'] for r in decision_rows if r['same_push'])),
        same_push_wait_seconds=float(sum(r['wait_seconds'] for r in decision_rows if r['same_push'])),
        same_push_delay_mean=float(np.mean([r['first_both_push_delay'] for r in decision_rows
                                            if r['same_push'] and r['first_both_push_delay'] is not None]))
        if any(r['same_push'] and r['first_both_push_delay'] is not None for r in decision_rows) else None,
        same_push_delay_count=int(sum(1 for r in decision_rows
                                      if r['same_push'] and r['first_both_push_delay'] is not None)),
        max_logp_error=max_logp_error,
        npz=str(out / 'episodes' / f'{episode_id}.npz'),
        summary_json=str(out / 'episodes' / f'{episode_id}.json'),
    )

    (out / 'episodes').mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / 'episodes' / f'{episode_id}.npz',
        obs=encoded, raw_obs=raw, mask0=mask0, mask1=mask1,
        a0=a0, a1=a1, logp=logp, value=value, reward=reward,
        periods=periods, terminal=terminal, probs=probs, logp_matrix=logp_matrix,
        trace_t=np.asarray([f['t'] for f in trace.frames], dtype=np.float32),
        trace_object=np.asarray([f['o'] for f in trace.frames], dtype=np.float32),
        trace_robots=np.asarray([f['r'] for f in trace.frames], dtype=np.float32),
        trace_contact=np.asarray([f['ct'] for f in trace.frames], dtype=bool),
        trace_stage=np.asarray([f['st'] for f in trace.frames], dtype='<U16'),
        trace_force=np.asarray([f['fo'] for f in trace.frames], dtype=np.float32),
        trace_distance=np.asarray([f['distance'] for f in trace.frames], dtype=np.float32),
    )
    atomic_json(out / 'episodes' / f'{episode_id}.json',
                dict(episode=episode, decisions=decision_rows, runs=runs,
                     actions=summary['actions'], execution_samples=summary['execution_samples'],
                     components=dict(summary['components']), skill_exits=dict(summary['skill_exits']),
                     initial_geometry=summary.get('initial_geometry')))
    return dict(episode=episode, decisions=decision_rows, runs=runs)


# ---------------------------------------------------------------------------
# main-side analysis
# ---------------------------------------------------------------------------

def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_npz_episode(path):
    data = np.load(path, allow_pickle=False)
    steps = []
    for i in range(len(data['a0'])):
        steps.append(Step(
            obs=data['obs'][i], a0=int(data['a0'][i]), a1=int(data['a1'][i]),
            logp=float(data['logp'][i]), value=float(data['value'][i]),
            reward=float(data['reward'][i]), mask0=data['mask0'][i], mask1=data['mask1'][i],
            terminal=bool(data['terminal'][i]), periods=int(data['periods'][i]),
        ))
    return data, steps


def _quantiles(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {}
    q = np.quantile(values, [0., .1, .25, .5, .75, .9, 1.])
    return dict(mean=float(values.mean()), q0=float(q[0]), q10=float(q[1]), q25=float(q[2]),
                q50=float(q[3]), q75=float(q[4]), q90=float(q[5]), q100=float(q[6]),
                positive_fraction=float((values > 0).mean()), n=int(len(values)))


def _group_grads(grads, params, groups):
    out = {}
    for group, selected in groups.items():
        values = [g.detach().clone() if g is not None else torch.zeros_like(p)
                  for p, g in zip(params, grads) if p in selected]
        out[group] = values
    return out


def _flat_norm(values):
    total = 0.0
    for value in values:
        total += float(value.double().square().sum())
    return math.sqrt(total)


def _flat_dot(a, b):
    total = 0.0
    for x, y in zip(a, b):
        total += float((x.double() * y.double()).sum())
    return total


def _cos(a, b):
    na, nb = _flat_norm(a), _flat_norm(b)
    if na == 0 or nb == 0:
        return None
    return _flat_dot(a, b) / (na * nb)


def _grad_dict(grads, params, groups):
    return _group_grads(grads, params, groups)


def batch_analysis(joint_policy, joint_data, episodes, episode_meta, batch_seed, out):
    """Reconstruct one 8-episode PPO batch and its first-order actor gradients."""
    ids = []
    for root in PANEL_ORDER:
        mode = 'random'
        model = 'joint_types'
        episode_id = f'{model}__{root}__{mode}__{batch_seed}'
        ids.append(episode_id)
    missing = [i for i in ids if i not in episodes]
    if missing:
        return dict(batch=batch_seed, status='incomplete', missing=missing,
                    reason='missing episode', episodes=ids)
    valid_ids = []
    invalid = []
    for i in ids:
        if episode_meta[i]['anomaly']:
            invalid.append(i)
        else:
            valid_ids.append(i)
    if invalid:
        status = 'incomplete'
    else:
        status = 'complete'
    all_steps = []
    step_task = []
    step_category = []
    step_success = []
    step_episode = []
    episode_records = []
    for i in valid_ids:
        meta = episode_meta[i]
        _, steps = _load_npz_episode(episodes[i])
        all_steps.extend(steps)
        step_task.extend([meta['task']] * len(steps))
        step_success.extend([bool(meta['success'])] * len(steps))
        step_episode.extend([i] * len(steps))
        for step in steps:
            if _same_push(step.a0, step.a1):
                step_category.append('same_push')
            elif _push_reposition(step.a0, step.a1):
                step_category.append('push_reposition')
            elif _both_brake(step.a0, step.a1):
                step_category.append('both_brake')
            else:
                step_category.append('other')
        episode_records.append(dict(episode_id=i, task=meta['task'], success=bool(meta['success']),
                                    seconds=float(meta['seconds']), transitions=len(steps)))
    if not all_steps:
        return dict(batch=batch_seed, status='incomplete', reason='no valid steps',
                    episodes=ids, invalid=invalid)
    n = len(all_steps)
    raw_adv = np.zeros(n, dtype=np.float64)
    targets = np.zeros(n, dtype=np.float64)
    rtg = np.zeros(n, dtype=np.float64)
    values = np.zeros(n, dtype=np.float64)
    offset = 0
    for i in valid_ids:
        meta = episode_meta[i]
        _, steps = _load_npz_episode(episodes[i])
        t, a = returns_and_advantages(steps, method='gae', trace_seconds=TRACE_SECONDS)
        raw_adv[offset:offset + len(steps)] = a
        targets[offset:offset + len(steps)] = t
        running = 0.0
        for j in range(len(steps) - 1, -1, -1):
            running += float(steps[j].reward)
            rtg[offset + j] = running
            values[offset + j] = float(steps[j].value)
        offset += len(steps)
    if raw_adv.std() > 1e-8:
        norm_adv = (raw_adv - raw_adv.mean()) / (raw_adv.std() + 1e-8)
    else:
        norm_adv = raw_adv.copy()
    obs = torch.tensor(np.stack([s.obs for s in all_steps]))
    masks = (torch.tensor(np.stack([s.mask0 for s in all_steps])),
             torch.tensor(np.stack([s.mask1 for s in all_steps])))
    a0 = torch.tensor([s.a0 for s in all_steps])
    a1 = torch.tensor([s.a1 for s in all_steps])
    old_logp = torch.tensor([s.logp for s in all_steps])
    adv = torch.tensor(norm_adv, dtype=torch.float32)
    with torch.no_grad():
        logp_check, _ = batch_logp_entropy(joint_policy, obs, masks, a0, a1)
        logp_error = float((logp_check - old_logp).abs().max())
        if logp_error > 2e-5:
            raise RuntimeError(f'batch old logp error {logp_error} > 2e-5')
    logp, entropy = batch_logp_entropy(joint_policy, obs, masks, a0, a1)
    task_array = np.asarray(step_task)
    category_array = np.asarray(step_category)
    success_array = np.asarray(step_success)
    task_masks = {name: torch.tensor(task_array == name) for name in TASK_NAMES if name != 'all'}
    actor_params = list(joint_policy.actor_parameters())
    body_params = set(joint_policy.body.parameters())
    base_params = set(joint_policy.types.parameters()) | set(joint_policy.scales.parameters()) | {joint_policy.persistence_bias}
    joint_params = set(joint_policy.joint_types.parameters()) if joint_policy.model_config.get('coupled') else set()
    groups = dict(body=body_params, base_heads=base_params, joint_types=joint_params,
                  all=set(actor_params))
    covered = set().union(*groups.values())
    if covered != set(actor_params):
        raise RuntimeError('actor parameter groups do not cover all actor parameters')

    L_actor = -(adv * logp).sum() / n
    L_entropy = -ENTROPY_COEF * entropy.sum() / n
    L_full = L_actor + L_entropy
    tasks = {}
    for name in ('dual', 'single', 'reposition'):
        tasks[name] = -(adv * logp * task_masks[name]).sum() / n
    L_other = tasks['single'] + tasks['reposition']
    L_recomb = abs(float(L_full.detach()) - float((L_actor + L_entropy).detach()))
    L_task_recomb = abs(float(L_actor.detach()) - float((tasks['dual'] + L_other).detach()))

    def grad_loss(loss):
        grads = torch.autograd.grad(loss, actor_params, retain_graph=True, allow_unused=True)
        return _grad_dict(grads, actor_params, groups)

    g_full = grad_loss(L_full)
    g_actor = grad_loss(L_actor)
    g_entropy = grad_loss(L_entropy)
    g_dual = grad_loss(tasks['dual'])
    g_single = grad_loss(tasks['single'])
    g_reposition = grad_loss(tasks['reposition'])
    g_other = {key: [a + b for a, b in zip(g_single[key], g_reposition[key])] for key in GROUPS}
    g_task_sum = {key: [a + b + c for a, b, c in zip(g_dual[key], g_single[key], g_reposition[key])] for key in GROUPS}
    g_recombined = {key: [a + b for a, b in zip(g_actor[key], g_entropy[key])] for key in GROUPS}
    def max_abs_diff(a, b):
        return max((float((x - y).abs().max()) for x, y in zip(a, b)), default=0.)

    recomb = {}
    for key in GROUPS:
        recomb[key] = max(max_abs_diff(g_task_sum[key], g_actor[key]),
                          max_abs_diff(g_recombined[key], g_full[key]))
    grad_norms = {}
    cosines = {}
    for key in GROUPS:
        grad_norms[key] = dict(
            full=_flat_norm(g_full[key]), actor=_flat_norm(g_actor[key]), entropy=_flat_norm(g_entropy[key]),
            dual=_flat_norm(g_dual[key]), single=_flat_norm(g_single[key]),
            reposition=_flat_norm(g_reposition[key]), other=_flat_norm(g_other[key]))
        cosines[key] = dict(
            dual_other=_cos(g_dual[key], g_other[key]),
            dual_entropy=_cos(g_dual[key], g_entropy[key]),
            other_entropy=_cos(g_other[key], g_entropy[key]),
        )
    # Advantage / reward-to-go summaries.
    adv_rows = []
    for task in ('dual', 'single', 'reposition', 'all'):
        tmask = np.ones(n, dtype=bool) if task == 'all' else (task_array == task)
        for category in CATEGORY_NAMES:
            cmask = np.ones(n, dtype=bool) if category == 'all' else (category_array == category)
            for success in (True, False, None):
                mask = tmask & cmask
                if success is not None:
                    mask &= (success_array == success)
                idx = np.flatnonzero(mask)
                if len(idx) == 0:
                    continue
                adv_rows.append(dict(
                    batch=batch_seed, task=task, category=category,
                    success='all' if success is None else bool(success), n=int(len(idx)),
                    raw_adv=_quantiles(raw_adv[idx]), norm_adv=_quantiles(norm_adv[idx]),
                    return_to_go=_quantiles(rtg[idx]),
                    value=_quantiles(values[idx]),
                    value_bias=_quantiles(rtg[idx] - values[idx]),
                ))
    # Return-to-go is already computed per episode above; do not sum across episodes.
    episode_rows = []
    for record in episode_records:
        episode_rows.append(record)
    summary = dict(
        batch=batch_seed, status=status, invalid=invalid, episodes=ids,
        n_steps=n, n_valid_episodes=len(valid_ids),
        task_counts={name: int((task_array == name).sum()) for name in ('dual', 'single', 'reposition')},
        category_counts={name: int((category_array == name).sum()) for name in CATEGORY_NAMES if name != 'all'},
        task_transitions={name: int((task_array == name).sum()) for name in ('dual', 'single', 'reposition')},
        task_episodes={name: int(sum(1 for m in episode_records if m['task'] == name)) for name in ('dual', 'single', 'reposition')},
        task_seconds={name: float(sum(m['seconds'] for m in episode_records if m['task'] == name)) for name in ('dual', 'single', 'reposition')},
        logp_error=logp_error, advantage_mean=float(norm_adv.mean()), advantage_std=float(norm_adv.std()),
        L_actor=float(L_actor.detach()), L_entropy=float(L_entropy.detach()), L_full=float(L_full.detach()),
        L_tasks={k: float(v.detach()) for k, v in tasks.items()}, L_other=float(L_other.detach()),
        L_recombination_error=L_recomb, L_task_recombination_error=L_task_recomb,
        grad_norms=grad_norms, cosines=cosines, gradient_recombination_error=recomb,
        episode_records=episode_records,
    )
    return dict(summary=summary, advantage_rows=adv_rows, g=g_full, g_actor=g_actor,
                g_entropy=g_entropy, g_dual=g_dual, g_single=g_single,
                g_reposition=g_reposition, g_other=g_other,
                actor_params=actor_params, groups=groups)


def _return_to_go(steps, index):
    total = 0.0
    for i in range(index, len(steps)):
        total += float(steps[i].reward)
    return total


def same_push_gradient(joint_policy, eval_inputs, batch_grads):
    """Gradient of mean log P(same-direction dual push) on fixed inputs."""
    actor_params = list(joint_policy.actor_parameters())
    body_params = set(joint_policy.body.parameters())
    base_params = set(joint_policy.types.parameters()) | set(joint_policy.scales.parameters()) | {joint_policy.persistence_bias}
    joint_params = set(joint_policy.joint_types.parameters()) if joint_policy.model_config.get('coupled') else set()
    groups = dict(body=body_params, base_heads=base_params, joint_types=joint_params, all=set(actor_params))
    valid = []
    for item in eval_inputs:
        logp = same_push_logprob(joint_policy, item['obs'], item['mask0'], item['mask1'])
        if logp is not None and bool(torch.isfinite(logp)):
            valid.append(logp)
    if not valid:
        return dict(status='no_valid_inputs', inputs=len(eval_inputs))
    logp_mean = torch.stack(valid).mean()
    p_values = [float(torch.exp(v).detach().item()) for v in valid]
    grads = torch.autograd.grad(logp_mean, actor_params, retain_graph=True, allow_unused=True)
    g_logp = _grad_dict(grads, actor_params, groups)
    rows = []
    for key in GROUPS:
        rows.append(dict(group=key, grad_norm=_flat_norm(g_logp[key])))
    dots = {}
    for name, grads_by_group in batch_grads.items():
        dots[name] = {}
        for group in GROUPS:
            dots[name][group] = -_flat_dot(g_logp[group], grads_by_group[group]) if grads_by_group else None
    # `dots` is already -<loss grad, logP grad>: positive means plain gradient
    # descent on that loss increases log P(same push) to first order.
    return dict(status='ok', n_inputs=len(valid), p_mean=float(np.mean(p_values)),
                p_geometric_mean=float(np.exp(np.mean([math.log(max(p, 1e-300)) for p in p_values]))),
                logp_mean=float(logp_mean.detach()), grad_norms=rows, first_order=dots)


def _preflight_initial_actions(checkpoints, panel_path, expected_json):
    train, _ = load_panel(panel_path)
    by_name = {i.name: i for i in train}
    report = {}
    for model, checkpoint in checkpoints.items():
        policy, normalizer, _ = load_checkpoint(checkpoint)
        policy.eval()
        report[model] = {}
        for root in DUAL_ROOTS:
            env = make_env(by_name[root], context='extended')
            obs, m0, m1 = observe(env)
            encoded = normalizer.normalise(obs)
            encoded[760:] = obs[760:]
            with torch.no_grad():
                actions, logp, value = policy.act(torch.tensor(encoded)[None, :], (m0, m1), deterministic=True)
            a0, a1 = int(actions[0]), int(actions[1])
            expected = None
            if expected_json.exists():
                data = json.loads(expected_json.read_text())
                for scene in data.get('checkpoints', {}).get(model, {}).get('scenes', []):
                    if scene.get('name') == root:
                        expected = dict(a0=scene['greedy_a0']['action'],
                                        a1=scene['conditional_given_greedy'][0]['action'])
            report[model][root] = dict(a0=a0, a1=a1, label0=ACTIONS[a0].label, label1=ACTIONS[a1].label,
                                       expected=expected, matches=(expected is None or
                                                                    (expected['a0'] == a0 and expected['a1'] == a1)))
    return report


def _gae_isolation_check():
    # Two synthetic episodes with terminal flags; concatenated GAE must reset at terminals.
    def mk(value, reward, terminal, periods=16):
        return Step(obs=np.zeros(944, dtype=np.float32), a0=44, a1=44, logp=0., value=value,
                    reward=reward, mask0=np.zeros(len(ACTIONS), dtype=bool),
                    mask1=np.zeros(len(ACTIONS), dtype=bool), terminal=terminal, periods=periods)
    e1 = [mk(1., 1., False), mk(2., 2., True)]
    e2 = [mk(3., 3., False), mk(4., 4., True)]
    t_all, a_all = returns_and_advantages(e1 + e2, method='gae', trace_seconds=TRACE_SECONDS)
    t1, a1 = returns_and_advantages(e1, method='gae', trace_seconds=TRACE_SECONDS)
    t2, a2 = returns_and_advantages(e2, method='gae', trace_seconds=TRACE_SECONDS)
    err = max(abs(float(x - y)) for x, y in zip(a_all, np.concatenate([a1, a2])))
    err = max(err, max(abs(float(x - y)) for x, y in zip(t_all, np.concatenate([t1, t2]))))
    return dict(max_error=float(err), ok=bool(err < 1e-6))


def _make_replay(out, selected_ids, episodes, episode_meta):
    """Build a self-contained HTML replay from recorded true-state traces."""
    from experiments.option_rl_v45.render_episodes import TEMPLATE
    payload_episodes = []
    for episode_id in selected_ids:
        meta = episode_meta[episode_id]
        npz = np.load(episodes[episode_id], allow_pickle=False)
        summary_path = Path(meta['summary_json'])
        summary = json.loads(summary_path.read_text())
        decisions = []
        start = 0.0
        for d in summary.get('actions', []):
            decisions.append(dict(t=start, a0=d['labels'][0], a1=d['labels'][1]))
            start = float(d['time'])
        frames = []
        for i in range(len(npz['trace_t'])):
            frames.append(dict(
                t=float(npz['trace_t'][i]),
                o=[float(v) for v in npz['trace_object'][i]],
                r=[[float(v) for v in row] for row in npz['trace_robots'][i]],
                ct=[bool(v) for v in npz['trace_contact'][i]],
                st=[str(v) for v in npz['trace_stage'][i]],
                fo=[float(v) for v in npz['trace_force'][i]],
            ))
        payload_episodes.append(dict(
            name=meta['episode_id'], goal=meta.get('goal'),
            radius=float(meta.get('radius', spec.GOAL_RADIUS)),
            object_size=[spec.OBJECT_WIDTH, spec.OBJECT_HEIGHT],
            start_goal_distance=round(float(meta['initial_distance']), 3),
            frames=frames, decisions=decisions,
            success=bool(meta['success']), termination=meta['termination'],
            summary=dict(return_=meta['return_'], seconds=meta['seconds'],
                         final_distance=meta['final_distance'], min_distance=meta['min_distance'],
                         termination=meta['termination']),
        ))
    # The D001 roots all use the panel's 5x4 arena; keep the arena from the run.
    arena = list(episode_meta[selected_ids[0]].get('arena', [5.0, 4.0])) if selected_ids else [5.0, 4.0]
    payload = dict(checkpoint='D001 frozen v15 | conditional greedy / random 16 | offline true-state replay',
                   arena=arena, episodes=payload_episodes)
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c'))
    html = html.replace('ARENA[1]', 'DATA.arena[1]').replace('ARENA[0]', 'DATA.arena[0]')
    path = Path(out) / 'replay.html'
    path.write_text(html)
    (Path(out) / 'replay_payload.json').write_text(json.dumps(payload, indent=1))
    return path


def _write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows:
        Path(path).write_text('')
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--independent', type=Path, required=True)
    parser.add_argument('--joint', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--deadline-seconds', type=float, default=900.)
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f'output directory must be new/empty: {out}')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'episodes').mkdir(exist_ok=True)
    started_at = time.time()
    started_monotonic = time.monotonic()
    checkpoints = dict(independent=args.independent.resolve(), joint_types=args.joint.resolve())
    input_hashes = dict(panel=_sha256(args.panel), **{k: _sha256(v) for k, v in checkpoints.items()})
    for key, expected in EXPECTED.items():
        actual = input_hashes['panel'] if key == 'panel' else input_hashes[key]
        if actual != expected:
            raise SystemExit(f'{key} SHA256 mismatch: {actual} != {expected}')

    # Preflight: initial greedy action reproduction and joint probability checks.
    expected_json = ROOT / 'outputs/option_rl_v15/greedy_coordination_001/summary.json'
    initial = _preflight_initial_actions(checkpoints, args.panel, expected_json)
    gae = _gae_isolation_check()
    preflight = dict(initial=initial, gae_isolation=gae, input_hashes=input_hashes,
                     expected_json=str(expected_json), expected_json_exists=expected_json.exists())
    atomic_json(out / 'preflight.json', preflight)
    if not gae['ok']:
        raise SystemExit('GAE termination isolation check failed')
    mismatch = [f'{m}/{r}' for m, roots in initial.items() for r, v in roots.items() if not v['matches']]
    if mismatch:
        raise SystemExit('initial greedy action mismatch: ' + ', '.join(mismatch))
    print(json.dumps(preflight, indent=1), flush=True)
    if args.self_check:
        print('self-check complete; no episodes run', flush=True)
        return

    # Official bounded run starts here.
    run_started = time.monotonic()
    jobs = []
    for model in ('independent', 'joint_types'):
        for root in DUAL_ROOTS:
            jobs.append(dict(model=model, checkpoint=str(checkpoints[model]), panel=str(args.panel),
                             root=root, mode='greedy', seed=-1, out=str(out)))
            for seed in RANDOM_SEEDS:
                jobs.append(dict(model=model, checkpoint=str(checkpoints[model]), panel=str(args.panel),
                                 root=root, mode='random', seed=seed, out=str(out)))
    for root in OTHER_ROOTS:
        for seed in BATCH_SEEDS:
            jobs.append(dict(model='joint_types', checkpoint=str(checkpoints['joint_types']),
                             panel=str(args.panel), root=root, mode='random', seed=seed, out=str(out)))
    if len(jobs) != 92:
        raise RuntimeError(f'job count {len(jobs)} != 92')

    episode_rows = []
    decision_rows = []
    run_rows = []
    errors = []
    incomplete_reason = None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=__import__('multiprocessing').get_context('spawn'),
                             initializer=_worker_init) as executor:
        futures = {executor.submit(run_job, job): job for job in jobs}
        pending = set(futures)
        while pending:
            remaining = args.deadline_seconds - (time.monotonic() - run_started)
            if remaining <= 0:
                incomplete_reason = 'wall_deadline_reached'
                for future in pending:
                    future.cancel()
                break
            done, pending = wait(pending, timeout=min(30.0, remaining), return_when=FIRST_COMPLETED)
            for future in done:
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # keep other completed work; no retry
                    errors.append(dict(job=job, error=repr(exc)))
                    print(json.dumps(dict(error=repr(exc), job=job), flush=True), file=sys.stderr)
                    continue
                episode_rows.append(result['episode'])
                decision_rows.extend(result['decisions'])
                run_rows.extend(result['runs'])
                print(json.dumps(dict(episode=result['episode']['episode_id'],
                                      success=result['episode']['success'],
                                      seconds=round(result['episode']['seconds'], 1),
                                      decisions=result['episode']['decisions'])), flush=True)
        else:
            pass
    actual_seconds = time.monotonic() - run_started
    _write_csv(out / 'episode_metrics.csv', episode_rows)
    _write_csv(out / 'decision_metrics.csv', decision_rows)
    _write_csv(out / 'same_push_runs.csv', run_rows)
    episode_meta = {e['episode_id']: e for e in episode_rows}

    # A. frozen policy: greedy vs random on the two dual roots.
    a_summary = {}
    for model in ('independent', 'joint_types'):
        for root in DUAL_ROOTS:
            greedy = [e for e in episode_rows if e['model'] == model and e['root'] == root and e['mode'] == 'greedy']
            random = [e for e in episode_rows if e['model'] == model and e['root'] == root and e['mode'] == 'random']
            a_summary[f'{model}/{root}'] = dict(
                greedy=dict(n=len(greedy), success=int(sum(e['success'] for e in greedy)),
                            return_=float(np.mean([e['return_'] for e in greedy])) if greedy else None,
                            seconds=float(np.mean([e['seconds'] for e in greedy])) if greedy else None,
                            min_distance=float(np.mean([e['min_distance'] for e in greedy])) if greedy else None),
                random=dict(n=len(random), success=int(sum(e['success'] for e in random)),
                            return_=float(np.mean([e['return_'] for e in random])) if random else None,
                            seconds=float(np.mean([e['seconds'] for e in random])) if random else None,
                            min_distance=float(np.mean([e['min_distance'] for e in random])) if random else None,
                            entered_goal_not_success=int(sum(e['entered_goal_not_success'] for e in random)),
                            anomalies=int(sum(e['anomaly'] for e in random)),
                            same_push_prob_mean=float(np.mean([e['same_push_prob_mean'] for e in random])) if random else None,
                            same_push_selected=int(sum(e['same_push_selected'] for e in random)),
                            same_push_selected_seconds=float(sum(e['same_push_selected_seconds'] for e in random)),
                            same_push_both_push_seconds=float(sum(e['same_push_both_push_seconds'] for e in random)),
                            same_push_wait_seconds=float(sum(e['same_push_wait_seconds'] for e in random))),
            )

    # B. four 8-episode batches from the joint model.
    joint_policy, joint_normalizer, joint_data = load_checkpoint(args.joint)
    joint_policy.eval()
    batch_results = []
    advantage_rows = []
    batch_grads = {}
    batch_summaries = []
    for batch_seed in BATCH_SEEDS:
        result = batch_analysis(joint_policy, joint_data, {e['episode_id']: e['npz'] for e in episode_rows},
                                episode_meta, batch_seed, out)
        if 'summary' in result:
            batch_summaries.append(result['summary'])
            advantage_rows.extend(result['advantage_rows'])
            batch_grads[f'batch_{batch_seed}'] = result['g']
        else:
            batch_summaries.append(result)
    _write_csv(out / 'advantage_summary.csv', advantage_rows)

    # Fixed evaluation inputs for log P(same push).
    eval_inputs = []
    for model in ('joint_types',):
        for root in DUAL_ROOTS:
            greedy_id = f'{model}__{root}__greedy__greedy'
            if greedy_id not in episode_meta:
                continue
            npz = np.load(episode_meta[greedy_id]['npz'], allow_pickle=False)
            for i in range(min(10, len(npz['a0']))):
                eval_inputs.append(dict(source=greedy_id, decision=i,
                                        obs=npz['obs'][i].copy(),
                                        mask0=npz['mask0'][i].copy(),
                                        mask1=npz['mask1'][i].copy()))
    # Deduplicate exact repeated observations; the first decision is the reset obs.
    unique = []
    seen = set()
    for item in eval_inputs:
        key = hashlib.sha256(item['obs'].tobytes()).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    eval_inputs = unique
    same_grad = same_push_gradient(joint_policy, eval_inputs, batch_grads) if eval_inputs else dict(status='no_inputs')
    atomic_json(out / 'same_push_gradient.json', same_grad)

    # Replay selection: two greedy failures plus first random success/failure per model/root.
    selected = []
    greedy_failures = [e['episode_id'] for e in episode_rows
                       if e['mode'] == 'greedy' and not e['success']]
    for episode_id in greedy_failures[:2]:
        if episode_id not in selected:
            selected.append(episode_id)
    for model in ('independent', 'joint_types'):
        for root in DUAL_ROOTS:
            random = [e for e in episode_rows if e['model'] == model and e['root'] == root and e['mode'] == 'random']
            first_success = next((e for e in random if e['success']), None)
            first_failure = next((e for e in random if not e['success']), None)
            for entry in (first_success, first_failure):
                if entry and entry['episode_id'] not in selected:
                    selected.append(entry['episode_id'])
    if selected:
        _make_replay(out, selected, {e['episode_id']: e['npz'] for e in episode_rows}, episode_meta)

    # Manifest and summary.
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(['git', 'status', '--porcelain'], cwd=ROOT, capture_output=True, text=True).stdout
    (out / 'git_status.txt').write_text(dirty)
    source_files = [Path(__file__), ROOT / 'experiments/option_rl_v16/train.py',
                    ROOT / 'experiments/option_rl_v16/interface.py', ROOT / 'experiments/option_rl_v15/interface.py',
                    ROOT / 'experiments/option_rl_v14/interface.py', ROOT / 'experiments/option_rl_v14/mode_env.py',
                    ROOT / 'experiments/option_rl_v13/controller.py', ROOT / 'experiments/option_rl_v13/mode_env.py',
                    ROOT / 'src/coimagine/random_start/env.py', ROOT / 'src/coimagine/random_start/spec.py']
    source_hashes = {str(p.relative_to(ROOT)): _sha256(p) for p in source_files if p.exists()}
    snapshot = out / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for p in source_files:
        if p.exists():
            dest = snapshot / str(p.relative_to(ROOT))
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    (out / 'source_hashes.json').write_text(json.dumps(source_hashes, indent=1))
    manifest = dict(
        task_version=TASK_VERSION, status='complete' if (not errors and not incomplete_reason) else 'incomplete',
        incomplete_reason=incomplete_reason, errors=errors, out=str(out),
        started_at=started_at, ended_at=time.time(), run_seconds=actual_seconds,
        wall_seconds_total=time.monotonic() - started_monotonic,
        workers=args.workers, deadline_seconds=args.deadline_seconds,
        jobs=len(jobs), completed_jobs=len(episode_rows), errors_count=len(errors),
        episode_budget=92, input_hashes=input_hashes, source_hashes=source_hashes,
        head=head, dirty_files=len(dirty.splitlines()),
        command=sys.argv, preflight=preflight,
    )
    atomic_json(out / 'manifest.json', manifest)
    summary = dict(
        task_version=TASK_VERSION, manifest=manifest, A=a_summary,
        B=dict(batches=batch_summaries, advantage_rows=len(advantage_rows)),
        same_push_gradient=same_grad, replay=selected,
    )
    atomic_json(out / 'summary.json', summary)
    print(json.dumps(dict(status=manifest['status'], completed=len(episode_rows),
                          run_seconds=round(actual_seconds, 1), errors=len(errors),
                          incomplete_reason=incomplete_reason), indent=1), flush=True)
    if errors or incomplete_reason:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
