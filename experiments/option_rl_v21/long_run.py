"""Bounded event-progress PPO run: wall-clock budget, telemetry, exact resume.

This module never executes on import. It reuses the verified event interface
(``event_env.py`` / ``interface.py``), the frozen v16 PPO update and the v14
legal dynamic scene generator; it adds only the long-run orchestration:
one fresh legal training root per episode, fixed validation/test panels, a
wall-clock budget, per-iteration history, dashboard, checkpoint and resume.

The reward, success criterion, controller, physics and action semantics are
unchanged. Training roots are new legal scenes from the same frozen generator
used by the v16 baseline; validation/test roots are disjoint fixed panels.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / 'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coimagine.random_start import layout, spec as R
from experiments.option_rl_v45.train import RunningMeanStd
from experiments.option_rl_v16 import scenes
from experiments.option_rl_v16.train import ppo_update
from experiments.option_rl_v21 import parallel
from experiments.option_rl_v21.event_env import EventEnv, REASONS
from experiments.option_rl_v21.diagnostics import REPORT_FIELDS, DEFINITION
from experiments.option_rl_v21.interface import Policy, metadata, validate_metadata
from experiments.option_rl_v21.monitor import write_dashboard
from experiments.option_rl_v21.train import run_episode
from experiments.option_rl_v15.live import atomic_json, read_json, status as live_status, stop_reason

EVENT_REASONS = REASONS
REWARD_KEYS = ('progress', 'distance_cost', 'time_cost', 'success_bonus', 'failure_penalty')
SOURCES = (
    'experiments/option_rl_v21/event_env.py',
    'experiments/option_rl_v21/interface.py',
    'experiments/option_rl_v21/train.py',
    'experiments/option_rl_v21/monitor.py',
    'experiments/option_rl_v21/long_run.py',
    'experiments/option_rl_v21/parallel.py',
    'experiments/option_rl_v21/supervise.py',
    'experiments/option_rl_v21/spec.py',
    'experiments/option_rl_v21/system.py',
    'experiments/option_rl_v21/yield_skill.py',
    'experiments/option_rl_v21/diagnostics.py',
    'experiments/option_rl_v19/unknown_shape.py',
    'experiments/option_rl_v19/spec.py',
    'experiments/option_rl_v19/system.py',
    'experiments/option_rl_v13/mode_env.py',
    'experiments/option_rl_v13/controller.py',
    'experiments/option_rl_v13/push_controller.py',
    'experiments/option_rl_v14/interface.py',
    'experiments/option_rl_v14/mode_env.py',
    'experiments/option_rl_v14/scenes.py',
    'experiments/option_rl_v14/learnability.py',
    'experiments/option_rl_v15/interface.py',
    'experiments/option_rl_v15/scenes.py',
    'experiments/option_rl_v15/live.py',
    'experiments/option_rl_v16/train.py',
    'experiments/option_rl_v16/learnability.py',
    'experiments/option_rl_v45/train.py',
    'experiments/option_rl_v45/monitor.py',
    'src/coimagine/random_start/env.py',
    'src/coimagine/random_start/spec.py',
    'src/coimagine/random_start/lidar.py',
    'src/coimagine/random_start/skills.py',
    'src/coimagine/random_start/boundary_follow.py',
    'src/coimagine/random_start/endpoint_tracking.py',
    'src/coimagine/random_start/effect_push.py',
    'src/coimagine/random_start/layout.py',
    'src/coimagine/random_start/geometry.py',
    'src/coimagine/random_start/anchor.py',
    'src/coimagine/random_start/peer_clearance.py',
    'src/coimagine/continuous2d/backend.py',
    'src/coimagine/continuous2d/environment.py',
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    """Freeze actual imported project dependencies as well as entry points."""
    names = set(SOURCES)
    for module in list(sys.modules.values()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        try:
            relative = str(path.relative_to(ROOT))
        except ValueError:
            continue
        if path.suffix == '.py' and relative.startswith(('src/coimagine/', 'experiments/')):
            names.add(relative)
    return tuple(sorted(names))


def aggregate(summaries):
    """Mean episode metrics plus event/action rates; every episode counts."""
    count = len(summaries)
    if count == 0:
        return {}
    reasons = Counter()
    actions = Counter()
    decisions = 0
    seconds = []
    for summary in summaries:
        reasons.update(summary['event_counts'])
        actions.update(summary['action_counts'])
        decisions += summary['decisions']
        seconds.extend(summary['decision_seconds'])

    def rate(key):
        return float(reasons.get(key, 0) / max(1, decisions))

    timeout = reasons.get('acquisition_timeout', 0) + reasons.get('progress_timeout', 0)
    result = dict(
        episodes=count,
        **{'return': float(np.mean([s['reward'] for s in summaries]))},
        success=float(np.mean([s['success'] for s in summaries])),
        progress=float(np.mean([s['progress'] for s in summaries])),
        final_distance=float(np.mean([s['final_distance'] for s in summaries])),
        initial_distance=float(np.mean([s['initial_distance'] for s in summaries])),
        decisions_mean=float(np.mean([s['decisions'] for s in summaries])),
        decision_seconds=float(np.mean(seconds)) if seconds else 0.,
        decision_p95_seconds=float(np.percentile(seconds, 95)) if seconds else 0.,
        contact_rate=float(np.mean([1. if s['contact'] else 0. for s in summaries])),
        action_push=float(actions.get('push', 0) / max(1, 2 * decisions)),
        action_reposition=float(actions.get('reposition', 0) / max(1, 2 * decisions)),
        action_wait=float(actions.get('wait', 0) / max(1, 2 * decisions)),
        action_yield=float(actions.get('yield', 0) / max(1, 2 * decisions)),
        event_progress_rate=rate('progress'),
        event_contact_rate=rate('contact'),
        event_contact_lost_rate=rate('contact_lost'),
        event_wait_rate=rate('wait'),
        event_failure_rate=rate('failure'),
        event_timeout_rate=float(timeout / max(1, decisions)),
        event_acquisition_timeout_rate=rate('acquisition_timeout'),
        event_progress_timeout_rate=rate('progress_timeout'),
        event_terminal_rate=rate('terminal'),
    )
    for key in REWARD_KEYS:
        result['reward_' + key] = float(np.mean([s['components'][key] for s in summaries]))
    for key in REPORT_FIELDS:
        result['diagnostic_' + key] = float(np.mean([s.get('diagnostics', {}).get(key, 0.) for s in summaries]))
    for key in ('yield_review', 'yield_blocked', 'direction_unavailable'):
        result['event_' + key + '_rate'] = rate(key)
    return result


def summarise_evaluation(summaries, instances, *, seconds, stochastic=False):
    """Aggregate a fixed-panel evaluation; incomplete runs are marked, not dropped."""
    if not summaries:
        return dict(complete=False, expected_roots=len(instances), completed_roots=0,
                    seconds=seconds, mode='stochastic' if stochastic else 'greedy')
    reason = Counter()
    actions = Counter()
    decisions = 0
    decision_seconds = []
    for summary in summaries:
        reason.update(summary['event_counts'])
        actions.update(summary['action_counts'])
        decisions += summary['decisions']
        decision_seconds.extend(summary['decision_seconds'])
    complete = len(summaries) == len(instances)
    outside = [s for s in summaries if s['initial_distance'] > R.GOAL_RADIUS]
    return dict(
        complete=complete, expected_roots=len(instances), completed_roots=len(summaries),
        seconds=seconds, mode='stochastic' if stochastic else 'greedy',
        success_rate=float(np.mean([s['success'] for s in summaries])),
        mean_progress=float(np.mean([s['progress'] for s in summaries])),
        mean_return=float(np.mean([s['reward'] for s in summaries])),
        mean_final_distance=float(np.mean([s['final_distance'] for s in summaries])),
        mean_decisions=float(np.mean([s['decisions'] for s in summaries])),
        mean_decision_seconds=float(np.mean(decision_seconds)) if decision_seconds else 0.,
        contact_rate=float(np.mean([1. if s['contact'] else 0. for s in summaries])),
        anomalies=int(sum(not s['eligible_for_ppo'] for s in summaries)),
        anomaly_rate=float(sum(not s['eligible_for_ppo'] for s in summaries) / len(summaries)),
        initially_inside_goal=len(summaries) - len(outside),
        outside_goal_success=float(np.mean([s['success'] for s in outside])) if outside and complete else None,
        event_rates={key: float(reason.get(key, 0) / max(1, decisions)) for key in EVENT_REASONS},
        action_mix={key: float(actions.get(key, 0) / max(1, 2 * decisions))
                    for key in ('push', 'reposition', 'wait', 'yield')},
        diagnostics={key: float(np.mean([s.get('diagnostics', {}).get(key, 0.) for s in summaries]))
                     for key in REPORT_FIELDS},
        episodes=summaries,
    )


def evaluate(policy, normalizer, instances, *, seed, stochastic=False, on_root=None, should_stop=None):
    """Serial evaluation on a fixed root list; the parallel path uses the same aggregation."""
    generator = torch.Generator().manual_seed(int(seed))
    summaries = []
    started = time.monotonic()
    for instance in instances:
        if should_stop is not None and should_stop():
            break
        _, summary = run_episode(policy, instance, normalizer, generator=generator,
                                 deterministic=not stochastic, update_stats=False,
                                 collect_observations=False)
        summaries.append(summary)
        if on_root is not None:
            on_root(len(summaries), len(instances), summary)
    return summarise_evaluation(summaries, instances, seconds=time.monotonic() - started,
                                stochastic=stochastic)


def save_state(path, *, policy, optimizers, normalizer, iteration, seed, history,
               ordinal, generator, best, config):
    data = dict(metadata=metadata(), policy=policy.state_dict(),
                optimizer={name: opt.state_dict() for name, opt in optimizers.items()},
                normalizer=dict(mean=normalizer.mean, var=normalizer.var, count=normalizer.count),
                obs_dim=policy.obs_dim, hidden=policy.hidden, iteration=iteration, seed=seed,
                history=history, next_scene_ordinal=ordinal,
                torch_rng=torch.get_rng_state(), sampling_rng=generator.get_state(),
                best=best, config=config)
    temp = path.with_name(path.name + '.tmp')
    torch.save(data, temp)
    temp.replace(path)


def load_state(path, *, lr):
    data = torch.load(path, map_location='cpu', weights_only=False)
    validate_metadata(data['metadata'])
    policy = Policy(data['obs_dim'], data['hidden'])
    policy.load_state_dict(data['policy'])
    normalizer = RunningMeanStd(data['obs_dim'])
    for key in ('mean', 'var', 'count'):
        setattr(normalizer, key, data['normalizer'][key])
    optimizers = dict(actor=torch.optim.Adam(policy.actor_parameters(), lr=lr),
                      critic=torch.optim.Adam(policy.critic_parameters(), lr=lr))
    if set(optimizers) != set(data['optimizer']):
        raise ValueError('optimizer schema mismatch')
    for name, opt in optimizers.items():
        opt.load_state_dict(data['optimizer'][name])
    torch.set_rng_state(data['torch_rng'])
    generator = torch.Generator()
    generator.set_state(data['sampling_rng'])
    return policy, normalizer, optimizers, generator, data


def config_of(args):
    return dict(seed=args.seed, episodes=args.episodes, hidden=args.hidden, epochs=args.epochs,
                lr=args.lr, entropy=args.entropy, trace_seconds=args.trace_seconds,
                target_kl=args.target_kl, panel=args.panel, eval_roots=args.eval_roots,
                eval_every=args.eval_every, eval_stochastic=args.eval_stochastic,
                max_attempt_factor=args.max_attempt_factor, workers=args.workers,
                test_seed=getattr(args, 'test_seed', None))


def long_run(args):
    if args.workers < 1:
        raise ValueError('workers must be positive')
    if not math.isfinite(args.hours) or args.hours <= 0 or args.iterations < 1 or args.episodes < 1 or args.eval_every < 1:
        raise ValueError('positive hours/iterations/episodes/eval_every required')
    if min(args.keep_snapshots, args.keep_evaluations) < 1:
        raise ValueError('positive retention counts required')
    if args.eval_roots < 1 or args.test_roots < 0:
        raise ValueError('eval_roots must be positive')

    torch.set_num_threads(1)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('long run requires an empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    (out / 'evaluations').mkdir(exist_ok=True)
    (out / 'snapshots').mkdir(exist_ok=True)

    if args.panel == 'dynamic':
        validation = scenes.panel(args.seed, args.eval_roots, 'validation')
        test_seed = args.seed if args.test_seed is None else args.test_seed
        test = scenes.panel(test_seed, args.test_roots, 'test') if args.test_roots else []
        dev_train = None
    elif args.panel == 'micro':
        from experiments.option_rl_v16.learnability import load_panel
        panel_path = Path(args.panel_path or ROOT / 'outputs/option_rl_v14/learnability_002/panel.json')
        dev_train, validation = load_panel(panel_path)
        validation = validation[:args.eval_roots]
        test = []
    else:
        raise ValueError('unknown panel')
    split_audit = layout.audit_split_content(dict(validation=validation, test=test,
                                                  **({'development_train': dev_train} if dev_train else {})))
    if not split_audit['ok']:
        raise ValueError('fixed panels overlap')
    frozen = {layout.content_hash(i) for i in validation + test}

    def next_instance(ordinal):
        if dev_train:
            return dev_train[ordinal % len(dev_train)]
        return scenes.scene(args.seed, ordinal, 'train')

    warmup = dev_train if dev_train else [scenes.scene(args.seed, i, 'train') for i in range(64)]
    first = EventEnv(warmup[0])
    obs_dim = len(first.observe()[0])

    config = config_of(args)
    args_json = {key: (str(value) if isinstance(value, Path) else value)
                 for key, value in vars(args).items()}
    source_hashes = {name: sha256(ROOT / name) for name in source_files()}
    manifest = dict(
        metadata=metadata(), config=config, args=args_json, obs_dim=obs_dim,
        trainer='event-progress-long-run-v2',
        scene_version=getattr(scenes, 'VERSION', None),
        training_scenes='one fresh legal training root per episode (v14 generator); fixed panels disjoint',
        evaluation_scenes='fixed validation roots; test roots reserved for final analysis',
        warmup_roots=len(warmup),
        gamma=1., advantage='gae by actual event periods, trace_seconds as lambda scale',
        normalization='warmup then batch update once after PPO (v16 protocol); first 760 dims normalized only',
        anomaly_rule='record all attempts; quarantine physical anomalies from PPO; all eval roots count',
        split_audit=split_audit,
        validation_instances=[i.as_dict() for i in validation],
        test_instances=[i.as_dict() for i in test],
        source_sha256=source_hashes,
        environment_config=first.env.config.as_dict(),
        physics_config=asdict(first.env.env.config),
        diagnostics=DEFINITION,
        test_seed=args.seed if args.test_seed is None else args.test_seed,
        retention=dict(periodic_snapshots=args.keep_snapshots,
                       latest_raw_evaluations=args.keep_evaluations,
                       evaluation_milestone=500, best_and_final='preserved'),
        command=' '.join([sys.executable, '-m', 'experiments.option_rl_v21.long_run'] + sys.argv[1:]),
    )
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for name in source_hashes:
        dest = out / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())

    live_status(out, phase='starting', iteration=0, episodes=args.episodes, eval_total=len(validation))
    history = []
    iteration = 0
    ordinal = 0
    best = None
    elapsed_offset = 0.
    policy = Policy(obs_dim, args.hidden)
    normalizer = RunningMeanStd(obs_dim)
    normalizer.update(np.stack([EventEnv(inst).observe()[0] for inst in warmup]))
    optimizers = dict(actor=torch.optim.Adam(policy.actor_parameters(), lr=args.lr),
                      critic=torch.optim.Adam(policy.critic_parameters(), lr=args.lr))
    if args.resume:
        resumed = torch.load(args.resume, map_location='cpu', weights_only=False)
        if resumed['config'] != config:
            raise ValueError('resume configuration mismatch')
        policy, normalizer, optimizers, generator, data = load_state(args.resume, lr=args.lr)
        history = data['history']
        iteration = data['iteration']
        ordinal = data['next_scene_ordinal']
        best = data['best']
        elapsed_offset = history[-1].get('run_seconds', 0.) if history else 0.
        (out / 'manifest.json').write_text(json.dumps(
            dict(manifest, resume=dict(path=str(Path(args.resume).resolve()),
                                       sha256=sha256(args.resume),
                                       iteration=data['iteration'])), indent=2) + '\n')

    executor = parallel.pool(args.workers) if args.workers > 1 else None
    started = time.monotonic()
    status = 'complete'
    write_dashboard(history, out / 'dashboard.html',
                    subtitle=f'{out} | event-progress | dynamic roots | gamma=1', live=read_json(out / 'live_status.json', {}))

    try:
        while iteration < args.iterations:
            if stop_reason(out):
                status = 'requested_stop'
                break
            if time.monotonic() - started >= args.hours * 3600:
                status = 'wall_budget_reached'
                break
            iteration += 1
            iteration_started = time.monotonic()
            batch = []
            attempts = []
            raw_observations = []
            valid_count = 0
            if executor is not None:
                batch, attempts, ordinal, raw_observations, valid_count = parallel.training_batch(
                    executor, policy, normalizer, directory=out / 'worker_snapshot',
                    seed=args.seed, ordinal=ordinal, episodes=args.episodes, frozen=frozen,
                    workers=args.workers, development_instances=dev_train,
                    should_stop=lambda: bool(stop_reason(out)) or time.monotonic()-started >= args.hours*3600,
                    progress=lambda done, total, valid: live_status(
                        out, phase='parallel_rollout', iteration=iteration, episode=valid,
                        episodes=args.episodes, jobs_done=done, jobs_total=total,
                        workers=args.workers, eval_total=len(validation)))
            else:
                for attempt in range(args.episodes * args.max_attempt_factor):
                    if stop_reason(out):
                        break
                    live_status(out, phase='rollout', iteration=iteration, episode=valid_count,
                                episodes=args.episodes, attempt=attempt, workers=args.workers,
                                eval_total=len(validation))
                    instance = next_instance(ordinal)
                    ordinal += 1
                    if layout.content_hash(instance) in frozen:
                        raise RuntimeError('dynamic training root duplicates a fixed holdout root')
                    steps, summary = run_episode(policy, instance, normalizer, generator=generator,
                                                 update_stats=False, collect_observations=True)
                    raw_observations.append(summary.pop('observations'))
                    attempts.append(summary)
                    if summary['eligible_for_ppo']:
                        batch.extend(steps)
                        valid_count += 1
                    if valid_count >= args.episodes:
                        break
            if valid_count < args.episodes:
                status = ('requested_stop' if stop_reason(out) else 'wall_budget_reached'
                          if time.monotonic()-started >= args.hours*3600 else 'insufficient_valid_rollouts')
                atomic_json(out/'partial_rollout.json', dict(attempted_iteration=iteration,
                    completed_episodes=valid_count, requested_episodes=args.episodes, attempts=attempts))
                save_state(out / 'checkpoint.pt', policy=policy, optimizers=optimizers, normalizer=normalizer,
                           iteration=iteration - 1, seed=args.seed, history=history, ordinal=ordinal,
                           generator=generator, best=best, config=config)
                break
            rollout_seconds = time.monotonic() - iteration_started

            live_status(out, phase='ppo', iteration=iteration, episode=valid_count,
                        episodes=args.episodes, workers=args.workers, eval_total=len(validation))
            stats = ppo_update(policy, optimizers, batch, epochs=args.epochs, entropy_coef=args.entropy,
                               advantage='gae', trace_seconds=args.trace_seconds, target_kl=args.target_kl)
            normalizer.update(np.concatenate(raw_observations, axis=0))

            agg = aggregate(attempts)
            row = dict(iteration=iteration,
                       explore_return=agg['return'], explore_success=agg['success'],
                       explore_progress=agg['progress'], explore_final_distance=agg['final_distance'],
                       explore_decisions=agg['decisions_mean'],
                       explore_decision_seconds=agg['decision_seconds'],
                       explore_contact_rate=agg['contact_rate'],
                       event_progress_rate=agg['event_progress_rate'],
                       event_contact_rate=agg['event_contact_rate'],
                       event_contact_lost_rate=agg['event_contact_lost_rate'],
                       event_wait_rate=agg['event_wait_rate'],
                       event_failure_rate=agg['event_failure_rate'],
                       event_timeout_rate=agg['event_timeout_rate'],
                       event_acquisition_timeout_rate=agg['event_acquisition_timeout_rate'],
                       event_progress_timeout_rate=agg['event_progress_timeout_rate'],
                       event_terminal_rate=agg['event_terminal_rate'],
                       action_push=agg['action_push'], action_reposition=agg['action_reposition'],
                       action_wait=agg['action_wait'],
                       action_yield=agg['action_yield'],
                       attempted_episodes=len(attempts), valid_episodes=valid_count,
                       anomaly_episodes=len(attempts) - valid_count,
                       train_anomaly_rate=(len(attempts) - valid_count) / len(attempts),
                       steps=len(batch), simulated_seconds=sum(s['seconds'] for s in attempts),
                       rollout_seconds=rollout_seconds,
                       sample_periods_per_wall_second=sum(s['periods'] for s in attempts) / max(rollout_seconds, 1e-9),
                       run_seconds=elapsed_offset + time.monotonic() - started,
                       **stats)
            for key in REWARD_KEYS:
                row['reward_' + key] = agg['reward_' + key]
            for key in REPORT_FIELDS:
                row['diagnostic_' + key] = agg['diagnostic_' + key]
            for key in ('yield_review', 'yield_blocked', 'direction_unavailable'):
                row['event_' + key + '_rate'] = agg['event_' + key + '_rate']
            history.append(row)
            atomic_json(out / 'history.json', history)
            save_state(out / 'checkpoint.pt', policy=policy, optimizers=optimizers, normalizer=normalizer,
                       iteration=iteration, seed=args.seed, history=history, ordinal=ordinal,
                       generator=generator, best=best, config=config)
            live_status(out, phase='iteration_saved', iteration=iteration, episode=valid_count,
                        episodes=args.episodes, workers=args.workers, eval_total=len(validation))
            write_dashboard(history, out / 'dashboard.html',
                            subtitle=f'{out} | event-progress | dynamic roots | gamma=1',
                            live=read_json(out / 'live_status.json', {}))

            if iteration == 1 or iteration % args.eval_every == 0:
                if not stop_reason(out) and time.monotonic() - started < args.hours * 3600:
                    mode = 'stochastic' if args.eval_stochastic else 'greedy'

                    def on_root(done, total, summary):
                        live_status(out, phase='validation', iteration=iteration, episode=valid_count,
                                    episodes=args.episodes, workers=args.workers, eval_done=done,
                                    eval_total=total, eval_root=summary['scene'], eval_mode=mode)

                    on_root(0, len(validation), {'scene': ''})
                    if executor is not None:
                        eval_started = time.monotonic()
                        summaries = parallel.evaluation_summaries(
                            executor, policy, normalizer, validation,
                            directory=out / 'worker_snapshot', stochastic=args.eval_stochastic,
                            seed=args.seed + 10000, on_root=on_root,
                            should_stop=lambda: bool(stop_reason(out)) or
                            time.monotonic() - started >= args.hours * 3600)
                        evaluation = summarise_evaluation(
                            summaries, validation, seconds=time.monotonic() - eval_started,
                            stochastic=args.eval_stochastic)
                    else:
                        evaluation = evaluate(policy, normalizer, validation,
                                              seed=args.seed + 10000, stochastic=args.eval_stochastic,
                                              on_root=on_root,
                                              should_stop=lambda: bool(stop_reason(out)) or
                                              time.monotonic() - started >= args.hours * 3600)
                    atomic_json(out / 'evaluations' / f'iteration_{iteration:06d}.json', evaluation)
                    if evaluation['complete']:
                        row.update(success_rate=evaluation['success_rate'],
                                   mean_progress=evaluation['mean_progress'],
                                   eval_return=evaluation['mean_return'],
                                   eval_final_distance=evaluation['mean_final_distance'],
                                   eval_decisions=evaluation['mean_decisions'],
                                   eval_decision_seconds=evaluation['mean_decision_seconds'],
                                   eval_contact_rate=evaluation['contact_rate'],
                                   eval_anomaly_rate=evaluation['anomaly_rate'],
                                   eval_seconds=evaluation['seconds'],
                                   eval_roots=evaluation['completed_roots'],
                                   eval_initial_inside_goal=evaluation['initially_inside_goal'],
                                   eval_outside_goal_success=evaluation['outside_goal_success'])
                        for key, value in evaluation['event_rates'].items():
                            row['eval_event_' + key + '_rate'] = value
                        for key, value in evaluation['action_mix'].items():
                            row['eval_action_' + key] = value
                        for key, value in evaluation['diagnostics'].items():
                            row['eval_diagnostic_' + key] = value
                        score = (evaluation['success_rate'], evaluation['mean_progress'])
                        if best is None or score > tuple(best):
                            best = list(score)
                            save_state(out / 'best.pt', policy=policy, optimizers=optimizers,
                                       normalizer=normalizer, iteration=iteration, seed=args.seed,
                                       history=history, ordinal=ordinal, generator=generator,
                                       best=best, config=config)
                            atomic_json(out/'best_evaluation.json', dict(iteration=iteration, **evaluation))
                    else:
                        row['eval_incomplete'] = True
                    atomic_json(out / 'history.json', history)
                    write_dashboard(history, out / 'dashboard.html',
                                    subtitle=f'{out} | event-progress | dynamic roots | gamma=1',
                                    live=read_json(out / 'live_status.json', {}))
            if args.snapshot_every and iteration % args.snapshot_every == 0:
                save_state(out / 'snapshots' / f'iteration_{iteration:06d}.pt', policy=policy,
                           optimizers=optimizers, normalizer=normalizer, iteration=iteration,
                           seed=args.seed, history=history, ordinal=ordinal, generator=generator,
                           best=best, config=config)
            row['iteration_seconds'] = time.monotonic() - iteration_started
            for path in sorted((out/'snapshots').glob('iteration_*.pt'))[:-args.keep_snapshots]:
                path.unlink()
            for path in sorted((out/'evaluations').glob('iteration_*.json'))[:-args.keep_evaluations]:
                index = int(path.stem.split('_')[-1])
                if index != 1 and index % 500:
                    path.unlink()
            row['run_seconds'] = elapsed_offset + time.monotonic() - started
            atomic_json(out / 'history.json', history)
            print(json.dumps({key: row.get(key) for key in
                              ('iteration', 'run_seconds', 'iteration_seconds', 'valid_episodes',
                               'steps', 'explore_return', 'explore_success', 'explore_progress',
                               'explore_decisions', 'event_progress_rate', 'event_failure_rate',
                               'success_rate', 'mean_progress', 'entropy', 'approx_kl')}), flush=True)

    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    completed_iteration = history[-1]['iteration'] if history else 0
    save_state(out / 'policy.pt', policy=policy, optimizers=optimizers, normalizer=normalizer,
               iteration=completed_iteration, seed=args.seed, history=history, ordinal=ordinal,
               generator=generator, best=best, config=config)
    result = dict(status=status, iterations=len(history), iteration=completed_iteration,
                  attempted_iteration=iteration,
                  run_seconds=elapsed_offset + time.monotonic() - started,
                  best=best, next_scene_ordinal=ordinal,
                  stop_reason=stop_reason(out), checkpoint='policy.pt',
                  event_config=metadata()['event_config'], config=config,
                  workers=args.workers, validation_roots=len(validation), test_roots=len(test))
    atomic_json(out / 'result.json', result)
    live_status(out, phase=status, iteration=iteration, episodes=args.episodes,
                workers=args.workers, eval_total=len(validation), result='result.json')
    write_dashboard(history, out / 'dashboard.html',
                    subtitle=f'{out} | event-progress | dynamic roots | gamma=1',
                    live=read_json(out / 'live_status.json', {}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--hours', type=float, default=12.)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--episodes', type=int, default=8)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--entropy', type=float, default=.02)
    parser.add_argument('--trace-seconds', type=float, default=10.)
    parser.add_argument('--target-kl', type=float, default=.03)
    parser.add_argument('--eval-every', type=int, default=25)
    parser.add_argument('--eval-roots', type=int, default=24)
    parser.add_argument('--test-roots', type=int, default=32)
    parser.add_argument('--test-seed', type=int, default=None)
    parser.add_argument('--eval-stochastic', action='store_true')
    parser.add_argument('--panel', choices=['dynamic', 'micro'], default='dynamic')
    parser.add_argument('--panel-path', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--snapshot-every', type=int, default=50)
    parser.add_argument('--keep-snapshots', type=int, default=8)
    parser.add_argument('--keep-evaluations', type=int, default=8)
    parser.add_argument('--max-attempt-factor', type=int, default=3)
    parser.add_argument('--workers', type=int, default=1)
    result = long_run(parser.parse_args())
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
