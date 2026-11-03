"""Bounded PPO runner for event options; no automatic execution on import."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from dataclasses import asdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT/'src'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from coimagine.random_start import env as E, spec as R
from experiments.option_rl_v45.train import RunningMeanStd
from experiments.option_rl_v16.train import Step, ppo_update
from experiments.option_rl_v16.learnability import load_panel
from experiments.option_rl_v21 import spec as S
from experiments.option_rl_v21.event_env import EventEnv, family
from experiments.option_rl_v21.interface import Policy, metadata, validate_metadata


def run_episode(policy, instance, normalizer, *, generator, update_stats=False, deterministic=False,
                record=False, collect_observations=False):
    event_env = EventEnv(instance)
    steps, events, raw_observations = [], [], []
    initial_distance = math.dist((instance.object_pose.x, instance.object_pose.y), instance.goal_centre)
    for _ in range(math.ceil(event_env.env.config.time_budget/R.CONTROL_DT)+1):
        obs, m0, m1 = event_env.observe()
        if collect_observations:
            raw_observations.append(obs.copy())
        if update_stats:
            normalizer.update(obs[None, :])
        encoded = normalizer.normalise(obs)
        encoded[760:] = obs[760:]
        with torch.no_grad():
            actions, logp, value = policy.act(torch.as_tensor(encoded)[None, :], (m0, m1),
                                               deterministic=deterministic, generator=generator)
        a0, a1 = [int(a.item()) for a in actions]
        event = event_env.step(a0, a1, record=record)
        terminal = event['termination'] is not None
        steps.append(Step(encoded, a0, a1, float(logp.item()), float(value.item()),
                          float(event['reward']), m0.copy(), m1.copy(), terminal, event['periods']))
        events.append(event)
        if terminal:
            break
    if event_env.env.termination is None:
        raise RuntimeError('physical horizon guard exceeded')
    eligible = event_env.env.termination not in E.ANOMALY_TERMINATIONS
    final_distance = float(event_env.env.object_distance())
    event_counts = Counter(e['reason'] for e in events)
    action_counts = Counter(family(S.ACTIONS[a]) for e in events for a in e['actions'])
    components = {key: float(sum(e['components'][key] for e in events))
                  for key in ('progress', 'distance_cost', 'time_cost', 'success_bonus', 'failure_penalty')}
    summary = dict(scene=instance.name, termination=event_env.env.termination,
                   success=bool(event_env.env.success), eligible_for_ppo=eligible,
                   periods=event_env.env.total_periods, seconds=event_env.env.total_periods*R.CONTROL_DT,
                   decisions=len(events), reward=float(event_env.env.reward_total), events=events,
                   initial_distance=float(initial_distance), final_distance=final_distance,
                   progress=float((initial_distance-final_distance)/max(initial_distance, 1e-6)),
                   contact=event_counts.get('contact', 0) > 0,
                   event_counts=dict(event_counts), action_counts=dict(action_counts),
                   decision_seconds=[e['elapsed_s'] for e in events], components=components)
    if collect_observations:
        summary['observations'] = np.stack(raw_observations)
    if not math.isclose(sum(s.reward for s in steps), summary['reward'], abs_tol=1e-7):
        raise RuntimeError('transition rewards do not sum to environment reward')
    return steps, summary


def save_checkpoint(path, policy, normalizer, optimizer, iteration, seed):
    torch.save(dict(metadata=metadata(), policy=policy.state_dict(),
                    optimizer={k: o.state_dict() for k, o in optimizer.items()},
                    normalizer=dict(mean=normalizer.mean, var=normalizer.var, count=normalizer.count),
                    obs_dim=policy.obs_dim, hidden=policy.hidden, iteration=iteration, seed=seed), path)


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_metadata(checkpoint['metadata'])
    policy = Policy(checkpoint['obs_dim'], checkpoint['hidden'])
    policy.load_state_dict(checkpoint['policy'])
    normalizer = RunningMeanStd(checkpoint['obs_dim'])
    for key in ('mean', 'var', 'count'):
        setattr(normalizer, key, checkpoint['normalizer'][key])
    return policy, normalizer, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--iterations', type=int, default=1)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eval-episodes', type=int, default=0,
                        help='perturbed panel cases per evaluation mode; 0 disables evaluation')
    args = parser.parse_args()
    if min(args.iterations, args.episodes, args.epochs) < 1 or args.eval_episodes < 0:
        parser.error('iterations, episodes and epochs must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)
    panel = ROOT/'outputs/option_rl_v14/learnability_002/panel.json'
    if not panel.exists():
        panel = ROOT/'docs/agent_exchange/state/testbed_panel.json'
    instances, validation = load_panel(panel)
    if args.eval_episodes > len(validation):
        parser.error('eval episodes exceed available perturbed instances')
    first = EventEnv(instances[0]); obs_dim = len(first.observe()[0])
    policy = Policy(obs_dim, args.hidden)
    normalizer = RunningMeanStd(obs_dim)
    optimizers = dict(actor=torch.optim.Adam(policy.actor_parameters(), lr=3e-4),
                      critic=torch.optim.Adam(policy.critic_parameters(), lr=3e-4))
    paths = ['experiments/option_rl_v21/event_env.py', 'experiments/option_rl_v21/interface.py',
             'experiments/option_rl_v21/train.py', 'experiments/option_rl_v19/unknown_shape.py',
             'experiments/option_rl_v19/spec.py', 'experiments/option_rl_v19/system.py',
             'experiments/option_rl_v21/spec.py', 'experiments/option_rl_v21/system.py',
             'experiments/option_rl_v21/yield_skill.py',
             'experiments/option_rl_v13/mode_env.py', 'experiments/option_rl_v13/controller.py',
             'experiments/option_rl_v13/push_controller.py', 'experiments/option_rl_v14/interface.py',
             'experiments/option_rl_v14/mode_env.py', 'experiments/option_rl_v15/interface.py',
             'experiments/option_rl_v15/scenes.py', 'experiments/option_rl_v16/learnability.py',
             'experiments/option_rl_v16/train.py', 'experiments/option_rl_v45/train.py',
             'src/coimagine/random_start/env.py', 'src/coimagine/random_start/spec.py',
             'src/coimagine/random_start/lidar.py', 'src/coimagine/random_start/skills.py',
             'src/coimagine/random_start/boundary_follow.py', 'src/coimagine/random_start/endpoint_tracking.py',
             'src/coimagine/random_start/effect_push.py', 'src/coimagine/random_start/layout.py',
             'src/coimagine/random_start/geometry.py', 'src/coimagine/random_start/anchor.py',
             'src/coimagine/random_start/peer_clearance.py', 'src/coimagine/continuous2d/backend.py',
             'src/coimagine/continuous2d/environment.py']
    for name in paths:
        snapshot = out/'source_snapshot'/name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT/name).read_bytes())
    manifest = dict(metadata=metadata(), args=vars(args), obs_dim=obs_dim,
                    panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
                    source_sha256={p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths})
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    history = []
    for iteration in range(args.iterations):
        batch, summaries = [], []
        for j in range(args.episodes):
            steps, summary = run_episode(policy, instances[(iteration*args.episodes+j) % len(instances)],
                                         normalizer, generator=generator, update_stats=True)
            summaries.append(summary)
            if summary['eligible_for_ppo']:
                batch.extend(steps)
        stats = ppo_update(policy, optimizers, batch, epochs=args.epochs)
        evaluation = {}
        for mode, deterministic in (('greedy', True), ('stochastic', False)):
            if not args.eval_episodes:
                break
            eval_rng = torch.Generator().manual_seed(args.seed + 10000)
            evals = [run_episode(policy, case, normalizer, generator=eval_rng,
                                 deterministic=deterministic)[1]
                     for case in validation[:args.eval_episodes]]
            evaluation[mode] = dict(episodes=evals, seed=args.seed + 10000,
                                    successes=sum(s['success'] for s in evals),
                                    count=len(evals), anomalies=sum(not s['eligible_for_ppo'] for s in evals))
        record = dict(iteration=iteration+1, episodes=summaries, ppo=stats,
                      evaluation=evaluation)
        history.append(record)
        (out/'history.json').write_text(json.dumps(history, indent=2)+'\n')
        save_checkpoint(out/'checkpoint.pt', policy, normalizer, optimizers, iteration+1, args.seed)
        print(json.dumps(dict(iteration=iteration+1, decisions=len(batch), ppo=stats,
                              termination=[s['termination'] for s in summaries])), flush=True)
    load_checkpoint(out/'checkpoint.pt')


if __name__ == '__main__':
    main()
