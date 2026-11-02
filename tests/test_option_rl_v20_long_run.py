"""Long-run orchestration checks: aggregation, dashboard, checkpoint, fault rules."""
import json
import math
from pathlib import Path

import numpy as np
import torch

from experiments.option_rl_v20.interface import Policy
from experiments.option_rl_v20.long_run import aggregate, evaluate, load_state, save_state
from experiments.option_rl_v20.monitor import write_dashboard
from experiments.option_rl_v20.supervise import fault
from experiments.option_rl_v45.train import RunningMeanStd


def summary(**overrides):
    base = dict(reward=1., success=True, progress=.5, final_distance=.1, initial_distance=.6,
                decisions=4, decision_seconds=[.5, .5, .5, .5], contact=True,
                event_counts={'contact': 1, 'progress': 2, 'terminal': 1},
                action_counts={'push': 6, 'wait': 2},
                components={'progress': .8, 'distance_cost': -.1, 'time_cost': -.2,
                            'success_bonus': 10., 'failure_penalty': 0.})
    base.update(overrides)
    return base


def test_aggregate_rates_rewards_and_actions():
    result = aggregate([summary(), summary(contact=False, success=False, reward=-1.,
                                           progress=-.2, event_counts={'progress': 3, 'terminal': 1},
                                           action_counts={'wait': 8},
                                           components={'progress': -.3, 'distance_cost': -.4,
                                                       'time_cost': -.4, 'success_bonus': 0.,
                                                       'failure_penalty': -2.})])
    assert result['episodes'] == 2
    assert result['return'] == 0.
    assert result['success'] == .5
    assert result['contact_rate'] == .5
    assert result['decisions_mean'] == 4.
    assert result['decision_seconds'] == .5
    assert math.isclose(result['event_progress_rate'], (2 + 3) / 8)
    assert math.isclose(result['event_contact_rate'], 1 / 8)
    assert math.isclose(result['event_terminal_rate'], 2 / 8)
    assert math.isclose(result['action_push'], 6 / 16)
    assert math.isclose(result['action_wait'], 10 / 16)
    assert math.isclose(result['action_reposition'], 0.)
    assert result['reward_success_bonus'] == 5.
    assert result['reward_failure_penalty'] == -1.


def test_evaluate_stops_early_and_is_incomplete():
    policy = Policy(958, 8)
    normalizer = RunningMeanStd(958)
    result = evaluate(policy, normalizer, [object()], seed=1, should_stop=lambda: True)
    assert result['complete'] is False and result['completed_roots'] == 0


def test_dashboard_embeds_history_and_status(tmp_path):
    history = [dict(iteration=1, explore_return=1.5, explore_success=.5, explore_progress=.2,
                    event_progress_rate=.7, action_push=.5, mean_progress=.1)]
    (tmp_path / 'live_status.json').write_text(json.dumps(
        dict(phase='rollout', iteration=4, episode=3, episodes=8, eval_done=2, eval_total=24,
             updated_local='2026-11-03T04:00:00+08:00')))
    path = write_dashboard(history, tmp_path / 'dashboard.html', subtitle='unit test')
    text = path.read_text()
    assert 'const H = [' in text and 'explore_return' in text
    assert 'unit test' in text and 'rollout' in text and '2/24' in text
    assert '{{' not in text and '}}' not in text
    assert 'v20 · event-progress PPO' in text


def test_checkpoint_round_trip_keeps_policy_history_and_rng(tmp_path):
    torch.manual_seed(7)
    policy = Policy(958, 8)
    normalizer = RunningMeanStd(958)
    normalizer.update(np.random.default_rng(3).normal(size=(4, 958)))
    optimizers = dict(actor=torch.optim.Adam(policy.actor_parameters(), lr=3e-4),
                      critic=torch.optim.Adam(policy.critic_parameters(), lr=3e-4))
    generator = torch.Generator().manual_seed(11)
    config = dict(seed=1, episodes=1, hidden=8, epochs=1, lr=3e-4, entropy=.02,
                  trace_seconds=10., target_kl=.03, panel='dynamic', eval_roots=1,
                  eval_every=1, eval_stochastic=False, max_attempt_factor=3)
    history = [dict(iteration=1, explore_return=0.)]
    path = tmp_path / 'checkpoint.pt'
    save_state(path, policy=policy, optimizers=optimizers, normalizer=normalizer, iteration=1,
               seed=1, history=history, ordinal=3, generator=generator, best=[.1, .2], config=config)
    x = torch.as_tensor(np.random.default_rng(4).normal(size=(1, 958)), dtype=torch.float32)
    masks = (torch.ones(1, 46, dtype=torch.bool), torch.ones(1, 46, dtype=torch.bool))
    with torch.no_grad():
        before = policy.distributions(x, masks)[0][0].probs
    restored, normalizer2, optimizers2, generator2, data = load_state(path, lr=3e-4)
    with torch.no_grad():
        after = restored.distributions(x, masks)[0][0].probs
    torch.testing.assert_close(before, after, rtol=0., atol=0.)
    np.testing.assert_array_equal(normalizer.mean, normalizer2.mean)
    assert data['iteration'] == 1 and data['next_scene_ordinal'] == 3
    assert data['history'] == history and data['best'] == [.1, .2]
    assert set(optimizers2) == {'actor', 'critic'}
    assert generator2.get_state().equal(generator.get_state())


def test_fault_rules_are_explicit_and_do_not_restart(tmp_path):
    plan = dict(run=str(tmp_path), deadline_unix=1e12)
    now = 1e9
    assert fault([summary()], {}, plan, now) is None
    assert fault([dict(explore_return=float('nan'))], {}, plan, now) == 'nonfinite_history'
    kl = [dict(approx_kl=.6) for _ in range(3)]
    assert fault(kl, {}, plan, now) == 'ppo_kl_over_0_5_three_iterations'
    anomalies = [dict(train_anomaly_rate=.3) for _ in range(3)]
    assert fault(anomalies, {}, plan, now) == 'train_anomaly_rate_ge_25pct_three_iterations'
    assert fault([summary()], dict(updated_unix=now - 400), plan, now) == 'heartbeat_stale_over_300s'
    (tmp_path / 'manifest.json').write_text(json.dumps(dict(
        source_sha256={'experiments/option_rl_v20/event_env.py': '0' * 64})))
    assert fault([summary()], {}, plan, now).startswith('runtime_source_changed:')


def test_snapshot_file_is_hash_named_and_pruned(tmp_path):
    from experiments.option_rl_v20 import parallel

    torch.manual_seed(9)
    policy = Policy(958, 8)
    normalizer = RunningMeanStd(958)
    normalizer.update(np.random.default_rng(5).normal(size=(3, 958)))
    path, digest = parallel.write_snapshot(tmp_path, policy, normalizer, keep=2)
    assert Path(path).exists() and Path(path).name == digest + '.pt'
    data = torch.load(path, map_location='cpu', weights_only=False)
    assert data['hash'] == digest and data['obs_dim'] == 958
    same_path, same_digest = parallel.write_snapshot(tmp_path, policy, normalizer, keep=2)
    assert same_path == path and same_digest == digest
    with torch.no_grad():
        policy.value.weight.add_(1.)
    parallel.write_snapshot(tmp_path, policy, normalizer, keep=2)
    with torch.no_grad():
        policy.value.weight.add_(1.)
    parallel.write_snapshot(tmp_path, policy, normalizer, keep=2)
    assert len(list(tmp_path.glob('*.pt'))) == 2


def test_parallel_two_worker_rollout_returns_real_steps(tmp_path):
    from experiments.option_rl_v16.scenes import scene
    from experiments.option_rl_v20 import parallel

    torch.manual_seed(13)
    policy = Policy(958, 8)
    normalizer = RunningMeanStd(958)
    executor = parallel.pool(2)
    try:
        steps, attempts, ordinal, observations, valid = parallel.training_batch(
            executor, policy, normalizer, directory=tmp_path / 'snapshots', seed=11,
            ordinal=0, episodes=1, frozen=set(), workers=2)
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
    assert len(attempts) == 1 and valid == 1 and ordinal == 1
    assert len(steps) == attempts[0]['decisions']
    assert steps[0].obs.shape == (958,)
    assert observations[0].shape == (len(steps), 958)
    assert attempts[0]['policy_snapshot'] and attempts[0]['worker_pid'] > 0
