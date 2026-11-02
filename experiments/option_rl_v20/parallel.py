"""Persistent spawn workers for event-progress rollouts.

Each worker holds a frozen copy of one policy/normalizer snapshot.  The snapshot
is written once per update to a hash-named local file (not pickled into every
job), so a 16-worker batch does not copy the model sixteen times over IPC.
Workers only sample; the main process owns PPO, the normalizer and checkpoints.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import multiprocessing as mp
import os
import time

import numpy as np
import torch

from coimagine.random_start import layout
from experiments.option_rl_v16 import scenes

_CACHE = {}


def worker_init():
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def pool(workers):
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    return ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn'),
                               initializer=worker_init)


def behaviour_seed(seed, index, stream=0):
    return int(np.random.SeedSequence([seed, index, 909, stream]).generate_state(1, dtype=np.uint64)[0] % (1 << 63))


def snapshot_digest(state, normalizer):
    digest = hashlib.sha256()
    for key, value in state.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().numpy().tobytes())
    digest.update(np.asarray(normalizer.mean).tobytes())
    digest.update(np.asarray(normalizer.var).tobytes())
    return digest.hexdigest()


def write_snapshot(directory, policy, normalizer, *, keep=3):
    """Write one frozen policy/normalizer snapshot; return its path and hash."""
    directory = os.fspath(directory)
    os.makedirs(directory, exist_ok=True)
    state = {key: value.detach().cpu().clone() for key, value in policy.state_dict().items()}
    digest = snapshot_digest(state, normalizer)
    path = os.path.join(directory, digest + '.pt')
    if not os.path.exists(path):
        data = dict(state=state, hidden=policy.hidden, obs_dim=policy.obs_dim, hash=digest,
                    mean=np.asarray(normalizer.mean).copy(), var=np.asarray(normalizer.var).copy(),
                    count=normalizer.count)
        temporary = path + '.tmp'
        torch.save(data, temporary)
        os.replace(temporary, path)
    files = sorted((os.path.join(directory, name) for name in os.listdir(directory)
                    if name.endswith('.pt')), key=os.path.getmtime)
    for old in files[:-keep]:
        try:
            os.unlink(old)
        except FileNotFoundError:
            pass
    return path, digest


def rollout_job(job):
    from experiments.option_rl_v20.interface import Policy
    from experiments.option_rl_v20.train import run_episode
    from experiments.option_rl_v45.train import RunningMeanStd

    started = time.monotonic()
    digest = job['snapshot_hash']
    if _CACHE.get('hash') != digest:
        data = torch.load(job['snapshot_path'], map_location='cpu', weights_only=False)
        if data['hash'] != digest:
            raise RuntimeError('worker snapshot hash mismatch')
        with torch.random.fork_rng(devices=[]):
            policy = Policy(data['obs_dim'], data['hidden'])
        policy.load_state_dict(data['state'])
        policy.eval()
        stats = RunningMeanStd(data['obs_dim'])
        stats.mean = data['mean'].copy()
        stats.var = data['var'].copy()
        stats.count = data['count']
        _CACHE.update(hash=digest, policy=policy, stats=stats)
    steps, summary = run_episode(_CACHE['policy'], job['instance'], _CACHE['stats'],
                                 generator=torch.Generator().manual_seed(job['seed']),
                                 deterministic=job.get('deterministic', False),
                                 update_stats=False,
                                 collect_observations=job.get('collect_observations', False))
    observations = summary.pop('observations', None)
    summary.update(worker_pid=os.getpid(), behaviour_seed=job['seed'], policy_snapshot=digest)
    return dict(steps=steps, summary=summary, observations=observations,
                wall_seconds=time.monotonic() - started)


def collect(executor, jobs, *, on_result=None, should_stop=None, timeout=900):
    """Run jobs; return results in declared job order (not completion order)."""
    if executor is None:
        results = []
        for job in jobs:
            if should_stop and should_stop():
                break
            value = rollout_job(job)
            results.append(value)
            if on_result:
                on_result(len(results), len(jobs), value)
        return results
    futures = {executor.submit(rollout_job, job): index for index, job in enumerate(jobs)}
    values = [None] * len(jobs)
    done = 0
    try:
        for future in as_completed(futures, timeout=timeout):
            if should_stop and should_stop():
                for pending in futures:
                    pending.cancel()
            if future.cancelled():
                continue
            value = future.result()
            values[futures[future]] = value
            done += 1
            if on_result:
                on_result(done, len(jobs), value)
    except TimeoutError:
        for pending in futures:
            pending.cancel()
        raise
    return [value for value in values if value is not None]


def training_batch(executor, policy, normalizer, *, directory, seed, ordinal, episodes, frozen,
                   workers, development_instances=None, should_stop=None, progress=None):
    """Sample until ``episodes`` valid episodes or three times that many attempts."""
    valid, attempts, observations = [], [], []
    snapshot_path, snapshot_hash = write_snapshot(directory, policy, normalizer)
    while len(valid) < episodes and len(attempts) < episodes * 3:
        if should_stop and should_stop():
            break
        count = min(episodes - len(valid), episodes * 3 - len(attempts))
        instances = []
        for _ in range(count):
            instance = (development_instances[ordinal % len(development_instances)]
                        if development_instances else scenes.scene(seed, ordinal))
            ordinal += 1
            if layout.content_hash(instance) in frozen:
                raise RuntimeError('parallel training root duplicates a fixed holdout root')
            instances.append(instance)
        jobs = [dict(instance=instance, snapshot_path=snapshot_path, snapshot_hash=snapshot_hash,
                     seed=behaviour_seed(seed, ordinal - count + index), collect_observations=True)
                for index, instance in enumerate(instances)]
        valid_now = len(valid)

        def on_result(done, total, result):
            nonlocal valid_now
            if result['summary']['eligible_for_ppo']:
                valid_now += 1
            if progress:
                progress(done, total, valid_now)

        results = collect(executor, jobs, on_result=on_result, should_stop=should_stop)
        for result in results:
            summary = result['summary']
            attempts.append(summary)
            if summary['eligible_for_ppo']:
                valid.append((result['steps'], summary))
                observations.append(result['observations'])
        if progress:
            progress(len(attempts), len(attempts), len(valid))
    steps = [step for episode_steps, _ in valid for step in episode_steps]
    return steps, attempts, ordinal, observations, len(valid)


def evaluation_summaries(executor, policy, normalizer, instances, *, directory, stochastic, seed,
                         on_root=None, should_stop=None):
    snapshot_path, snapshot_hash = write_snapshot(directory, policy, normalizer)
    jobs = [dict(instance=instance, snapshot_path=snapshot_path, snapshot_hash=snapshot_hash,
                 seed=behaviour_seed(seed, instance.index, 1), deterministic=not stochastic,
                 collect_observations=False) for instance in instances]

    def progress(done, total, result):
        if on_root:
            on_root(done, total, result['summary'])

    results = collect(executor, jobs, on_result=progress, should_stop=should_stop)
    return [result['summary'] for result in results]
