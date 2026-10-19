"""Parallelism must preserve rollout values, job order, policy isolation and resume state."""
import concurrent.futures
import importlib
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def runner():
    import sys
    directory = str(Path(__file__).resolve().parents[1] / "experiments/random_start_reposition_v1")
    sys.path.insert(0, directory)
    return importlib.import_module("long_train")


def test_spawn_rollouts_equal_serial_and_preserve_policy(runner):
    r = runner
    torch.manual_seed(2)
    instances, _ = r.rl.layout.build_split("L0", "train", count=2)
    config = r.rl.E.EnvConfig(time_budget=.6, joint_max_seconds=.2)
    policy = r.rl.JointPolicy(len(r.rl.E.RandomStartEnv(instances[0]).observation_fields()), hidden=8)
    state = r.snapshot(policy)
    digest = r.snapshot_hash(state)
    jobs = [(instance, state, 8, i, False, config, False) for i, instance in enumerate(instances)]
    serial = [r.rollout_job(job) for job in jobs]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn"),
                                               initializer=r.worker_init) as executor:
        parallel = r.collect(executor, jobs)
        repeated = r.collect(executor, jobs)
    for reference, actual, repeat in zip(serial, parallel, repeated):
        assert reference["instance"] == actual["instance"] == repeat["instance"]
        assert reference["return"] == actual["return"] == repeat["return"]
        assert actual["policy_hash"] == digest
        assert len(reference["decisions"]) == len(actual["decisions"])
        for old, new, again in zip(reference["decisions"], actual["decisions"], repeat["decisions"]):
            for key in ("action", "logp", "value", "return_to_go"):
                assert old[key] == new[key] == again[key]
            assert np.array_equal(old["observations"], new["observations"].numpy())
    assert r.snapshot_hash(r.snapshot(policy)) == digest


def test_normal_checkpoint_restore_matches_uninterrupted_updates(runner, tmp_path):
    r = runner
    torch.set_num_threads(1)
    torch.manual_seed(3)
    instance = r.rl.layout.build_split("L0", "train", count=1)[0][0]
    config = r.rl.E.EnvConfig(time_budget=.2, joint_max_seconds=.1)
    dim = len(r.rl.E.RandomStartEnv(instance).observation_fields())
    policy = r.rl.JointPolicy(dim, hidden=8)
    initial_weights = {key: value.clone() for key, value in policy.state_dict().items()}
    optimizer = torch.optim.AdamW(policy.parameters(), lr=r.rl.LEARNING_RATE, weight_decay=0)
    manifest = dict(config=dict(seed=3), deadline_unix=12345)
    def step(model, opt, index):
        env = r.rl.E.RandomStartEnv(instance, config)
        ep = r.rl.rollout(env, model, generator=torch.Generator().manual_seed(index))
        r.rl.update(model, opt, [ep])
    for index in range(1, 5):
        step(policy, optimizer, index)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}
    policy.load_state_dict(initial_weights)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=r.rl.LEARNING_RATE, weight_decay=0)
    for index in range(1, 3):
        step(policy, optimizer, index)
    checkpoint = tmp_path / "resume.pt"
    r.save_checkpoint(checkpoint, policy, optimizer, [dict(update=2)], {}, None, manifest)
    restored = r.rl.JointPolicy(dim, hidden=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=r.rl.LEARNING_RATE, weight_decay=0)
    state = r.load_checkpoint(checkpoint, restored, restored_optimizer, manifest)
    assert state["update"] == 2
    for index in range(3, 5):
        step(restored, restored_optimizer, index)
    assert all(torch.equal(value, expected[key]) for key, value in restored.state_dict().items())
    with pytest.raises(ValueError):
        r.load_checkpoint(checkpoint, restored, restored_optimizer,
                          dict(config=dict(seed=4), deadline_unix=12345))


def test_balanced_epoch_covers_all_instances_and_is_resume_deterministic(runner):
    first = [runner.balanced_indices(0, i) for i in range(1, 33)]
    assert all(len(set(batch)) == 8 for batch in first)
    assert sorted(i for batch in first for i in batch) == list(range(256))
    assert runner.balanced_indices(0, 17) == first[16]
    second = [runner.balanced_indices(0, i) for i in range(33, 65)]
    assert sorted(i for batch in second for i in batch) == list(range(256))
    assert first != second
