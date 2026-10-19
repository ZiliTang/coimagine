"""Resumable two-arm checkpoint contract and the new rollout diagnostics."""

import importlib
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def runner():
    directory = str(Path(__file__).resolve().parents[1] /
                    "experiments/random_start_reposition_v1")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("long_train")


def tiny_manifest(arm="stopped"):
    return dict(config=dict(seed=7, episodes_per_update=1, updates=4, hidden=8,
                            arm=arm, critic_shared_grad=(arm == "shared"),
                            panel_contract={"train": {"count": 1, "sha256": "x"}},
                            learning_rate=3e-4, weight_decay=0.0),
                deadline_unix=1234567890)


def test_checkpoint_roundtrip_restores_optimizer_best_and_rng(runner, tmp_path):
    r = runner
    torch.manual_seed(5)
    policy = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0)
    manifest = tiny_manifest()
    history = [dict(update=3, success=0.0)]
    best = dict(update=0, score=[0.0, -1.0], selection="fixed-validation rule")
    path = tmp_path / "resume.pt"
    r.save_stability_checkpoint(path, policy, optimizer, history, {"initial": 1}, best, manifest)
    restored = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4, weight_decay=0)
    state = r.load_stability_checkpoint(path, restored, restored_optimizer, manifest)
    assert state["update"] == 3
    assert state["best"] == best
    assert state["initial"] == {"initial": 1}
    assert all(torch.equal(value, restored.state_dict()[key])
               for key, value in policy.state_dict().items())
    assert state["critic_contract"] == restored.contract()


@pytest.mark.parametrize("change", ["critic_mode", "deadline", "split_version", "config"])
def test_resume_rejects_contract_mismatch(runner, tmp_path, change):
    r = runner
    torch.manual_seed(6)
    policy = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=3e-4, weight_decay=0)
    manifest = tiny_manifest()
    path = tmp_path / "resume.pt"
    r.save_stability_checkpoint(path, policy, optimizer, [dict(update=1)], {}, None, manifest)
    other = tiny_manifest()
    if change == "critic_mode":
        other["config"]["arm"] = "shared"
        other["config"]["critic_shared_grad"] = True
        restored = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=True)
    elif change == "deadline":
        other["deadline_unix"] += 1
        restored = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
    elif change == "split_version":
        restored = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["split_version"] = "legacy"
        torch.save(state, path)
    else:
        other["config"]["learning_rate"] = 1e-3
        restored = r.rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4, weight_decay=0)
    with pytest.raises(ValueError, match="Resume rejected"):
        r.load_stability_checkpoint(path, restored, restored_optimizer, other)


def test_stopped_arm_resume_matches_uninterrupted_updates(runner, tmp_path):
    r = runner
    torch.set_num_threads(1)
    torch.manual_seed(8)
    instance = r.rl.layout.build_split("L0", "train", count=1)[0][0]
    config = r.rl.E.EnvConfig(time_budget=.2, joint_max_seconds=.1)
    dim = len(r.rl.E.RandomStartEnv(instance).observation_fields())
    manifest = tiny_manifest()

    def step(model, optimizer, index):
        environment = r.rl.E.RandomStartEnv(instance, config)
        episode = r.rl.rollout(environment, model,
                               generator=torch.Generator().manual_seed(index))
        r.rl.update(model, optimizer, [episode])

    policy = r.rl.JointPolicy(dim, hidden=8, critic_shared_grad=False)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=r.rl.LEARNING_RATE, weight_decay=0)
    for index in range(1, 5):
        step(policy, optimizer, index)
    expected = {key: value.clone() for key, value in policy.state_dict().items()}

    torch.manual_seed(8)
    interrupted = r.rl.JointPolicy(dim, hidden=8, critic_shared_grad=False)
    interrupted_optimizer = torch.optim.AdamW(interrupted.parameters(),
                                              lr=r.rl.LEARNING_RATE, weight_decay=0)
    for index in range(1, 3):
        step(interrupted, interrupted_optimizer, index)
    path = tmp_path / "resume.pt"
    r.save_stability_checkpoint(path, interrupted, interrupted_optimizer,
                                [dict(update=2)], {}, None, manifest)
    restored = r.rl.JointPolicy(dim, hidden=8, critic_shared_grad=False)
    restored_optimizer = torch.optim.AdamW(restored.parameters(),
                                           lr=r.rl.LEARNING_RATE, weight_decay=0)
    r.load_stability_checkpoint(path, restored, restored_optimizer, manifest)
    for index in range(3, 5):
        step(restored, restored_optimizer, index)
    assert all(torch.equal(value, expected[key])
               for key, value in restored.state_dict().items())


def test_rollout_logs_short_skills_exits_and_repeat_runs(runner):
    r = runner
    instance = r.rl.layout.build_split("L0", "train", count=1)[0][0]
    config = r.rl.E.EnvConfig(time_budget=.4, joint_max_seconds=.2)
    policy = r.rl.JointPolicy(len(r.rl.E.RandomStartEnv(instance).observation_fields()),
                              hidden=8)
    episode = r.rl.rollout(r.rl.E.RandomStartEnv(instance, config), policy,
                           generator=torch.Generator().manual_seed(0))
    for key in ("reward_components", "exits", "short_decisions", "short_time_fraction",
                "repeated_action_pairs", "longest_repeat_run", "mean_repeat_run",
                "action_indices"):
        assert key in episode
    assert 0.0 <= episode["short_time_fraction"] <= 1.0
    assert episode["longest_repeat_run"] >= 1
    assert len(episode["action_indices"]) == episode["skills_used"]
    assert sum(episode["reward_components"][key] for key in
               ("progress", "distance_cost", "time_cost",
                "success_bonus", "failure_penalty")) == pytest.approx(
        episode["return"], abs=1e-6)


def test_advantage_group_statistics_cover_all_decisions(runner):
    r = runner
    base = dict(prefix_before=1.0, value=0.5)
    episodes = [
        dict(success=True, decisions=[dict(base, periods=3, return_to_go=5.0)]),
        dict(success=False, decisions=[dict(base, periods=1, return_to_go=0.0),
                                       dict(base, periods=9, return_to_go=1.0)])]
    stats = r.rl.advantage_group_statistics(episodes)
    assert set(stats) == {"success", "failure_short", "failure_long"}
    assert sum(group["decisions"] for group in stats.values()) == 3
    assert stats["success"]["decisions"] == 1
    assert stats["failure_short"]["decisions"] == 1
    assert stats["failure_long"]["decisions"] == 1
