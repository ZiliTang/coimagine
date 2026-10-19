"""The single training variable: critic backpropagation into the shared representation.

Arm A (``shared``) is the existing baseline. Arm B (``stopped``) feeds the value head a
detached hidden state. These tests pin the intended semantics, including the fact that arm B
is *not* an independent critic and does *not* freeze the value head.
"""

import importlib.util
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def rl():
    path = Path(__file__).resolve().parents[1] / "experiments/random_start_reposition_v1/rl.py"
    spec = importlib.util.spec_from_file_location("random_start_critic_grad_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_episodes(rl, policy, count=2):
    episodes = []
    generator = torch.Generator().manual_seed(5)
    for episode_index in range(count):
        hidden = policy.initial_hidden()
        decisions = []
        for index in range(2):
            frame = torch.full((1, 4), 0.2 + index + 0.1 * episode_index)
            inner = torch.full((3, 4), 0.7 + index - 0.1 * episode_index)
            with torch.no_grad():
                hidden = policy.advance(frame, hidden)
                first, second, logp, _ = policy.sample(
                    hidden, generator=generator, deterministic=False)
                value = policy.value(hidden).item()
                hidden = policy.advance(inner, hidden)
            decisions.append(dict(
                boundary_frame=frame, observations=inner, action=[first, second],
                logp=logp, value=value, periods=1 if index == 0 else 5,
                return_to_go=1.0 + index + episode_index, prefix_before=1.0))
        episodes.append(dict(decisions=decisions, success=bool(episode_index == 0)))
    return episodes


def matched_arms(rl, seed=21):
    torch.manual_seed(seed)
    shared = rl.JointPolicy(4, hidden=8, critic_shared_grad=True)
    stopped = rl.JointPolicy(4, hidden=8, critic_shared_grad=False)
    stopped.load_state_dict(shared.state_dict())
    return shared, stopped


def test_arms_are_forward_identical_before_any_update(rl):
    shared, stopped = matched_arms(rl)
    assert shared.critic_shared_grad is True and stopped.critic_shared_grad is False
    assert shared.contract()["critic_mode"] == "shared"
    assert stopped.contract()["critic_mode"] == "stopped"
    # the switch adds no parameters: same keys and shapes
    assert list(shared.state_dict()) == list(stopped.state_dict())
    assert all(shared.state_dict()[key].shape == stopped.state_dict()[key].shape
               for key in shared.state_dict())
    episodes = synthetic_episodes(rl, shared)
    shared_batch = rl.flatten(episodes)
    stopped_batch = rl.flatten(episodes)
    for a, b in zip(shared_batch["entries"], stopped_batch["entries"]):
        assert a["target"] == pytest.approx(b["target"], abs=1e-7)
        assert a["advantage"] == pytest.approx(b["advantage"], abs=1e-7)
    for decision, hidden_shared in rl.decision_states(shared, episodes[0]):
        # recompute the same pre-action state under both arms and compare every forward read
        hidden_stopped = hidden_shared.detach().clone()
        logp_a, entropy_a = shared.evaluate(hidden_shared, *decision["action"])
        logp_b, entropy_b = stopped.evaluate(hidden_stopped, *decision["action"])
        assert float(logp_a.detach()) == pytest.approx(float(logp_b.detach()), abs=1e-7)
        assert float(entropy_a.detach()) == pytest.approx(float(entropy_b.detach()), abs=1e-7)
        assert float(shared.value(hidden_shared).detach()) == pytest.approx(
            float(stopped.value(hidden_stopped).detach()), abs=1e-7)
        assert rl.joint_distribution_panel(shared, episodes)  # exact tables remain finite
        table_a = rl.joint_log_distribution(shared, hidden_shared)
        table_b = rl.joint_log_distribution(stopped, hidden_stopped)
        assert float(table_a.exp().sum()) == pytest.approx(1.0, abs=1e-6)
        assert torch.allclose(table_a, table_b, atol=1e-7)


def test_zero_actor_advantage_moves_only_value_head_in_stopped_arm(rl, monkeypatch):
    _, stopped = matched_arms(rl)
    episodes = synthetic_episodes(rl, stopped)
    original_flatten = rl.flatten

    def zero_advantages(eps):
        batch = original_flatten(eps)
        for entry in batch["entries"]:
            entry["advantage"] = 0.0
        return batch

    monkeypatch.setattr(rl, "flatten", zero_advantages)
    monkeypatch.setattr(rl, "UPDATE_EPOCHS", 1)
    before = {key: value.detach().clone() for key, value in stopped.state_dict().items()}
    optimizer = torch.optim.AdamW(stopped.parameters(), lr=1e-3, weight_decay=0)
    rl.update(stopped, optimizer, episodes)
    after = stopped.state_dict()
    for key in before:
        if key.startswith("value_head."):
            assert not torch.equal(before[key], after[key]), "value head must keep training"
        else:
            assert torch.equal(before[key], after[key]), (
                f"arm B changed {key} without any actor advantage")


def test_zero_actor_advantage_does_move_shared_representation_in_shared_arm(rl, monkeypatch):
    shared, _ = matched_arms(rl)
    episodes = synthetic_episodes(rl, shared)
    original_flatten = rl.flatten

    def zero_advantages(eps):
        batch = original_flatten(eps)
        for entry in batch["entries"]:
            entry["advantage"] = 0.0
        return batch

    monkeypatch.setattr(rl, "flatten", zero_advantages)
    monkeypatch.setattr(rl, "UPDATE_EPOCHS", 1)
    before = {key: value.detach().clone() for key, value in shared.state_dict().items()}
    optimizer = torch.optim.AdamW(shared.parameters(), lr=1e-3, weight_decay=0)
    rl.update(shared, optimizer, episodes)
    after = shared.state_dict()
    for key in ("encoder.0.weight", "gru.weight_ih", "value_head.weight"):
        assert not torch.equal(before[key], after[key]), (
            f"arm A value loss should move {key} through the shared representation")


def test_stopped_arm_actor_loss_still_trains_shared_representation(rl, monkeypatch):
    _, stopped = matched_arms(rl)
    episodes = synthetic_episodes(rl, stopped)
    monkeypatch.setattr(rl, "UPDATE_EPOCHS", 1)
    monkeypatch.setattr(rl, "VALUE_WEIGHT", 0.0)
    before = {key: value.detach().clone() for key, value in stopped.state_dict().items()}
    optimizer = torch.optim.AdamW(stopped.parameters(), lr=1e-3, weight_decay=0)
    rl.update(stopped, optimizer, episodes)
    after = stopped.state_dict()
    for key in ("encoder.0.weight", "gru.weight_ih", "head_first.weight",
                "head_second.weight"):
        assert not torch.equal(before[key], after[key]), (
            f"the actor loss alone must still train {key} in arm B")
    # the value head is part of the same optimizer, but VALUE_WEIGHT=0 keeps it unchanged here
    assert torch.equal(before["value_head.weight"], after["value_head.weight"])


def test_grouped_critic_shared_gradient_is_zero_in_stopped_arm(rl):
    shared, stopped = matched_arms(rl)
    shared_episodes = synthetic_episodes(rl, shared)
    stopped_episodes = [dict(episode) for episode in shared_episodes]
    shared_report = rl.grouped_gradient_audit(shared, shared_episodes)
    stopped_report = rl.grouped_gradient_audit(stopped, stopped_episodes)
    assert shared_report["shared_critic_norm"] > 0.0
    assert stopped_report["shared_critic_norm"] == 0.0
    # the actor part of the shared representation is unaffected by the switch
    assert stopped_report["shared_actor_norm"] == pytest.approx(
        shared_report["shared_actor_norm"], rel=1e-6, abs=1e-9)
    # the value head still receives its own gradient in arm B
    assert stopped_report["gradient_norms"]["critic/success"] > 0.0
    assert stopped_report["critic_mode"] == "stopped"
