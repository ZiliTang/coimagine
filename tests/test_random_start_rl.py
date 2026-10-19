"""Training semantics, not just unchanged-policy probability replay."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture(scope="module")
def rl():
    path = Path(__file__).resolve().parents[1] / "experiments/random_start_reposition_v1/rl.py"
    spec = importlib.util.spec_from_file_location("random_start_rl_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_episode(rl, policy):
    hidden = policy.initial_hidden()
    decisions = []
    for index in range(2):
        frame = torch.full((1, 4), 0.2 + index)
        inner = torch.full((3, 4), 0.7 + index)
        with torch.no_grad():
            hidden = policy.advance(frame, hidden)
            first, second, logp, _ = policy.sample(hidden, deterministic=True)
            value = policy.value(hidden).item()
            hidden = policy.advance(inner, hidden)
        decisions.append(dict(boundary_frame=frame, observations=inner,
                              action=[first, second], logp=logp, value=value,
                              return_to_go=1.0 + index, prefix_before=1.0))
    return dict(decisions=decisions, success=False)


def test_update_critic_uses_same_pre_action_state_as_rollout(rl, monkeypatch):
    torch.manual_seed(11)
    policy = rl.JointPolicy(4, hidden=8)
    episode = synthetic_episode(rl, policy)
    recorded = []
    hook = policy.value_head.register_forward_hook(
        lambda _module, _args, output: recorded.append(output.detach().item()))
    monkeypatch.setattr(rl, "UPDATE_EPOCHS", 1)
    rl.update(policy, torch.optim.AdamW(policy.parameters(), lr=0.0), [episode])
    hook.remove()
    assert recorded == pytest.approx([d["value"] for d in episode["decisions"]], abs=1e-7)


def test_gradient_stops_at_previous_decision_but_trains_intervening_frames(rl, monkeypatch):
    torch.manual_seed(12)
    policy = rl.JointPolicy(4, hidden=8)
    episode = synthetic_episode(rl, policy)
    first_boundary = episode["decisions"][0]["boundary_frame"].requires_grad_()
    first_inner = episode["decisions"][0]["observations"].requires_grad_()
    batch = rl.flatten([episode])
    batch["entries"][0]["advantage"] = 0.0
    batch["entries"][1]["advantage"] = 1.0
    monkeypatch.setattr(rl, "flatten", lambda _episodes: batch)
    monkeypatch.setattr(rl, "UPDATE_EPOCHS", 1)
    monkeypatch.setattr(rl, "VALUE_WEIGHT", 0.0)
    monkeypatch.setattr(rl, "ENTROPY_WEIGHT", 0.0)
    rl.update(policy, torch.optim.AdamW(policy.parameters(), lr=0.0), [episode])
    assert first_boundary.grad is None or torch.count_nonzero(first_boundary.grad) == 0
    assert first_inner.grad is not None and torch.linalg.vector_norm(first_inner.grad) > 0


def test_sampling_uses_supplied_generator_without_consuming_global_rng(rl):
    policy = rl.JointPolicy(4, hidden=8)
    hidden = policy.initial_hidden()
    state = torch.get_rng_state().clone()
    first = [policy.sample(hidden, generator=torch.Generator().manual_seed(7))[:2]
             for _ in range(8)]
    assert torch.equal(state, torch.get_rng_state())
    assert len(set(first)) == 1


def test_local_generator_gives_reproducible_sequence_and_logp(rl):
    policy = rl.JointPolicy(4, hidden=8)
    hidden = policy.initial_hidden()
    def draws(seed):
        generator = torch.Generator().manual_seed(seed)
        return [policy.sample(hidden, generator=generator) for _ in range(20)]
    first = draws(8)
    torch.rand(30)
    assert first == draws(8)
    assert [x[:2] for x in first] != [x[:2] for x in draws(9)]
    for a, b, logp, _ in first:
        assert policy.evaluate(hidden, a, b)[0].item() == pytest.approx(logp, abs=1e-6)


def test_semimarkov_targets_match_hand_calculation(rl):
    gamma = 0.997
    # Rewards here are already weighted by the episode discount prefix.
    decisions = [dict(return_to_go=2.0 + gamma ** 10 * 3.0,
                      prefix_before=1.0, value=0.0),
                 dict(return_to_go=gamma ** 10 * 3.0,
                      prefix_before=gamma ** 10, value=0.0)]
    batch = rl.flatten([dict(decisions=decisions, success=False)])
    assert [e["target"] for e in batch["entries"]] == pytest.approx(
        [2.0 + gamma ** 10 * 3.0, 3.0])
    assert np.isfinite([e["advantage"] for e in batch["entries"]]).all()


def test_better_action_probability_increases_through_real_update(rl, monkeypatch):
    torch.manual_seed(15)
    policy = rl.JointPolicy(4, hidden=8)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=0.03)
    monkeypatch.setattr(rl, "UPDATE_EPOCHS", 1)
    frame = torch.zeros((1, 4))
    def probability():
        with torch.no_grad():
            h = policy.advance(frame, policy.initial_hidden())
            return torch.softmax(policy.head(h)[0], dim=-1)[0, 0].item()
    initial = probability()
    for _ in range(25):
        episodes = []
        with torch.no_grad():
            h = policy.advance(frame, policy.initial_hidden())
            for action, reward in ((0, 1.0), (1, -1.0)):
                logp, _ = policy.evaluate(h, action, 0)
                episodes.append(dict(success=reward > 0, decisions=[dict(
                    boundary_frame=frame, observations=torch.zeros((0, 4)),
                    action=[action, 0], logp=logp.item(), value=policy.value(h).item(),
                    return_to_go=reward, prefix_before=1.0)]))
        rl.update(policy, optimizer, episodes)
    # Only two of the 15 actions appear in this hand-worked fixture. Check direction and
    # ranking, not an unjustified absolute probability threshold for the other 13 actions.
    assert probability() > initial + 0.2
    with torch.no_grad():
        hidden = policy.advance(frame, policy.initial_hidden())
        assert policy.head(hidden)[0].argmax().item() == 0


def test_failure_return_retains_real_distance_and_progress(rl):
    instance = rl.layout.build_split("L0", "train", count=1)[0][0]
    def total(potentials):
        environment = rl.E.RandomStartEnv(instance)
        result = 0.0
        return environment._reward_terms(potentials, success=False, budget=True)["total"]
    assert total([1.0, 0.5, 0.3]) > total([1.0, 1.3, 1.8])


def test_dense_reward_hand_calculation_terminal_and_duration(rl):
    instance = rl.layout.build_split("L0", "train", count=1)[0][0]
    environment = rl.E.RandomStartEnv(instance)
    terms = environment._reward_terms([1.0, 0.8, 0.6], success=False, budget=True)
    assert terms["progress"] == pytest.approx(0.4)
    assert terms["distance_cost"] == pytest.approx(-0.02 * 0.05 * (0.8 + 0.6))
    assert terms["time_cost"] == pytest.approx(-0.004)
    assert terms["failure_penalty"] == -2
    assert terms["total"] == pytest.approx(0.4 - 0.0014 - 0.004 - 2)
    # A failed last step moving farther cannot manufacture a positive progress reward.
    assert environment._reward_terms([1.0, 1.8], success=False, budget=True)["progress"] == pytest.approx(-0.8)
    short = environment._reward_terms([1.0] * 3, success=False, budget=True)
    long = environment._reward_terms([1.0] * 101, success=False, budget=True)
    assert long["total"] < short["total"]
    both = environment._reward_terms([0.2, 0.1], success=True, budget=True)
    assert both["success_bonus"] == 10 and both["failure_penalty"] == 0


def test_dense_reward_no_jitter_profit_and_invariant_to_skill_segmentation(rl):
    instance = rl.layout.build_split("L0", "train", count=1)[0][0]
    environment = rl.E.RandomStartEnv(instance)
    reward = environment._reward_terms
    jitter = reward([1.0, 0.9, 1.0], success=False, budget=False)
    assert jitter["progress"] == 0 and jitter["total"] < 0
    whole = reward([1.0, 0.8, 0.6], success=False, budget=True)
    split = (reward([1.0, 0.8], success=False, budget=False)["total"]
             + reward([0.8, 0.6], success=False, budget=True)["total"])
    assert whole["total"] == pytest.approx(split)
    # Same endpoint and duration: earlier approach has a smaller distance integral.
    assert reward([1.0, 0.5, 0.5], success=False, budget=True)["total"] > reward(
        [1.0, 1.0, 0.5], success=False, budget=True)["total"]


def test_real_rollout_consumes_each_preterminal_frame_once_and_replays_value(rl, monkeypatch):
    instance = rl.layout.build_split("L0", "train", count=1)[0][0]
    environment = rl.E.RandomStartEnv(instance, rl.E.EnvConfig(
        time_budget=.6, joint_max_seconds=0.2))
    policy = rl.JointPolicy(len(environment.observation_fields()), hidden=8)
    consumed = []
    original = policy.advance
    def advance(observations, hidden):
        consumed.extend(observations.detach().numpy().copy())
        return original(observations, hidden)
    monkeypatch.setattr(policy, "advance", advance)
    episode = rl.rollout(environment, policy, generator=torch.Generator().manual_seed(4))
    assert len(consumed) == environment.total_periods
    fields = environment.observation_fields()
    elapsed = fields.index("elapsed_fraction")
    assert [row[elapsed] for row in consumed] == pytest.approx([
        tick * rl.CONTROL_DT / environment.config.time_budget
        for tick in range(environment.total_periods)], abs=1e-7)
    with torch.no_grad():
        for decision, hidden in rl.decision_states(policy, episode):
            assert policy.value(hidden).item() == pytest.approx(decision["value"], abs=1e-7)


def test_pass_requires_three_consecutive_recent_evaluation_points(rl):
    def history(points):
        return [dict(dev_success_deterministic=value) for value in points]
    assert not rl.sustained_pass(history([0.8, 0.8]), 0.0)
    assert not rl.sustained_pass(history([0.8, 0.4, 0.8]), 0.0)
    assert rl.sustained_pass(history([0.0, 0.8, 0.8, 0.8]), 0.0)
    assert not rl.sustained_pass(history([0.8, 0.8, 0.8, 0.0]), 0.0)
    assert not rl.sustained_pass(history([0.8, 0.8, 0.8]), 0.7)


def test_value_fit_metrics_evaluate_pre_action_targets(rl):
    policy = rl.JointPolicy(4, hidden=8)
    episode = synthetic_episode(rl, policy)
    for decision in episode["decisions"]:
        decision["return_to_go"] = decision["value"]
    metrics = rl.value_fit_metrics(policy, [episode])
    assert metrics["mse"] == pytest.approx(0.0, abs=1e-12)
    assert metrics["explained_variance"] == pytest.approx(1.0, abs=1e-10)


def test_bounded_retrain_exports_curves_and_report(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1] / "experiments/random_start_reposition_v1/run_bounded_retrain.py"
    spec = importlib.util.spec_from_file_location("bounded_retrain_report_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "OUT", tmp_path)
    (tmp_path / "results").mkdir()
    panel = dict(success=0.0, return_mean=0.2, median_final_distance=0.9)
    history = [dict(update=5, rollout_return_mean=0.1, rollout_success=0.0,
                    dev_return_mean=0.2, dev_success_deterministic=0.0,
                    dev_stochastic_evaluation=panel,
                    critic_before=dict(explained_variance=-0.1),
                    critic_after=dict(explained_variance=0.2))]
    result = dict(history=history, initial_deterministic=panel,
                  final_deterministic=panel, final_stochastic=panel,
                  validation_deterministic=panel, pass_signal=False)
    (tmp_path / "results/rl_diagnostic.json").write_text(json.dumps(result))
    module.summarise()
    assert (tmp_path / "results/curves.png").stat().st_size > 0
    assert "False" in (tmp_path / "README.md").read_text()
    assert "rollout_return_mean" in (tmp_path / "results/curve.csv").read_text()


def test_reward_accounting_matches_direct_environment_components(rl):
    path = Path(__file__).resolve().parents[1] / "experiments/random_start_reposition_v1/reward_breakdown.py"
    spec = importlib.util.spec_from_file_location("reward_accounting_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    instance = rl.layout.build_split("L0", "train", count=1)[0][0]
    env = rl.E.RandomStartEnv(instance, rl.E.EnvConfig(time_budget=.4, joint_max_seconds=.2))
    initial = env.object_distance()
    env.joint_step(13, 13)
    env.joint_step(13, 13)
    actual = module.decompose(env.reward_total, initial, env.object_distance(), env.total_periods,
                              float(env.success), env.config.as_dict())
    for key in ("progress", "time_cost", "success_bonus", "failure_penalty"):
        assert actual[key] == pytest.approx(env.reward_components[key], abs=1e-12)
    assert actual["distance_cost_residual"] == pytest.approx(env.reward_components["distance_cost"], abs=1e-12)
    assert sum(actual[key] for key in module.COMPONENTS) == pytest.approx(env.reward_total, abs=1e-12)
