from types import SimpleNamespace

import numpy as np
import torch

from tools import cap_abc_event_ppo as event


def make_review(monkeypatch, mode):
    monkeypatch.setattr(event, "pusher", lambda _: (0, (1.0, 0.0)))
    monkeypatch.setattr(event.s, "observe", lambda _: {"robots": [{"own": {"x": 0.0, "y": 0.0}}]})
    env = SimpleNamespace(total_periods=0, active_target=1)
    env.public = lambda: {"tick": env.total_periods}
    world = SimpleNamespace(env=env, decision_count=0)
    queries = []

    def choose(_, obs, action, left, reason):
        queries.append((obs["tick"], left, reason))

    review = event.PushReview(mode, choose)(world, {"target_id": 1})

    def tick(x, *, stage="drive"):
        env.total_periods += 1
        feedback = {"robots": [{
            "own": {"x": x, "y": 0.0}, "contacting_object": True,
            "execution": {"direction_valid": True, "stage": stage},
        }]}
        assert review(feedback, 20-env.total_periods) is None

    return tick, queries


def test_contact_progress_and_braking(monkeypatch):
    tick, queries = make_review(monkeypatch, "progress")
    for x in (0.05, 0.10, 0.15, 0.25, 0.36):
        tick(x)
    tick(0.70, stage="brake")
    tick(0.71)
    tick(0.90)
    assert queries == [(3, 17, "contact"), (5, 15, "progress"), (8, 12, "progress")]


def test_time_reviews_use_command_relative_ticks(monkeypatch):
    tick, queries = make_review(monkeypatch, "time")
    for _ in range(9):
        tick(0.0)
    assert queries == [(4, 16, "time"), (8, 12, "time")]


def test_review_masks_duration_only_restarts_and_duplicate_proposals(monkeypatch):
    monkeypatch.setattr(event.base, "candidate_inputs", lambda *args: np.zeros((3, event.base.SIZE)))
    active = {"a0": 7, "a1": 44, "target_id": 1, "ticks": 8}
    alternate = dict(active, a0=9, ticks=20)
    proposals = [dict(active, ticks=20), alternate, alternate]
    rows = event.candidate_rows(None, [], None, proposals, active)
    assert rows[:, -1].tolist() == [0, 1, 0, 1]


def test_training_resume_verifies_latest_actor_and_optimizer(tmp_path):
    model = event.Policy()
    optimizer = torch.optim.Adam(model.parameters())
    objective = model.actor[0].weight.square().sum()
    objective.backward()
    optimizer.step()
    event.save_training_round(tmp_path, "progress", 1, model, optimizer, {"rooms": ["a"]})
    event.save_training_round(tmp_path, "progress", 2, model, optimizer, {"rooms": ["b"]})
    restored = event.Policy()
    restored_optimizer = torch.optim.Adam(restored.parameters())
    assert event.restore_latest(tmp_path, "progress", restored, restored_optimizer, 4) == 2
    assert not event.state_path(tmp_path, "progress", 1).exists()
    assert not event.actor_path(tmp_path, "progress", 1).exists()
    assert torch.equal(model.actor[0].weight, restored.actor[0].weight)
    assert torch.equal(optimizer.state[model.actor[0].weight]["exp_avg"],
                       restored_optimizer.state[restored.actor[0].weight]["exp_avg"])
