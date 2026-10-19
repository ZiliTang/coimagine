"""Check diagnostic gradients against the production PPO update, including clipping."""
import importlib
import json
import sys
from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="module")
def audit():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] /
                           "experiments/random_start_reposition_v1"))
    return importlib.import_module("diagnose_v4")


def test_group_gradients_sum_to_real_update(audit, monkeypatch):
    r = audit.rl
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = r.JointPolicy(4, hidden=8)
    episodes = []
    for ep_index in range(2):
        hidden = model.initial_hidden()
        decisions = []
        for i in range(2):
            frame = torch.randn(1, 4)
            inner = torch.randn(2, 4)
            with torch.no_grad():
                hidden = model.advance(frame, hidden)
                a, b, logp, _ = model.sample(hidden, deterministic=True)
                value = model.value(hidden).item()
                hidden = model.advance(inner, hidden)
            decisions.append(dict(boundary_frame=frame, observations=inner,
                                  action=[a, b], logp=logp, value=value,
                                  periods=1 if i == 0 else 5,
                                  return_to_go=100.0*(ep_index+1), prefix_before=1.0))
        episodes.append(dict(decisions=decisions, success=ep_index == 0))
    report, expected = audit.gradient_audit(model, episodes)
    assert report["counts"] == {"success": 2, "failure_short": 1, "failure_long": 1}
    assert report["logp_replay_max_abs"] < 1e-6
    monkeypatch.setattr(r, "UPDATE_EPOCHS", 1)
    r.update(model, torch.optim.AdamW(model.parameters(), lr=0, weight_decay=0), episodes)
    params = list(model.parameters())
    actual = audit.flat_grad(params, [p.grad for p in params])
    expected *= report["global_clip_scale"]
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_old_training_manifest_cannot_silently_change_holdouts(audit, tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"config": {}}))
    with pytest.raises(ValueError, match="split version"):
        audit.runner.train(tmp_path, resume=True)
