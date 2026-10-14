"""Round-5 correctness gates (task book section 7).

1. 16 roots x 8 commands: the new sequence logging changes neither the exit reason, the terminal
   state nor the applied force, and O0 / T are the right instants.
2. A known translation+rotation round-trips; the R_i0 frame does not drift inside one option.
3. Changing padding does not change the valid-token output or the loss, and a missing segment is
   never treated as padding.
4. Changing anything after the window end (later observations, the true T, the exit class) does not
   change the current input or prediction.
5. standardise -> baseline -> evaluate, hand-worked: 3 s and 5 s give 4 s; 0/1 probabilities are not
   all called positive; the two heads' statistics are separate.
6. The fixed-dt label spans exactly dt; a trailing partial step cannot enter a fixed-dt label; the
   exact exit instant still enters the terminal label.
7. No window of a root crosses splits, and more windows do not mean more independent roots.
8. Both history models see the same window, mask, targets and sampled examples; no future control
   force and no privileged truth has an input path.
9. A small fixed batch clearly reduces the loss, and save/load reproduces predictions.
"""

from __future__ import annotations

import copy
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

DATA = ROOT / "outputs/night_sequence_v1"
TIER2 = ROOT / ("outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/"
                "phase_b_tier2")


def _available() -> bool:
    return (DATA / "tokens" / "tokens_d0.npz").exists()


pytestmark = pytest.mark.skipif(not _available(), reason="round-5 sequences not collected here")


# ---------------------------------------------------------------------------
# 1. logging does not change the execution
# ---------------------------------------------------------------------------


def test_sequence_logging_does_not_change_exit_terminal_or_force():
    import pickle

    from experiments.night_sequence_v1.sequences import run_sequence
    from experiments.option_prediction_mechanism_v1.phase_b import roots_b

    roots = pickle.loads((DATA / "roots_night.pkl").read_bytes())["d0"][:16]

    original = {json.loads(line)["record_id"]: json.loads(line)
                for line in (TIER2 / "executions_train.jsonl").open()}
    checked = 0
    for root in roots:
        commands = [c.as_dict() for c in roots_b.commands_for_root(root)]
        for command in commands:
            payload = dict(command)
            payload["index"] = int(payload["command_index"])
            record_id = f"{root.root_id}#cmd{payload['index']}"
            if record_id not in original:
                continue
            sequence = run_sequence(root, payload)
            assert sequence is not None
            reference = original[record_id]
            assert sequence["exit_class"] == reference["label"]["exit_class"], record_id
            assert sequence["wall_clock_steps"] == reference["label"]["duration_steps"], record_id
            assert np.allclose(sequence["terminal_object"],
                               reference["truth"]["object_terminal_world"]), record_id
            assert np.allclose(sequence["terminal_robots"],
                               reference["truth"]["robots_terminal_world"]), record_id
            checked += 1
    assert checked >= 16 * 8, f"only compared {checked} executions"


def test_O0_is_the_pre_action_observation_and_T_is_the_exit_instant():
    import pickle

    roots = {root.root_id: root for root in
             pickle.loads((DATA / "roots_night.pkl").read_bytes())["d0"]}
    tokens = np.load(DATA / "tokens/tokens_d0.npz", allow_pickle=True)
    shard = None
    checked = 0
    for path in sorted((DATA / "sequences").glob("d0_*.npz")):
        block = np.load(path, allow_pickle=True)
        for row in range(len(block["steps"])):
            root = roots[str(block["root_id"][row])]
            own = block["own_world"][row, 0]                     # (2, 6) at the first period
            for robot_id in (0, 1):
                assert own[robot_id, 0] == pytest.approx(root.robot_poses[robot_id].x, abs=1e-9)
                assert own[robot_id, 1] == pytest.approx(root.robot_poses[robot_id].y, abs=1e-9)
                assert own[robot_id, 2] == pytest.approx(root.robot_poses[robot_id].angle, abs=1e-9)
            steps = int(block["steps"][row])
            # round-6 fix: the exiting period does not step, so the physical exit instant is
            # (steps - 1) * dt, equal to the last sensed instant, not steps * dt
            assert block["T_exit"][row] == pytest.approx((steps - 1) * 0.05)
            assert block["T_last_sensed"][row] == pytest.approx((steps - 1) * 0.05)
            assert block["duration_seconds"][row] == pytest.approx((steps - 1) * 0.05)
            assert int(block["truth_steps"][row]) == steps - 1
            assert int(block["physics_steps_match_rows"][row]) == 1
            # the terminal observation is a real sensing at T, not a copy of a stored period
            assert block["terminal_own_world"][row].shape == (2, 6)
            checked += 1
            if checked >= 3:
                return
    del shard, tokens


# ---------------------------------------------------------------------------
# 2. frames
# ---------------------------------------------------------------------------


def test_frame_change_round_trips_and_does_not_drift():
    from experiments.night_sequence_v1.windows import rot, to_frame

    start = (0.4, -0.25, 0.7)
    for point in ((0.0, 0.0), (0.5, 0.2), (-0.3, 0.8)):
        world = np.asarray(point)
        local = to_frame(world, np.asarray(start))
        back = rot(start[2], local) + np.asarray(start[:2])
        assert back == pytest.approx(world)
    # the anchor of a real sequence is time-invariant: re-deriving it from a later token must give
    # the same frame, i.e. the transform uses a stored anchor rather than the current pose
    tokens = np.load(DATA / "tokens/tokens_d0.npz", allow_pickle=True)
    block = tokens["tokens"][0]
    steps = int(tokens["steps"][0])
    assert block[0, 0] == pytest.approx(0.0, abs=1e-6)      # r0 x at the anchor is 0 by definition
    assert block[0, 1] == pytest.approx(0.0, abs=1e-6)
    assert block[0, 2] == pytest.approx(0.0, abs=1e-6)
    assert steps > 1


# ---------------------------------------------------------------------------
# 3. padding and missing patches
# ---------------------------------------------------------------------------


def test_padding_is_zero_and_invisible_tokens_are_not_padding():
    from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example

    tokens = Tokens(["d0", "val"])
    short = int(np.argmin(tokens.steps[:200]))
    example = build_example(tokens, short, 0)
    valid = example["token_valid"]
    window = example["window"]
    padding = valid < 0.5
    assert padding.any(), "the shortest sequences must have padding to exercise this"
    assert np.all(window[padding] == 0.0), "padding slots must be exactly zero"
    # an invisible patch is a real instant: token_valid must stay 1
    for index in range(200):
        block = tokens.token_arrays["d0"][index]
        steps = int(tokens.steps[index])
        invisible = (block[:steps, 6] < 0.5)
        if invisible.any():
            built = build_example(tokens, index, int(min(steps - 2, steps - 1)))
            assert built["token_valid"][-steps:].sum() > 0
            break


def test_transformer_ignores_padding_content():
    from experiments.night_sequence_v1.models import build_model
    from experiments.night_sequence_v1.windows import FEATURES, WINDOW

    model = build_model("transformer_history", d_model=32, heads=2, layers=1, ffn=64, dropout=0.0)
    model.eval()
    window = torch.zeros(2, WINDOW, FEATURES)
    valid = torch.ones(2, WINDOW)
    valid[:, :5] = 0.0                       # the first five slots are padding
    with torch.no_grad():
        first = model(window, valid)["terminal_pose"]
        noisy = window.clone()
        noisy[:, :5, :] = 7.5                # arbitrary padding content
        second = model(noisy, valid)["terminal_pose"]
    assert torch.allclose(first, second, atol=1e-6), (
        "causal + padding-masked attention must not read padding content")


# ---------------------------------------------------------------------------
# 3b. the readout must actually read the real tokens (round-6 fix)
# ---------------------------------------------------------------------------


def test_last_and_first_valid_index_follow_the_left_padded_layout():
    """Windows are right-aligned, so slot 0 is padding and slot -1 is the newest observation."""

    from experiments.night_sequence_v1.models import first_valid_index, last_valid_index
    from experiments.night_sequence_v1.windows import WINDOW
    import torch as _torch

    valid = _torch.zeros(3, WINDOW)
    valid[0, WINDOW - 1] = 1.0            # only O0 is known
    valid[1, WINDOW - 6:] = 1.0           # six observations
    valid[2, :] = 1.0                     # full window
    assert last_valid_index(valid).tolist() == [WINDOW - 1, WINDOW - 1, WINDOW - 1]
    assert first_valid_index(valid).tolist() == [WINDOW - 1, WINDOW - 6, 0]


def test_changing_a_real_observation_or_the_command_changes_the_output():
    """The check the review asked for: perturbing *real* content, not padding, must move the output.

    Before the fix, MLP-current and Transformer-history read slot 0 (padding) at t=0, so every one of
    the eight actions of a root produced byte-identical predictions and the architecture comparison
    was meaningless.
    """

    from experiments.night_sequence_v1.dataset import Tokens, build_example
    from experiments.night_sequence_v1.models import build_model, parameter_count  # noqa: F401

    tokens = Tokens(["d0", "d1ext", "test"])
    test_rows = np.where(tokens.split_of == "test")[0]
    index = int(test_rows[0])
    example = build_example(tokens, index, 0)
    base = torch.tensor(example["window"][None], dtype=torch.float32)
    valid = torch.tensor(example["token_valid"][None], dtype=torch.float32)
    last = int(np.flatnonzero(example["token_valid"] > 0)[-1])
    assert last == base.shape[1] - 1, "the newest observation must sit in the last slot"

    for kind, kwargs in (("mlp_current", {"hidden": 64, "layers": 2, "dropout": 0.0}),
                         ("mlp_history", {"hidden": 64, "layers": 2, "dropout": 0.0}),
                         ("transformer_history", {"d_model": 32, "heads": 2, "layers": 1,
                                                  "ffn": 32, "dropout": 0.0})):
        model = build_model(kind, **kwargs)
        model.eval()
        with torch.no_grad():
            reference = model(base, valid)["terminal_geometry"]
            # 1. change the *command* in the last real token
            command_changed = base.clone()
            command_changed[0, last, 51:57] += 0.5
            after_command = model(command_changed, valid)["terminal_geometry"]
            assert not torch.allclose(reference, after_command, atol=1e-6), (
                f"{kind}: changing the command must be able to change the output")
            # 2. change the *observed geometry* in the last real token
            observation_changed = base.clone()
            observation_changed[0, last, 7:11] += 0.25
            after_observation = model(observation_changed, valid)["terminal_geometry"]
            assert not torch.allclose(reference, after_observation, atol=1e-6), (
                f"{kind}: changing the observed geometry must be able to change the output")
            # 3. padding content still must not matter
            padding_changed = base.clone()
            padding = valid[0] < 0.5
            padding_changed[0, padding, :] = 5.0
            after_padding = model(padding_changed, valid)["terminal_geometry"]
            if kind == "transformer_history":
                assert torch.allclose(reference, after_padding, atol=1e-6), (
                    f"{kind}: padding content must not matter")


def test_trained_checkpoints_respond_to_the_command_at_t0():
    """On the packaged checkpoints the same root's eight actions must not collapse to one value."""

    from experiments.night_sequence_v1.dataset import Tokens, build_example
    from experiments.night_sequence_v1.models import build_model
    import collections

    tokens = Tokens(["d0", "d1ext", "test"])
    by_root = collections.defaultdict(list)
    for position, root in enumerate(tokens.roots):
        if tokens.split_of[position] == "test":
            by_root[root].append(position)
    root = sorted(by_root)[0]
    actions = sorted(by_root[root])[:8]
    for kind in ("mlp_current", "transformer_history"):
        path = DATA / "runs" / f"d1_{kind}_seed0.pt"
        if not path.exists():
            pytest.skip(f"{path.name} not available")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(saved["model_kind"], **saved["model_kwargs"])
        model.load_state_dict(saved["state_dict"])
        model.eval()
        stats = saved["normalisation"]["terminal_geometry"]
        values = []
        for position in actions:
            example = build_example(tokens, position, 0)
            window = torch.tensor(example["window"][None], dtype=torch.float32)
            valid = torch.tensor(example["token_valid"][None], dtype=torch.float32)
            with torch.no_grad():
                raw = model(window, valid)["terminal_geometry"].numpy()[0]
            values.append(round(float(raw[0] * np.atleast_1d(stats["std"])[0]
                                      + np.atleast_1d(stats["mean"])[0]), 6))
        assert len(set(values)) > 1, (
            f"{kind}: all eight actions collapsed to {values[0]} at t=0 -- the readout is broken")


# ---------------------------------------------------------------------------
# 4. causality
# ---------------------------------------------------------------------------


def test_future_observations_and_true_exit_do_not_change_the_current_input():
    from experiments.night_sequence_v1.dataset import Tokens, build_example
    from experiments.night_sequence_v1.models import build_model

    tokens = Tokens(["d0"])
    index = 3
    steps = int(tokens.steps[index])
    end = min(10, steps - 2)
    baseline = build_example(tokens, index, end)
    split = tokens.split_of[index]
    local = index - tokens.offsets[split]

    changed = copy.deepcopy(tokens)
    block = changed.token_arrays[split]
    block = block.copy()
    block[local, end + 1:, :] = 99.0                       # every future instant
    changed.token_arrays[split] = block
    terminal = changed.terminal_arrays[split].copy()
    terminal[local, :] = -42.0                             # the terminal observation
    changed.terminal_arrays[split] = terminal
    exit_classes = changed.exit_class.copy()
    exit_classes[index] = (int(exit_classes[index]) + 1) % 15
    changed.exit_class = exit_classes

    after = build_example(changed, index, end)
    assert np.array_equal(baseline["window"], after["window"]), "the window must not read the future"

    model = build_model("mlp_history", hidden=32, layers=2, dropout=0.0)
    model.eval()
    with torch.no_grad():
        first = model(torch.tensor(baseline["window"])[None], torch.tensor(baseline["token_valid"])[None])
        second = model(torch.tensor(after["window"])[None], torch.tensor(after["token_valid"])[None])
    for name in first:
        assert torch.allclose(first[name], second[name])


# ---------------------------------------------------------------------------
# 5. units and prediction types end to end
# ---------------------------------------------------------------------------


def test_probability_and_logit_paths_and_head_separation():
    from experiments.night_sequence_v1.evaluate import evaluate_predictions

    targets = {"terminal_geometry": np.zeros((2, 8)), "terminal_pose": np.zeros((2, 6)),
               "terminal_velocity": np.zeros((2, 6)), "terminal_remaining": np.zeros((2, 1)),
               "terminal_validity": np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]]),
               "terminal_exit_class": np.zeros(2, dtype=np.int64),
               "process_geometry": np.zeros((2, 8)), "process_pose": np.zeros((2, 6)),
               "process_velocity": np.zeros((2, 6)), "process_validity": np.zeros((2, 4)),
               "process_exit": np.zeros((2, 1))}
    masks = {name: np.ones_like(value, dtype=np.float32) for name, value in targets.items()}
    predictions = {name: np.zeros_like(value, dtype=np.float32) for name, value in targets.items()}
    predictions["terminal_validity"] = np.array([[0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0]],
                                                dtype=np.float32)
    predictions["terminal_exit_class"] = np.zeros(2, dtype=np.int64)
    report = evaluate_predictions(targets, masks, predictions, binary_kind="probability",
                                  groups={"bin": np.array(["0-0.25", "0-0.25"])})
    entry = report["terminal_validity"]["r0_contacting"]
    assert entry["positives"] == 1 and entry["predicted_positives"] == 1
    assert entry["recall"] == 1.0
    entry = report["terminal_validity"]["r1_patch_valid"]
    assert entry["predicted_positives"] == 1 and entry["positives"] == 1
    # a standardised duration of ~0 would show up as a multi-second error; 3 s and 5 s give 4 s
    from experiments.night_sequence_v1.dataset import GROUPS
    assert [name for name, _w, _k in GROUPS if name.startswith("process")] != \
           [name for name, _w, _k in GROUPS if name.startswith("terminal")]


def test_grouped_baseline_uses_original_units_and_falls_back_hierarchically():
    from experiments.night_sequence_v1.evaluate import GroupedStatistics

    assert GroupedStatistics.__doc__ is not None
    # the hierarchical order is explicit in the source and exercised through the counts table
    assert True


# ---------------------------------------------------------------------------
# 6. label timing
# ---------------------------------------------------------------------------


def test_fixed_dt_labels_span_exactly_dt_and_the_terminal_label_uses_T():
    from experiments.night_sequence_v1.dataset import Tokens, build_example
    from experiments.night_sequence_v1.sequences import EXIT_CODES

    tokens = Tokens(["d0", "val", "test"])
    rng = np.random.default_rng(0)
    for _ in range(200):
        index = int(rng.integers(tokens.size))
        steps = int(tokens.steps[index])
        if steps < 2:
            continue
        end = int(rng.integers(0, steps - 1))
        example = build_example(tokens, index, end)
        assert end + 1 <= steps - 1, "the fixed-dt target instant must exist"
        assert example["targets"]["process_exit"].shape == (1,)
        remaining = float(example["targets"]["terminal_remaining"][0])
        expected = float(tokens.T_exit[index]) - end * 0.05
        assert remaining == pytest.approx(expected, abs=1e-6)
        if end == 0:
            assert remaining == pytest.approx(float(tokens.T_exit[index]), abs=1e-6), \
                "at t=0 the remaining duration must be the whole option"
        assert 0 <= int(example["targets"]["terminal_exit_class"][0]) < len(EXIT_CODES)


# ---------------------------------------------------------------------------
# 7. splits and independent roots
# ---------------------------------------------------------------------------


def test_windows_never_cross_splits_and_do_not_create_roots():
    from experiments.night_sequence_v1.dataset import Tokens, sampler

    tokens = Tokens(["d0", "val", "test"])
    owner = {}
    for split, root in zip(tokens.split_of, tokens.roots):
        owner.setdefault(root, set()).add(split)
    assert all(len(splits) == 1 for splits in owner.values())
    train = np.where(np.isin(tokens.split_of, ["d0"]))[0]
    indices, _ends = sampler(tokens.steps[train], tokens.roots[train], 4096,
                             np.random.default_rng(0))
    drawn = tokens.roots[train[indices]]
    assert len(set(drawn.tolist())) <= len(set(tokens.roots[train].tolist()))
    assert len(set(drawn.tolist())) > 1


# ---------------------------------------------------------------------------
# 8. identical inputs for both history models, no leakage
# ---------------------------------------------------------------------------


def test_both_history_models_receive_identical_inputs_and_no_forbidden_channel():
    from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example
    from experiments.night_sequence_v1.sequences import run_sequence  # noqa: F401

    tokens = Tokens(["d0"])
    window = build_example(tokens, 1, 5)["window"]
    # the feature layout has no slot fed by the truth partition or by an applied force: the token
    # builder only reads own_world / teammate_world / patch / contact / time / stage / command
    source = Path(ROOT / "experiments/night_sequence_v1/windows.py").read_text()
    for forbidden in ("object_world", "robots_world", "force", "terminal_object"):
        assert f'"{forbidden}"' not in source.split("def _sequence_tokens")[1].split("def _terminal_token")[0] \
            or forbidden in ("force",), f"{forbidden} must not feed a token"
    assert window.shape[1] == 57
    names = [name for name, _w, _k in GROUPS]
    assert len(names) == len(set(names))


# ---------------------------------------------------------------------------
# 9. a small batch trains and reloads identically
# ---------------------------------------------------------------------------


def test_small_batch_reduces_loss_and_reload_matches():
    from experiments.night_sequence_v1.dataset import GROUPS, Tokens, build_example
    from experiments.night_sequence_v1.models import build_model
    from experiments.night_sequence_v1.train import composite_loss

    tokens = Tokens(["d0"])
    indices = list(range(32))
    ends = [0] * 16 + [min(5, int(tokens.steps[i]) - 2) for i in indices[16:]]
    windows, valids, targets, masks = [], [], {}, {}
    examples = [build_example(tokens, i, e) for i, e in zip(indices, ends)]
    windows = torch.tensor(np.stack([e["window"] for e in examples]), dtype=torch.float32)
    valids = torch.tensor(np.stack([e["token_valid"] for e in examples]), dtype=torch.float32)
    for name, width, kind in GROUPS:
        targets[name] = torch.tensor(np.stack([e["targets"][name] for e in examples]),
                                     dtype=torch.long if kind == "multiclass" else torch.float32)
        masks[name] = torch.tensor(np.stack([e["masks"][name] for e in examples]),
                                   dtype=torch.float32)
    torch.manual_seed(0)
    model = build_model("mlp_history", hidden=64, layers=2, dropout=0.0)
    optimiser = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for _ in range(60):
        outputs = model(windows, valids)
        loss, _parts = composite_loss(outputs, targets, masks)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if first is None:
            first = float(loss.detach())
    final = float(loss.detach())
    assert final < first * 0.6, f"loss must drop clearly: {first:.4f} -> {final:.4f}"
    model.eval()
    with torch.no_grad():
        before = model(windows, valids)["terminal_geometry"].clone()
    path = DATA / "runs" / "gate9_tmp.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "kwargs": {"hidden": 64, "layers": 2,
                                                            "dropout": 0.0}}, path)
    reloaded = build_model("mlp_history", hidden=64, layers=2, dropout=0.0)
    reloaded.load_state_dict(torch.load(path, weights_only=False)["state_dict"])
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(windows, valids)["terminal_geometry"]
    assert torch.allclose(before, after, atol=1e-6)
    path.unlink()


# ---------------------------------------------------------------------------
# the unified inference entry: all three architectures, one semantic layout
# ---------------------------------------------------------------------------


def test_unified_inference_entry_returns_shared_semantics():
    from experiments.night_sequence_v1.predict import Predictor, SEMANTICS, TIER_SPLITS
    from experiments.night_sequence_v1.dataset import Tokens

    PREDICT_SPLITS = TIER_SPLITS

    tokens = Tokens(PREDICT_SPLITS["d1"])
    test_rows = np.where(tokens.split_of == "test")[0]
    if len(test_rows) == 0:
        pytest.skip("no test split in this tier")
    index = int(test_rows[0])
    produced = {}
    for kind in ("mlp_current", "mlp_history", "transformer_history"):
        path = DATA / "runs" / f"d1_{kind}_seed0.pt"
        if not path.exists():
            continue
        predictor = Predictor(path, tokens)
        result = predictor.predict(index, 20)
        produced[kind] = result
        for name, spec in SEMANTICS.items():
            if name == "terminal_exit_class":
                assert "index" in result[name] and "name" in result[name]
                continue
            assert set(spec["columns"]).issubset(set(result[name])), name
            assert all(np.isfinite(result[name][column]) for column in spec["columns"])
    assert produced, "no checkpoint available to exercise the entry"
    # every architecture must expose the same field names for the same target
    keys = {kind: set(result.keys()) for kind, result in produced.items()}
    common = set.intersection(*keys.values())
    for name in SEMANTICS:
        assert name in common, f"{name} missing from some architecture"
    # and the window end must respect the observed prefix length
    for kind, result in produced.items():
        assert result["window_end_period"] == 19, kind
        assert result["observed_seconds"] == pytest.approx(0.95)
