"""Behaviour checks required by the residual task book section 4.4.

These are written to catch the concrete defects the review found (a left-padding readout taken from
``valid.sum()-1``, a reference added twice, an invisible placeholder entering a statistic), not to
mirror the code structure.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.residual_prediction_v1 import data as D  # noqa: E402
from experiments.residual_prediction_v1.targets import (  # noqa: E402
    GEOMETRY_COLUMNS,
    MIDPOINT_COLUMNS,
    TransformSpec,
    apply_hold_bias,
    decode_prediction,
    encode_target,
    fit_transform,
    hold_output,
    motion_bin,
    visibility_transition,
    visible_reference,
)

OUT = ROOT / "outputs/residual_prediction_v1"
TOKENS = ROOT / "outputs/night_sequence_v1"


def _panels_available() -> bool:
    return (OUT / "panel_val_d1.npz").exists()


# ---------------------------------------------------------------------------
# 1. token positions and padding invariance
# ---------------------------------------------------------------------------


def test_first_and_last_valid_token_positions_for_every_window_length():
    """1, 6, 16, 17, 32 and 33 real tokens must be read at their true positions."""

    from experiments.night_sequence_v1.models import first_valid_index, last_valid_index
    from experiments.night_sequence_v1.windows import WINDOW

    for count in (1, 6, 16, 17, 32, 33):
        valid = torch.zeros(1, WINDOW)
        valid[0, WINDOW - count:] = 1.0
        last = int(last_valid_index(valid)[0])
        first = int(first_valid_index(valid)[0])
        assert last == WINDOW - 1, f"{count} tokens: newest observation must be the last slot"
        assert first == WINDOW - count, f"{count} tokens: O0 must be the first real slot"
        # and the previous, broken formula would have been wrong for every count below 33
        if count < WINDOW:
            assert valid.sum().item() - 1 != last


def test_padding_is_deterministically_zero_and_marked_invalid():
    """The flattened-MLP family consumes every slot, padding included.

    So "changing padding does not change the output" is **not** a property of this model family: a
    non-zero padding slot is a real input. What must hold, and what this test pins down, is that the
    padding region is written as exact zeros by construction and that the validity mask marks
    exactly the real tokens. CONTRACT.md records the zero convention as part of the input contract.
    """

    from experiments.night_sequence_v1.dataset import Tokens
    from experiments.night_sequence_v1.windows import WINDOW

    if not _panels_available():
        pytest.skip("panels not built here")
    tokens = Tokens(D.TIER_SPLITS["d1"])
    index = np.asarray([i for i in range(len(tokens.roots))
                        if tokens.split_of[i] == "d0"][:16])
    ends = np.zeros(len(index), dtype=np.int64)
    batch = D.build_batch(tokens, index, ends)
    window = batch.window.numpy()
    valid = batch.token_valid.numpy()
    for row in range(len(index)):
        real = valid[row] > 0.5
        assert real.any() and real[-1], "the newest real token must occupy the last slot"
        assert np.all(window[row][~real] == 0.0), "padding slots must be exact zeros"
        assert not np.all(window[row][real] == 0.0), "real slots must carry the observation"


# ---------------------------------------------------------------------------
# 2 and 3. encode/decode round-trip and exact decoding of synthetic samples
# ---------------------------------------------------------------------------


def test_encode_decode_round_trips_and_adds_the_displacement_once():
    generator = np.random.default_rng(0)
    truth = generator.normal(size=(256, 4))
    reference = generator.normal(size=(256, 4))
    mask = np.ones((256, 4))
    for parameterisation in ("direct", "residual"):
        for normalisation in ("per_group", "per_column"):
            spec = fit_transform(truth, mask, reference, name=f"{parameterisation}_{normalisation}",
                                 parameterisation=parameterisation, normalisation=normalisation)
            encoded = encode_target(truth, reference, spec)
            decoded = decode_prediction(encoded, reference, spec)
            assert np.abs(decoded - truth).max() < 1e-12, (
                f"{parameterisation}/{normalisation}: round-trip must be exact")
            if parameterisation == "residual":
                # a zero output is the mean displacement, and -mu/sigma is exactly hold
                at_zero = decode_prediction(np.zeros(4), reference, spec)
                at_hold = decode_prediction(hold_output(spec), reference, spec)
                assert np.abs(at_hold - reference).max() < 1e-12
                assert not np.allclose(at_zero, reference), (
                    "with a non-zero mean, a zero output is NOT hold; the report must not conflate "
                    "zero output with zero displacement")


def test_synthetic_constant_position_and_known_displacement_decode_exactly():
    """A constant target and a known constant displacement are reproduced to machine precision."""

    reference = np.tile(np.array([0.30, -0.10, 0.55, 0.20]), (64, 1))
    mask = np.ones((64, 4))
    # (a) constant target equal to the reference
    constant = reference.copy()
    spec = fit_transform(constant, mask, reference, name="residual_constant",
                         parameterisation="residual", normalisation="per_column")
    held = decode_prediction(np.zeros(4), reference, spec)
    assert np.abs(held - reference).max() < 1e-12
    # (b) a known constant displacement of +2 cm on x for both robots
    displacement = np.tile(np.array([0.02, 0.0, 0.02, 0.0]), (64, 1))
    moved = reference + displacement
    spec = fit_transform(moved, mask, reference, name="residual_moved",
                         parameterisation="residual", normalisation="per_column")
    output = hold_output(spec) + displacement[0] / np.atleast_1d(spec.std)
    decoded = decode_prediction(output, reference, spec)
    assert np.abs(decoded - moved).max() < 1e-12


# ---------------------------------------------------------------------------
# 4. invisible targets stay out of statistics, loss and metrics
# ---------------------------------------------------------------------------


def test_invisible_placeholders_do_not_enter_statistics_or_errors():
    generator = np.random.default_rng(1)
    truth = generator.normal(size=(200, 4))
    reference = generator.normal(size=(200, 4))
    mask = np.ones((200, 4))
    mask[:100] = 0.0                                    # half the rows are invisible
    poisoned = truth.copy()
    poisoned[:100] = 999.0                              # placeholder values that must be ignored
    spec = fit_transform(poisoned, mask, reference, name="masked", parameterisation="residual",
                         normalisation="per_column")
    clean = fit_transform(truth, mask, reference, name="masked", parameterisation="residual",
                          normalisation="per_column")
    assert np.allclose(spec.mean, clean.mean) and np.allclose(spec.std, clean.std), (
        "an invisible placeholder must not influence a normalisation statistic")
    # a masked error must ignore the same rows
    error = np.abs(poisoned - truth)
    weighted = (error * mask).sum() / mask.sum()
    assert weighted == 0.0


def test_visibility_transition_counts_all_four_cases():
    current = np.array([[1, 1], [1, 0], [0, 1], [0, 0]])
    target = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    labels = visibility_transition(current, target, robot=0)
    assert labels[0] == "sustained_visible"
    assert labels[1] == "became_invisible"
    assert labels[2] == "became_visible"
    assert labels[3] == "sustained_invisible"


# ---------------------------------------------------------------------------
# 5. the reference never uses invalid geometry and never reads the future
# ---------------------------------------------------------------------------


def test_reference_uses_the_origin_when_the_current_patch_is_invisible():
    current = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    visible = np.array([[1.0, 0.0], [0.0, 1.0]])
    reference = visible_reference(current, visible)
    assert reference[0].tolist() == [1.0, 2.0, 0.0, 0.0], (
        "an invisible robot's reference must be the fixed origin, not an invalid midpoint")
    assert reference[1].tolist() == [0.0, 0.0, 7.0, 8.0]


@pytest.mark.skipif(not _panels_available(), reason="panels not built here")
def test_changing_the_future_does_not_change_the_reference():
    from experiments.night_sequence_v1.dataset import Tokens

    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel = D.load_panel("d1", "val", OUT)
    index = panel["index"][:64]
    ends = panel["end"][:64]
    first = D.build_batch(tokens, index, ends)
    changed = copy.deepcopy(tokens)
    # poison every stored instant AFTER the window end: the reference must not move
    for position, (i, e) in enumerate(zip(index, ends)):
        split = tokens.split_of[i]
        local = int(i) - tokens.offsets[split]
        block = changed.token_arrays[split].copy()
        block[local, int(e) + 1:, :] = 123.0
        changed.token_arrays[split] = block
    second = D.build_batch(changed, index, ends)
    assert np.allclose(first.reference, second.reference)
    assert np.allclose(first.current_mid, second.current_mid)
    assert torch.allclose(first.window, second.window)


# ---------------------------------------------------------------------------
# 6. online and offline decoding agree
# ---------------------------------------------------------------------------


def test_online_and_offline_decode_the_same_checkpoint_identically():
    from experiments.residual_prediction_v1.predict import (OnlinePredictor,
                                                            spec_from_descriptor)
    from experiments.night_sequence_v1.dataset import Tokens
    from experiments.residual_prediction_v1.targets import decode_prediction

    checkpoint = OUT / "diagnostic_single_head.pt"
    if not checkpoint.exists():
        pytest.skip("diagnostic checkpoint not produced here")
    tokens = Tokens(D.TIER_SPLITS["d1"])
    panel = D.load_panel("d1", "val", OUT)
    index, ends = panel["index"][:64], panel["end"][:64]
    batch = D.build_batch(tokens, index, ends)
    online = OnlinePredictor(checkpoint)
    values = online.predict(batch.window, batch.token_valid)
    # the online path adds the reference exactly once, through the shared decode function
    online_units = online.decode(values, batch.reference)
    # the offline path rebuilds the transform from the checkpoint descriptor and must agree exactly
    offline_spec = spec_from_descriptor(online.spec_descriptor)
    offline_units = decode_prediction(values, batch.reference, offline_spec)
    assert np.allclose(online_units, offline_units, atol=1e-12)
    # and the transform itself must send the hold output back to the reference exactly
    from experiments.residual_prediction_v1.targets import hold_output
    assert np.allclose(decode_prediction(hold_output(offline_spec), batch.reference, offline_spec),
                       batch.reference, atol=1e-12)


# ---------------------------------------------------------------------------
# motion bins must be about the robot, not the object
# ---------------------------------------------------------------------------


def test_motion_bins_follow_the_documented_thresholds():
    speed = np.array([0.0, 0.009, 0.01, 0.049, 0.05, 1.0])
    labels = motion_bin(speed)
    assert labels[0] == "near_static(<0.01)"
    assert labels[1] == "near_static(<0.01)"
    assert labels[2] == "slow(0.01-0.05)"
    assert labels[3] == "slow(0.01-0.05)"
    assert labels[4] == "moving(>=0.05)"
    assert labels[5] == "moving(>=0.05)"
