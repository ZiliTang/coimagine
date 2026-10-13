"""Round 3, phase B tests: input truth leakage, masks, split grouping, branch recovery.

The contract names these four risks explicitly (section 5.6). They are cheap and read the real
phase-B artifacts, so a later change that leaks ground truth into the inputs, silently zero-fills a
missing label, splits a root across datasets, or makes branch replay non-deterministic fails here.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from experiments.option_prediction_mechanism_v1.phase_b.dataset_b import (  # noqa: E402
    build_features,
    build_supervised,
    load_records,
)

DATA = ROOT / "outputs/option_prediction_mechanism_v1/option-prediction-mechanism-v1/phase_b"


def _records():
    if not (DATA / "records_tier1.jsonl").exists():
        pytest.skip("phase-B tier-1 dataset not collected on this machine")
    return load_records(DATA / "records_tier1.jsonl")


def test_deployable_inputs_do_not_read_object_ground_truth():
    """Move the object's true start pose; the deployable vector must not change at all."""

    record = _records()[0]
    baseline_x, baseline_priv = build_features(record)
    for shift in ((1.0, 0.0), (0.0, 0.5), (0.0, 0.0)):
        perturbed = copy.deepcopy(record)
        object_state = perturbed["truth"]["object_start_world"]
        perturbed["truth"]["object_start_world"] = [
            object_state[0] + shift[0], object_state[1] + shift[1],
            object_state[2] + 0.3, object_state[3] + 0.05, object_state[4] - 0.02,
            object_state[5] + 0.1,
        ]
        moved_x, moved_priv = build_features(perturbed)
        assert moved_x == pytest.approx(baseline_x), (
            "the deployable input changed when only the object's hidden truth changed")
        assert moved_priv != pytest.approx(baseline_priv), (
            "the privileged slot should be the only place the object truth enters")


def test_missing_terminal_geometry_is_masked_not_zero_filled():
    records = _records()
    data = build_supervised(records)
    mask = data.targets["terminal_geometry"]["mask"]
    truth = data.targets["terminal_geometry"]["y"]  # raw until _standardise runs
    # every record whose terminal geometry is unobservable must be masked, and the raw label may
    # be any value -- what matters is that the mask, not a substituted zero, carries the absence
    unobservable = [index for index, record in enumerate(records)
                    if record["label"]["robot0"]["mask_terminal_geometry"] == 0]
    assert unobservable, "the dataset should keep at least one unobservable terminal patch"
    for index in unobservable:
        assert mask[index, 0] == 0 and mask[index, 1] == 0
        assert mask[index, 2] == 0 and mask[index, 3] == 0
    observable = mask[:, 0] > 0
    assert observable.any()
    assert np.isfinite(truth[observable]).all()


def test_no_root_appears_in_two_splits():
    records = _records()
    owner: dict = {}
    for record in records:
        root = record["root_id"]
        split = record["split"]
        if root in owner:
            assert owner[root] == split, f"{root} appears in both {owner[root]} and {split}"
        owner[root] = split
    assert set(owner.values()) == {"train", "val", "test"}
    # the eight commands of a root stay together, so a branch can never cross the split boundary
    counts: dict = {}
    for record in records:
        counts[(record["root_id"], record["split"])] = \
            counts.get((record["root_id"], record["split"]), 0) + 1
    assert all(value == 8 for value in counts.values())


def test_branch_recovery_is_bit_exact():
    """Re-running one command from the same root must reproduce the identical snapshot."""

    from coimagine.continuous2d.config import default_config
    from coimagine.local_contact_option.runner import run_execution, build_snapshot
    from experiments.option_prediction_mechanism_v1.phase_b import collect_b
    from experiments.option_prediction_mechanism_v1.phase_b import roots_b

    roots, _reports = roots_b.build_phase_b_roots({"train": 2, "val": 0, "test": 0})
    root = roots["train"][0]
    command = roots_b.commands_for_root(root)[0]
    prepared = roots_b.apply_command(root, command)
    option = collect_b.frozen_option_config()
    variant = collect_b.VARIANT
    first, _rows, _report = run_execution(prepared, variant, "V", option, default_config())
    second, _rows2, _report2 = run_execution(prepared, variant, "V", option, default_config())
    assert first["terminal_object"] == second["terminal_object"]
    assert first["terminal_robots"] == second["terminal_robots"]
    assert first["metrics"]["exit_reason"] == second["metrics"]["exit_reason"]
    assert first["metrics"]["wall_clock_steps"] == second["metrics"]["wall_clock_steps"]
    # and the snapshot really is shared across the eight commands of that root
    hashes = {
        collect_b.snapshot_sha256(build_snapshot(roots_b.apply_command(root, c), variant,
                                                 default_config())[0])
        for c in roots_b.commands_for_root(root)
    }
    assert len(hashes) == 1


def test_persistence_baseline_uses_train_statistics():
    records = _records()
    data = build_supervised(records)
    train = data.split == "train"
    durations = data.targets["duration"]["y"][train, 0]
    assert data.train_duration_median == pytest.approx(float(np.median(durations)))
    classes = data.targets["exit_class"]["y"][train]
    assert data.train_exit_majority == int(np.bincount(classes).argmax())
    validity = data.targets["observation_validity"]["y"][train]
    assert data.train_validity_majority.tolist() == \
        (validity.mean(axis=0) >= 0.5).astype(float).tolist()
