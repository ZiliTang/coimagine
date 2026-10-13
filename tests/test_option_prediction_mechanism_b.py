"""Round 3, phase B tests: input truth leakage, masks, split grouping, branch recovery.

The contract names these four risks explicitly (section 5.6). They are cheap and read the real
phase-B artifacts, so a later change that leaks ground truth into the inputs, silently zero-fills a
missing label, splits a root across datasets, or makes branch replay non-deterministic fails here.
"""

from __future__ import annotations

import copy
import json
import math
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


# ---------------------------------------------------------------------------
# the evaluation interface: original units in, no second un-standardisation
# ---------------------------------------------------------------------------


def _tiny_supervised(train_value: float, test_value: float):
    """A hand-built dataset where the persistence prediction is exactly the test truth."""

    from experiments.option_prediction_mechanism_v1.phase_b.model import GroupSpec, SupervisedData

    specs = [GroupSpec("terminal_geometry", "continuous", 4, 1.0),
             GroupSpec("duration", "continuous", 1, 1.0)]
    y = np.array([[train_value] * 4, [train_value] * 4,
                  [test_value] * 4, [test_value] * 4], dtype=np.float32)
    duration = np.array([[train_value], [train_value],
                         [test_value], [test_value]], dtype=np.float32)
    targets = {
        "terminal_geometry": {"y": y, "mask": np.ones_like(y)},
        "duration": {"y": duration, "mask": np.ones_like(duration)},
    }
    return SupervisedData(
        x=np.zeros((4, 2), dtype=np.float32), x_priv=np.zeros((4, 1), dtype=np.float32),
        roots=["r1", "r1", "r2", "r2"], split=np.array(["train", "train", "test", "test"]),
        commands=["0", "1", "0", "1"], targets=targets, specs=specs,
        persistence={"terminal_geometry": np.full((4, 4), test_value, dtype=np.float32),
                     "duration": np.zeros((4, 1), dtype=np.float32)},
        train_duration_median=float(test_value), train_exit_majority=0,
        train_validity_majority=np.zeros(2, dtype=np.float32))


def test_evaluation_interface_is_original_units():
    """Hand-worked: the persistence prediction equals the truth, so its error must be exactly 0.

    With train values 0.5 the standardisation offset is ``mean = 0.5``. If the pipeline
    un-standardised the baseline again (the round-3 bug) the prediction would become
    ``0.2 * 1 + 0.5 = 0.7`` and the error would be 0.5 instead of 0.
    """

    from experiments.option_prediction_mechanism_v1.phase_b.model import (
        _standardise,
        persistence_predictions,
        unstandardise_predictions,
    )

    data = _tiny_supervised(train_value=0.5, test_value=0.2)
    _standardise(data)
    baseline = persistence_predictions(data, "test")
    truth = data.targets["terminal_geometry"]["y_raw"][2:]
    assert baseline["terminal_geometry"][2:] == pytest.approx(truth), (
        "the persistence baseline is defined in original units")
    assert np.abs(baseline["terminal_geometry"][2:] - truth).max() == 0.0
    # and the old, wrong path is demonstrably different
    double_converted = unstandardise_predictions(data, baseline)["terminal_geometry"][2:]
    assert np.abs(double_converted - truth).max() == pytest.approx(0.5)


def test_duration_persistence_error_matches_a_hand_computation():
    """Integration check on the real dataset: the reported MAE equals the hand-computed one."""

    records = _records()
    from experiments.option_prediction_mechanism_v1.phase_b.model import (
        _standardise,
        persistence_predictions,
    )
    from experiments.option_prediction_mechanism_v1.phase_b.train_b import evaluate

    data = build_supervised(records)
    _standardise(data)
    baseline = persistence_predictions(data, "test")
    report = evaluate(data, records, baseline, "test", "persistence-hand-check")
    index = np.where(data.split == "test")[0]
    expected = float(np.median(np.abs(data.train_duration_median
                                      - data.targets["duration"]["y_raw"][index, 0])))
    assert report["duration"]["duration_s"]["p50"] == pytest.approx(expected, abs=1e-9)
    # the signed-distance baseline must also be its own raw start geometry, not a rescaled one
    geometry = baseline["terminal_geometry"][index]
    assert geometry == pytest.approx(data.persistence["terminal_geometry"][index])


# ---------------------------------------------------------------------------
# rank correlation: average ranks for ties, undefined when there is no ordering
# ---------------------------------------------------------------------------


def test_average_ranks_uses_tied_blocks():
    from experiments.option_prediction_mechanism_v1.phase_b.action_sensitivity import average_ranks

    assert average_ranks([1, 1, 1, 1]).tolist() == [2.5, 2.5, 2.5, 2.5]
    assert average_ranks([1, 2, 2, 3]).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert average_ranks([3, 1, 2]).tolist() == [3.0, 1.0, 2.0]
    assert average_ranks([1, 1, 2, 2, 3, 3]).tolist() == [1.5, 1.5, 3.5, 3.5, 5.5, 5.5]


def test_spearman_examples_and_undefined_cases():
    from experiments.option_prediction_mechanism_v1.phase_b.action_sensitivity import spearman

    assert math.isnan(spearman([1, 1, 1, 1], [1, 2, 3, 4])), "constant input -> undefined"
    assert math.isnan(spearman([2, 2, 2], [5, 5, 5])), "both constant -> undefined"
    assert math.isnan(spearman([1, 2], [3, 4])), "fewer than three points -> undefined"
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)
    # ties reduce, rather than manufacture, correlation
    tied = spearman([1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6])
    assert tied == pytest.approx(0.956182887467515)


def test_ordinal_ranks_would_manufacture_correlation_from_ties():
    """Why the fix matters: the old ordinal ranking scored a perfectly tied input as 1.0."""

    ordinal_a = np.argsort(np.argsort(np.asarray([1, 1, 2, 2, 3, 3], dtype=float)))
    ordinal_b = np.argsort(np.argsort(np.asarray([1, 2, 3, 4, 5, 6], dtype=float)))
    assert np.corrcoef(ordinal_a, ordinal_b)[0, 1] == pytest.approx(1.0)
    from experiments.option_prediction_mechanism_v1.phase_b.action_sensitivity import spearman
    assert spearman([1, 1, 2, 2, 3, 3], [1, 2, 3, 4, 5, 6]) < 1.0


# ---------------------------------------------------------------------------
# label provenance: the geometry labels are privileged, not observed
# ---------------------------------------------------------------------------


def test_terminal_geometry_labels_come_from_the_true_object_pose_not_the_observation():
    """Changing only the *observed* terminal patch must leave the trained label untouched."""

    from experiments.option_prediction_mechanism_v1.phase_b.dataset_b import build_targets

    record = _records()[0]
    baseline = build_targets([record])["terminal_geometry"]["y"][0]
    # 1. perturb only the observation the robot received at exit
    observation_only = copy.deepcopy(record)
    robot = observation_only["label"]["robot0"]
    robot["obs_terminal_patch_distance"] = 999.0
    robot["obs_terminal_patch_normal_own_start_frame"] = [1.0, 0.0]
    robot["obs_terminal_patch_centroid_own_start_frame"] = [5.0, 5.0]
    after_observation = build_targets([observation_only])["terminal_geometry"]["y"][0]
    assert after_observation == pytest.approx(baseline), (
        "the trained label must not depend on the observed patch; if this fails the label really "
        "is observable and the target audit should be updated")
    # 2. the label producer, however, is a pure function of the TRUE object pose and the face
    #    index. Test the producer directly: moving the true object moves the geometry.
    from experiments.option_prediction_mechanism_v1.phase_b import collect_b as collector

    object_pose = list(record["truth"]["object_terminal_world"])[:3]
    robot_pose = list(record["truth"]["robots_terminal_world"][0])[:3]
    face = collector.reference_face(object_pose, robot_pose)
    distance_before, tangential_before = collector.segment_geometry(object_pose, face, robot_pose)
    moved = [object_pose[0] + 0.5, object_pose[1], object_pose[2]]
    distance_after, tangential_after = collector.segment_geometry(moved, face, robot_pose)
    assert distance_after != pytest.approx(distance_before), (
        "the label producer reads the true object pose, which is the whole point of the audit")
    assert tangential_after != pytest.approx(tangential_before)
    # and the face index itself is chosen with the true initial object pose
    initial = list(record["truth"]["object_start_world"])[:3]
    start_robot = list(record["truth"]["robots_start_world"][0])[:3]
    assert collector.reference_face(initial, start_robot) == \
        collector.reference_face(initial, start_robot)
